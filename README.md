# cuda-vmm-sysmem-fallback

A transparent `LD_PRELOAD` shim that gives Linux the same GPU memory oversubscription behavior as Windows WDDM. When a CUDA application exhausts VRAM, the shim intercepts the allocation at the Driver API level and creates a **split allocation** — first portion in VRAM, overflow in system RAM — behind a single contiguous GPU virtual address. No application changes required.

## The Problem

```mermaid
flowchart LR
    subgraph Windows["Windows (WDDM)"]
        direction TB
        W1["cudaMalloc(20GB)"] --> W2["WDDM allocates:<br/>16GB VRAM + 4GB sysmem"]
        W2 --> W3["GPU computes all layers<br/>VRAM @ 288 GB/s<br/>overflow @ PCIe speed"]
        style W3 fill:#2d6a4f,color:#fff
    end

    subgraph Linux["Linux (default)"]
        direction TB
        L1["cudaMalloc(20GB)"] --> L2["cudaErrorMemoryAllocation"]
        L2 --> L3["App routes overflow<br/>to CPU compute"]
        style L3 fill:#9d0208,color:#fff
    end

    subgraph Shim["Linux + this shim"]
        direction TB
        S1["cudaMalloc(20GB)"] --> S2["Shim intercepts OOM<br/>VMM split allocation"]
        S2 --> S3["GPU computes all layers<br/>VRAM @ 288 GB/s<br/>overflow @ PCIe speed"]
        style S3 fill:#2d6a4f,color:#fff
    end
```

On Windows, NVIDIA's WDDM driver transparently backs overflowing allocations with system RAM. On Linux, `cudaMalloc()` fails and applications fall back to CPU compute. NVIDIA has an [open feature request](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/663) for Linux sysmem fallback — closed without action.

This project bridges that gap at the userspace level.

## How It Works

```mermaid
sequenceDiagram
    participant App as CUDA Application
    participant Shim as VMM Fallback Shim
    participant Driver as libcuda.so
    participant GPU as GPU Hardware

    App->>Shim: cuMemAlloc(20GB)
    Shim->>Driver: cuMemAlloc(20GB)
    Driver-->>Shim: CUDA_ERROR_OUT_OF_MEMORY

    Note over Shim: Query real VRAM free (~14GB)

    Shim->>Driver: cuMemAddressReserve(20GB)
    Driver-->>Shim: contiguous VA range

    Shim->>Driver: cuMemCreate(14GB, DEVICE)
    Driver-->>Shim: VRAM physical handle

    Shim->>Driver: cuMemCreate(6GB, HOST_NUMA)
    Driver-->>Shim: sysmem physical handle

    Shim->>Driver: cuMemMap(va+0, 14GB, vram)
    Shim->>Driver: cuMemMap(va+14GB, 6GB, sysmem)
    Shim->>Driver: cuMemSetAccess(va, 20GB, GPU_RW)

    Shim-->>App: valid GPU pointer

    Note over GPU: Computes on all data<br/>VRAM portion @ 288 GB/s<br/>sysmem portion @ PCIe speed
```

The shim also intercepts:
- **NVML** (`nvmlDeviceGetMemoryInfo`) — spoofs inflated VRAM so Ollama's layer planner assigns all layers to GPU
- **`cuMemGetInfo`** — reports VRAM + sysmem capacity
- **`cuGetProcAddress`** / **`dlsym`** — intercepts bundled cudart that bypasses LD_PRELOAD symbol search order
- **`GGML_CUDA_ENABLE_UNIFIED_MEMORY`** — automatically unset (forces `cudaMalloc` path for interception)

## Interception Architecture

```mermaid
graph TD
    subgraph Application
        A1[Ollama / vLLM / PyTorch]
        A2[Bundled libcudart.so]
    end

    subgraph Shim["libcuda_vmm_fallback.so (LD_PRELOAD)"]
        I1["dlsym hook<br/><i>catches cuGetProcAddress<br/>+ nvmlDeviceGetMemoryInfo</i>"]
        I2["cuMemAlloc_v2 → split alloc"]
        I3["cuMemFree_v2 → VMM cleanup"]
        I4["cuMemGetInfo_v2 → spoof free"]
        I5["nvmlDeviceGetMemoryInfo → spoof free"]
        I6["GGML_CUDA_ENABLE_UNIFIED_MEMORY → unset"]
    end

    subgraph System
        D1[libcuda.so<br/>CUDA Driver API]
        D2[libnvidia-ml.so<br/>NVML]
        D3[nvidia.ko<br/>GPU Hardware]
    end

    A1 --> A2
    A2 -->|dlsym| I1
    I1 -->|redirect| I2
    I1 -->|redirect| I5
    I2 --> D1
    I3 --> D1
    I4 --> D1
    I5 --> D2
    D1 --> D3

    style Shim fill:#1a1a2e,color:#e0e0e0
    style I2 fill:#2d6a4f,color:#fff
    style I5 fill:#2d6a4f,color:#fff
```

## Memory Layout (Split Allocation)

```mermaid
block-beta
    columns 1

    block:VA["GPU Virtual Address Space (contiguous 20GB)"]
        columns 2
        VRAM["VRAM Portion<br/>14 GB @ 288 GB/s<br/>(first layers)"] :1
        SYSMEM["Sysmem Portion<br/>6 GB @ PCIe speed<br/>(overflow layers)"] :1
    end

    block:PHYS["Physical Backing"]
        columns 2
        P1["GPU GDDR6<br/>16 GB total"] :1
        P2["Host DDR5<br/>96 GB total"] :1
    end

    VRAM --> P1
    SYSMEM --> P2

    style VRAM fill:#2d6a4f,color:#fff
    style SYSMEM fill:#e76f51,color:#fff
    style P1 fill:#264653,color:#fff
    style P2 fill:#264653,color:#fff
```

## Benchmark Results

Tested on RTX 4060 Ti 16GB (PCIe x8 Gen4), 96GB DDR5-5200, qwen3:32b Q4_K_M (~20GB model).

| Approach | tok/s | vs Baseline | Mechanism |
|----------|------:|:-----------:|-----------|
| Baseline (GGML CPU fallback) | **6.39** | — | CPU computes overflow @ DDR5 83 GB/s |
| **VMM split (13.1GB VRAM + 5.3GB sysmem)** | **4.63** | 0.72x | GPU reads overflow @ PCIe ~10 GB/s |
| UVM + cudaMemPrefetchAsync | 1.29 | 0.20x | Page migration overhead |
| All-sysmem VMM | 0.50 | 0.08x | All weights over PCIe |
| UVM + cudaMemAdvise hints | 0.44 | 0.07x | Zero-copy but UVM overhead |
| Raw UVM (no hints) | 0.22 | 0.03x | Page fault stalls |

```mermaid
xychart-beta
    title "Inference Speed by Approach (tok/s, higher is better)"
    x-axis ["CPU fallback", "VMM split", "UVM prefetch", "All sysmem", "UVM hints", "Raw UVM"]
    y-axis "Tokens per second" 0 --> 7
    bar [6.39, 4.63, 1.29, 0.50, 0.44, 0.22]
```

**Key findings:**
- VMM split is **10x faster** than raw UVM and **2x faster** than all other GPU-compute overflow approaches
- CPU fallback still wins by 1.4x on PCIe x8 because DDR5 (83 GB/s) is 5x faster than PCIe x8 Gen4 (15.75 GB/s) for overflow reads
- On PCIe x16 Gen4/Gen5 the bandwidth gap narrows and the shim should match or exceed CPU fallback
- WDDM achieves better results via hardware-level page residency in the GPU MMU — Linux UVM's software fault handling is orders of magnitude slower

## Why WDDM Is Faster

```mermaid
flowchart TD
    subgraph WDDM["Windows WDDM Page Residency"]
        W1["GPU MMU hardware fault"] --> W2["VidMm kernel driver<br/>bulk DMA migration"]
        W2 --> W3["Pages in VRAM<br/>GPU reads @ 288 GB/s"]
        W3 --> W4["Cold pages evicted<br/>to sysmem transparently"]
    end

    subgraph UVM["Linux UVM Page Fault"]
        L1["GPU MMU hardware fault"] --> L2["nvidia-uvm kernel interrupt"]
        L2 --> L3["Software fault handler<br/>CPU-initiated DMA"]
        L3 --> L4["TLB flush + retry<br/>per 4KB page"]
    end

    subgraph VMM["This Shim (VMM Split)"]
        V1["No page faults"] --> V2["VRAM portion:<br/>direct GPU access"]
        V1 --> V3["Sysmem portion:<br/>GPU reads over PCIe<br/>(zero-copy, pinned)"]
    end

    style W3 fill:#2d6a4f,color:#fff
    style L4 fill:#9d0208,color:#fff
    style V2 fill:#2d6a4f,color:#fff
    style V3 fill:#e76f51,color:#fff
```

WDDM's advantage is **dynamic page residency** managed at the hardware level. Pages migrate between VRAM and sysmem based on access patterns, with the working set (current layer weights) always in VRAM. Linux has no equivalent kernel-level mechanism — the NVIDIA Linux driver explicitly declined to implement this ([issue #663](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/663), closed Sept 2024).

This shim does the next best thing: a **static split** where the first N bytes live in VRAM permanently and the overflow lives in pinned system RAM with direct GPU page table access. No page faults, no migration overhead.

## Usage

```bash
# Basic: run any CUDA app with the shim
LD_PRELOAD=./libcuda_vmm_fallback.so ollama serve

# Systemd service override
# /etc/systemd/system/ollama.service.d/override.conf:
# [Service]
# Environment="LD_PRELOAD=/usr/local/lib/libcuda_vmm_fallback.so"

# Environment variables:
#   CUDA_VMM_FALLBACK_LOG_LEVEL    0=silent 1=fallbacks 2=all (default: 1)
#   CUDA_VMM_FALLBACK_MAX_SYSMEM   Max sysmem bytes (default: 50% RAM)
#   CUDA_VMM_FALLBACK_DISABLE      Set to 1 to passthrough all calls
```

**Important:** The shim automatically unsets `GGML_CUDA_ENABLE_UNIFIED_MEMORY` because GGML's `cudaMallocManaged` path bypasses the shim's `cuMemAlloc` interception.

## Requirements

- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- NVIDIA Linux driver 535+ (open or proprietary kernel modules)
- CUDA toolkit 12.0+ (for VMM API support)
- Linux kernel 5.6+

## Building

```bash
make                  # builds libcuda_vmm_fallback.so
make test             # builds test suite (requires nvcc)
sudo make install     # installs to /usr/local/lib/
```

## Testing

```bash
# Full test suite: VRAM fill + sysmem overflow + GPU kernel verification
CUDA_VMM_FALLBACK_LOG_LEVEL=2 LD_PRELOAD=./libcuda_vmm_fallback.so ./tests/test_alloc

# NVML spoofing verification
LD_PRELOAD=./libcuda_vmm_fallback.so nvidia-smi --query-gpu=memory.free,memory.total --format=csv
```

## Prior Art

- [Nixie](https://arxiv.org/abs/2601.11743) — LD_PRELOAD shim for GPU memory virtualization (RTX 5090, no public source)
- [NVIDIA CUDA VMM API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html) — underlying API
- [WDDM 2.0 GPU Virtual Memory](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-memory-in-wddm-2-0) — Windows mechanism this replicates
- NVIDIA open-gpu-kernel-modules [#663](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/663) — closed feature request

## Status

**Working prototype.** Split VMM allocation verified with Ollama + qwen3:32b (4.63 tok/s). NVML spoofing verified. Test suite passes. Performance limited by PCIe bandwidth for overflow portion — PCIe x16 or Gen5 would close the gap with CPU fallback.

## License

MIT
