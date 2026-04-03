# cuda-vmm-sysmem-fallback

A transparent `LD_PRELOAD` shim that gives Linux the same GPU memory oversubscription behavior as Windows WDDM. When a CUDA application exhausts VRAM, the shim intercepts the allocation at the Driver API level and redirects it to system RAM via CUDA's Virtual Memory Management API — with GPU page table mappings for direct PCIe access. The GPU computes on all data, whether it lives in VRAM or system RAM. No application changes required.

## The Problem

On Windows, NVIDIA's WDDM driver transparently backs failed `cudaMalloc()` calls with system RAM. GPU page tables map the overflow memory, and CUDA kernels read it directly over PCIe. Applications don't know the difference. A 32B parameter LLM on a 16GB GPU just works — the 4GB overflow silently lives in system RAM.

On Linux, `cudaMalloc()` returns `cudaErrorMemoryAllocation` when VRAM is full. Applications must handle this themselves. Most inference engines (llama.cpp, Ollama) respond by routing overflow compute to the CPU instead of the GPU — resulting in dramatically lower performance. NVIDIA has an [open feature request](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/663) for Linux sysmem fallback with no response.

This project bridges that gap at the userspace level.

## How It Works

```
Without shim (Linux default):
  cudaMalloc(20GB) on 16GB GPU → cudaErrorMemoryAllocation → app falls back to CPU

With shim:
  cudaMalloc(20GB) on 16GB GPU
    → cuMemAlloc intercepted by shim
    → First 16GB allocated in VRAM (normal)
    → Remaining 4GB:
      1. cuMemCreate() with CU_MEM_LOCATION_TYPE_HOST → physical chunk in system RAM
      2. cuMemMap() → map into GPU virtual address space
      3. cuMemSetAccess() → GPU page table entries for direct PCIe reads
    → Application receives valid GPU pointer
    → CUDA kernels compute on all data — overflow reads from RAM at PCIe speed
```

The shim intercepts at the **CUDA Driver API** level (`cuMemAlloc` in `libcuda.so`), not the Runtime API (`cudaMalloc` in `libcudart.so`). Since every CUDA runtime ultimately calls through the single system `libcuda.so`, this works for all CUDA applications regardless of which runtime they bundle.

## Architecture

```
┌──────────────────────────────────────────┐
│  Any CUDA Application                    │
│  (Ollama, vLLM, PyTorch, TRT-LLM, etc.) │
├──────────────────────────────────────────┤
│  libcudart.so (Runtime API)              │
│  (may be bundled per-application)        │
├──────────────────────────────────────────┤
│  ┌──────────────────────────────────┐    │
│  │  libcuda_vmm_fallback.so         │    │  ← LD_PRELOAD shim
│  │  Intercepts: cuMemAlloc,         │    │
│  │    cuMemFree, cuMemGetInfo       │    │
│  │  Fallback: CUDA VMM API          │    │
│  │    (cuMemCreate + cuMemMap +     │    │
│  │     cuMemSetAccess)              │    │
│  └──────────────────────────────────┘    │
├──────────────────────────────────────────┤
│  libcuda.so (Driver API)                 │
│  (single instance per system)            │
├──────────────────────────────────────────┤
│  nvidia.ko → GPU Hardware                │
└──────────────────────────────────────────┘
```

## Usage

```bash
# Run any CUDA application with the shim
LD_PRELOAD=/usr/local/lib/libcuda_vmm_fallback.so ollama serve

# Or add to a systemd service
# In /etc/systemd/system/ollama.service.d/override.conf:
# Environment="LD_PRELOAD=/usr/local/lib/libcuda_vmm_fallback.so"
```

## Requirements

- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- NVIDIA Linux driver 535+ (open or proprietary kernel modules)
- CUDA toolkit 12.0+ (for VMM API support)
- Linux kernel 5.6+ (for `LD_PRELOAD` with modern glibc)

## Building

```bash
make
sudo make install  # installs to /usr/local/lib/
```

## How This Differs From Existing Approaches

| Approach | Scope | Mechanism | Limitation |
|----------|-------|-----------|------------|
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | llama.cpp only | `cudaMallocManaged` (UVM page faults) | GGML still routes overflow compute to CPU |
| vLLM `--cpu-offload-gb` | vLLM only | Pinned memory + explicit copies | Synchronous weight shuttling, slow |
| Windows WDDM Sysmem Fallback | All Windows CUDA apps | Driver-level `cudaMalloc` redirect | Windows only |
| **This project** | **All Linux CUDA apps** | **Driver API interception + CUDA VMM** | **PCIe bandwidth ceiling** |

## Performance Expectations

GPU compute on overflow data is limited by PCIe bandwidth:

| PCIe Config | Bandwidth | 4GB overflow per token | Theoretical max tok/s |
|-------------|-----------|----------------------|----------------------|
| x16 Gen4 | 31.5 GB/s | ~127ms | ~7.9 |
| x8 Gen4 | 15.75 GB/s | ~254ms | ~3.9 |
| x16 Gen5 | 63 GB/s | ~63ms | ~15.8 |

For models with small overflow (1-4GB beyond VRAM), CPU compute may actually be faster than GPU-over-PCIe because the CPU reads from DDR5 at ~83 GB/s without traversing the PCIe bus. This shim is most beneficial for **large overflows** where GPU tensor core throughput exceeds CPU SIMD throughput per byte.

## Prior Art

- [Nixie](https://arxiv.org/abs/2601.11743) (arXiv 2601.11743, Jan 2026) — LD_PRELOAD shim using CUDA VMM for transparent GPU memory virtualization. Tested on RTX 5090. No public source code.
- [NVIDIA CUDA VMM API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html) — the underlying API this project uses.
- [NVIDIA Blog: Introducing Low-Level GPU Virtual Memory Management](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/) — tutorial on the VMM API.
- [vectorAddMMAP sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAddMMAP) — NVIDIA's reference implementation of VMM-based allocation.
- [WDDM 2.0 GPU Virtual Memory](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-memory-in-wddm-2-0) — the Windows mechanism this project replicates.

## Status

**Pre-implementation.** Architecture designed, API surface identified, reference materials collected. See `dev-docs/` (not tracked in git) for investigation notes and roadmap.

## License

MIT
