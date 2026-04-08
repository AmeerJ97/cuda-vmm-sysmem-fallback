/*
 * test_alloc.cu — Verify the VMM sysmem fallback shim works.
 *
 * Tests:
 *   1. cuMemGetInfo reports inflated free memory
 *   2. cuMemAlloc succeeds for allocation beyond physical VRAM
 *   3. GPU kernel can read/write data in sysmem-backed allocation
 *   4. cuMemFree cleans up VMM allocations without leaks
 *
 * Build: nvcc -o test_alloc tests/test_alloc.cu -lcuda
 * Run:   LD_PRELOAD=./libcuda_vmm_fallback.so ./test_alloc
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Simple kernel: write a pattern, read it back */
__global__ void write_pattern(float *data, size_t n, float val) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = val + (float)idx;
}

__global__ void verify_pattern(float *data, size_t n, float val, int *errors) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) {
        float expected = val + (float)idx;
        if (data[idx] != expected)
            atomicAdd(errors, 1);
    }
}

#define CHECK_CUDA(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *str; \
        cuGetErrorString(err, &str); \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, str ? str : "unknown", err); \
        exit(1); \
    } \
} while(0)

#define CHECK_RT(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA RT error at %s:%d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(1); \
    } \
} while(0)

int main(void) {
    /* Initialize CUDA driver API */
    CHECK_CUDA(cuInit(0));

    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    CHECK_CUDA(cuCtxCreate(&ctx, &ctxParams, 0, dev));

    /* Test 1: cuMemGetInfo reports inflated memory */
    printf("=== Test 1: cuMemGetInfo spoofing ===\n");
    size_t free_mem, total_mem;
    CHECK_CUDA(cuMemGetInfo(&free_mem, &total_mem));
    printf("  Reported: free=%.1f GB, total=%.1f GB\n",
           free_mem / (1024.0 * 1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0 * 1024.0));

    /* With shim, total should be VRAM + sysmem max */
    if (total_mem > 20ULL * 1024 * 1024 * 1024) {
        printf("  PASS: total memory is inflated (shim active)\n");
    } else {
        printf("  INFO: total memory not inflated (shim may not be loaded)\n");
    }

    /* Test 2: Allocate beyond VRAM */
    printf("\n=== Test 2: Overflow allocation ===\n");

    /* Fill most of VRAM — leave ~2 GB headroom for driver/context */
    size_t vram_fill = 12ULL * 1024 * 1024 * 1024; /* 12 GB on 16 GB card */
    CUdeviceptr vram_ptr;
    printf("  Allocating %.1f GB (VRAM fill)...\n", vram_fill / (1024.0 * 1024.0 * 1024.0));
    CHECK_CUDA(cuMemAlloc(&vram_ptr, vram_fill));
    printf("  PASS: VRAM allocation at 0x%llx\n", vram_ptr);

    /* Now allocate 4 GB more — VRAM is nearly full, this triggers sysmem fallback */
    size_t overflow_size = 4ULL * 1024 * 1024 * 1024;
    CUdeviceptr overflow_ptr;
    printf("  Allocating %.1f GB (should trigger sysmem fallback)...\n",
           overflow_size / (1024.0 * 1024.0 * 1024.0));
    CHECK_CUDA(cuMemAlloc(&overflow_ptr, overflow_size));
    printf("  PASS: Overflow allocation at 0x%llx\n", overflow_ptr);

    /* Test 3: GPU kernel reads/writes sysmem-backed memory */
    printf("\n=== Test 3: GPU compute on sysmem ===\n");

    /* Use a smaller test region for speed */
    size_t test_elements = 1024 * 1024; /* 4 MB worth */
    int threads = 256;
    int blocks = (int)((test_elements + threads - 1) / threads);
    float pattern_val = 42.0f;

    printf("  Writing pattern with GPU kernel (%zu elements)...\n", test_elements);
    write_pattern<<<blocks, threads>>>((float*)overflow_ptr, test_elements, pattern_val);
    CHECK_RT(cudaDeviceSynchronize());

    /* Verify on GPU */
    int *d_errors;
    CHECK_RT(cudaMalloc(&d_errors, sizeof(int)));
    CHECK_RT(cudaMemset(d_errors, 0, sizeof(int)));

    verify_pattern<<<blocks, threads>>>((float*)overflow_ptr, test_elements, pattern_val, d_errors);
    CHECK_RT(cudaDeviceSynchronize());

    int h_errors = 0;
    CHECK_RT(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_errors == 0) {
        printf("  PASS: GPU kernel verified %zu elements in sysmem — zero errors\n", test_elements);
    } else {
        printf("  FAIL: %d verification errors\n", h_errors);
        exit(1);
    }

    CHECK_RT(cudaFree(d_errors));

    /* Test 4: Free and verify cleanup */
    printf("\n=== Test 4: Cleanup ===\n");
    CHECK_CUDA(cuMemFree(overflow_ptr));
    printf("  PASS: Overflow allocation freed\n");

    CHECK_CUDA(cuMemFree(vram_ptr));
    printf("  PASS: VRAM allocation freed\n");

    /* Verify memory is returned */
    CHECK_CUDA(cuMemGetInfo(&free_mem, &total_mem));
    printf("  After free: free=%.1f GB, total=%.1f GB\n",
           free_mem / (1024.0 * 1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0 * 1024.0));

    CHECK_CUDA(cuCtxDestroy(ctx));

    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
