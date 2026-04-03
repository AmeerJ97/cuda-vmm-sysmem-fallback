/*
 * cuda-vmm-sysmem-fallback — WDDM-style system memory fallback for Linux
 *
 * LD_PRELOAD shim that intercepts CUDA Driver API allocation calls.
 * When VRAM is exhausted, transparently allocates in system RAM via
 * CUDA VMM API (cuMemCreate + cuMemMap + cuMemSetAccess) with GPU
 * page table mappings for direct PCIe access.
 *
 * Usage: LD_PRELOAD=./libcuda_vmm_fallback.so <any_cuda_app>
 *
 * Environment variables:
 *   CUDA_VMM_FALLBACK_LOG_LEVEL    0=silent 1=fallbacks 2=all (default: 1)
 *   CUDA_VMM_FALLBACK_MAX_SYSMEM   Max sysmem in bytes (default: 50% of RAM)
 *   CUDA_VMM_FALLBACK_DISABLE      Set to 1 to passthrough all calls
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

/* CUDA Driver API types — we declare these manually to avoid
 * requiring the CUDA toolkit headers at build time. */

typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef void* CUmemGenericAllocationHandle;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2

/* TODO: Fill in remaining CUDA VMM type definitions from cuda.h:
 *   CUmemAllocationProp, CUmemAccessDesc, CUmemLocationType, etc.
 *   These are needed for cuMemCreate, cuMemMap, cuMemSetAccess.
 *   See: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
 */

/* ── Allocation Tracking ──────────────────────────────────────────── */

typedef enum {
    ALLOC_VRAM,     /* Normal VRAM allocation (passthrough) */
    ALLOC_SYSMEM,   /* Fallback: physical chunk in system RAM, mapped to GPU VA */
} alloc_location_t;

typedef struct alloc_entry {
    CUdeviceptr         ptr;        /* GPU virtual address */
    size_t              size;       /* Allocation size in bytes */
    alloc_location_t    location;   /* Where the physical memory lives */
    /* For ALLOC_SYSMEM: */
    /* CUmemGenericAllocationHandle phys_handle; */
    /* CUdeviceptr                  va_base;     */
    struct alloc_entry* next;       /* Linked list (replace with hash map) */
} alloc_entry_t;

static alloc_entry_t*  g_alloc_list = NULL;
static pthread_mutex_t g_alloc_lock = PTHREAD_MUTEX_INITIALIZER;
static size_t          g_sysmem_used = 0;
static size_t          g_sysmem_max  = 0;  /* Set from env or 50% of RAM */
static int             g_log_level   = 1;
static int             g_disabled    = 0;

/* ── Logging ──────────────────────────────────────────────────────── */

#define LOG_FALLBACK(fmt, ...) \
    do { if (g_log_level >= 1) fprintf(stderr, "[cuda-vmm-fallback] " fmt "\n", ##__VA_ARGS__); } while(0)

#define LOG_DEBUG(fmt, ...) \
    do { if (g_log_level >= 2) fprintf(stderr, "[cuda-vmm-fallback] " fmt "\n", ##__VA_ARGS__); } while(0)

/* ── Initialization ───────────────────────────────────────────────── */

__attribute__((constructor))
static void shim_init(void) {
    const char* env;

    env = getenv("CUDA_VMM_FALLBACK_DISABLE");
    if (env && atoi(env)) {
        g_disabled = 1;
        return;
    }

    env = getenv("CUDA_VMM_FALLBACK_LOG_LEVEL");
    if (env) g_log_level = atoi(env);

    env = getenv("CUDA_VMM_FALLBACK_MAX_SYSMEM");
    if (env) {
        g_sysmem_max = (size_t)atoll(env);
    } else {
        /* Default: 50% of system RAM */
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        g_sysmem_max = (size_t)(pages * page_size) / 2;
    }

    LOG_FALLBACK("initialized: max_sysmem=%.1f GB, log_level=%d",
                 g_sysmem_max / (1024.0 * 1024.0 * 1024.0), g_log_level);
}

/* ── cuMemAlloc Interception ──────────────────────────────────────── */

/*
 * This is the core of the shim. cuMemAlloc is the Driver API allocation
 * function. Every CUDA runtime (libcudart.so) calls this through libcuda.so.
 *
 * Interception strategy:
 *   1. dlsym(RTLD_NEXT, "cuMemAlloc") → get real function
 *   2. Call real cuMemAlloc
 *   3. If CUDA_SUCCESS → track as ALLOC_VRAM, return
 *   4. If CUDA_ERROR_OUT_OF_MEMORY →
 *      a. cuMemCreate() with CU_MEM_LOCATION_TYPE_HOST
 *      b. cuMemAddressReserve() for GPU virtual address
 *      c. cuMemMap() to connect virtual → physical
 *      d. cuMemSetAccess() to grant GPU read/write
 *      e. Track as ALLOC_SYSMEM, return the GPU virtual address
 */

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    typedef CUresult (*real_fn_t)(CUdeviceptr*, size_t);
    static real_fn_t real_cuMemAlloc = NULL;

    if (!real_cuMemAlloc) {
        real_cuMemAlloc = (real_fn_t)dlsym(RTLD_NEXT, "cuMemAlloc");
        if (!real_cuMemAlloc) {
            LOG_FALLBACK("FATAL: cannot resolve real cuMemAlloc");
            return 1;
        }
    }

    if (g_disabled) {
        return real_cuMemAlloc(dptr, bytesize);
    }

    /* Try VRAM first */
    CUresult err = real_cuMemAlloc(dptr, bytesize);

    if (err == CUDA_SUCCESS) {
        LOG_DEBUG("cuMemAlloc(%zu) → VRAM", bytesize);
        /* TODO: Track as ALLOC_VRAM */
        return CUDA_SUCCESS;
    }

    if (err != CUDA_ERROR_OUT_OF_MEMORY) {
        return err;  /* Some other error, don't intercept */
    }

    /* ── VRAM exhausted — sysmem fallback ── */

    if (g_sysmem_used + bytesize > g_sysmem_max) {
        LOG_FALLBACK("cuMemAlloc(%zu) DENIED: sysmem limit reached (%.1f/%.1f GB)",
                     bytesize,
                     g_sysmem_used / (1024.0*1024.0*1024.0),
                     g_sysmem_max / (1024.0*1024.0*1024.0));
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    /*
     * TODO: Implement VMM-based sysmem fallback:
     *
     * 1. cuMemCreate(&phys_handle, bytesize, &alloc_prop, 0)
     *    where alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
     *          alloc_prop.location.type = CU_MEM_LOCATION_TYPE_HOST
     *          alloc_prop.location.id = 0
     *
     * 2. cuMemAddressReserve(&va_ptr, bytesize, alignment, 0, 0)
     *    Reserve GPU virtual address range
     *
     * 3. cuMemMap(va_ptr, bytesize, 0, phys_handle, 0)
     *    Map virtual range to physical sysmem allocation
     *
     * 4. cuMemSetAccess(va_ptr, bytesize, &access_desc, 1)
     *    where access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
     *          access_desc.location.id = 0  (GPU device 0)
     *          access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
     *
     * 5. *dptr = va_ptr
     *    Return the GPU virtual address to the caller
     *
     * See: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
     * See: cuda-samples/Samples/0_Introduction/vectorAddMMAP/
     */

    LOG_FALLBACK("cuMemAlloc(%zu) → sysmem fallback NOT YET IMPLEMENTED", bytesize);
    return CUDA_ERROR_OUT_OF_MEMORY;
}

/* ── cuMemFree Interception ───────────────────────────────────────── */

CUresult cuMemFree(CUdeviceptr dptr) {
    typedef CUresult (*real_fn_t)(CUdeviceptr);
    static real_fn_t real_cuMemFree = NULL;

    if (!real_cuMemFree) {
        real_cuMemFree = (real_fn_t)dlsym(RTLD_NEXT, "cuMemFree");
        if (!real_cuMemFree) return 1;
    }

    if (g_disabled) {
        return real_cuMemFree(dptr);
    }

    /*
     * TODO: Look up dptr in allocation tracking.
     * If ALLOC_VRAM: pass through to real cuMemFree.
     * If ALLOC_SYSMEM:
     *   1. cuMemUnmap(dptr, size)
     *   2. cuMemAddressFree(dptr, size)
     *   3. cuMemRelease(phys_handle)
     *   4. Remove from tracking, decrement g_sysmem_used
     */

    return real_cuMemFree(dptr);
}

/* ── cuMemGetInfo Interception ────────────────────────────────────── */

/*
 * Spoof available memory to prevent applications from pre-rejecting
 * allocations. Report free = real VRAM free + available sysmem.
 * This causes Ollama's llama_params_fit_impl to think all layers fit.
 */
CUresult cuMemGetInfo(size_t* free, size_t* total) {
    typedef CUresult (*real_fn_t)(size_t*, size_t*);
    static real_fn_t real_cuMemGetInfo = NULL;

    if (!real_cuMemGetInfo) {
        real_cuMemGetInfo = (real_fn_t)dlsym(RTLD_NEXT, "cuMemGetInfo");
        if (!real_cuMemGetInfo) return 1;
    }

    CUresult err = real_cuMemGetInfo(free, total);
    if (err != CUDA_SUCCESS || g_disabled) return err;

    size_t real_free = *free;
    size_t sysmem_avail = g_sysmem_max - g_sysmem_used;

    *free  = real_free + sysmem_avail;
    *total = *total + g_sysmem_max;

    LOG_DEBUG("cuMemGetInfo: real_free=%.1f GB, spoofed_free=%.1f GB",
              real_free / (1024.0*1024.0*1024.0),
              *free / (1024.0*1024.0*1024.0));

    return CUDA_SUCCESS;
}
