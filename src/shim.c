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
#include <stdint.h>
#include <stdatomic.h>

/* ── CUDA Driver API Types ───────────────────────────────────────────
 * Declared manually to avoid requiring CUDA toolkit headers at build time.
 * Values match cuda.h from CUDA 13.1.1.
 */

typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef int CUdevice;

#define CUDA_SUCCESS                    0
#define CUDA_ERROR_OUT_OF_MEMORY        2
#define CUDA_ERROR_NOT_INITIALIZED      3
#define CUDA_ERROR_INVALID_VALUE        11

/* ── CUDA VMM Enums ──────────────────────────────────────────────── */

typedef enum {
    CU_MEM_LOCATION_TYPE_INVALID         = 0x0,
    CU_MEM_LOCATION_TYPE_DEVICE          = 0x1,
    CU_MEM_LOCATION_TYPE_HOST            = 0x2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA       = 0x3,
    CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 0x4,
} CUmemLocationType;

typedef enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,
    CU_MEM_ALLOCATION_TYPE_PINNED  = 0x1,
} CUmemAllocationType;

typedef enum {
    CU_MEM_HANDLE_TYPE_NONE = 0x0,
} CUmemAllocationHandleType;

typedef enum {
    CU_MEM_ACCESS_FLAGS_PROT_NONE      = 0x0,
    CU_MEM_ACCESS_FLAGS_PROT_READ      = 0x1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
} CUmemAccess_flags;

typedef enum {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM     = 0x0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1,
} CUmemAllocationGranularity_flags;

/* ── CUDA VMM Structs ────────────────────────────────────────────── */

typedef struct {
    CUmemLocationType type;
    int id;
} CUmemLocation;

typedef struct {
    CUmemAllocationType type;
    CUmemAllocationHandleType requestedHandleTypes;
    CUmemLocation location;
    void *win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
} CUmemAllocationProp;

typedef struct {
    CUmemLocation location;
    CUmemAccess_flags flags;
} CUmemAccessDesc;

/* Device attribute IDs */
#define CU_DEVICE_ATTRIBUTE_VMM_SUPPORTED       102
#define CU_DEVICE_ATTRIBUTE_HOST_NUMA_VMM       141

/* ── cuGetProcAddress Types ──────────────────────────────────────── */

typedef unsigned long long cuuint64_t;
typedef enum {
    CU_GET_PROC_ADDRESS_SUCCESS             = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND    = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2,
} CUdriverProcAddressQueryResult;

typedef CUresult (*fn_cuGetProcAddress_v2)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);

/* ── VMM Function Pointer Types ──────────────────────────────────── */

typedef CUresult (*fn_cuMemAlloc_v2)(CUdeviceptr*, size_t);
typedef CUresult (*fn_cuMemFree_v2)(CUdeviceptr);
typedef CUresult (*fn_cuMemGetInfo_v2)(size_t*, size_t*);
typedef CUresult (*fn_cuMemCreate)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
typedef CUresult (*fn_cuMemAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
typedef CUresult (*fn_cuMemMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
typedef CUresult (*fn_cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
typedef CUresult (*fn_cuMemUnmap)(CUdeviceptr, size_t);
typedef CUresult (*fn_cuMemAddressFree)(CUdeviceptr, size_t);
typedef CUresult (*fn_cuMemRelease)(CUmemGenericAllocationHandle);
typedef CUresult (*fn_cuMemGetAllocationGranularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
typedef CUresult (*fn_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*fn_cuDeviceGetAttribute)(int*, int, CUdevice);

/* ── Real Function Pointers (resolved via dlsym) ─────────────────── */

static _Atomic(fn_cuGetProcAddress_v2) real_cuGetProcAddress_v2;
static _Atomic(fn_cuMemAlloc_v2)       real_cuMemAlloc_v2;
static _Atomic(fn_cuMemFree_v2)        real_cuMemFree_v2;
static _Atomic(fn_cuMemGetInfo_v2)     real_cuMemGetInfo_v2;
static fn_cuMemCreate                  real_cuMemCreate;
static fn_cuMemAddressReserve          real_cuMemAddressReserve;
static fn_cuMemMap                     real_cuMemMap;
static fn_cuMemSetAccess               real_cuMemSetAccess;
static fn_cuMemUnmap                   real_cuMemUnmap;
static fn_cuMemAddressFree             real_cuMemAddressFree;
static fn_cuMemRelease                 real_cuMemRelease;
static fn_cuMemGetAllocationGranularity real_cuMemGetAllocationGranularity;
static fn_cuDeviceGet                  real_cuDeviceGet;
static fn_cuDeviceGetAttribute         real_cuDeviceGetAttribute;

/* ── Allocation Tracking Hash Map ────────────────────────────────── */

#define ALLOC_MAP_SIZE 4096

typedef enum {
    ALLOC_VRAM,
    ALLOC_SYSMEM,
} alloc_location_t;

typedef struct {
    CUdeviceptr                  ptr;
    size_t                       size;        /* requested size */
    size_t                       padded_size; /* rounded to granularity */
    alloc_location_t             location;
    CUmemGenericAllocationHandle phys_handle;
    int                          occupied;
} alloc_entry_t;

static alloc_entry_t    g_alloc_map[ALLOC_MAP_SIZE];
static pthread_rwlock_t g_alloc_rwlock = PTHREAD_RWLOCK_INITIALIZER;

static inline size_t alloc_map_hash(CUdeviceptr ptr) {
    uint64_t v = (uint64_t)ptr;
    v = (v >> 21) ^ (v >> 12) ^ v;
    return (size_t)(v % ALLOC_MAP_SIZE);
}

static int alloc_map_insert(CUdeviceptr ptr, size_t size, size_t padded_size,
                            alloc_location_t location, CUmemGenericAllocationHandle handle) {
    /* Caller must hold write lock */
    size_t idx = alloc_map_hash(ptr);
    for (size_t i = 0; i < ALLOC_MAP_SIZE; i++) {
        size_t slot = (idx + i) % ALLOC_MAP_SIZE;
        if (!g_alloc_map[slot].occupied) {
            g_alloc_map[slot].ptr = ptr;
            g_alloc_map[slot].size = size;
            g_alloc_map[slot].padded_size = padded_size;
            g_alloc_map[slot].location = location;
            g_alloc_map[slot].phys_handle = handle;
            g_alloc_map[slot].occupied = 1;
            return 0;
        }
    }
    return -1; /* map full */
}

static alloc_entry_t *alloc_map_lookup(CUdeviceptr ptr) {
    /* Caller must hold at least read lock */
    size_t idx = alloc_map_hash(ptr);
    for (size_t i = 0; i < ALLOC_MAP_SIZE; i++) {
        size_t slot = (idx + i) % ALLOC_MAP_SIZE;
        if (!g_alloc_map[slot].occupied)
            return NULL;
        if (g_alloc_map[slot].ptr == ptr)
            return &g_alloc_map[slot];
    }
    return NULL;
}

static void alloc_map_remove(CUdeviceptr ptr) {
    /* Caller must hold write lock. Backward-shift deletion to maintain
     * linear probing invariant — no tombstones, no orphaned entries. */
    size_t idx = alloc_map_hash(ptr);
    size_t i;
    int found = 0;

    /* Find the entry */
    for (size_t n = 0; n < ALLOC_MAP_SIZE; n++) {
        i = (idx + n) % ALLOC_MAP_SIZE;
        if (!g_alloc_map[i].occupied)
            return;
        if (g_alloc_map[i].ptr == ptr) {
            found = 1;
            break;
        }
    }
    if (!found)
        return;

    /* Backward-shift: scan forward, shifting entries back to fill the gap */
    size_t j = i;
    for (;;) {
        g_alloc_map[i].occupied = 0;

        for (;;) {
            j = (j + 1) % ALLOC_MAP_SIZE;
            if (!g_alloc_map[j].occupied)
                return; /* hit empty slot — all chains intact */

            size_t k = alloc_map_hash(g_alloc_map[j].ptr);

            /* Does entry at j need to shift to gap at i?
             * Yes if its natural hash k is NOT in the range (i, j] mod SIZE.
             * Equivalently: the gap at i is in the probe path [k, j). */
            if (i <= j) {
                if (i < k && k <= j) continue; /* k in (i,j] — no shift */
            } else { /* wrapped around */
                if (i < k || k <= j) continue; /* k in (i,j] mod SIZE */
            }
            break; /* entry at j must shift to i */
        }

        g_alloc_map[i] = g_alloc_map[j];
        i = j;
    }
}

/* ── Global State ────────────────────────────────────────────────── */

/* Protected by g_alloc_rwlock (write lock) */
static size_t  g_sysmem_used         = 0;
static size_t  g_sysmem_peak         = 0;

/* Atomics — updated from multiple threads without the alloc lock */
static _Atomic int g_sysmem_alloc_count  = 0;
static _Atomic int g_sysmem_alloc_total  = 0;
static _Atomic int g_vram_alloc_count    = 0;

/* Immutable after init */
static size_t  g_sysmem_max          = 0;
static size_t  g_granularity         = 0;
static int     g_device_id           = 0;
static int     g_log_level           = 1;
static int     g_disabled            = 0;

/* VMM init state — protected by pthread_once */
static pthread_once_t g_vmm_once     = PTHREAD_ONCE_INIT;
static int     g_vmm_supported       = 0;

/* ── Logging ─────────────────────────────────────────────────────── */

#define LOG_FALLBACK(fmt, ...) \
    do { if (g_log_level >= 1) fprintf(stderr, "[cuda-vmm-fallback] " fmt "\n", ##__VA_ARGS__); } while(0)

#define LOG_DEBUG(fmt, ...) \
    do { if (g_log_level >= 2) fprintf(stderr, "[cuda-vmm-fallback] " fmt "\n", ##__VA_ARGS__); } while(0)

/* ── Helper: Resolve a CUDA symbol via dlsym ─────────────────────── */

static void *resolve_cuda_sym(const char *name) {
    void *sym = dlsym(RTLD_NEXT, name);
    if (!sym) {
        void *lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (lib) {
            sym = dlsym(lib, name);
            dlclose(lib);
        }
    }
    return sym;
}

/* ── Intercepted Function Pointer Resolution ─────────────────────── */

static fn_cuMemAlloc_v2 get_real_cuMemAlloc_v2(void) {
    fn_cuMemAlloc_v2 fn = atomic_load_explicit(&real_cuMemAlloc_v2, memory_order_acquire);
    if (fn) return fn;
    fn = (fn_cuMemAlloc_v2)resolve_cuda_sym("cuMemAlloc_v2");
    if (fn)
        atomic_store_explicit(&real_cuMemAlloc_v2, fn, memory_order_release);
    return fn;
}

static fn_cuMemFree_v2 get_real_cuMemFree_v2(void) {
    fn_cuMemFree_v2 fn = atomic_load_explicit(&real_cuMemFree_v2, memory_order_acquire);
    if (fn) return fn;
    fn = (fn_cuMemFree_v2)resolve_cuda_sym("cuMemFree_v2");
    if (fn)
        atomic_store_explicit(&real_cuMemFree_v2, fn, memory_order_release);
    return fn;
}

static fn_cuMemGetInfo_v2 get_real_cuMemGetInfo_v2(void) {
    fn_cuMemGetInfo_v2 fn = atomic_load_explicit(&real_cuMemGetInfo_v2, memory_order_acquire);
    if (fn) return fn;
    fn = (fn_cuMemGetInfo_v2)resolve_cuda_sym("cuMemGetInfo_v2");
    if (fn)
        atomic_store_explicit(&real_cuMemGetInfo_v2, fn, memory_order_release);
    return fn;
}

/* ── VMM One-Time Initialization ─────────────────────────────────── */

static void vmm_init_once(void) {
    /* Resolve VMM function pointers */
    real_cuMemCreate = (fn_cuMemCreate)resolve_cuda_sym("cuMemCreate");
    real_cuMemAddressReserve = (fn_cuMemAddressReserve)resolve_cuda_sym("cuMemAddressReserve");
    real_cuMemMap = (fn_cuMemMap)resolve_cuda_sym("cuMemMap");
    real_cuMemSetAccess = (fn_cuMemSetAccess)resolve_cuda_sym("cuMemSetAccess");
    real_cuMemUnmap = (fn_cuMemUnmap)resolve_cuda_sym("cuMemUnmap");
    real_cuMemAddressFree = (fn_cuMemAddressFree)resolve_cuda_sym("cuMemAddressFree");
    real_cuMemRelease = (fn_cuMemRelease)resolve_cuda_sym("cuMemRelease");
    real_cuMemGetAllocationGranularity = (fn_cuMemGetAllocationGranularity)resolve_cuda_sym("cuMemGetAllocationGranularity");
    real_cuDeviceGet = (fn_cuDeviceGet)resolve_cuda_sym("cuDeviceGet");
    real_cuDeviceGetAttribute = (fn_cuDeviceGetAttribute)resolve_cuda_sym("cuDeviceGetAttribute");

    if (!real_cuMemCreate || !real_cuMemAddressReserve || !real_cuMemMap ||
        !real_cuMemSetAccess || !real_cuMemUnmap || !real_cuMemAddressFree ||
        !real_cuMemRelease || !real_cuMemGetAllocationGranularity ||
        !real_cuDeviceGet || !real_cuDeviceGetAttribute) {
        LOG_FALLBACK("VMM init FAILED: could not resolve all VMM symbols");
        return;
    }

    CUdevice dev;
    CUresult err = real_cuDeviceGet(&dev, 0);
    if (err != CUDA_SUCCESS) {
        LOG_FALLBACK("VMM init FAILED: cuDeviceGet returned %d", err);
        return;
    }
    g_device_id = dev;

    int vmm_attr = 0;
    err = real_cuDeviceGetAttribute(&vmm_attr, CU_DEVICE_ATTRIBUTE_VMM_SUPPORTED, dev);
    if (err != CUDA_SUCCESS || !vmm_attr) {
        LOG_FALLBACK("VMM init FAILED: device does not support VMM (attr=%d, err=%d)", vmm_attr, err);
        return;
    }

    int numa_vmm_attr = 0;
    err = real_cuDeviceGetAttribute(&numa_vmm_attr, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VMM, dev);
    if (err != CUDA_SUCCESS || !numa_vmm_attr) {
        LOG_FALLBACK("WARNING: HOST_NUMA VMM not reported (attr=%d, err=%d), trying anyway", numa_vmm_attr, err);
    }

    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id = 0;

    err = real_cuMemGetAllocationGranularity(&g_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (err != CUDA_SUCCESS || g_granularity == 0) {
        LOG_FALLBACK("VMM init FAILED: cuMemGetAllocationGranularity returned %d (granularity=%zu)", err, g_granularity);
        return;
    }

    /* All checks passed — publish as supported. Order matters:
     * g_granularity and g_device_id are already set above.
     * g_vmm_supported is read by other threads after pthread_once returns. */
    g_vmm_supported = 1;

    LOG_FALLBACK("VMM initialized: device=%d, granularity=%zu bytes (%.1f MB), max_sysmem=%.1f GB",
                 g_device_id, g_granularity, g_granularity / (1024.0 * 1024.0),
                 g_sysmem_max / (1024.0 * 1024.0 * 1024.0));
}

static int ensure_vmm_ready(void) {
    pthread_once(&g_vmm_once, vmm_init_once);
    return g_vmm_supported;
}

/* ── VMM Sysmem Allocation ───────────────────────────────────────── */

static CUresult vmm_alloc_sysmem(CUdeviceptr *dptr, size_t bytesize) {
    size_t padded = ((bytesize + g_granularity - 1) / g_granularity) * g_granularity;

    /* Admission check under write lock — prevents TOCTOU overcommit */
    pthread_rwlock_wrlock(&g_alloc_rwlock);
    if (g_sysmem_used + padded > g_sysmem_max) {
        pthread_rwlock_unlock(&g_alloc_rwlock);
        LOG_FALLBACK("cuMemAlloc(%zu) DENIED: sysmem limit (%.1f/%.1f GB)",
                     bytesize,
                     g_sysmem_used / (1024.0 * 1024.0 * 1024.0),
                     g_sysmem_max / (1024.0 * 1024.0 * 1024.0));
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    /* Reserve the space before releasing the lock */
    g_sysmem_used += padded;
    pthread_rwlock_unlock(&g_alloc_rwlock);

    /* 1. Create physical memory in system RAM */
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id = 0;

    CUmemGenericAllocationHandle handle;
    CUresult err = real_cuMemCreate(&handle, padded, &prop, 0);
    if (err != CUDA_SUCCESS) {
        LOG_FALLBACK("cuMemCreate FAILED: err=%d, size=%zu", err, padded);
        goto fail_unreserve;
    }

    /* 2. Reserve GPU virtual address range */
    CUdeviceptr va_ptr = 0;
    err = real_cuMemAddressReserve(&va_ptr, padded, g_granularity, 0, 0);
    if (err != CUDA_SUCCESS) {
        LOG_FALLBACK("cuMemAddressReserve FAILED: err=%d, size=%zu", err, padded);
        real_cuMemRelease(handle);
        goto fail_unreserve;
    }

    /* 3. Map virtual → physical */
    err = real_cuMemMap(va_ptr, padded, 0, handle, 0);
    if (err != CUDA_SUCCESS) {
        LOG_FALLBACK("cuMemMap FAILED: err=%d, va=0x%llx, size=%zu", err, va_ptr, padded);
        real_cuMemAddressFree(va_ptr, padded);
        real_cuMemRelease(handle);
        goto fail_unreserve;
    }

    /* 4. Grant GPU read/write access */
    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = g_device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    err = real_cuMemSetAccess(va_ptr, padded, &access, 1);
    if (err != CUDA_SUCCESS) {
        LOG_FALLBACK("cuMemSetAccess FAILED: err=%d, va=0x%llx", err, va_ptr);
        real_cuMemUnmap(va_ptr, padded);
        real_cuMemAddressFree(va_ptr, padded);
        real_cuMemRelease(handle);
        goto fail_unreserve;
    }

    /* 5. Track allocation (under write lock) */
    pthread_rwlock_wrlock(&g_alloc_rwlock);
    if (alloc_map_insert(va_ptr, bytesize, padded, ALLOC_SYSMEM, handle) != 0) {
        pthread_rwlock_unlock(&g_alloc_rwlock);
        LOG_FALLBACK("alloc_map_insert FAILED: map full");
        real_cuMemUnmap(va_ptr, padded);
        real_cuMemAddressFree(va_ptr, padded);
        real_cuMemRelease(handle);
        goto fail_unreserve;
    }
    if (g_sysmem_used > g_sysmem_peak)
        g_sysmem_peak = g_sysmem_used;
    pthread_rwlock_unlock(&g_alloc_rwlock);

    atomic_fetch_add_explicit(&g_sysmem_alloc_count, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_sysmem_alloc_total, 1, memory_order_relaxed);

    *dptr = va_ptr;

    int active = atomic_load_explicit(&g_sysmem_alloc_count, memory_order_relaxed);
    LOG_FALLBACK("cuMemAlloc(%zu) → sysmem fallback at 0x%llx (padded %zu) "
                 "[active: %d allocs, %.1f/%.1f GB sysmem]",
                 bytesize, va_ptr, padded, active,
                 g_sysmem_used / (1024.0 * 1024.0 * 1024.0),
                 g_sysmem_max / (1024.0 * 1024.0 * 1024.0));

    return CUDA_SUCCESS;

fail_unreserve:
    pthread_rwlock_wrlock(&g_alloc_rwlock);
    g_sysmem_used -= padded;
    pthread_rwlock_unlock(&g_alloc_rwlock);
    return CUDA_ERROR_OUT_OF_MEMORY;
}

/* ── cuMemAlloc_v2 Interception ──────────────────────────────────── */

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    fn_cuMemAlloc_v2 fn = get_real_cuMemAlloc_v2();
    if (!fn) {
        LOG_FALLBACK("FATAL: cannot resolve cuMemAlloc_v2");
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (g_disabled)
        return fn(dptr, bytesize);

    CUresult err = fn(dptr, bytesize);

    if (err == CUDA_SUCCESS) {
        LOG_DEBUG("cuMemAlloc(%zu) → VRAM at 0x%llx", bytesize, *dptr);
        pthread_rwlock_wrlock(&g_alloc_rwlock);
        if (alloc_map_insert(*dptr, bytesize, bytesize, ALLOC_VRAM, 0) != 0)
            LOG_DEBUG("WARNING: alloc map full, VRAM alloc at 0x%llx untracked", *dptr);
        pthread_rwlock_unlock(&g_alloc_rwlock);
        atomic_fetch_add_explicit(&g_vram_alloc_count, 1, memory_order_relaxed);
        return CUDA_SUCCESS;
    }

    if (err != CUDA_ERROR_OUT_OF_MEMORY)
        return err;

    /* ── VRAM exhausted — attempt sysmem fallback ── */

    if (!ensure_vmm_ready())
        return CUDA_ERROR_OUT_OF_MEMORY;

    return vmm_alloc_sysmem(dptr, bytesize);
}

/* ── cuMemFree_v2 Interception ───────────────────────────────────── */

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    fn_cuMemFree_v2 fn = get_real_cuMemFree_v2();
    if (!fn)
        return CUDA_ERROR_NOT_INITIALIZED;

    if (g_disabled || dptr == 0)
        return fn(dptr);

    /* Look up in allocation map */
    pthread_rwlock_wrlock(&g_alloc_rwlock);
    alloc_entry_t *entry = alloc_map_lookup(dptr);

    if (!entry || entry->location == ALLOC_VRAM) {
        if (entry)
            alloc_map_remove(dptr);
        pthread_rwlock_unlock(&g_alloc_rwlock);
        return fn(dptr);
    }

    /* Sysmem allocation — extract info and remove from map under lock */
    size_t padded = entry->padded_size;
    CUmemGenericAllocationHandle handle = entry->phys_handle;
    alloc_map_remove(dptr);
    g_sysmem_used -= padded;
    pthread_rwlock_unlock(&g_alloc_rwlock);

    /* VMM cleanup — order matters: unmap before free, free before release */
    CUresult err;

    err = real_cuMemUnmap(dptr, padded);
    if (err != CUDA_SUCCESS) {
        LOG_FALLBACK("WARNING: cuMemUnmap(0x%llx, %zu) failed: %d", dptr, padded, err);
        return err; /* Don't proceed if unmap failed */
    }

    err = real_cuMemAddressFree(dptr, padded);
    if (err != CUDA_SUCCESS)
        LOG_FALLBACK("WARNING: cuMemAddressFree(0x%llx, %zu) failed: %d", dptr, padded, err);

    err = real_cuMemRelease(handle);
    if (err != CUDA_SUCCESS)
        LOG_FALLBACK("WARNING: cuMemRelease failed: %d", err);

    atomic_fetch_sub_explicit(&g_sysmem_alloc_count, 1, memory_order_relaxed);

    int remaining = atomic_load_explicit(&g_sysmem_alloc_count, memory_order_relaxed);
    LOG_DEBUG("cuMemFree(0x%llx) → sysmem freed %zu bytes [remaining: %d allocs, %.1f GB]",
              dptr, padded, remaining,
              g_sysmem_used / (1024.0 * 1024.0 * 1024.0));

    return CUDA_SUCCESS;
}

/* ── cuMemGetInfo_v2 Interception ────────────────────────────────── */

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    fn_cuMemGetInfo_v2 fn = get_real_cuMemGetInfo_v2();
    if (!fn)
        return CUDA_ERROR_NOT_INITIALIZED;

    CUresult err = fn(free, total);
    if (err != CUDA_SUCCESS || g_disabled)
        return err;

    size_t real_free = *free;

    /* Guard against unsigned underflow */
    size_t used = g_sysmem_used; /* snapshot — may be stale, that's OK for spoofing */
    size_t sysmem_avail = (used <= g_sysmem_max) ? (g_sysmem_max - used) : 0;

    *free  = real_free + sysmem_avail;
    *total = *total + g_sysmem_max;

    LOG_DEBUG("cuMemGetInfo: real_free=%.1f GB, spoofed_free=%.1f GB (sysmem_avail=%.1f GB)",
              real_free / (1024.0 * 1024.0 * 1024.0),
              *free / (1024.0 * 1024.0 * 1024.0),
              sysmem_avail / (1024.0 * 1024.0 * 1024.0));

    return CUDA_SUCCESS;
}

/* ── Legacy v1 Wrappers ──────────────────────────────────────────── */

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult cuMemFree(CUdeviceptr dptr) {
    return cuMemFree_v2(dptr);
}

CUresult cuMemGetInfo(size_t *free, size_t *total) {
    return cuMemGetInfo_v2(free, total);
}

/* ── cuGetProcAddress_v2 Interception ────────────────────────────── */
/* This is the critical interception point. NVIDIA's bundled cudart uses
 * dlsym(handle, "cuGetProcAddress_v2") to get a function resolver, then
 * calls it to resolve ALL driver API functions. By intercepting this,
 * we redirect cuMemAlloc_v2, cuMemFree_v2, and cuMemGetInfo_v2 lookups
 * to our implementations. This is the ONLY way to intercept calls from
 * bundled CUDA runtimes (Ollama, PyTorch, etc.) that dlopen libcuda.so
 * directly and bypass LD_PRELOAD symbol search order. */

static fn_cuGetProcAddress_v2 get_real_cuGetProcAddress_v2(void) {
    fn_cuGetProcAddress_v2 fn = atomic_load_explicit(&real_cuGetProcAddress_v2, memory_order_acquire);
    if (fn) return fn;
    fn = (fn_cuGetProcAddress_v2)resolve_cuda_sym("cuGetProcAddress_v2");
    if (fn)
        atomic_store_explicit(&real_cuGetProcAddress_v2, fn, memory_order_release);
    return fn;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    fn_cuGetProcAddress_v2 fn = get_real_cuGetProcAddress_v2();
    if (!fn) return CUDA_ERROR_NOT_INITIALIZED;

    /* Let the real resolver do its work first */
    CUresult err = fn(symbol, pfn, cudaVersion, flags, symbolStatus);
    if (err != CUDA_SUCCESS || g_disabled)
        return err;

    LOG_DEBUG("cuGetProcAddress('%s', ver=%d, flags=%llu) → %p", symbol, cudaVersion, flags, *pfn);

    /* Redirect cuGetProcAddress itself — cudart re-resolves it to get the v2 version,
     * then uses the returned pointer for ALL subsequent lookups. If we don't redirect this,
     * our interception is a one-shot that gets bypassed immediately. */
    if (strcmp(symbol, "cuGetProcAddress") == 0) {
        *pfn = (void*)cuGetProcAddress_v2;
        LOG_DEBUG("cuGetProcAddress('%s') → self-redirect to shim", symbol);
        return CUDA_SUCCESS;
    }

    /* Redirect known symbols to our implementations */
    if (strcmp(symbol, "cuMemAlloc_v2") == 0 || strcmp(symbol, "cuMemAlloc") == 0) {
        /* Stash the real function pointer if we haven't yet */
        if (!atomic_load_explicit(&real_cuMemAlloc_v2, memory_order_acquire) && *pfn)
            atomic_store_explicit(&real_cuMemAlloc_v2, (fn_cuMemAlloc_v2)*pfn, memory_order_release);
        *pfn = (void*)cuMemAlloc_v2;
        LOG_DEBUG("cuGetProcAddress('%s') → redirected to shim", symbol);
    }
    else if (strcmp(symbol, "cuMemFree_v2") == 0 || strcmp(symbol, "cuMemFree") == 0) {
        if (!atomic_load_explicit(&real_cuMemFree_v2, memory_order_acquire) && *pfn)
            atomic_store_explicit(&real_cuMemFree_v2, (fn_cuMemFree_v2)*pfn, memory_order_release);
        *pfn = (void*)cuMemFree_v2;
        LOG_DEBUG("cuGetProcAddress('%s') → redirected to shim", symbol);
    }
    else if (strcmp(symbol, "cuMemGetInfo_v2") == 0 || strcmp(symbol, "cuMemGetInfo") == 0) {
        if (!atomic_load_explicit(&real_cuMemGetInfo_v2, memory_order_acquire) && *pfn)
            atomic_store_explicit(&real_cuMemGetInfo_v2, (fn_cuMemGetInfo_v2)*pfn, memory_order_release);
        *pfn = (void*)cuMemGetInfo_v2;
        LOG_DEBUG("cuGetProcAddress('%s') → redirected to shim", symbol);
    }

    return CUDA_SUCCESS;
}

/* Also intercept v1 cuGetProcAddress for completeness */
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
}

/* ── dlsym Interception ─────────────────────────────────────────── */
/* NVIDIA's bundled cudart calls dlsym(handle, "cuGetProcAddress_v2") with
 * a direct handle to libcuda.so, bypassing LD_PRELOAD search order.
 * We intercept dlsym itself to redirect cuGetProcAddress_v2 lookups to
 * our version, which then intercepts all subsequent driver API resolutions.
 *
 * We use dlvsym(RTLD_NEXT) to get the real dlsym — this works because
 * dlvsym is NOT intercepted by us, so RTLD_NEXT resolves to glibc's
 * actual dlvsym. */

static void *(*g_real_dlsym)(void*, const char*) = NULL;

static void ensure_real_dlsym(void) {
    if (g_real_dlsym) return;
    /* Try GLIBC_2.34 first (newer glibc), then 2.2.5 */
    g_real_dlsym = (void*(*)(void*,const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (!g_real_dlsym)
        g_real_dlsym = (void*(*)(void*,const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
}

void *dlsym(void *handle, const char *symbol) {
    ensure_real_dlsym();
    if (!g_real_dlsym) {
        /* Catastrophic: can't resolve real dlsym. Should never happen. */
        return NULL;
    }

    void *result = g_real_dlsym(handle, symbol);

    /* Only intercept cuGetProcAddress lookups — everything else passes through */
    if (!g_disabled && symbol) {
        if (strcmp(symbol, "cuGetProcAddress_v2") == 0 || strcmp(symbol, "cuGetProcAddress") == 0) {
            /* Stash the real cuGetProcAddress for our own use */
            if (result && !atomic_load_explicit(&real_cuGetProcAddress_v2, memory_order_acquire))
                atomic_store_explicit(&real_cuGetProcAddress_v2, (fn_cuGetProcAddress_v2)result, memory_order_release);
            LOG_DEBUG("dlsym('%s') → redirected to shim cuGetProcAddress_v2", symbol);
            return (void*)cuGetProcAddress_v2;
        }
    }

    return result;
}

/* ── Initialization ──────────────────────────────────────────────── */

__attribute__((constructor))
static void shim_init(void) {
    const char *env;

    env = getenv("CUDA_VMM_FALLBACK_DISABLE");
    if (env && atoi(env)) {
        g_disabled = 1;
        return;
    }

    env = getenv("CUDA_VMM_FALLBACK_LOG_LEVEL");
    if (env) g_log_level = atoi(env);

    env = getenv("CUDA_VMM_FALLBACK_MAX_SYSMEM");
    if (env) {
        char *endptr;
        long long val = strtoll(env, &endptr, 10);
        if (endptr != env && val > 0)
            g_sysmem_max = (size_t)val;
    }

    if (g_sysmem_max == 0) {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0)
            g_sysmem_max = ((size_t)pages * (size_t)page_size) / 2;
        else
            g_sysmem_max = 8ULL * 1024 * 1024 * 1024; /* 8 GB fallback */
    }

    LOG_FALLBACK("loaded: max_sysmem=%.1f GB, log_level=%d",
                 g_sysmem_max / (1024.0 * 1024.0 * 1024.0), g_log_level);
}

/* ── Cleanup ─────────────────────────────────────────────────────── */

__attribute__((destructor))
static void shim_fini(void) {
    if (g_disabled)
        return;

    int total_sysmem = atomic_load_explicit(&g_sysmem_alloc_total, memory_order_relaxed);
    int total_vram = atomic_load_explicit(&g_vram_alloc_count, memory_order_relaxed);

    if (total_sysmem > 0 || total_vram > 0) {
        LOG_FALLBACK("session summary: %d VRAM allocs, %d sysmem allocs (lifetime), "
                     "%.1f GB peak sysmem",
                     total_vram, total_sysmem,
                     g_sysmem_peak / (1024.0 * 1024.0 * 1024.0));
    }
}
