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
 *   CUDA_VMM_FALLBACK_POOL_SYSMEM  Optional pre-reserved host VMM pool in bytes
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
#include <errno.h>

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

/* ── NVML Types (for Ollama layer planner interception) ─────────── */
/* Ollama's Go scheduler calls nvmlDeviceGetMemoryInfo to pre-compute
 * n_gpu_layers. Without spoofing NVML, Ollama assigns overflow layers
 * to CPU before any CUDA allocation happens. */

typedef void *nvmlDevice_t;
typedef int    nvmlReturn_t;
#define NVML_SUCCESS 0

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef struct {
    unsigned int       version;
    unsigned long long total;
    unsigned long long reserved;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_v2_t;

typedef nvmlReturn_t (*fn_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*);
typedef nvmlReturn_t (*fn_nvmlDeviceGetMemoryInfo_v2)(nvmlDevice_t, nvmlMemory_v2_t*);

static fn_nvmlDeviceGetMemoryInfo    real_nvmlDeviceGetMemoryInfo;
static fn_nvmlDeviceGetMemoryInfo_v2 real_nvmlDeviceGetMemoryInfo_v2;

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
typedef CUresult (*fn_cuCtxGetDevice)(CUdevice*);

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
static fn_cuCtxGetDevice               real_cuCtxGetDevice;

/* ── Allocation Tracking Hash Map ────────────────────────────────── */

#define ALLOC_MAP_SIZE 4096

typedef enum {
    ALLOC_VRAM,
    ALLOC_SYSMEM,
    ALLOC_SPLIT,    /* first portion in VRAM, remainder in sysmem */
} alloc_location_t;

typedef struct {
    CUdeviceptr                  ptr;
    CUdevice                     device;
    size_t                       size;        /* requested size */
    size_t                       padded_size; /* total VA range (rounded to granularity) */
    alloc_location_t             location;
    CUmemGenericAllocationHandle phys_handle; /* sysmem handle (or sole handle for ALLOC_SYSMEM) */
    /* Split allocation fields — only valid when location == ALLOC_SPLIT */
    CUmemGenericAllocationHandle vram_handle; /* VRAM physical handle */
    size_t                       vram_size;   /* bytes mapped in VRAM */
    size_t                       sysmem_size; /* bytes mapped in sysmem */
    size_t                       sysmem_pool_offset; /* offset in pooled host handle */
    int                          sysmem_from_pool;
    int                          occupied;
} alloc_entry_t;

static alloc_entry_t    g_alloc_map[ALLOC_MAP_SIZE];
static pthread_rwlock_t g_alloc_rwlock = PTHREAD_RWLOCK_INITIALIZER;

#define DEVICE_CACHE_SIZE 16

typedef struct {
    CUdevice device;
    size_t granularity;
    int initialized;
    int vmm_supported;
} device_state_t;

static device_state_t g_device_states[DEVICE_CACHE_SIZE];
static pthread_mutex_t g_device_state_lock = PTHREAD_MUTEX_INITIALIZER;

#define SYSMEM_POOL_MAX_RANGES 256

typedef struct {
    size_t offset;
    size_t size;
} sysmem_pool_range_t;

typedef struct {
    CUmemGenericAllocationHandle handle;
    size_t size;
    size_t granularity;
    int initialized;
    sysmem_pool_range_t free_ranges[SYSMEM_POOL_MAX_RANGES];
    size_t free_range_count;
} sysmem_pool_t;

static sysmem_pool_t g_sysmem_pool;
static pthread_mutex_t g_sysmem_pool_lock = PTHREAD_MUTEX_INITIALIZER;

static inline size_t alloc_map_hash(CUdeviceptr ptr) {
    uint64_t v = (uint64_t)ptr;
    v = (v >> 21) ^ (v >> 12) ^ v;
    return (size_t)(v % ALLOC_MAP_SIZE);
}

static int alloc_map_insert(CUdeviceptr ptr, CUdevice device, size_t size, size_t padded_size,
                            alloc_location_t location, CUmemGenericAllocationHandle handle) {
    /* Caller must hold write lock */
    size_t idx = alloc_map_hash(ptr);
    for (size_t i = 0; i < ALLOC_MAP_SIZE; i++) {
        size_t slot = (idx + i) % ALLOC_MAP_SIZE;
        if (!g_alloc_map[slot].occupied) {
            g_alloc_map[slot].ptr = ptr;
            g_alloc_map[slot].device = device;
            g_alloc_map[slot].size = size;
            g_alloc_map[slot].padded_size = padded_size;
            g_alloc_map[slot].location = location;
            g_alloc_map[slot].phys_handle = handle;
            g_alloc_map[slot].vram_handle = 0;
            g_alloc_map[slot].vram_size = 0;
            g_alloc_map[slot].sysmem_size = 0;
            g_alloc_map[slot].sysmem_pool_offset = 0;
            g_alloc_map[slot].sysmem_from_pool = 0;
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
static size_t  g_sysmem_pool_target  = 0;
static int     g_log_level           = 1;
static int     g_disabled            = 0;

/* VMM symbol init state — protected by pthread_once */
static pthread_once_t g_vmm_once     = PTHREAD_ONCE_INIT;
static int     g_vmm_symbols_ready   = 0;

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

static size_t get_sysmem_used_snapshot(void) {
    size_t used;
    pthread_rwlock_rdlock(&g_alloc_rwlock);
    used = g_sysmem_used;
    pthread_rwlock_unlock(&g_alloc_rwlock);
    return used;
}

static size_t get_sysmem_available_snapshot(void) {
    size_t used = get_sysmem_used_snapshot();
    return (used <= g_sysmem_max) ? (g_sysmem_max - used) : 0;
}

static int sysmem_pool_add_free_range_locked(size_t offset, size_t size) {
    if (size == 0)
        return 0;

    size_t insert = 0;
    while (insert < g_sysmem_pool.free_range_count &&
           g_sysmem_pool.free_ranges[insert].offset < offset) {
        insert++;
    }

    if (g_sysmem_pool.free_range_count >= SYSMEM_POOL_MAX_RANGES)
        return -1;

    for (size_t i = g_sysmem_pool.free_range_count; i > insert; i--)
        g_sysmem_pool.free_ranges[i] = g_sysmem_pool.free_ranges[i - 1];

    g_sysmem_pool.free_ranges[insert].offset = offset;
    g_sysmem_pool.free_ranges[insert].size = size;
    g_sysmem_pool.free_range_count++;

    if (insert > 0) {
        sysmem_pool_range_t *prev = &g_sysmem_pool.free_ranges[insert - 1];
        sysmem_pool_range_t *curr = &g_sysmem_pool.free_ranges[insert];
        if (prev->offset + prev->size == curr->offset) {
            prev->size += curr->size;
            for (size_t i = insert; i + 1 < g_sysmem_pool.free_range_count; i++)
                g_sysmem_pool.free_ranges[i] = g_sysmem_pool.free_ranges[i + 1];
            g_sysmem_pool.free_range_count--;
            insert--;
        }
    }

    if (insert + 1 < g_sysmem_pool.free_range_count) {
        sysmem_pool_range_t *curr = &g_sysmem_pool.free_ranges[insert];
        sysmem_pool_range_t *next = &g_sysmem_pool.free_ranges[insert + 1];
        if (curr->offset + curr->size == next->offset) {
            curr->size += next->size;
            for (size_t i = insert + 1; i + 1 < g_sysmem_pool.free_range_count; i++)
                g_sysmem_pool.free_ranges[i] = g_sysmem_pool.free_ranges[i + 1];
            g_sysmem_pool.free_range_count--;
        }
    }

    return 0;
}

static int ensure_sysmem_pool_ready(size_t granularity) {
    if (g_sysmem_pool_target == 0)
        return 0;

    pthread_mutex_lock(&g_sysmem_pool_lock);
    if (g_sysmem_pool.initialized) {
        pthread_mutex_unlock(&g_sysmem_pool_lock);
        return 1;
    }

    size_t pool_size = ((g_sysmem_pool_target + granularity - 1) / granularity) * granularity;
    CUmemAllocationProp host_prop;
    memset(&host_prop, 0, sizeof(host_prop));
    host_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    host_prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    host_prop.location.id = 0;

    CUresult err = real_cuMemCreate(&g_sysmem_pool.handle, pool_size, &host_prop, 0);
    if (err != CUDA_SUCCESS) {
        pthread_mutex_unlock(&g_sysmem_pool_lock);
        LOG_FALLBACK("WARNING: sysmem pool reservation FAILED: err=%d, size=%zu", err, pool_size);
        return 0;
    }

    g_sysmem_pool.size = pool_size;
    g_sysmem_pool.granularity = granularity;
    g_sysmem_pool.initialized = 1;
    g_sysmem_pool.free_range_count = 1;
    g_sysmem_pool.free_ranges[0].offset = 0;
    g_sysmem_pool.free_ranges[0].size = pool_size;
    pthread_mutex_unlock(&g_sysmem_pool_lock);

    LOG_FALLBACK("reserved sysmem VMM pool: %.1f GB", pool_size / (1024.0 * 1024.0 * 1024.0));
    return 1;
}

static int sysmem_pool_alloc(size_t size, size_t *offset_out, CUmemGenericAllocationHandle *handle_out) {
    if (g_sysmem_pool_target == 0 || !offset_out || !handle_out)
        return 0;

    pthread_mutex_lock(&g_sysmem_pool_lock);
    if (!g_sysmem_pool.initialized) {
        pthread_mutex_unlock(&g_sysmem_pool_lock);
        return 0;
    }

    for (size_t i = 0; i < g_sysmem_pool.free_range_count; i++) {
        sysmem_pool_range_t *range = &g_sysmem_pool.free_ranges[i];
        if (range->size < size)
            continue;

        *offset_out = range->offset;
        *handle_out = g_sysmem_pool.handle;
        range->offset += size;
        range->size -= size;
        if (range->size == 0) {
            for (size_t j = i; j + 1 < g_sysmem_pool.free_range_count; j++)
                g_sysmem_pool.free_ranges[j] = g_sysmem_pool.free_ranges[j + 1];
            g_sysmem_pool.free_range_count--;
        }
        pthread_mutex_unlock(&g_sysmem_pool_lock);
        return 1;
    }

    pthread_mutex_unlock(&g_sysmem_pool_lock);
    return 0;
}

static void sysmem_pool_free(size_t offset, size_t size) {
    if (g_sysmem_pool_target == 0 || size == 0)
        return;

    pthread_mutex_lock(&g_sysmem_pool_lock);
    if (g_sysmem_pool.initialized &&
        sysmem_pool_add_free_range_locked(offset, size) != 0) {
        LOG_FALLBACK("WARNING: sysmem pool free-range table exhausted");
    }
    pthread_mutex_unlock(&g_sysmem_pool_lock);
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
    real_cuCtxGetDevice = (fn_cuCtxGetDevice)resolve_cuda_sym("cuCtxGetDevice");

    if (!real_cuMemCreate || !real_cuMemAddressReserve || !real_cuMemMap ||
        !real_cuMemSetAccess || !real_cuMemUnmap || !real_cuMemAddressFree ||
        !real_cuMemRelease || !real_cuMemGetAllocationGranularity ||
        !real_cuDeviceGet || !real_cuDeviceGetAttribute || !real_cuCtxGetDevice) {
        LOG_FALLBACK("VMM init FAILED: could not resolve all VMM symbols");
        return;
    }

    g_vmm_symbols_ready = 1;
}

typedef struct {
    CUdevice device;
    size_t granularity;
} vmm_device_config_t;

static int get_active_device(CUdevice *device) {
    if (!device)
        return 0;
    pthread_once(&g_vmm_once, vmm_init_once);
    if (!g_vmm_symbols_ready)
        return 0;

    CUresult ctx_err = real_cuCtxGetDevice(device);
    if (ctx_err == CUDA_SUCCESS)
        return 1;

    CUresult dev_err = real_cuDeviceGet(device, 0);
    if (dev_err == CUDA_SUCCESS) {
        LOG_FALLBACK("WARNING: cuCtxGetDevice failed (%d), falling back to device 0", ctx_err);
        return 1;
    }

    LOG_FALLBACK("VMM init FAILED: could not determine active device (ctx_err=%d, dev0_err=%d)", ctx_err, dev_err);
    return 0;
}

static int get_vmm_device_config(CUdevice device, vmm_device_config_t *config) {
    if (!config)
        return 0;

    pthread_once(&g_vmm_once, vmm_init_once);
    if (!g_vmm_symbols_ready)
        return 0;

    pthread_mutex_lock(&g_device_state_lock);

    device_state_t *slot = NULL;
    device_state_t *empty = NULL;
    for (size_t i = 0; i < DEVICE_CACHE_SIZE; i++) {
        if (g_device_states[i].initialized && g_device_states[i].device == device) {
            slot = &g_device_states[i];
            break;
        }
        if (!g_device_states[i].initialized && !empty)
            empty = &g_device_states[i];
    }

    if (!slot) {
        if (!empty) {
            pthread_mutex_unlock(&g_device_state_lock);
            LOG_FALLBACK("VMM init FAILED: device cache full");
            return 0;
        }

        int vmm_attr = 0;
        CUresult err = real_cuDeviceGetAttribute(&vmm_attr, CU_DEVICE_ATTRIBUTE_VMM_SUPPORTED, device);
        if (err != CUDA_SUCCESS || !vmm_attr) {
            pthread_mutex_unlock(&g_device_state_lock);
            LOG_FALLBACK("VMM init FAILED: device %d does not support VMM (attr=%d, err=%d)", device, vmm_attr, err);
            return 0;
        }

        int numa_vmm_attr = 0;
        err = real_cuDeviceGetAttribute(&numa_vmm_attr, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VMM, device);
        if (err != CUDA_SUCCESS || !numa_vmm_attr)
            LOG_FALLBACK("WARNING: device %d HOST_NUMA VMM not reported (attr=%d, err=%d), trying anyway", device, numa_vmm_attr, err);

        CUmemAllocationProp prop;
        memset(&prop, 0, sizeof(prop));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = 0;

        size_t granularity = 0;
        err = real_cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (err != CUDA_SUCCESS || granularity == 0) {
            pthread_mutex_unlock(&g_device_state_lock);
            LOG_FALLBACK("VMM init FAILED: device %d cuMemGetAllocationGranularity returned %d (granularity=%zu)",
                         device, err, granularity);
            return 0;
        }

        empty->device = device;
        empty->granularity = granularity;
        empty->initialized = 1;
        empty->vmm_supported = 1;
        slot = empty;

        LOG_FALLBACK("VMM initialized: device=%d, granularity=%zu bytes (%.1f MB), max_sysmem=%.1f GB",
                     device, granularity, granularity / (1024.0 * 1024.0),
                     g_sysmem_max / (1024.0 * 1024.0 * 1024.0));
    }

    config->device = slot->device;
    config->granularity = slot->granularity;
    pthread_mutex_unlock(&g_device_state_lock);
    return slot->vmm_supported;
}

/* ── VMM Split Allocation ────────────────────────────────────────── */
/* When cudaMalloc OOMs, instead of putting everything in sysmem,
 * create a SPLIT allocation: first portion in VRAM, remainder in
 * sysmem. Both mapped into a single contiguous GPU virtual address.
 * GGML sees one pointer. GPU reads VRAM at 288 GB/s, sysmem at PCIe speed. */

static CUresult vmm_alloc_split(CUdeviceptr *dptr, size_t bytesize) {
    vmm_device_config_t config;
    CUdevice device;
    if (!get_active_device(&device) || !get_vmm_device_config(device, &config))
        return CUDA_ERROR_OUT_OF_MEMORY;

    size_t granularity = config.granularity;
    size_t padded = ((bytesize + granularity - 1) / granularity) * granularity;
    CUresult err;
    CUresult result_err = CUDA_ERROR_OUT_OF_MEMORY;
    CUdeviceptr va_ptr = 0;
    CUmemGenericAllocationHandle vram_handle = 0;
    CUmemGenericAllocationHandle sysmem_handle = 0;
    int va_reserved = 0;
    int vram_created = 0;
    int sysmem_created = 0;
    int vram_mapped = 0;
    int sysmem_mapped = 0;

    /* Query real VRAM to determine split point */
    fn_cuMemGetInfo_v2 meminfo_fn = get_real_cuMemGetInfo_v2();
    size_t vram_free = 0, vram_total = 0;
    if (meminfo_fn)
        meminfo_fn(&vram_free, &vram_total);

    /* Leave 512 MB headroom for KV cache, compute buffers, driver */
    size_t headroom = 512ULL * 1024 * 1024;
    size_t vram_usable = (vram_free > headroom) ? (vram_free - headroom) : 0;
    /* Round down to granularity */
    vram_usable = (vram_usable / granularity) * granularity;

    size_t sysmem_portion = padded - vram_usable;
    size_t reserved_sysmem = 0;
    size_t sysmem_pool_offset = 0;
    int sysmem_from_pool = 0;
    if (vram_usable >= padded) {
        /* Shouldn't happen (we got here because cudaMalloc OOM'd), but handle it */
        sysmem_portion = 0;
        vram_usable = padded;
    }
    if (sysmem_portion == 0) {
        /* Everything fits in VRAM — shouldn't reach here, but just in case */
        vram_usable = padded;
    }

    /* Admission check for sysmem portion */
    if (sysmem_portion > 0) {
        pthread_rwlock_wrlock(&g_alloc_rwlock);
        if (g_sysmem_used + sysmem_portion > g_sysmem_max) {
            pthread_rwlock_unlock(&g_alloc_rwlock);
            LOG_FALLBACK("cuMemAlloc(%zu) DENIED: sysmem limit (%.1f/%.1f GB)",
                         bytesize,
                         g_sysmem_used / (1024.0 * 1024.0 * 1024.0),
                         g_sysmem_max / (1024.0 * 1024.0 * 1024.0));
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        g_sysmem_used += sysmem_portion;
        reserved_sysmem = sysmem_portion;
        pthread_rwlock_unlock(&g_alloc_rwlock);
    }

    /* 1. Reserve one contiguous VA range for the full allocation */
    err = real_cuMemAddressReserve(&va_ptr, padded, granularity, 0, 0);
    if (err != CUDA_SUCCESS) {
        LOG_FALLBACK("cuMemAddressReserve FAILED: err=%d, size=%zu", err, padded);
        result_err = err;
        goto fail_unreserve;
    }
    va_reserved = 1;

    /* 2. Create VRAM physical memory for the first portion */
    if (vram_usable > 0) {
        CUmemAllocationProp vram_prop;
        memset(&vram_prop, 0, sizeof(vram_prop));
        vram_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        vram_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        vram_prop.location.id = device;

        err = real_cuMemCreate(&vram_handle, vram_usable, &vram_prop, 0);
        if (err != CUDA_SUCCESS) {
            LOG_FALLBACK("cuMemCreate(VRAM, %zu) FAILED: err=%d — falling back to all-sysmem",
                         vram_usable, err);
            /* Fall back: put everything in sysmem if the larger reservation fits. */
            pthread_rwlock_wrlock(&g_alloc_rwlock);
            size_t extra_sysmem = padded - reserved_sysmem;
            if (g_sysmem_used + extra_sysmem > g_sysmem_max) {
                pthread_rwlock_unlock(&g_alloc_rwlock);
                result_err = CUDA_ERROR_OUT_OF_MEMORY;
                LOG_FALLBACK("cuMemAlloc(%zu) DENIED: all-sysmem fallback exceeds limit", bytesize);
                goto fail_unreserve;
            }
            g_sysmem_used += extra_sysmem;
            reserved_sysmem = padded;
            pthread_rwlock_unlock(&g_alloc_rwlock);
            sysmem_portion = padded;
            vram_usable = 0;
        } else {
            vram_created = 1;
        }
    }

    /* 3. Create sysmem physical memory for the overflow portion */
    if (sysmem_portion > 0) {
        if (ensure_sysmem_pool_ready(granularity) &&
            sysmem_pool_alloc(sysmem_portion, &sysmem_pool_offset, &sysmem_handle)) {
            sysmem_from_pool = 1;
        } else {
            CUmemAllocationProp host_prop;
            memset(&host_prop, 0, sizeof(host_prop));
            host_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            host_prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
            host_prop.location.id = 0;

            err = real_cuMemCreate(&sysmem_handle, sysmem_portion, &host_prop, 0);
            if (err != CUDA_SUCCESS) {
                LOG_FALLBACK("cuMemCreate(HOST, %zu) FAILED: err=%d", sysmem_portion, err);
                result_err = err;
                goto fail_unreserve;
            }
            sysmem_created = 1;
        }
    }

    /* 4. Map VRAM portion at the start of the VA range */
    if (vram_usable > 0) {
        err = real_cuMemMap(va_ptr, vram_usable, 0, vram_handle, 0);
        if (err != CUDA_SUCCESS) {
            LOG_FALLBACK("cuMemMap(VRAM) FAILED: err=%d", err);
            result_err = err;
            goto fail_unreserve;
        }
        vram_mapped = 1;
    }

    /* 5. Map sysmem portion right after the VRAM portion */
    if (sysmem_portion > 0) {
        err = real_cuMemMap(va_ptr + vram_usable, sysmem_portion, sysmem_pool_offset, sysmem_handle, 0);
        if (err != CUDA_SUCCESS) {
            LOG_FALLBACK("cuMemMap(HOST) FAILED: err=%d", err);
            result_err = err;
            goto fail_unreserve;
        }
        sysmem_mapped = 1;
    }

    /* 6. Grant GPU read/write access to the entire range */
    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    err = real_cuMemSetAccess(va_ptr, padded, &access, 1);
    if (err != CUDA_SUCCESS) {
        LOG_FALLBACK("cuMemSetAccess FAILED: err=%d", err);
        result_err = err;
        goto fail_unreserve;
    }

    /* 7. Track allocation */
    alloc_location_t loc_type = (vram_usable > 0 && sysmem_portion > 0) ? ALLOC_SPLIT :
                                (sysmem_portion > 0) ? ALLOC_SYSMEM : ALLOC_VRAM;

    pthread_rwlock_wrlock(&g_alloc_rwlock);
    if (alloc_map_insert(va_ptr, device, bytesize, padded, loc_type, sysmem_handle) != 0) {
        pthread_rwlock_unlock(&g_alloc_rwlock);
        LOG_FALLBACK("alloc_map_insert FAILED: map full");
        result_err = CUDA_ERROR_OUT_OF_MEMORY;
        goto fail_unreserve;
    }
    /* Store split details in the entry */
    alloc_entry_t *entry = alloc_map_lookup(va_ptr);
    if (entry) {
        entry->vram_handle = vram_handle;
        entry->vram_size = vram_usable;
        entry->sysmem_size = sysmem_portion;
        entry->sysmem_from_pool = sysmem_from_pool;
        entry->sysmem_pool_offset = sysmem_pool_offset;
    }
    if (g_sysmem_used > g_sysmem_peak)
        g_sysmem_peak = g_sysmem_used;
    pthread_rwlock_unlock(&g_alloc_rwlock);

    atomic_fetch_add_explicit(&g_sysmem_alloc_count, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_sysmem_alloc_total, 1, memory_order_relaxed);

    *dptr = va_ptr;

    LOG_FALLBACK("cuMemAlloc(%zu) → SPLIT: %.1f GB VRAM + %.1f GB sysmem at 0x%llx "
                 "[%.1f/%.1f GB sysmem used]",
                 bytesize,
                 vram_usable / (1024.0 * 1024.0 * 1024.0),
                 sysmem_portion / (1024.0 * 1024.0 * 1024.0),
                 va_ptr,
                 g_sysmem_used / (1024.0 * 1024.0 * 1024.0),
                 g_sysmem_max / (1024.0 * 1024.0 * 1024.0));

    return CUDA_SUCCESS;

fail_unreserve:
    if (sysmem_mapped)
        real_cuMemUnmap(va_ptr + vram_usable, sysmem_portion);
    if (vram_mapped)
        real_cuMemUnmap(va_ptr, vram_usable);
    if (va_reserved)
        real_cuMemAddressFree(va_ptr, padded);
    if (sysmem_from_pool)
        sysmem_pool_free(sysmem_pool_offset, sysmem_portion);
    else if (sysmem_created)
        real_cuMemRelease(sysmem_handle);
    if (vram_created)
        real_cuMemRelease(vram_handle);
    if (reserved_sysmem > 0) {
        pthread_rwlock_wrlock(&g_alloc_rwlock);
        g_sysmem_used -= reserved_sysmem;
        pthread_rwlock_unlock(&g_alloc_rwlock);
    }
    return result_err;
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
        CUdevice device = 0;
        if (!get_active_device(&device))
            device = 0;
        pthread_rwlock_wrlock(&g_alloc_rwlock);
        if (alloc_map_insert(*dptr, device, bytesize, bytesize, ALLOC_VRAM, 0) != 0)
            LOG_DEBUG("WARNING: alloc map full, VRAM alloc at 0x%llx untracked", *dptr);
        pthread_rwlock_unlock(&g_alloc_rwlock);
        atomic_fetch_add_explicit(&g_vram_alloc_count, 1, memory_order_relaxed);
        return CUDA_SUCCESS;
    }

    if (err != CUDA_ERROR_OUT_OF_MEMORY)
        return err;

    /* ── VRAM exhausted — attempt sysmem fallback ── */
    return vmm_alloc_split(dptr, bytesize);
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

    /* VMM allocation (ALLOC_SYSMEM or ALLOC_SPLIT) — extract info under lock */
    size_t padded = entry->padded_size;
    CUmemGenericAllocationHandle sysmem_handle = entry->phys_handle;
    CUmemGenericAllocationHandle vram_handle = entry->vram_handle;
    size_t vram_size = entry->vram_size;
    size_t sysmem_size = entry->sysmem_size;
    size_t sysmem_pool_offset = entry->sysmem_pool_offset;
    int sysmem_from_pool = entry->sysmem_from_pool;
    alloc_location_t loc = entry->location;
    alloc_map_remove(dptr);
    if (loc == ALLOC_SPLIT)
        g_sysmem_used -= sysmem_size;
    else
        g_sysmem_used -= padded;
    pthread_rwlock_unlock(&g_alloc_rwlock);

    /* VMM cleanup — order matters: unmap before free, free before release */
    CUresult err;

    err = real_cuMemUnmap(dptr, padded);
    if (err != CUDA_SUCCESS) {
        LOG_FALLBACK("WARNING: cuMemUnmap(0x%llx, %zu) failed: %d", dptr, padded, err);
        return err;
    }

    err = real_cuMemAddressFree(dptr, padded);
    if (err != CUDA_SUCCESS)
        LOG_FALLBACK("WARNING: cuMemAddressFree(0x%llx, %zu) failed: %d", dptr, padded, err);

    if (vram_size > 0 && vram_handle) {
        err = real_cuMemRelease(vram_handle);
        if (err != CUDA_SUCCESS)
            LOG_FALLBACK("WARNING: cuMemRelease(VRAM) failed: %d", err);
    }
    if (sysmem_from_pool) {
        sysmem_pool_free(sysmem_pool_offset, sysmem_size);
    } else if (sysmem_handle) {
        err = real_cuMemRelease(sysmem_handle);
        if (err != CUDA_SUCCESS)
            LOG_FALLBACK("WARNING: cuMemRelease(sysmem) failed: %d", err);
    }

    atomic_fetch_sub_explicit(&g_sysmem_alloc_count, 1, memory_order_relaxed);

    int remaining = atomic_load_explicit(&g_sysmem_alloc_count, memory_order_relaxed);
    LOG_DEBUG("cuMemFree(0x%llx) → sysmem freed %zu bytes [remaining: %d allocs, %.1f GB]",
              dptr, padded, remaining,
              get_sysmem_used_snapshot() / (1024.0 * 1024.0 * 1024.0));

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
    size_t sysmem_avail = get_sysmem_available_snapshot();

    *free  = real_free + sysmem_avail;

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

/* ── NVML Interception ──────────────────────────────────────────── */
/* Ollama's Go scheduler calls nvmlDeviceGetMemoryInfo via dlopen'd
 * libnvidia-ml.so.1 to decide n_gpu_layers. Spoof free/total so
 * Ollama assigns ALL layers to GPU, letting cuMemAlloc_v2 OOM trigger
 * our VMM sysmem fallback for overflow allocations. */

/* ── Real dlsym (forward declaration) ───────────────────────────── */
/* Needed here so resolve_nvml_sym can bypass our dlsym interception.
 * Full definition is in the "dlsym Interception" section below. */
static void *(*g_real_dlsym)(void*, const char*);
static void ensure_real_dlsym(void);

/* Resolve NVML symbol without going through our dlsym hook */
static void *resolve_nvml_sym(const char *name) {
    void *lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) return NULL;
    ensure_real_dlsym();
    void *sym = NULL;
    if (g_real_dlsym)
        sym = g_real_dlsym(lib, name);
    dlclose(lib);
    return sym;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
    /* Real pointer is captured by our dlsym hook when Ollama resolves it.
     * If not captured yet, resolve directly from libnvidia-ml.so.1. */
    if (!real_nvmlDeviceGetMemoryInfo)
        real_nvmlDeviceGetMemoryInfo = (fn_nvmlDeviceGetMemoryInfo)
            resolve_nvml_sym("nvmlDeviceGetMemoryInfo");
    if (!real_nvmlDeviceGetMemoryInfo || g_disabled)
        return real_nvmlDeviceGetMemoryInfo ? real_nvmlDeviceGetMemoryInfo(device, memory) : 1;

    nvmlReturn_t ret = real_nvmlDeviceGetMemoryInfo(device, memory);
    if (ret != NVML_SUCCESS) return ret;

    unsigned long long real_free = memory->free;
    unsigned long long sysmem_avail = (unsigned long long)get_sysmem_available_snapshot();
    memory->free += sysmem_avail;

    LOG_DEBUG("nvmlDeviceGetMemoryInfo: real_free=%.1f GB → spoofed_free=%.1f GB",
                 real_free / (1024.0 * 1024.0 * 1024.0),
                 memory->free / (1024.0 * 1024.0 * 1024.0));
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory) {
    if (!real_nvmlDeviceGetMemoryInfo_v2)
        real_nvmlDeviceGetMemoryInfo_v2 = (fn_nvmlDeviceGetMemoryInfo_v2)
            resolve_nvml_sym("nvmlDeviceGetMemoryInfo_v2");
    if (!real_nvmlDeviceGetMemoryInfo_v2 || g_disabled) {
        if (real_nvmlDeviceGetMemoryInfo_v2)
            return real_nvmlDeviceGetMemoryInfo_v2(device, memory);
        return 1;
    }

    nvmlReturn_t ret = real_nvmlDeviceGetMemoryInfo_v2(device, memory);
    if (ret != NVML_SUCCESS) return ret;

    unsigned long long real_free = memory->free;
    unsigned long long sysmem_avail = (unsigned long long)get_sysmem_available_snapshot();
    memory->free += sysmem_avail;

    LOG_DEBUG("nvmlDeviceGetMemoryInfo_v2: real_free=%.1f GB → spoofed_free=%.1f GB",
                 real_free / (1024.0 * 1024.0 * 1024.0),
                 memory->free / (1024.0 * 1024.0 * 1024.0));
    return NVML_SUCCESS;
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

/* g_real_dlsym declared above (forward declaration for resolve_nvml_sym) */

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

    /* Intercept cuGetProcAddress and NVML memory queries */
    if (!g_disabled) {
        if (strcmp(symbol, "cuGetProcAddress_v2") == 0 || strcmp(symbol, "cuGetProcAddress") == 0) {
            /* Stash the real cuGetProcAddress for our own use */
            if (result && !atomic_load_explicit(&real_cuGetProcAddress_v2, memory_order_acquire))
                atomic_store_explicit(&real_cuGetProcAddress_v2, (fn_cuGetProcAddress_v2)result, memory_order_release);
            LOG_DEBUG("dlsym('%s') → redirected to shim cuGetProcAddress_v2", symbol);
            return (void*)cuGetProcAddress_v2;
        }
        /* Intercept NVML memory queries — Ollama dlopen's libnvidia-ml.so.1
         * and resolves nvmlDeviceGetMemoryInfo by name */
        if (strcmp(symbol, "nvmlDeviceGetMemoryInfo") == 0) {
            if (result && !real_nvmlDeviceGetMemoryInfo)
                real_nvmlDeviceGetMemoryInfo = (fn_nvmlDeviceGetMemoryInfo)result;
            LOG_DEBUG("dlsym('%s') → redirected to shim", symbol);
            return (void*)nvmlDeviceGetMemoryInfo;
        }
        if (strcmp(symbol, "nvmlDeviceGetMemoryInfo_v2") == 0) {
            if (result && !real_nvmlDeviceGetMemoryInfo_v2)
                real_nvmlDeviceGetMemoryInfo_v2 = (fn_nvmlDeviceGetMemoryInfo_v2)result;
            LOG_DEBUG("dlsym('%s') → redirected to shim", symbol);
            return (void*)nvmlDeviceGetMemoryInfo_v2;
        }
    }

    return result;
}

/* ── Initialization ──────────────────────────────────────────────── */

__attribute__((constructor))
static void shim_init(void) {
    const char *env;
    size_t physical_ram = 0;
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0)
        physical_ram = (size_t)pages * (size_t)page_size;

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
        errno = 0;
        unsigned long long val = strtoull(env, &endptr, 10);
        if (endptr == env || *endptr != '\0' || errno == ERANGE || val == 0) {
            LOG_FALLBACK("WARNING: ignoring invalid CUDA_VMM_FALLBACK_MAX_SYSMEM='%s'", env);
        } else if (physical_ram > 0 && val > (unsigned long long)physical_ram) {
            LOG_FALLBACK("WARNING: ignoring CUDA_VMM_FALLBACK_MAX_SYSMEM='%s' above physical RAM", env);
        } else {
            g_sysmem_max = (size_t)val;
        }
    }

    env = getenv("CUDA_VMM_FALLBACK_POOL_SYSMEM");
    if (env) {
        char *endptr;
        errno = 0;
        unsigned long long val = strtoull(env, &endptr, 10);
        if (endptr == env || *endptr != '\0' || errno == ERANGE) {
            LOG_FALLBACK("WARNING: ignoring invalid CUDA_VMM_FALLBACK_POOL_SYSMEM='%s'", env);
        } else {
            g_sysmem_pool_target = (size_t)val;
        }
    }

    if (g_sysmem_max == 0) {
        if (physical_ram > 0)
            g_sysmem_max = physical_ram / 2;
        else
            g_sysmem_max = 8ULL * 1024 * 1024 * 1024; /* 8 GB fallback */
    }

    if (g_sysmem_pool_target > g_sysmem_max) {
        LOG_FALLBACK("WARNING: clamping sysmem pool from %.1f GB to max_sysmem %.1f GB",
                     g_sysmem_pool_target / (1024.0 * 1024.0 * 1024.0),
                     g_sysmem_max / (1024.0 * 1024.0 * 1024.0));
        g_sysmem_pool_target = g_sysmem_max;
    }

    /* Disable GGML unified memory — the shim replaces UVM's role.
     * With GGML_CUDA_ENABLE_UNIFIED_MEMORY set, GGML uses cudaMallocManaged
     * (which bypasses our cuMemAlloc_v2 interception) and reads /proc/meminfo
     * (which bypasses our cuMemGetInfo_v2 spoofing). Without it, GGML uses
     * cudaMalloc → cuMemAlloc_v2, and our shim catches OOM. */
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY")) {
        unsetenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
        LOG_FALLBACK("unset GGML_CUDA_ENABLE_UNIFIED_MEMORY (shim replaces UVM)");
    }

    LOG_FALLBACK("loaded: max_sysmem=%.1f GB, pool_sysmem=%.1f GB, log_level=%d",
                 g_sysmem_max / (1024.0 * 1024.0 * 1024.0),
                 g_sysmem_pool_target / (1024.0 * 1024.0 * 1024.0),
                 g_log_level);
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
