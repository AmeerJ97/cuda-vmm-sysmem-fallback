/*
 * stochastic.c — Failure-mode fallback: deterministic → stochastic
 *
 * When deterministic tiering fails, explore alternative placement strategies
 * via Thompson sampling. Each strategy maintains a Beta posterior updated
 * from observed successes/failures.
 */

#define _GNU_SOURCE
#include "stochastic.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>

/* ── Simple xorshift64* PRNG (no libc dependencies) ─────────────── */

static uint64_t prng_state = 0x9e3779b97f4a7c15ULL;

static uint64_t xorshift64star(void) {
    prng_state ^= prng_state >> 12;
    prng_state ^= prng_state << 25;
    prng_state ^= prng_state >> 27;
    return prng_state * 0x2545F4914F6CDD1DULL;
}

static double random_uniform(void) {
    /* 53-bit precision uniform in [0,1) */
    return (double)(xorshift64star() >> 11) * (1.0 / (1ULL << 53));
}

/* Seed from time + pid for process uniqueness */
static void prng_seed(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    prng_state ^= (uint64_t)ts.tv_sec;
    prng_state ^= (uint64_t)ts.tv_nsec;
    prng_state ^= (uint64_t)getpid();
    /* Warm up */
    (void)xorshift64star();
    (void)xorshift64star();
    (void)xorshift64star();
}

/* ── Beta sampler (Thompson sampling) ─────────────────────────────
 * Beta(a,b) where a,b are positive integers.
 * Sample: X = Y/(Y+Z) where Y~Gamma(a,1), Z~Gamma(b,1)
 * Gamma(n,1) for integer n = -ln(product of n uniforms)
 */

static double sample_beta_int(int a, int b) {
    if (a <= 0 || b <= 0) return 0.5;

    double log_y = 0.0;
    for (int i = 0; i < a; i++) {
        double u = random_uniform();
        if (u <= 0.0) u = 1e-15;  /* avoid log(0) */
        log_y += -log(u);
    }

    double log_z = 0.0;
    for (int i = 0; i < b; i++) {
        double u = random_uniform();
        if (u <= 0.0) u = 1e-15;
        log_z += -log(u);
    }

    /* X = Y/(Y+Z) = 1/(1 + Z/Y) */
    double log_ratio = log_z - log_y;
    /* Clamp to avoid overflow */
    if (log_ratio > 50.0) return 0.0;
    if (log_ratio < -50.0) return 1.0;
    return 1.0 / (1.0 + exp(log_ratio));
}

/* ── Strategy state ─────────────────────────────────────────────── */

typedef struct {
    _Atomic uint32_t successes;
    _Atomic uint32_t failures;
    _Atomic uint32_t attempts;   /* total attempts (success + failure) */
} strat_state_t;

static strat_state_t g_strat[STRAT_COUNT];
static _Atomic int g_consec_failures;
static _Atomic int g_total_attempts;
static _Atomic int g_stochastic_mode;  /* 0 = off, 1 = degraded, 2 = critical */

/* Consecutive failures before engaging stochastic fallback */
#define STOCH_FALLBACK_THRESHOLD 2
#define STOCH_CRITICAL_THRESHOLD 4

/* Sysmem pressure thresholds */
#define SYSMEM_PRESSURE_WARN   0.75
#define SYSMEM_PRESSURE_CRITICAL 0.90

void stochastic_init(void) {
    prng_seed();
    stochastic_reset();
}

void stochastic_reset(void) {
    for (int i = 0; i < STRAT_COUNT; i++) {
        /* Uniform prior: Beta(1,1) = uniform. Start with 1 success, 1 failure
         * so the sampler has variance even with zero observations. */
        g_strat[i].successes = 1;
        g_strat[i].failures = 1;
        g_strat[i].attempts = 0;
    }
    g_consec_failures = 0;
    g_total_attempts = 0;
    g_stochastic_mode = 0;
}

/* ── Health checking ────────────────────────────────────────────── */

tier_health_t stochastic_check_health(size_t sysmem_used, size_t sysmem_max) {
    int consec = g_consec_failures;
    double pressure = (sysmem_max > 0) ? (double)sysmem_used / (double)sysmem_max : 0.0;

    if (consec >= STOCH_CRITICAL_THRESHOLD || pressure >= SYSMEM_PRESSURE_CRITICAL)
        return TIER_HEALTH_CRITICAL;

    if (consec >= STOCH_FALLBACK_THRESHOLD || pressure >= SYSMEM_PRESSURE_WARN)
        return TIER_HEALTH_DEGRADED;

    return TIER_HEALTH_OK;
}

bool stochastic_should_fallback(size_t bytes) {
    (void)bytes;
    int consec = g_consec_failures;
    return consec >= STOCH_FALLBACK_THRESHOLD;
}

/* ── Strategy selection ─────────────────────────────────────────── */

strat_id_t stochastic_select_strategy(void) {
    double best_sample = -1.0;
    strat_id_t best_strat = STRAT_ALL_SYSMEM;  /* safest default */

    for (int i = 0; i < STRAT_COUNT; i++) {
        int s = (int)g_strat[i].successes;
        int f = (int)g_strat[i].failures;
        double sample = sample_beta_int(s, f);

        if (sample > best_sample) {
            best_sample = sample;
            best_strat = (strat_id_t)i;
        }
    }

    atomic_fetch_add_explicit(&g_strat[best_strat].attempts, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_total_attempts, 1, memory_order_relaxed);

    return best_strat;
}

void stochastic_record_result(strat_id_t strat, bool success) {
    if (strat < 0 || strat >= STRAT_COUNT) return;

    if (success) {
        atomic_fetch_add_explicit(&g_strat[strat].successes, 1, memory_order_relaxed);
        atomic_store_explicit(&g_consec_failures, 0, memory_order_relaxed);
    } else {
        atomic_fetch_add_explicit(&g_strat[strat].failures, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_consec_failures, 1, memory_order_relaxed);
    }

    /* Update stochastic mode for monitoring */
    int consec = g_consec_failures;
    if (consec >= STOCH_CRITICAL_THRESHOLD)
        atomic_store_explicit(&g_stochastic_mode, 2, memory_order_relaxed);
    else if (consec >= STOCH_FALLBACK_THRESHOLD)
        atomic_store_explicit(&g_stochastic_mode, 1, memory_order_relaxed);
    else
        atomic_store_explicit(&g_stochastic_mode, 0, memory_order_relaxed);
}

/* ── Strategy configuration ─────────────────────────────────────── */

static const size_t HEADROOM_DEFAULT = 512ULL * 1024 * 1024;
static const size_t HEADROOM_REDUCE  = 256ULL * 1024 * 1024;
static const size_t HEADROOM_INCREASE = 1024ULL * 1024 * 1024;

size_t stochastic_headroom_for_strategy(strat_id_t strat) {
    switch (strat) {
        case STRAT_REDUCE_HEADROOM:  return HEADROOM_REDUCE;
        case STRAT_INCREASE_HEADROOM: return HEADROOM_INCREASE;
        case STRAT_ALL_SYSMEM:       return HEADROOM_DEFAULT;  /* all in sysmem, headroom irrelevant */
        case STRAT_ALL_VRAM:         return 0;  /* try to use all VRAM */
        default:                     return HEADROOM_DEFAULT;
    }
}

bool strat_forces_all_sysmem(strat_id_t strat) {
    return (strat == STRAT_ALL_SYSMEM);
}

bool strat_forces_all_vram(strat_id_t strat) {
    return (strat == STRAT_ALL_VRAM);
}

const char *stochastic_strat_name(strat_id_t strat) {
    switch (strat) {
        case STRAT_REDUCE_HEADROOM:  return "reduce_headroom";
        case STRAT_INCREASE_HEADROOM: return "increase_headroom";
        case STRAT_ALL_SYSMEM:       return "all_sysmem";
        case STRAT_ALL_VRAM:         return "all_vram";
        default:                     return "unknown";
    }
}
