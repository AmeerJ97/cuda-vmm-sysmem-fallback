/*
 * stochastic.h — Failure-mode fallback: deterministic → stochastic
 *
 * When deterministic tiering fails (OOM, fragmentation, performance cliff),
 * fall back to stochastic methods that explore alternative placement strategies.
 *
 * Strategies:
 *   0: REDUCE_HEADROOM  (256 MB instead of 512 MB)
 *   1: INCREASE_HEADROOM (1024 MB instead of 512 MB)
 *   2: ALL_SYSMEM        (force 100% sysmem)
 *   3: ALL_VRAM          (try to cram everything into VRAM)
 *
 * Each strategy maintains a Beta posterior (successes, failures).
 * Thompson sampling selects the strategy with highest sampled posterior.
 */

#ifndef STOCHASTIC_H
#define STOCHASTIC_H

#include <stddef.h>
#include <stdbool.h>

/* Strategy IDs */
typedef enum {
    STRAT_REDUCE_HEADROOM = 0,
    STRAT_INCREASE_HEADROOM,
    STRAT_ALL_SYSMEM,
    STRAT_ALL_VRAM,
    STRAT_COUNT
} strat_id_t;

/* Health status */
typedef enum {
    TIER_HEALTH_OK = 0,
    TIER_HEALTH_DEGRADED,   /* 1-2 consecutive failures */
    TIER_HEALTH_CRITICAL,   /* 3+ consecutive failures or sysmem > 90% */
} tier_health_t;

/* Initialize stochastic subsystem (call once at library load) */
void stochastic_init(void);

/* Check current health based on recent failure history and sysmem pressure */
tier_health_t stochastic_check_health(size_t sysmem_used, size_t sysmem_max);

/* Decide whether to use stochastic fallback for this allocation */
bool stochastic_should_fallback(size_t bytes);

/* Select a strategy via Thompson sampling. Returns strategy ID. */
strat_id_t stochastic_select_strategy(void);

/* Record outcome of a strategy attempt */
void stochastic_record_result(strat_id_t strat, bool success);

/* Get headroom override for a strategy (0 = use default) */
size_t stochastic_headroom_for_strategy(strat_id_t strat);

/* Check if strategy forces all-sysmem or all-vram */
bool strat_forces_all_sysmem(strat_id_t strat);
bool strat_forces_all_vram(strat_id_t strat);

/* Get human-readable strategy name */
const char *stochastic_strat_name(strat_id_t strat);

/* Reset all posteriors (e.g., on process fork or model switch) */
void stochastic_reset(void);

#endif
