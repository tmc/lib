#include "registry.h"

#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

static struct cudaintercept_replay_v1 replay;
/* 0 empty, 1 publishing, 2 immutable; no memory is reclaimed. */
static atomic_uint replay_state;
static atomic_int replay_enabled, process_disabled, replay_disabled;
static _Thread_local struct cudaintercept_replay_v1 binding;
static _Thread_local int bound;

#define TEXT(field, value) (sizeof(value) <= sizeof(e->field) && \
    memcmp(e->field, value, sizeof(value)) == 0)

static uint32_t contract(const struct cudaintercept_replay_v1 *e) {
    if (!e || e->version != CUDAINTERCEPT_REPLAY_VERSION || e->size != sizeof *e)
        return CUDAINTERCEPT_METADATA_ABI;
    if (!TEXT(operation, CUDAINTERCEPT_REPLAY_OPERATION)) return CUDAINTERCEPT_OPERATION;
    if (!e->caller[0] || !memchr(e->caller, 0, sizeof e->caller)) return CUDAINTERCEPT_CALLER;
    if (!e->caller_generation || !e->context_generation || !e->module_generation)
        return CUDAINTERCEPT_GENERATION;
    if (!e->context) return CUDAINTERCEPT_CONTEXT;
    if (!e->function) return CUDAINTERCEPT_REGISTRY_MISS;
    if (!e->stream) return CUDAINTERCEPT_STREAM;
    const uint32_t widths[4] = {8, 8, 8, 4}, kinds[4] = {1, 1, 2, 3};
    if (memcmp(e->parameter_bytes, widths, sizeof widths) ||
        memcmp(e->parameter_kinds, kinds, sizeof kinds)) return CUDAINTERCEPT_ARGUMENT_ABI;
    if (e->n != 8) return CUDAINTERCEPT_SHAPE;
    if (e->dtype != 1) return CUDAINTERCEPT_DTYPE;
    if (e->layout != 1) return CUDAINTERCEPT_LAYOUT;
    if (e->alignment != 4) return CUDAINTERCEPT_ALIGNMENT;
    for (int i = 0; i < 3; i++) {
        if (e->bytes[i] != 32) return CUDAINTERCEPT_SHAPE;
        if (e->strides[i] != 4) return CUDAINTERCEPT_LAYOUT;
        if (!e->regions[i] || e->regions[i] > UINT64_MAX - 32) return CUDAINTERCEPT_REGIONS;
        if (e->regions[i] % 4) return CUDAINTERCEPT_ALIGNMENT;
        for (int j = 0; j < i; j++)
            if (e->regions[i] < e->regions[j] + 32 && e->regions[j] < e->regions[i] + 32)
                return CUDAINTERCEPT_REGIONS;
    }
    if (e->workspace_bytes) return CUDAINTERCEPT_WORKSPACE;
    if (!TEXT(target, "sm_121")) return CUDAINTERCEPT_TARGET;
    if (!TEXT(nvrtc, CUDAINTERCEPT_REPLAY_NVRTC) || !TEXT(builtins, CUDAINTERCEPT_REPLAY_BUILTINS) ||
        !TEXT(flags, CUDAINTERCEPT_REPLAY_FLAGS)) return CUDAINTERCEPT_TOOLCHAIN;
    if (!TEXT(baseline, CUDAINTERCEPT_REPLAY_BASELINE)) return CUDAINTERCEPT_BASELINE_ARTIFACT;
    if (!TEXT(candidate, CUDAINTERCEPT_REPLAY_CANDIDATE)) return CUDAINTERCEPT_CANDIDATE_ARTIFACT;
    if (!e->capture) return CUDAINTERCEPT_CAPTURE_UNKNOWN;
    if (e->capture == 3) return CUDAINTERCEPT_GRAPH_EXECUTION;
    if (e->capture != 1) return CUDAINTERCEPT_CAPTURE;
    if (e->caller_owned != 1) return CUDAINTERCEPT_OWNERSHIP_UNKNOWN;
    if (!TEXT(population, CUDAINTERCEPT_REPLAY_POPULATION)) return CUDAINTERCEPT_POPULATION;
    const uint32_t grid[3] = {1, 1, 1}, block[3] = {32, 1, 1};
    if (memcmp(e->grid, grid, sizeof grid) || memcmp(e->block, block, sizeof block) || e->shared_bytes)
        return CUDAINTERCEPT_GEOMETRY;
    return 0;
}

uint32_t cudaintercept_replay_register_v1(const struct cudaintercept_replay_v1 *e) {
    uint32_t reason = contract(e);
    if (reason) return reason;
    unsigned empty = 0;
    if (!atomic_compare_exchange_strong_explicit(&replay_state, &empty, 1,
        memory_order_acquire, memory_order_relaxed)) return CUDAINTERCEPT_ALREADY_REGISTERED;
    replay = *e;
    atomic_store_explicit(&replay_state, 2, memory_order_release);
    return 0;
}

uint32_t cudaintercept_replay_bind_v1(const struct cudaintercept_replay_v1 *e) {
    bound = 0;
    if (!e) return 0;
    if (e->version != CUDAINTERCEPT_REPLAY_VERSION || e->size != sizeof *e) {
        bound = -1;
        return CUDAINTERCEPT_METADATA_ABI;
    }
    binding = *e;
    bound = 1;
    return 0;
}

int cudaintercept_replay_enable_v1(int v) {
    if (v != 0 && v != 1) return -1;
    atomic_store_explicit(&replay_enabled, v, memory_order_release);
    return 0;
}
int cudaintercept_process_disable_v1(int v) {
    if (v != 0 && v != 1) return -1;
    atomic_store_explicit(&process_disabled, v, memory_order_release);
    return 0;
}
int cudaintercept_replay_disable_v1(int v) {
    if (v != 0 && v != 1) return -1;
    atomic_store_explicit(&replay_disabled, v, memory_order_release);
    return 0;
}

uint32_t cudaintercept_replay_decide_v1(uint64_t function, uint64_t stream,
    const uint32_t grid[3], const uint32_t block[3], uint32_t shared,
    int params_present, int extra_present, int reentrant, uint64_t *generation) {
    int had_binding = bound == 1;
    int invalid_binding = bound < 0;
    bound = 0; /* Consume before forwarding, including recursive calls. */
    *generation = had_binding ? binding.caller_generation : 0;
    if (reentrant) return CUDAINTERCEPT_REENTRANT;
    if (atomic_load_explicit(&process_disabled, memory_order_acquire)) return CUDAINTERCEPT_PROCESS_DISABLED;
    if (!atomic_load_explicit(&replay_enabled, memory_order_acquire)) return CUDAINTERCEPT_DEFAULT_OFF;
    if (atomic_load_explicit(&replay_disabled, memory_order_acquire)) return CUDAINTERCEPT_OPERATION_DISABLED;
    unsigned state = atomic_load_explicit(&replay_state, memory_order_acquire);
    if (!state) return CUDAINTERCEPT_REGISTRY_MISS;
    if (state != 2) return CUDAINTERCEPT_REGISTRY_UNAVAILABLE;
    if (invalid_binding) return CUDAINTERCEPT_METADATA_ABI;
    if (!had_binding && function != replay.function) return CUDAINTERCEPT_REGISTRY_MISS;
    if (!had_binding) return CUDAINTERCEPT_NO_BINDING;
    const struct cudaintercept_replay_v1 *e = &binding;
    uint32_t reason = contract(e);
    if (reason) return reason;
    if (function != replay.function || function != e->function) return CUDAINTERCEPT_REGISTRY_MISS;
    if (memcmp(e->caller, replay.caller, sizeof e->caller)) return CUDAINTERCEPT_CALLER;
    if (e->caller_generation != replay.caller_generation || e->context_generation != replay.context_generation ||
        e->module_generation != replay.module_generation) return CUDAINTERCEPT_GENERATION;
    if (e->context != replay.context) return CUDAINTERCEPT_CONTEXT;
    if (stream != replay.stream || stream != e->stream) return CUDAINTERCEPT_STREAM;
    if (memcmp(e->regions, replay.regions, sizeof e->regions)) return CUDAINTERCEPT_REGIONS;
    if (memcmp(grid, e->grid, sizeof e->grid) || memcmp(block, e->block, sizeof e->block) || shared != e->shared_bytes)
        return CUDAINTERCEPT_GEOMETRY;
    if (!params_present || extra_present) return CUDAINTERCEPT_ARGUMENTS;
    /* No adapter or synchronous-dispatch contract has been qualified. */
    return CUDAINTERCEPT_ADAPTER_UNQUALIFIED;
}

struct decision_slot {
    _Atomic uint64_t commit;
    struct cudaintercept_decision_v1 record;
};
static struct decision_slot decisions[CUDAINTERCEPT_DECISION_CAPACITY];
static _Atomic uint64_t decision_head, decision_tail, decision_dropped;
static atomic_int decisions_enabled;
static pthread_mutex_t decision_drain_mu = PTHREAD_MUTEX_INITIALIZER;

int cudaintercept_decisions_enable_v1(int enable) {
    if (enable != 0 && enable != 1) return -1;
    atomic_store_explicit(&decisions_enabled, enable, memory_order_release);
    return 0;
}

void cudaintercept_decision_record_v1(uint32_t reason, uint64_t generation,
    uint64_t function, uint64_t stream, int result) {
    if (!atomic_load_explicit(&decisions_enabled, memory_order_acquire)) return;
    uint64_t h;
    for (;;) {
        uint64_t t = atomic_load_explicit(&decision_tail, memory_order_acquire);
        h = atomic_load_explicit(&decision_head, memory_order_relaxed);
        if (h == UINT64_MAX || h - t >= CUDAINTERCEPT_DECISION_CAPACITY) {
            atomic_fetch_add_explicit(&decision_dropped, 1, memory_order_relaxed);
            return;
        }
        if (atomic_compare_exchange_weak_explicit(&decision_head, &h, h + 1,
            memory_order_relaxed, memory_order_relaxed)) break;
    }
    struct decision_slot *s = &decisions[h % CUDAINTERCEPT_DECISION_CAPACITY];
    s->record = (struct cudaintercept_decision_v1){CUDAINTERCEPT_DECISION_VERSION,
        sizeof s->record, h, reason, result, generation, function, stream};
    atomic_store_explicit(&s->commit, h + 1, memory_order_release);
}

long cudaintercept_decision_drain_v1(struct cudaintercept_decision_v1 *out, size_t cap, uint64_t *dropped) {
    if (!atomic_load_explicit(&decisions_enabled, memory_order_acquire)) {
        if (dropped) *dropped = 0;
        return -1;
    }
    pthread_mutex_lock(&decision_drain_mu);
    uint64_t t = atomic_load_explicit(&decision_tail, memory_order_relaxed);
    uint64_t h = atomic_load_explicit(&decision_head, memory_order_relaxed);
    size_t n = 0;
    for (; n < cap && t != h; n++, t++) {
        struct decision_slot *s = &decisions[t % CUDAINTERCEPT_DECISION_CAPACITY];
        if (atomic_load_explicit(&s->commit, memory_order_acquire) != t + 1) break;
        out[n] = s->record;
    }
    atomic_store_explicit(&decision_tail, t, memory_order_release);
    pthread_mutex_unlock(&decision_drain_mu);
    if (dropped) *dropped = atomic_load_explicit(&decision_dropped, memory_order_relaxed);
    return (long)n;
}
