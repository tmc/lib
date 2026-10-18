/*
 * Interposition shim for the libcuda driver symbol cuLaunchKernel.
 *
 * It covers only calls that bind cuLaunchKernel by symbol name.
 * cuLaunchKernel_ptsz, cuLaunchKernelEx, and launches whose entry point
 * was resolved through cuGetProcAddress (as libcudart does) bypass it.
 *
 * The opt-in launch observer is described in observe.h.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "observe.h"

typedef int (*cuLaunchKernel_t)(void* f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, void* hStream, void** kernelParams, void** extra);

static cuLaunchKernel_t real_cuLaunchKernel = NULL;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static _Thread_local int resolving;

static void resolve_symbol(void) {
    if (resolving) return;
    resolving = 1;
    real_cuLaunchKernel = (cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
    resolving = 0;
}

/*
 * The observer ring. observe_head counts claimed seqs and observe_tail counts
 * drained ones; both only increase, and tail <= head. A launch claims seq h,
 * and with it slot h % CUDAINTERCEPT_CAPACITY, only while
 * h - tail < CUDAINTERCEPT_CAPACITY, so a slot is never claimed while it
 * holds an undrained record or a drain is copying it. Once a slot's record
 * is written, its commit word is set to seq+1. No seq is ever UINT64_MAX, so
 * seq+1 never wraps to 0, the value of a never-used slot, and a commit word
 * left from an earlier lap never equals the seq+1 a drain expects.
 *
 * Launches claim with C11 atomics rather than a mutex so that recording
 * never blocks a launch. Drains, which are off the launch path, take
 * drain_mu so that only one drain advances observe_tail at a time.
 */
struct observe_slot {
    _Atomic uint64_t commit;
    struct cudaintercept_record record;
};

static pthread_once_t observe_once = PTHREAD_ONCE_INIT;
static int observing;
static struct observe_slot observe_ring[CUDAINTERCEPT_CAPACITY];
static _Atomic uint64_t observe_head;
static _Atomic uint64_t observe_tail;
static _Atomic uint64_t observe_dropped;
static _Atomic uint64_t observe_threads;
static _Thread_local uint64_t observe_thread;
static pthread_mutex_t drain_mu = PTHREAD_MUTEX_INITIALIZER;

static void observe_init(void) {
    const char *v = getenv("CUDAINTERCEPT_OBSERVE");
    observing = v != NULL && strcmp(v, "1") == 0;
}

/* observe records one forwarded launch. It calls no function. */
static void observe(void* f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                    unsigned int sharedMemBytes, void* hStream, void** kernelParams, void** extra,
                    int result) {
    uint64_t h;
    for (;;) {
        /* Load tail before head: tail never exceeds the head loaded after it. */
        uint64_t t = atomic_load_explicit(&observe_tail, memory_order_acquire);
        h = atomic_load_explicit(&observe_head, memory_order_relaxed);
        if (h == UINT64_MAX || h - t >= CUDAINTERCEPT_CAPACITY) {
            atomic_fetch_add_explicit(&observe_dropped, 1, memory_order_relaxed);
            return;
        }
        if (atomic_compare_exchange_weak_explicit(&observe_head, &h, h + 1,
                                                  memory_order_relaxed, memory_order_relaxed))
            break;
    }
    if (observe_thread == 0)
        observe_thread = atomic_fetch_add_explicit(&observe_threads, 1, memory_order_relaxed) + 1;
    struct observe_slot *s = &observe_ring[h % CUDAINTERCEPT_CAPACITY];
    struct cudaintercept_record *r = &s->record;
    r->version = CUDAINTERCEPT_RECORD_VERSION;
    r->size = sizeof *r;
    r->seq = h;
    r->function = (uint64_t)(uintptr_t)f;
    r->stream = (uint64_t)(uintptr_t)hStream;
    r->thread = observe_thread;
    r->grid[0] = gridDimX;
    r->grid[1] = gridDimY;
    r->grid[2] = gridDimZ;
    r->block[0] = blockDimX;
    r->block[1] = blockDimY;
    r->block[2] = blockDimZ;
    r->shared_mem_bytes = sharedMemBytes;
    r->flags = CUDAINTERCEPT_ARGUMENTS_UNAVAILABLE |
               (kernelParams ? CUDAINTERCEPT_PARAMS_PRESENT : 0) |
               (extra ? CUDAINTERCEPT_EXTRA_PRESENT : 0);
    r->result = result;
    r->reserved = 0;
    atomic_store_explicit(&s->commit, h + 1, memory_order_release);
}

long cudaintercept_drain(struct cudaintercept_record *out, size_t cap, uint64_t *dropped) {
    pthread_once(&observe_once, observe_init);
    if (!observing) {
        if (dropped) *dropped = 0;
        return -1;
    }
    pthread_mutex_lock(&drain_mu);
    uint64_t t = atomic_load_explicit(&observe_tail, memory_order_relaxed);
    uint64_t h = atomic_load_explicit(&observe_head, memory_order_relaxed);
    size_t n = 0;
    for (; n < cap && t != h; n++, t++) {
        struct observe_slot *s = &observe_ring[t % CUDAINTERCEPT_CAPACITY];
        if (atomic_load_explicit(&s->commit, memory_order_acquire) != t + 1)
            break; /* still being recorded */
        out[n] = s->record;
    }
    atomic_store_explicit(&observe_tail, t, memory_order_release);
    pthread_mutex_unlock(&drain_mu);
    if (dropped) *dropped = atomic_load_explicit(&observe_dropped, memory_order_relaxed);
    return (long)n;
}

int cuLaunchKernel(void* f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                   unsigned int sharedMemBytes, void* hStream, void** kernelParams, void** extra) {
    /* pthread_once cannot be reentered while this thread resolves the symbol. */
    if (resolving) return 3; /* CUDA_ERROR_NOT_INITIALIZED: fail closed. */
    pthread_once(&resolve_once, resolve_symbol);
    if (!real_cuLaunchKernel) return 3; /* CUDA_ERROR_NOT_INITIALIZED: fail closed. */
    int result = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    /* Observation follows forwarding and cannot change the result. */
    pthread_once(&observe_once, observe_init);
    if (observing)
        observe(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream, kernelParams, extra, result);
    return result;
}

/*
 * The exported definition must have exactly the type cuLaunchKernel_t
 * names. The ABI tests check cuLaunchKernel_t against cuda.h, so this ties
 * that check to the symbol the shim exports.
 */
_Static_assert(__builtin_types_compatible_p(__typeof__(&cuLaunchKernel), cuLaunchKernel_t),
               "cuLaunchKernel definition differs from cuLaunchKernel_t");
