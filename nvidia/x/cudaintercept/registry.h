/*
 * One caller-owned operation: tmc.replay.finite-n8.fp32.v1.
 * For i=0..7, y[i]=x[i]+w[i], x[i]+=100; w must remain unchanged.
 * The finite input population is x[i]=i+2, w[i]=5+2*i, y[i]=-8-i.
 * Historical artifact evidence does not qualify this interposer adapter.
 * Every v1 metadata-only decision forwards original, including a full match.
 * The separate v2 owner requires immutable independent admission plus a real
 * canary; its production admission table is empty. No public flag or callback
 * can authorize replacement.
 */
#ifndef CUDAINTERCEPT_REGISTRY_H
#define CUDAINTERCEPT_REGISTRY_H

#include <stdint.h>
#include "observe.h"

#define CUDAINTERCEPT_REPLAY_VERSION 1
#define CUDAINTERCEPT_REPLAY_OPERATION "tmc.replay.finite-n8.fp32.v1"
#define CUDAINTERCEPT_REPLAY_POPULATION "x=i+2,w=5+2*i,y=-8-i;i=0..7"
#define CUDAINTERCEPT_REPLAY_BASELINE "7d71d7ebc631e8644ed8f068857835dd70a2d567a07a2c306f85f9153b62d344"
#define CUDAINTERCEPT_REPLAY_CANDIDATE "040710b55474fdbfbcf703b6d8258ac0cc9776d97e82ad7b4e37dabb961f6c4a"
#define CUDAINTERCEPT_REPLAY_NVRTC "64db9c3fc128f4a57a22c314fda08831a8e3a725d439a70076a0fb98f7bea1b7"
#define CUDAINTERCEPT_REPLAY_BUILTINS "84f1c5031bfe1146375b367b2c7fb8d31ea00430be7172177eeb5d04ab813b69"
#define CUDAINTERCEPT_REPLAY_FLAGS "--gpu-architecture=sm_121 --std=c++17 --fmad=false --ftz=false"

/* Fixed-width values; device addresses below are not host pointers. */
struct cudaintercept_replay_v1 {
    uint32_t version, size;
    char operation[64], caller[64];
    uint64_t caller_generation, context_generation, module_generation;
    uint64_t context, function, stream;
    uint64_t regions[3];          /* x, y, w device addresses, disjoint */
    uint32_t bytes[3], strides[3]; /* 32 bytes each, 4-byte strides */
    uint32_t parameter_bytes[4];  /* 8,8,8,4 */
    uint32_t parameter_kinds[4];  /* 1: rw FP32 device pointer, 2: const FP32
                                  * device pointer, 3: unsigned 32-bit scalar */
    uint32_t n, dtype, layout, alignment; /* 8,1 (FP32),1 (contiguous),4 */
    uint32_t grid[3], block[3], shared_bytes, workspace_bytes;
    uint32_t capture;             /* 0 unknown, 1 ordinary noncapture, 2 capture, 3 graph execution/mutation */
    uint32_t caller_owned;        /* 1: caller asserts lifetime below; 0 unknown */
    char population[64], target[16]; /* exact finite population, sm_121 */
    char baseline[65], candidate[65], nvrtc[65], builtins[65], flags[80];
};

/*
 * Registration and binding are outside the hook. Inputs must be readable
 * host storage for the duration of the call; both APIs copy all metadata.
 * The caller asserts the context is current on this thread and retains its
 * context, modules, ordinary stream and all three buffers through actual
 * completion. Copying metadata does not establish that lifetime or contents.
 * No kernelParams/extra array or Go pointer is retained or dereferenced.
 *
 * Register installs at most one immutable entry for the process lifetime.
 * A concurrent duplicate returns ALREADY_REGISTERED; there is no retirement
 * or reuse of function handles. Generation changes refuse rather than update
 * the entry. A fresh process is needed to register a different generation.
 * Registration validates the finite contract; it does not load any artifact.
 *
 * Bind copies a caller declaration for the next cuLaunchKernel on this thread.
 * Every forwarded launch consumes it, including a mismatch or disabled call.
 * It is not inherited by a nested launch. Bind(NULL) clears the declaration.
 * Both return a decision reason (0 on success). Invalid version/size marks
 * the next forwarded launch with an invalid binding (METADATA_ABI), subject
 * to higher-priority policy and registry refusals. That launch consumes it.
 * Without a binding, an unrelated CUfunction reports REGISTRY_MISS and the
 * registered CUfunction reports NO_BINDING. Explicit capture=3 reports
 * GRAPH_EXECUTION refusal; the shim never hooks graph-launch APIs or infers
 * graph or stream state from pointers or kernels.
 * Callers must not mutate inputs concurrently with either API. These APIs
 * must not be called from a signal handler or hook.
 */
uint32_t cudaintercept_replay_register_v1(const struct cudaintercept_replay_v1 *entry);
uint32_t cudaintercept_replay_bind_v1(const struct cudaintercept_replay_v1 *call);

/*
 * Policy requests default off. A request to enable only evaluates guards;
 * it cannot admit replacement. Emergency disable and operation disable each
 * override that request. All three switches accept 0 or 1 (other values
 * return -1 without changing policy). Acquire/release atomics make changes
 * visible to subsequent decisions; an in-flight decision may use the prior
 * value. These switches issue no eligibility; v1 declarations always forward.
 * The separate owned v2 path must additionally possess its private capsule.
 */
int cudaintercept_replay_enable_v1(int enable);
int cudaintercept_process_disable_v1(int disable);
int cudaintercept_replay_disable_v1(int disable);

/* Internal shim boundary: consumes only the copied declaration, not arguments. */
uint32_t cudaintercept_replay_decide_v1(uint64_t function, uint64_t stream,
    const uint32_t grid[3], const uint32_t block[3], uint32_t shared,
    int params_present, int extra_present, int reentrant, uint64_t *generation);
void cudaintercept_decision_record_v1(uint32_t reason, uint64_t generation,
    uint64_t function, uint64_t stream, int result);

/* Private translation-unit bridge, hidden from the dynamic public API. Only
 * replay.c's concrete producer may publish/arm it after independent admission
 * and completed canary. This is not a hostile-native-code security boundary. */
#define CUDAINTERCEPT_PRIVATE __attribute__((visibility("hidden")))
struct cudaintercept_owned_capsule_v2 {
    struct cudaintercept_replay_v1 operation;
    uint64_t candidate, process;
    uint64_t modules[2];
    char driver_image[65];
    char canary[65];
    char digest[65];
};
struct cudaintercept_owned_frame_v2 {
    const struct cudaintercept_owned_capsule_v2 *owner;
    void **params;
    uint32_t refusal, decision;
};
CUDAINTERCEPT_PRIVATE uint32_t cudaintercept_owned_contract_v2(const struct cudaintercept_replay_v1 *);
CUDAINTERCEPT_PRIVATE uint32_t cudaintercept_owned_publish_v2(const struct cudaintercept_owned_capsule_v2 *);
CUDAINTERCEPT_PRIVATE int cudaintercept_owned_arm_v2(struct cudaintercept_owned_frame_v2 *);
CUDAINTERCEPT_PRIVATE void cudaintercept_owned_clear_v2(void);
CUDAINTERCEPT_PRIVATE int cudaintercept_owned_prewarm_v2(uint64_t expected_launch);
CUDAINTERCEPT_PRIVATE int cudaintercept_owned_in_hook_v2(void);
CUDAINTERCEPT_PRIVATE int cudaintercept_owned_decide_v2(uint64_t, uint64_t,
    const uint32_t[3], const uint32_t[3], uint32_t, void **, void **, int,
    uint32_t *, uint64_t *, uint64_t *);
CUDAINTERCEPT_PRIVATE void cudaintercept_owned_record_v2(uint32_t, uint64_t,
    uint64_t, uint64_t, uint64_t, int);

#endif
