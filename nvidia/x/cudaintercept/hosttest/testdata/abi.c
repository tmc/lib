/*
 * Compile-only ABI checks. Nothing here runs.
 *
 * The record layout section needs only observe.h. The real-header section,
 * enabled by CUDAINTERCEPT_HAVE_CUDA_H, needs the real CUDA driver cuda.h
 * and shim-abi.h, which the test generates from interpose.c:
 * SHIM_RET and SHIM_P0..SHIM_P10 are the shim typedef's return and parameter
 * types, and SHIM_NOT_INITIALIZED is its fail-closed result literal.
 */
#include <stddef.h>

#include "observe.h"

#define FIELD(name, offset) \
    _Static_assert(offsetof(struct cudaintercept_record, name) == offset, #name)

_Static_assert(CUDAINTERCEPT_RECORD_VERSION == 1, "version");
_Static_assert(sizeof(struct cudaintercept_record) == 80, "size");
_Static_assert(_Alignof(struct cudaintercept_record) == 8, "align");
FIELD(version, 0);
FIELD(size, 4);
FIELD(seq, 8);
FIELD(function, 16);
FIELD(stream, 24);
FIELD(thread, 32);
FIELD(grid, 40);
FIELD(block, 52);
FIELD(shared_mem_bytes, 64);
FIELD(flags, 68);
FIELD(result, 72);
FIELD(reserved, 76);

/* Separately versioned additive decision ABI; the launch ABI above stays v1. */
_Static_assert(CUDAINTERCEPT_DECISION_VERSION == 1, "decision version");
_Static_assert(sizeof(struct cudaintercept_decision_v1) == 48, "decision size");
_Static_assert(offsetof(struct cudaintercept_decision_v1, reason) == 16, "decision reason");
_Static_assert(offsetof(struct cudaintercept_decision_v1, result) == 20, "decision result");
_Static_assert(offsetof(struct cudaintercept_decision_v1, caller_generation) == 24, "decision generation");
_Static_assert(offsetof(struct cudaintercept_decision_v1, function) == 32, "decision function");
_Static_assert(offsetof(struct cudaintercept_decision_v1, stream) == 40, "decision stream");

#ifdef CUDAINTERCEPT_HAVE_REGISTRY_H
#include "registry.h"
_Static_assert(CUDAINTERCEPT_REPLAY_VERSION == 1, "replay version");
_Static_assert(sizeof(struct cudaintercept_replay_v1) == 744, "replay size");
_Static_assert(offsetof(struct cudaintercept_replay_v1, caller_generation) == 136, "replay generation");
_Static_assert(offsetof(struct cudaintercept_replay_v1, regions) == 184, "replay regions");
_Static_assert(offsetof(struct cudaintercept_replay_v1, parameter_kinds) == 248, "replay pointer intent");
_Static_assert(offsetof(struct cudaintercept_replay_v1, baseline) == 400, "replay baseline");
_Static_assert(offsetof(struct cudaintercept_replay_v1, candidate) == 465, "replay candidate");
typedef uint32_t (*replay_input_t)(const struct cudaintercept_replay_v1 *);
typedef int (*policy_t)(int);
typedef long (*decision_drain_t)(struct cudaintercept_decision_v1 *, size_t, uint64_t *);
#define API_TYPE(name, type) _Static_assert(__builtin_types_compatible_p(__typeof__(&name), type), #name)
API_TYPE(cudaintercept_replay_register_v1, replay_input_t);
API_TYPE(cudaintercept_replay_bind_v1, replay_input_t);
API_TYPE(cudaintercept_replay_enable_v1, policy_t);
API_TYPE(cudaintercept_process_disable_v1, policy_t);
API_TYPE(cudaintercept_replay_disable_v1, policy_t);
API_TYPE(cudaintercept_decisions_enable_v1, policy_t);
API_TYPE(cudaintercept_decision_drain_v1, decision_drain_t);
#endif

#ifdef CUDAINTERCEPT_HAVE_CUDA_H
#include <cuda.h>

#include "shim-abi.h"

#define SAME_SHAPE(a, b) (sizeof(a) == sizeof(b) && _Alignof(a) == _Alignof(b))
#define SAME_TYPE(a, b) __builtin_types_compatible_p(a, b)

/*
 * The reference parameter list, checked against the header's own
 * declaration so that it cannot drift from cuda.h.
 */
typedef CUresult reference_t(CUfunction, unsigned int, unsigned int, unsigned int,
                             unsigned int, unsigned int, unsigned int, unsigned int,
                             CUstream, void **, void **);
_Static_assert(SAME_TYPE(__typeof__(cuLaunchKernel), reference_t), "cuLaunchKernel prototype");

_Static_assert(sizeof(CUresult) == sizeof(int), "CUresult size");
_Static_assert(SAME_TYPE(SHIM_RET, int), "shim result type");
_Static_assert(CUDA_SUCCESS == 0, "CUDA_SUCCESS");
_Static_assert(CUDA_ERROR_NOT_INITIALIZED == SHIM_NOT_INITIALIZED, "fail-closed literal");

/* CUfunction and CUstream are struct pointers; the shim passes void *. */
_Static_assert(SAME_SHAPE(SHIM_P0, CUfunction), "f");
_Static_assert(SAME_TYPE(SHIM_P1, unsigned int), "gridDimX");
_Static_assert(SAME_TYPE(SHIM_P2, unsigned int), "gridDimY");
_Static_assert(SAME_TYPE(SHIM_P3, unsigned int), "gridDimZ");
_Static_assert(SAME_TYPE(SHIM_P4, unsigned int), "blockDimX");
_Static_assert(SAME_TYPE(SHIM_P5, unsigned int), "blockDimY");
_Static_assert(SAME_TYPE(SHIM_P6, unsigned int), "blockDimZ");
_Static_assert(SAME_TYPE(SHIM_P7, unsigned int), "sharedMemBytes");
_Static_assert(SAME_SHAPE(SHIM_P8, CUstream), "hStream");
_Static_assert(SAME_TYPE(SHIM_P9, void **), "kernelParams");
_Static_assert(SAME_TYPE(SHIM_P10, void **), "extra");

/*
 * Preprocessing with -E prints the symbol cuLaunchKernel expands to,
 * which is cuLaunchKernel_ptsz under CUDA_API_PER_THREAD_DEFAULT_STREAM.
 */
#define CUDAINTERCEPT_STR(x) #x
#define CUDAINTERCEPT_XSTR(x) CUDAINTERCEPT_STR(x)
__attribute__((unused)) static const char cudaintercept_symbol[] = "cudaintercept-symbol=" CUDAINTERCEPT_XSTR(cuLaunchKernel);
#endif
