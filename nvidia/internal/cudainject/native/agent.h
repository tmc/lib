#ifndef CUDAPROF_NATIVE_AGENT_H
#define CUDAPROF_NATIVE_AGENT_H

#include <stdatomic.h>
#include <stdint.h>
#include "buffer.h"

struct cudaprof_native_agent {
	struct cudaprof_native_identity identity;
	struct cudaprof_native_queue *queue;
	_Atomic uint32_t state;
	_Atomic uint32_t active_callbacks;
	_Atomic uint32_t shutdown_requested;
	_Atomic uint64_t next_sequence;
};

const char *cudaprof_native_architecture_name(uint16_t);
int cudaprof_native_agent_init(struct cudaprof_native_agent *,
	struct cudaprof_native_queue *, const struct cudaprof_native_identity *);
int cudaprof_native_agent_bootstrap(struct cudaprof_native_agent *);
int cudaprof_native_agent_mark_no_cuda(struct cudaprof_native_agent *);
int cudaprof_native_agent_mark_injection_failure(struct cudaprof_native_agent *);
int cudaprof_native_agent_mark_cupti_unavailable(struct cudaprof_native_agent *);
int cudaprof_native_agent_mark_cupti_conflict(struct cudaprof_native_agent *);
int cudaprof_native_agent_callback_enter(struct cudaprof_native_agent *);
void cudaprof_native_agent_callback_leave(struct cudaprof_native_agent *);
int cudaprof_native_agent_emit(struct cudaprof_native_agent *, uint64_t timestamp_ns,
	const void *payload, uint32_t payload_length);
int cudaprof_native_agent_shutdown_begin(struct cudaprof_native_agent *);
int cudaprof_native_agent_shutdown_finalize(struct cudaprof_native_agent *);
uint32_t cudaprof_native_agent_state(const struct cudaprof_native_agent *);

#endif
