#include "agent.h"

#include <string.h>

const char *
cudaprof_native_architecture_name(uint16_t architecture)
{
	switch (architecture) {
	case CUDAPROF_NATIVE_ARCH_AMD64: return "amd64";
	case CUDAPROF_NATIVE_ARCH_ARM64: return "arm64";
	default: return "unknown";
	}
}

static int
valid_identity(const struct cudaprof_native_identity *identity)
{
	return identity != NULL && identity->magic == CUDAPROF_NATIVE_MAGIC &&
		identity->protocol_version == CUDAPROF_NATIVE_PROTOCOL_VERSION &&
		identity->abi_version == CUDAPROF_NATIVE_ABI_VERSION &&
		identity->architecture == CUDAPROF_NATIVE_COMPILED_ARCH &&
		identity->session_nonce != 0 && identity->process_generation != 0 &&
		identity->pid != 0;
}

int
cudaprof_native_agent_init(struct cudaprof_native_agent *agent,
	struct cudaprof_native_queue *queue, const struct cudaprof_native_identity *identity)
{
	if (agent == NULL || queue == NULL || !valid_identity(identity))
		return CUDAPROF_NATIVE_INCOMPATIBLE;
	memset(agent, 0, sizeof(*agent));
	agent->identity = *identity;
	agent->queue = queue;
	atomic_init(&agent->state, CUDAPROF_NATIVE_BOOTSTRAP);
	atomic_init(&agent->active_callbacks, 0);
	atomic_init(&agent->shutdown_requested, 0);
	atomic_init(&agent->next_sequence, 1);
	return CUDAPROF_NATIVE_OK;
}

static int
emit_state(struct cudaprof_native_agent *agent, uint32_t state)
{
	struct cudaprof_native_event event;
	memset(&event, 0, sizeof(event));
	event.identity = agent->identity;
	event.sequence = atomic_fetch_add_explicit(&agent->next_sequence, 1, memory_order_relaxed);
	event.state = state;
	int result = cudaprof_native_queue_push(agent->queue, &event);
	if (result == CUDAPROF_NATIVE_FULL)
		atomic_store_explicit(&agent->state, CUDAPROF_NATIVE_BUFFER_EXHAUSTED, memory_order_release);
	return result;
}

int
cudaprof_native_agent_bootstrap(struct cudaprof_native_agent *agent)
{
	if (agent == NULL)
		return CUDAPROF_NATIVE_INVALID_ARGUMENT;
	if (atomic_load_explicit(&agent->state, memory_order_acquire) != CUDAPROF_NATIVE_BOOTSTRAP)
		return CUDAPROF_NATIVE_BUSY;
	if (emit_state(agent, CUDAPROF_NATIVE_BOOTSTRAP) != CUDAPROF_NATIVE_OK)
		return CUDAPROF_NATIVE_FULL;
	atomic_store_explicit(&agent->state, CUDAPROF_NATIVE_READY, memory_order_release);
	return emit_state(agent, CUDAPROF_NATIVE_READY);
}

static int
set_terminal(struct cudaprof_native_agent *agent, uint32_t state)
{
	if (agent == NULL)
		return CUDAPROF_NATIVE_INVALID_ARGUMENT;
	uint32_t current = atomic_load_explicit(&agent->state, memory_order_acquire);
	if (current == CUDAPROF_NATIVE_DRAINING || current == CUDAPROF_NATIVE_FINALIZED)
		return CUDAPROF_NATIVE_CLOSED;
	atomic_store_explicit(&agent->state, state, memory_order_release);
	return emit_state(agent, state);
}

int cudaprof_native_agent_mark_no_cuda(struct cudaprof_native_agent *a) { return set_terminal(a, CUDAPROF_NATIVE_NO_CUDA); }
int cudaprof_native_agent_mark_injection_failure(struct cudaprof_native_agent *a) { return set_terminal(a, CUDAPROF_NATIVE_INJECTION_FAILURE); }
int cudaprof_native_agent_mark_cupti_unavailable(struct cudaprof_native_agent *a) { return set_terminal(a, CUDAPROF_NATIVE_CUPTI_UNAVAILABLE); }
int cudaprof_native_agent_mark_cupti_conflict(struct cudaprof_native_agent *a) { return set_terminal(a, CUDAPROF_NATIVE_CUPTI_CONFLICT); }

int
cudaprof_native_agent_callback_enter(struct cudaprof_native_agent *agent)
{
	if (agent == NULL || atomic_load_explicit(&agent->shutdown_requested, memory_order_acquire))
		return CUDAPROF_NATIVE_CLOSED;
	atomic_fetch_add_explicit(&agent->active_callbacks, 1, memory_order_acq_rel);
	if (atomic_load_explicit(&agent->shutdown_requested, memory_order_acquire)) {
		atomic_fetch_sub_explicit(&agent->active_callbacks, 1, memory_order_acq_rel);
		return CUDAPROF_NATIVE_CLOSED;
	}
	return CUDAPROF_NATIVE_OK;
}

void
cudaprof_native_agent_callback_leave(struct cudaprof_native_agent *agent)
{
	if (agent != NULL)
		atomic_fetch_sub_explicit(&agent->active_callbacks, 1, memory_order_acq_rel);
}

int
cudaprof_native_agent_emit(struct cudaprof_native_agent *agent, uint64_t timestamp_ns,
	const void *payload, uint32_t payload_length)
{
	if (agent == NULL || (payload_length != 0 && payload == NULL) ||
		payload_length > CUDAPROF_NATIVE_MAX_PAYLOAD)
		return CUDAPROF_NATIVE_INVALID_ARGUMENT;
	if (atomic_load_explicit(&agent->shutdown_requested, memory_order_acquire))
		return CUDAPROF_NATIVE_CLOSED;
	struct cudaprof_native_event event;
	memset(&event, 0, sizeof(event));
	event.identity = agent->identity;
	event.sequence = atomic_fetch_add_explicit(&agent->next_sequence, 1, memory_order_relaxed);
	event.timestamp_ns = timestamp_ns;
	event.state = CUDAPROF_NATIVE_CUDA_OBSERVED;
	event.payload_length = payload_length;
	if (payload_length != 0)
		memcpy(event.payload, payload, payload_length);
	atomic_store_explicit(&agent->state, CUDAPROF_NATIVE_CUDA_OBSERVED, memory_order_release);
	int result = cudaprof_native_queue_push(agent->queue, &event);
	if (result == CUDAPROF_NATIVE_FULL)
		atomic_store_explicit(&agent->state, CUDAPROF_NATIVE_BUFFER_EXHAUSTED, memory_order_release);
	return result;
}

int
cudaprof_native_agent_shutdown_begin(struct cudaprof_native_agent *agent)
{
	if (agent == NULL)
		return CUDAPROF_NATIVE_INVALID_ARGUMENT;
	atomic_store_explicit(&agent->shutdown_requested, 1, memory_order_release);
	atomic_store_explicit(&agent->state, CUDAPROF_NATIVE_DRAINING, memory_order_release);
	return emit_state(agent, CUDAPROF_NATIVE_DRAINING);
}

int
cudaprof_native_agent_shutdown_finalize(struct cudaprof_native_agent *agent)
{
	if (agent == NULL)
		return CUDAPROF_NATIVE_INVALID_ARGUMENT;
	if (!atomic_load_explicit(&agent->shutdown_requested, memory_order_acquire))
		return CUDAPROF_NATIVE_BUSY;
	if (atomic_load_explicit(&agent->active_callbacks, memory_order_acquire) != 0 ||
		cudaprof_native_queue_pending(agent->queue) != 0) {
		atomic_store_explicit(&agent->state, CUDAPROF_NATIVE_SHUTDOWN_UNRESOLVED, memory_order_release);
		return CUDAPROF_NATIVE_BUSY;
	}
	atomic_store_explicit(&agent->state, CUDAPROF_NATIVE_FINALIZED, memory_order_release);
	return CUDAPROF_NATIVE_OK;
}

uint32_t
cudaprof_native_agent_state(const struct cudaprof_native_agent *agent)
{
	return agent == NULL ? CUDAPROF_NATIVE_INJECTION_FAILURE :
		atomic_load_explicit(&agent->state, memory_order_acquire);
}
