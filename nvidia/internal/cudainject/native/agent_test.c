#include "agent.h"

#include <assert.h>
#include <string.h>

static struct cudaprof_native_identity
identity(void)
{
	struct cudaprof_native_identity id = {
		.magic = CUDAPROF_NATIVE_MAGIC,
		.protocol_version = CUDAPROF_NATIVE_PROTOCOL_VERSION,
		.abi_version = CUDAPROF_NATIVE_ABI_VERSION,
		.architecture = CUDAPROF_NATIVE_COMPILED_ARCH,
		.session_nonce = 41,
		.process_generation = 7,
		.pid = 1234,
	};
	return id;
}

int
main(void)
{
	struct cudaprof_native_slot slots[8];
	struct cudaprof_native_queue queue;
	assert(cudaprof_native_queue_init(&queue, slots, 8) == CUDAPROF_NATIVE_OK);
	struct cudaprof_native_agent agent;
	struct cudaprof_native_identity id = identity();
	assert(cudaprof_native_agent_init(&agent, &queue, &id) == CUDAPROF_NATIVE_OK);
	assert(cudaprof_native_agent_bootstrap(&agent) == CUDAPROF_NATIVE_OK);
	assert(cudaprof_native_agent_state(&agent) == CUDAPROF_NATIVE_READY);
	assert(cudaprof_native_agent_callback_enter(&agent) == CUDAPROF_NATIVE_OK);
	const char payload[] = "kernel";
	assert(cudaprof_native_agent_emit(&agent, 99, payload, sizeof(payload) - 1) == CUDAPROF_NATIVE_OK);
	assert(cudaprof_native_agent_shutdown_begin(&agent) == CUDAPROF_NATIVE_OK);
	assert(cudaprof_native_agent_shutdown_finalize(&agent) == CUDAPROF_NATIVE_BUSY);
	cudaprof_native_agent_callback_leave(&agent);
	assert(cudaprof_native_agent_callback_enter(&agent) == CUDAPROF_NATIVE_CLOSED);
	struct cudaprof_native_event event;
	while (cudaprof_native_queue_pop(&queue, &event) == CUDAPROF_NATIVE_OK)
		assert(event.identity.session_nonce == 41);
	assert(cudaprof_native_agent_shutdown_finalize(&agent) == CUDAPROF_NATIVE_OK);
	assert(cudaprof_native_agent_state(&agent) == CUDAPROF_NATIVE_FINALIZED);
	assert(cudaprof_native_agent_emit(&agent, 100, payload, sizeof(payload) - 1) == CUDAPROF_NATIVE_CLOSED);
	assert(cudaprof_native_agent_emit(&agent, 100, payload, CUDAPROF_NATIVE_MAX_PAYLOAD + 1) == CUDAPROF_NATIVE_INVALID_ARGUMENT);

	struct cudaprof_native_slot small_slots[2];
	struct cudaprof_native_queue small;
	assert(cudaprof_native_queue_init(&small, small_slots, 2) == CUDAPROF_NATIVE_OK);
	assert(cudaprof_native_queue_push(&small, &event) == CUDAPROF_NATIVE_OK);
	assert(cudaprof_native_queue_push(&small, &event) == CUDAPROF_NATIVE_OK);
	assert(cudaprof_native_queue_push(&small, &event) == CUDAPROF_NATIVE_FULL);
	assert(cudaprof_native_queue_dropped(&small) == 1);

	id.architecture = CUDAPROF_NATIVE_ARCH_UNKNOWN;
	assert(cudaprof_native_agent_init(&agent, &queue, &id) == CUDAPROF_NATIVE_INCOMPATIBLE);
	assert(strcmp(cudaprof_native_architecture_name(CUDAPROF_NATIVE_COMPILED_ARCH),
		CUDAPROF_NATIVE_COMPILED_ARCH == CUDAPROF_NATIVE_ARCH_AMD64 ? "amd64" : "arm64") == 0);
	return 0;
}
