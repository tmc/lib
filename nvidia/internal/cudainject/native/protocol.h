#ifndef CUDAPROF_NATIVE_PROTOCOL_H
#define CUDAPROF_NATIVE_PROTOCOL_H

/*
 * The native agent protocol is deliberately independent of CUDA headers.  A
 * controller may reject an agent before it gives the target any work to do.
 */
#include <stdint.h>

#define CUDAPROF_NATIVE_MAGIC UINT32_C(0x4350444e) /* "CPDN" */
#define CUDAPROF_NATIVE_PROTOCOL_VERSION UINT16_C(1)
#define CUDAPROF_NATIVE_ABI_VERSION UINT16_C(1)
#define CUDAPROF_NATIVE_MAX_PAYLOAD UINT32_C(4096)

enum cudaprof_native_architecture {
	CUDAPROF_NATIVE_ARCH_UNKNOWN = 0,
	CUDAPROF_NATIVE_ARCH_AMD64 = 1,
	CUDAPROF_NATIVE_ARCH_ARM64 = 2
};

enum cudaprof_native_state {
	CUDAPROF_NATIVE_BOOTSTRAP = 1,
	CUDAPROF_NATIVE_READY = 2,
	CUDAPROF_NATIVE_CUDA_OBSERVED = 3,
	CUDAPROF_NATIVE_DRAINING = 4,
	CUDAPROF_NATIVE_FINALIZED = 5,
	CUDAPROF_NATIVE_NO_CUDA = 6,
	CUDAPROF_NATIVE_INJECTION_FAILURE = 7,
	CUDAPROF_NATIVE_TRANSPORT_FAILURE = 8,
	CUDAPROF_NATIVE_BUFFER_EXHAUSTED = 9,
	CUDAPROF_NATIVE_CUPTI_UNAVAILABLE = 10,
	CUDAPROF_NATIVE_CUPTI_CONFLICT = 11,
	CUDAPROF_NATIVE_SHUTDOWN_UNRESOLVED = 12
};

enum cudaprof_native_result {
	CUDAPROF_NATIVE_OK = 0,
	CUDAPROF_NATIVE_INVALID_ARGUMENT = 1,
	CUDAPROF_NATIVE_INCOMPATIBLE = 2,
	CUDAPROF_NATIVE_FULL = 3,
	CUDAPROF_NATIVE_BUSY = 4,
	CUDAPROF_NATIVE_CLOSED = 5,
	CUDAPROF_NATIVE_UNSUPPORTED = 6
};

struct cudaprof_native_identity {
	uint32_t magic;
	uint16_t protocol_version;
	uint16_t abi_version;
	uint16_t architecture;
	uint16_t reserved;
	uint64_t session_nonce;
	uint64_t process_generation;
	uint32_t pid;
	/* Zero means that CUPTI was not loaded or its version was not recorded. */
	uint32_t cupti_version;
};

struct cudaprof_native_event {
	struct cudaprof_native_identity identity;
	uint64_t sequence;
	uint64_t timestamp_ns;
	uint32_t state;
	uint32_t payload_length;
	uint8_t payload[CUDAPROF_NATIVE_MAX_PAYLOAD];
};

_Static_assert(sizeof(struct cudaprof_native_identity) == 40,
	"native identity ABI changed");
_Static_assert(sizeof(struct cudaprof_native_event) == 4160,
	"native event ABI changed");

#if defined(__x86_64__) || defined(_M_X64)
#define CUDAPROF_NATIVE_COMPILED_ARCH CUDAPROF_NATIVE_ARCH_AMD64
#elif defined(__aarch64__) || defined(_M_ARM64)
#define CUDAPROF_NATIVE_COMPILED_ARCH CUDAPROF_NATIVE_ARCH_ARM64
#else
#define CUDAPROF_NATIVE_COMPILED_ARCH CUDAPROF_NATIVE_ARCH_UNKNOWN
#endif

#endif
