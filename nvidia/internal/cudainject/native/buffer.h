#ifndef CUDAPROF_NATIVE_BUFFER_H
#define CUDAPROF_NATIVE_BUFFER_H

#include <stdatomic.h>
#include <stddef.h>
#include "protocol.h"

/* A fixed-storage MPMC-safe queue. Push never waits or allocates. */
struct cudaprof_native_slot {
	_Atomic size_t sequence;
	struct cudaprof_native_event event;
};

struct cudaprof_native_queue {
	struct cudaprof_native_slot *slots;
	size_t capacity;
	_Atomic size_t enqueue;
	_Atomic size_t dequeue;
	_Atomic uint64_t dropped;
};

int cudaprof_native_queue_init(struct cudaprof_native_queue *,
	struct cudaprof_native_slot *, size_t capacity);
int cudaprof_native_queue_push(struct cudaprof_native_queue *,
	const struct cudaprof_native_event *);
int cudaprof_native_queue_pop(struct cudaprof_native_queue *,
	struct cudaprof_native_event *);
uint64_t cudaprof_native_queue_dropped(const struct cudaprof_native_queue *);
size_t cudaprof_native_queue_pending(const struct cudaprof_native_queue *);

#endif
