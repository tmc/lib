#include "buffer.h"

#include <stdint.h>

static int
is_power_of_two(size_t n)
{
	/* The sequence-counter algorithm needs at least two slots to
	 * distinguish a full queue from the next wrap of a one-slot queue. */
	return n >= 2 && (n & (n - 1)) == 0;
}

int
cudaprof_native_queue_init(struct cudaprof_native_queue *q,
	struct cudaprof_native_slot *slots, size_t capacity)
{
	if (q == NULL || slots == NULL || !is_power_of_two(capacity))
		return CUDAPROF_NATIVE_INVALID_ARGUMENT;
	q->slots = slots;
	q->capacity = capacity;
	atomic_init(&q->enqueue, 0);
	atomic_init(&q->dequeue, 0);
	atomic_init(&q->dropped, 0);
	for (size_t i = 0; i < capacity; i++)
		atomic_init(&slots[i].sequence, i);
	return CUDAPROF_NATIVE_OK;
}

int
cudaprof_native_queue_push(struct cudaprof_native_queue *q,
	const struct cudaprof_native_event *event)
{
	if (q == NULL || event == NULL || event->payload_length > CUDAPROF_NATIVE_MAX_PAYLOAD)
		return CUDAPROF_NATIVE_INVALID_ARGUMENT;
	size_t position = atomic_load_explicit(&q->enqueue, memory_order_relaxed);
	struct cudaprof_native_slot *slot = &q->slots[position & (q->capacity - 1)];
	size_t sequence = atomic_load_explicit(&slot->sequence, memory_order_acquire);
	intptr_t difference = (intptr_t)sequence - (intptr_t)position;
	if (difference != 0 || !atomic_compare_exchange_strong_explicit(&q->enqueue, &position,
		position + 1, memory_order_relaxed, memory_order_relaxed)) {
		/* A callback never retries: contention is accounted for as loss. */
		atomic_fetch_add_explicit(&q->dropped, 1, memory_order_relaxed);
		return CUDAPROF_NATIVE_FULL;
	}
	slot->event = *event;
	atomic_store_explicit(&slot->sequence, position + 1, memory_order_release);
	return CUDAPROF_NATIVE_OK;
}

int
cudaprof_native_queue_pop(struct cudaprof_native_queue *q,
	struct cudaprof_native_event *event)
{
	if (q == NULL || event == NULL)
		return CUDAPROF_NATIVE_INVALID_ARGUMENT;
	size_t position = atomic_load_explicit(&q->dequeue, memory_order_relaxed);
	struct cudaprof_native_slot *slot = &q->slots[position & (q->capacity - 1)];
	size_t sequence = atomic_load_explicit(&slot->sequence, memory_order_acquire);
	intptr_t difference = (intptr_t)sequence - (intptr_t)(position + 1);
	if (difference < 0)
		return CUDAPROF_NATIVE_CLOSED;
	if (difference > 0 || !atomic_compare_exchange_strong_explicit(&q->dequeue, &position,
		position + 1, memory_order_relaxed, memory_order_relaxed))
		return CUDAPROF_NATIVE_BUSY;
	*event = slot->event;
	atomic_store_explicit(&slot->sequence, position + q->capacity, memory_order_release);
	return CUDAPROF_NATIVE_OK;
}

uint64_t
cudaprof_native_queue_dropped(const struct cudaprof_native_queue *q)
{
	return q == NULL ? 0 : atomic_load_explicit(&q->dropped, memory_order_relaxed);
}

size_t
cudaprof_native_queue_pending(const struct cudaprof_native_queue *q)
{
	if (q == NULL)
		return 0;
	size_t in = atomic_load_explicit(&q->enqueue, memory_order_acquire);
	size_t out = atomic_load_explicit(&q->dequeue, memory_order_acquire);
	return in >= out ? in - out : 0;
}
