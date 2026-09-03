// SPDX-License-Identifier: Apache-2.0

// Package kvcache provides logical KV block leases, radix-tree prefix caching,
// sliding-window attention sinks, and chunked prefill scheduling.
//
// Store coordinates block identity and lifetime without prescribing a memory
// topology. Backend packages remain responsible for allocation, synchronization,
// transfer, and reclamation of CUDA device memory or Metal unified memory.
package kvcache
