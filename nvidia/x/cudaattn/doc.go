// Package cudaattn provides causal attention CUDA source and JIT compilation.
//
// CausalFlashAttentionTemplate is a single-sequence decode kernel. Launch it
// with one single-thread block per query head and shared memory for seqLen
// float32 scores. It does not implement batch addressing.
//
// BatchedPagedAttentionTemplate provides the multi-sequence production path.
// Callers must satisfy [Config.Validate], use a positive power-of-two thread
// block size, allocate [PagedSharedMemoryBytes], and inspect the device status
// after synchronization.
package cudaattn
