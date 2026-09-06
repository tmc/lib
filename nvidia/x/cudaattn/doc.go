// Package cudaattn provides causal attention CUDA source and JIT compilation.
//
// CausalFlashAttentionTemplate is a single-sequence decode kernel. Launch it
// with one block per query head, any block width, and shared memory from
// [DecodeSharedMemoryBytes]. It does not implement batch addressing.
//
// BatchedPagedAttentionTemplate provides multi-sequence paged attention. Callers
// must satisfy [Config.Validate], use a positive power-of-two thread block size,
// allocate [PagedSharedMemoryBytes], and inspect the device status after
// synchronization. The llama-infer CUDA serving path does not use this kernel.
package cudaattn
