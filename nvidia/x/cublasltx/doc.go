// Package cublasltx provides an owning cuBLASLt matrix multiplication engine.
//
// [NewEngine] allocates a handle and a 32 MiB workspace on a CUDA device.
// [Engine.MatMul] computes D = A*B for packed column-major float32 matrices:
// A has shape m by k, B has shape k by n, and D has shape m by n.
// The input and output slices remain owned by the caller.
//
// An Engine serializes host submissions, but MatMul does not wait for GPU
// completion. Because calls share workspace, use an ordered stream or wait for
// earlier work to finish before submitting through another stream. A nil stream
// selects the CUDA default stream. Keep buffers alive until work completes and
// synchronize before reading results on the host or calling [Engine.Close].
// A closed Engine cannot be reused.
package cublasltx
