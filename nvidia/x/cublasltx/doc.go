// Package cublasltx provides an owning cuBLASLt matrix multiplication engine.
//
// [NewEngine] allocates a handle and a 32 MiB workspace on a CUDA device.
// [Engine.MatMul], [Engine.MatMulF16], and [Engine.MatMulBF16] compute D = A*B
// for column-major matrices.
// [Engine.MatMulRowMajor], [Engine.MatMulRowMajorF16], and [Engine.MatMulRowMajorBF16]
// compute D = A*B for row-major matrices.
// [Engine.LinearRowMajor], [Engine.LinearRowMajorF16], [Engine.LinearRowMajorBF16],
// and [Engine.LinearRowMajorF16ToF32] compute transformer linear projections Y = X*W^T
// for row-major matrices without transposition overhead.
// [Engine.MatMulRaw] allows full control over compute type, scale type, data types,
// and transposed operands via [MatMulConfig].
//
// An Engine serializes host submissions, but matrix multiplications do not wait for GPU
// completion. Because calls share workspace, use an ordered stream or wait for
// earlier work to finish before submitting through another stream. A nil stream
// selects the CUDA default stream. Keep buffers alive until work completes and
// synchronize before reading results on the host or calling [Engine.Close].
// A closed Engine cannot be reused.
package cublasltx
