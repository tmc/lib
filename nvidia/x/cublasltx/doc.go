// Package cublasltx provides an owning cuBLASLt matrix multiplication engine.
//
// [NewEngine] allocates a handle and a default 32 MiB workspace on a CUDA device.
// The workspace allocation can be customized or supplied by the caller using
// [WithWorkspaceBytes] or [WithWorkspace].
//
// [Engine.MatMul], [Engine.MatMulF16], and [Engine.MatMulBF16] compute D = A*B
// for column-major matrices.
// [Engine.MatMulRowMajor], [Engine.MatMulRowMajorF16], and [Engine.MatMulRowMajorBF16]
// compute D = A*B for row-major matrices.
// [Engine.LinearRowMajor], [Engine.LinearRowMajorF16], [Engine.LinearRowMajorBF16],
// and [Engine.LinearRowMajorF16ToF32] compute transformer linear projections Y = X*W^T
// for row-major matrices without transposition overhead.
// [Engine.MatMulRaw] allows full control over compute type, scale type, data types,
// transposed operands, and per-operation workspace override via [MatMulConfig].
//
// Callers that manage their own cuBLASLt descriptors can choose a kernel
// deterministically: [Heuristic] lists the heuristic's candidates,
// [AlgoConfigOf] reads a candidate's [AlgoConfig], and [Pick] selects the
// first candidate a predicate accepts, or else the first. [Version] reports
// the loaded library version, against which such a predicate is usually tuned.
//
// An Engine serializes host submissions, but matrix multiplications do not wait for GPU
// completion. When using the default shared workspace across multiple streams, host submission
// alone does not order work on the device. To run concurrent operations on separate streams safely,
// use [Engine.WithWorkspace] to derive a child Engine with dedicated workspace per stream,
// or provide a stream-local [MatMulConfig.Workspace] on each invocation.
// A nil stream selects the CUDA default stream. Keep buffers alive until work completes and
// synchronize before reading results on the host or calling [Engine.Close].
// A closed Engine cannot be reused.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cublasltx
