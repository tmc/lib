// Package cudaref provides CPU reference operations for cudair graphs.
//
// # Graph evaluation
//
// [Evaluate] and [EvaluateSnapshot] evaluate graphs with float32 buffers. The
// evaluator supports the unary operations relu, relu_grad, gelu, gelu_grad,
// silu, exp, log, sqrt, and neg; the binary operations add, sub, mul, and div;
// matrix multiplication; sum, max, and mean reductions; and reshape, transpose,
// slice, and concat. RMSNorm and RoPE nodes are admitted by snapshots but are
// unsupported by this evaluator. An empty mean reduction returns NaN for each
// output cell; when there are no output cells, the result is empty.
//
// The float32 buffer interface does not define dtype-specific conversion or
// overflow policies.
//
// [MatMul], [Softmax], and [OnlineAttentionRow] are also available as standalone
// CPU operations. A completely masked [OnlineAttentionRow] returns a zero
// vector.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudaref
