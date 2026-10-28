// Package frontend imports model graphs from HuggingFace config.json files,
// StableHLO MLIR text and ONNX protobuf files into the hopper term
// language of [github.com/tmc/lib/nvidia/x/egraph/hopper].
//
// Each importer lowers its input to the same small set of term operators:
// elementwise maps with FP32 lambdas, last-axis reductions, reshapes and
// x·wᵀ matmuls with weight leaves. Composite operators are expanded to one
// canonical decomposition (RMSNorm as x·rsqrt(Σx²/D+ε)·γ, SiLU as
// x·sigmoid(x)) and splat constants are folded into lambdas. [Optimize]
// then saturates the e-graph with the hopper rules plus [FuseMaps], which
// composes chains of maps and merges duplicate operands, and extracts the
// cheapest term under [hopper.Cost]. When two inputs describe the same
// computation in the same FP32 operation order, the optimized terms are
// equal up to the names of their tensor leaves ([CanonicalFingerprint]).
//
// The importers are pure Go with no dependencies outside the standard
// library and this module: [ImportONNXWire] carries its own protobuf
// wire-format reader for the ONNX messages it needs. Supported inputs:
//
//   - [ImportHFConfig]: LlamaForCausalLM, Qwen3ForCausalLM,
//     DeepseekV3ForCausalLM and NemotronHForCausalLM configs. It builds
//     one decoder layer's MLP sub-block (post-attention RMSNorm, SwiGLU or
//     squared-ReLU MLP, residual) and maps every layer-0 weight name to its
//     shape. The attention core and routed experts are listed in
//     [ImportResult.Unlowered], not lowered.
//   - [ImportStableHLO]: the pretty-printed form of one func.func using
//     add, subtract, multiply, divide, maximum, minimum, negate, abs,
//     exponential, log, sqrt, rsqrt, logistic, cosine, sine, convert,
//     constant (splat), broadcast_in_dim, reshape, reduce (add or maximum
//     over the last dimension) and dot_general (rank 2, one contracting
//     dimension).
//   - [ImportONNXWire]: a ModelProto whose graph uses MatMul, Gemm, Add,
//     Sub, Mul, Div, Neg, Abs, Exp, Log, Sqrt, Reciprocal, Sigmoid, Relu,
//     Pow (exponent 2 or 0.5), ReduceSum, ReduceMean (last axis), Cast,
//     Reshape, Identity and SimplifiedLayerNormalization.
//
// Every importer reports malformed input as an error carrying a line
// number or byte offset; none panics.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package frontend
