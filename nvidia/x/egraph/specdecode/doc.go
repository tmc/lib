// Package specdecode generates Hopper and Blackwell CUDA C++ kernels for
// speculative decoding, tree draft verification, grammar-constrained token
// sampling, and adaptive-depth early exit.
//
// Speculative decoding and verification:
//   - [SpeculativeDecodeController] verifies a linear chain of draft tokens
//     against target tokens on the device and sets a CUDA graph conditional
//     handle to continue or terminate the decode loop without CPU round-trips.
//   - [DraftTreeTopology] and [BuildDraftTreeTopology] validate a draft token
//     tree and compute the ancestor masks for [hopper.MTPPagedAttention].
//   - [AcceptDraftTreeGreedy] and [AcceptDraftTreeSampled] verify draft trees
//     under greedy decoding and multi-candidate speculative sampling
//     (SpecInfer), using [Philox4x32] for reproducible device-matching draws.
//   - [TreeSpecVerify] emits device-resident page reservation and draft-tree
//     verification kernels for a conditional WHILE graph loop.
//
// Grammar-constrained sampling and adaptive early exit:
//   - [GrammarBitmask], [SamplingParams], [SampleTokenFused], and
//     [GrammarSampler] sample tokens under a packed XGrammar/llguidance
//     vocabulary bitmask with repetition penalty, temperature, min-p, and
//     rejection-sampled top-k/top-p using the Gumbel-max trick.
//   - [EarlyExitConfig] and [SimulateAdaptiveDepthLayer] implement
//     token-adaptive early exit (hidden-state cosine similarity, probe entropy,
//     or top-2 logit margin) with warp-ballot token compaction and optional
//     Blackwell Cluster Launch Control tile cancellation.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package specdecode
