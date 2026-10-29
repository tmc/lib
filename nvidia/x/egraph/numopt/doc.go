// Package numopt chooses a tensor-core precision format for each K-tile of a
// layer by trading modelled accuracy loss against modelled cycles.
//
// Each K-tile of a tcgen05 GEMM may use a different kind of tcgen05.mma, all
// accumulating into the same FP32 TMEM accumulator. Given per-tile
// sensitivity statistics ([TileSensitivity], computed by [Calibrate]),
// [SolveParetoFrontier] returns every assignment of formats to tiles that is
// not dominated in (cycles, expected loss drift), and the fastest one within
// a drift budget. The frontier is exact for the model: it is built by a
// Pareto merge over tiles with integer cycle costs, so no assignment of the
// model is better in both objectives than a frontier point. [Validate]
// rejects assignments the target cannot run. [PlanKLoop] groups the tiles of
// an assignment by format, and [EmitKLoop] emits CUDA C++ that issues one
// contiguous sub-loop of tcgen05.mma per format.
//
// Everything numeric here is a model, not a measurement:
//
//   - The drift of a tile in format f is ½·Tr(H)·σ²(f), where σ²(f) is a
//     per-format relative mean squared error for Gaussian data, grown by the
//     tile's weight kurtosis for block-scaled formats and by activation
//     dynamic range beyond the format's range. See [NoiseVariance].
//   - The cycles of a tile are the larger of its math and its operand-byte
//     time under assumed relative throughputs. See [CycleCost].
//
// Neither has been fitted to hardware or to model perplexity. The frontier
// is optimal for the model, and only as good as the model.
//
// Targets are the tcgen05 targets of [github.com/tmc/lib/nvidia/x/tcgen05]:
// sm_100a, sm_103a and sm_107a, and their f variants. The Rubin formats
// ([PrecRubin3BitLUT], [PrecRubinUE5M3]) are allowed only on sm_107a and
// sm_107f. sm_121a (GB10) is refused: it has no tcgen05 on the device.
// [EmitKLoop] uses the instruction spellings that package already emits; it
// has no spelling for UE5M3 operands and refuses them. The emitted source is
// not compiled or run by this package's default tests; TestEmitKLoopNative
// compiles it with nvcc when CUDA_NUMOPT_NATIVE=1 and a GPU lock is held.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package numopt
