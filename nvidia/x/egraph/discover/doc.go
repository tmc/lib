// Package discover implements automated synthesis, exhaustive verification,
// mutant rejection, and hardware-cost ranking of PTX/SASS sub-byte micro-idioms
// and macro tile/warp-specialization schedules across Hopper (sm_90a),
// Blackwell (sm_100a, sm_103a, sm_120a, sm_121a), and Vera Rubin (sm_107a, sm_107f).
//
// Micro-idiom discovery ([DiscoverSubByteIdiom]) enumerates target-legal PTX/SASS
// instruction sequences for 4-bit E2M1 dequantization, fused dynamic block-scale
// quantization, and residual-activation-quantization epilogues. Every candidate
// is verified exhaustively over all 16 E2M1 nibbles (0x0..0xF) and all 256 packed
// byte pairs (0x00..0xFF) against IEEE/OCP reference evaluators, checked for
// 1-bit constant/LUT mutant rejection, filtered against target ISA legality, and
// ranked along a cycle-and-register Pareto frontier.
//
// Macro schedule discovery ([DiscoverMacroSchedule]) explores tile geometry,
// pipeline depth, CTA cluster group size, Tensor Memory (TMEM) column allocation,
// and warp-specialized register budgets, adapting schedules in response to
// [OnlineStallFeedback] and persisting results in a JSON [Rulebook].
//
// Program search ([RunMCTSDiscovery]) is Monte Carlo tree search with UCB1
// over straight-line PTX basic blocks ([Program]) in a carry-free 32-bit
// alphabet: not, and, or, xor, shl, shr, shf.l.wrap, lop3 and prmt. Starting
// from a baseline program, it rewrites cones of instructions into single
// lop3 or prmt instructions, forwards operands, reorders independent
// instructions and tries random replacements, rewarding verified programs by
// their saving in modelled cycles ([CostModel]), peak live registers
// ([PeakRegs]) and instruction count. Candidates are checked against a
// [SpecFunc] on 64 boundary vectors and 10,000 pseudo-random vectors
// ([Oracle]); a new best program must also be proven equivalent to the
// baseline by [ProveEquivalent], which enumerates each result bit's input
// support. [Oracle.CheckMutants] verifies that single-point mutants of the
// result are rejected, and [Rule] promotes the result to an
// e-graph rewrite.
//
// Cycle counts come from a latency model, not from hardware. Discovered
// programs are emitted as PTX ([Program.PTX]) and as CUDA device functions
// with inline PTX ([Program.CUDA]); they run on the host through
// [Program.Eval]. The CUDA_EGRAPH_DISCOVER_NATIVE=1 test compiles and runs
// them on a GPU.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package discover
