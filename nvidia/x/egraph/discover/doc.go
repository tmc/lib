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
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package discover
