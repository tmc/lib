// Package qualify provides host-side formal and adversarial qualification gates,
// PTXAS and ABBA paired-timing autotuning, and profile-guided bottleneck
// classification and schedule refinement for GPU kernels and execution plans.
//
// Qualification combines static CUDA C++ and launcher validation, host-emulated
// concurrency and warpgroup hazard checking via [cudahostemu.CheckHazards],
// symbolic SASS equivalence verification for NVFP4 bit-routing and E4M3 scale
// conversion, dual-target numerical agreement against Tile IR bytecode
// modules, and Blackwell sm_100a Tensor Memory (TMEM) trace verification via
// [cudatmem.VerifyTrace].
//
// Deterministic mode: [AuditPTXDeterminism] flags floating-point add
// reductions through atomics and timer-derived stochastic-rounding bits in
// PTX; [AuditSchedule] and [BatchInvarianceCheck] check that a reduction
// schedule gives a row the same bits at every batch size and split count;
// [EmitDeterministicStochasticRoundPTX] emits cvt.rs rounding keyed by a
// [Philox4x32] counter, and [DeterministicCUDA] is its CUDA C++ twin.
package qualify
