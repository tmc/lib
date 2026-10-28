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
//
// Silent data corruption: [VerifyTileABFT] checks a GEMM tile with row and
// column checksums against a tolerance that bounds every FP32 accumulation
// order, and locates a corrupted element; [EmitABFTAndWatchdogPTX] emits the
// Tensor Memory epilogue check, a %globaltimer-bounded mbarrier wait and a
// [FaultRecord] ring writer.
//
// Power pacing: [AnalyzePowerTransient] models the dI/dt step of a
// megakernel's clusters entering their Tensor Core phase together and the
// inductive droop it causes; [SolveOptimalPacing] picks the cheapest
// nanosleep stagger that keeps the modeled droop under a limit, and
// [EmitPowerPacedProloguePTX] emits it.
//
// Cross-target equivalence: [ProveCrossISAEquivalence] enumerates every
// multiply-accumulate of an NVFP4 GEMM as lowered for sm_90a (wgmma, with
// the [bitopt] FP4 decode), sm_100a, sm_103a and sm_107a (tcgen05 with
// Tensor Memory) and sm_120a and sm_121a (mma.sync), using the instruction
// layouts in [cudahostemu] and [tcgen05], and checks that each computes
// every scaled term exactly once; [VerifyAllTargetISAs] runs it against
// sm_90a for all six targets.
package qualify
