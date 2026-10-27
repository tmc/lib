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
package qualify
