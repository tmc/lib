// Package warpsynth chooses warp-specialization parameters for a persistent
// CTA, checks the resulting mbarrier ring protocol for deadlock and races,
// and emits the kernel skeleton.
//
// [Synthesize] takes per-role demands ([RoleDemand]: TMA producer, MMA
// issuer, epilogue math, fabric storer), the tile's shared-memory footprint
// and the TMEM buffers the pipeline needs, and returns a [Schedule]: warps
// and setmaxnreg targets per role, the shared-memory ring depth D, TMEM
// column offsets and whether the MMA issuer releases a shared-memory stage
// on its commit rather than after the epilogue.
//
// # Search
//
// The search is exhaustive enumeration in pure Go, not an SMT solver; no
// solver is linked. It is complete for the space it enumerates: every warp
// split with each role a whole number of warpgroups and a total in
// [Problem.Totals], every ring depth D in 2..8, and both release modes. Two
// dimensions are decided without enumeration, and each is exact. Register
// targets do not enter the cost model, and the register constraint is
// monotone in each target, so a split is feasible exactly when the minimum
// targets fit; leftover registers are then granted greedily. TMEM buffers
// have power-of-two widths, so placing them largest first at offsets aligned
// to their width packs them without gaps, and the packing fits exactly when
// the rounded total does.
//
// The invariants every returned schedule satisfies, which [Schedule.Check]
// re-verifies, are:
//
//   - 32·Σ W_r·R_r is at most the CTA register pool, 32·T·L, where T is the
//     total warp count and L the launch register count (at most 65536);
//   - every R_r is a multiple of 8 in 24..256, the setmaxnreg range ptxas
//     13.0 accepts at sm_90a and sm_121a (see package cudareg);
//   - every W_r is a multiple of 4, so setmaxnreg runs per warpgroup;
//   - TMEM intervals are disjoint and end within the target's column limit;
//   - D·tile + epilogue + 16·(mbarriers) bytes fit the shared-memory cap.
//
// The cost model is a model. A role's per-tile cycles are its demand's
// WorkCyclesTile, divided by the warp ratio when the role scales with warps;
// the steady-state period is the larger of the slowest role and the time a
// ring stage stays occupied divided by D. None of it was measured.
//
// # Protocol check
//
// [VerifyDeadlockFreedom] builds the ring protocol the schedule implies
// (full and empty mbarriers per stage, phase parity (k/D)&1, one elected
// arrival per role) and explores it as an explicit-state system: every
// reachable interleaving of role steps, breadth first, for the requested
// tile count, then the same run under fixed producer-first, consumer-first
// and round-robin schedulers. It reports a deadlock (no role can step before
// all tiles finish), a race (a stage overwritten before its release), a
// stale read, or a parity alias: a wait that passes because the barrier's
// phase parity matches while its phase count does not. Each role's tile step
// is modelled as one atomic action (its waits, its stage accesses and its
// arrivals); interleavings inside a step are not explored, and the warps of
// a role act as one agent, as the emitted skeleton makes them do with a
// role-local named barrier before the elected arrival.
//
// # Emission
//
// [Schedule.EmitWarpSpecializedSkeletonPTX] emits a PTX entry with the role
// dispatch on %tid.x/32 (not %warpid, which PTX ISA documents as volatile),
// setmaxnreg.dec/inc per role, mbarrier initialization and the
// parity-tracking wait loops, with comments where TMA, MMA and epilogue work
// go. Nothing in this package assembles it; the native test does that, with
// ptxas, when CUDA_WARPSYNTH_NATIVE=1. [Schedule.EmitHostEmuCUDA] emits the
// same protocol as CUDA C++ for package cudahostemu, whose adversarial
// mbarrier model runs it on the host CPU; that executes the C++ skeleton,
// not the PTX.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package warpsynth
