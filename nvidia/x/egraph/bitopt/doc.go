// Package bitopt finds short instruction sequences that move the bits of
// packed words into fixed register layouts, choosing the packing as well.
//
// A low-bit weight format is decoded by placing each code's bits where a
// wider type wants them: E2M1 magnitudes at bf16 bits 6-8 and the sign at
// bit 15 give e2m1·2^-126, for example. The packing of codes into words is
// free when weights are repacked offline, so the cost of the decode depends
// on the packing. [Solve] searches packings and programs together and
// returns the cheapest program in its model, or proves that none is cheaper
// than a bound.
//
// The model: a program computes moves of its input words (shifts,
// rotations, funnel shifts and byte permutes, one instruction each; the
// words themselves are free), and builds each output register as the OR of
// masked moves (one LOP3 per part). A program costs one per move plus one
// per part. Every operation routes bits without combining them, so the
// value of each output bit is one input bit or zero, and [Solution.Check]
// proves a program correct for all inputs by evaluating it on the unit
// vectors.
//
// [Solution.CUDA] emits the program as a device function, and
// [Solution.Source] gives the packing: which input bit each output bit
// reads.
//
// # Epilogue DAGs
//
// [SynthesizeEpilogueDAG] shortens a straight-line, multi-output epilogue
// ([EpilogueDAG]) of integer, float32 and E2M1-conversion operations: it
// merges common subexpressions, rebalances associative chains, replaces
// small windows with smaller programs found by bounded enumeration, and
// list-schedules the result against its tcgen05.ld load groups to lower
// the live register count. Unlike [Solve], it is not exact: the search is
// bounded and equivalence is tested on boundary and random vectors
// ([TestVectors], [CheckEquivalent]), not proved. The E2M1 rounding and
// float32 NaN results are this package's model of the PTX instructions,
// not observed on hardware.
//
// [EpilogueDAG.PTX] emits the result for sm_100a. [VerifyWaitDataflow]
// checks that no register a tcgen05.ld loads is touched before a
// tcgen05.wait::ld, and [CoalesceWaits] sinks and merges those waits.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package bitopt
