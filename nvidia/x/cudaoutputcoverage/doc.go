// Package cudaoutputcoverage checks which logical CUDA output elements were
// written, independently of numerical output agreement.
//
// Contracts name a buffer and a declared logical domain. Domain ranges exclude
// padding explicitly. ExactlyOnce requires one store per element; AtLeastOnce
// requires a store; WriterCount declares a positive expected store count;
// Accumulated declares an atomic or reduction accumulator. Applying ExactlyOnce
// to an atomic or reduction output is a contract error.
//
// The two-fill arm supports only deterministic, ordinary AtLeastOnce FP32
// outputs. Fill patterns have distinct finite values and differ numerically
// from the reference on the entire domain. Readback comparisons against fills
// are bitwise. An element retaining its own fill in both runs is unwritten.
// Other differences between runs are reported separately as nondeterminism.
// A kernel deliberately producing the selected fill patterns can fool this
// arm; write counts and accumulated outputs require instrumented store evidence.
// No store-address hook is supplied by this package.
//
// Reports describe output elements written for the admitted launch and domain,
// not instruction coverage, code coverage, or a proof of numerical correctness.
package cudaoutputcoverage
