// Package cudaref provides an independent CPU reference implementation and numerical oracle
// for validating GPU kernel operations and semantic IR execution graphs.
//
// # Graph Evaluation
//
// Computation graphs ([cudair.Graph] or [cudair.Snapshot]) can be evaluated directly on the CPU
// using [Evaluate] or [EvaluateSnapshot]. The evaluator executes the full authoritative operation
// registry (elementwise arithmetic, transcendental activations, matrix multiplication, reductions,
// and shape transformations) with strict floating-point tracking.
//
// # IEEE-754 Numerical Policies
//
// The reference implementation defines explicit policies for edge cases:
//   - NaN propagation: arithmetic on NaN yields NaN; activations preserve NaN.
//   - Infinities: division by signed zero yields positive/negative infinity; exp(-Inf) is +0.0.
//   - Signed zero: preserve distinction between +0.0 and -0.0 in negation and reciprocal operations.
//   - Empty reduction sets: sum reduction on an empty set yields 0.0; max reduction yields -Inf.
//   - Completely masked attention: rows with all elements masked return an exact zero vector without NaNs.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudaref
