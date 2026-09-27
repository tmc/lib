// Package cudasearch provides workload-bound search strategies (Beam and MCTS)
// over e-graph program rewrite transitions for CUDA execution plans.
//
// In GPU program synthesis and kernel optimization, equality saturation can produce
// combinatorial expansion. cudasearch explores candidate rewriting sequences using
// bounded beam search or seeded Monte Carlo Tree Search (MCTS).
//
// Candidates are evaluated against concrete workload bindings (tensor shapes, dtypes,
// input/weight identities, and numerical policies) using two-phase evaluation:
// Phase 1 screening via interleaved trials (cudatrial), and Phase 2 attribution
// with hardware stall profiling guidance.
//
// In addition to program rewrite search, cudasearch supports structured
// mapspace search over tensor loop nests ([LinearPrunedSearch]), pruning
// candidate factorizations that fail shared memory or register capacity
// constraints before permutation and bypass evaluation.
//
// Basic usage:
//
//	bindings := cudasearch.WorkloadBindings{
//		Shape:           cudashape.MustNew(1, 2048, 4096),
//		DType:           cudair.Float16,
//		NumericalPolicy: cudasearch.NumericalPolicyExact,
//	}
//
//	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
//		BeamWidth: 4,
//		MaxDepth:  8,
//	})
//	result, err := beam.Search(ctx, state, rules, evaluator)
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudasearch
