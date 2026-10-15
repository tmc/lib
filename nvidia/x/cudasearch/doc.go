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
// constraints before permutation, spatial and bypass evaluation. Supplied
// spatial meshes must divide the factored tile counts. Whole-warp meshes must
// also fit the caller's thread bound; those checks are not device launch or
// residency qualification. The selected logical retention recipe accounts for
// permutation, spatial ownership and bypass. With SpatialProfile and SplitK,
// search includes tagged partial-C scratch and the declared direct-DRAM binary
// reduction tree. MapSpaceSearchResult.TotalDRAM includes both phases; Movement
// is the GEMM phase. These are exact selected-recipe traffic/capacity quantities,
// not universal minimum traffic, native or theoretical hardware evidence.
//
// [SearchMemory] freezes a canonical bottleneck category, target-specific
// strategy catalog and compatible trajectory rewards once per Search. Priors
// order legal transitions; they do not change objective costs or validate a
// rewrite. [ClassifyBottleneck] is a deterministic producer-declared heuristic
// and reports PendingHardware. Its percentages cannot establish native evidence.
// [CollectBottleneckProfile] separately collects bounded whole-operation CUPTI
// counters under a live GPU lock. [ObserveCandidateTrajectory] admits completed
// executable ABBA trials for cross-task replay. Both retain exact execution
// identities; a hardware-observed category remains a heuristic.
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
