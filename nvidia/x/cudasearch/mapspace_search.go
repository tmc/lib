package cudasearch

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// MapSpaceSearchStats tracks mapspace search progression and pruning efficacy.
type MapSpaceSearchStats struct {
	FactorizationsTested int
	FactorizationsPruned int
	DownstreamSkipped    int
	EvaluatedCandidates  int
	BestIntensity        float64
}

// LinearPrunedSearch performs ordered iteration over a factored mapspace,
// pruning downstream permutation and bypass combinations whenever an
// IndexFactorization fails capacity constraints.
type LinearPrunedSearch struct {
	Factorizations []cudalayout.IndexFactorization
	Permutations   []cudalayout.LoopPermutation
	Bypasses       []cudalayout.DatatypeBypass
	Constraints    cudalayout.HardwareConstraints
	DTypes         map[cudalayout.TensorRole]int
	Problem        map[string]int
}

// NewLinearPrunedSearch constructs a LinearPrunedSearch instance.
func NewLinearPrunedSearch(
	facts []cudalayout.IndexFactorization,
	perms []cudalayout.LoopPermutation,
	bypasses []cudalayout.DatatypeBypass,
	constraints cudalayout.HardwareConstraints,
	dtypes map[cudalayout.TensorRole]int,
	problem map[string]int,
) *LinearPrunedSearch {
	if len(perms) == 0 {
		perms = []cudalayout.LoopPermutation{cudalayout.NewLoopPermutation(nil)}
	}
	if len(bypasses) == 0 {
		bypasses = []cudalayout.DatatypeBypass{cudalayout.NewDatatypeBypass()}
	}
	return &LinearPrunedSearch{
		Factorizations: append([]cudalayout.IndexFactorization(nil), facts...),
		Permutations:   append([]cudalayout.LoopPermutation(nil), perms...),
		Bypasses:       append([]cudalayout.DatatypeBypass(nil), bypasses...),
		Constraints:    constraints,
		DTypes:         dtypes,
		Problem:        problem,
	}
}

// Split divides the factorization space across n partitions.
func (s *LinearPrunedSearch) Split(n int) ([]*LinearPrunedSearch, error) {
	if n <= 0 {
		return nil, fmt.Errorf("split count %d must be positive", n)
	}
	total := len(s.Factorizations)
	if total == 0 {
		return nil, nil
	}
	if n > total {
		n = total
	}
	result := make([]*LinearPrunedSearch, 0, n)
	base, extra := total/n, total%n
	start := 0
	for i := 0; i < n; i++ {
		count := base
		if i < extra {
			count++
		}
		partFacts := s.Factorizations[start : start+count]
		result = append(result, &LinearPrunedSearch{
			Factorizations: append([]cudalayout.IndexFactorization(nil), partFacts...),
			Permutations:   append([]cudalayout.LoopPermutation(nil), s.Permutations...),
			Bypasses:       append([]cudalayout.DatatypeBypass(nil), s.Bypasses...),
			Constraints:    s.Constraints,
			DTypes:         s.DTypes,
			Problem:        s.Problem,
		})
		start += count
	}
	return result, nil
}

// Search runs pruned search over the mapspace.
func (s *LinearPrunedSearch) Search(ctx context.Context) (MapSpaceSearchStats, error) {
	stats := MapSpaceSearchStats{}
	downstreamPerFact := len(s.Permutations) * len(s.Bypasses)

	// Convert dtypes to string map for cudaroofline
	dtypesStr := map[string]int{
		"A": s.DTypes[cudalayout.TensorA],
		"B": s.DTypes[cudalayout.TensorB],
		"C": s.DTypes[cudalayout.TensorC],
	}

	for _, fact := range s.Factorizations {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		stats.FactorizationsTested++
		if err := fact.Validate(s.Problem); err != nil {
			stats.FactorizationsPruned++
			stats.DownstreamSkipped += downstreamPerFact
			continue
		}

		// A mapping can fit only with a supplied bypass. Prune it only when every
		// requested bypass configuration fails capacity validation.
		fits := false
		for _, bypass := range s.Bypasses {
			if err := cudalayout.ValidateCapacity(fact, bypass, s.DTypes, s.Constraints); err == nil {
				fits = true
				break
			}
		}
		if !fits {
			stats.FactorizationsPruned++
			stats.DownstreamSkipped += downstreamPerFact
			continue // Skip downstream permutation and bypass evaluations
		}

		sharedTile := fact.TileShape(cudalayout.LevelShared)

		for _, perm := range s.Permutations {
			_ = perm // Permutation determines nest ordering for schedule generation
			for _, bypass := range s.Bypasses {
				if err := ctx.Err(); err != nil {
					return stats, err
				}

				// Check bypass-specific capacity
				if err := cudalayout.ValidateCapacity(fact, bypass, s.DTypes, s.Constraints); err != nil {
					stats.DownstreamSkipped++
					continue
				}

				bypassedMap := map[string]bool{
					"A": bypass.IsBypassed(cudalayout.LevelShared, cudalayout.TensorA) || bypass.IsBypassed(cudalayout.LevelL2, cudalayout.TensorA),
					"B": bypass.IsBypassed(cudalayout.LevelShared, cudalayout.TensorB) || bypass.IsBypassed(cudalayout.LevelL2, cudalayout.TensorB),
					"C": bypass.IsBypassed(cudalayout.LevelShared, cudalayout.TensorC) || bypass.IsBypassed(cudalayout.LevelL2, cudalayout.TensorC),
				}

				dm, err := cudaroofline.AnalyticalDataMovement(s.Problem, sharedTile, dtypesStr, bypassedMap)
				if err != nil {
					return stats, fmt.Errorf("mapspace data movement: %w", err)
				}
				if dm.OperationalIntensity > stats.BestIntensity {
					stats.BestIntensity = dm.OperationalIntensity
				}
				stats.EvaluatedCandidates++
			}
		}
	}

	return stats, nil
}
