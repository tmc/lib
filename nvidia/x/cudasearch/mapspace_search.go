package cudasearch

import (
	"context"
	"fmt"
	"slices"

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

// Search runs pruned search over the mapspace. Orders must include each
// non-unit factor at the specified level; unit-factor loops may be omitted.
// Scoring uses order-independent data movement floors, not schedule latency.
// It rejects a candidate count that cannot be represented by its int statistics.
func (s *LinearPrunedSearch) Search(ctx context.Context) (MapSpaceSearchStats, error) {
	stats := MapSpaceSearchStats{}
	downstreamPerFact, err := mapspaceDownstreamCount(len(s.Factorizations), len(s.Permutations), len(s.Bypasses))
	if err != nil {
		return stats, err
	}

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
			if !validMapSpacePermutation(fact, perm, s.Problem) {
				stats.DownstreamSkipped += len(s.Bypasses)
				continue
			}
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

// mapspaceDownstreamCount checks both the per-factorization product and the
// total count. Every skipped or evaluated candidate belongs to that total, so
// subsequent statistics additions cannot overflow.
func mapspaceDownstreamCount(factorizations, permutations, bypasses int) (int, error) {
	if factorizations < 0 || permutations < 0 || bypasses < 0 {
		return 0, fmt.Errorf("mapspace candidate count is negative")
	}
	if factorizations == 0 {
		return 0, nil
	}
	maxInt := int(^uint(0) >> 1)
	if bypasses != 0 && permutations > maxInt/bypasses {
		return 0, fmt.Errorf("mapspace downstream count overflows int")
	}
	downstream := permutations * bypasses
	if downstream != 0 && factorizations > maxInt/downstream {
		return 0, fmt.Errorf("mapspace candidate count overflows int")
	}
	return downstream, nil
}

func validMapSpacePermutation(fact cudalayout.IndexFactorization, perm cudalayout.LoopPermutation, problem map[string]int) bool {
	for level, order := range perm.Orders {
		if level < cudalayout.LevelRegister || level > cudalayout.LevelDRAM || !slices.Contains(fact.Levels, level) {
			return false
		}
		var dimensions []string
		for dimension, factor := range fact.Factors[level] {
			if factor > 1 {
				dimensions = append(dimensions, dimension)
			}
		}
		for _, dimension := range order {
			if _, ok := problem[dimension]; ok && !slices.Contains(dimensions, dimension) {
				dimensions = append(dimensions, dimension)
			}
		}
		one := cudalayout.LoopPermutation{Orders: map[cudalayout.StorageLevel][]string{level: order}}
		if err := one.Validate(dimensions); err != nil {
			return false
		}
	}
	return true
}
