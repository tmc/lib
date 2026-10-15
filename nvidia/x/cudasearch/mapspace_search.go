package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"maps"
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
	// UnsupportedCandidates is a subset of DownstreamSkipped: geometrically
	// valid, capacity-fitting candidates whose spatial traffic is
	// not modeled by the selected profile (including absent profile). These
	// never contribute to EvaluatedCandidates or Best.
	UnsupportedCandidates int
	// BestIntensity is the greatest logical DRAM FLOPs/byte. Equal intensities
	// retain the first candidate. This is not a latency or occupancy estimate.
	BestIntensity float64
	Best          *MapSpaceSearchResult
}

// MapSpaceSearchResult retains a detached mapping and its logical
// traffic. Mutating it does not change the search inputs.
type MapSpaceSearchResult struct {
	Mapping         cudalayout.MapSpace
	Movement        cudaroofline.HierarchyMovement
	SpatialProfile  *cudaroofline.GEMMSpatialProfile
	SpatialMovement *cudaroofline.SpatialHierarchyMovement
}

// LinearPrunedSearch performs ordered iteration over a factored mapspace,
// pruning downstream permutation, spatial and bypass combinations whenever an
// IndexFactorization fails capacity constraints.
type LinearPrunedSearch struct {
	Factorizations []cudalayout.IndexFactorization
	Permutations   []cudalayout.LoopPermutation
	// SpatialTilings supplies the spatial axis. An empty slice selects one
	// identity mesh. Explicit warp meshes require a positive thread limit.
	SpatialTilings []cudalayout.SpatialTiling
	// SpatialProfile opts into the declared logical owner-major model. Nil
	// preserves unsupported nonidentity accounting. LaneMesh is borrowed input.
	SpatialProfile *cudaroofline.GEMMSpatialProfile
	Bypasses       []cudalayout.DatatypeBypass
	Constraints    cudalayout.HardwareConstraints
	DTypes         map[cudalayout.TensorRole]int
	Problem        map[string]int
	defaultOrder   bool
}

// NewLinearPrunedSearch constructs a LinearPrunedSearch instance. With no
// permutations, it selects B/M/N/K (outermost first) at each listed level.
// Supplied permutations are used as written; absent non-unit orders are invalid.
func NewLinearPrunedSearch(
	facts []cudalayout.IndexFactorization,
	perms []cudalayout.LoopPermutation,
	bypasses []cudalayout.DatatypeBypass,
	constraints cudalayout.HardwareConstraints,
	dtypes map[cudalayout.TensorRole]int,
	problem map[string]int,
) *LinearPrunedSearch {
	defaultOrder := len(perms) == 0
	if defaultOrder {
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
		defaultOrder:   defaultOrder,
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
			SpatialTilings: append([]cudalayout.SpatialTiling(nil), s.SpatialTilings...),
			SpatialProfile: s.SpatialProfile,
			Bypasses:       append([]cudalayout.DatatypeBypass(nil), s.Bypasses...),
			Constraints:    s.Constraints,
			DTypes:         s.DTypes,
			Problem:        s.Problem,
			defaultOrder:   s.defaultOrder,
		})
		start += count
	}
	return result, nil
}

// Search runs pruned search over the mapspace. Orders must include each
// non-unit factor at the specified level; unit-factor loops may be omitted.
// Scoring uses AnalyticalGEMMHierarchy's full temporal retention contract and
// maximizes logical DRAM operational intensity, retaining the first tie.
// With SpatialProfile, supported owner-major spatial traffic is included.
// Without it, nonidentity spatial candidates remain unsupported skips.
// Spatial K reductions remain explicitly unmodeled. No physical cache behavior,
// device residency, synchronization or schedule latency is modeled.
// It rejects a candidate count that cannot be represented by its int statistics.
func (s *LinearPrunedSearch) Search(ctx context.Context) (MapSpaceSearchStats, error) {
	stats := MapSpaceSearchStats{}
	if len(s.Factorizations) == 0 {
		return stats, nil
	}
	spatials := s.SpatialTilings
	if len(spatials) == 0 {
		spatials = []cudalayout.SpatialTiling{{}}
	}
	orders, err := mapspaceDownstreamCount(1, len(s.Permutations), len(spatials))
	if err != nil {
		return stats, err
	}
	downstreamPerFact, err := mapspaceDownstreamCount(len(s.Factorizations), orders, len(s.Bypasses))
	if err != nil {
		return stats, err
	}

	if err := ctx.Err(); err != nil {
		return stats, err
	}
	if err := validateMapSpaceInputs(s.Problem, s.DTypes); err != nil {
		return stats, err
	}

	if s.SpatialProfile != nil {
		lanes := 1
		for dim, n := range s.SpatialProfile.LaneMesh {
			if s.Problem[dim] <= 0 || n <= 0 || n > 32/lanes {
				return stats, fmt.Errorf("invalid spatial lane mesh: %w", cudalayout.ErrUnsupported)
			}
			lanes *= n
		}
	}
	for _, fact := range s.Factorizations {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		stats.FactorizationsTested++
		if !validMapSpaceFactorization(fact, s.Problem) {
			stats.FactorizationsPruned++
			stats.DownstreamSkipped += downstreamPerFact
			continue
		}

		// A mapping can fit only with a supplied bypass. Prune it only when every
		// requested bypass configuration fails capacity validation.
		fits := false
		for _, spatial := range spatials {
			if err := ctx.Err(); err != nil {
				return stats, err
			}
			if !s.validSpatial(fact, spatial) {
				continue
			}
			for _, bypass := range s.Bypasses {
				if err := ctx.Err(); err != nil {
					return stats, err
				}
				if validMapSpaceBypass(fact, bypass) && cudalayout.ValidateCapacity(fact, bypass, s.DTypes, s.Constraints) == nil {
					fits = true
					break
				}
			}
		}
		if !fits {
			stats.FactorizationsPruned++
			stats.DownstreamSkipped += downstreamPerFact
			continue // Skip downstream permutation and bypass evaluations
		}

		for _, perm := range s.Permutations {
			if s.defaultOrder && len(perm.Orders) == 0 {
				orders := make(map[cudalayout.StorageLevel][]string, len(fact.Levels))
				for _, level := range fact.Levels {
					orders[level] = []string{"B", "M", "N", "K"}
				}
				perm = cudalayout.NewLoopPermutation(orders)
			}
			if err := ctx.Err(); err != nil {
				return stats, err
			}
			if !validMapSpacePermutation(fact, perm, s.Problem) {
				stats.DownstreamSkipped += len(spatials) * len(s.Bypasses)
				continue
			}
			for _, spatial := range spatials {
				if err := ctx.Err(); err != nil {
					return stats, err
				}
				if !s.validSpatial(fact, spatial) {
					stats.DownstreamSkipped += len(s.Bypasses)
					continue
				}
				for _, bypass := range s.Bypasses {
					if err := ctx.Err(); err != nil {
						return stats, err
					}

					// Check bypass-specific capacity
					if !validMapSpaceBypass(fact, bypass) || cudalayout.ValidateCapacity(fact, bypass, s.DTypes, s.Constraints) != nil {
						stats.DownstreamSkipped++
						continue
					}

					if s.SpatialProfile == nil && !identityMapSpaceSpatial(spatial) {
						stats.UnsupportedCandidates++
						stats.DownstreamSkipped++
						continue
					}
					mapping := cudalayout.MapSpace{Problem: s.Problem, Factorization: fact, Permutation: perm, Spatial: spatial, Bypass: bypass}
					var movement cudaroofline.HierarchyMovement
					var spatialMovement *cudaroofline.SpatialHierarchyMovement
					if s.SpatialProfile != nil {
						report, err := cudaroofline.AnalyticalGEMMSpatial(mapping, s.DTypes, *s.SpatialProfile)
						if errors.Is(err, cudalayout.ErrUnmodeled) {
							stats.UnsupportedCandidates++
							stats.DownstreamSkipped++
							continue
						}
						if errors.Is(err, cudalayout.ErrUnsupported) {
							stats.DownstreamSkipped++
							continue
						}
						if err != nil {
							return stats, fmt.Errorf("mapspace spatial data movement: %w", err)
						}
						spatialMovement = &report
						movement = report.Movement
					} else {
						var err error
						movement, err = cudaroofline.AnalyticalGEMMHierarchy(mapping, s.DTypes)
						if err != nil {
							return stats, fmt.Errorf("mapspace data movement: %w", err)
						}
					}
					if stats.Best == nil || movement.DRAM.OperationalIntensity > stats.BestIntensity {
						stats.BestIntensity = movement.DRAM.OperationalIntensity
						stats.Best = &MapSpaceSearchResult{Mapping: cloneMapSpace(mapping), Movement: movement, SpatialMovement: spatialMovement}
						if s.SpatialProfile != nil {
							stats.Best.SpatialProfile = &cudaroofline.GEMMSpatialProfile{LaneMesh: maps.Clone(s.SpatialProfile.LaneMesh)}
						}
					}
					stats.EvaluatedCandidates++
				}
			}
		}
	}

	return stats, nil
}

func (s *LinearPrunedSearch) validSpatial(fact cudalayout.IndexFactorization, spatial cudalayout.SpatialTiling) bool {
	if err := spatial.Validate(fact, s.Problem); err != nil {
		return false
	}
	if len(spatial.WarpMesh) != 0 {
		// CUDA warp meshes require 32 threads per warp. The caller's
		// configured block size is a conservative capacity bound, not a
		// measured residency or launch fact.
		warps := spatial.TotalWarps()
		if warps <= 0 || s.Constraints.MaxThreadsPerBlock <= 0 || warps > s.Constraints.MaxThreadsPerBlock/32 {
			return false
		}
	}
	if s.SpatialProfile != nil {
		lanes := 1
		for _, n := range s.SpatialProfile.LaneMesh {
			lanes *= n
		} // validated before iteration
		if lanes > 1 && (s.Constraints.MaxThreadsPerBlock <= 0 || spatial.TotalWarps() > s.Constraints.MaxThreadsPerBlock/lanes) {
			return false
		}
	}
	return true
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
	for _, level := range fact.Levels {
		for dim, factor := range fact.Factors[level] {
			if factor > 1 && !slices.Contains(perm.Orders[level], dim) {
				return false
			}
		}
	}
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

func validateMapSpaceInputs(problem map[string]int, dtypes map[cudalayout.TensorRole]int) error {
	if len(problem) != 4 {
		return fmt.Errorf("mapspace requires exactly B/M/N/K: %w", cudalayout.ErrUnsupported)
	}
	for _, dim := range []string{"B", "M", "N", "K"} {
		if problem[dim] <= 0 {
			return fmt.Errorf("mapspace requires positive %s: %w", dim, cudalayout.ErrUnsupported)
		}
	}
	if len(dtypes) != 3 {
		return fmt.Errorf("mapspace requires exactly A/B/C byte sizes: %w", cudalayout.ErrUnsupported)
	}
	for _, role := range []cudalayout.TensorRole{cudalayout.TensorA, cudalayout.TensorB, cudalayout.TensorC} {
		if dtypes[role] <= 0 {
			return fmt.Errorf("mapspace requires positive %s byte size: %w", role, cudalayout.ErrUnsupported)
		}
	}
	return nil
}

func validMapSpaceFactorization(fact cudalayout.IndexFactorization, problem map[string]int) bool {
	if len(fact.Levels) == 0 || fact.Levels[len(fact.Levels)-1] != cudalayout.LevelDRAM {
		return false
	}
	for i, level := range fact.Levels {
		if level < cudalayout.LevelRegister || level > cudalayout.LevelDRAM || (i > 0 && level <= fact.Levels[i-1]) {
			return false
		}
	}
	for level := range fact.Factors {
		if !slices.Contains(fact.Levels, level) {
			return false
		}
	}
	return fact.Validate(problem) == nil
}

func validMapSpaceBypass(fact cudalayout.IndexFactorization, bypass cudalayout.DatatypeBypass) bool {
	for level, roles := range bypass.Bypassed {
		if !slices.Contains(fact.Levels, level) {
			return false
		}
		for role, bypassed := range roles {
			if role < cudalayout.TensorA || role > cudalayout.TensorC || (level == cudalayout.LevelDRAM && bypassed) {
				return false
			}
		}
	}
	return true
}

func identityMapSpaceSpatial(spatial cudalayout.SpatialTiling) bool {
	for _, mesh := range []map[string]int{spatial.CTAMesh, spatial.WarpMesh} {
		for _, count := range mesh {
			if count != 1 {
				return false
			}
		}
	}
	return true
}

func cloneMapSpace(mapping cudalayout.MapSpace) cudalayout.MapSpace {
	mapping.Problem = maps.Clone(mapping.Problem)
	mapping.Factorization = cudalayout.NewIndexFactorization(mapping.Factorization.Levels, mapping.Factorization.Factors)
	mapping.Permutation = cudalayout.NewLoopPermutation(mapping.Permutation.Orders)
	mapping.Spatial = cudalayout.NewSpatialTiling(mapping.Spatial.CTAMesh, mapping.Spatial.WarpMesh)
	tensors := make(map[cudalayout.StorageLevel]map[cudalayout.TensorRole]bool, len(mapping.Bypass.Bypassed))
	for level, roles := range mapping.Bypass.Bypassed {
		tensors[level] = maps.Clone(roles)
	}
	mapping.Bypass = cudalayout.DatatypeBypass{Bypassed: tensors}
	return mapping
}
