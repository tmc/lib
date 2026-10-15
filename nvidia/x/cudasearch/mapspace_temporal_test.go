package cudasearch

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

var searchTemporalRoles = []cudalayout.TensorRole{cudalayout.TensorA, cudalayout.TensorB, cudalayout.TensorC}
var searchTemporalDims = []string{"B", "M", "N", "K"}

func searchTemporalMapping() cudalayout.MapSpace {
	levels := []cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelL2, cudalayout.LevelDRAM}
	factors := map[cudalayout.StorageLevel]map[string]int{
		cudalayout.LevelRegister: {"B": 1, "M": 2, "N": 1, "K": 1},
		cudalayout.LevelShared:   {"B": 1, "M": 1, "N": 2, "K": 2},
		cudalayout.LevelL2:       {"B": 1, "M": 2, "N": 1, "K": 1},
		cudalayout.LevelDRAM:     {"B": 2, "M": 1, "N": 1, "K": 2},
	}
	orders := map[cudalayout.StorageLevel][]string{}
	for _, level := range levels {
		orders[level] = []string{"B", "M", "N", "K"}
	}
	return cudalayout.MapSpace{
		Problem:       map[string]int{"B": 2, "M": 4, "N": 2, "K": 4},
		Factorization: cudalayout.NewIndexFactorization(levels, factors),
		Permutation:   cudalayout.NewLoopPermutation(orders),
		Bypass:        cudalayout.NewDatatypeBypass(),
	}
}

// searchTemporalTrace executes scalar iterations, tracks the actual projected tile
// keys and first visits, and charges the nearest retained source. It does not
// use the estimator's algebraic run counts or any production footprint helper.
func searchTemporalTrace(mapping cudalayout.MapSpace, dtypes map[cudalayout.TensorRole]int) []cudaroofline.HierarchyLevel {
	levels := mapping.Factorization.Levels
	var result []cudaroofline.HierarchyLevel
	tiles := make([][4]int, len(levels))
	projected := [3][]int{{0, 1, 3}, {0, 2, 3}, {0, 1, 2}}
	cumulative := [4]int{1, 1, 1, 1}
	for i, level := range levels {
		for j, dim := range searchTemporalDims {
			if factor := mapping.Factorization.Factors[level][dim]; factor != 0 {
				cumulative[j] *= factor
			}
		}
		tiles[i] = cumulative
		entry := cudaroofline.HierarchyLevel{Level: level, Tensors: map[cudalayout.TensorRole]cudaroofline.TensorMovement{}}
		for roleIndex, role := range searchTemporalRoles {
			bytes := uint64(0)
			if !mapping.Bypass.IsBypassed(level, role) {
				bytes = uint64(dtypes[role])
				for _, dim := range projected[roleIndex] {
					bytes *= uint64(cumulative[dim])
				}
			}
			entry.Tensors[role] = cudaroofline.TensorMovement{ResidentBytes: bytes}
			entry.ResidentBytes += bytes
		}
		result = append(result, entry)
	}
	charge := func(child int, role cudalayout.TensorRole, read, write uint64) {
		for i := child + 1; i < len(levels); i++ {
			if mapping.Bypass.IsBypassed(levels[i], role) {
				continue
			}
			entry := result[i].Tensors[role]
			entry.ReadBytes += read
			entry.WriteBytes += write
			result[i].Tensors[role] = entry
			return
		}
		panic("invalid test fixture: no source")
	}
	type cache struct {
		last  [4]int
		valid bool
		seen  map[[4]int]bool
	}
	caches := make([][3]cache, len(levels))
	for i := range caches {
		for j := range searchTemporalRoles {
			caches[i][j].seen = map[[4]int]bool{}
		}
	}
	type scalarLoop struct{ dim, count, stride int }
	var loops []scalarLoop
	for i := len(levels) - 1; i >= 0; i-- {
		for _, name := range mapping.Permutation.Orders[levels[i]] {
			for dim, expected := range searchTemporalDims {
				if name != expected {
					continue
				}
				stride := 1
				if i > 0 {
					stride = tiles[i-1][dim]
				}
				count := mapping.Factorization.Factors[levels[i]][name]
				if count == 0 {
					count = 1
				}
				loops = append(loops, scalarLoop{dim, count, stride})
			}
		}
	}
	var coords [4]int
	outputs := map[[3]int]bool{}
	var walk func(int)
	walk = func(at int) {
		if at < len(loops) {
			loop := loops[at]
			old := coords[loop.dim]
			for n := 0; n < loop.count; n++ {
				coords[loop.dim] = old + n*loop.stride
				walk(at + 1)
			}
			coords[loop.dim] = old
			return
		}
		for i, level := range levels[:len(levels)-1] {
			for roleIndex, role := range searchTemporalRoles {
				if mapping.Bypass.IsBypassed(level, role) {
					continue
				}
				var key [4]int
				for _, dim := range projected[roleIndex] {
					key[dim] = coords[dim] / tiles[i][dim]
				}
				state := &caches[i][roleIndex]
				if state.valid && state.last == key {
					continue
				}
				bytes := result[i].Tensors[role].ResidentBytes
				if role == cudalayout.TensorC {
					read := uint64(0)
					if state.seen[key] {
						read = bytes
					}
					charge(i, role, read, bytes)
				} else {
					charge(i, role, bytes, 0)
				}
				state.last, state.valid, state.seen[key] = key, true, true
			}
		}
		charge(-1, cudalayout.TensorA, uint64(dtypes[cudalayout.TensorA]), 0)
		charge(-1, cudalayout.TensorB, uint64(dtypes[cudalayout.TensorB]), 0)
		key := [3]int{coords[0], coords[1], coords[2]}
		read := uint64(0)
		if outputs[key] {
			read = uint64(dtypes[cudalayout.TensorC])
		}
		charge(-1, cudalayout.TensorC, read, uint64(dtypes[cudalayout.TensorC]))
		outputs[key] = true
	}
	walk(0)
	return result
}

// The scalar oracle above is retained from the independently traced hierarchy
// fixtures. Search is checked against accesses, not another estimator call.
func TestMapspaceTemporalTrace(t *testing.T) {
	dtypes := map[cudalayout.TensorRole]int{cudalayout.TensorA: 2, cudalayout.TensorB: 3, cudalayout.TensorC: 4}
	for direction, order := range [][]string{{"B", "M", "N", "K"}, {"B", "K", "N", "M"}} {
		for mask := 0; mask < 512; mask++ {
			mapping := searchTemporalMapping()
			for _, level := range mapping.Factorization.Levels {
				mapping.Permutation.Orders[level] = order
			}
			for level := 0; level < 3; level++ {
				for role := 0; role < 3; role++ {
					mapping.Bypass.SetBypass(cudalayout.StorageLevel(level), searchTemporalRoles[role], mask&(1<<(level*3+role)) != 0)
				}
			}
			search := NewLinearPrunedSearch([]cudalayout.IndexFactorization{mapping.Factorization}, []cudalayout.LoopPermutation{mapping.Permutation}, []cudalayout.DatatypeBypass{mapping.Bypass}, cudalayout.HardwareConstraints{}, dtypes, mapping.Problem)
			stats, err := search.Search(context.Background())
			if err != nil || stats.EvaluatedCandidates != 1 || stats.Best == nil {
				t.Fatalf("direction %d mask %d: %+v, %v", direction, mask, stats, err)
			}
			want := searchTemporalTrace(mapping, dtypes)
			if !reflect.DeepEqual(stats.Best.Movement.Levels, want) {
				t.Fatalf("direction %d mask %d: got %+v want %+v", direction, mask, stats.Best.Movement.Levels, want)
			}
			dram := want[len(want)-1]
			var read, write uint64
			for _, role := range searchTemporalRoles {
				read += dram.Tensors[role].ReadBytes
				write += dram.Tensors[role].WriteBytes
			}
			got := stats.Best.Movement.DRAM
			if got.ReadBytes != read || got.WriteBytes != write || got.TotalBytes != read+write || stats.BestIntensity != 128/float64(read+write) {
				t.Fatalf("direction %d mask %d: DRAM %+v, intensity %g, want read %d write %d", direction, mask, got, stats.BestIntensity, read, write)
			}
			if write < 2*4*2*4 || math.IsNaN(stats.BestIntensity) || math.IsInf(stats.BestIntensity, 0) {
				t.Fatalf("invalid C writes/intensity: %+v", stats)
			}
		}
	}
}

func temporalOrderSearch() *LinearPrunedSearch {
	levels := []cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelL2, cudalayout.LevelDRAM}
	factors := map[cudalayout.StorageLevel]map[string]int{}
	for _, level := range levels {
		factors[level] = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
	}
	factors[cudalayout.LevelDRAM] = map[string]int{"B": 1, "M": 2, "N": 2, "K": 2}
	fact := cudalayout.NewIndexFactorization(levels, factors)
	permutations := []cudalayout.LoopPermutation{
		cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{cudalayout.LevelDRAM: {"K", "M", "N"}}),
		cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{cudalayout.LevelDRAM: {"M", "N", "K"}}),
	}
	return NewLinearPrunedSearch([]cudalayout.IndexFactorization{fact}, permutations, nil, cudalayout.HardwareConstraints{}, map[cudalayout.TensorRole]int{cudalayout.TensorA: 1, cudalayout.TensorB: 1, cudalayout.TensorC: 1}, map[string]int{"B": 1, "M": 2, "N": 2, "K": 2})
}

func TestMapspaceTemporalWinner(t *testing.T) {
	search := temporalOrderSearch()
	stats, err := search.Search(context.Background())
	if err != nil || stats.EvaluatedCandidates != 2 || stats.Best == nil {
		t.Fatalf("stats=%+v error=%v", stats, err)
	}
	if stats.BestIntensity != 16.0/20 || stats.Best.Movement.DRAM.TotalBytes != 20 || !reflect.DeepEqual(stats.Best.Mapping.Permutation.Orders[cudalayout.LevelDRAM], []string{"M", "N", "K"}) {
		t.Fatalf("wrong temporal winner: %+v", stats.Best)
	}
	search.Permutations = search.Permutations[:1]
	other, err := search.Search(context.Background())
	if err != nil || other.Best == nil || other.Best.Movement.DRAM.TotalBytes != 24 || other.Best.Movement.DRAM.TensorTransfers["C"] != 12 {
		t.Fatalf("C eviction/reload missing: %+v %v", other.Best, err)
	}
}

func TestMapspaceTemporalFirstTieAndDetached(t *testing.T) {
	search := temporalOrderSearch()
	search.Permutations = search.Permutations[1:]
	second := cudalayout.NewLoopPermutation(search.Permutations[0].Orders)
	second.Orders[cudalayout.LevelRegister] = []string{"B", "M", "N", "K"} // All unit; same cost.
	search.Permutations = append(search.Permutations, second)
	search.SpatialTilings = []cudalayout.SpatialTiling{cudalayout.NewSpatialTiling(map[string]int{"M": 1}, nil)}
	bypass := cudalayout.NewDatatypeBypass()
	bypass.SetBypass(cudalayout.LevelShared, cudalayout.TensorA, false)
	search.Bypasses = []cudalayout.DatatypeBypass{bypass}
	before := cloneMapSpace(cudalayout.MapSpace{Problem: search.Problem, Factorization: search.Factorizations[0], Permutation: search.Permutations[0], Spatial: search.SpatialTilings[0], Bypass: search.Bypasses[0]})
	stats, err := search.Search(context.Background())
	if err != nil || stats.Best == nil {
		t.Fatal(stats, err)
	}
	if !reflect.DeepEqual(stats.Best.Mapping, before) {
		t.Fatal("first tie not retained")
	}
	stats.Best.Mapping.Problem["M"] = 9
	stats.Best.Mapping.Factorization.Levels[0] = cudalayout.LevelDRAM
	stats.Best.Mapping.Factorization.Factors[cudalayout.LevelDRAM]["M"] = 9
	stats.Best.Mapping.Permutation.Orders[cudalayout.LevelDRAM][0] = "K"
	stats.Best.Mapping.Spatial.CTAMesh["M"] = 9
	stats.Best.Mapping.Bypass.Bypassed[cudalayout.LevelShared][cudalayout.TensorA] = true
	got := cudalayout.MapSpace{Problem: search.Problem, Factorization: search.Factorizations[0], Permutation: search.Permutations[0], Spatial: search.SpatialTilings[0], Bypass: search.Bypasses[0]}
	if !reflect.DeepEqual(got, before) {
		t.Fatal("best evidence aliases input")
	}
	stats.Best.Movement.Levels[0].Tensors[cudalayout.TensorA] = cudaroofline.TensorMovement{}
	again, err := search.Search(context.Background())
	if err != nil || again.Best == nil || again.Best.Movement.Levels[0].Tensors[cudalayout.TensorA].ResidentBytes != 1 {
		t.Fatal("result movement retained mutable search state", err)
	}
}

func TestMapspaceTemporalInputs(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*LinearPrunedSearch)
	}{
		{"missing batch", func(s *LinearPrunedSearch) { delete(s.Problem, "B") }},
		{"unknown dimension", func(s *LinearPrunedSearch) { s.Problem["Q"] = 1 }},
		{"zero extent", func(s *LinearPrunedSearch) { s.Problem["M"] = 0 }},
		{"missing dtype", func(s *LinearPrunedSearch) { delete(s.DTypes, cudalayout.TensorC) }},
		{"unknown dtype", func(s *LinearPrunedSearch) { s.DTypes[cudalayout.TensorRole(99)] = 1 }},
		{"zero dtype", func(s *LinearPrunedSearch) { s.DTypes[cudalayout.TensorA] = 0 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := temporalOrderSearch()
			tt.mutate(s)
			s.Constraints.MaxSharedMemoryPerBlock = 1 // Invalid metadata must not hide behind pruning.
			stats, err := s.Search(context.Background())
			if !errors.Is(err, cudalayout.ErrUnsupported) || stats.EvaluatedCandidates != 0 || stats.Best != nil {
				t.Fatalf("stats=%+v error=%v", stats, err)
			}
		})
	}
}

func TestMapspaceTemporalDomainAndDefault(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*LinearPrunedSearch)
	}{
		{"missing active order", func(s *LinearPrunedSearch) {
			s.Permutations = []cudalayout.LoopPermutation{{Orders: map[cudalayout.StorageLevel][]string{cudalayout.LevelShared: nil}}}
		}},
		{"explicit empty order", func(s *LinearPrunedSearch) { s.Permutations = []cudalayout.LoopPermutation{{}} }},
		{"missing DRAM", func(s *LinearPrunedSearch) { s.Factorizations[0].Levels = s.Factorizations[0].Levels[:3] }},
		{"unlisted factors", func(s *LinearPrunedSearch) {
			s.Factorizations[0].Factors[cudalayout.StorageLevel(99)] = map[string]int{"M": 1}
		}},
		{"duplicate levels", func(s *LinearPrunedSearch) { s.Factorizations[0].Levels[1] = cudalayout.LevelRegister }},
		{"unknown bypass", func(s *LinearPrunedSearch) {
			s.Bypasses[0].SetBypass(cudalayout.LevelShared, cudalayout.TensorRole(99), true)
		}},
		{"DRAM bypass", func(s *LinearPrunedSearch) { s.Bypasses[0].SetBypass(cudalayout.LevelDRAM, cudalayout.TensorC, true) }},
		{"unlisted bypass", func(s *LinearPrunedSearch) {
			s.Bypasses[0].SetBypass(cudalayout.StorageLevel(99), cudalayout.TensorC, false)
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := temporalOrderSearch()
			tt.mutate(s)
			stats, err := s.Search(context.Background())
			if err != nil || stats.EvaluatedCandidates != 0 || stats.Best != nil || stats.DownstreamSkipped != len(s.Permutations) {
				t.Fatalf("stats=%+v error=%v", stats, err)
			}
		})
	}
	s := temporalOrderSearch()
	s = NewLinearPrunedSearch(s.Factorizations, nil, nil, s.Constraints, s.DTypes, s.Problem)
	stats, err := s.Search(context.Background())
	if err != nil || stats.EvaluatedCandidates != 1 || stats.Best == nil || !reflect.DeepEqual(stats.Best.Mapping.Permutation.Orders[cudalayout.LevelDRAM], []string{"B", "M", "N", "K"}) {
		t.Fatalf("default stats=%+v error=%v", stats, err)
	}
	parts, err := s.Split(1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := parts[0].Search(context.Background())
	if err != nil || !reflect.DeepEqual(got, stats) {
		t.Fatalf("split lost default: %+v %v", got, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	stats, err = s.Search(ctx)
	if !errors.Is(err, context.Canceled) || stats.EvaluatedCandidates != 0 || stats.Best != nil {
		t.Fatal(stats, err)
	}
}

func TestMapspaceTemporalOverflow(t *testing.T) {
	s := temporalOrderSearch()
	s.DTypes[cudalayout.TensorC] = int(^uint(0) >> 1)
	stats, err := s.Search(context.Background())
	if err == nil || stats.EvaluatedCandidates != 0 || stats.Best != nil {
		t.Fatalf("overflow admitted: %+v %v", stats, err)
	}
}

func TestMapspaceTemporalFactorizationAndBypass(t *testing.T) {
	search := temporalOrderSearch()
	search.Permutations = search.Permutations[1:]
	first := search.Factorizations[0]
	second := cudalayout.NewIndexFactorization(first.Levels, first.Factors)
	second.Factors[cudalayout.LevelL2] = map[string]int{"B": 1, "M": 2, "N": 2, "K": 2}
	second.Factors[cudalayout.LevelDRAM] = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
	// Give the second factorization its active L2 order explicitly. The same
	// permutation remains valid for the first factorization's unit L2 loops.
	search.Permutations[0].Orders[cudalayout.LevelL2] = []string{"M", "N", "K"}
	search.Factorizations = append(search.Factorizations, second)
	stats, err := search.Search(context.Background())
	if err != nil || stats.Best == nil || stats.EvaluatedCandidates != 2 || stats.Best.Movement.DRAM.TotalBytes != 12 || stats.Best.Mapping.Factorization.Factors[cudalayout.LevelL2]["M"] != 2 {
		t.Fatalf("full L2 factorization not scored: %+v %v", stats, err)
	}
	search.Factorizations = search.Factorizations[1:]
	for _, tt := range []struct {
		level cudalayout.StorageLevel
		want  uint64
	}{
		{cudalayout.LevelShared, 12}, {cudalayout.LevelL2, 16},
	} {
		bypass := cudalayout.NewDatatypeBypass()
		bypass.SetBypass(tt.level, cudalayout.TensorA, true)
		search.Bypasses = []cudalayout.DatatypeBypass{bypass}
		stats, err := search.Search(context.Background())
		if err != nil || stats.Best == nil || stats.Best.Movement.DRAM.TotalBytes != tt.want || stats.Best.Movement.DRAM.TensorTransfers["C"] != 4 {
			t.Fatalf("bypass at %s: %+v %v, want %d bytes with four C writes", tt.level, stats.Best, err, tt.want)
		}
	}
}
