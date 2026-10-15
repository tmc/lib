package cudasearch_test

import (
	"context"
	"fmt"
	"reflect"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudasearch"
)

// exactCostSearch has two M tiles and two temporal K tiles above Shared.
// L2 retains A/B but bypasses C, whose Shared keys are 0,1,0,1 for K,M
// versus 0,0,1,1 for M,K. Increasing h changes tile extents, not loop count
// in the analytical estimator. Zero capacity limits make no device-fit claim.
func exactCostSearch(h uint64, spatial, split bool) *cudasearch.LinearPrunedSearch {
	levels := []cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelL2, cudalayout.LevelDRAM}
	factors := make(map[cudalayout.StorageLevel]map[string]int)
	for _, level := range levels {
		factors[level] = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
	}
	factors[cudalayout.LevelRegister]["K"] = int(h)
	factors[cudalayout.LevelL2]["M"] = 2
	factors[cudalayout.LevelL2]["K"] = 2
	bypass := cudalayout.NewDatatypeBypass()
	bypass.SetBypass(cudalayout.LevelL2, cudalayout.TensorC, true)
	dtypes := map[cudalayout.TensorRole]int{cudalayout.TensorA: 4, cudalayout.TensorB: 4, cudalayout.TensorC: 4}
	k := 2 * int(h)
	if split {
		// Two CTA partial owners, each still visiting two temporal K tiles.
		factors[cudalayout.LevelL2]["K"] = 4
		k *= 2
		dtypes[cudalayout.TensorC] = 1
	}
	permutation := func(order []string) cudalayout.LoopPermutation {
		orders := make(map[cudalayout.StorageLevel][]string)
		for _, level := range levels {
			orders[level] = []string{"B", "M", "N", "K"}
		}
		orders[cudalayout.LevelL2] = order
		return cudalayout.NewLoopPermutation(orders)
	}
	s := cudasearch.NewLinearPrunedSearch(
		[]cudalayout.IndexFactorization{cudalayout.NewIndexFactorization(levels, factors)},
		[]cudalayout.LoopPermutation{permutation([]string{"K", "M", "B", "N"}), permutation([]string{"M", "K", "B", "N"})},
		[]cudalayout.DatatypeBypass{bypass}, cudalayout.HardwareConstraints{}, dtypes,
		map[string]int{"B": 1, "M": 2, "N": 1, "K": k})
	if spatial {
		s.SpatialProfile = &cudaroofline.GEMMSpatialProfile{}
	}
	if split {
		s.SplitK = true
		s.SpatialTilings = []cudalayout.SpatialTiling{cudalayout.NewSpatialTiling(map[string]int{"K": 2}, nil)}
	}
	return s
}

func exactCostRun(t *testing.T, s *cudasearch.LinearPrunedSearch, evaluated int) cudasearch.MapSpaceSearchStats {
	t.Helper()
	stats, err := s.Search(context.Background())
	if err != nil || stats.FactorizationsTested != 1 || stats.FactorizationsPruned != 0 || stats.DownstreamSkipped != 0 || stats.UnsupportedCandidates != 0 || stats.EvaluatedCandidates != evaluated || stats.Best == nil {
		t.Fatalf("search: %+v, error %v; want %d admitted evaluations", stats, err, evaluated)
	}
	return stats
}

func TestMapspaceExactCostCloseCounts(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("close-cost witness requires 64-bit admitted extents")
	}
	for _, tt := range []struct {
		name              string
		spatial, split    bool
		inputs, bad, good uint64
		useful            uint64
	}{
		{"temporal", false, false, 24, 24, 8, 8},
		{"spatial identity", true, false, 24, 24, 8, 8},
		// A32h+B16h. Each of two owners has C read1/write2 versus
		// read0/write1 per M cell: phase1 C12 versus C4. The two-cell
		// merge reads4/writes2, so whole-recipe C18 versus C10.
		{"split K", true, true, 48, 18, 10, 16},
	} {
		for _, h := range []uint64{1 << 54, 1 << 55} {
			for _, reverse := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/h%d/reverse%t", tt.name, h, reverse), func(t *testing.T) {
					s := exactCostSearch(h, tt.spatial, tt.split)
					want := []uint64{tt.inputs*h + tt.bad, tt.inputs*h + tt.good}
					var intensity [2]float64
					for i, perm := range s.Permutations {
						one := *s
						one.Permutations = []cudalayout.LoopPermutation{perm}
						stats := exactCostRun(t, &one, 1)
						dram := stats.Best.TotalDRAM()
						c := []uint64{tt.bad, tt.good}[i]
						if dram.TotalBytes != want[i] || dram.TensorTransfers["A"] != tt.inputs*h*2/3 || dram.TensorTransfers["B"] != tt.inputs*h/3 || dram.TensorTransfers["C"] != c {
							t.Fatalf("candidate %d: %+v; want total %d and C%d", i, dram, want[i], c)
						}
						if (stats.Best.SpatialMovement != nil) != tt.spatial || (stats.Best.SplitK != nil) != tt.split {
							t.Fatalf("candidate used the wrong scoring route: %+v", stats.Best)
						}
						if tt.split && (stats.Best.Movement.DRAM.TotalBytes+6 != dram.TotalBytes || stats.Best.SplitK.ReductionAdditions != 2) {
							t.Fatalf("merge not included in whole-recipe traffic: %+v", stats.Best.SplitK)
						}
						intensity[i] = stats.BestIntensity
						if intensity[i] != float64(tt.useful*h)/float64(want[i]) {
							t.Fatal("reported intensity changed", stats.BestIntensity)
						}
					}
					if want[0] <= want[1] || intensity[0] != intensity[1] {
						t.Fatalf("witness lacks unequal exact costs with rounded intensity tie: %v, %v", want, intensity)
					}
					better := s.Permutations[1]
					if reverse {
						s.Permutations[0], s.Permutations[1] = s.Permutations[1], s.Permutations[0]
					}
					stats := exactCostRun(t, s, 2)
					if stats.Best.TotalDRAM().TotalBytes != want[1] || stats.BestIntensity != intensity[1] || !reflect.DeepEqual(stats.Best.Mapping.Permutation, better) {
						t.Fatalf("rounded intensity hid cheaper candidate: bytes%d want%d, order%v", stats.Best.TotalDRAM().TotalBytes, want[1], stats.Best.Mapping.Permutation)
					}
				})
			}
		}
	}
}

func TestMapspaceExactCostFirstTie(t *testing.T) {
	for _, tt := range []struct {
		name           string
		spatial, split bool
	}{
		{"temporal", false, false},
		{"spatial identity", true, false},
		{"split K", true, true},
	} {
		for _, reverse := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/reverse%t", tt.name, reverse), func(t *testing.T) {
				s := exactCostSearch(1, tt.spatial, tt.split)
				second := exactCostSearch(1, tt.spatial, tt.split).Permutations[1]
				second.Orders[cudalayout.LevelL2] = []string{"B", "M", "K", "N"}
				s.Permutations = []cudalayout.LoopPermutation{s.Permutations[1], second}
				if reverse {
					s.Permutations[0], s.Permutations[1] = s.Permutations[1], s.Permutations[0]
				}
				var cost uint64
				for i, perm := range s.Permutations {
					one := *s
					one.Permutations = []cudalayout.LoopPermutation{perm}
					stats := exactCostRun(t, &one, 1)
					if i == 0 {
						cost = stats.Best.TotalDRAM().TotalBytes
					} else if stats.Best.TotalDRAM().TotalBytes != cost {
						t.Fatal("tie candidates have different exact costs")
					}
				}
				stats := exactCostRun(t, s, 2)
				if stats.Best.TotalDRAM().TotalBytes != cost || !reflect.DeepEqual(stats.Best.Mapping.Permutation, s.Permutations[0]) {
					t.Fatal("exact tie replaced the first mapping", stats)
				}
			})
		}
	}
}
