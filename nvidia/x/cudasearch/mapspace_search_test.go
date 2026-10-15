package cudasearch

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

func TestLinearPrunedSearchSkipOnFailure(t *testing.T) {
	levels := []cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelDRAM}
	problem := map[string]int{"B": 1, "M": 256, "N": 256, "K": 256}
	dtypes := map[cudalayout.TensorRole]int{
		cudalayout.TensorA: 2,
		cudalayout.TensorB: 2,
		cudalayout.TensorC: 4,
	}

	// Fact 1: Valid tile M=64, N=64, K=64 (32,768 B <= 101,376 B)
	factValid := cudalayout.NewIndexFactorization(levels, map[cudalayout.StorageLevel]map[string]int{
		cudalayout.LevelRegister: {"B": 1, "M": 8, "N": 8, "K": 1},
		cudalayout.LevelShared:   {"B": 1, "M": 8, "N": 8, "K": 64},
		cudalayout.LevelDRAM:     {"B": 1, "M": 4, "N": 4, "K": 4},
	})

	// Fact 2: Invalid tile M=256, N=256, K=128 (> 101,376 B)
	factHuge := cudalayout.NewIndexFactorization(levels, map[cudalayout.StorageLevel]map[string]int{
		cudalayout.LevelRegister: {"B": 1, "M": 16, "N": 16, "K": 1},
		cudalayout.LevelShared:   {"B": 1, "M": 16, "N": 16, "K": 128},
		cudalayout.LevelDRAM:     {"B": 1, "M": 1, "N": 1, "K": 2},
	})

	// Fact 3: Another valid tile M=32, N=64, K=64 (20,480 B) with 128B TMA alignment
	factSmall := cudalayout.NewIndexFactorization(levels, map[cudalayout.StorageLevel]map[string]int{
		cudalayout.LevelRegister: {"B": 1, "M": 4, "N": 2, "K": 1},
		cudalayout.LevelShared:   {"B": 1, "M": 8, "N": 32, "K": 64},
		cudalayout.LevelDRAM:     {"B": 1, "M": 8, "N": 4, "K": 4},
	})

	facts := []cudalayout.IndexFactorization{factValid, factHuge, factSmall}

	perms := []cudalayout.LoopPermutation{
		cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{
			cudalayout.LevelRegister: {"M", "N"}, cudalayout.LevelDRAM: {"M", "N", "K"}, cudalayout.LevelShared: {"M", "N", "K"},
		}),
		cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{
			cudalayout.LevelRegister: {"M", "N"}, cudalayout.LevelDRAM: {"M", "N", "K"}, cudalayout.LevelShared: {"K", "M", "N"},
		}),
		cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{
			cudalayout.LevelRegister: {"M", "N"}, cudalayout.LevelDRAM: {"M", "N", "K"}, cudalayout.LevelShared: {"N", "K", "M"},
		}),
	}

	bypass1 := cudalayout.NewDatatypeBypass()
	bypass2 := cudalayout.NewDatatypeBypass()
	bypass2.SetBypass(cudalayout.LevelShared, cudalayout.TensorC, true)
	bypasses := []cudalayout.DatatypeBypass{bypass1, bypass2}

	constraints := cudalayout.HardwareConstraints{
		MaxSharedMemoryPerBlock: 101376, // GB10 SM121
		MaxRegistersPerBlock:    65536,
		MaxThreadsPerBlock:      128,
		TMAAlignmentBytes:       128,
	}

	search := NewLinearPrunedSearch(facts, perms, bypasses, constraints, dtypes, problem)
	stats, err := search.Search(context.Background())
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	downstreamPerFact := len(perms) * len(bypasses) // 3 * 2 = 6

	if stats.FactorizationsTested != 3 {
		t.Errorf("FactorizationsTested = %d, want 3", stats.FactorizationsTested)
	}
	if stats.FactorizationsPruned != 1 {
		t.Errorf("FactorizationsPruned = %d, want 1", stats.FactorizationsPruned)
	}
	expectedSkipped := 1 * downstreamPerFact // 6
	if stats.DownstreamSkipped != expectedSkipped {
		t.Errorf("DownstreamSkipped = %d, want %d", stats.DownstreamSkipped, expectedSkipped)
	}
	expectedEvaluated := 2 * downstreamPerFact // 12
	if stats.EvaluatedCandidates != expectedEvaluated {
		t.Errorf("EvaluatedCandidates = %d, want %d", stats.EvaluatedCandidates, expectedEvaluated)
	}

	// Test Split
	splits, err := search.Split(2)
	if err != nil {
		t.Fatalf("Split failed: %v", err)
	}
	if len(splits) != 2 {
		t.Errorf("len(splits) = %d, want 2", len(splits))
	}
	totalInSplits := len(splits[0].Factorizations) + len(splits[1].Factorizations)
	if totalInSplits != 3 {
		t.Errorf("total factorizations in splits = %d, want 3", totalInSplits)
	}
}

func TestLinearPrunedSearchKeepsBypassOnlyFit(t *testing.T) {
	search, bypasses := bypassFitSearch()
	search.Bypasses = bypasses[:2]
	stats, err := search.Search(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.FactorizationsPruned != 0 || stats.EvaluatedCandidates != 1 || stats.DownstreamSkipped != 1 {
		t.Fatalf("stats = %+v, want the C-bypassed mapping evaluated and one failing variant skipped", stats)
	}
}

func TestLinearPrunedSearchRejectsMismatchedFactorization(t *testing.T) {
	search, _ := bypassFitSearch()
	search.Bypasses = []cudalayout.DatatypeBypass{cudalayout.NewDatatypeBypass()}
	search.Bypasses[0].SetBypass(cudalayout.LevelShared, cudalayout.TensorC, true)
	fact := search.Factorizations[0]
	fact.Factors[cudalayout.LevelDRAM]["M"] = 2
	search.Factorizations = []cudalayout.IndexFactorization{fact}
	stats, err := search.Search(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.FactorizationsPruned != 1 || stats.DownstreamSkipped != 1 || stats.EvaluatedCandidates != 0 {
		t.Fatalf("stats = %+v, want one invalid factorization pruned", stats)
	}
}

func TestLinearPrunedSearchRechecksEachBypass(t *testing.T) {
	search, bypasses := bypassFitSearch()
	search.Bypasses = bypasses
	stats, err := search.Search(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.DownstreamSkipped != 2 || stats.EvaluatedCandidates != 1 {
		t.Fatalf("stats = %+v, want 2 bypass variants skipped and 1 evaluated", stats)
	}
}

func TestLinearPrunedSearchSplit(t *testing.T) {
	for _, tt := range []struct{ total, n int }{{5, 4}, {7, 4}, {3, 3}, {1, 4}, {0, 2}} {
		t.Run(fmt.Sprintf("%d_%d", tt.total, tt.n), func(t *testing.T) {
			facts := make([]cudalayout.IndexFactorization, tt.total)
			for i := range facts {
				facts[i] = cudalayout.NewIndexFactorization(
					[]cudalayout.StorageLevel{cudalayout.LevelRegister},
					map[cudalayout.StorageLevel]map[string]int{cudalayout.LevelRegister: {"M": i + 1}},
				)
			}
			search := NewLinearPrunedSearch(facts, nil, nil, cudalayout.HardwareConstraints{}, nil, nil)
			parts, err := search.Split(tt.n)
			if err != nil {
				t.Fatal(err)
			}
			wantParts := tt.n
			if wantParts > tt.total {
				wantParts = tt.total
			}
			if len(parts) != wantParts {
				t.Fatalf("Split returned %d partitions, want %d", len(parts), wantParts)
			}
			seen := make([]int, 0, tt.total)
			partitionSizes := make([]int, 0, len(parts))
			for _, part := range parts {
				if part == nil {
					t.Fatal("Split returned a nil partition")
				}
				partitionSizes = append(partitionSizes, len(part.Factorizations))
				if tt.total > 0 && len(part.Factorizations) == 0 {
					t.Fatal("Split returned an empty partition")
				}
				for _, fact := range part.Factorizations {
					seen = append(seen, fact.Factors[cudalayout.LevelRegister]["M"])
				}
			}
			if len(partitionSizes) > 1 {
				minSize, maxSize := partitionSizes[0], partitionSizes[0]
				for _, size := range partitionSizes[1:] {
					if size < minSize {
						minSize = size
					}
					if size > maxSize {
						maxSize = size
					}
				}
				if maxSize-minSize > 1 {
					t.Fatalf("partition sizes = %v, want sizes differing by at most one", partitionSizes)
				}
			}
			if len(seen) != tt.total {
				t.Fatalf("partition total = %d, want %d", len(seen), tt.total)
			}
			for i, got := range seen {
				if got != i+1 {
					t.Fatalf("partition order = %v, want 1..%d", seen, tt.total)
				}
			}
		})
	}
}

func bypassFitSearch() (*LinearPrunedSearch, []cudalayout.DatatypeBypass) {
	levels := []cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelDRAM}
	fact := cudalayout.NewIndexFactorization(levels, map[cudalayout.StorageLevel]map[string]int{
		cudalayout.LevelRegister: {"B": 1, "M": 1, "N": 1, "K": 1},
		cudalayout.LevelShared:   {"B": 1, "M": 256, "N": 256, "K": 64},
		cudalayout.LevelDRAM:     {"B": 1, "M": 1, "N": 1, "K": 1},
	})
	problem := map[string]int{"B": 1, "M": 256, "N": 256, "K": 64}
	dtypes := map[cudalayout.TensorRole]int{cudalayout.TensorA: 2, cudalayout.TensorB: 2, cudalayout.TensorC: 4}
	noBypass := cudalayout.NewDatatypeBypass()
	bypassC := cudalayout.NewDatatypeBypass()
	bypassC.SetBypass(cudalayout.LevelShared, cudalayout.TensorC, true)
	bypassA := cudalayout.NewDatatypeBypass()
	bypassA.SetBypass(cudalayout.LevelShared, cudalayout.TensorA, true)
	bypasses := []cudalayout.DatatypeBypass{noBypass, bypassC, bypassA}
	constraints := cudalayout.HardwareConstraints{MaxSharedMemoryPerBlock: 101376, MaxRegistersPerBlock: 65536, MaxThreadsPerBlock: 128}
	search := NewLinearPrunedSearch([]cudalayout.IndexFactorization{fact}, nil, bypasses, constraints, dtypes, problem)
	return search, bypasses
}

func TestSplitPartitionsAreCopies(t *testing.T) {
	facts := []cudalayout.IndexFactorization{cudalayout.NewIndexFactorization(nil, nil)}
	search := NewLinearPrunedSearch(facts, nil, nil, cudalayout.HardwareConstraints{}, nil, nil)
	parts, err := search.Split(1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(parts[0].Factorizations[0], facts[0]) {
		t.Fatal("split factorization differs from input")
	}
}
