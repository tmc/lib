package cudasearch

import (
	"context"
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
			cudalayout.LevelShared: {"M", "N", "K"},
		}),
		cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{
			cudalayout.LevelShared: {"K", "M", "N"},
		}),
		cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{
			cudalayout.LevelShared: {"N", "K", "M"},
		}),
	}

	bypass1 := cudalayout.NewDatatypeBypass()
	bypass2 := cudalayout.NewDatatypeBypass()
	bypass2.SetBypass(cudalayout.LevelShared, cudalayout.TensorC, true)
	bypasses := []cudalayout.DatatypeBypass{bypass1, bypass2}

	constraints := cudalayout.HardwareConstraints{
		MaxSharedMemoryPerBlock: 101376, // GB10 SM121
		MaxRegistersPerBlock:    65536,
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
