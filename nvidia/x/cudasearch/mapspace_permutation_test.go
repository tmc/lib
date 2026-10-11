package cudasearch

import (
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

func TestLinearPrunedSearchPermutationOrders(t *testing.T) {
	for _, tt := range []struct {
		name  string
		level cudalayout.StorageLevel
		order []string
		batch int
		valid bool
	}{
		{"active loops", cudalayout.LevelShared, []string{"M", "N", "K"}, 1, true},
		{"reordered loops", cudalayout.LevelShared, []string{"K", "N", "M"}, 1, true},
		{"explicit unit loop", cudalayout.LevelShared, []string{"B", "M", "N", "K"}, 1, true},
		{"empty unit-factor level", cudalayout.LevelRegister, nil, 1, true},
		{"explicit unit-factor level", cudalayout.LevelRegister, []string{"B", "M", "N", "K"}, 1, true},
		{"missing active loop", cudalayout.LevelShared, []string{"M", "N"}, 1, false},
		{"duplicate active loop", cudalayout.LevelShared, []string{"M", "M", "N", "K"}, 1, false},
		{"duplicate unit loop", cudalayout.LevelShared, []string{"B", "B", "M", "N", "K"}, 1, false},
		{"unknown loop", cudalayout.LevelShared, []string{"M", "N", "K", "Q"}, 1, false},
		{"absent level", cudalayout.LevelL2, []string{"M", "N", "K"}, 1, false},
		{"unknown level", cudalayout.StorageLevel(99), []string{"M", "N", "K"}, 1, false},
		{"non-unit batch required", cudalayout.LevelShared, []string{"M", "N", "K"}, 2, false},
		{"non-unit batch included", cudalayout.LevelShared, []string{"B", "M", "N", "K"}, 2, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			search, bypasses := bypassFitSearch()
			search.Bypasses = bypasses[1:2]
			search.Problem["B"] = tt.batch
			search.Factorizations[0].Factors[cudalayout.LevelShared]["B"] = tt.batch
			search.Constraints.MaxSharedMemoryPerBlock = 128 << 10
			search.Permutations = []cudalayout.LoopPermutation{{Orders: map[cudalayout.StorageLevel][]string{tt.level: tt.order}}}
			stats, err := search.Search(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			wantEvaluated, wantSkipped := 0, 1
			if tt.valid {
				wantEvaluated, wantSkipped = 1, 0
			}
			if stats.FactorizationsTested != 1 || stats.FactorizationsPruned != 0 || stats.EvaluatedCandidates != wantEvaluated || stats.DownstreamSkipped != wantSkipped {
				t.Fatalf("stats = %+v, want one valid factorization, %d evaluations, %d skipped", stats, wantEvaluated, wantSkipped)
			}
		})
	}
}

func TestLinearPrunedSearchPermutationSkipCounts(t *testing.T) {
	search, bypasses := bypassFitSearch()
	search.Bypasses = bypasses
	search.Permutations = []cudalayout.LoopPermutation{
		{Orders: map[cudalayout.StorageLevel][]string{cudalayout.LevelShared: {"M", "N"}}},
		{Orders: map[cudalayout.StorageLevel][]string{cudalayout.LevelShared: {"K", "M", "N"}}},
	}
	stats, err := search.Search(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.FactorizationsPruned != 0 || stats.DownstreamSkipped != 5 || stats.EvaluatedCandidates != 1 || stats.BestIntensity <= 0 {
		t.Fatalf("stats = %+v, want three invalid-order combinations and two capacity combinations skipped", stats)
	}
}

func TestLinearPrunedSearchPermutationRejectsUnmodeledLevel(t *testing.T) {
	search, bypasses := bypassFitSearch()
	search.Bypasses = bypasses[1:2]
	level := cudalayout.StorageLevel(99)
	search.Factorizations[0].Levels = append(search.Factorizations[0].Levels, level)
	search.Factorizations[0].Factors[level] = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
	search.Permutations = []cudalayout.LoopPermutation{{Orders: map[cudalayout.StorageLevel][]string{level: nil}}}
	stats, err := search.Search(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.EvaluatedCandidates != 0 || stats.DownstreamSkipped != 1 {
		t.Fatalf("stats = %+v, want the unmodeled storage level skipped", stats)
	}
}

func TestLinearPrunedSearchPermutationChecksEveryLevel(t *testing.T) {
	for _, tt := range []struct {
		name     string
		register []string
		shared   []string
		valid    bool
	}{
		{"different active factors", nil, []string{"M", "N", "K"}, true},
		{"invalid register order", []string{"B", "B"}, []string{"M", "N", "K"}, false},
		{"invalid shared order", nil, []string{"M", "N"}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			search, bypasses := bypassFitSearch()
			search.Bypasses = bypasses[1:2]
			search.Permutations = []cudalayout.LoopPermutation{{Orders: map[cudalayout.StorageLevel][]string{
				cudalayout.LevelRegister: tt.register,
				cudalayout.LevelShared:   tt.shared,
			}}}
			stats, err := search.Search(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			want := 0
			if tt.valid {
				want = 1
			}
			if stats.EvaluatedCandidates != want || stats.DownstreamSkipped != 1-want {
				t.Fatalf("stats = %+v, want %d evaluations", stats, want)
			}
		})
	}
}
