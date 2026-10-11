package cudasearch

import (
	"context"
	"math/big"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

func TestMapspaceCardinalityBounds(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                          string
		facts, permutations, bypasses int
		want                          int
		refuse                        bool
	}{
		{"small", 3, 2, 5, 10, false},
		{"exact downstream", 1, maxInt, 1, maxInt, false},
		{"exact total", maxInt, 1, 1, 1, false},
		{"downstream overflow", 1, maxInt, 2, 0, true},
		{"downstream wraps positive", 1, maxInt, 3, 0, true},
		{"downstream wraps zero", 1, maxInt/2 + 1, 4, 0, true},
		{"total overflow", 2, maxInt/2 + 1, 1, 0, true},
		{"total wraps positive", maxInt, 3, 1, 0, true},
		{"near total limit", maxInt / 3, 3, 1, 3, false},
		{"zero bypasses", maxInt, maxInt, 0, 0, false},
		{"zero permutations", maxInt, 0, maxInt, 0, false},
		{"zero facts bounded downstream", 0, maxInt, 1, 0, false},
		{"zero facts unbounded downstream", 0, maxInt, 2, 0, false},
		{"negative fact count", -1, 2, 3, 0, true},
		{"negative permutation count", 1, -1, 3, 0, true},
		{"negative bypass count", 1, 2, -1, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := mapspaceDownstreamCount(tc.facts, tc.permutations, tc.bypasses)
			if tc.refuse {
				if err == nil || got != 0 {
					t.Fatalf("unrepresentable count admitted: %d, %v", got, err)
				}
			} else if err != nil || got != tc.want {
				t.Fatalf("count = %d, %v; want %d", got, err, tc.want)
			}
		})
	}
}

func TestMapspaceCardinalityConservation(t *testing.T) {
	search, bypasses := bypassFitSearch()
	search.Bypasses = bypasses
	search.Factorizations = append(search.Factorizations, cudalayout.IndexFactorization{})
	search.Permutations = []cudalayout.LoopPermutation{
		cudalayout.NewLoopPermutation(nil),
		cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{cudalayout.LevelShared: {"unknown"}}),
	}
	stats, err := search.Search(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := len(search.Factorizations) * len(search.Permutations) * len(search.Bypasses)
	if stats.FactorizationsTested != 2 || stats.FactorizationsPruned != 1 || stats.EvaluatedCandidates != 1 || stats.DownstreamSkipped != want-1 {
		t.Fatalf("candidate counts not conserved: %+v; total=%d", stats, want)
	}
}

func FuzzMapspaceCardinality(f *testing.F) {
	maxInt := uint64(^uint(0) >> 1)
	for _, seed := range [][3]uint64{{0, 0, 0}, {3, 2, 5}, {1, maxInt, 1}, {1, maxInt, 2}, {maxInt, 2, 2}} {
		f.Add(seed[0], seed[1], seed[2])
	}
	f.Fuzz(func(t *testing.T, facts, permutations, bypasses uint64) {
		facts &= maxInt
		permutations &= maxInt
		bypasses &= maxInt
		limit := new(big.Int).SetUint64(maxInt)
		downstream := new(big.Int).Mul(new(big.Int).SetUint64(permutations), new(big.Int).SetUint64(bypasses))
		total := new(big.Int).Mul(new(big.Int).Set(downstream), new(big.Int).SetUint64(facts))
		wantError := facts != 0 && (downstream.Cmp(limit) > 0 || total.Cmp(limit) > 0)
		if facts == 0 {
			downstream.SetUint64(0)
		}
		got, err := mapspaceDownstreamCount(int(facts), int(permutations), int(bypasses))
		if wantError {
			if err == nil || got != 0 {
				t.Fatalf("overflow accepted: %d, %v", got, err)
			}
		} else if err != nil || uint64(got) != downstream.Uint64() {
			t.Fatalf("exact product differs: %d, %v; want %s", got, err, downstream)
		}
	})
}
