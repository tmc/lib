package egraph

import (
	"errors"
	"math"
	"slices"
	"testing"
)

func TestAttentionMaskDeclaredSemantics(t *testing.T) {
	scores := []float32{0, 0, 0, 0}
	values := []float32{10, 20, 40, 80}
	tests := []struct {
		name     string
		mask     func() ([]bool, error)
		wantMask []bool
		want     float32
	}{
		{"dense", func() ([]bool, error) { return nil, nil }, nil, 37.5},
		{"causal_includes_query", func() ([]bool, error) { return CausalAttentionMaskChecked(1, 4) }, []bool{false, false, true, true}, 15},
		{"tree_prefix", func() ([]bool, error) { return SparseTreeMaskChecked([]int{0, 2, 2}, 4) }, []bool{false, true, false, true}, 25},
		{"fully_masked", func() ([]bool, error) { return SparseTreeMaskChecked(nil, 4) }, []bool{true, true, true, true}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask, err := tt.mask()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(mask, tt.wantMask) {
				t.Fatalf("mask=%v, want independently declared %v", mask, tt.wantMask)
			}
			for _, tile := range []int{1, 2, 3, 4} {
				got, err := OnlineAttentionRow(scores, values, mask, 4, 1, tile)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != 1 || math.Float32bits(got[0]) != math.Float32bits(tt.want) {
					t.Fatalf("tile %d output %v, want independent mean %v", tile, got, tt.want)
				}
			}
			reference, err := MaterializedAttentionReference(scores, values, mask, 4, 1)
			if err != nil || len(reference) != 1 || math.Float32bits(reference[0]) != math.Float32bits(tt.want) {
				t.Fatal("materialized mask semantics differ from independent mean", reference, err)
			}
			mutant := slices.Clone(mask)
			if len(mutant) == 0 {
				mutant = make([]bool, 4)
			}
			mutant[0] = !mutant[0]
			got, err := OnlineAttentionRow(scores, values, mutant, 4, 1, 2)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 1 || got[0] == tt.want {
				t.Fatal("altered mask semantics survived independent oracle", got)
			}
		})
	}
}

func TestAttentionMaskCheckedRefusals(t *testing.T) {
	for _, test := range []struct {
		name string
		run  func() ([]bool, error)
	}{
		{"negative_query", func() ([]bool, error) { return CausalAttentionMaskChecked(-1, 4) }},
		{"query_outside_sequence", func() ([]bool, error) { return CausalAttentionMaskChecked(4, 4) }},
		{"negative_sequence", func() ([]bool, error) { return SparseTreeMaskChecked(nil, -1) }},
		{"negative_tree_key", func() ([]bool, error) { return SparseTreeMaskChecked([]int{-1}, 4) }},
		{"tree_key_outside_sequence", func() ([]bool, error) { return SparseTreeMaskChecked([]int{4}, 4) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := test.run(); !errors.Is(err, ErrInvalidMaskIndex) {
				t.Fatal("invalid predicate admitted", err)
			}
		})
	}
}
