package cudalayout

import (
	"errors"
	"math"
	"testing"
)

func spatialTestFactors() (IndexFactorization, map[string]int) {
	return NewIndexFactorization([]StorageLevel{LevelRegister, LevelShared, LevelDRAM}, map[StorageLevel]map[string]int{
		LevelRegister: {"B": 1, "M": 1, "N": 1, "K": 1},
		LevelShared:   {"B": 1, "M": 4, "N": 4, "K": 4},
		LevelDRAM:     {"B": 1, "M": 2, "N": 2, "K": 2},
	}), map[string]int{"B": 1, "M": 8, "N": 8, "K": 8}
}

func TestSpatialTilingValidate(t *testing.T) {
	fact, problem := spatialTestFactors()
	tests := []struct {
		name  string
		cta   map[string]int
		warp  map[string]int
		want  error
		warps int
	}{
		{"identity", nil, nil, nil, 1},
		{"two axes", map[string]int{"M": 2, "N": 2}, map[string]int{"M": 2, "N": 2}, nil, 4},
		{"reduction subdivision", map[string]int{"K": 2}, map[string]int{"K": 4}, nil, 4},
		{"CTA does not divide outer tiles", map[string]int{"M": 4}, nil, ErrUnsupported, 1},
		{"warp does not divide inner tiles", nil, map[string]int{"M": 3}, ErrUnsupported, 3},
		{"unknown CTA dimension", map[string]int{"X": 1}, nil, ErrUnsupported, 1},
		{"unknown warp dimension", nil, map[string]int{"X": 1}, ErrUnsupported, 1},
		{"zero CTA factor", map[string]int{"M": 0}, nil, ErrUnsupported, 1},
		{"negative warp factor", nil, map[string]int{"M": -1}, ErrUnsupported, 0},
		{"warp product overflow", nil, map[string]int{"M": math.MaxInt, "N": 2}, ErrOverflow, 0},
		{"CTA product overflow", map[string]int{"M": math.MaxInt, "N": 2}, nil, ErrOverflow, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewSpatialTiling(tt.cta, tt.warp)
			err := s.Validate(fact, problem)
			if tt.want == nil && err != nil || tt.want != nil && !errors.Is(err, tt.want) {
				t.Fatalf("Validate() = %v, want %v", err, tt.want)
			}
			if got := s.TotalWarps(); got != tt.warps {
				t.Fatalf("TotalWarps() = %d, want %d", got, tt.warps)
			}
		})
	}
	for _, level := range []StorageLevel{LevelRegister, LevelShared} {
		broken := NewIndexFactorization(fact.Levels, fact.Factors)
		delete(broken.Factors, level)
		if err := NewSpatialTiling(nil, map[string]int{"M": 2}).Validate(broken, problem); err == nil {
			t.Fatalf("missing %s factors were accepted", level)
		}
	}
}
