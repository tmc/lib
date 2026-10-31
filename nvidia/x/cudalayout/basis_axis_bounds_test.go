package cudalayout_test

import (
	"errors"
	"math"
	"math/bits"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

func TestBasisAxisMaxIntNoPanic(t *testing.T) {
	defer func() {
		if p := recover(); p != nil {
			t.Errorf("basis offset panicked for admitted axis: %v", p)
		}
	}()
	l, err := cudalayout.NewLayout(cudalayout.IntShape(2), cudalayout.BasisStride(math.MaxInt, 1))
	if err != nil {
		if !errors.Is(err, cudalayout.ErrOverflow) {
			t.Fatalf("constructor: %v, want ErrOverflow", err)
		}
		return
	}
	canary := []int{71, 83}
	got, err := l.BasisOffset(1, canary)
	if !errors.Is(err, cudalayout.ErrOverflow) || !slices.Equal(got, canary) || !slices.Equal(canary, []int{71, 83}) {
		t.Errorf("offset = %v, %v; canary = %v, want unchanged canary and ErrOverflow", got, err, canary)
	}
}

func TestBasisAxisBounds(t *testing.T) {
	for _, tt := range []struct {
		name string
		axis int
		want error
	}{
		{"negative", -1, cudalayout.ErrUnsupported},
		{"length overflow", math.MaxInt, cudalayout.ErrOverflow},
		{"byte overflow", math.MaxInt - 1, cudalayout.ErrOverflow},
		{"first byte overflow", math.MaxInt / (bits.UintSize / 8), cudalayout.ErrOverflow},
		{"largest representable vector", math.MaxInt/(bits.UintSize/8) - 1, nil},
		{"zero", 0, nil},
		{"nonzero", 3, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := cudalayout.NewLayout(cudalayout.IntShape(2), cudalayout.BasisStride(tt.axis, 1))
			if !errors.Is(err, tt.want) {
				t.Fatalf("NewLayout axis %d = %v, want %v", tt.axis, err, tt.want)
			}
		})
	}
}

func TestBasisAxisReuse(t *testing.T) {
	l, err := cudalayout.NewLayout(cudalayout.TupleShape(cudalayout.IntShape(2), cudalayout.IntShape(3)), cudalayout.TupleStride(cudalayout.BasisStride(0, 2), cudalayout.BasisStride(2, -3)))
	if err != nil {
		t.Fatal(err)
	}
	dst := []int{91, 92, 93, 94, 95}
	got, err := l.BasisOffset(5, dst)
	if err != nil || !slices.Equal(got, []int{2, 0, -6}) || &got[0] != &dst[0] || !slices.Equal(dst[3:], []int{94, 95}) {
		t.Fatalf("offset = %v, %v; backing = %v", got, err, dst)
	}
	for _, i := range []int{-1, l.Size()} {
		before := slices.Clone(dst)
		got, err := l.BasisOffset(i, dst)
		if !errors.Is(err, cudalayout.ErrOutOfBounds) || !slices.Equal(got, before) || !slices.Equal(dst, before) {
			t.Fatalf("offset(%d) = %v, %v; backing = %v", i, got, err, dst)
		}
	}
}
