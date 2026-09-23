package builder

import (
	"math"
	"testing"
)

func TestInvalidMasksLeaveBuilderUnchanged(t *testing.T) {
	for _, mask := range []Type{{Shape: []int64{2, 3}, ElementType: "f32"}, {Shape: []int64{2, 4}, ElementType: "i1"}, {Shape: []int64{-1, 3}, ElementType: "i1"}, {Shape: []int64{1, 1, 1, 3}, ElementType: "i1"}} {
		typ := func(shape ...int64) Type { return Type{Shape: shape, ElementType: "f32"} }
		b := NewFunc("main", typ(2, 2, 4), typ(2, 3, 4), typ(2, 3, 5), mask)
		if _, err := b.MaskedScaledDotProductAttention(b.Arg(0), b.Arg(1), b.Arg(2), b.Arg(3)); err == nil {
			t.Fatalf("accepted mask %s", mask)
		}
		if len(b.Ops) != 0 || b.nextID != 0 {
			t.Fatal("invalid mask changed builder")
		}
	}
}

func TestCausalMaskValidation(t *testing.T) {
	for _, dims := range [][3]int64{{0, 1, 0}, {1, 0, 0}, {1, 1, -1}, {2, 1, math.MaxInt64}, {math.MaxInt64, 2, 0}} {
		b := NewFunc("main")
		if _, err := b.CausalMask(dims[0], dims[1], dims[2]); err == nil {
			t.Fatalf("accepted %v", dims)
		}
		if len(b.Ops) != 0 || b.nextID != 0 {
			t.Fatal("invalid causal mask changed builder")
		}
	}
}

func TestAndValidation(t *testing.T) {
	for _, typ := range []Type{{Shape: []int64{3}, ElementType: "i1"}, {Shape: []int64{2}, ElementType: "i32"}} {
		b := NewFunc("main", Type{Shape: []int64{2}, ElementType: "i1"}, typ)
		if _, err := b.And(b.Arg(0), b.Arg(1)); err == nil {
			t.Fatalf("accepted %s", typ)
		}
		if len(b.Ops) != 0 || b.nextID != 0 {
			t.Fatal("invalid and changed builder")
		}
	}
}
