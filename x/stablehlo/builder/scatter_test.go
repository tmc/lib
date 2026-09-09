package builder

import (
	"math"
	"testing"
)

func TestScatterRowsValidation(t *testing.T) {
	f := func(shape ...int64) Type { return Type{Shape: shape, ElementType: "f32"} }
	i := func(shape ...int64) Type { return Type{Shape: shape, ElementType: "i64"} }
	for _, tt := range []struct {
		name                      string
		operand, indices, updates Type
	}{
		{"scalar", f(), i(1), f()},
		{"rank", f(3, 2), i(1), f(1)},
		{"index scalar", f(3, 2), i(), f(1, 2)},
		{"index type", f(3, 2), f(1), f(1, 2)},
		{"update count", f(3, 2), i(2), f(1, 2)},
		{"row shape", f(3, 2), i(1), f(1, 3)},
		{"element type", f(3, 2), i(1), i(1, 2)},
		{"dynamic", f(-1, 2), i(1), f(1, 2)},
		{"dynamic empty", f(0, -1), i(0), f(0, -1)},
		{"invalid index shape", f(3, 2), i(-2), f(1, 2)},
		{"overflow", f(math.MaxInt64, 2), i(1), f(1, 2)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := NewFunc("main", tt.operand, tt.indices, tt.updates)
			if _, err := b.ScatterRows(b.Arg(0), b.Arg(1), b.Arg(2)); err == nil {
				t.Fatal("accepted invalid scatter")
			}
			if len(b.Ops) != 0 || b.nextID != 0 {
				t.Fatal("error changed builder")
			}
		})
	}
}

func TestScatterRows16Indices(t *testing.T) {
	for _, element := range []string{"i16", "ui16"} {
		t.Run(element, func(t *testing.T) {
			b := NewFunc("main", Type{Shape: []int64{3, 2}, ElementType: "f32"}, Type{Shape: []int64{2}, ElementType: element}, Type{Shape: []int64{2, 2}, ElementType: "f32"})
			if _, err := b.ScatterRows(b.Arg(0), b.Arg(1), b.Arg(2)); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestBatchedCausalMaskValidation(t *testing.T) {
	i := func(shape ...int64) Type { return Type{Shape: shape, ElementType: "i64"} }
	for _, tt := range []struct {
		name              string
		q, k              int64
		positions, counts Type
	}{
		{"zero length", 0, 2, i(2), i(2)},
		{"negative length", 2, -1, i(2), i(2)},
		{"type", 2, 2, i(2), Type{Shape: []int64{2}, ElementType: "i32"}},
		{"shape", 2, 2, i(2), i(1)},
		{"dynamic", 2, 2, i(-1), i(-1)},
		{"dynamic empty", 2, 2, i(0, -1), i(0, -1)},
		{"overflow", 2, 2, i(math.MaxInt64), i(math.MaxInt64)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := NewFunc("main", tt.positions, tt.counts)
			if _, err := b.BatchedCausalMask(tt.q, tt.k, b.Arg(0), b.Arg(1)); err == nil {
				t.Fatal("accepted invalid mask")
			}
			if len(b.Ops) != 0 || b.nextID != 0 {
				t.Fatal("error changed builder")
			}
		})
	}
}
