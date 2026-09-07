package builder

import (
	"math"
	"testing"
)

func TestDynamicUpdateValidation(t *testing.T) {
	for _, tt := range []struct {
		name            string
		operand, update Type
		starts          []Type
	}{
		{"too large", Type{Shape: []int64{2}, ElementType: "f32"}, Type{Shape: []int64{3}, ElementType: "f32"}, []Type{{ElementType: "i64"}}},
		{"dynamic", Type{Shape: []int64{-1}, ElementType: "f32"}, Type{Shape: []int64{1}, ElementType: "f32"}, []Type{{ElementType: "i64"}}},
		{"index vector", Type{Shape: []int64{2}, ElementType: "f32"}, Type{Shape: []int64{1}, ElementType: "f32"}, []Type{{Shape: []int64{1}, ElementType: "i64"}}},
		{"missing index", Type{Shape: []int64{2}, ElementType: "f32"}, Type{Shape: []int64{1}, ElementType: "f32"}, nil},
		{"mixed indices", Type{Shape: []int64{2, 2}, ElementType: "f32"}, Type{Shape: []int64{1, 1}, ElementType: "f32"}, []Type{{ElementType: "i32"}, {ElementType: "i64"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			types := append([]Type{tt.operand, tt.update}, tt.starts...)
			b := NewFunc("main", types...)
			if _, err := b.DynamicUpdateSlice(b.Arg(0), b.Arg(1), b.Args[2:]); err == nil {
				t.Fatal("accepted invalid update")
			}
			if len(b.Ops) != 0 || b.nextID != 0 {
				t.Fatal("invalid update changed builder")
			}
		})
	}
}

func TestIntegerConstantValidation(t *testing.T) {
	for _, tt := range []struct {
		value int64
		typ   Type
	}{
		{math.MaxInt32 + 1, Type{ElementType: "i32"}}, {math.MinInt32 - 1, Type{ElementType: "i32"}}, {0, Type{ElementType: "f32"}}, {0, Type{Shape: []int64{-1}, ElementType: "i64"}},
	} {
		b := NewFunc("main")
		if _, err := b.IntegerConstant(tt.value, tt.typ); err == nil {
			t.Fatalf("accepted %v", tt)
		}
		if len(b.Ops) != 0 || b.nextID != 0 {
			t.Fatal("invalid constant changed builder")
		}
	}
}
