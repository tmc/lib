package builder

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

func TestInvalidShapesDoNotEmit(t *testing.T) {
	matrix := Type{Shape: []int64{2, 3}, ElementType: "f32"}
	arg := Value{ID: "%arg0", Type: matrix}
	scalar := Value{ID: "%zero", Type: Type{ElementType: "f32"}}
	for _, tt := range []struct {
		name  string
		build func(*FuncBuilder) (Value, error)
	}{
		{"dot out of range", func(b *FuncBuilder) (Value, error) {
			return b.DotGeneral(arg, arg, nil, nil, []int64{2}, []int64{0}, matrix)
		}},
		{"dot overlap", func(b *FuncBuilder) (Value, error) {
			return b.DotGeneral(arg, arg, []int64{0}, []int64{0}, []int64{0}, []int64{0}, matrix)
		}},
		{"dot length", func(b *FuncBuilder) (Value, error) { return b.DotGeneral(arg, arg, nil, nil, []int64{1}, nil, matrix) }},
		{"dot contraction", func(b *FuncBuilder) (Value, error) {
			return b.DotGeneral(arg, arg, nil, nil, []int64{1}, []int64{0}, matrix)
		}},
		{"dot result", func(b *FuncBuilder) (Value, error) {
			return b.DotGeneral(arg, arg, nil, nil, []int64{1}, []int64{1}, matrix)
		}},
		{"reshape count", func(b *FuncBuilder) (Value, error) {
			return b.Reshape(arg, Type{Shape: []int64{5}, ElementType: "f32"})
		}},
		{"reshape overflow", func(b *FuncBuilder) (Value, error) {
			return b.Reshape(arg, Type{Shape: []int64{math.MaxInt64, 2}, ElementType: "f32"})
		}},
		{"transpose duplicate", func(b *FuncBuilder) (Value, error) { return b.Transpose(arg, []int64{0, 0}, matrix) }},
		{"transpose result", func(b *FuncBuilder) (Value, error) { return b.Transpose(arg, []int64{1, 0}, matrix) }},
		{"slice zero stride", func(b *FuncBuilder) (Value, error) {
			return b.Slice(arg, []int64{0, 0}, []int64{2, 3}, []int64{1, 0}, matrix)
		}},
		{"slice bounds", func(b *FuncBuilder) (Value, error) {
			return b.Slice(arg, []int64{0, 0}, []int64{3, 3}, []int64{1, 1}, matrix)
		}},
		{"concat empty", func(b *FuncBuilder) (Value, error) { return b.Concatenate(nil, 0, matrix) }},
		{"concat result", func(b *FuncBuilder) (Value, error) { return b.Concatenate([]Value{arg, arg}, 0, matrix) }},
		{"broadcast duplicate", func(b *FuncBuilder) (Value, error) { return b.BroadcastInDim(arg, []int64{0, 0}, matrix) }},
		{"reduce initial", func(b *FuncBuilder) (Value, error) { return b.Reduce(arg, arg, []int64{1}, matrix, "add") }},
		{"reduce result", func(b *FuncBuilder) (Value, error) { return b.Reduce(arg, scalar, []int64{1}, matrix, "add") }},
		{"rmsnorm axis", func(b *FuncBuilder) (Value, error) { return b.RMSNorm(arg, scalar, 9, 1e-5) }},
		{"rmsnorm gamma", func(b *FuncBuilder) (Value, error) { return b.RMSNorm(arg, scalar, 1, 1e-5) }},
		{"constant integer", func(b *FuncBuilder) (Value, error) { return b.Constant(1, Type{ElementType: "i32"}) }},
		{"constant dynamic", func(b *FuncBuilder) (Value, error) {
			return b.Constant(1, Type{Shape: []int64{-1}, ElementType: "f32"})
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := NewFunc("main", matrix)
			if _, err := tt.build(b); err == nil {
				t.Fatal("invalid operation accepted")
			}
			if len(b.Ops) != 0 || b.nextID != 0 {
				t.Fatalf("failed operation changed builder: %+v", b)
			}
		})
	}
}

func TestDotShape(t *testing.T) {
	for _, tt := range []struct {
		name           string
		lhs, rhs       []int64
		lb, rb, lc, rc []int64
		want           []int64
	}{
		{"matmul", []int64{2, 3}, []int64{3, 4}, nil, nil, []int64{1}, []int64{0}, []int64{2, 4}},
		{"batched", []int64{5, 2, 3}, []int64{5, 3, 4}, []int64{0}, []int64{0}, []int64{2}, []int64{1}, []int64{5, 2, 4}},
		{"dynamic", []int64{-1, 2, 3}, []int64{5, 3, 4}, []int64{0}, []int64{0}, []int64{2}, []int64{1}, []int64{5, 2, 4}},
		{"scalar", nil, nil, nil, nil, nil, nil, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := dotShape(Type{Shape: tt.lhs}, Type{Shape: tt.rhs}, tt.lb, tt.rb, tt.lc, tt.rc)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("got %v want %v", got, tt.want)
			}
			for i, n := range got {
				if n != tt.want[i] {
					t.Fatalf("got %v want %v", got, tt.want)
				}
			}
		})
	}
}

func TestConstantPreservesFloat32(t *testing.T) {
	for _, value := range []float32{0, 1, 1e-8, math.SmallestNonzeroFloat32, math.MaxFloat32, -123.4567} {
		b := NewFunc("main")
		if _, err := b.Constant(value, Type{ElementType: "f32"}); err != nil {
			t.Fatal(err)
		}
		literal := strings.Split(strings.Split(b.Ops[0], "dense<")[1], ">")[0]
		got, err := strconv.ParseFloat(literal, 32)
		if err != nil || math.Float32bits(float32(got)) != math.Float32bits(value) {
			t.Fatalf("%g encoded as %q (decoded %g, %v)", value, literal, got, err)
		}
	}
}
