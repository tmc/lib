package gomlx

import (
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeArithmeticExtra(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	for _, tc := range []struct {
		name  string
		input any
		build func(*graph.Node) *graph.Node
		want  any
	}{
		{"logical", []bool{true, false}, graph.LogicalNot, []bool{false, true}},
		{"signed bits", []int64{0, -1, 1<<53 + 1}, graph.BitwiseNot, []int64{-1, 0, ^(int64(1<<53 + 1))}},
		{"unsigned bits", []uint64{0, 1<<63 + 1, ^uint64(0)}, graph.BitwiseNot, []uint64{^uint64(0), ^(uint64(1<<63 + 1)), 0}},
		{"rsqrt", []float64{1, 4, 16}, graph.Rsqrt, []float64{1, .5, .25}},
		{"product rows", [][]int64{{1<<53 + 1, 1}, {-2, 3}}, func(x *graph.Node) *graph.Node { return graph.ReduceMultiply(x, 1) }, []int64{1<<53 + 1, -6}},
		{"product all", [][]float32{{2, 3}, {4, 5}}, func(x *graph.Node) *graph.Node { return graph.ReduceMultiply(x) }, float32(120)},
		{"unsigned product", []uint64{1<<63 + 1, 1}, func(x *graph.Node) *graph.Node { return graph.ReduceMultiply(x) }, uint64(1<<63 + 1)},
		{"overflow product", []uint8{100, 3}, func(x *graph.Node) *graph.Node { return graph.ReduceMultiply(x) }, uint8(44)},
		{"scalar product", float32(7), func(x *graph.Node) *graph.Node { return graph.ReduceMultiply(x) }, float32(7)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, err := graph.NewExec(b, tc.build)
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Call(tc.input)
			if err != nil {
				t.Fatalf("%+v", err)
			}
			defer out[0].FinalizeAll()
			if got := out[0].Value(); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestArithmeticExtraValidation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dtype dtypes.DType
		call  func(compute.Function, compute.Value) (compute.Value, error)
	}{
		{"logical integer", dtypes.Int32, func(f compute.Function, v compute.Value) (compute.Value, error) { return f.LogicalNot(v) }},
		{"bitwise float", dtypes.Float32, func(f compute.Function, v compute.Value) (compute.Value, error) { return f.BitwiseNot(v) }},
		{"rsqrt integer", dtypes.Int32, func(f compute.Function, v compute.Value) (compute.Value, error) { return f.Rsqrt(v) }},
		{"product bool", dtypes.Bool, func(f compute.Function, v compute.Value) (compute.Value, error) { return f.ReduceProduct(v) }},
		{"product duplicate axis", dtypes.Float32, func(f compute.Function, v compute.Value) (compute.Value, error) { return f.ReduceProduct(v, 0, 0) }},
		{"product invalid axis", dtypes.Float32, func(f compute.Function, v compute.Value) (compute.Value, error) { return f.ReduceProduct(v, 2) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := (&Backend{}).Builder("invalid").Main()
			v, err := f.Parameter("x", shapes.Make(tc.dtype, 2, 3), nil)
			if err != nil {
				t.Fatal(err)
			}
			raw := f.(*function).raw
			n := len(raw.Ops)
			if _, err := tc.call(f, v); err == nil {
				t.Fatal("accepted invalid operand")
			}
			if len(raw.Ops) != n {
				t.Fatal("invalid operation emitted IR")
			}
		})
	}
}
