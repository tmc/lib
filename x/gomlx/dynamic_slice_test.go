package gomlx

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeDynamicSlice(t *testing.T) {
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
	x := [][]int64{{9007199254740993, 2, 3}, {4, 5, 9007199254740995}}
	for _, tc := range []struct {
		name   string
		starts []any
		want   [][]int64
	}{
		{"negative", []any{int64(math.MinInt64), int64(-2)}, [][]int64{{9007199254740993, 2}}},
		{"high", []any{int64(math.MaxInt64), int64(math.MaxInt64)}, [][]int64{{5, 9007199254740995}}},
		{"vector", []any{[]int32{1, 1}}, [][]int64{{5, 9007199254740995}}},
		{"int8", []any{int8(1), int8(1)}, [][]int64{{5, 9007199254740995}}},
		{"uint8", []any{uint8(255), uint8(255)}, [][]int64{{5, 9007199254740995}}},
		{"uint32", []any{uint32(math.MaxUint32), uint32(math.MaxUint32)}, [][]int64{{5, 9007199254740995}}},
		{"uint64", []any{uint64(math.MaxUint64), uint64(math.MaxUint64)}, [][]int64{{5, 9007199254740995}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			args := append([]any{x}, tc.starts...)
			got, err := testutil.Exec1(b, args, func(f compute.Function, v []compute.Value) (compute.Value, error) {
				return f.DynamicSlice(v[0], v[1:], []int{1, 2})
			})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
	got, err := testutil.Exec1(b, []any{true}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
		return f.DynamicSlice(v[0], nil, nil)
	})
	if err != nil || got != true {
		t.Fatalf("scalar: got %v err %v", got, err)
	}
}

func TestNativeDynamicSliceGradients(t *testing.T) {
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
	e, err := graph.NewExec(b, func(x, u, row, col *graph.Node) []*graph.Node {
		starts := []*graph.Node{row, col}
		cut := graph.DynamicSlice(x, starts, []int{2, 2})
		updated := graph.DynamicUpdateSlice(x, u, starts)
		grad := graph.Gradient(graph.ReduceAllSum(graph.Square(updated)), x, u)
		return []*graph.Node{cut, updated, graph.Gradient(graph.ReduceAllSum(graph.Square(cut)), x)[0], grad[0], grad[1]}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	x := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}
	u := [][]float32{{20, 21}, {22, 23}}
	for _, starts := range [][2]int64{{0, 0}, {1, 2}, {-99, 99}, {99, -99}} {
		row, col := int(starts[0]), int(starts[1])
		row = max(0, min(1, row))
		col = max(0, min(2, col))
		makeMatrix := func(rows, cols int) [][]float32 {
			v := make([][]float32, rows)
			for i := range v {
				v[i] = make([]float32, cols)
			}
			return v
		}
		cut, updated, dx, dxUpdate, du := makeMatrix(2, 2), makeMatrix(3, 4), makeMatrix(3, 4), makeMatrix(3, 4), makeMatrix(2, 2)
		for i := range x {
			copy(updated[i], x[i])
			for j, v := range x[i] {
				dxUpdate[i][j] = 2 * v
			}
		}
		for i := range u {
			for j, v := range u[i] {
				cut[i][j] = x[row+i][col+j]
				updated[row+i][col+j] = v
				dx[row+i][col+j] = 2 * cut[i][j]
				dxUpdate[row+i][col+j] = 0
				du[i][j] = 2 * v
			}
		}
		out, err := e.Call(x, u, starts[0], starts[1])
		if err != nil {
			t.Fatal(err)
		}
		want := []any{cut, updated, dx, dxUpdate, du}
		for i, v := range out {
			got := v.Value()
			v.FinalizeAll()
			if !reflect.DeepEqual(got, want[i]) {
				t.Fatalf("starts %v output %d: got %v want %v", starts, i, got, want[i])
			}
		}
	}
}

func TestDynamicSliceValidation(t *testing.T) {
	for _, name := range []string{"negative size", "oversized", "size count", "index count", "index rank", "float index", "bool index", "mixed indices", "short vector", "foreign"} {
		t.Run(name, func(t *testing.T) {
			f := (&Backend{}).Builder("slice").Main().(*function)
			param := func(fn compute.Function, n string, s shapes.Shape) compute.Value {
				v, err := fn.Parameter(n, s, nil)
				if err != nil {
					t.Fatal(err)
				}
				return v
			}
			x := param(f, "x", shapes.Make(dtypes.Float32, 2, 3))
			sizes := []int{1, 2}
			indexShape := shapes.Make(dtypes.Int64)
			switch name {
			case "negative size":
				sizes[0] = -1
			case "oversized":
				sizes[1] = 4
			case "size count":
				sizes = sizes[:1]
			case "index rank":
				indexShape = shapes.Make(dtypes.Int64, 1, 1)
			case "float index":
				indexShape = shapes.Make(dtypes.Float32)
			case "bool index":
				indexShape = shapes.Make(dtypes.Bool)
			}
			a := param(f, "a", indexShape)
			other := indexShape
			if name == "mixed indices" {
				other = shapes.Make(dtypes.Int32)
			}
			b := param(f, "b", other)
			starts := []compute.Value{a, b}
			if name == "index count" {
				starts = nil
			}
			if name == "short vector" {
				starts = []compute.Value{param(f, "short", shapes.Make(dtypes.Int64, 1))}
			}
			if name == "foreign" {
				starts[1] = param((&Backend{}).Builder("other").Main(), "z", shapes.Make(dtypes.Int64))
			}
			n := len(f.raw.Ops)
			_, err := f.DynamicSlice(x, starts, sizes)
			if err == nil || len(f.raw.Ops) != n {
				t.Fatal("invalid contract accepted or emitted IR")
			}
		})
	}
}

func TestNativeDynamicSliceEmpty(t *testing.T) {
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
	for _, cols := range []int{0, 3} {
		for _, dtype := range []dtypes.DType{dtypes.Bool, dtypes.Float16, dtypes.BFloat16, dtypes.Float32, dtypes.Float64, dtypes.Int8, dtypes.Int32, dtypes.Int64, dtypes.Uint8, dtypes.Uint32, dtypes.Uint64} {
			shape := shapes.Make(dtype, 2, cols)
			in, err := b.BufferFromFlatData(0, newFlat(dtype, 2*cols), shape)
			if err != nil {
				t.Fatal(err)
			}
			defer in.Finalize()
			builder := b.Builder("empty_slice")
			f := builder.Main()
			x, err := f.Parameter("x", shape, nil)
			if err != nil {
				t.Fatal(err)
			}
			start, err := f.Constant([]int64{99})
			if err != nil {
				t.Fatal(err)
			}
			y, err := f.DynamicSlice(x, []compute.Value{start, start}, []int{2, 0})
			if err != nil {
				t.Fatal(err)
			}
			if err := f.Return([]compute.Value{y}, nil); err != nil {
				t.Fatal(err)
			}
			e, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Execute([]compute.Buffer{in}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range out {
				got, err := v.Shape()
				if err != nil {
					t.Fatal(err)
				}
				if got.DType != dtype || !reflect.DeepEqual(got.Dimensions, []int{2, 0}) {
					t.Fatalf("input %v: got shape %v", shape, got)
				}
				if err := v.ToFlatData(newFlat(dtype, 0)); err != nil {
					t.Fatal(err)
				}
				if err := v.Finalize(); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}
