package gomlx

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/gomlx/core/graph"
	"os"
	"reflect"
	"testing"
)

func TestNativePad(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	for _, tc := range []struct {
		name              string
		input, fill, want any
		config            []compute.PadAxis
	}{
		{"interior", []int64{9007199254740993, 2}, int64(-1), []int64{-1, 9007199254740993, -1, 2, -1}, []compute.PadAxis{{Start: 1, End: 1, Interior: 1}}},
		{"crop after dilation", []int32{1, 2, 3}, int32(9), []int32{9, 2, 9}, []compute.PadAxis{{Start: -1, End: -1, Interior: 1}}},
		{"missing axes", [][]uint64{{1, 2}, {3, 4}}, uint64(9223372036854775809), [][]uint64{{9223372036854775809, 9223372036854775809}, {1, 2}, {3, 4}}, []compute.PadAxis{{Start: 1}}},
		{"bool", []bool{true}, false, []bool{false, true, false}, []compute.PadAxis{{Start: 1, End: 1}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{tc.input, tc.fill}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
				return f.Pad(v[0], v[1], tc.config...)
			})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
}

func TestNativeSumMeanPoolGradient(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	e := graph.MustNewExec(b, func(x *graph.Node) []*graph.Node {
		sum := graph.SumPool(x).Window(2).Strides(1).Done()
		mean := graph.MeanPool(x).Window(2).Strides(1).Done()
		return []*graph.Node{graph.Gradient(graph.ReduceAllSum(graph.Square(sum)), x)[0], graph.Gradient(graph.ReduceAllSum(graph.Square(mean)), x)[0]}
	})
	defer e.Finalize()
	out, err := e.Call([][][]float32{{{1}, {2}, {3}}})
	if err != nil {
		t.Fatal(err)
	}
	want := []any{[][][]float32{{{6}, {16}, {10}}}, [][][]float32{{{1.5}, {4}, {2.5}}}}
	for i, v := range out {
		defer v.FinalizeAll()
		if !reflect.DeepEqual(v.Value(), want[i]) {
			t.Fatalf("gradient %d got %v want %v", i, v.Value(), want[i])
		}
	}
}

func TestNativeMaxMinPoolGradient(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	e := graph.MustNewExec(b, func(x *graph.Node) []*graph.Node {
		max := graph.MaxPool(x).Window(2).Strides(1).Done()
		min := graph.MinPool(x).Window(2).Strides(1).Done()
		return []*graph.Node{graph.Gradient(graph.ReduceAllSum(graph.Square(max)), x)[0], graph.Gradient(graph.ReduceAllSum(graph.Square(min)), x)[0]}
	})
	defer e.Finalize()
	for _, tc := range []struct {
		name            string
		input, max, min []float32
	}{
		{"overlap", []float32{1, 3, 2}, []float32{0, 12, 0}, []float32{2, 0, 4}},
		{"ties", []float32{2, 2, 2}, []float32{4, 4, 0}, []float32{4, 4, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := make([][]float32, len(tc.input))
			for i, v := range tc.input {
				x[i] = []float32{v}
			}
			out, err := e.Call([][][]float32{x})
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range out {
				defer v.FinalizeAll()
				flat := tc.max
				if i == 1 {
					flat = tc.min
				}
				want := make([][]float32, len(flat))
				for j, n := range flat {
					want[j] = []float32{n}
				}
				if !reflect.DeepEqual(v.Value(), [][][]float32{want}) {
					t.Fatalf("gradient %d got %v want %v", i, v.Value(), want)
				}
			}
		})
	}
}

func TestPadAndSelectionValidation(t *testing.T) {
	for _, name := range []string{"fill rank", "fill dtype", "interior", "axis count", "selection shape", "selection dtype", "selection stride"} {
		t.Run(name, func(t *testing.T) {
			f := (&Backend{}).Builder("padding").Main()
			param := func(n string, s shapes.Shape) compute.Value {
				v, err := f.Parameter(n, s, nil)
				if err != nil {
					t.Fatal(err)
				}
				return v
			}
			x := param("x", shapes.Make(dtypes.Float32, 3))
			fillShape := shapes.Make(dtypes.Float32)
			if name == "fill rank" {
				fillShape = shapes.Make(dtypes.Float32, 1)
			}
			if name == "fill dtype" {
				fillShape = shapes.Make(dtypes.Int32)
			}
			fill := param("fill", fillShape)
			sourceShape := shapes.Make(dtypes.Float32, 2)
			if name == "selection shape" {
				sourceShape = shapes.Make(dtypes.Float32, 3)
			}
			if name == "selection dtype" {
				sourceShape = shapes.Make(dtypes.Int32, 2)
			}
			source := param("source", sourceShape)
			config := []compute.PadAxis{{Start: 1}}
			if name == "interior" {
				config[0].Interior = -1
			}
			if name == "axis count" {
				config = append(config, compute.PadAxis{})
			}
			raw := f.(*function).raw
			n := len(raw.Ops)
			var err error
			switch name {
			case "selection shape", "selection dtype", "selection stride":
				stride := 1
				if name == "selection stride" {
					stride = 0
				}
				_, err = f.SelectAndScatterMax(x, source, []int{2}, []int{stride}, nil)
			default:
				_, err = f.Pad(x, fill, config...)
			}
			if err == nil {
				t.Fatal("accepted invalid parameters")
			}
			if len(raw.Ops) != n {
				t.Fatal("invalid parameters emitted IR")
			}
		})
	}
}
