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

func TestNativeArgMinMax(t *testing.T) {
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
		axis  int
		dtype dtypes.DType
		min   bool
		want  any
	}{
		{"max tie", []float32{3, 5, 5, 2}, 0, dtypes.Int32, false, int32(1)},
		{"min tie", []float64{3, -5, -5, 2}, 0, dtypes.Int64, true, int64(1)},
		{"nan max", []float64{3, math.NaN(), math.NaN(), 8}, 0, dtypes.Uint32, false, uint32(1)},
		{"nan min", []float64{3, math.NaN(), math.NaN(), -8}, 0, dtypes.Uint8, true, uint8(1)},
		{"signed exact", []int64{1 << 53, 1<<53 + 1, 1 << 53}, 0, dtypes.Int8, false, int8(1)},
		{"unsigned exact", []uint64{1<<63 + 2, 1<<63 + 1, 1<<63 + 1}, 0, dtypes.Uint64, true, uint64(1)},
		{"columns", [][]int32{{2, 0, 7}, {-3, 4, 2}}, 0, dtypes.Int32, false, []int32{0, 1, 0}},
		{"rows", [][]int32{{2, 0, 7}, {-3, 4, 2}}, 1, dtypes.Int64, true, []int64{1, 0}},
		{"all inf", []float32{float32(math.Inf(1)), float32(math.Inf(1))}, 0, dtypes.Int32, true, int32(0)},
		{"all min integer", []int64{math.MinInt64, math.MinInt64}, 0, dtypes.Int64, false, int64(0)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{tc.input}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
				return f.ArgMinMax(v[0], tc.axis, tc.dtype, tc.min)
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

func TestArgMinMaxValidation(t *testing.T) {
	for _, tc := range []struct {
		shape shapes.Shape
		axis  int
		dtype dtypes.DType
	}{
		{shapes.Make(dtypes.Bool, 3), 0, dtypes.Int32},
		{shapes.Make(dtypes.Float32), 0, dtypes.Int32},
		{shapes.Make(dtypes.Float32, 3), -1, dtypes.Int32},
		{shapes.Make(dtypes.Float32, 3), 1, dtypes.Int32},
		{shapes.Make(dtypes.Float32, 3), 0, dtypes.Float32},
	} {
		f := (&Backend{}).Builder("argminmax").Main()
		x, err := f.Parameter("x", tc.shape, nil)
		if err != nil {
			t.Fatal(err)
		}
		raw := f.(*function).raw
		n := len(raw.Ops)
		if _, err := f.ArgMinMax(x, tc.axis, tc.dtype, false); err == nil {
			t.Fatal("accepted invalid argument")
		}
		if len(raw.Ops) != n {
			t.Fatal("invalid argminmax emitted IR")
		}
	}
}

func TestNativeArgMinMaxGraph(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
		return []*graph.Node{graph.ArgMax(x, -1, dtypes.Int8), graph.ArgMax(x, -1, dtypes.Int64)}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	input := make([]float32, 260)
	input[130] = 10
	input[258] = 10
	out, err := e.Call(input)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		defer v.FinalizeAll()
	}
	if got := out[0].Value(); got != int8(-126) {
		t.Fatalf("narrow index %v, want -126 after conversion", got)
	}
	if got := out[1].Value(); got != int64(130) {
		t.Fatalf("wide index %v, want first tie130", got)
	}
}

func TestNativeArgMinMaxEmptyAxis(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	for _, min := range []bool{false, true} {
		builder := b.Builder("emptyarg")
		f := builder.Main()
		shape := shapes.Make(dtypes.Float32, 2, 0)
		x, err := f.Parameter("x", shape, nil)
		if err != nil {
			t.Fatal(err)
		}
		y, err := f.ArgMinMax(x, 1, dtypes.Int32, min)
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
		in, err := b.BufferFromFlatData(0, []float32{}, shape)
		if err != nil {
			t.Fatal(err)
		}
		defer in.Finalize()
		out, err := e.Execute([]compute.Buffer{in}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range out {
			defer v.Finalize()
			got := make([]int32, 2)
			if err := v.ToFlatData(got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, []int32{0, 0}) {
				t.Fatalf("empty reduction %v", got)
			}
		}
	}
}
