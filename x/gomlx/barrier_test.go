package gomlx

import (
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/backendtest"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/gomlx/core/graph"
)

func TestBarrierIR(t *testing.T) {
	f := (&Backend{}).Builder("barrier").Main().(*function)
	x := barrierParameter(t, f, "x", shapes.Make(dtypes.Float32))
	y := barrierParameter(t, f, "y", shapes.Make(dtypes.Int32, 2))
	z := barrierParameter(t, f, "z", shapes.Make(dtypes.Bool))
	outs, err := f.OptimizationBarrier(x, y)
	if err != nil {
		t.Fatal(err)
	}
	ready, err := f.SchedulingBarrier(outs[0], y, z)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{ready, outs[1]}, nil); err != nil {
		t.Fatal(err)
	}
	if len(f.raw.Ops) != 2 {
		t.Fatalf("got %d operations, want 2", len(f.raw.Ops))
	}
	for i, want := range []string{`"stablehlo.optimization_barrier"(%arg0, %arg1)`, `"stablehlo.optimization_barrier"(%barrier0_0, %arg1, %arg2)`} {
		if !strings.Contains(f.raw.Ops[i], want) {
			t.Fatalf("operation %d = %q, want dependency structure %q", i, f.raw.Ops[i], want)
		}
	}
}

func TestBarrierValidation(t *testing.T) {
	f := (&Backend{}).Builder("barrier").Main().(*function)
	x := barrierParameter(t, f, "x", shapes.Make(dtypes.Float32))
	foreignFunction := (&Backend{}).Builder("foreign").Main().(*function)
	foreign := barrierParameter(t, foreignFunction, "y", shapes.Make(dtypes.Float32))
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"empty optimization barrier", func() error { _, err := f.OptimizationBarrier(); return err }},
		{"nil operand", func() error { _, err := f.OptimizationBarrier(nil); return err }},
		{"foreign operand", func() error { _, err := f.OptimizationBarrier(x, foreign); return err }},
		{"foreign dependency", func() error { _, err := f.SchedulingBarrier(x, foreign); return err }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := len(f.raw.Ops)
			if err := tc.call(); err == nil {
				t.Fatal("accepted invalid barrier")
			}
			if len(f.raw.Ops) != before {
				t.Fatal("invalid barrier emitted IR")
			}
		})
	}
	if err := f.Return([]compute.Value{x}, nil); err != nil {
		t.Fatal(err)
	}
	before := len(f.raw.Ops)
	if _, err := f.OptimizationBarrier(x); err == nil {
		t.Fatal("accepted optimization barrier after return")
	}
	if _, err := f.SchedulingBarrier(x); err == nil {
		t.Fatal("accepted scheduling barrier after return")
	}
	if len(f.raw.Ops) != before {
		t.Fatal("barrier after return emitted IR")
	}
}

func TestNativeBarriers(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	backendtest.TestBarriers(t, b)

	negativeZero := math.Float64frombits(1 << 63)
	inputs := []any{[]float64{math.Inf(1), math.Inf(-1), math.NaN(), negativeZero}, []int64{math.MinInt64, -1, 0, math.MaxInt64}, []bool{true, false}}
	got, err := testutil.Exec1(b, inputs, func(f compute.Function, values []compute.Value) (compute.Value, error) {
		return f.SchedulingBarrier(values[0], values[1], values[2])
	})
	if err != nil {
		t.Fatal(err)
	}
	want := inputs[0].([]float64)
	values := got.([]float64)
	for i := range want {
		if math.Float64bits(values[i]) != math.Float64bits(want[i]) {
			t.Fatalf("value %d bits = %016x, want %016x", i, math.Float64bits(values[i]), math.Float64bits(want[i]))
		}
	}
}

func TestNativeGradientCheckpointBarrier(t *testing.T) {
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
		y := graph.Square(x.Checkpoint()).StopCheckpoint()
		loss := graph.ReduceAllSum(y)
		return []*graph.Node{y, graph.Gradient(loss, x)[0]}
	})
	defer e.Finalize()
	got, err := e.Call([]float64{1, -2, 3})
	if err != nil {
		t.Fatal(err)
	}
	want := []any{[]float64{1, 4, 9}, []float64{2, -4, 6}}
	for i, value := range got {
		defer value.FinalizeAll()
		if !reflect.DeepEqual(value.Value(), want[i]) {
			t.Fatalf("output %d = %v, want %v", i, value.Value(), want[i])
		}
	}
}

func TestSchedulingBarrierNoDependencies(t *testing.T) {
	f := (&Backend{}).Builder("barrier").Main().(*function)
	x := barrierParameter(t, f, "x", shapes.Make(dtypes.Float32))
	got, err := f.SchedulingBarrier(x)
	if err != nil {
		t.Fatal(err)
	}
	if got != x || len(f.raw.Ops) != 0 {
		t.Fatal("no-dependency barrier changed value or emitted IR")
	}
}

func barrierParameter(t *testing.T, f *function, name string, shape shapes.Shape) compute.Value {
	t.Helper()
	value, err := f.Parameter(name, shape, nil)
	if err != nil {
		t.Fatal(err)
	}
	return value
}
