package gomlx

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeDynamicUnary(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	builder := b.Builder("unary")
	f := builder.Main().(*function)
	s := shapes.MakeDynamic(dtypes.Float32, []int{-1, 3}, []string{"batch", ""})
	x, err := f.Parameter("x", s, nil)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name  string
		build func(compute.Value) (compute.Value, error)
		want  func(float64) float64
	}{
		{"exp", f.Exp, math.Exp}, {"log", f.Log, math.Log}, {"sqrt", f.Sqrt, math.Sqrt}, {"rsqrt", f.Rsqrt, func(x float64) float64 { return 1 / math.Sqrt(x) }},
		{"sin", f.Sin, math.Sin}, {"cos", f.Cos, math.Cos}, {"abs", f.Abs, math.Abs}, {"expm1", f.Expm1, math.Expm1}, {"log1p", f.Log1p, math.Log1p},
		{"ceil", f.Ceil, math.Ceil}, {"floor", f.Floor, math.Floor}, {"round", f.Round, math.RoundToEven}, {"tanh", f.Tanh, math.Tanh},
		{"logistic", f.Logistic, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }}, {"neg", f.Neg, func(x float64) float64 { return -x }},
	}
	outputs := make([]compute.Value, len(tests))
	for i, tt := range tests {
		outputs[i], err = tt.build(x)
		if err != nil {
			t.Fatal(tt.name, err)
		}
		got, err := f.Shape(outputs[i])
		if err != nil || !reflect.DeepEqual(got, s) {
			t.Fatalf("%s shape=%v want=%v: %v", tt.name, got, s, err)
		}
	}
	if err = f.Return(outputs, nil); err != nil {
		t.Fatal(err)
	}
	compiled, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	defer compiled.Finalize()
	e := compiled.(*executable)
	for run, batch := range []int{2, 4, 2} {
		data := make([]float32, batch*3)
		for i := range data {
			data[i] = float32(i%3)*.5 + .25
		}
		input, err := b.BufferFromFlatData(0, data, shapes.Make(dtypes.Float32, batch, 3))
		if err != nil {
			t.Fatal(err)
		}
		out, err := e.Execute([]compute.Buffer{input}, nil, 0)
		input.Finalize()
		if err != nil {
			t.Fatal(err)
		}
		for i, tt := range tests {
			got := make([]float32, len(data))
			if err = out[i].ToFlatData(got); err != nil {
				t.Fatal(err)
			}
			out[i].Finalize()
			for j, v := range got {
				checkUnaryNumber(t, tt.name, float64(v), tt.want(float64(data[j])))
			}
		}
		if want := []int{1, 2, 2}[run]; e.compileCalls != want {
			t.Fatalf("compile calls=%d want=%d", e.compileCalls, want)
		}
	}
}

func checkUnaryNumber(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 1e-5 {
		t.Fatalf("%s got=%g want=%g", name, got, want)
	}
}

func TestNativeDynamicUnaryGradients(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	builds := 0
	e, err := graph.NewExec(dynamicConformanceBackend{b}, func(x, y *graph.Node) []*graph.Node {
		builds++
		exp := graph.Exp(x)
		log := graph.Log(x)
		div := graph.Gradient(graph.ReduceAllSum(graph.Div(x, y)), x, y)
		return []*graph.Node{exp, log, graph.Gradient(graph.ReduceAllSum(exp), x)[0], graph.Gradient(graph.ReduceAllSum(log), x)[0], div[0], div[1], graph.Log(exp)}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	e.WithDynamicAxes([]string{"batch", ""}, []string{"batch", ""})
	for _, batch := range []int{2, 4, 2} {
		x := make([][]float32, batch)
		y := make([][]float32, batch)
		for i := range x {
			x[i] = []float32{.25, .75, 1.25}
			y[i] = []float32{2, 3, 4}
		}
		out, err := e.Call(x, y)
		if err != nil {
			t.Fatal(err)
		}
		for k, v := range out {
			got := v.Value().([][]float32)
			for i := range x {
				for j := range x[i] {
					a, d := float64(x[i][j]), float64(y[i][j])
					want := []float64{math.Exp(a), math.Log(a), math.Exp(a), 1 / a, 1 / d, -a / (d * d), a}[k]
					checkUnaryNumber(t, "gradient", float64(got[i][j]), want)
				}
			}
			v.FinalizeAll()
		}
	}
	if builds != 1 {
		t.Fatalf("graph builds=%d want=1", builds)
	}
}

func TestDynamicUnaryValidation(t *testing.T) {
	f := (&Backend{}).Builder("invalid-unary").Main().(*function)
	for _, dtype := range []dtypes.DType{dtypes.Bool, dtypes.Int32} {
		x, err := f.Parameter(dtype.String(), shapes.Make(dtype, 2), nil)
		if err != nil {
			t.Fatal(err)
		}
		before := len(f.raw.Ops)
		if _, err = f.Exp(x); err == nil {
			t.Fatal("invalid exp accepted")
		}
		if dtype == dtypes.Bool {
			if _, err = f.Neg(x); err == nil {
				t.Fatal("bool neg accepted")
			}
		}
		if len(f.raw.Ops) != before {
			t.Fatal("invalid operation emitted IR")
		}
	}
	x, err := f.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	before := len(f.raw.Ops)
	if _, err = f.unary(x, "unknown"); err == nil {
		t.Fatal("unknown unary accepted")
	}
	if len(f.raw.Ops) != before {
		t.Fatal("unknown emitted IR")
	}
	if err = f.Return([]compute.Value{x}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err = f.Exp(x); err == nil {
		t.Fatal("post-return exp accepted")
	}
}

func TestNativeDynamicUnaryRuntimePrefix(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	builder := b.Builder("unary-prefix")
	f := builder.Main()
	must := func(v compute.Value, err error) compute.Value {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	x := must(f.Parameter("x", shapes.Make(dtypes.Float32), nil))
	n := must(f.ConvertDType(must(f.Exp(x)), dtypes.Int32))
	y := must(f.DynamicIota(dtypes.Int32, 0, compute.DynamicDimensionSpec{Name: "count", Value: n}))
	if err = f.Return([]compute.Value{y}, nil); err != nil {
		t.Fatal(err)
	}
	compiled, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	defer compiled.Finalize()
	e := compiled.(*executable)
	for run, v := range []float32{1, 1.5, 1} {
		input, err := b.BufferFromFlatData(0, []float32{v}, shapes.Make(dtypes.Float32))
		if err != nil {
			t.Fatal(err)
		}
		out, err := e.Execute([]compute.Buffer{input}, nil, 0)
		input.Finalize()
		if err != nil {
			t.Fatal(err)
		}
		size := []int{2, 4, 2}[run]
		shape, err := out[0].Shape()
		if err != nil || !reflect.DeepEqual(shape.Dimensions, []int{size}) {
			t.Fatalf("shape=%v want=%d: %v", shape, size, err)
		}
		got := make([]int32, size)
		if err = out[0].ToFlatData(got); err != nil {
			t.Fatal(err)
		}
		out[0].Finalize()
		for i, v := range got {
			if v != int32(i) {
				t.Fatalf("iota=%v", got)
			}
		}
		if e.compileCalls != []int{1, 2, 2}[run] {
			t.Fatalf("compile calls=%d", e.compileCalls)
		}
	}
}

func TestDynamicUnaryClosedBuilder(t *testing.T) {
	f := (&Backend{}).Builder("closed-unary").Main().(*function)
	x, err := f.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	f.owner.closed = true
	before := len(f.raw.Ops)
	if _, err = f.Exp(x); err == nil {
		t.Fatal("closed builder accepted exp")
	}
	if len(f.raw.Ops) != before || len(f.dynamicOps) != 0 {
		t.Fatal("closed builder mutated")
	}
}

func TestDynamicUnaryNamedIota(t *testing.T) {
	f := (&Backend{}).Builder("named-unary").Main().(*function)
	x, err := f.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{-1, 3}, []string{"batch", ""}), nil)
	if err != nil {
		t.Fatal(err)
	}
	y, err := f.Exp(x)
	if err != nil {
		t.Fatal(err)
	}
	shape, err := f.Shape(y)
	if err != nil {
		t.Fatal(err)
	}
	name := ""
	if len(shape.AxisNames) > 0 {
		name = shape.AxisNames[0]
	}
	if name == "" {
		t.Fatal("name-only iota cannot select the unary batch axis")
	}
	if _, err = f.DynamicIota(dtypes.Int32, 0, compute.DynamicDimensionSpec{Name: name}); err != nil {
		t.Fatalf("name-only iota from unary shape: %v", err)
	}
}
