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

// This test-only view opens upstream conformance gates without changing the
// backend's public capability claim.
type dynamicConformanceBackend struct{ *Backend }

func (b dynamicConformanceBackend) Capabilities() compute.Capabilities {
	c := b.Backend.Capabilities().Clone()
	c.DynamicShapes = compute.DynamicShapesRecompiling
	c.DynamicAxes = true
	for _, op := range []compute.OpType{compute.OpTypeDynamicReshape, compute.OpTypeDynamicBroadcastInDim, compute.OpTypeDynamicIota, compute.OpTypeDynamicPad, compute.OpTypeDynamicDimensionSize, compute.OpTypeDynamicShape} {
		c.Operations[op] = true
	}
	return c
}

func TestNativeDynamicClosureGraph(t *testing.T) {
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
	e, err := graph.NewExec(dynamicConformanceBackend{b}, func(x, scalar *graph.Node) *graph.Node {
		builds++
		y := graph.Add(x, scalar)
		y = graph.Slice(y, graph.AxisRange(), graph.AxisRange(0, 2))
		y = graph.Concatenate([]*graph.Node{y, y}, 1)
		return graph.ReduceSum(y, 1)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	e.WithDynamicAxes([]string{"batch", ""}, nil)
	for _, batch := range []int{2, 4, 2} {
		data := make([][]float32, batch)
		want := make([]float32, batch)
		for i := range data {
			data[i] = []float32{float32(i*3 + 1), float32(i*3 + 2), float32(i*3 + 3)}
			want[i] = 2 * (data[i][0] + data[i][1] + 20)
		}
		out, err := e.Call(data, float32(10))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(out[0].Value(), want) {
			t.Fatalf("got %v want %v", out[0].Value(), want)
		}
		for _, v := range out {
			v.FinalizeAll()
		}
	}
	if builds != 1 {
		t.Fatalf("GoMLX graph rebuilt %d times, want 1", builds)
	}
}

func TestDynamicClosureValidation(t *testing.T) {
	f := (&Backend{}).Builder("invalid-closure").Main().(*function)
	a, err := f.Parameter("a", shapes.MakeDynamic(dtypes.Float32, []int{-1, 2}, []string{"batch", ""}), nil)
	if err != nil {
		t.Fatal(err)
	}
	other, err := f.Parameter("other", shapes.MakeDynamic(dtypes.Float32, []int{-1, 3}, []string{"items", ""}), nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, build := range []func() (compute.Value, error){
		func() (compute.Value, error) { return f.Concatenate(1, a, other) },
		func() (compute.Value, error) { return f.Slice(a, []int{1, 0}, []int{-1, 2}, nil) },
		func() (compute.Value, error) { return f.Slice(a, []int{-1, 0}, []int{2, 2}, nil) },
		func() (compute.Value, error) { return f.Add(a, other) },
	} {
		before := len(f.raw.Ops)
		if _, err := build(); err == nil {
			t.Fatal("invalid dynamic operation accepted")
		}
		if len(f.raw.Ops) != before {
			t.Fatal("invalid operation emitted IR")
		}
	}
	// Constants and scalar arithmetic remain available after rejected operations.
	scalar, err := f.Constant([]float32{2})
	if err != nil {
		t.Fatal(err)
	}
	result, err := f.Mul(a, scalar)
	if err != nil {
		t.Fatal(err)
	}
	shape, err := f.Shape(result)
	if err != nil {
		t.Fatal(err)
	}
	if shape.AxisName(0) != "batch" || shape.Dimensions[0] != shapes.DynamicDim {
		t.Fatalf("broadcast lost axis name: %v", shape)
	}
}

func TestNativeDynamicCapabilityBoundary(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	builder := b.Builder("dynamic-boundary")
	f := builder.Main()
	x, err := f.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{-1}, []string{"batch"}), nil)
	if err != nil {
		t.Fatal(err)
	}
	y, err := f.Erf(x)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{y}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Compile(); err == nil {
		t.Fatal("unrecorded dynamic Erf unexpectedly compiled; reassess capability boundary")
	}
	if b.Capabilities().DynamicShapes != compute.DynamicShapesNone {
		t.Fatal("partial replay advertised dynamic capability")
	}
}

func TestNativeDynamicStaticNamedSlice(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	builder := b.Builder("static-named-slice")
	f := builder.Main()
	x, err := f.Parameter("x", shapes.Make(dtypes.Int32, 6), nil)
	if err != nil {
		t.Fatal(err)
	}
	named, err := f.DynamicReshape(x, compute.DynamicDimensionSpec{Name: "rows"})
	if err != nil {
		t.Fatal(err)
	}
	sliced, err := f.Slice(named, []int{0}, []int{2}, nil)
	if err != nil {
		t.Fatal(err)
	}
	shape, err := f.Shape(sliced)
	if err != nil {
		t.Fatal(err)
	}
	if shape.AxisName(0) != "rows" || shape.Dimensions[0] != 2 {
		t.Fatalf("static slice metadata=%v", shape)
	}
	if _, err := f.DynamicIota(dtypes.Int32, 0, compute.DynamicDimensionSpec{Name: "rows"}); err == nil {
		t.Fatal("static name entered dynamic axis namespace")
	}
	if err := f.Return([]compute.Value{sliced}, nil); err != nil {
		t.Fatal(err)
	}
	e, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	input, err := b.BufferFromFlatData(0, []int32{1, 2, 3, 4, 5, 6}, shapes.Make(dtypes.Int32, 6))
	if err != nil {
		t.Fatal(err)
	}
	out, err := e.Execute([]compute.Buffer{input}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int32, 2)
	if err := out[0].ToFlatData(got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []int32{1, 2}) {
		t.Fatalf("slice=%v", got)
	}
}
