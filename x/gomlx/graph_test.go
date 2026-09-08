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

func TestNativeGoMLXGraph(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN to run native graph acceptance")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		b.Finalize()
		if b.closeErr != nil {
			t.Error(b.closeErr)
		}
	}()
	e, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
		weights := graph.Const(x.Graph(), [][]float32{{1, 2}, {3, 4}})
		bias := graph.Const(x.Graph(), []float32{10, 20})
		return graph.Add(graph.MatMul(x, weights), bias)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	for _, tc := range []struct{ input, want [][]float32 }{
		{[][]float32{{1, 2}, {3, 4}}, [][]float32{{17, 30}, {25, 42}}},
		{[][]float32{{2, 0}, {0, 2}}, [][]float32{{12, 24}, {16, 28}}},
	} {
		out, err := e.Call(tc.input)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != 1 {
			t.Fatalf("got %d outputs", len(out))
		}
		got := out[0].Value()
		out[0].FinalizeAll()
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("got %v, want %v", got, tc.want)
		}
		t.Logf("GoMLX graph result: %v", got)
	}
}

func TestNativeDonation(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		b.Finalize()
		if b.closeErr != nil {
			t.Error(b.closeErr)
		}
	}()
	builder := b.Builder("donate")
	f := builder.Main()
	shape := shapes.Make(dtypes.Float32, 2)
	p, err := f.Parameter("x", shape, nil)
	if err != nil {
		t.Fatal(err)
	}
	data := []float32{10, 20}
	c, err := f.Constant(data, 2)
	if err != nil {
		t.Fatal(err)
	}
	data[0] = 99 // Constant must own its data before compilation.
	sum, err := f.Add(p, c)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{sum}, nil); err != nil {
		t.Fatal(err)
	}
	e, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		e.Finalize()
		if e.(*executable).closeErr != nil {
			t.Error(e.(*executable).closeErr)
		}
	}()
	in, err := b.BufferFromFlatData(0, []float32{1, 2}, shape)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := in.Finalize(); err != nil {
			t.Error(err)
		}
	}()
	out, err := e.Execute([]compute.Buffer{in}, []bool{true}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		defer func() {
			if err := v.Finalize(); err != nil {
				t.Error(err)
			}
		}()
	}
	if _, err := in.Shape(); err == nil {
		t.Fatal("donated input still valid")
	}
	if _, err := e.Execute([]compute.Buffer{in}, nil, 0); err == nil {
		t.Fatal("accepted consumed input")
	}
	got := make([]float32, 2)
	if err := out[0].ToFlatData(got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []float32{11, 22}) {
		t.Fatalf("constant ownership: got %v", got)
	}
	t.Log("donated input invalidated; mutated constant source did not affect output")
}

func TestNativeGradient(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		b.Finalize()
		if b.closeErr != nil {
			t.Error(b.closeErr)
		}
	}()
	e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
		loss := graph.ReduceAllSum(graph.Mul(x, x))
		grad := graph.Gradient(loss, x)[0]
		return []*graph.Node{loss, grad}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	out, err := e.Call([][]float32{{-2, 0, 3}, {1, -4, 2}})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		defer v.FinalizeAll()
	}
	if len(out) != 2 {
		t.Fatalf("got %d outputs", len(out))
	}
	if got := out[0].Value(); !reflect.DeepEqual(got, float32(34)) {
		t.Fatalf("loss: %v", got)
	}
	want := [][]float32{{-4, 0, 6}, {2, -8, 4}}
	if got := out[1].Value(); !reflect.DeepEqual(got, want) {
		t.Fatalf("gradient: got %v, want %v", got, want)
	}
	t.Log("GoMLX reverse autodiff: loss 34, gradient [[-4 0 6] [2 -8 4]]")
}
