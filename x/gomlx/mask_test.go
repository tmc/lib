package gomlx

import (
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeWhereAndCompare(t *testing.T) {
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
	e, err := graph.NewExec(b, func(x, y, mask *graph.Node) []*graph.Node {
		less := graph.LessThan(x, y)
		equal := graph.Equal(x, y)
		enabled := graph.LogicalAnd(less, mask)
		return []*graph.Node{less, equal, enabled, graph.Where(enabled, x, y), graph.Identity(mask)}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	out, err := e.Call([]float32{-4, 2, 3, 8}, []float32{1, 2, 9, -1}, []bool{true, false, false, true})
	if err != nil {
		t.Fatalf("%+v", err)
	}
	want := []any{[]bool{true, false, true, false}, []bool{false, true, false, false}, []bool{true, false, false, false}, []float32{-4, 2, 9, -1}, []bool{true, false, false, true}}
	for i, tensor := range out {
		defer tensor.FinalizeAll()
		if got := tensor.Value(); !reflect.DeepEqual(got, want[i]) {
			t.Fatalf("output %d = %v, want %v", i, got, want[i])
		}
	}
}

func TestNativeWhereScalar(t *testing.T) {
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
	e, err := graph.NewExec(b, func(x, condition *graph.Node) []*graph.Node {
		g := x.Graph()
		return []*graph.Node{
			graph.Where(condition, x, graph.Const(g, int64(-7))),
			graph.Where(graph.Const(g, []bool{true, false, true}), graph.Const(g, int64(5)), graph.Const(g, int64(9))),
			graph.Equal(x, graph.Const(g, int64(9007199254740993))),
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	for _, condition := range []bool{true, false} {
		input := []int64{9007199254740993, -4, 0}
		out, err := e.Call(input, condition)
		if err != nil {
			t.Fatalf("%+v", err)
		}
		selected := input
		if !condition {
			selected = []int64{-7, -7, -7}
		}
		want := []any{selected, []int64{5, 9, 5}, []bool{true, false, false}}
		for i, tensor := range out {
			defer tensor.FinalizeAll()
			if got := tensor.Value(); !reflect.DeepEqual(got, want[i]) {
				t.Fatalf("condition %v output %d = %v, want %v", condition, i, got, want[i])
			}
		}
	}
}

func TestNativeMaskedSoftmax(t *testing.T) {
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
	e, err := graph.NewExec(b, func(x, mask *graph.Node) *graph.Node { return graph.MaskedSoftmax(x, mask) })
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	for _, hidden := range []float32{-10000, 10000} {
		out, err := e.Call([][]float32{{0, hidden, 0}, {hidden, 2, 3}}, [][]bool{{true, false, true}, {false, false, false}})
		if err != nil {
			t.Fatalf("%+v", err)
		}
		defer out[0].FinalizeAll()
		want := [][]float32{{0.5, 0, 0.5}, {0, 0, 0}}
		if got := out[0].Value(); !reflect.DeepEqual(got, want) {
			t.Fatalf("hidden %v: got %v, want %v", hidden, got, want)
		}
	}
}
