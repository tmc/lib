package gomlx

import (
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeInt8(t *testing.T) {
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
	values := make([]int8, 256)
	neg, square := make([]int8, 256), make([]int8, 256)
	wide := make([]int32, 256)
	negative := make([]bool, 256)
	for i := range values {
		values[i] = int8(i)
		neg[i] = -values[i]
		square[i] = values[i] * values[i]
		wide[i] = int32(values[i])
		negative[i] = values[i] < 0
	}
	e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
		g := x.Graph()
		return []*graph.Node{x, graph.Const(g, values), graph.Neg(x), graph.Pow(x, graph.Const(g, int8(2))), graph.ConvertDType(x, dtypes.Int32), graph.LessThan(x, graph.Const(g, int8(0))), graph.ReduceAllMax(x), graph.ReduceAllSum(x)}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	out, err := e.Call(values)
	if err != nil {
		t.Fatal(err)
	}
	wants := []any{values, values, neg, square, wide, negative, int8(127), int8(-128)}
	for i, want := range wants {
		defer out[i].FinalizeAll()
		if got := out[i].Value(); !reflect.DeepEqual(got, want) {
			t.Fatalf("output %d: got %v, want %v", i, got, want)
		}
	}
}

func TestNativeUint8(t *testing.T) {
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
	values, neg, square := make([]uint8, 256), make([]uint8, 256), make([]uint8, 256)
	wide := make([]int32, 256)
	lower := make([]bool, 256)
	for i := range values {
		values[i] = uint8(i)
		neg[i] = -values[i]
		square[i] = values[i] * values[i]
		wide[i] = int32(i)
		lower[i] = i < 128
	}
	e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
		g := x.Graph()
		return []*graph.Node{x, graph.Const(g, values), graph.Neg(x), graph.Pow(x, graph.Const(g, uint8(2))), graph.ConvertDType(x, dtypes.Int32), graph.LessThan(x, graph.Const(g, uint8(128))), graph.ReduceAllMax(x), graph.ReduceAllSum(x)}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	out, err := e.Call(values)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []any{values, values, neg, square, wide, lower, uint8(255), uint8(128)} {
		defer out[i].FinalizeAll()
		if got := out[i].Value(); !reflect.DeepEqual(got, want) {
			t.Fatalf("output %d: got %v, want %v", i, got, want)
		}
	}
}
