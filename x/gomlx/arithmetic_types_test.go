package gomlx

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeTypedArithmetic(t *testing.T) {
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
		name                             string
		input, exponent, negative, power any
	}{
		{"int32", []int32{-3, 0, 2, -2147483648}, []int32{3, 0, 31, 1}, []int32{3, 0, -2, -2147483648}, []int32{-27, 1, -2147483648, -2147483648}},
		{"int64", []int64{-3, 9007199254740993, 2, math.MinInt64}, []int64{3, 1, 63, 1}, []int64{3, -9007199254740993, -2, math.MinInt64}, []int64{-27, 9007199254740993, math.MinInt64, math.MinInt64}},
		{"uint32", []uint32{0, 2, math.MaxUint32}, []uint32{0, 31, 2}, []uint32{0, math.MaxUint32 - 1, 1}, []uint32{1, 1 << 31, 1}},
		{"uint64", []uint64{0, 2, math.MaxUint64}, []uint64{0, 63, 2}, []uint64{0, math.MaxUint64 - 1, 1}, []uint64{1, 1 << 63, 1}},
		{"float64", []float64{-3, 9007199254740991, 2}, []float64{3, 1, -2}, []float64{3, -9007199254740991, -2}, []float64{-27, 9007199254740991, .25}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, err := graph.NewExec(b, func(x, y *graph.Node) []*graph.Node { return []*graph.Node{graph.Neg(x), graph.Pow(x, y)} })
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Call(tc.input, tc.exponent)
			if err != nil {
				t.Fatal(err)
			}
			for i, want := range []any{tc.negative, tc.power} {
				defer out[i].FinalizeAll()
				if got := out[i].Value(); !reflect.DeepEqual(got, want) {
					t.Fatalf("output %d: got %v, want %v", i, got, want)
				}
			}
		})
	}
}

func TestNativeFloat64Reduction(t *testing.T) {
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
	e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node { return []*graph.Node{graph.ReduceAllMax(x), graph.ReduceAllSum(x)} })
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	for _, tc := range []struct {
		name     string
		input    []float64
		max, sum float64
	}{
		{"negative", []float64{-1e8, -1e6, -1e16}, -1e6, -10000000101000000},
		{"precision", []float64{9007199254740991, 1}, 9007199254740991, 9007199254740992},
		{"infinity", []float64{math.Inf(-1), math.Inf(-1)}, math.Inf(-1), math.Inf(-1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := e.Call(tc.input)
			if err != nil {
				t.Fatal(err)
			}
			for i, want := range []float64{tc.max, tc.sum} {
				defer out[i].FinalizeAll()
				if got := out[i].Value().(float64); got != want {
					t.Fatalf("output %d: got %g, want %g", i, got, want)
				}
			}
		})
	}
}
