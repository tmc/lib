package gomlx

import (
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeIntegerGraph(t *testing.T) {
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
		name                   string
		input, bias, want, sum any
	}{
		{"int32", []int32{2147483647, -2147483648}, int32(1), []int32{-2147483648, -2147483647}, int32(1)},
		{"int64", []int64{9007199254740993, -9007199254740992}, int64(2), []int64{9007199254740995, -9007199254740990}, int64(5)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
				bias := graph.Const(x.Graph(), tc.bias)
				added := graph.Add(x, bias)
				reshaped := graph.Reshape(added, 1, 2)
				return []*graph.Node{graph.Reshape(reshaped, 2), graph.ReduceAllSum(reshaped)}
			})
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Call(tc.input)
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range out {
				defer v.FinalizeAll()
			}
			if len(out) != 2 {
				t.Fatalf("got %d outputs", len(out))
			}
			if got := out[0].Value(); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("integer values: got %v, want %v", got, tc.want)
			}
			if got := out[1].Value(); !reflect.DeepEqual(got, tc.sum) {
				t.Fatalf("integer sum: got %v, want %v", got, tc.sum)
			}
			t.Logf("exact typed result %v, reduced scalar %v", tc.want, tc.sum)
		})
	}
}

func TestNativeIntegerProducts(t *testing.T) {
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
		name                           string
		input, weight, product, square any
	}{
		{"int32", [][]int32{{1, 2}, {3, 4}}, [][]int32{{2, 0}, {0, 2}}, [][]int32{{2, 4}, {6, 8}}, [][]int32{{1, 4}, {9, 16}}},
		{"int64", [][]int64{{1, 2}, {3, 4}}, [][]int64{{2, 0}, {0, 2}}, [][]int64{{2, 4}, {6, 8}}, [][]int64{{1, 4}, {9, 16}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
				return []*graph.Node{graph.MatMul(x, graph.Const(x.Graph(), tc.weight)), graph.Mul(x, x)}
			})
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Call(tc.input)
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range out {
				defer v.FinalizeAll()
			}
			if !reflect.DeepEqual(out[0].Value(), tc.product) || !reflect.DeepEqual(out[1].Value(), tc.square) {
				t.Fatal("integer product mismatch")
			}
		})
	}
}
