package gomlx

import (
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
)

func TestNativeGatherIndices(t *testing.T) {
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
	table := [][]int64{{9007199254740993, -2}, {3, 4}, {5, -9007199254740993}}
	for _, tc := range []struct {
		name    string
		indices any
		axis    int
	}{
		{"implicit", []int32{-1, 2, 99, 0}, 1},
		{"implicit i16", []int16{-1, 2, 99, 0}, 1},
		{"implicit ui16", []uint16{0, 2, 99, 0}, 1},
		{"leading vector", [][]int32{{-1, 2, 99, 0}}, 0},
		{"trailing vector", [][]int64{{-1}, {2}, {99}, {0}}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{table, tc.indices}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
				return f.Gather(v[0], v[1], tc.axis, []int{1}, []int{0}, []int{0}, []int{1, 2}, false)
			})
			if err != nil {
				t.Fatal(err)
			}
			want := [][]int64{table[0], table[2], table[2], table[0]}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("got%v want%v", got, want)
			}
		})
	}
	store := model.NewStore()
	defer store.Finalize()
	e, err := model.NewExec(b, store, func(scope *model.Scope, x *graph.Node) *graph.Node {
		scope = scope.WithInitializer(func(g *graph.Graph, s shapes.Shape) *graph.Node {
			return graph.Const(g, [][]float32{{1, 2}, {3, 4}, {5, 6}})
		})
		return layers.Embedding(scope, x, dtypes.Float32, 3, 2)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	out, err := e.Call([]int32{2, 0, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer out[0].FinalizeAll()
	if got := out[0].Value(); !reflect.DeepEqual(got, [][]float32{{5, 6}, {1, 2}, {5, 6}}) {
		t.Fatalf("embedding %v", got)
	}
}

func TestGatherValidation(t *testing.T) {
	f := (&Backend{}).Builder("gather").Main()
	x, err := f.Parameter("x", shapes.Make(dtypes.Float32, 3, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	idx, err := f.Parameter("idx", shapes.Make(dtypes.Int32, 2, 1), nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name                              string
		axis                              int
		offset, collapsed, mapping, sizes []int
	}{
		{"axis", 3, []int{1}, []int{0}, []int{0}, []int{1, 2}},
		{"size", 1, []int{1}, []int{0}, []int{0}, []int{1, 3}},
		{"duplicate mapping", 1, []int{1}, []int{0}, []int{0, 0}, []int{1, 2}},
		{"dynamic size", 1, []int{1}, []int{0}, []int{0}, []int{-1, 2}},
		{"bad collapsed size", 1, []int{1}, []int{0}, []int{0}, []int{2, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := f.(*function).raw
			before := len(raw.Ops)
			if _, err := f.Gather(x, idx, tc.axis, tc.offset, tc.collapsed, tc.mapping, tc.sizes, false); err == nil {
				t.Fatal("accepted invalid gather")
			}
			if len(raw.Ops) != before {
				t.Fatal("invalid gather emitted operation")
			}
		})
	}
}
