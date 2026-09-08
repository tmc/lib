package gomlx

import (
	"os"
	"reflect"
	"sort"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

func TestNativeSortFunc(t *testing.T) {
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
	keys := [][]float32{{2, 1, 2, 0}, {1, 3, 1, 3}}
	ids := [][]int64{{9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996}, {9007199254740997, 9007199254740998, 9007199254740999, 9007199254741000}}
	for _, axis := range []int{0, 1, -1} {
		e, err := graph.NewExec(b, func(k, id, reverse *graph.Node) []*graph.Node {
			c := graph.NewClosure(k.Graph(), func(g *graph.Graph) []*graph.Node {
				lhs := graph.Parameter(g, "lhs", shapes.Make(dtypes.Float32))
				rhs := graph.Parameter(g, "rhs", shapes.Make(dtypes.Float32))
				graph.Parameter(g, "lhsID", shapes.Make(dtypes.Int64))
				graph.Parameter(g, "rhsID", shapes.Make(dtypes.Int64))
				return []*graph.Node{graph.Where(reverse, graph.GreaterThan(lhs, rhs), graph.LessThan(lhs, rhs))}
			})
			return graph.SortFunc(c, axis, true, k, id)
		})
		if err != nil {
			t.Fatal(err)
		}
		for _, reverse := range []bool{false, true} {
			wantKeys := [][]float32{append([]float32(nil), keys[0]...), append([]float32(nil), keys[1]...)}
			wantIDs := [][]int64{append([]int64(nil), ids[0]...), append([]int64(nil), ids[1]...)}
			lines, n := 2, 4
			if axis == 0 {
				lines, n = 4, 2
			}
			for line := 0; line < lines; line++ {
				order := make([]int, n)
				for i := range order {
					order[i] = i
				}
				key := func(i int) float32 {
					if axis == 0 {
						return keys[i][line]
					}
					return keys[line][i]
				}
				sort.SliceStable(order, func(i, j int) bool {
					if reverse {
						return key(order[i]) > key(order[j])
					}
					return key(order[i]) < key(order[j])
				})
				for dst, src := range order {
					if axis == 0 {
						wantKeys[dst][line] = keys[src][line]
						wantIDs[dst][line] = ids[src][line]
					} else {
						wantKeys[line][dst] = keys[line][src]
						wantIDs[line][dst] = ids[line][src]
					}
				}
			}
			out, err := e.Call(keys, ids, reverse)
			if err != nil {
				t.Fatalf("axis%d reverse%t: %+v", axis, reverse, err)
			}
			gotKeys, gotIDs := out[0].Value(), out[1].Value()
			for _, v := range out {
				v.FinalizeAll()
			}
			if !reflect.DeepEqual(gotKeys, wantKeys) || !reflect.DeepEqual(gotIDs, wantIDs) {
				t.Fatalf("axis%d reverse%t got %v %v want %v %v", axis, reverse, gotKeys, gotIDs, wantKeys, wantIDs)
			}
		}
		e.Finalize()
	}
}

func TestNativeSortTensorCapture(t *testing.T) {
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
	e, err := graph.NewExec(b, func(input, weights *graph.Node) *graph.Node {
		c := graph.NewClosure(input.Graph(), func(g *graph.Graph) []*graph.Node {
			left := graph.Parameter(g, "left", shapes.Make(dtypes.Float32))
			right := graph.Parameter(g, "right", shapes.Make(dtypes.Float32))
			center := graph.ReduceAllSum(weights)
			l, r := graph.Sub(left, center), graph.Sub(right, center)
			return []*graph.Node{graph.LessThan(graph.Mul(l, l), graph.Mul(r, r))}
		})
		return graph.SortFunc(c, 0, true, input)[0]
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	empty := tensors.FromShape(shapes.Make(dtypes.Float32, 0))
	defer empty.FinalizeAll()
	for _, tc := range []struct {
		name    string
		weights any
		want    []float32
	}{
		{"matrix", [][]float32{{1, 2}, {3, 4}}, []float32{3, 2, 1}},
		{"changed matrix", [][]float32{{0, 0}, {0, 0}}, []float32{1, 2, 3}},
		{"empty", empty, []float32{1, 2, 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := e.Call([]float32{3, 1, 2}, tc.weights)
			if err != nil {
				t.Fatal(err)
			}
			defer out[0].FinalizeAll()
			if got := out[0].Value(); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got%v want%v", got, tc.want)
			}
		})
	}
}

func TestSortValidation(t *testing.T) {
	main := (&Backend{}).Builder("sort").Main()
	value := func(v compute.Value, err error) compute.Value {
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	x := value(main.Parameter("x", shapes.Make(dtypes.Float32, 2), nil))
	y := value(main.Parameter("y", shapes.Make(dtypes.Float32, 3), nil))
	c, err := main.Closure()
	if err != nil {
		t.Fatal(err)
	}
	l := value(c.Parameter("l", shapes.Make(dtypes.Float32), nil))
	r := value(c.Parameter("r", shapes.Make(dtypes.Float32), nil))
	pred := value(c.LessThan(l, r))
	if err := c.Return([]compute.Value{pred}, nil); err != nil {
		t.Fatal(err)
	}
	bad, err := main.Closure()
	if err != nil {
		t.Fatal(err)
	}
	bl := value(bad.Parameter("l", shapes.Make(dtypes.Float32), nil))
	value(bad.Parameter("r", shapes.Make(dtypes.Float32), nil))
	if err := bad.Return([]compute.Value{bl}, nil); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		c      compute.Function
		axis   int
		inputs []compute.Value
	}{
		{"empty", c, 0, nil}, {"axis", c, 2, []compute.Value{x}}, {"dimensions", c, 0, []compute.Value{x, y}},
		{"signature", c, 0, []compute.Value{x, x}}, {"result", bad, 0, []compute.Value{x}}, {"scope", main, 0, []compute.Value{x}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := len(main.(*function).raw.Ops)
			if _, err := main.Sort(tc.c, tc.axis, true, tc.inputs...); err == nil {
				t.Fatal("accepted invalid sort")
			}
			if len(main.(*function).raw.Ops) != before {
				t.Fatal("invalid sort emitted IR")
			}
		})
	}
}
