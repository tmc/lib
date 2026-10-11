package cudair_test

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func TestReadinessBinaryShapeAdmission(t *testing.T) {
	for _, shape := range [][]int{{2, 3}, {1, 3}, {2, 2}, {3, 2}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{2, 3}, cudair.Float32)
			b := g.AddInput("b", shape, cudair.Float32)
			y, err := g.AddBinaryChecked("add", x, b)
			if !reflect.DeepEqual(shape, x.Shape) {
				if !errors.Is(err, cudair.ErrShapeMismatch) || y != nil {
					t.Fatalf("unequal shapes admitted: node=%v err=%v", y, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := g.SetOutputs(y); err != nil {
				t.Fatal(err)
			}
			s, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			got, err := cudaref.EvaluateSnapshot(s, map[string][]float32{"x": {1, 2, 3, 4, 5, 6}, "b": {10, 20, 30, 10, 20, 30}})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got[y.Name], []float32{11, 22, 33, 14, 25, 36}) {
				t.Fatalf("output = %v", got)
			}
			// A checked builder is insufficient when the public graph can be edited.
			b.Shape = []int{2, 2}
			if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrShapeMismatch) {
				t.Fatalf("edited broadcast extent accepted: %v", err)
			}
		})
	}
}

func TestReadinessUnsupportedStatefulOrder(t *testing.T) {
	for _, reorder := range []bool{false, true} {
		g := cudair.NewGraph()
		x := g.AddInput("x", []int{1}, cudair.Float32)
		a := g.AddUnary("neg", x)
		b := g.AddUnary("relu", a)
		a.Effect, a.StorageAlias = cudair.EffectMutating, x
		b.Effect, b.StorageAlias = cudair.EffectMutating, x
		if reorder {
			b.Inputs[0] = x
			nodes := g.Nodes()
			nodes[1], nodes[2] = nodes[2], nodes[1]
		}
		if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrUnsupportedEffect) {
			t.Fatalf("reorder=%v: unsupported writes admitted: %v", reorder, err)
		}
	}
	// The supported alias is a read-only reshape, followed by pure arithmetic.
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	alias := g.AddReshape(x, []int{1, 1})
	y := g.AddUnary("neg", alias)
	_ = g.SetOutputs(y)
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	got, err := cudaref.EvaluateSnapshot(s, map[string][]float32{"x": {-2}})
	if err != nil || !reflect.DeepEqual(got[y.Name], []float32{2}) {
		t.Fatalf("supported alias control: %v %v", got, err)
	}
	alias.StorageAlias = alias
	if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrUnsupportedEffect) {
		t.Fatalf("alias cycle admitted: %v", err)
	}
}

func ExampleSnapshot_Contains() {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	g.AddUnary("relu", x)
	s, err := g.Snapshot()
	if err != nil {
		fmt.Println(err)
		return
	}
	foreign := cudair.NewGraph().AddInput("x", []int{1}, cudair.Float32)
	fmt.Println(s.Contains(x), s.Contains(s.NodeByID(x.ID)), s.Contains(foreign))
	// Output: true true false
}

func TestReadinessScalarBroadcast(t *testing.T) {
	for _, shape := range [][]int{nil, {1}, {1, 1, 1}} {
		for _, left := range []bool{false, true} {
			t.Run(fmt.Sprintf("%v/left=%v", shape, left), func(t *testing.T) {
				g := cudair.NewGraph()
				x := g.AddInput("x", []int{2, 3}, cudair.Float32)
				scalar := g.AddInput("scalar", shape, cudair.Float32)
				a, b := x, scalar
				if left {
					a, b = scalar, x
				}
				y, err := g.AddBinaryChecked("sub", a, b)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(y.Shape, x.Shape) {
					t.Fatalf("shape=%v", y.Shape)
				}
				_ = g.SetOutputs(y)
				s, err := g.Snapshot()
				if err != nil {
					t.Fatal(err)
				}
				got, err := cudaref.EvaluateSnapshot(s, map[string][]float32{"x": {1, 2, 3, 4, 5, 6}, "scalar": {2}})
				want := []float32{-1, 0, 1, 2, 3, 4}
				if left {
					want = []float32{1, 0, -1, -2, -3, -4}
				}
				if err != nil || !reflect.DeepEqual(got[y.Name], want) {
					t.Fatalf("scalar result=%v err=%v want=%v", got, err, want)
				}
				if _, err := cudaref.EvaluateSnapshot(s, map[string][]float32{"x": {1, 2, 3, 4, 5, 6}, "scalar": {}}); err == nil {
					t.Fatal("missing scalar binding accepted")
				}
				// The existing bounded Tile C++ consumer supports scalar operands up to rank 2.
				if len(shape) <= 2 {
					if _, _, err := tileir.GenerateSemanticTileCPP(s, tileir.LowerOptions{KernelName: "scalar_control"}); err != nil {
						t.Fatalf("Tile scalar consumer: %v", err)
					}
				}
				scalar.Shape = []int{1, 2}
				if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrShapeMismatch) {
					t.Fatalf("nonscalar extent mutant admitted: %v", err)
				}
			})
		}
	}
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2, 3}, cudair.Float32)
	scalar := g.AddInput("scalar", nil, cudair.Float16)
	if _, err := g.AddBinaryChecked("add", x, scalar); !errors.Is(err, cudair.ErrTypeMismatch) {
		t.Fatalf("mixed scalar dtype admitted: %v", err)
	}
}
