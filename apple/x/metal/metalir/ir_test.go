package metalir

import (
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/apple/x/metal/metalshape"
)

func TestOperationRefusesInvalidInputs(t *testing.T) {
	sentinel := errors.New("invalid source")
	for _, kind := range []string{"unary", "binary", "matmul", "rope"} {
		t.Run(kind, func(t *testing.T) {
			g := NewGraph()
			bad := &Node{Shape: []int{2, 2}, Type: Float32, ValidationError: sentinel}
			good := g.AddInput("good", []int{2, 2}, Float32)
			var n *Node
			var err error
			switch kind {
			case "unary":
				n = g.AddUnary("silu", bad)
				_, err = n.SizeBytesChecked(nil)
			case "binary":
				n, err = g.AddBinaryChecked("add", bad, good)
			case "matmul":
				n, err = g.AddMatMulChecked(bad, good)
			case "rope":
				n, err = g.AddRoPEChecked(bad, good, 10000)
			}
			if !errors.Is(err, sentinel) {
				t.Fatalf("node=%v err=%v", n, err)
			}
		})
	}
	for _, op := range []string{"binary", "matmul"} {
		t.Run(op+" mixed type", func(t *testing.T) {
			g := NewGraph()
			a := g.AddInput("a", []int{2, 2}, Float32)
			b := g.AddInput("b", []int{2, 2}, Float16)
			var err error
			if op == "binary" {
				_, err = g.AddBinaryChecked("add", a, b)
			} else {
				_, err = g.AddMatMulChecked(a, b)
			}
			if err == nil {
				t.Fatal("accepted mixed input types")
			}
		})
	}
}

func TestSizeBytesRejectsUnknownType(t *testing.T) {
	for _, dtype := range []DataType{-1, Uint8 + 1} {
		n := &Node{Shape: []int{3}, Type: dtype}
		if _, err := n.SizeBytesChecked(nil); err == nil {
			t.Fatalf("accepted type %d", dtype)
		}
		g := NewGraph()
		if n, err := g.AddInputWithShapeExpr("unknown", nil, dtype); err == nil || n != nil || len(g.Nodes) != 0 {
			t.Fatalf("accepted symbolic input type %d: node=%v error=%v", dtype, n, err)
		}
		if n := g.AddInput("unknown", nil, dtype); n.ValidationError == nil {
			t.Fatalf("input type %d has no validation error", dtype)
		}
	}
}

func TestShapesAndMemoryPlan(t *testing.T) {
	g := NewGraph()
	shape := []int{2, 3}
	input := g.AddInput("a", shape, Float32)
	shape[0] = 99
	if got, err := input.SizeBytesChecked(nil); err != nil || got != 24 {
		t.Fatalf("size=%d err=%v", got, err)
	}
	a := g.AddUnary("silu", input)
	b := g.AddUnary("silu", a)
	plan, err := PlanMemory(g, 16)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Offsets[a.ID] != 0 || plan.Offsets[b.ID] != 32 || plan.TotalBytes != 56 {
		t.Fatalf("plan=%+v", plan)
	}
	b.ID = a.ID
	if _, err := PlanMemory(g, 16); err == nil {
		t.Fatal("accepted duplicate node ID")
	}
	n := &Node{Shape: []int{math.MaxInt, 2}, Type: Float32}
	if _, err := n.NumElementsChecked(nil); err == nil {
		t.Fatal("accepted element overflow")
	}
	symbol, _ := metalshape.DimensionSymbol("n")
	sg := NewGraph()
	sn, _ := sg.AddInputWithShapeExpr("symbol", []metalshape.Expr{symbol}, Float16)
	if got, err := sn.SizeBytesChecked(map[string]int64{"n": 7}); err != nil || got != 14 {
		t.Fatalf("symbol size=%d err=%v", got, err)
	}
	if _, err := sn.SizeBytesChecked(nil); err == nil {
		t.Fatal("accepted unknown symbol")
	}
	input.Shape[0] = 3
	if _, err := input.ResolveShape(nil); err == nil {
		t.Fatal("accepted contradictory concrete shape")
	}
}

func TestCheckedConstructorsRefuseNilGraph(t *testing.T) {
	var g *Graph
	input := &Node{Shape: []int{2, 2}, Type: Float32}
	constructors := []func() (*Node, error){
		func() (*Node, error) { return g.AddInputWithShapeExpr("input", nil, Float32) },
		func() (*Node, error) { return g.AddBinaryChecked("add", input, input) },
		func() (*Node, error) { return g.AddMatMulChecked(input, input) },
		func() (*Node, error) { return g.AddRoPEChecked(input, input, 10000) },
		func() (*Node, error) { return g.AddRMSNormChecked(input, input, 0.001) },
	}
	for _, construct := range constructors {
		if n, err := construct(); n != nil || err == nil {
			t.Fatalf("node=%v error=%v", n, err)
		}
	}
}
