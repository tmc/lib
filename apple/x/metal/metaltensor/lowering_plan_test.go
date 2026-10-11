package metaltensor

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/metalir"
)

func tensorTestInput(id int, name string, shape []int) *egraph.Expr {
	return &egraph.Expr{Op: "input", Sort: "float32", Data: graphInputData{ID: id, Name: name, Shape: shape, Type: metalir.Float32}}
}

func TestLoweringRejectsConflictingInputIdentity(t *testing.T) {
	a := tensorTestInput(7, "shared", []int{2, 3})
	b := tensorTestInput(7, "shared", []int{3, 4})
	expr := &egraph.Expr{Op: "matmul", Sort: "float32", Data: TensorNodeData{Shape: []int{2, 4}, Type: metalir.Float32}, Children: []*egraph.Expr{a, b}}
	if plan, err := LowerTensorExecutionPlan(expr, nil); err == nil || plan != nil {
		t.Fatalf("plan=%v err=%v", plan, err)
	}
}

func TestLoweringDeduplicatesEqualInputIdentity(t *testing.T) {
	shape := []int{2, 3}
	a := tensorTestInput(7, "shared", shape)
	b := tensorTestInput(7, "shared", []int{2, 3})
	expr := &egraph.Expr{Op: "add", Sort: "float32", Data: TensorNodeData{Shape: []int{2, 3}, Type: metalir.Float32}, Children: []*egraph.Expr{a, b}}
	plan, err := LowerTensorExecutionPlan(expr, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Inputs) != 1 || len(plan.InputLayouts) != 1 || len(plan.Stages) != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	if plan.Stages[0].Grid.Width != 6 || !strings.Contains(plan.Stages[0].Source, "kernel void") {
		t.Fatalf("stage=%+v", plan.Stages[0])
	}
	shape[0] = 99
	if plan.InputLayouts[0].Shape[0] != 2 {
		t.Fatal("plan aliases input shape")
	}
}

func TestLoweringRefusesMalformedExpressions(t *testing.T) {
	for _, kind := range []string{"nil child", "cycle", "sort", "unknown operator", "shape"} {
		t.Run(kind, func(t *testing.T) {
			input := tensorTestInput(1, "a", []int{2})
			expr := &egraph.Expr{Op: "silu", Sort: "float32", Data: TensorNodeData{Shape: []int{2}, Type: metalir.Float32}, Children: []*egraph.Expr{input}}
			switch kind {
			case "nil child":
				expr.Children[0] = nil
			case "cycle":
				expr.Children[0] = expr
			case "sort":
				input.Sort = "float16"
			case "unknown operator":
				expr.Op = "opaque"
			case "shape":
				expr.Data = TensorNodeData{Shape: []int{3}, Type: metalir.Float32}
			}
			if plan, err := LowerTensorExecutionPlan(expr, nil); err == nil || plan != nil {
				t.Fatalf("plan=%v err=%v", plan, err)
			}
		})
	}
}

func TestLoweringRefusesInvalidLeafShapes(t *testing.T) {
	for _, shape := range [][]int{{-1}, {0, math.MaxInt}, {65536, 65536}} {
		if plan, err := LowerTensorExecutionPlan(tensorTestInput(1, "invalid", shape), nil); err == nil || plan != nil {
			t.Fatalf("accepted leaf shape %v: plan=%v error=%v", shape, plan, err)
		}
	}
}
