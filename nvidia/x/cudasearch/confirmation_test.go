package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func screenedFixture(t *testing.T) *CandidatePlan {
	t.Helper()
	e, _, _, _ := executableFixture(t)
	e.Workload.InputIdentities = map[string]string{"x": "screen-input"}
	s := executableState()
	operand := s.Graph.Add(egraph.ENode{Op: "operand", Data: int64(7)})
	s.Roots["root"] = s.Graph.Add(egraph.ENode{Op: "selected", Children: []egraph.Id{operand}})
	c, err := e.Evaluate(context.Background(), s, s.Roots["root"])
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestConfirmSelectedExpression(t *testing.T) {
	screened := screenedFixture(t)
	e, base, compiled, _ := executableFixture(t)
	e.Workload.InputIdentities = map[string]string{"x": "held-input"}
	e.References["out"] = []float32{3}
	base.value, compiled.value = 3, 3
	compile := e.Compile
	e.Compile = func(ctx context.Context, expr *egraph.Expr, b WorkloadBindings) (Executable, error) {
		if expr.Op != "selected" || len(expr.Children) != 1 || expr.Children[0].Data != int64(7) || expr == screened.Expr || expr.Children[0] == screened.Expr.Children[0] {
			t.Fatal("lost or aliased selected expression")
		}
		return compile(ctx, expr, b)
	}
	c, err := e.Confirm(context.Background(), screened)
	if err != nil {
		t.Fatal(err)
	}
	if c.ConfirmationOf != screened.ID || c.InputKey == screened.InputKey || c.WorkloadKey == screened.WorkloadKey || len(c.Trials) != 4 || c.Comparison["out"].Elements != 1 || compiled.closed != 1 {
		t.Fatalf("confirmation=%+v", c)
	}
}

func TestConfirmRefusals(t *testing.T) {
	for _, reason := range []string{"same_inputs", "missing_inputs", "renamed_workload", "estimated", "unvalidated", "timing", "tolerance", "cycle", "nil_child", "canceled", "wrong_output"} {
		t.Run(reason, func(t *testing.T) {
			screened := screenedFixture(t)
			e, _, compiled, order := executableFixture(t)
			e.Workload.InputIdentities = map[string]string{"x": "held-input"}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch reason {
			case "same_inputs":
				e.Workload.InputIdentities["x"] = "screen-input"
			case "missing_inputs":
				e.Workload.InputIdentities = nil
			case "renamed_workload":
				e.Workload.Name = "different"
				e.Workload.InputIdentities["x"] = "screen-input"
			case "estimated":
				screened.CostSource = CostEstimated
			case "unvalidated":
				screened.Validated = false
			case "timing":
				e.TimingKind = cudatrial.TimingGPUBusy
			case "tolerance":
				e.Policy.Absolute = 1
			case "cycle":
				screened.Expr.Children = []*egraph.Expr{screened.Expr}
			case "nil_child":
				screened.Expr.Children = []*egraph.Expr{nil}
			case "canceled":
				cancel()
			case "wrong_output":
				compiled.value = 2
			}
			c, err := e.Confirm(ctx, screened)
			if err == nil || c != nil {
				t.Fatalf("candidate=%v error=%v", c, err)
			}
			if reason == "wrong_output" {
				if !errors.Is(err, ErrVerificationFailed) || len(e.Receipts) != 0 || e.Comparison["out"].MaxAbsolute != 1 || compiled.closed != 1 {
					t.Fatalf("comparison=%v err=%v", e.Comparison, err)
				}
			} else if len(*order) != 0 {
				t.Fatal("executed invalid confirmation")
			}
		})
	}
}

func ExampleExecutableEvaluator_Confirm() {
	// Host-only fixture. Real callers use private executable buffers and a live lease.
	order := []string{}
	x := &testExecutable{id: "image", value: 1, kind: cudatrial.TimingSummedKernel, duration: 1, order: &order}
	e := &ExecutableEvaluator{Workload: WorkloadBindings{Name: "example", InputNames: []string{"x"}, InputIdentities: map[string]string{"x": "screen"}, NumericalPolicy: NumericalPolicyExact},
		Baseline: x, References: map[string][]float32{"out": {1}}, Rounds: 2, TimingKind: cudatrial.TimingSummedKernel, HardwareLockChecker: func() bool { return true },
		Compile: func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { y := *x; return &y, nil },
	}
	s := executableState()
	screened, _ := e.Evaluate(context.Background(), s, s.Roots["root"])
	e.Workload.InputIdentities["x"] = "held"
	confirmed, err := e.Confirm(context.Background(), screened)
	fmt.Println(err, confirmed.ConfirmationOf, len(confirmed.Trials))
	// Output: <nil> image 4
}
