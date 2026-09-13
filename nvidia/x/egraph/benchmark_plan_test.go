package egraph

import "testing"

func TestBenchmarkPlanCoversTerminalOps(t *testing.T) {
	for _, op := range TerminalOps {
		t.Run(op, func(t *testing.T) {
			plan, err := benchmarkPlan(TuningCandidate{Expr: &Expr{Op: op}})
			if err != nil {
				t.Fatalf("benchmarkPlan(%q): %v", op, err)
			}
			if len(plan.buffers) == 0 || plan.grid == nil {
				t.Fatalf("benchmarkPlan(%q) returned incomplete plan", op)
			}
		})
	}
}
