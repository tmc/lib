package egraph

import (
	"testing"
)

func TestEGraphSaturation(t *testing.T) {
	eg := New()
	x := eg.Add(NewLeaf("x"))
	wNorm := eg.Add(NewLeaf("w_norm"))
	eps := eg.Add(NewLeaf("eps"))
	wProj := eg.Add(NewLeaf("w_proj"))

	normX := eg.Add(NewNode("rmsnorm", x, wNorm, eps))
	proj := eg.Add(NewNode("gemv", normX, wProj))

	rules := []*Rewrite{
		MustRewrite("fuse_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)"),
	}

	Saturate(eg, rules, 10)
	expr, cost, err := ExtractBest(eg, proj, FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("extract failed: %v", err)
	}
	if expr.Op != "fused_rmsnorm_gemv" {
		t.Fatalf("expected fused_rmsnorm_gemv, got %s", expr.Op)
	}
	if cost <= 0 {
		t.Fatalf("expected positive cost, got %f", cost)
	}
}
