package egraph

import (
	"strings"
	"testing"
)

func TestNewRewriteVariables(t *testing.T) {
	for _, test := range []struct {
		name string
		lhs  string
		rhs  string
		want string
	}{
		{"bound", "(+ ?x ?y)", "(+ ?y ?x)", ""},
		{"repeated", "(+ ?x ?x)", "?x", ""},
		{"unbound", "(+ ?x 0)", "(+ ?x ?y)", "?y"},
		{"deterministic", "?x", "(+ ?z ?y)", "?y, ?z"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewRewrite(test.name, test.lhs, test.rhs)
			if test.want == "" {
				if err != nil {
					t.Fatalf("NewRewrite: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("NewRewrite error = %v, want text %q", err, test.want)
			}
		})
	}
}

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

func TestExtractTopK(t *testing.T) {
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

	// Extract Top 2 candidates
	cands, err := ExtractTopK(eg, proj, 2, FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("ExtractTopK failed: %v", err)
	}

	if len(cands) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(cands))
	}

	// Winner (lowest cost) should be the fused kernel
	if cands[0].Expr.Op != "fused_rmsnorm_gemv" {
		t.Errorf("expected candidate 0 to be fused_rmsnorm_gemv, got %s", cands[0].Expr.Op)
	}

	// Second candidate should be the discrete gemv
	if cands[1].Expr.Op != "gemv" {
		t.Errorf("expected candidate 1 to be gemv, got %s", cands[1].Expr.Op)
	}

	if cands[0].Cost >= cands[1].Cost {
		t.Errorf("expected candidate 0 cost (%f) < candidate 1 cost (%f)", cands[0].Cost, cands[1].Cost)
	}
}
