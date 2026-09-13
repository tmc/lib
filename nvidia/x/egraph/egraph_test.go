package egraph_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestUnionFindAndRebuild(t *testing.T) {
	eg := egraph.New()

	// a = 1, b = 1, c = 2
	a := eg.Add(egraph.NewLeaf("1"))
	b := eg.Add(egraph.NewLeaf("1"))
	c := eg.Add(egraph.NewLeaf("2"))

	if a != b {
		t.Errorf("expected hash-consing to deduplicate leaf '1', got a=%d, b=%d", a, b)
	}

	// (+ a c) and (+ b c)
	sum1 := eg.Add(egraph.NewNode("+", a, c))
	sum2 := eg.Add(egraph.NewNode("+", b, c))

	if sum1 != sum2 {
		t.Errorf("expected (+ a c) and (+ b c) to match since a == b, got sum1=%d, sum2=%d", sum1, sum2)
	}

	// Add x, y, (+ x c) and (+ y c)
	x := eg.Add(egraph.NewLeaf("x"))
	y := eg.Add(egraph.NewLeaf("y"))
	sumX := eg.Add(egraph.NewNode("+", x, c))
	sumY := eg.Add(egraph.NewNode("+", y, c))

	if sumX == sumY {
		t.Error("sumX and sumY should initially be in distinct e-classes")
	}

	// Union x and y -> congruence closure should repair (+ x c) == (+ y c)
	eg.Union(x, y)
	eg.Rebuild()

	if eg.Find(sumX) != eg.Find(sumY) {
		t.Errorf("expected congruence closure to merge sumX and sumY after union(x, y), got %d vs %d",
			eg.Find(sumX), eg.Find(sumY))
	}
}

func TestEqualitySaturationArithmetic(t *testing.T) {
	eg := egraph.New()

	// Expr: (* x 2)
	x := eg.Add(egraph.NewLeaf("x"))
	two := eg.Add(egraph.NewLeaf("2"))
	expr := eg.Add(egraph.NewNode("*", x, two))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("mul_by_two", "(* ?a 2)", "(<< ?a 1)"),
		egraph.MustRewrite("shift_to_add", "(<< ?a 1)", "(+ ?a ?a)"),
	}

	iters, totalMerges := egraph.Saturate(eg, rules, 10)
	if iters == 0 || totalMerges == 0 {
		t.Fatalf("expected rewrites to apply, got iters=%d, merges=%d", iters, totalMerges)
	}

	best, cost, err := egraph.ExtractBest(eg, expr, egraph.AstSizeCost{})
	if err != nil {
		t.Fatalf("extraction failed: %v", err)
	}

	t.Logf("Extracted best expr: %s (cost: %.2f)", best, cost)
}

func TestNewRewriteVariables(t *testing.T) {
	tests := []struct {
		name string
		lhs  string
		rhs  string
		want string
	}{
		{"bound", "(+ ?x ?y)", "(+ ?y ?x)", ""},
		{"repeated", "(+ ?x ?x)", "?x", ""},
		{"unbound", "(+ ?x 0)", "(+ ?x ?y)", "?y"},
		{"deterministic", "?x", "(+ ?z ?y)", "?y, ?z"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := egraph.NewRewrite(test.name, test.lhs, test.rhs)
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

func TestCUDAMegakernelFusionRewrites(t *testing.T) {
	eg := egraph.New()

	// High-level Transformer feed-forward graph:
	// x -> GEMV(x, w_gate) -> SiLU
	// x -> GEMV(x, w_up)
	// (SiLU * Up) -> GEMV(..., w_down)
	x := eg.Add(egraph.NewLeaf("x"))
	wGate := eg.Add(egraph.NewLeaf("w_gate"))
	wUp := eg.Add(egraph.NewLeaf("w_up"))
	wDown := eg.Add(egraph.NewLeaf("w_down"))

	gateProj := eg.Add(egraph.NewNode("gemv", x, wGate))
	siluNode := eg.Add(egraph.NewNode("silu", gateProj))
	upProj := eg.Add(egraph.NewNode("gemv", x, wUp))
	swigluMult := eg.Add(egraph.NewNode("mul", siluNode, upProj))
	ffnOut := eg.Add(egraph.NewNode("gemv", swigluMult, wDown))

	rules := []*egraph.Rewrite{
		// RMSNorm + QKV Projection
		egraph.MustRewrite("fuse_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)"),

		// SwiGLU + Down Projection Megakernel
		egraph.MustRewrite("fuse_swiglu_block",
			"(gemv (mul (silu (gemv ?x ?w_g)) (gemv ?x ?w_u)) ?w_d)",
			"(fused_swiglu_gemv ?x ?w_g ?w_u ?w_d)"),
	}

	iters, merges := egraph.Saturate(eg, rules, 10)
	if iters == 0 || merges == 0 {
		t.Fatalf("expected fusion rewrite to match, got iters=%d, merges=%d", iters, merges)
	}

	// Extract with FusedMegakernelCost model
	best, cost, err := egraph.ExtractBest(eg, ffnOut, egraph.FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("megakernel extraction failed: %v", err)
	}

	if !strings.Contains(best.String(), "fused_swiglu_gemv") {
		t.Errorf("expected extracted best kernel to be 'fused_swiglu_gemv', got: %s", best.String())
	}
	t.Logf("Optimal Megakernel Plan extracted: %s (cost: %.2f)", best, cost)

	// Test code generation from AST
	cudaSrc, err := egraph.GenerateCUDA(best, egraph.CodegenOptions{
		KernelName: "Kernel_FusedSwiGLU",
		Precision:  "float",
	})
	if err != nil {
		t.Fatalf("GenerateCUDA failed: %v", err)
	}
	if !strings.Contains(cudaSrc, "kernel_fused_swiglu") && !strings.Contains(cudaSrc, "Kernel_FusedSwiGLU") {
		t.Errorf("expected generated CUDA source to contain Kernel_FusedSwiGLU:\n%s", cudaSrc)
	}
}

func TestAutotuner(t *testing.T) {
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	w := eg.Add(egraph.NewLeaf("w"))
	mul := eg.Add(egraph.NewNode("matmul", x, w))

	tuner := egraph.NewTuner(nil) // nil dev defaults to instant extraction
	cand, err := tuner.SelectBest(nil, eg, mul, egraph.BudgetInstant)
	if err != nil {
		t.Fatalf("SelectBest failed: %v", err)
	}

	if cand == nil || cand.BlockDim != 128 {
		t.Errorf("unexpected candidate: %+v", cand)
	}
}

func TestExtractTopK(t *testing.T) {
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	wG := eg.Add(egraph.NewLeaf("w_g"))
	wU := eg.Add(egraph.NewLeaf("w_u"))
	wD := eg.Add(egraph.NewLeaf("w_d"))

	gX := eg.Add(egraph.NewNode("gemv", x, wG))
	siluG := eg.Add(egraph.NewNode("silu", gX))
	uX := eg.Add(egraph.NewNode("gemv", x, wU))
	h := eg.Add(egraph.NewNode("mul", siluG, uX))
	ffnOut := eg.Add(egraph.NewNode("gemv", h, wD))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("fuse_swiglu_block",
			"(gemv (mul (silu (gemv ?x ?w_g)) (gemv ?x ?w_u)) ?w_d)",
			"(fused_swiglu_gemv ?x ?w_g ?w_u ?w_d)"),
	}

	egraph.Saturate(eg, rules, 10)

	cands, err := egraph.ExtractTopK(eg, ffnOut, 2, egraph.FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("ExtractTopK failed: %v", err)
	}

	if len(cands) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(cands))
	}

	if cands[0].Expr.Op != "fused_swiglu_gemv" {
		t.Errorf("expected candidate 0 to be fused_swiglu_gemv, got %s", cands[0].Expr.Op)
	}
	if cands[1].Expr.Op != "gemv" {
		t.Errorf("expected candidate 1 to be gemv, got %s", cands[1].Expr.Op)
	}
	if cands[0].Cost >= cands[1].Cost {
		t.Errorf("expected candidate 0 cost (%f) < candidate 1 cost (%f)", cands[0].Cost, cands[1].Cost)
	}
}

func TestPDLNotPreferredOverLowerable(t *testing.T) {
	eg := egraph.New()

	x := eg.Add(egraph.NewLeaf("x"))
	w1 := eg.Add(egraph.NewLeaf("w1"))
	w2 := eg.Add(egraph.NewLeaf("w2"))

	stage1 := eg.Add(egraph.NewNode("gemv", x, w1))
	stage2 := eg.Add(egraph.NewNode("gemv", stage1, w2))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("pdl_overlap_gemv",
			"(gemv (gemv ?x ?w1) ?w2)",
			"(pdl_gemv_pipeline ?x ?w1 ?w2)"),
		egraph.MustRewrite("fuse_gemv_pipeline",
			"(gemv (gemv ?x ?w1) ?w2)",
			"(fused_gemv_pipeline ?x ?w1 ?w2)"),
	}

	egraph.Saturate(eg, rules, 10)

	best, cost, err := egraph.ExtractBest(eg, stage2, egraph.FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("ExtractBest failed: %v", err)
	}

	if strings.Contains(best.String(), "pdl_") {
		t.Errorf("pdl_ node must not be preferred over lowerable node, got %s", best.String())
	}
	if !strings.Contains(best.String(), "fused_gemv_pipeline") {
		t.Errorf("expected best plan to select fused_gemv_pipeline, got %s", best.String())
	}
	t.Logf("Extracted plan: %s (cost: %.2f)", best, cost)
}

