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
