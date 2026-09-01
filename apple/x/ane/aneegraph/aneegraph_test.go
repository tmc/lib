//go:build darwin

package aneegraph

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/apple/x/metal/egraph"
)

func TestGenerateMILSwiGLU(t *testing.T) {
	expr := &egraph.Expr{
		Op: "fused_swiglu_gemv",
		Children: []*egraph.Expr{
			{Op: "x"},
			{Op: "w_gate"},
			{Op: "w_up"},
			{Op: "w_down"},
		},
	}

	opts := ANEOptions{
		SpatialDim:       1,
		HiddenDim:        64,
		IntermediateDim:  128,
		WeightBlobOffset: 64,
	}

	milText, err := GenerateMIL(expr, opts)
	if err != nil {
		t.Fatalf("GenerateMIL failed: %v", err)
	}

	if !strings.Contains(milText, "program(1.3)") {
		t.Errorf("expected program(1.3), got:\n%s", milText)
	}
	if !strings.Contains(milText, "conv_gate") || !strings.Contains(milText, "conv_up") || !strings.Contains(milText, "swiglu_product") {
		t.Errorf("expected fused SwiGLU ops in generated MIL, got:\n%s", milText)
	}
}

func TestGenerateMILRMSNorm(t *testing.T) {
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	eps := eg.Add(egraph.NewLeaf("eps"))
	wProj := eg.Add(egraph.NewLeaf("w_proj"))

	normX := eg.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	proj := eg.Add(egraph.NewNode("gemv", normX, wProj))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("fuse_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)"),
	}

	egraph.Saturate(eg, rules, 5)
	bestExpr, _, err := egraph.ExtractBest(eg, proj, egraph.FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("ExtractBest failed: %v", err)
	}

	opts := ANEOptions{
		SpatialDim:       1,
		HiddenDim:        64,
		IntermediateDim:  128,
		WeightBlobOffset: 64,
	}

	milText, err := GenerateMIL(bestExpr, opts)
	if err != nil {
		t.Fatalf("GenerateMIL failed: %v", err)
	}

	if !strings.Contains(milText, "program(1.3)") {
		t.Errorf("expected program(1.3), got:\n%s", milText)
	}
	if !strings.Contains(milText, "conv") {
		t.Errorf("expected conv op in generated MIL, got:\n%s", milText)
	}
}

func Example() {
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	eps := eg.Add(egraph.NewLeaf("eps"))
	wProj := eg.Add(egraph.NewLeaf("w_proj"))

	normX := eg.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	proj := eg.Add(egraph.NewNode("gemv", normX, wProj))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("fuse_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)"),
	}

	egraph.Saturate(eg, rules, 5)
	bestExpr, _, _ := egraph.ExtractBest(eg, proj, egraph.FusedMegakernelCost{})

	milText, _ := GenerateMIL(bestExpr, ANEOptions{HiddenDim: 64, SpatialDim: 1})
	fmt.Printf("Generated ANE MIL valid: %t\n", strings.Contains(milText, "program(1.3)"))

	// Output:
	// Generated ANE MIL valid: true
}
