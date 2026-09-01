package egraph

import (
	"testing"
)

func TestHeterogeneousPartition(t *testing.T) {
	// Build a combined LLaMA block: Residual + Self-Attention (GPU) + SwiGLU FFN (ANE)
	root := &Expr{
		Op: "add",
		Children: []*Expr{
			{
				Op: "causal_flash_attention",
				Children: []*Expr{
					{Op: "q"},
					{Op: "k"},
					{Op: "v"},
				},
			},
			{
				Op: "fused_swiglu_gemv",
				Children: []*Expr{
					{Op: "h"},
					{Op: "w_gate"},
					{Op: "w_up"},
					{Op: "w_down"},
				},
			},
		},
	}

	pg := HeterogeneousPartition(root)

	if len(pg.ANEExprs) != 1 || pg.ANEExprs[0].Op != "fused_swiglu_gemv" {
		t.Errorf("expected 1 ANE SwiGLU expr, got: %v", pg.ANEExprs)
	}

	if len(pg.GPUExprs) < 1 {
		t.Errorf("expected GPU attention expressions, got: %v", pg.GPUExprs)
	}

	t.Logf("Heterogeneous Partition Succeeded:")
	t.Logf("  ANE Subgraphs (Neural Engine Cores): %d", len(pg.ANEExprs))
	t.Logf("  GPU Subgraphs (Metal GPU Cores):     %d", len(pg.GPUExprs))
}
