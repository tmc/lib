package main

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestQwenConfigDefaults(t *testing.T) {
	cfg := DefaultQwenConfig()
	if cfg.HeadDim() != 128 {
		t.Errorf("expected HeadDim 128, got %d", cfg.HeadDim())
	}
	if cfg.NumHiddenLayers != 2 {
		t.Errorf("expected 2 layers, got %d", cfg.NumHiddenLayers)
	}
}

func TestQwenEGraphMegakernelPlan(t *testing.T) {
	eg := egraph.New()

	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	eps := eg.Add(egraph.NewLeaf("eps"))
	wQ := eg.Add(egraph.NewLeaf("w_q"))
	wK := eg.Add(egraph.NewLeaf("w_k"))
	wV := eg.Add(egraph.NewLeaf("w_v"))
	pos := eg.Add(egraph.NewLeaf("pos"))
	theta := eg.Add(egraph.NewLeaf("theta"))
	kvCache := eg.Add(egraph.NewLeaf("kv_cache"))

	normX := eg.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	q := eg.Add(egraph.NewNode("gemv", normX, wQ))
	k := eg.Add(egraph.NewNode("gemv", normX, wK))
	v := eg.Add(egraph.NewNode("gemv", normX, wV))
	qRope := eg.Add(egraph.NewNode("rope", q, pos, theta))
	kRope := eg.Add(egraph.NewNode("rope", k, pos, theta))
	kvStore := eg.Add(egraph.NewNode("store_kv", qRope, kRope, v, kvCache))

	wGate := eg.Add(egraph.NewLeaf("w_gate"))
	wUp := eg.Add(egraph.NewLeaf("w_up"))
	wDown := eg.Add(egraph.NewLeaf("w_down"))
	gateProj := eg.Add(egraph.NewNode("gemv", x, wGate))
	upProj := eg.Add(egraph.NewNode("gemv", x, wUp))
	siluNode := eg.Add(egraph.NewNode("silu", gateProj))
	swigluMult := eg.Add(egraph.NewNode("mul", siluNode, upProj))
	ffnOut := eg.Add(egraph.NewNode("gemv", swigluMult, wDown))

	layerOut := eg.Add(egraph.NewNode("add", kvStore, ffnOut))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("fuse_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)"),
		egraph.MustRewrite("fuse_rope_store_kv",
			"(store_kv (rope ?q ?pos ?th) (rope ?k ?pos ?th) ?v ?cache)",
			"(fused_rope_store_kv ?q ?k ?v ?pos ?th ?cache)"),
		egraph.MustRewrite("fuse_swiglu_gemv",
			"(gemv (mul (silu (gemv ?x ?w_g)) (gemv ?x ?w_u)) ?w_d)",
			"(fused_swiglu_gemv ?x ?w_g ?w_u ?w_d)"),
	}

	egraph.Saturate(eg, rules, 10)
	bestPlan, cost, err := egraph.ExtractBest(eg, layerOut, egraph.FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("failed extracting best plan: %v", err)
	}

	planStr := bestPlan.String()
	if !strings.Contains(planStr, "fused_swiglu_gemv") || !strings.Contains(planStr, "fused_rmsnorm_gemv") {
		t.Errorf("expected plan to contain fused megakernels, got: %s", planStr)
	}
	t.Logf("Qwen optimal plan: %s (cost: %.2f)", planStr, cost)
}
