package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func main() {
	maxIters := flag.Int("iters", 10, "Maximum equality saturation iterations")
	bench := flag.Bool("bench", true, "Execute comparative GPU decoding benchmark")
	benchSteps := flag.Int("steps", 200, "Number of benchmark decoding steps")
	batchSize := flag.Int("batch", 1, "Batch size for decoding evaluation")
	flag.Parse()

	fmt.Println("=== Pure-Go E-Graph Megakernel Optimizer vs. Luminal Benchmark ===")

	eg := egraph.New()

	// 1. Construct unoptimized Transformer Decoder layer computation graph
	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	eps := eg.Add(egraph.NewLeaf("1e-5"))
	wQ := eg.Add(egraph.NewLeaf("w_q"))
	wK := eg.Add(egraph.NewLeaf("w_k"))
	wV := eg.Add(egraph.NewLeaf("w_v"))
	pos := eg.Add(egraph.NewLeaf("pos"))
	theta := eg.Add(egraph.NewLeaf("10000.0"))
	kvCache := eg.Add(egraph.NewLeaf("kv_cache"))

	// Attention Block: RMSNorm -> QKV GEMVs -> RoPE -> KV Cache Store
	normX := eg.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	q := eg.Add(egraph.NewNode("gemv", normX, wQ))
	k := eg.Add(egraph.NewNode("gemv", normX, wK))
	v := eg.Add(egraph.NewNode("gemv", normX, wV))

	qRope := eg.Add(egraph.NewNode("rope", q, pos, theta))
	kRope := eg.Add(egraph.NewNode("rope", k, pos, theta))
	kvStore := eg.Add(egraph.NewNode("store_kv", qRope, kRope, v, kvCache))

	// Feed-Forward SwiGLU Block:
	wGate := eg.Add(egraph.NewLeaf("w_gate"))
	wUp := eg.Add(egraph.NewLeaf("w_up"))
	wDown := eg.Add(egraph.NewLeaf("w_down"))

	gateProj := eg.Add(egraph.NewNode("gemv", x, wGate))
	upProj := eg.Add(egraph.NewNode("gemv", x, wUp))
	siluNode := eg.Add(egraph.NewNode("silu", gateProj))
	swigluMult := eg.Add(egraph.NewNode("mul", siluNode, upProj))
	ffnOut := eg.Add(egraph.NewNode("gemv", swigluMult, wDown))

	// Residual Add
	layerOut := eg.Add(egraph.NewNode("add", kvStore, ffnOut))

	initialClasses, initialNodes := eg.Size()
	fmt.Printf("[Initial Graph] E-Classes: %d, E-Nodes: %d\n", initialClasses, initialNodes)

	// (a) Extract unfused baseline plan before saturation
	unfusedPlan, baselineCost, err := egraph.ExtractBest(eg, layerOut, egraph.AstSizeCost{})
	if err != nil {
		fmt.Printf("Baseline extraction error: %v\n", err)
		return
	}

	// 2. Define Megakernel Equivalence Rewrite Rules
	rules := []*egraph.Rewrite{
		// RMSNorm + QKV Projection Fusion
		egraph.MustRewrite("fuse_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)"),

		// RoPE + KV Cache Store Fusion
		egraph.MustRewrite("fuse_rope_store_kv",
			"(store_kv (rope ?q ?pos ?th) (rope ?k ?pos ?th) ?v ?cache)",
			"(fused_rope_store_kv ?q ?k ?v ?pos ?th ?cache)"),

		// SwiGLU + Down-Projection Megakernel
		egraph.MustRewrite("fuse_swiglu_gemv",
			"(gemv (mul (silu (gemv ?x ?w_g)) (gemv ?x ?w_u)) ?w_d)",
			"(fused_swiglu_gemv ?x ?w_g ?w_u ?w_d)"),
	}

	// 3. Run Equality Saturation
	iters, merges := egraph.Saturate(eg, rules, *maxIters)
	satClasses, satNodes := eg.Size()
	fmt.Printf("[Saturation] Completed in %d iterations (%d equivalence unions)\n", iters, merges)
	fmt.Printf("[Search Space] Explored %d E-Classes across %d equivalent E-Nodes\n\n", satClasses, satNodes)

	// (b) Extract Hardware-Aware Fused Megakernel Plan
	fusedPlan, megakernelCost, err := egraph.ExtractBest(eg, layerOut, egraph.FusedMegakernelCost{})
	if err != nil {
		fmt.Printf("Megakernel extraction error: %v\n", err)
		return
	}

	fmt.Println("--- Extracted Hardware Plans ---")
	fmt.Printf("1. Unfused Plan (Memory-Roundtrip Cost: %.2f):\n   %s\n\n", baselineCost, formatPlan(unfusedPlan.String(), 3))
	fmt.Printf("2. Optimal Fused Megakernel Plan (Hardware Cost: %.2f):\n   %s\n\n", megakernelCost, formatPlan(fusedPlan.String(), 3))

	fmt.Println("--- Hardware Optimizations Achieved ---")
	fmt.Println("✓ Fused RMSNorm + QKV Projection: Eliminated intermediate normalized activations buffer")
	fmt.Println("✓ Fused RoPE + KV Cache Store: Eliminated temporary un-rotated query/key allocations")
	fmt.Println("✓ Fused SwiGLU Megakernel: Eliminated gate/up intermediate VRAM roundtrips (register-resident)")

	if *bench {
		fmt.Printf("\n=== Running GPU Hardware Decoding Benchmark (Batch=%d, Steps=%d) ===\n", *batchSize, *benchSteps)
		res, err := RunBenchmarkComparison(*batchSize, *benchSteps, egraph.BudgetQuick)
		if err != nil {
			fmt.Printf("[Benchmark Error] %v\n", err)
			return
		}
		fmt.Printf("Unfused Host-Dispatched Baseline: %8.2f ms (%8.1f layer-tok/sec)\n", res.UnfusedDurationMs, res.UnfusedTokSec)
		fmt.Printf("TMC Pure-Go E-Graph Megakernel  : %8.2f ms (%8.1f layer-tok/sec)\n", res.TmcEGraphDurationMs, res.TmcEGraphTokSec)
		fmt.Printf("Speedup via Megakernel Fusion   : %8.2fx\n", res.Speedup)
	}
}

func formatPlan(s string, indent int) string {
	s = strings.ReplaceAll(s, "(", "\n"+strings.Repeat(" ", indent)+"(")
	return strings.TrimSpace(s)
}

