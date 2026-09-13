package egraph_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func Example() {
	eg := egraph.New()

	// Represent expression: (gemv (rmsnorm x w_norm 1e-5) w_qkv)
	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	eps := eg.Add(egraph.NewLeaf("1e-5"))
	wQKV := eg.Add(egraph.NewLeaf("w_qkv"))

	norm := eg.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	qkvProj := eg.Add(egraph.NewNode("gemv", norm, wQKV))

	// Define megakernel rewrite
	rule := egraph.MustRewrite("fuse_rmsnorm_gemv",
		"(gemv (rmsnorm ?x ?w_n ?e) ?w_q)",
		"(fused_rmsnorm_gemv ?x ?w_n ?e ?w_q)",
	)

	// Saturate equivalence graph
	egraph.Saturate(eg, []*egraph.Rewrite{rule}, 5)

	// Extract optimal kernel plan
	best, _, err := egraph.ExtractBest(eg, qkvProj, egraph.FusedMegakernelCost{})
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	fmt.Println(best)
	// Output:
	// (fused_rmsnorm_gemv x w_norm 1e-5 w_qkv)
}

func ExampleTransformerTileRules() {
	eg := egraph.New()

	x := eg.Add(egraph.NewLeaf("x"))
	wg := eg.Add(egraph.NewLeaf("w_g"))
	wu := eg.Add(egraph.NewLeaf("w_u"))
	wd := eg.Add(egraph.NewLeaf("w_d"))

	gate := eg.Add(egraph.NewNode("matmul", x, wg))
	up := eg.Add(egraph.NewNode("matmul", x, wu))
	silu := eg.Add(egraph.NewNode("silu", gate))
	act := eg.Add(egraph.NewNode("mul", silu, up))
	ffnOut := eg.Add(egraph.NewNode("matmul", act, wd))

	egraph.Saturate(eg, egraph.TransformerTileRules(), 10)

	best, _, err := egraph.ExtractBest(eg, ffnOut, egraph.FusedMegakernelCost{})
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	fmt.Println(best.Op)
	// Output:
	// tile_swiglu
}

func ExampleSubByteRules() {
	eg := egraph.New()

	x := eg.Add(egraph.NewLeaf("x"))
	w := eg.Add(egraph.NewLeaf("w"))
	scale := eg.Add(egraph.NewLeaf("scale"))

	dequant := eg.Add(egraph.NewNode("dequantize_fp4_e2m1", w, scale))
	mul := eg.Add(egraph.NewNode("matmul", x, dequant))

	egraph.Saturate(eg, egraph.SubByteRules(), 5)

	best, _, err := egraph.ExtractBest(eg, mul, egraph.FusedMegakernelCost{})
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	fmt.Println(best)
	// Output:
	// (mmaf_scaled x (ptx_unswizzle_fp4_e2m1 w) scale)
}

