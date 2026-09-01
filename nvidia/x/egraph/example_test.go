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
