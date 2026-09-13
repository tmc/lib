package egraph_test

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
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

func ExampleDequantizeIQ2XXS() {
	// 66-byte IQ2_XXS block with scale d = 1.0 (0x3c00 in float16)
	block := make([]byte, egraph.IQ2XXSBlockBytes)
	block[0] = 0x00
	block[1] = 0x3c

	dst := make([]float32, egraph.IQ2XXSBlockSize)
	if err := egraph.DequantizeIQ2XXS(dst, block); err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	fmt.Printf("first 4 weights: %.3f %.3f %.3f %.3f\n", dst[0], dst[1], dst[2], dst[3])
	// Output:
	// first 4 weights: 0.125 0.125 0.125 0.125
}

func ExampleGenerateTileIR() {
	// Build an AST expression: RMSNorm -> GEMV
	x := egraph.NewLeafNode("x")
	wNorm := egraph.NewLeafNode("w_norm")
	eps := egraph.NewLeafNode("eps")
	wQKV := egraph.NewLeafNode("w_qkv")

	norm := egraph.NewNodeExpr("rmsnorm", x, wNorm, eps)
	qkvProj := egraph.NewNodeExpr("gemv", norm, wQKV)

	// Lower directly to Tile IR assembly text (auto-fusing to tile_rmsnorm_qkv)
	textIR, err := egraph.GenerateTileIR(qkvProj, egraph.CodegenOptions{
		KernelName: "fused_rmsnorm_qkv",
		Precision:  "f16",
	})
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	fmt.Println(strings.Contains(textIR, "cuda_tile.entry @fused_rmsnorm_qkv"))
	fmt.Println(strings.Contains(textIR, "cuda_tile.rsqrt"))
	fmt.Println(strings.Contains(textIR, "cuda_tile.mma_f"))
	// Output:
	// true
	// true
	// true
}

type exampleShadowExecutor struct{}

func (e *exampleShadowExecutor) ExecuteShadow(ctx context.Context, s *cuda.Stream, cand egraph.TuningCandidate) (time.Duration, error) {
	if cand.BlockDim == 256 {
		return 35 * time.Microsecond, nil
	}
	return 50 * time.Microsecond, nil
}

func ExampleShadowTuner() {
	eg := egraph.New()

	// Represent expression: (gemv (rmsnorm x w_norm 1e-5) w_qkv)
	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	eps := eg.Add(egraph.NewLeaf("1e-5"))
	wQKV := eg.Add(egraph.NewLeaf("w_qkv"))

	norm := eg.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	qkvProj := eg.Add(egraph.NewNode("gemv", norm, wQKV))

	rule := egraph.MustRewrite("fuse_rmsnorm_gemv",
		"(gemv (rmsnorm ?x ?w_n ?e) ?w_q)",
		"(fused_rmsnorm_gemv ?x ?w_n ?e ?w_q)",
	)
	egraph.Saturate(eg, []*egraph.Rewrite{rule}, 5)

	candidates, err := egraph.ExtractCandidatePlans(eg, qkvProj, []int{128, 256})
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	// Throttle gating: sample memory bus saturation
	dramUtilization := 0.40 // 40% memory bus saturation (below 80% threshold)
	throttle := egraph.NewMemoryThrottle(0.80, func() float64 {
		return dramUtilization
	})

	tuner, err := egraph.NewShadowTuner(nil,
		egraph.WithThrottle(throttle),
		egraph.WithShadowExecutor(&exampleShadowExecutor{}),
	)
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}
	defer tuner.Close()

	ctx := context.Background()
	results, err := tuner.EvaluateCandidates(ctx, candidates[:2])
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	for _, r := range results {
		fmt.Printf("candidate: %s (block %d)\n", r.Candidate.Expr.Op, r.Candidate.BlockDim)
	}

	// Compare candidate Block(256) vs baseline Block(128)
	baseline := candidates[0]
	cand256 := candidates[1]
	shouldPromote := tuner.ShouldPromote(baseline, cand256, egraph.PromotionPolicy{MinObservations: 1})
	fmt.Printf("promote Block(256): %v\n", shouldPromote)

	// Output:
	// candidate: fused_rmsnorm_gemv (block 128)
	// candidate: fused_rmsnorm_gemv (block 256)
	// promote Block(256): true
}
