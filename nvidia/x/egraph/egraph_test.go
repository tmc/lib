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

func TestCublasLtNotPreferredOverLowerable(t *testing.T) {
	eg := egraph.New()
	a := eg.Add(egraph.NewLeaf("a"))
	b := eg.Add(egraph.NewLeaf("b"))
	mul := eg.Add(egraph.NewNode("matmul", a, b))

	// An e-graph containing both matmul and candidate cublaslt_gemm must not
	// select the unlowerable host library operator in FusedMegakernelCost.
	eg.Add(egraph.NewNode("cublaslt_gemm", b, a))

	best, cost, err := egraph.ExtractBest(eg, mul, egraph.FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("ExtractBest failed: %v", err)
	}
	if best.Op == "cublaslt_gemm" {
		t.Errorf("unlowerable cublaslt_gemm must not be preferred over matmul, got %s (cost %.2f)", best.String(), cost)
	}
}

func TestCostModelRewardsNoUnlowerableOp(t *testing.T) {
	costModel := egraph.FusedMegakernelCost{}

	// Ghost / unlowerable operators must be penalized and NEVER receive a reward (< 1.0).
	ghostOps := []string{
		"fused_unknown_megakernel",
		"fused_ghost_reduction",
		"tile_nonexistent_op",
		"tile_arbitrary_unsupported",
		"custom_unsupported_activation",
	}

	for _, op := range ghostOps {
		cost := costModel.NodeCost(egraph.ENode{Op: op}, []float64{0.0, 0.0})
		if cost < 1.0 {
			t.Errorf("unlowerable ghost op %q received favorable cost %.2f; want >= 1.0", op, cost)
		}
		if egraph.IsLowerableOp(op) {
			t.Errorf("IsLowerableOp(%q) = true, want false", op)
		}
	}

	// Lowerable operators must be accepted by IsLowerableOp and receive rewarded costs (< 1.0).
	lowerableOps := []struct {
		op         string
		childCosts []float64
		maxCost    float64
	}{
		{"fused_swiglu_gemv", []float64{0.0, 0.0}, 0.5},
		{"tile_swiglu", []float64{0.0, 0.0}, 0.5},
		{"fused_swiglu_q8_0", []float64{0.0, 0.0}, 0.5},
		{"fused_rmsnorm_gemv", []float64{0.0, 0.0}, 0.5},
		{"tile_rmsnorm_qkv", []float64{0.0, 0.0}, 0.5},
		{"tile_rmsnorm_proj", []float64{0.0, 0.0}, 0.5},
		{"fused_gemv_pipeline", []float64{0.0, 0.0}, 0.5},
		{"tile_mma", []float64{0.0, 0.0}, 0.5},
		// PDL pipeline with independent preamble work (childCosts[1] > 0) receives 0.6 baseCost.
		{"pdl_gemv_pipeline", []float64{0.0, 0.1, 0.1}, 0.9},
	}

	for _, tc := range lowerableOps {
		if !egraph.IsLowerableOp(tc.op) {
			t.Errorf("IsLowerableOp(%q) = false, want true", tc.op)
		}
		cost := costModel.NodeCost(egraph.ENode{Op: tc.op, Children: make([]egraph.Id, len(tc.childCosts))}, tc.childCosts)
		if cost > tc.maxCost {
			t.Errorf("lowerable op %q cost = %.2f; want <= %.2f", tc.op, cost, tc.maxCost)
		}
	}

	// Entry-syncing PDL without preamble work must be penalized (>= 2.0).
	entrySyncCost := costModel.NodeCost(egraph.ENode{Op: "pdl_entry_sync"}, []float64{0.0, 0.0})
	if entrySyncCost < 2.0 {
		t.Errorf("entry-syncing PDL cost = %.2f, want >= 2.0", entrySyncCost)
	}
}

func TestLowerabilityPredicates(t *testing.T) {
	tests := []struct {
		op          string
		wantCUDA    bool
		wantTileIR  bool
		wantUnified bool
	}{
		{"fused_swiglu_gemv", true, false, true},
		{"fused_gemv_pipeline", true, false, true},
		{"gemv", false, false, false},
		{"matmul", false, false, false},
		{"tile_rmsnorm_gemv_swiglu", false, true, true},
		{"tile_rmsnorm_qkv_split", false, true, true},
		{"tile_gelu", false, true, true},
		{"tile_geglu", false, true, true},
		{"tile_mma_fp8", false, true, true},
		{"mma_anything", false, false, false},
		{"tma_future", false, false, false},
		{"gemma_proj", false, false, false},
		{"comma_split", false, false, false},
		{"assume_aligned_x", false, false, false},
		{"fp8_cast", false, false, false},
		{"fp4_cast", false, false, false},
		{"fused_nonexistent_ghost", false, false, false},
		{"random_custom_op", false, false, false},
	}

	for _, tc := range tests {
		gotCUDA := egraph.IsLowerableCUDAOp(tc.op)
		if gotCUDA != tc.wantCUDA {
			t.Errorf("IsLowerableCUDAOp(%q) = %v, want %v", tc.op, gotCUDA, tc.wantCUDA)
		}
		gotTileIR := egraph.IsLowerableTileIROp(tc.op)
		if gotTileIR != tc.wantTileIR {
			t.Errorf("IsLowerableTileIROp(%q) = %v, want %v", tc.op, gotTileIR, tc.wantTileIR)
		}
		gotUnified := egraph.IsLowerableOp(tc.op)
		if gotUnified != tc.wantUnified {
			t.Errorf("IsLowerableOp(%q) = %v, want %v", tc.op, gotUnified, tc.wantUnified)
		}
	}
	// Test CanLower on expressions
	validExpr := &egraph.Expr{
		Op: "fused_gemv_pipeline",
		Children: []*egraph.Expr{
			{Op: "x"},
			{Op: "w1"},
			{Op: "w2"},
		},
	}
	if !egraph.CanLower(validExpr) {
		t.Errorf("CanLower(validExpr) = false, want true")
	}

	invalidExpr := &egraph.Expr{
		Op: "fused_nonexistent_ghost",
		Children: []*egraph.Expr{
			{Op: "x"},
		},
	}
	if egraph.CanLower(invalidExpr) {
		t.Errorf("CanLower(invalidExpr) = true, want false")
	}
}

func TestLowerabilityDifferential(t *testing.T) {
	// Corpus of operators: valid CUDA ops, valid Tile IR ops, invalid ops,
	// invented prefix ops, and decoy substring matches.
	corpus := []string{
		// Valid CUDA ops
		"fused_swiglu_gemv", "tile_swiglu", "fused_swiglu_q8_0", "fused_rmsnorm_gemv",
		"tile_rmsnorm_qkv", "tile_rmsnorm_proj", "fused_gemv_pipeline", "q8_0_matvec",
		"u2_simd", "u2_scalar", "q2_k_matvec", "iq2_xxs_matvec", "fused_rope_store_kv",
		"pdl_gemv_pipeline", "pdl_fused_swiglu", "add",

		// Valid Tile IR ops
		"tile_rmsnorm_gemv_swiglu", "tile_rmsnorm_swiglu", "fused_rmsnorm_gemv_swiglu", "fused_rmsnorm_swiglu",
		"tile_rmsnorm_qkv_split", "fused_rmsnorm_qkv_split",
		"tile_geglu", "fused_geglu", "tile_gelu", "fused_gelu",
		"tile_mma", "tile_mma_aligned", "mma",
		"tile_mma_fp8", "tile_fp8_gemm", "tile_fp8", "fp8_matmul",
		"mmaf_scaled", "tile_fp4_gemm", "tile_fp4", "fp4_matmul",

		// Invalid / ghost / invented ops that MUST NOT be lowerable by either backend
		"mma_anything", "tma_future", "assume_aligned_x", "fp8_cast", "fp4_cast",
		"gemma_proj", "comma_split", "fused_anything", "tile_unknown", "pdl",
		"tile_gamma_scale", "dequant_gamma", "gamma_rms", "zzz_total_nonsense",
		"matmul", "gemv", "rmsnorm", "silu", "gelu", "swish",
	}

	costModel := egraph.FusedMegakernelCost{}

	for _, op := range corpus {
		t.Run(op, func(t *testing.T) {
			expr := &egraph.Expr{Op: op}
			if op == "add" {
				expr.Children = []*egraph.Expr{{Op: "x"}, {Op: "y"}}
			}

			// 1. CUDA differential: IsLowerableCUDAOp(op) == (GenerateCUDA succeeds)
			_, cudaErr := egraph.GenerateCUDA(expr, egraph.CodegenOptions{})
			cudaLowerable := (cudaErr == nil)
			if egraph.IsLowerableCUDAOp(op) != cudaLowerable {
				t.Errorf("CUDA differential mismatch for %q: IsLowerableCUDAOp=%v, GenerateCUDA err=%v",
					op, egraph.IsLowerableCUDAOp(op), cudaErr)
			}

			// 2. Tile IR differential: IsLowerableTileIROp(op) == (Lower and GenerateDeclarativeTile succeed)
			node := &egraph.Node{Op: op}
			_, tileErr := egraph.NewLowerer(egraph.CodegenOptions{}).Lower(node)
			tileLowerable := (tileErr == nil)
			if egraph.IsLowerableTileIROp(op) != tileLowerable {
				t.Errorf("Tile IR Lower differential mismatch for %q: IsLowerableTileIROp=%v, Lower err=%v",
					op, egraph.IsLowerableTileIROp(op), tileErr)
			}

			_, declErr := egraph.GenerateDeclarativeTile(expr, egraph.CodegenOptions{})
			declLowerable := (declErr == nil)
			if egraph.IsLowerableTileIROp(op) != declLowerable {
				t.Errorf("Tile IR Declarative differential mismatch for %q: IsLowerableTileIROp=%v, GenerateDeclarativeTile err=%v",
					op, egraph.IsLowerableTileIROp(op), declErr)
			}

			// 3. Combined lowerability: IsLowerableOp(op) == (IsLowerableCUDAOp(op) || IsLowerableTileIROp(op))
			wantCombined := egraph.IsLowerableCUDAOp(op) || egraph.IsLowerableTileIROp(op)
			if egraph.IsLowerableOp(op) != wantCombined {
				t.Errorf("IsLowerableOp(%q) = %v, want %v", op, egraph.IsLowerableOp(op), wantCombined)
			}

			// 4. Cost model check: If op is unlowerable, cost model MUST penalize it (>= 1.0)
			if !wantCombined {
				cost := costModel.NodeCost(egraph.ENode{Op: op}, []float64{0.0, 0.0})
				if cost < 1.0 {
					t.Errorf("unlowerable op %q received favorable cost %.2f in cost model; want >= 1.0", op, cost)
				}
			}
		})
	}
}

func TestGenerateCUDA_NewLowerings(t *testing.T) {
	opts := egraph.CodegenOptions{
		KernelName: "TestKernel",
		Precision:  "float",
	}

	// fused_gemv_pipeline must compile to valid CUDA C++
	pipelineExpr := &egraph.Expr{
		Op: "fused_gemv_pipeline",
		Children: []*egraph.Expr{
			{Op: "x"},
			{Op: "W1"},
			{Op: "W2"},
		},
	}
	src, err := egraph.GenerateCUDA(pipelineExpr, opts)
	if err != nil {
		t.Fatalf("GenerateCUDA(fused_gemv_pipeline) failed: %v", err)
	}
	if !strings.Contains(src, "extern __shared__ float s_stage1[]") {
		t.Errorf("GenerateCUDA output missing expected substring:\n%s", src)
	}

	// Unfused mathematical placeholder ops must be rejected by GenerateCUDA
	for _, op := range []string{"gemv", "matmul"} {
		expr := &egraph.Expr{
			Op: op,
			Children: []*egraph.Expr{
				{Op: "A"},
				{Op: "B"},
			},
		}
		_, err := egraph.GenerateCUDA(expr, opts)
		if err == nil {
			t.Errorf("GenerateCUDA(%q) unexpectedly succeeded; want placeholder rejection", op)
		}
	}
}
