package egraph_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestKernelASTSplit(t *testing.T) {
	tests := []struct {
		name              string
		expr              *egraph.Expr
		depInputs         []string
		wantIndepInputs   []string
		wantDepInputs     []string
		wantPreambleCount int
	}{
		{
			name: "gemv_weight_fetch",
			expr: &egraph.Expr{
				Op: "gemv",
				Children: []*egraph.Expr{
					{Op: "x"},
					{Op: "w"},
				},
			},
			depInputs:         []string{"x"},
			wantIndepInputs:   []string{"w"},
			wantDepInputs:     []string{"x"},
			wantPreambleCount: 1,
		},
		{
			name: "quantized_unpack_gemv",
			expr: &egraph.Expr{
				Op: "gemv",
				Children: []*egraph.Expr{
					{Op: "x"},
					{
						Op: "ptx_unpack_q8_0",
						Children: []*egraph.Expr{
							{Op: "w"},
							{Op: "scale"},
						},
					},
				},
			},
			depInputs:         []string{"x"},
			wantIndepInputs:   []string{"scale", "w"},
			wantDepInputs:     []string{"x"},
			wantPreambleCount: 1,
		},
		{
			name: "gemv_with_bias",
			expr: &egraph.Expr{
				Op: "add",
				Children: []*egraph.Expr{
					{
						Op: "gemv",
						Children: []*egraph.Expr{
							{Op: "x"},
							{Op: "w"},
						},
					},
					{Op: "bias"},
				},
			},
			depInputs:         []string{"x"},
			wantIndepInputs:   []string{"bias", "w"},
			wantDepInputs:     []string{"x"},
			wantPreambleCount: 2,
		},
		{
			name: "fused_swiglu",
			expr: &egraph.Expr{
				Op: "fused_swiglu_gemv",
				Children: []*egraph.Expr{
					{Op: "x"},
					{Op: "w_gate"},
					{Op: "w_up"},
					{Op: "w_down"},
				},
			},
			depInputs:         []string{"x"},
			wantIndepInputs:   []string{"w_down", "w_gate", "w_up"},
			wantDepInputs:     []string{"x"},
			wantPreambleCount: 3,
		},
		{
			name: "fused_rmsnorm",
			expr: &egraph.Expr{
				Op: "fused_rmsnorm_gemv",
				Children: []*egraph.Expr{
					{Op: "x"},
					{Op: "w_norm"},
					{Op: "w_proj"},
				},
			},
			depInputs:         []string{"x"},
			wantIndepInputs:   []string{"w_norm", "w_proj"},
			wantDepInputs:     []string{"x"},
			wantPreambleCount: 2,
		},
		{
			name: "pdl_gemv_pipeline_node",
			expr: &egraph.Expr{
				Op: "pdl_gemv_pipeline",
				Children: []*egraph.Expr{
					{Op: "x"},
					{Op: "w1"},
					{Op: "w2"},
				},
			},
			wantIndepInputs:   []string{"w2"},
			wantDepInputs:     []string{"prev"},
			wantPreambleCount: 1,
		},
		{
			name: "existing_pdl_split_ast",
			expr: &egraph.Expr{
				Op: "pdl_split",
				Children: []*egraph.Expr{
					{
						Op: "pdl_preamble",
						Children: []*egraph.Expr{
							{Op: "w"},
							{Op: "bias"},
						},
					},
					{Op: "griddepcontrol_wait"},
					{
						Op: "gemv",
						Children: []*egraph.Expr{
							{Op: "x"},
							{Op: "w"},
						},
					},
				},
			},
			depInputs:         []string{"x"},
			wantIndepInputs:   []string{"bias", "w"},
			wantDepInputs:     []string{"x"},
			wantPreambleCount: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ks, err := egraph.SplitKernel(tc.expr, tc.depInputs...)
			if err != nil {
				t.Fatalf("SplitKernel failed: %v", err)
			}

			if len(ks.Preamble) != tc.wantPreambleCount {
				t.Errorf("preamble count = %d, want %d", len(ks.Preamble), tc.wantPreambleCount)
			}

			if !sliceEqual(ks.IndependentInputs, tc.wantIndepInputs) {
				t.Errorf("IndependentInputs = %v, want %v", ks.IndependentInputs, tc.wantIndepInputs)
			}

			if !sliceEqual(ks.DependentInputs, tc.wantDepInputs) {
				t.Errorf("DependentInputs = %v, want %v", ks.DependentInputs, tc.wantDepInputs)
			}

			// Validate AST representation
			ast := ks.AST()
			if ast.Op != "pdl_split" {
				t.Errorf("AST Op = %q, want %q", ast.Op, "pdl_split")
			}
			if len(ast.Children) != 3 {
				t.Fatalf("AST Children count = %d, want 3", len(ast.Children))
			}
			if ast.Children[1].Op != "griddepcontrol_wait" {
				t.Errorf("AST Barrier Op = %q, want %q", ast.Children[1].Op, "griddepcontrol_wait")
			}

			// Validate convenience wrapper
			wrapperAST, err := egraph.SplitPreambleBody(tc.expr, tc.depInputs...)
			if err != nil {
				t.Fatalf("SplitPreambleBody failed: %v", err)
			}
			if wrapperAST.String() != ast.String() {
				t.Errorf("SplitPreambleBody AST = %s, want %s", wrapperAST.String(), ast.String())
			}
		})
	}
}

func TestGeneratePDLCUDA(t *testing.T) {
	tests := []struct {
		name         string
		expr         *egraph.Expr
		opts         egraph.CodegenOptions
		wantContains []string
	}{
		{
			name: "pdl_gemv_pipeline",
			expr: &egraph.Expr{
				Op: "pdl_gemv_pipeline",
				Children: []*egraph.Expr{
					{Op: "x"},
					{Op: "w1"},
					{Op: "w2"},
				},
			},
			opts: egraph.CodegenOptions{KernelName: "PDLGEMVPipelineKernel"},
			wantContains: []string{
				"PDLGEMVPipelineKernel",
				"cudaTriggerProgrammaticLaunchCompletion()",
				"cudaGridDependencySynchronize()",
				"asm volatile(\"griddepcontrol.wait;\":::\"memory\");",
				"(a) PDL Preamble",
				"(b) PDL Barrier",
				"(c) PDL Dependent Body",
			},
		},
		{
			name: "pdl_split_ast",
			expr: &egraph.Expr{
				Op: "pdl_split",
				Children: []*egraph.Expr{
					{Op: "w2"},
					{Op: "griddepcontrol_wait"},
					{
						Op: "gemv",
						Children: []*egraph.Expr{
							{Op: "x"},
							{Op: "w2"},
						},
					},
				},
			},
			opts: egraph.CodegenOptions{KernelName: "PDLSplitKernel"},
			wantContains: []string{
				"PDLSplitKernel",
				"cudaGridDependencySynchronize()",
				"asm volatile(\"griddepcontrol.wait;\":::\"memory\");",
				"(a) PDL Preamble",
				"(b) PDL Barrier",
				"(c) PDL Dependent Body",
			},
		},
		{
			name: "fused_swiglu_pdl_split",
			expr: &egraph.Expr{
				Op: "fused_swiglu_gemv",
				Children: []*egraph.Expr{
					{Op: "x"},
					{Op: "w_gate"},
					{Op: "w_up"},
					{Op: "w_down"},
				},
			},
			opts: egraph.CodegenOptions{KernelName: "PDLSwiGLUKernel", EnablePDL: true},
			wantContains: []string{
				"PDLSwiGLUKernel",
				"w_prefetch",
				"cudaGridDependencySynchronize()",
				"asm volatile(\"griddepcontrol.wait;\":::\"memory\");",
				"silu",
			},
		},
		{
			name: "fused_rmsnorm_pdl_split",
			expr: &egraph.Expr{
				Op: "fused_rmsnorm_gemv",
				Children: []*egraph.Expr{
					{Op: "x"},
					{Op: "w_norm"},
					{Op: "w_proj"},
				},
			},
			opts: egraph.CodegenOptions{KernelName: "PDLRMSNormKernel", EnablePDL: true},
			wantContains: []string{
				"PDLRMSNormKernel",
				"w_norm_val",
				"cudaGridDependencySynchronize()",
				"asm volatile(\"griddepcontrol.wait;\":::\"memory\");",
				"rsqrtf",
			},
		},
		{
			name: "q8_0_matvec_pdl_split",
			expr: &egraph.Expr{
				Op: "q8_0_matvec",
				Children: []*egraph.Expr{
					{Op: "w"},
					{Op: "x"},
				},
			},
			opts: egraph.CodegenOptions{KernelName: "PDLQ8_0Kernel", EnablePDL: true},
			wantContains: []string{
				"PDLQ8_0Kernel",
				"scale_prefetch",
				"cudaGridDependencySynchronize()",
				"asm volatile(\"griddepcontrol.wait;\":::\"memory\");",
				"warp_sum",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src, err := egraph.GenerateCUDA(tc.expr, tc.opts)
			if err != nil {
				t.Fatalf("GenerateCUDA failed: %v", err)
			}

			for _, want := range tc.wantContains {
				if !strings.Contains(src, want) {
					t.Errorf("generated CUDA missing %q\n--- Source ---\n%s", want, src)
				}
			}

			// Also verify KernelSplit.GenerateCUDA method
			ks, err := egraph.SplitKernel(tc.expr)
			if err == nil {
				splitSrc, err := ks.GenerateCUDA(tc.opts)
				if err != nil {
					t.Fatalf("ks.GenerateCUDA failed: %v", err)
				}
				if !strings.Contains(splitSrc, "cudaGridDependencySynchronize()") {
					t.Errorf("ks.GenerateCUDA missing synchronization barrier")
				}
			}
		})
	}
}

func TestPDLCostModelSelection(t *testing.T) {
	t.Run("PreambleWorkPresentsEarnsOverlapReward", func(t *testing.T) {
		eg := egraph.New()

		x := eg.Add(egraph.NewLeaf("x"))
		w1 := eg.Add(egraph.NewLeaf("w1"))
		w2 := eg.Add(egraph.NewLeaf("w2"))

		gemv1 := eg.Add(egraph.NewNode("gemv", x, w1))
		gemv2 := eg.Add(egraph.NewNode("gemv", gemv1, w2))

		// Only PDL rewrite rules (no monolithic fusion rule)
		rules := egraph.PDLRules()
		egraph.Saturate(eg, rules, 10)

		best, cost, err := egraph.ExtractBest(eg, gemv2, egraph.FusedMegakernelCost{})
		if err != nil {
			t.Fatalf("ExtractBest failed: %v", err)
		}

		// Must select pdl_gemv_pipeline because 3.60 < 7.00 (unfused sequential)
		if !strings.Contains(best.String(), "pdl_gemv_pipeline") {
			t.Errorf("expected plan to select pdl_gemv_pipeline, got %s", best.String())
		}
		if cost > 4.0 {
			t.Errorf("expected earned PDL cost < 4.0, got %.2f", cost)
		}
		t.Logf("Extracted PDL plan: %s (cost: %.2f)", best, cost)
	})

	t.Run("QuantizedUnpackPreambleSelected", func(t *testing.T) {
		eg := egraph.New()

		x := eg.Add(egraph.NewLeaf("x"))
		w1 := eg.Add(egraph.NewLeaf("w1"))
		w2 := eg.Add(egraph.NewLeaf("w2"))
		scale := eg.Add(egraph.NewLeaf("scale"))

		unpack := eg.Add(egraph.NewNode("ptx_unpack_q8_0", w2, scale))
		gemv1 := eg.Add(egraph.NewNode("gemv", x, w1))
		gemv2 := eg.Add(egraph.NewNode("gemv", gemv1, unpack))

		rules := egraph.PDLRules()
		egraph.Saturate(eg, rules, 10)

		best, cost, err := egraph.ExtractBest(eg, gemv2, egraph.FusedMegakernelCost{})
		if err != nil {
			t.Fatalf("ExtractBest failed: %v", err)
		}

		if !strings.Contains(best.String(), "pdl_gemv_pipeline") {
			t.Errorf("expected best plan to select pdl_gemv_pipeline with unpacking preamble, got %s", best.String())
		}
		t.Logf("Extracted unpacking PDL plan: %s (cost: %.2f)", best, cost)
	})

	t.Run("FullyFusedMegakernelPreferredOverPDL", func(t *testing.T) {
		eg := egraph.New()

		x := eg.Add(egraph.NewLeaf("x"))
		w1 := eg.Add(egraph.NewLeaf("w1"))
		w2 := eg.Add(egraph.NewLeaf("w2"))

		gemv1 := eg.Add(egraph.NewNode("gemv", x, w1))
		gemv2 := eg.Add(egraph.NewNode("gemv", gemv1, w2))

		rules := []*egraph.Rewrite{
			egraph.MustRewrite("pdl_overlap_gemv",
				"(gemv (gemv ?x ?w1) ?w2)",
				"(pdl_gemv_pipeline ?x ?w1 ?w2)"),
			egraph.MustRewrite("fuse_gemv_pipeline",
				"(gemv (gemv ?x ?w1) ?w2)",
				"(fused_gemv_pipeline ?x ?w1 ?w2)"),
		}

		egraph.Saturate(eg, rules, 10)

		best, cost, err := egraph.ExtractBest(eg, gemv2, egraph.FusedMegakernelCost{})
		if err != nil {
			t.Fatalf("ExtractBest failed: %v", err)
		}

		// When full fusion is possible, fused megakernel (3.20) beats PDL pipeline (3.60)
		if strings.Contains(best.String(), "pdl_") {
			t.Errorf("pdl_ node must not be preferred over lowerable fused node, got %s", best.String())
		}
		if !strings.Contains(best.String(), "fused_gemv_pipeline") {
			t.Errorf("expected best plan to select fused_gemv_pipeline, got %s", best.String())
		}
		t.Logf("Extracted fused plan: %s (cost: %.2f)", best, cost)
	})

	t.Run("EntrySyncWithoutPreambleUnearned", func(t *testing.T) {
		costModel := egraph.FusedMegakernelCost{}

		// Node with independent preamble work
		earnedNode := egraph.ENode{
			Op:       "pdl_gemv_pipeline",
			Children: []egraph.Id{1, 2, 3},
		}
		earnedCost := costModel.NodeCost(earnedNode, []float64{1.0, 1.0, 1.0})

		// Node explicitly tagged as entry-syncing / no preamble
		unearnedNode := egraph.ENode{
			Op:       "pdl_entry_sync_pipeline",
			Children: []egraph.Id{1, 2, 3},
		}
		unearnedCost := costModel.NodeCost(unearnedNode, []float64{1.0, 1.0, 1.0})

		if earnedCost >= unearnedCost {
			t.Errorf("earnedCost (%.2f) must be strictly less than unearnedCost (%.2f)", earnedCost, unearnedCost)
		}
		t.Logf("Earned PDL Cost: %.2f, Unearned PDL Cost: %.2f", earnedCost, unearnedCost)
	})
}

func TestPDLKernelCompilesNVRTC(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()

	src, err := egraph.GenerateCUDA(&egraph.Expr{
		Op: "pdl_gemv_pipeline",
		Children: []*egraph.Expr{
			{Op: "x"},
			{Op: "w1"},
			{Op: "w2"},
		},
	}, egraph.CodegenOptions{
		KernelName: "TestPDLCompilation",
	})
	if err != nil {
		t.Fatalf("GenerateCUDA failed: %v", err)
	}

	compiled, err := nvrtcx.Compile(src, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed for PDL kernel: %v\n--- Source ---\n%s", err, src)
	}

	ptx := compiled.PTX()
	if !strings.Contains(ptx, "griddepcontrol.wait;") {
		t.Errorf("PTX missing griddepcontrol.wait; instruction")
	}
	t.Logf("Successfully compiled PDL kernel: %d bytes PTX with griddepcontrol.wait;", len(ptx))
}

func sliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func Example_pdlSplit() {
	// A consumer kernel that performs GEMV and adds bias, where 'x' arrives
	// from a predecessor kernel.
	expr := &egraph.Expr{
		Op: "add",
		Children: []*egraph.Expr{
			{
				Op: "gemv",
				Children: []*egraph.Expr{
					{Op: "x"},
					{Op: "w"},
				},
			},
			{Op: "bias"},
		},
	}

	// Split the kernel into independent preamble loads (w, bias) and dependent body (gemv)
	// separated by griddepcontrol.wait.
	ks, err := egraph.SplitKernel(expr, "x")
	if err != nil {
		panic(err)
	}

	fmt.Println("Preamble inputs:", ks.IndependentInputs)
	fmt.Println("Dependent inputs:", ks.DependentInputs)
	fmt.Println("AST:", ks.AST())
	// Output:
	// Preamble inputs: [bias w]
	// Dependent inputs: [x]
	// AST: (pdl_split (pdl_preamble w bias) griddepcontrol_wait (add (gemv x w) bias))
}
