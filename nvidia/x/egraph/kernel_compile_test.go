package egraph_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// TestGeneratedKernelsCompile compiles the output of GenerateCUDA for every
// operator it dispatches on (egraph.TerminalOps). The generator's own test
// asserts only that the kernel name it passed in appears in the output, which
// is true of any body whatsoever; this gate checks that what comes out is a
// valid, compilable CUDA program.
//
// It needs the toolkit but no GPU.
func TestGeneratedKernelsCompile(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	for _, op := range egraph.TerminalOps {
		t.Run(op, func(t *testing.T) {
			expr := &egraph.Expr{Op: op, Children: []*egraph.Expr{{Op: "x"}, {Op: "w"}}}
			src, err := egraph.GenerateCUDA(expr, egraph.CodegenOptions{KernelName: "K"})
			if err != nil {
				t.Fatalf("GenerateCUDA: %v", err)
			}
			if _, err := cudatest.CompileSource(src, major, minor); err != nil {
				t.Fatalf("generated %s kernel does not compile: %v\n%s", op, err, src)
			}
		})
	}
}

// TestScaledFP8TemplateCompiles covers the hand-written template in fp8.go,
// which GenerateCUDA does not produce.
func TestScaledFP8TemplateCompiles(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	if _, err := cudatest.CompileSource(egraph.ScaledFP8GEMVTemplate, major, minor); err != nil {
		t.Fatalf("ScaledFP8GEMVTemplate does not compile: %v", err)
	}
}

// TestUnsupportedPrecisionRejected checks that a precision GenerateCUDA cannot
// emit valid source for is refused rather than returned.
//
// Half-precision ("half", "float16") is supported via uint16_t storage and
// inline PTX cvt instructions, while quantization formats ("q8_0", "u2", "q2_k", "iq2_xxs")
// are supported via self-contained block structures. Unsupported types like
// "double", "float64", "int64", and "__nv_bfloat16" are rejected.
func TestUnsupportedPrecisionRejected(t *testing.T) {
	for _, prec := range []string{"double", "float64", "int64", "__nv_bfloat16"} {
		t.Run(prec, func(t *testing.T) {
			expr := &egraph.Expr{Op: "fused_swiglu_gemv"}
			src, err := egraph.GenerateCUDA(expr, egraph.CodegenOptions{KernelName: "K", Precision: prec})
			if err == nil {
				t.Fatalf("GenerateCUDA accepted %q and returned %d bytes of source", prec, len(src))
			}
			if !strings.Contains(err.Error(), prec) {
				t.Errorf("error %q does not name the precision", err)
			}
		})
	}
}

// TestSupportedPrecisionsAccepted checks that supported floating point and
// quantized precision formats are accepted by GenerateCUDA.
func TestSupportedPrecisionsAccepted(t *testing.T) {
	for _, prec := range []string{"float", "float32", "half", "float16", "q8_0", "u2", "q2_k", "iq2_xxs"} {
		t.Run(prec, func(t *testing.T) {
			expr := &egraph.Expr{Op: "fused_swiglu_gemv"}
			src, err := egraph.GenerateCUDA(expr, egraph.CodegenOptions{KernelName: "K", Precision: prec})
			if err != nil {
				t.Fatalf("GenerateCUDA rejected supported precision %q: %v", prec, err)
			}
			if len(src) == 0 {
				t.Fatalf("GenerateCUDA returned empty source for precision %q", prec)
			}
		})
	}
}

// TestUnsupportedOperatorRejected checks that an operator GenerateCUDA has no
// kernel for is refused. This covers all PlaceholderOps as well as unpriced
// operators like "softmax" and the empty operator.
//
// The default branch used to emit a kernel that copied its input to its
// output. It carried the requested kernel name and a signature unrelated to
// the operator asked for, and it compiled and ran, so a caller asking for a
// gemv received a memcpy and no diagnostic. Autotune generates from whatever
// ExtractBest returns, so it would have benchmarked that copy -- the fastest
// candidate in any field -- and reported it as the tuned kernel.
func TestUnsupportedOperatorRejected(t *testing.T) {
	testOps := append([]string{"softmax", ""}, egraph.PlaceholderOps...)
	for _, op := range testOps {
		t.Run(op, func(t *testing.T) {
			expr := &egraph.Expr{Op: op, Children: []*egraph.Expr{{Op: "x"}, {Op: "w"}}}
			src, err := egraph.GenerateCUDA(expr, egraph.CodegenOptions{KernelName: "K"})
			if err == nil {
				t.Fatalf("GenerateCUDA(%q) returned source instead of an error:\n%s", op, src)
			}
			if !strings.Contains(err.Error(), "no CUDA kernel") {
				t.Errorf("GenerateCUDA(%q) error = %v, want it to name the missing kernel", op, err)
			}
		})
	}
}

// TestOperatorTaxonomy systematically validates the two-tier operator model:
//
// 1. TerminalOps and PlaceholderOps are strictly disjoint.
// 2. Every TerminalOp is recognized by IsTerminalOp, rejected by IsPlaceholderOp,
//    and produces valid CUDA code via GenerateCUDA.
// 3. Every PlaceholderOp is recognized by IsPlaceholderOp, rejected by IsTerminalOp,
//    is refused by GenerateCUDA ("no CUDA kernel"), and is priced by FusedMegakernelCost
//    with a specific non-default heuristic cost (> 0).
func TestOperatorTaxonomy(t *testing.T) {
	// Check disjointness
	terminalSet := make(map[string]bool)
	for _, op := range egraph.TerminalOps {
		if terminalSet[op] {
			t.Errorf("duplicate terminal operator: %q", op)
		}
		terminalSet[op] = true
	}

	placeholderSet := make(map[string]bool)
	for _, op := range egraph.PlaceholderOps {
		if placeholderSet[op] {
			t.Errorf("duplicate placeholder operator: %q", op)
		}
		if terminalSet[op] {
			t.Errorf("operator %q is listed as both terminal and placeholder", op)
		}
		placeholderSet[op] = true
	}

	// Verify terminal operator predicates and codegen
	for _, op := range egraph.TerminalOps {
		if !egraph.IsTerminalOp(op) {
			t.Errorf("IsTerminalOp(%q) = false, want true", op)
		}
		if egraph.IsPlaceholderOp(op) {
			t.Errorf("IsPlaceholderOp(%q) = true, want false for terminal op", op)
		}
		expr := &egraph.Expr{Op: op, Children: []*egraph.Expr{{Op: "x"}, {Op: "w"}}}
		src, err := egraph.GenerateCUDA(expr, egraph.CodegenOptions{KernelName: "TaxonomyTest"})
		if err != nil {
			t.Errorf("GenerateCUDA(%q) failed: %v", op, err)
		}
		if len(src) == 0 {
			t.Errorf("GenerateCUDA(%q) returned empty source", op)
		}
	}

	// Verify placeholder operator predicates, codegen rejection, and cost model coverage
	costModel := egraph.FusedMegakernelCost{}
	for _, op := range egraph.PlaceholderOps {
		if !egraph.IsPlaceholderOp(op) {
			t.Errorf("IsPlaceholderOp(%q) = false, want true", op)
		}
		if egraph.IsTerminalOp(op) {
			t.Errorf("IsTerminalOp(%q) = true, want false for placeholder op", op)
		}
		expr := &egraph.Expr{Op: op, Children: []*egraph.Expr{{Op: "x"}, {Op: "w"}}}
		src, err := egraph.GenerateCUDA(expr, egraph.CodegenOptions{KernelName: "TaxonomyTest"})
		if err == nil {
			t.Errorf("GenerateCUDA(%q) unexpectedly succeeded:\n%s", op, src)
		} else if !strings.Contains(err.Error(), "no CUDA kernel") {
			t.Errorf("GenerateCUDA(%q) error = %v, want error to contain \"no CUDA kernel\"", op, err)
		}

		// Verify cost model recognizes the placeholder and assigns a non-zero cost
		node := egraph.ENode{Op: op}
		cost := costModel.NodeCost(node, nil)
		if cost <= 0 {
			t.Errorf("FusedMegakernelCost.NodeCost(%q) = %v, want > 0", op, cost)
		}
	}
}
