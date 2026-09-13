package egraph_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// codegenOps are every operator GenerateCUDA has a kernel for. Anything else
// is an error, which TestUnsupportedOperatorRejected covers.
var codegenOps = []string{
	"fused_swiglu_gemv",
	"fused_swiglu_q8_0",
	"fused_rmsnorm_gemv",
	"q8_0_matvec",
	"u2_simd",
	"u2_scalar",
	"q2_k_matvec",
	"fused_rope_store_kv",
	"add",
}

// TestGeneratedKernelsCompile compiles the output of GenerateCUDA for every
// operator it dispatches on. The generator's own test asserts only that the
// kernel name it passed in appears in the output, which is true of any body
// whatsoever; this gate checks that what comes out is a CUDA program.
//
// It needs the toolkit but no GPU.
func TestGeneratedKernelsCompile(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	for _, op := range codegenOps {
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
// inline PTX cvt instructions, while quantization formats ("q8_0", "u2", "q2_k")
// are supported via self-contained block structures. Unsupported types like
// "double", "float64", "int64", "iq2_xxs", and "__nv_bfloat16" are rejected.
func TestUnsupportedPrecisionRejected(t *testing.T) {
	for _, prec := range []string{"double", "float64", "int64", "__nv_bfloat16", "iq2_xxs"} {
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
	for _, prec := range []string{"float", "float32", "half", "float16", "q8_0", "u2", "q2_k"} {
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
// kernel for is refused.
//
// The default branch used to emit a kernel that copied its input to its
// output. It carried the requested kernel name and a signature unrelated to
// the operator asked for, and it compiled and ran, so a caller asking for a
// gemv received a memcpy and no diagnostic. Autotune generates from whatever
// ExtractBest returns, so it would have benchmarked that copy -- the fastest
// candidate in any field -- and reported it as the tuned kernel.
func TestUnsupportedOperatorRejected(t *testing.T) {
	for _, op := range []string{"gemv", "matmul", "softmax", ""} {
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
