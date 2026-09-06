package egraph_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// TestBenchmarkCandidateArity launches every kernel this package generates
// through the autotuner's benchmark path.
//
// BenchmarkCandidate always passes exactly four arguments -- two inputs, an
// output and an element count -- which is the add kernel's signature. The two
// fused kernels take seven parameters. Passing four to a kernel expecting seven
// is a launch ABI error: the driver does not check arity, so the missing
// parameters are read from whatever follows the argument array, and the kernel
// dereferences those as device pointers. Nothing in the generator, the nvrtc
// compile gate, or a status-code check can see it.
func TestBenchmarkCandidateArity(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()

	for _, op := range []string{"add", "fused_rmsnorm_gemv", "fused_swiglu_gemv"} {
		t.Run(op, func(t *testing.T) {
			expr := &egraph.Expr{Op: op, Children: []*egraph.Expr{{Op: "x"}, {Op: "w"}}}
			src, err := egraph.GenerateCUDA(expr, egraph.CodegenOptions{KernelName: "K"})
			if err != nil {
				t.Fatalf("GenerateCUDA: %v", err)
			}
			cand := egraph.TuningCandidate{
				Expr:       expr,
				BlockDim:   128,
				KernelName: "K",
				CUDASource: src,
			}
			d, err := egraph.BenchmarkCandidate(dev, cand, 5)
			t.Logf("op=%s duration=%v err=%v", op, d, err)
			if err != nil {
				t.Fatalf("BenchmarkCandidate(%s) failed: %v", op, err)
			}
		})
	}
}
