package egraph_test

import (
	"context"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestAutotunerInstant(t *testing.T) {
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	w := eg.Add(egraph.NewLeaf("w"))
	mul := eg.Add(egraph.NewNode("matmul", x, w))

	tuner := egraph.NewTuner(nil) // nil dev defaults to instant extraction
	cand, err := tuner.SelectBest(context.Background(), eg, mul, egraph.BudgetInstant)
	if err != nil {
		t.Fatalf("SelectBest failed: %v", err)
	}

	if cand == nil || cand.BlockDim != 128 {
		t.Errorf("unexpected candidate: %+v", cand)
	}

	key := cand.Key()
	if !strings.Contains(key, "b128") {
		t.Errorf("expected candidate key to contain b128, got %q", key)
	}
}

// TestAutotunerWithHardware tunes an operator this package can actually
// generate, and checks the returned candidate was benchmarked.
//
// It used to tune "matmul", for which there is no kernel, and assert only that
// the result was non-nil. Every candidate was discarded and SelectBest returned
// the unbenchmarked default, so the test passed over an autotuner that had
// tuned nothing -- and would have passed just as well if the benchmark path
// were removed entirely.
func TestAutotunerWithHardware(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()

	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	w := eg.Add(egraph.NewLeaf("w"))
	root := eg.Add(egraph.NewNode("fused_rmsnorm_gemv", x, w))

	tuner := egraph.NewTuner(dev)
	cand, err := tuner.SelectBest(context.Background(), eg, root, egraph.BudgetQuick)
	if err != nil {
		t.Fatalf("SelectBest with hardware failed: %v", err)
	}
	if cand == nil {
		t.Fatal("expected non-nil candidate from autotuner")
	}
	// A candidate that was never benchmarked carries no source: SelectBest
	// only fills CUDASource on the path that compiles and launches it.
	if cand.CUDASource == "" {
		t.Errorf("candidate %q has no CUDA source, so it was never benchmarked", cand.KernelName)
	}
	if !strings.Contains(cand.KernelName, "Kernel_Auto_Block") {
		t.Errorf("candidate %q is the untuned default, not a benchmarked block size", cand.KernelName)
	}
	t.Logf("Selected best candidate: %s with blockDim=%d", cand.KernelName, cand.BlockDim)
}

// TestAutotunerRejectsUngeneratableOperator checks that tuning an operator with
// no kernel is an error rather than an unbenchmarked default carrying no
// source, which a caller would then try to compile.
func TestAutotunerRejectsUngeneratableOperator(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()

	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	w := eg.Add(egraph.NewLeaf("w"))
	root := eg.Add(egraph.NewNode("matmul", x, w))

	cand, err := egraph.NewTuner(dev).SelectBest(context.Background(), eg, root, egraph.BudgetQuick)
	if err == nil {
		t.Fatalf("SelectBest returned candidate %+v for an operator with no kernel", cand)
	}
	if !strings.Contains(err.Error(), "no candidate could be benchmarked") {
		t.Errorf("error = %v, want it to say nothing could be benchmarked", err)
	}
}
