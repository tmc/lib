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
	root := eg.Add(egraph.NewNode("matmul", x, w))
	cand, err := egraph.NewTuner(nil).SelectBest(context.Background(), eg, root, egraph.BudgetInstant)
	if err != nil || cand == nil || cand.BlockDim != 128 {
		t.Fatalf("candidate=%+v err=%v", cand, err)
	}
	if !strings.Contains(cand.Key(), "b128") {
		t.Errorf("key=%q", cand.Key())
	}
	defaultKey := cand.Key()
	other := *cand
	other.KernelName = "Kernel_Auto_Block128"
	if other.Key() == defaultKey {
		t.Fatal("default and benchmark candidates share a cache key")
	}
}

// TestAutotunerWithHardware tunes an operator that GenerateCUDA can lower and
// checks that the returned candidate was actually compiled and launched.
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
	cand, err := egraph.NewTuner(dev).SelectBest(context.Background(), eg, root, egraph.BudgetQuick)
	if err != nil {
		t.Fatal(err)
	}
	if cand == nil || cand.CUDASource == "" || !strings.Contains(cand.KernelName, "Kernel_Auto_Block") {
		t.Fatalf("unbenchmarked candidate: %+v", cand)
	}
}

// TestAutotunerRejectsUngeneratableOperator prevents a failed benchmark sweep
// from being reported as a successful, sourceless default candidate.
// TestAutotunerRejectsUngeneratableOperator checks that an operator with no
// kernel is refused rather than returned as an unbenchmarked default.
func TestAutotunerRejectsUngeneratableOperator(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	w := eg.Add(egraph.NewLeaf("w"))
	root := eg.Add(egraph.NewNode("matmul", x, w))
	cand, err := egraph.NewTuner(dev).SelectBest(context.Background(), eg, root, egraph.BudgetQuick)
	if err == nil || cand != nil || !strings.Contains(err.Error(), "no candidate could be benchmarked") {
		t.Fatalf("candidate=%+v err=%v", cand, err)
	}
}
