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

// TestAutotunerWithHardware tunes an operator this package can generate and
// checks that the returned candidate was compiled and launched rather than
// returned as the unbenchmarked default.
//
// It runs with BudgetQuick, which is shorter than one NVRTC compile on GB10
// (100-150ms): the budget stops the search between candidates and must not
// cancel the compile already in progress.
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
	const prefix = "Kernel_Auto_fused_rmsnorm_gemv_Block"
	if cand == nil || cand.CUDASource == "" || cand.GridDim == 0 || !strings.HasPrefix(cand.KernelName, prefix) {
		t.Fatalf("unbenchmarked candidate: %+v", cand)
	}
}

// TestAutotunerRejectsUngeneratableOperator checks that an operator with no
// benchmark launch plan is refused rather than returned as a sourceless,
// unbenchmarked default.
func TestAutotunerRejectsUngeneratableOperator(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	w := eg.Add(egraph.NewLeaf("w"))
	root := eg.Add(egraph.NewNode("matmul", x, w))
	cand, err := egraph.NewTuner(dev).SelectBest(context.Background(), eg, root, egraph.BudgetQuick)
	if err == nil || cand != nil || !strings.Contains(err.Error(), `no benchmark launch plan for operator "matmul"`) {
		t.Fatalf("candidate=%+v err=%v", cand, err)
	}
}
