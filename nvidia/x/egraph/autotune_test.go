package egraph_test

import (
	"context"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
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

func TestAutotunerWithHardware(t *testing.T) {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("CUDA device 0 not available: %v", err)
	}
	defer dev.Close()

	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	w := eg.Add(egraph.NewLeaf("w"))
	mul := eg.Add(egraph.NewNode("matmul", x, w))

	tuner := egraph.NewTuner(dev)
	cand, err := tuner.SelectBest(context.Background(), eg, mul, egraph.BudgetQuick)
	if err != nil {
		t.Fatalf("SelectBest with hardware failed: %v", err)
	}

	if cand == nil {
		t.Fatal("expected non-nil candidate from autotuner")
	}

	t.Logf("Selected best candidate: %s with blockDim=%d", cand.KernelName, cand.BlockDim)
}
