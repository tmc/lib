// SPDX-License-Identifier: Apache-2.0

package egraph

import (
	"testing"
	"time"
)

func TestPGOFeedbackCostModel(t *testing.T) {
	fallback := FusedMegakernelCost{}
	model := NewPGOFeedbackCostModel(fallback)

	node := ENode{Op: "fused_rmsnorm_gemv"}

	// Before measurements, should return fallback cost
	initialCost := model.NodeCost(node, nil)
	if initialCost != 1.0 {
		t.Errorf("initial cost = %f, want fallback 1.0", initialCost)
	}

	// Record real measurement of 450ns = 0.45us
	model.RecordMeasurement("fused_rmsnorm_gemv", 450*time.Nanosecond)
	model.RecordMeasurement("fused_rmsnorm_gemv", 550*time.Nanosecond)

	measuredCost := model.NodeCost(node, nil)
	expectedUs := 0.5 // (450 + 550) / 2 / 1000 = 0.5us
	if measuredCost != expectedUs {
		t.Errorf("measured cost = %f, want %f", measuredCost, expectedUs)
	}
}

func TestPGOTrackerTelemetry(t *testing.T) {
	tracker := NewPGOTracker("Fused Megakernel Layer 0..15")

	tracker.RecordIteration("Baseline Discrete Passes", 4.12, false)
	tracker.RecordIteration("Naive RMSNorm+GEMV Fusion", 2.85, false)
	tracker.RecordIteration("Auto-Tuned TG [32x8] + SwiGLU", 2.10, true)

	tracker.RecordTrial("FusedMegakernel", [3]int{16, 16, 1}, 285.0, 100)
	tracker.RecordTrial("FusedMegakernel", [3]int{32, 8, 1}, 210.0, 100)
	tracker.RecordTrial("FusedMegakernel", [3]int{64, 4, 1}, 250.0, 100)

	tracker.RecordHotSwap("FusedMegakernel", "PSO_TG_16x16", "PSO_TG_32x8", "Lower latency measured in trial")
	tracker.LockConvergence(true)

	telemetry := tracker.Telemetry()

	if len(telemetry.History) != 3 {
		t.Fatalf("history length = %d, want 3", len(telemetry.History))
	}
	if telemetry.History[2].SpeedupPct <= 0 {
		t.Errorf("expected positive speedup percentage, got %f", telemetry.History[2].SpeedupPct)
	}
	if !telemetry.History[2].Active {
		t.Errorf("expected iteration 2 to be active")
	}
	if telemetry.History[0].Active {
		t.Errorf("expected iteration 0 to be inactive")
	}

	if len(telemetry.AutoTuningTrials) != 3 {
		t.Fatalf("trials length = %d, want 3", len(telemetry.AutoTuningTrials))
	}
	if telemetry.Convergence.OptimalTG != [3]int{32, 8, 1} {
		t.Errorf("optimal TG = %v, want [32 8 1]", telemetry.Convergence.OptimalTG)
	}
	if !telemetry.Convergence.Locked {
		t.Errorf("expected convergence locked")
	}
	if len(telemetry.HotSwaps) != 1 {
		t.Fatalf("hot swaps length = %d, want 1", len(telemetry.HotSwaps))
	}
	if telemetry.CumulativeSpeedup <= 1.0 {
		t.Errorf("expected cumulative speedup > 1.0, got %f", telemetry.CumulativeSpeedup)
	}
}
