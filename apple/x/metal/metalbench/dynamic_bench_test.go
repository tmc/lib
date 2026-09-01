//go:build darwin

package metalbench

import (
	"testing"
	"time"

	"github.com/tmc/lib/apple/x/metal"
)

func TestDynamicScalingHarness(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	const (
		dim    = 2048
		outDim = 2048
	)

	// Define multi-phase workload schedule with dynamic regime shifts
	phases := []WorkloadPhase{
		{Name: "Phase_1_Single_User_Decode", BatchSize: 1, SeqLen: 256, Steps: 30},
		{Name: "Phase_2_Traffic_Surge_Batch_16", BatchSize: 16, SeqLen: 512, Steps: 30},
		{Name: "Phase_3_Peak_Load_Batch_32", BatchSize: 32, SeqLen: 1024, Steps: 30},
		{Name: "Phase_4_Cooldown_Batch_1", BatchSize: 1, SeqLen: 2048, Steps: 30},
	}

	// 1. Initialize Engines
	staticEngine, err := NewStaticBaselineEngine(dev, dim, outDim)
	if err != nil {
		t.Fatalf("Failed to initialize static engine: %v", err)
	}

	adaptiveEngine, err := NewAdaptivePGOEngine(dev, dim, outDim)
	if err != nil {
		t.Fatalf("Failed to initialize adaptive engine: %v", err)
	}

	t.Logf("==========================================================================================")
	t.Logf("              DYNAMIC INFERENCE SCALING BENCHMARK: STATIC VS ADAPTIVE PGO                 ")
	t.Logf("==========================================================================================")

	var (
		staticTotalTokens   int
		staticTotalDuration time.Duration
		adaptTotalTokens    int
		adaptTotalDuration  time.Duration
	)

	for _, phase := range phases {
		// Run Static Engine
		staticRes, err := staticEngine.ExecutePhase(phase)
		if err != nil {
			t.Fatalf("Static engine phase %q failed: %v", phase.Name, err)
		}
		staticTotalTokens += staticRes.StepTokens
		staticTotalDuration += staticRes.TotalTime

		// Run Adaptive Engine
		adaptRes, err := adaptiveEngine.ExecutePhase(phase)
		if err != nil {
			t.Fatalf("Adaptive engine phase %q failed: %v", phase.Name, err)
		}
		adaptTotalTokens += adaptRes.StepTokens
		adaptTotalDuration += adaptRes.TotalTime

		speedup := adaptRes.Throughput / staticRes.Throughput
		t.Logf("[Phase: %-30s] B=%2d | Static: %8.1f tok/s (%6.1f µs/step) | Adaptive: %8.1f tok/s (%6.1f µs/step) | Speedup: %.2fx",
			phase.Name,
			phase.BatchSize,
			staticRes.Throughput,
			staticRes.AvgStepUs,
			adaptRes.Throughput,
			adaptRes.AvgStepUs,
			speedup,
		)
	}

	overallStaticTokPerSec := float64(staticTotalTokens) / staticTotalDuration.Seconds()
	overallAdaptTokPerSec := float64(adaptTotalTokens) / adaptTotalDuration.Seconds()
	overallSpeedup := overallAdaptTokPerSec / overallStaticTokPerSec

	t.Logf("==========================================================================================")
	t.Logf("OVERALL SCHEDULE THROUGHPUT:")
	t.Logf("  Static Baseline:    %8.1f tok/s (Total time: %v)", overallStaticTokPerSec, staticTotalDuration)
	t.Logf("  Adaptive Dynamic:   %8.1f tok/s (Total time: %v)", overallAdaptTokPerSec, adaptTotalDuration)
	t.Logf("  Aggregate Speedup:  %.2fx", overallSpeedup)
	t.Logf("==========================================================================================")

	if overallSpeedup < 1.0 {
		t.Errorf("Expected adaptive dynamic engine to outperform static baseline, got %.2fx", overallSpeedup)
	}
}
