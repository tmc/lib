package nvtelemetry_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestReplayEquivalence(t *testing.T) {
	manifest := &nvtelemetry.Manifest{
		RunID:              "replay-test-1",
		SchemaVersion:      nvtelemetry.SchemaVersion,
		CollectorVersion:   nvtelemetry.CollectorVersion,
		StartTimeMonotonic: 1_000_000_000,
		EndTimeMonotonic:   10_000_000_000,
		Devices: []nvtelemetry.DeviceMetadata{
			{UUID: "GPU-REPLAY", Model: "L4"},
		},
	}

	phase1 := nvtelemetry.PhaseMarker{
		Name:                "warmup",
		StartMonotonicNanos: 1_000_000_000,
		EndMonotonicNanos:   4_000_000_000,
	}
	phase2 := nvtelemetry.PhaseMarker{
		Name:                "measurement",
		StartMonotonicNanos: 4_000_000_000,
		EndMonotonicNanos:   10_000_000_000,
	}

	samples := []nvtelemetry.Sample{
		{
			Seq:             1,
			QueryStartNanos: 2_000_000_000,
			QueryEndNanos:   2_002_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					Unit:       nvtelemetry.UnitPercent,
					DeviceUUID: "GPU-REPLAY",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      20.0,
				},
				{
					Name:       nvtelemetry.MetricTotalEnergyMJ,
					Unit:       nvtelemetry.UnitMillijoules,
					DeviceUUID: "GPU-REPLAY",
					Evidence:   nvtelemetry.EvidenceMeasured,
					IntValue:   100_000,
					IsInt:      true,
				},
			},
		},
		{
			Seq:             2,
			QueryStartNanos: 3_000_000_000,
			QueryEndNanos:   3_002_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					Unit:       nvtelemetry.UnitPercent,
					DeviceUUID: "GPU-REPLAY",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      40.0,
				},
				{
					Name:       nvtelemetry.MetricTotalEnergyMJ,
					Unit:       nvtelemetry.UnitMillijoules,
					DeviceUUID: "GPU-REPLAY",
					Evidence:   nvtelemetry.EvidenceMeasured,
					IntValue:   120_000,
					IsInt:      true,
				},
			},
		},
		{
			Seq:             3,
			QueryStartNanos: 5_000_000_000,
			QueryEndNanos:   5_002_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					Unit:       nvtelemetry.UnitPercent,
					DeviceUUID: "GPU-REPLAY",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      80.0,
				},
				{
					Name:       nvtelemetry.MetricTotalEnergyMJ,
					Unit:       nvtelemetry.UnitMillijoules,
					DeviceUUID: "GPU-REPLAY",
					Evidence:   nvtelemetry.EvidenceMeasured,
					IntValue:   150_000,
					IsInt:      true,
				},
			},
		},
		{
			Seq:             4,
			QueryStartNanos: 8_000_000_000,
			QueryEndNanos:   8_002_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					Unit:       nvtelemetry.UnitPercent,
					DeviceUUID: "GPU-REPLAY",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      90.0,
				},
				{
					Name:       nvtelemetry.MetricTotalEnergyMJ,
					Unit:       nvtelemetry.UnitMillijoules,
					DeviceUUID: "GPU-REPLAY",
					Evidence:   nvtelemetry.EvidenceMeasured,
					IntValue:   200_000,
					IsInt:      true,
				},
			},
		},
	}

	// Write stream to buffer
	var buf bytes.Buffer
	w := nvtelemetry.NewWriter(&buf, nvtelemetry.CollectorConfig{})
	if err := w.WriteManifest(manifest); err != nil {
		t.Fatalf("WriteManifest failed: %v", err)
	}
	if err := w.WritePhase(&phase1); err != nil {
		t.Fatalf("WritePhase 1 failed: %v", err)
	}
	if err := w.WriteSample(&samples[0]); err != nil {
		t.Fatalf("WriteSample 0 failed: %v", err)
	}
	if err := w.WriteSample(&samples[1]); err != nil {
		t.Fatalf("WriteSample 1 failed: %v", err)
	}
	if err := w.WritePhase(&phase2); err != nil {
		t.Fatalf("WritePhase 2 failed: %v", err)
	}
	if err := w.WriteSample(&samples[2]); err != nil {
		t.Fatalf("WriteSample 2 failed: %v", err)
	}
	if err := w.WriteSample(&samples[3]); err != nil {
		t.Fatalf("WriteSample 3 failed: %v", err)
	}

	// Replay stream
	replayedManifest, windowSummaries, energySummaries, err := nvtelemetry.Replay(&buf, 0, nvtelemetry.WindowOptions{})
	if err != nil {
		t.Fatalf("Replay failed: %v", err)
	}

	if replayedManifest.RunID != "replay-test-1" {
		t.Errorf("RunID = %s, want replay-test-1", replayedManifest.RunID)
	}

	// Check measurement phase window summary
	measSum, ok := windowSummaries["measurement"]
	if !ok || len(measSum) == 0 {
		t.Fatalf("missing measurement window summary")
	}

	// In measurement phase: samples at t=5s (80%) and t=8s (90%)
	// dt = 3s, avg = 85.0%
	if math.Abs(measSum[0].TimeWeightedMean-85.0) > 0.01 {
		t.Errorf("TimeWeightedMean = %f, want 85.0", measSum[0].TimeWeightedMean)
	}

	// Check measurement phase energy summary
	measEnergy, ok := energySummaries["measurement"]
	if !ok || len(measEnergy) == 0 {
		t.Fatalf("missing measurement energy summary")
	}
	// Start energy at 5s = 150_000 mJ, end at 8s = 200_000 mJ -> delta = 50_000 mJ = 50.0 J
	if math.Abs(measEnergy[0].EnergyConsumedJoules-50.0) > 0.01 {
		t.Errorf("EnergyConsumedJoules = %f, want 50.0", measEnergy[0].EnergyConsumedJoules)
	}
}
