package nvtelemetry_test

import (
	"bytes"
	"context"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
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

func TestReplayV4PreservesPrecisionAndCoverageStates(t *testing.T) {
	const device = "GPU-replay-v4"
	const large = uint64(1)<<53 + 1
	const next = large + 10
	coverageNumerator, coverageDenominator := uint64(5), uint64(10)
	sampler := &precisionSampler{
		devices: []nvtelemetry.DeviceMetadata{{UUID: device}},
		batches: make([][]nvtelemetry.Reading, 2),
	}
	for i, value := range []uint64{large, next} {
		start := int64(100 + i*10)
		interval := &tracecontract.Interval{Start: start, End: start + 10, Clock: "mono-test"}
		measuredCoverage := &nvtelemetry.SamplingCoverage{
			Status: "measured", Basis: "time", Scope: "activity-union/eligible-window-v1",
			Numerator:   &nvtelemetry.Number{Uint: &coverageNumerator},
			Denominator: &nvtelemetry.Number{Uint: &coverageDenominator}, Unit: "ns",
		}
		unknownCoverage := &nvtelemetry.SamplingCoverage{Status: "unknown", Reason: "not_reported"}
		sampler.batches[i] = []nvtelemetry.Reading{
			{
				Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules,
				DeviceUUID: device, Evidence: nvtelemetry.EvidenceMeasured,
				Number: &nvtelemetry.Number{Uint: &value},
				Observation: &nvtelemetry.Observation{
					SourceClass: "synthetic", Kind: "counter", CounterEpoch: "boot-v4",
					Interval: interval,
					Observer: nvtelemetry.Observer{
						ID: "synthetic-activity", Version: "1", Mode: "active", Pass: "pass-1",
						Population: "eligible activity intervals", SamplingCoverage: measuredCoverage,
					},
				},
			},
			{
				Name: "v4-unknown-coverage-gauge", Unit: "count",
				DeviceUUID: device, Evidence: nvtelemetry.EvidenceMeasured,
				Number: &nvtelemetry.Number{Uint: u64ptr(1)},
				Observation: &nvtelemetry.Observation{
					SourceClass: "synthetic", Kind: "gauge", Interval: interval,
					Observer: nvtelemetry.Observer{SamplingCoverage: unknownCoverage},
				},
			},
		}
	}

	clock := nvtelemetry.NewFakeClock(100, 1_000)
	var stream bytes.Buffer
	cfg := nvtelemetry.CollectorConfig{SchemaVersion: nvtelemetry.SchemaVersionV4}
	writer := nvtelemetry.NewWriter(&stream, cfg)
	collector := nvtelemetry.NewCollector("replay-v4-precision", sampler, clock, cfg, writer)
	if err := collector.InitStream(); err != nil {
		t.Fatalf("init V4 collector: %v", err)
	}
	for i := range sampler.batches {
		if err := collector.Step(context.Background()); err != nil {
			t.Fatalf("collect V4 sample %d: %v", i, err)
		}
		clock.Advance(10 * time.Nanosecond)
	}
	if _, err := collector.Stop(); err != nil {
		t.Fatalf("stop V4 collector: %v", err)
	}
	wire := append([]byte(nil), stream.Bytes()...)
	manifest, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(wire), 0)
	if err != nil {
		t.Fatalf("strictly read V4 stream: %v", err)
	}
	if manifest.SchemaVersion != nvtelemetry.SchemaVersionV4 || len(samples) != 2 {
		t.Fatalf("strict reader returned schema %d and %d samples", manifest.SchemaVersion, len(samples))
	}
	seenEnergy := 0
	seenMeasured, seenUnknown := false, false
	wantEnergy := []uint64{large, next}
	for _, sample := range samples {
		for _, reading := range sample.Readings {
			if reading.Number == nil {
				t.Fatalf("strict reader lost Number for %s", reading.Name)
			}
			coverage := reading.Observation.Observer.SamplingCoverage
			if coverage == nil {
				t.Fatalf("strict reader lost V4 sampling coverage for %s", reading.Name)
			}
			switch coverage.Status {
			case "measured":
				seenMeasured = true
				if coverage.Basis != "time" || coverage.Scope != "activity-union/eligible-window-v1" || coverage.Numerator == nil || coverage.Denominator == nil || coverage.Numerator.Uint == nil || coverage.Denominator.Uint == nil || *coverage.Numerator.Uint != coverageNumerator || *coverage.Denominator.Uint != coverageDenominator || coverage.Unit != "ns" || coverage.Reason != "" {
					t.Fatalf("measured coverage changed: %+v", coverage)
				}
			case "unknown":
				seenUnknown = true
				if coverage.Reason != "not_reported" || coverage.Numerator != nil || coverage.Denominator != nil || coverage.Unit != "" {
					t.Fatalf("unknown coverage acquired measured operands: %+v", coverage)
				}
			}
			if reading.Name == nvtelemetry.MetricTotalEnergyMJ {
				seenEnergy++
				if seenEnergy > len(wantEnergy) || reading.Number.Uint == nil || *reading.Number.Uint != wantEnergy[seenEnergy-1] {
					t.Fatalf("strict reader changed raw V4 energy number: %+v", reading.Number)
				}
			}
		}
	}
	if seenEnergy != 2 || !seenMeasured || !seenUnknown {
		t.Fatalf("strict reader missed V4 evidence: energy=%d measured=%v unknown=%v", seenEnergy, seenMeasured, seenUnknown)
	}

	replayedManifest, windows, energies, err := nvtelemetry.Replay(bytes.NewReader(wire), 0, nvtelemetry.WindowOptions{})
	if err != nil {
		t.Fatalf("replay V4 stream: %v", err)
	}
	if replayedManifest.SchemaVersion != nvtelemetry.SchemaVersionV4 {
		t.Fatalf("replay schema = %d, want V4", replayedManifest.SchemaVersion)
	}
	var energyWindow *nvtelemetry.WindowSummary
	for i := range windows["full_run"] {
		if windows["full_run"][i].MetricName == nvtelemetry.MetricTotalEnergyMJ {
			energyWindow = &windows["full_run"][i]
			break
		}
	}
	if energyWindow == nil {
		t.Fatal("V4 replay lost energy metric window")
	}
	if energyWindow.SampleValuesStatus != nvtelemetry.WindowSampleValuesUnavailable || energyWindow.SampleValuesReason != nvtelemetry.ReasonIntegerPrecision {
		t.Fatalf("V4 replay did not expose precision refusal: %+v", energyWindow)
	}
	if energyWindow.SampleCount != 2 || energyWindow.CoveredDurationNanos != 10 || energyWindow.Min != 0 || energyWindow.Max != 0 || energyWindow.StartValue != 0 || energyWindow.EndValue != 0 {
		t.Fatalf("V4 precision refusal changed summary population or endpoints: %+v", energyWindow)
	}
	if len(energies["full_run"]) != 1 {
		t.Fatalf("got %d V4 energy summaries, want 1", len(energies["full_run"]))
	}
	energy := energies["full_run"][0]
	if !energy.Valid || energy.StartEnergyMJ != large || energy.EndEnergyMJ != next || energy.EnergyConsumedMJ != 10 {
		t.Fatalf("V4 exact energy accounting changed: %+v", energy)
	}
}
