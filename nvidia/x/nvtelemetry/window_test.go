package nvtelemetry_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestTimeWeightedMeanDiffersFromArithmeticMean(t *testing.T) {
	// Sample 0 at t = 0: val = 10
	// Sample 1 at t = 1_000_000_000 (1s): val = 10
	// Sample 2 at t = 10_000_000_000 (10s): val = 90
	samples := []nvtelemetry.Sample{
		{
			QueryStartNanos: 0,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      10.0,
				},
			},
		},
		{
			QueryStartNanos: 1_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      10.0,
				},
			},
		},
		{
			QueryStartNanos: 10_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      90.0,
				},
			},
		},
	}

	summaries, _ := nvtelemetry.AggregateWindow("test", 0, 10_000_000_000, samples, nvtelemetry.WindowOptions{})
	if len(summaries) != 1 {
		t.Fatalf("len(summaries) = %d, want 1", len(summaries))
	}

	s := summaries[0]
	// Arithmetic mean: (10 + 10 + 90) / 3 = 36.666...
	expectedArith := (10.0 + 10.0 + 90.0) / 3.0
	// Time-weighted mean:
	// Interval 0->1s (1s): avg=(10+10)/2=10. Area = 10.
	// Interval 1->10s (9s): avg=(10+90)/2=50. Area = 450.
	// Total area = 460. Total duration = 10s. Weighted mean = 46.0.
	expectedWeighted := 46.0

	if math.Abs(s.ArithmeticMean-expectedArith) > 0.01 {
		t.Errorf("ArithmeticMean = %f, want %f", s.ArithmeticMean, expectedArith)
	}
	if math.Abs(s.TimeWeightedMean-expectedWeighted) > 0.01 {
		t.Errorf("TimeWeightedMean = %f, want %f", s.TimeWeightedMean, expectedWeighted)
	}

	if s.ArithmeticMean == s.TimeWeightedMean {
		t.Errorf("expected ArithmeticMean (%f) != TimeWeightedMean (%f)", s.ArithmeticMean, s.TimeWeightedMean)
	}
}

func TestV2UnavailableEnergyDoesNotBecomeReset(t *testing.T) {
	type point struct {
		value     uint64
		available bool
	}
	tests := []struct {
		name   string
		values []point
		valid  bool
		reset  bool
		reason string
	}{
		{"increasing", []point{{10, true}, {20, true}}, true, false, ""},
		{"unavailable gap", []point{{10, true}, {0, false}, {20, true}}, false, false, nvtelemetry.ReasonGapExceeded},
		{"actual reset", []point{{20, true}, {10, true}}, false, true, nvtelemetry.ReasonCounterReset},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var stream bytes.Buffer
			writer := nvtelemetry.NewWriter(&stream, nvtelemetry.CollectorConfig{})
			manifest := nvtelemetry.Manifest{
				RunID: "v2-unavailable-gap", SchemaVersion: nvtelemetry.SchemaVersionV2,
				Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-v2-gap"}}, StartTimeMonotonic: 10,
			}
			if err := writer.WriteManifest(&manifest); err != nil {
				t.Fatalf("write V2 manifest: %v", err)
			}
			for i, point := range tt.values {
				reading := nvtelemetry.Reading{
					Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules,
					DeviceUUID: "GPU-v2-gap", Evidence: nvtelemetry.EvidenceUnavailable,
					UnavailableReason: nvtelemetry.ReasonUnsupported, IsInt: true,
				}
				if point.available {
					reading.Evidence = nvtelemetry.EvidenceMeasured
					reading.UnavailableReason = ""
					reading.IntValue = point.value
				}
				sample := nvtelemetry.Sample{
					Seq: uint64(i + 1), QueryStartNanos: int64(10 + i*10), QueryEndNanos: int64(10 + i*10),
					Readings: []nvtelemetry.Reading{reading},
				}
				if err := writer.WriteSample(&sample); err != nil {
					t.Fatalf("write V2 sample %d: %v", i, err)
				}
			}
			manifest.TotalSamples = uint64(len(tt.values))
			manifest.EndTimeMonotonic = int64(10 + len(tt.values)*10)
			manifest.IsComplete = true
			if err := writer.WriteManifest(&manifest); err != nil {
				t.Fatalf("write V2 completion: %v", err)
			}

			_, _, energies, err := nvtelemetry.Replay(bytes.NewReader(stream.Bytes()), 0, nvtelemetry.WindowOptions{})
			if err != nil {
				t.Fatalf("replay V2 stream: %v", err)
			}
			if len(energies["full_run"]) != 1 {
				t.Fatalf("got %d V2 energy summaries, want 1", len(energies["full_run"]))
			}
			got := energies["full_run"][0]
			if got.Valid != tt.valid || got.ResetDetected != tt.reset || got.Reason != tt.reason {
				t.Fatalf("energy = %+v; want valid=%v reset=%v reason=%q", got, tt.valid, tt.reset, tt.reason)
			}
			if tt.name == "unavailable gap" && (got.StartEnergyMJ != 10 || got.EndEnergyMJ != 20) {
				t.Fatalf("gap lost exact endpoints: %+v", got)
			}
		})
	}
}

func TestCumulativeEnergyResetDetected(t *testing.T) {
	samples := []nvtelemetry.Sample{
		{
			QueryStartNanos: 0,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricTotalEnergyMJ,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					IntValue:   100_000,
					IsInt:      true,
				},
			},
		},
		{
			QueryStartNanos: 5_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricTotalEnergyMJ,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					IntValue:   120_000,
					IsInt:      true,
				},
			},
		},
		{
			QueryStartNanos: 10_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricTotalEnergyMJ,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					IntValue:   5_000, // RESET! Counter decreased
					IsInt:      true,
				},
			},
		},
	}

	_, energy := nvtelemetry.AggregateWindow("test", 0, 10_000_000_000, samples, nvtelemetry.WindowOptions{})
	if len(energy) != 1 {
		t.Fatalf("len(energy) = %d, want 1", len(energy))
	}

	e := energy[0]
	if !e.ResetDetected {
		t.Errorf("ResetDetected = false, want true")
	}
	if e.Valid {
		t.Errorf("Valid = true, want false when reset is detected")
	}
	if e.Reason != nvtelemetry.ReasonCounterReset {
		t.Errorf("Reason = %s, want %s", e.Reason, nvtelemetry.ReasonCounterReset)
	}
}

func TestWindowCoverageAndGaps(t *testing.T) {
	// Sample at 2s and 8s within a 10s window
	samples := []nvtelemetry.Sample{
		{
			QueryStartNanos: 2_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      50.0,
				},
			},
		},
		{
			QueryStartNanos: 8_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      50.0,
				},
			},
		},
	}

	summaries, _ := nvtelemetry.AggregateWindow("test", 0, 10_000_000_000, samples, nvtelemetry.WindowOptions{})
	if len(summaries) != 1 {
		t.Fatalf("len(summaries) = %d, want 1", len(summaries))
	}

	s := summaries[0]
	// Covered duration is 8s - 2s = 6s. Window is 10s.
	// Coverage ratio = 6s / 10s = 0.6
	if math.Abs(s.CoverageRatio-0.6) > 0.01 {
		t.Errorf("CoverageRatio = %f, want 0.6", s.CoverageRatio)
	}
	if s.MaxGapNanos != 6_000_000_000 {
		t.Errorf("MaxGapNanos = %d, want 6_000_000_000", s.MaxGapNanos)
	}
}

func TestWindowDoesNotBridgeUnavailableGaps(t *testing.T) {
	// Sample at 0s (available), 4s (unavailable), 6s (unavailable), 10s (available)
	samples := []nvtelemetry.Sample{
		{
			QueryStartNanos: 0,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      50.0,
				},
			},
		},
		{
			QueryStartNanos: 4_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:              nvtelemetry.MetricGPUUtilization,
					DeviceUUID:        "GPU-TEST",
					Evidence:          nvtelemetry.EvidenceUnavailable,
					UnavailableReason: nvtelemetry.ReasonQueryFailure,
				},
			},
		},
		{
			QueryStartNanos: 6_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:              nvtelemetry.MetricGPUUtilization,
					DeviceUUID:        "GPU-TEST",
					Evidence:          nvtelemetry.EvidenceUnavailable,
					UnavailableReason: nvtelemetry.ReasonQueryFailure,
				},
			},
		},
		{
			QueryStartNanos: 10_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      50.0,
				},
			},
		},
	}

	summaries, _ := nvtelemetry.AggregateWindow("test", 0, 10_000_000_000, samples, nvtelemetry.WindowOptions{})
	if len(summaries) != 1 {
		t.Fatalf("len(summaries) = %d, want 1", len(summaries))
	}
	s := summaries[0]
	// Between 0s and 10s, all intervals touch an unavailable sample.
	// Therefore, covered duration MUST NOT bridge across the unavailable samples!
	if s.CoveredDurationNanos != 0 {
		t.Errorf("CoveredDurationNanos = %d, want 0 (must not bridge unavailable gaps)", s.CoveredDurationNanos)
	}
	if s.CoverageRatio != 0.0 {
		t.Errorf("CoverageRatio = %f, want 0.0", s.CoverageRatio)
	}
}

func TestWindowMaxAllowedGapEnforcement(t *testing.T) {
	// Sample at 0s, 5s, 10s (all available)
	samples := []nvtelemetry.Sample{
		{
			QueryStartNanos: 0,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      40.0,
				},
			},
		},
		{
			QueryStartNanos: 5_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      50.0,
				},
			},
		},
		{
			QueryStartNanos: 10_000_000_000,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					DeviceUUID: "GPU-TEST",
					Evidence:   nvtelemetry.EvidenceMeasured,
					Value:      60.0,
				},
			},
		},
	}

	// 1. With MaxAllowedGapNanos = 3s: gap of 5s exceeds threshold, so neither interval is covered
	optsStrict := nvtelemetry.WindowOptions{MaxAllowedGapNanos: 3_000_000_000}
	summariesStrict, _ := nvtelemetry.AggregateWindow("test", 0, 10_000_000_000, samples, optsStrict)
	if summariesStrict[0].CoveredDurationNanos != 0 {
		t.Errorf("Strict gap: CoveredDurationNanos = %d, want 0", summariesStrict[0].CoveredDurationNanos)
	}

	// 2. With MaxAllowedGapNanos = 6s: gap of 5s is acceptable, full window is covered
	optsLenient := nvtelemetry.WindowOptions{MaxAllowedGapNanos: 6_000_000_000}
	summariesLenient, _ := nvtelemetry.AggregateWindow("test", 0, 10_000_000_000, samples, optsLenient)
	if summariesLenient[0].CoveredDurationNanos != 10_000_000_000 {
		t.Errorf("Lenient gap: CoveredDurationNanos = %d, want 10_000_000_000", summariesLenient[0].CoveredDurationNanos)
	}
	if summariesLenient[0].CoverageRatio != 1.0 {
		t.Errorf("Lenient gap: CoverageRatio = %f, want 1.0", summariesLenient[0].CoverageRatio)
	}
}

func TestEnergySummaryRetainsWindowIdentity(t *testing.T) {
	samples := []nvtelemetry.Sample{
		{QueryStartNanos: 10, Readings: []nvtelemetry.Reading{{Name: nvtelemetry.MetricTotalEnergyMJ, DeviceUUID: "GPU", Unit: nvtelemetry.UnitMillijoules, Evidence: nvtelemetry.EvidenceMeasured, IsInt: true, IntValue: 1000}}},
		{QueryStartNanos: 20, Readings: []nvtelemetry.Reading{{Name: nvtelemetry.MetricTotalEnergyMJ, DeviceUUID: "GPU", Unit: nvtelemetry.UnitMillijoules, Evidence: nvtelemetry.EvidenceMeasured, IsInt: true, IntValue: 2000}}},
	}
	_, energy := nvtelemetry.AggregateWindow("measurement", 10, 20, samples, nvtelemetry.WindowOptions{})
	if len(energy) != 1 {
		t.Fatalf("got %d summaries", len(energy))
	}
	e := energy[0]
	if e.StartMonoNanos != 10 || e.EndMonoNanos != 20 || e.TotalDurationNanos != 10 || e.CoveredDurationNanos != 10 || e.CoverageRatio != 1 {
		t.Fatalf("missing window identity: %+v", e)
	}
	p := nvtelemetry.IntegratePowerToEnergy("measurement", "GPU", []nvtelemetry.SamplePoint{{Timestamp: 10, Value: 1000, Available: true}, {Timestamp: 20, Value: 1000, Available: true}}, 10, 20, nvtelemetry.WindowOptions{})
	if p.StartMonoNanos != 10 || p.EndMonoNanos != 20 || p.TotalDurationNanos != 10 {
		t.Fatalf("missing power window identity: %+v", p)
	}
}
