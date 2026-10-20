package nvtelemetry_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func windowAdmissionSamples(name, firstUnit, secondUnit string) []nvtelemetry.Sample {
	return []nvtelemetry.Sample{
		{QueryStartNanos: 0, Readings: []nvtelemetry.Reading{{Name: name, Unit: firstUnit, DeviceUUID: "GPU-input", Value: 1000, Evidence: nvtelemetry.EvidenceMeasured}}},
		{QueryStartNanos: 1e9, Readings: []nvtelemetry.Reading{{Name: name, Unit: secondUnit, DeviceUUID: "GPU-input", Value: 1, Evidence: nvtelemetry.EvidenceMeasured}}},
	}
}

func TestWindowInputUnitAdmission(t *testing.T) {
	for _, tt := range []struct {
		name, metric, first, second string
		valid                       bool
	}{
		{"canonical", nvtelemetry.MetricPowerUsageMW, nvtelemetry.UnitMilliwatts, nvtelemetry.UnitMilliwatts, true},
		{"mixed", nvtelemetry.MetricPowerUsageMW, nvtelemetry.UnitMilliwatts, "W", false},
		{"uniform wrong", nvtelemetry.MetricPowerUsageMW, "W", "W", false},
		{"missing registered", nvtelemetry.MetricPowerUsageMW, "", "", false},
		{"custom", "application.gauge", "widgets", "widgets", true},
		{"custom dimensionless", "application.gauge", "", "", true},
		{"mixed custom", "application.gauge", "widgets", "gadgets", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			samples := windowAdmissionSamples(tt.metric, tt.first, tt.second)
			legacy, _ := nvtelemetry.AggregateWindow("test", 0, 1e9, samples, nvtelemetry.WindowOptions{})
			strict, _, err := nvtelemetry.AggregateWindowStrict("test", 0, 1e9, samples, nvtelemetry.WindowOptions{})
			if !tt.valid {
				if len(legacy) != 0 || err == nil || len(strict) != 0 {
					t.Fatalf("invalid units admitted: legacy=%+v strict=%+v err=%v", legacy, strict, err)
				}
				return
			}
			for _, got := range [][]nvtelemetry.WindowSummary{legacy, strict} {
				if err != nil || len(got) != 1 || got[0].Unit != tt.first || got[0].TimeWeightedMean != 500.5 || got[0].CoverageRatio != 1 {
					t.Fatalf("valid units: summaries=%+v err=%v", got, err)
				}
			}
		})
	}
}

func TestWindowInputMalformedGroupPreservesSibling(t *testing.T) {
	samples := windowAdmissionSamples(nvtelemetry.MetricPowerUsageMW, nvtelemetry.UnitMilliwatts, "W")
	for i := range samples {
		samples[i].Readings = append(samples[i].Readings, nvtelemetry.Reading{
			Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent,
			DeviceUUID: "GPU-input", Value: 25, Evidence: nvtelemetry.EvidenceMeasured,
		})
	}
	got, energy := nvtelemetry.AggregateWindow("test", 0, 1e9, samples, nvtelemetry.WindowOptions{})
	if len(got) != 1 || len(energy) != 0 || got[0].MetricName != nvtelemetry.MetricGPUUtilization || got[0].TimeWeightedMean != 25 {
		t.Fatalf("malformed group contaminated sibling: %+v %+v", got, energy)
	}
	// An invalid cumulative-energy unit must not produce a valid energy delta.
	samples = windowAdmissionSamples(nvtelemetry.MetricTotalEnergyMJ, "J", "J")
	for i := range samples {
		samples[i].Readings[0].IsInt = true
		samples[i].Readings[0].IntValue = uint64(i + 1)
	}
	got, energy = nvtelemetry.AggregateWindow("test", 0, 1e9, samples, nvtelemetry.WindowOptions{})
	if len(got) != 0 || len(energy) != 0 {
		t.Fatalf("wrong-unit energy admitted: %+v %+v", got, energy)
	}
}

func TestWindowInputPowerTimeAdmission(t *testing.T) {
	for _, tt := range []struct {
		name  string
		times []int64
	}{
		{"replayed interval", []int64{0, 1e9, 0, 1e9}},
		{"descending", []int64{1e9, 0}},
		{"identical duplicate", []int64{0, 0, 1e9}},
		{"middle duplicate", []int64{0, 5e8, 5e8, 1e9}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var points []nvtelemetry.SamplePoint
			for _, ts := range tt.times {
				points = append(points, nvtelemetry.SamplePoint{Timestamp: ts, Value: 1000, Available: true})
			}
			got := nvtelemetry.IntegratePowerToEnergy("test", "GPU-input", points, 0, 1e9, nvtelemetry.WindowOptions{})
			if got.Valid || got.Reason != "non_increasing_sample_time" || got.EnergyConsumedJoules != 0 || got.CoveredDurationNanos != 0 {
				t.Fatalf("invalid time sequence admitted: %+v", got)
			}
		})
	}
	points := []nvtelemetry.SamplePoint{{Timestamp: 0, Value: 1000, Available: true}, {Timestamp: 1e9, Value: 1000, Available: true}}
	got := nvtelemetry.IntegratePowerToEnergy("test", "GPU-input", points, 0, 1e9, nvtelemetry.WindowOptions{})
	if !got.Valid || got.EnergyConsumedJoules != 1 || got.CoveredDurationNanos != 1e9 || got.TotalDurationNanos != 1e9 || got.CoverageRatio != 1 {
		t.Fatalf("monotone positive: %+v", got)
	}
	points[1].Timestamp = 0
	points[1].Value = 2000
	if got := nvtelemetry.IntegratePowerToEnergy("test", "GPU-input", points, 0, 1e9, nvtelemetry.WindowOptions{}); got.Valid || got.Reason != "non_increasing_sample_time" {
		t.Fatalf("conflicting duplicate admitted: %+v", got)
	}
	points[1].Available = false
	if got := nvtelemetry.IntegratePowerToEnergy("test", "GPU-input", points, 0, 1e9, nvtelemetry.WindowOptions{}); got.Reason != "non_increasing_sample_time" {
		t.Fatalf("unavailable duplicate escaped time admission: %+v", got)
	}
}

func TestWindowInputPowerGapCoverage(t *testing.T) {
	for _, tt := range []struct {
		name        string
		unavailable bool
		gap         int64
		covered     int64
		joules      float64
	}{
		{"unavailable", true, 0, 250_000_000, 0.25},
		{"maximum gap", false, 300_000_000, 500_000_000, 0.5},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var points []nvtelemetry.SamplePoint
			for _, timestamp := range []int64{0, 250_000_000, 750_000_000, 1e9} {
				points = append(points, nvtelemetry.SamplePoint{Timestamp: timestamp, Value: 1000, Available: true})
			}
			points[2].Available = !tt.unavailable
			got := nvtelemetry.IntegratePowerToEnergy("test", "GPU-input", points, 0, 1e9, nvtelemetry.WindowOptions{MaxAllowedGapNanos: tt.gap})
			if !got.Valid || got.CoveredDurationNanos != tt.covered || got.EnergyConsumedJoules != tt.joules || got.CoverageRatio != float64(tt.covered)/1e9 || got.CoveredDurationNanos > got.TotalDurationNanos {
				t.Fatalf("gap coverage: %+v", got)
			}
		})
	}
}
