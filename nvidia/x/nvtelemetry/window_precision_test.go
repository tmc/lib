package nvtelemetry_test

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

const precisionDevice = "GPU-window-precision"

type precisionSampler struct {
	devices []nvtelemetry.DeviceMetadata
	batches [][]nvtelemetry.Reading
	next    int
}

func (s *precisionSampler) Devices() []nvtelemetry.DeviceMetadata {
	return s.devices
}

func (s *precisionSampler) Sample(context.Context) ([]nvtelemetry.Reading, error) {
	if s.next == len(s.batches) {
		return nil, nil
	}
	readings := s.batches[s.next]
	s.next++
	return readings, nil
}

type precisionMetric struct {
	name  string
	unit  string
	first *nvtelemetry.Number
	last  *nvtelemetry.Number
	kind  string
}

func replayV3Precision(t *testing.T, metrics []precisionMetric) ([]nvtelemetry.Sample, map[string][]nvtelemetry.WindowSummary, map[string][]nvtelemetry.EnergySummary) {
	t.Helper()
	batches := make([][]nvtelemetry.Reading, 2)
	for _, metric := range metrics {
		for i, number := range []*nvtelemetry.Number{metric.first, metric.last} {
			observation := &nvtelemetry.Observation{Kind: metric.kind}
			if metric.kind == "counter" {
				observation.CounterEpoch = "boot-1"
			}
			batches[i] = append(batches[i], nvtelemetry.Reading{
				Name: metric.name, Unit: metric.unit, DeviceUUID: precisionDevice,
				Evidence: nvtelemetry.EvidenceMeasured, Number: number, Observation: observation,
			})
		}
	}

	clock := nvtelemetry.NewFakeClock(100, 1_000)
	sampler := &precisionSampler{
		devices: []nvtelemetry.DeviceMetadata{{UUID: precisionDevice}},
		batches: batches,
	}
	var stream bytes.Buffer
	cfg := nvtelemetry.CollectorConfig{SchemaVersion: nvtelemetry.SchemaVersionV3}
	writer := nvtelemetry.NewWriter(&stream, cfg)
	collector := nvtelemetry.NewCollector("window-precision", sampler, clock, cfg, writer)
	if err := collector.InitStream(); err != nil {
		t.Fatalf("init collector: %v", err)
	}
	for i := range batches {
		if err := collector.Step(context.Background()); err != nil {
			t.Fatalf("collect sample %d: %v", i, err)
		}
		clock.Advance(10 * time.Nanosecond)
	}
	if _, err := collector.Stop(); err != nil {
		t.Fatalf("stop collector: %v", err)
	}
	raw := append([]byte(nil), stream.Bytes()...)
	manifest, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0)
	if err != nil {
		t.Fatalf("read V3 stream strictly: %v", err)
	}
	if manifest.SchemaVersion != nvtelemetry.SchemaVersionV3 {
		t.Fatalf("schema version = %d, want V3", manifest.SchemaVersion)
	}
	for _, sample := range samples {
		for _, reading := range sample.Readings {
			if reading.Number == nil {
				t.Fatalf("reader lost canonical number for %s", reading.Name)
			}
		}
	}
	_, windows, energy, err := nvtelemetry.Replay(bytes.NewReader(raw), 0, nvtelemetry.WindowOptions{})
	if err != nil {
		t.Fatalf("replay V3 stream: %v", err)
	}
	return samples, windows, energy
}

func TestIndependentV3WindowSummaryPreservesUintAbove2Pow53(t *testing.T) {
	const two53 = uint64(1) << 53
	const large = two53 + 1
	const largeNext = large + 10
	negLarge, negLargeNext := -int64(large), -int64(large+2)
	minInt, maxInt := int64(math.MinInt64), int64(math.MaxInt64)
	maxUint, maxUintPrev := uint64(math.MaxUint64), uint64(math.MaxUint64-1)
	energyLast := large + 10
	metrics := []precisionMetric{
		{name: "precision_uint", unit: "count", first: &nvtelemetry.Number{Uint: u64ptr(large)}, last: &nvtelemetry.Number{Uint: u64ptr(largeNext)}, kind: "gauge"},
		{name: "precision_int", unit: "count", first: &nvtelemetry.Number{Int: i64ptr(negLarge)}, last: &nvtelemetry.Number{Int: i64ptr(negLargeNext)}, kind: "gauge"},
		{name: "precision_max_int", unit: "count", first: &nvtelemetry.Number{Int: i64ptr(maxInt)}, last: &nvtelemetry.Number{Int: i64ptr(maxInt)}, kind: "gauge"},
		{name: "precision_max_uint", unit: "count", first: &nvtelemetry.Number{Uint: u64ptr(maxUintPrev)}, last: &nvtelemetry.Number{Uint: u64ptr(maxUint)}, kind: "gauge"},
		{name: "precision_min_int", unit: "count", first: &nvtelemetry.Number{Int: i64ptr(minInt)}, last: &nvtelemetry.Number{Int: i64ptr(minInt)}, kind: "gauge"},
		{name: "precision_exact", unit: "count", first: &nvtelemetry.Number{Uint: u64ptr(two53)}, last: &nvtelemetry.Number{Uint: u64ptr(two53 + 2)}, kind: "gauge"},
		{name: "precision_zero", unit: "count", first: &nvtelemetry.Number{Uint: u64ptr(0)}, last: &nvtelemetry.Number{Uint: u64ptr(0)}, kind: "gauge"},
		{name: nvtelemetry.MetricTotalEnergyMJ, unit: nvtelemetry.UnitMillijoules, first: &nvtelemetry.Number{Uint: u64ptr(large)}, last: &nvtelemetry.Number{Uint: u64ptr(energyLast)}, kind: "counter"},
	}
	samples, windows, energy := replayV3Precision(t, metrics)
	if len(samples) != 2 {
		t.Fatalf("got %d V3 samples, want 2", len(samples))
	}
	gotNumbers := make(map[string][]*nvtelemetry.Number)
	for _, sample := range samples {
		for _, reading := range sample.Readings {
			gotNumbers[reading.Name] = append(gotNumbers[reading.Name], reading.Number)
		}
	}
	if got := gotNumbers["precision_uint"]; len(got) != 2 || *got[0].Uint != large || *got[1].Uint != largeNext {
		t.Fatalf("uint64 Number arm changed in reader: %+v", got)
	}
	if got := gotNumbers["precision_int"]; len(got) != 2 || *got[0].Int != negLarge || *got[1].Int != negLargeNext {
		t.Fatalf("int64 Number arm changed in reader: %+v", got)
	}
	if got := gotNumbers["precision_max_uint"]; len(got) != 2 || *got[0].Uint != maxUintPrev || *got[1].Uint != maxUint {
		t.Fatalf("MaxUint64 Number arm changed in reader: %+v", got)
	}
	if got := gotNumbers["precision_min_int"]; len(got) != 2 || *got[0].Int != minInt || *got[1].Int != minInt {
		t.Fatalf("MinInt64 Number arm changed in reader: %+v", got)
	}

	rows := make(map[string]nvtelemetry.WindowSummary)
	for _, summary := range windows["full_run"] {
		rows[summary.MetricName] = summary
	}
	for _, name := range []string{"precision_uint", "precision_int", "precision_max_int", "precision_max_uint", nvtelemetry.MetricTotalEnergyMJ} {
		summary, ok := rows[name]
		if !ok {
			t.Fatalf("inexact metric row %q disappeared from replay", name)
		}
		if summary.SampleValuesStatus != nvtelemetry.WindowSampleValuesUnavailable || summary.SampleValuesReason != nvtelemetry.ReasonIntegerPrecision {
			t.Fatalf("%s status/reason = %q/%q", name, summary.SampleValuesStatus, summary.SampleValuesReason)
		}
		if summary.Min != 0 || summary.Max != 0 || summary.StartValue != 0 || summary.EndValue != 0 {
			t.Fatalf("%s exposed rounded extrema/endpoints: %+v", name, summary)
		}
		if summary.SampleCount != 2 || summary.TotalDurationNanos != 20 || summary.ArithmeticMean == 0 || summary.TimeWeightedMean == 0 {
			t.Fatalf("%s lost row metadata or floating means: %+v", name, summary)
		}
	}
	for _, name := range []string{"precision_exact", "precision_zero", "precision_min_int"} {
		summary, ok := rows[name]
		if !ok || summary.SampleValuesStatus != "" || summary.SampleValuesReason != "" {
			t.Fatalf("exact sample values changed legacy status for %q: %+v", name, summary)
		}
	}
	if got := rows["precision_exact"]; got.StartValue != float64(two53) || got.EndValue != float64(two53+2) {
		t.Fatalf("exact binary64 boundary values changed: %+v", got)
	}
	if got := rows["precision_min_int"]; got.Min != float64(minInt) || got.Max != float64(minInt) {
		t.Fatalf("exact MinInt64 values refused: %+v", got)
	}
	if len(energy["full_run"]) != 1 {
		t.Fatalf("got %d energy summaries, want 1", len(energy["full_run"]))
	}
	e := energy["full_run"][0]
	if !e.Valid || e.ResetDetected || e.StartEnergyMJ != large || e.EndEnergyMJ != energyLast || e.EnergyConsumedMJ != 10 {
		t.Fatalf("exact EnergySummary changed: %+v", e)
	}
	encoded, err := json.Marshal(rows["precision_uint"])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(encoded, []byte(`"sample_values_status":"unavailable"`)) || !bytes.Contains(encoded, []byte(nvtelemetry.ReasonIntegerPrecision)) {
		t.Fatalf("precision refusal missing from public JSON summary: %s", encoded)
	}
}

func u64ptr(value uint64) *uint64 { return &value }
func i64ptr(value int64) *int64   { return &value }

func TestV3PrecisionStatusDoesNotClaimMetricUnavailable(t *testing.T) {
	const inexact = uint64(1)<<53 + 1
	metrics := []precisionMetric{{
		name: "inexact-with-gap", unit: "count",
		first: &nvtelemetry.Number{Uint: u64ptr(inexact)}, last: &nvtelemetry.Number{Uint: u64ptr(inexact + 2)}, kind: "gauge",
	}}
	samples, windows, _ := replayV3Precision(t, metrics)
	if len(samples) != 2 || len(windows["full_run"]) != 1 {
		t.Fatal("inexact sample values removed the metric row")
	}
	s := windows["full_run"][0]
	if s.SampleValuesStatus != nvtelemetry.WindowSampleValuesUnavailable || s.SampleCount != 2 || s.CoveredDurationNanos != 10 {
		t.Fatalf("sample-value refusal was confused with metric availability: %+v", s)
	}
	if s.SampleValuesReason != nvtelemetry.ReasonIntegerPrecision {
		t.Fatalf("sample-value reason = %q", s.SampleValuesReason)
	}
}

func TestV3CounterMetadataControlsEnergy(t *testing.T) {
	value := func(n uint64, epoch string, reset bool) nvtelemetry.Reading {
		return nvtelemetry.Reading{
			Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules,
			DeviceUUID: precisionDevice, Evidence: nvtelemetry.EvidenceMeasured,
			Number:      &nvtelemetry.Number{Uint: u64ptr(n)},
			Observation: &nvtelemetry.Observation{Kind: "counter", CounterEpoch: epoch, CounterReset: reset},
		}
	}
	unavailable := func(epoch string) nvtelemetry.Reading {
		return nvtelemetry.Reading{
			Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules,
			DeviceUUID: precisionDevice, Evidence: nvtelemetry.EvidenceUnavailable,
			UnavailableReason: nvtelemetry.ReasonUnsupported,
			Observation:       &nvtelemetry.Observation{Kind: "counter", CounterEpoch: epoch},
		}
	}
	tests := []struct {
		name   string
		points []nvtelemetry.Reading
		valid  bool
		reset  bool
		reason string
	}{
		{"zero-delta", []nvtelemetry.Reading{value(0, "boot-1", false), value(0, "boot-1", false)}, true, false, ""},
		{"counter-reset", []nvtelemetry.Reading{value(100, "boot-1", false), value(110, "boot-1", true)}, false, true, nvtelemetry.ReasonCounterReset},
		{"epoch-change", []nvtelemetry.Reading{value(100, "boot-1", false), value(110, "boot-2", false)}, false, true, nvtelemetry.ReasonCounterReset},
		{"missing-epoch", []nvtelemetry.Reading{value(100, "", false), value(110, "", false)}, false, false, "missing_counter_epoch"},
		{"unavailable-gap", []nvtelemetry.Reading{value(100, "boot-1", false), unavailable("boot-1"), value(110, "boot-1", false)}, false, false, nvtelemetry.ReasonGapExceeded},
		{"invalid-kind", []nvtelemetry.Reading{value(100, "boot-1", false), value(110, "boot-1", false)}, false, false, "invalid_counter_type"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			points := append([]nvtelemetry.Reading(nil), tt.points...)
			if tt.name == "invalid-kind" {
				points[0].Observation.Kind = "gauge"
				points[1].Observation.Kind = "gauge"
			}
			var stream bytes.Buffer
			writer := nvtelemetry.NewWriter(&stream, nvtelemetry.CollectorConfig{})
			manifest := nvtelemetry.Manifest{
				RunID: "v3-counter-metadata", SchemaVersion: nvtelemetry.SchemaVersionV3,
				Devices: []nvtelemetry.DeviceMetadata{{UUID: precisionDevice}}, StartTimeMonotonic: 100,
			}
			if err := writer.WriteManifest(&manifest); err != nil {
				t.Fatalf("write V3 manifest: %v", err)
			}
			for i := range points {
				sample := nvtelemetry.Sample{
					Seq: uint64(i + 1), QueryStartNanos: int64(100 + i*10), QueryEndNanos: int64(100 + i*10),
					Readings: []nvtelemetry.Reading{points[i]},
				}
				if err := writer.WriteSample(&sample); err != nil {
					t.Fatalf("write V3 sample %d: %v", i, err)
				}
			}
			manifest.TotalSamples = uint64(len(points))
			manifest.EndTimeMonotonic = int64(100 + (len(points)-1)*10)
			manifest.IsComplete = true
			if err := writer.WriteManifest(&manifest); err != nil {
				t.Fatalf("write V3 completion: %v", err)
			}
			_, _, energies, err := nvtelemetry.Replay(bytes.NewReader(stream.Bytes()), 0, nvtelemetry.WindowOptions{})
			if err != nil {
				t.Fatalf("replay V3 energy stream: %v", err)
			}
			if len(energies["full_run"]) != 1 {
				t.Fatalf("got %d V3 energy summaries, want 1", len(energies["full_run"]))
			}
			got := energies["full_run"][0]
			if got.Valid != tt.valid || got.ResetDetected != tt.reset || got.Reason != tt.reason {
				t.Fatalf("energy = %+v; want valid=%v reset=%v reason=%q", got, tt.valid, tt.reset, tt.reason)
			}
		})
	}
}
