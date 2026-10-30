package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

const v3Device = "GPU-v3window"

// v3Point is one V3 reading at a sample time. A nil Number with an empty
// Reason is an available reading without a scalar (for example histogram-only).
type v3Point struct {
	ts     int64
	num    *nvtelemetry.Number
	obs    *nvtelemetry.Observation
	reason string
}

func u64(v uint64) *nvtelemetry.Number  { return &nvtelemetry.Number{Uint: &v} }
func i64(v int64) *nvtelemetry.Number   { return &nvtelemetry.Number{Int: &v} }
func f64(v float64) *nvtelemetry.Number { return &nvtelemetry.Number{Float: &v} }

func counter(reset bool) *nvtelemetry.Observation {
	return &nvtelemetry.Observation{Kind: "counter", CounterEpoch: "boot-1", CounterReset: reset}
}

// replayV3 writes a V3 stream with the real Writer, then replays it with the
// public Replay consumer and returns the full_run summaries for metric.
func replayV3(t *testing.T, metric, unit string, points []v3Point) ([]nvtelemetry.WindowSummary, []nvtelemetry.EnergySummary) {
	t.Helper()
	var b bytes.Buffer
	w := nvtelemetry.NewWriter(&b, nvtelemetry.CollectorConfig{})
	m := nvtelemetry.Manifest{
		RunID:              "v3window",
		SchemaVersion:      nvtelemetry.SchemaVersionV3,
		Devices:            []nvtelemetry.DeviceMetadata{{UUID: v3Device}},
		StartTimeMonotonic: 100,
		EndTimeMonotonic:   400,
	}
	if err := w.WriteManifest(&m); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	for i, p := range points {
		r := nvtelemetry.Reading{Name: metric, Unit: unit, DeviceUUID: v3Device, Evidence: nvtelemetry.EvidenceMeasured, Number: p.num, Observation: p.obs}
		if p.reason != "" {
			r.Evidence = nvtelemetry.EvidenceUnavailable
			r.UnavailableReason = p.reason
		}
		s := nvtelemetry.Sample{Seq: uint64(i + 1), QueryStartNanos: p.ts, QueryEndNanos: p.ts, Readings: []nvtelemetry.Reading{r}}
		if err := w.WriteSample(&s); err != nil {
			t.Fatalf("write sample %d: %v", i, err)
		}
	}
	m.TotalSamples = uint64(len(points))
	m.IsComplete = true
	if err := w.WriteManifest(&m); err != nil {
		t.Fatalf("write final manifest: %v", err)
	}
	_, ws, es, err := nvtelemetry.Replay(&b, 0, nvtelemetry.WindowOptions{})
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	return ws["full_run"], es["full_run"]
}

func TestV3WindowGauge(t *testing.T) {
	tests := []struct {
		name     string
		points   []v3Point
		want     float64
		count    int
		coverage int64
	}{
		{"float-250mW", []v3Point{{ts: 150, num: f64(250)}, {ts: 250, num: f64(250)}}, 250, 2, 100},
		{"uint-250mW", []v3Point{{ts: 150, num: u64(250)}, {ts: 250, num: u64(250)}}, 250, 2, 100},
		{"int-250mW", []v3Point{{ts: 150, num: i64(250)}, {ts: 250, num: i64(250)}}, 250, 2, 100},
		{"measured-zero", []v3Point{{ts: 150, num: f64(0)}, {ts: 250, num: f64(0)}}, 0, 2, 100},
		// An unavailable reading between two available ones breaks coverage
		// and does not contribute a zero to the mean.
		{"unavailable-gap", []v3Point{{ts: 150, num: f64(250)}, {ts: 200, reason: "not_supported"}, {ts: 250, num: f64(250)}}, 250, 2, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ws, _ := replayV3(t, nvtelemetry.MetricPowerUsageMW, nvtelemetry.UnitMilliwatts, tt.points)
			if len(ws) != 1 {
				t.Fatalf("got %d summaries, want 1: %+v", len(ws), ws)
			}
			s := ws[0]
			if s.ArithmeticMean != tt.want || s.TimeWeightedMean != tt.want || s.Min != tt.want || s.Max != tt.want {
				t.Errorf("mean/tw/min/max = %v/%v/%v/%v, want %v", s.ArithmeticMean, s.TimeWeightedMean, s.Min, s.Max, tt.want)
			}
			if s.SampleCount != tt.count {
				t.Errorf("SampleCount = %d, want %d", s.SampleCount, tt.count)
			}
			if s.CoveredDurationNanos != tt.coverage {
				t.Errorf("CoveredDurationNanos = %d, want %d", s.CoveredDurationNanos, tt.coverage)
			}
		})
	}
}

// TestV3WindowGaugeAllUnavailable checks that readings with no scalar produce
// no summary rather than a measured zero.
func TestV3WindowGaugeAllUnavailable(t *testing.T) {
	ws, _ := replayV3(t, nvtelemetry.MetricPowerUsageMW, nvtelemetry.UnitMilliwatts, []v3Point{
		{ts: 150, reason: "not_supported"},
		{ts: 250, reason: "not_supported"},
	})
	if len(ws) != 0 {
		t.Fatalf("got summaries %+v, want none", ws)
	}
}

func TestV3WindowEnergy(t *testing.T) {
	const base = uint64(1)<<53 + 1
	hist := &nvtelemetry.Observation{Kind: "histogram", Histogram: &nvtelemetry.Histogram{Bounds: []string{"+Inf"}, Counts: []uint64{1}}}
	tests := []struct {
		name   string
		points []v3Point
		valid  bool
		delta  uint64
		reset  bool
		reason string
	}{
		{"exact-above-2^53", []v3Point{{ts: 150, num: u64(base), obs: counter(false)}, {ts: 250, num: u64(base + 10), obs: counter(false)}}, true, 10, false, ""},
		{"int-counter", []v3Point{{ts: 150, num: i64(1000), obs: counter(false)}, {ts: 250, num: i64(1010), obs: counter(false)}}, true, 10, false, ""},
		{"measured-zero-delta", []v3Point{{ts: 150, num: u64(0), obs: counter(false)}, {ts: 250, num: u64(0), obs: counter(false)}}, true, 0, false, ""},
		{"counter-reset", []v3Point{{ts: 150, num: u64(base), obs: counter(false)}, {ts: 250, num: u64(base + 10), obs: counter(true)}}, false, 0, true, nvtelemetry.ReasonCounterReset},
		{"epoch-change", []v3Point{{ts: 150, num: u64(5), obs: counter(false)}, {ts: 250, num: u64(15), obs: &nvtelemetry.Observation{Kind: "counter", CounterEpoch: "boot-2"}}}, false, 0, true, nvtelemetry.ReasonCounterReset},
		{"gauge-kind", []v3Point{{ts: 150, num: u64(5), obs: &nvtelemetry.Observation{Kind: "gauge"}}, {ts: 250, num: u64(15), obs: &nvtelemetry.Observation{Kind: "gauge"}}}, false, 0, false, "invalid_counter_type"},
		{"no-observation", []v3Point{{ts: 150, num: u64(5)}, {ts: 250, num: u64(15)}}, false, 0, false, "invalid_counter_type"},
		{"float-counter", []v3Point{{ts: 150, num: f64(5), obs: counter(false)}, {ts: 250, num: f64(15), obs: counter(false)}}, false, 0, false, "invalid_counter_type"},
		{"negative-end", []v3Point{{ts: 150, num: i64(5), obs: counter(false)}, {ts: 250, num: i64(-1), obs: counter(false)}}, false, 0, false, nvtelemetry.ReasonBlank},
		{"histogram-end", []v3Point{{ts: 150, num: u64(5), obs: counter(false)}, {ts: 250, obs: hist}}, false, 0, false, nvtelemetry.ReasonBlank},
		{"start-reset", []v3Point{{ts: 150, num: u64(5), obs: counter(true)}, {ts: 250, num: u64(15), obs: counter(false)}}, false, 0, true, nvtelemetry.ReasonCounterReset},
		{"decrease-across-unavailable", []v3Point{{ts: 150, num: u64(15), obs: counter(false)}, {ts: 200, reason: "not_supported"}, {ts: 250, num: u64(5), obs: counter(false)}}, false, 0, true, nvtelemetry.ReasonCounterReset},
		{"non-monotonic-middle", []v3Point{{ts: 150, num: u64(5), obs: counter(false)}, {ts: 200, num: u64(20), obs: counter(false)}, {ts: 250, num: u64(10), obs: counter(false)}}, false, 0, true, nvtelemetry.ReasonCounterReset},
		{"missing-epoch", []v3Point{{ts: 150, num: u64(5), obs: &nvtelemetry.Observation{Kind: "counter"}}, {ts: 250, num: u64(15), obs: &nvtelemetry.Observation{Kind: "counter"}}}, false, 0, false, "missing_counter_epoch"},
		{"unavailable-middle", []v3Point{{ts: 150, num: u64(5), obs: counter(false)}, {ts: 200, reason: "not_supported"}, {ts: 250, num: u64(15), obs: counter(false)}}, false, 0, false, nvtelemetry.ReasonGapExceeded},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, es := replayV3(t, nvtelemetry.MetricTotalEnergyMJ, nvtelemetry.UnitMillijoules, tt.points)
			if len(es) != 1 {
				t.Fatalf("got %d energy summaries, want 1: %+v", len(es), es)
			}
			e := es[0]
			if e.Valid != tt.valid || e.Reason != tt.reason || e.ResetDetected != tt.reset {
				t.Fatalf("Valid/Reason/Reset = %v/%q/%v, want %v/%q/%v (%+v)", e.Valid, e.Reason, e.ResetDetected, tt.valid, tt.reason, tt.reset, e)
			}
			if e.EnergyConsumedMJ != tt.delta {
				t.Errorf("EnergyConsumedMJ = %d, want %d", e.EnergyConsumedMJ, tt.delta)
			}
			if tt.name == "exact-above-2^53" && (e.StartEnergyMJ != base || e.EndEnergyMJ != base+10) {
				t.Errorf("Start/End = %d/%d, want %d/%d", e.StartEnergyMJ, e.EndEnergyMJ, base, base+10)
			}
		})
	}
}

// TestV2ReplayUnchanged pins the replay output of the canonical V1 and V2
// fixtures to digests recorded before V3 window support.
func TestV2ReplayUnchanged(t *testing.T) {
	tests := []struct {
		file string
		want string
	}{
		{"canonical_v1.jsonl", "c0ad0477d935ea827f8c2a05470afaaa0f021b3ebe9c3e4b3954c719d023a4ec"},
		{"canonical_v2.jsonl", "84f8225a200c445dd54261caf1e7ae355124b920bb2069ef386cc1e557a1e297"},
	}
	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			f, err := os.Open(filepath.Join("testdata", tt.file))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			_, ws, es, err := nvtelemetry.Replay(f, 0, nvtelemetry.WindowOptions{})
			if err != nil {
				t.Fatalf("replay: %v", err)
			}
			if len(ws) == 0 {
				t.Fatal("no window summaries")
			}
			b, err := json.Marshal(struct {
				W map[string][]nvtelemetry.WindowSummary
				E map[string][]nvtelemetry.EnergySummary
			}{ws, es})
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(b)
			if got := hex.EncodeToString(sum[:]); got != tt.want {
				t.Errorf("digest = %s, want %s\n%s", got, tt.want, b)
			}
		})
	}
}

// TestV2WindowEnergyUnchanged covers the legacy IntValue energy path, which
// the canonical fixtures do not exercise.
func TestV2WindowEnergyUnchanged(t *testing.T) {
	const base = uint64(1)<<53 + 1
	tests := []struct {
		name   string
		values []uint64
		valid  bool
		delta  uint64
		reason string
	}{
		{"exact-above-2^53", []uint64{base, base + 10}, true, 10, ""},
		{"decrease-is-reset", []uint64{base + 10, base}, false, 0, nvtelemetry.ReasonCounterReset},
		{"equal-is-zero-delta", []uint64{base, base, base}, true, 0, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var samples []nvtelemetry.Sample
			for i, v := range tt.values {
				samples = append(samples, nvtelemetry.Sample{Seq: uint64(i + 1), QueryStartNanos: int64(150 + 100*i), Readings: []nvtelemetry.Reading{{
					Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules, DeviceUUID: v3Device,
					Evidence: nvtelemetry.EvidenceMeasured, IsInt: true, IntValue: v, Value: float64(v),
				}}})
			}
			_, es := nvtelemetry.AggregateWindow("w", 100, 400, samples, nvtelemetry.WindowOptions{})
			if len(es) != 1 {
				t.Fatalf("got %d energy summaries, want 1", len(es))
			}
			e := es[0]
			if e.Valid != tt.valid || e.Reason != tt.reason || e.EnergyConsumedMJ != tt.delta {
				t.Errorf("Valid/Reason/Delta = %v/%q/%d, want %v/%q/%d", e.Valid, e.Reason, e.EnergyConsumedMJ, tt.valid, tt.reason, tt.delta)
			}
		})
	}
}
