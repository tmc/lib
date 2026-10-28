package cudadiag_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func mixedV5Protocol() ([]nvtelemetry.Reading, cudadiag.TelemetryOptions) {
	r, opt := mixedProtocol()
	setProtocolSamplingCoverage(r, &opt, &nvtelemetry.SamplingCoverage{Status: "unknown", Basis: "time", Scope: "activity-window-v1"})
	return r, opt
}

func TestTelemetryV5VersionOnly(t *testing.T) {
	r, opt := mixedV5Protocol()
	var previous cudadiag.TelemetryDiagnosis
	for _, version := range []int{nvtelemetry.SchemaVersionV4, nvtelemetry.SchemaVersionV5} {
		raw := collectMixedVersion(t, r, opt, version)
		m, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0)
		if err != nil || m.SchemaVersion != version || len(samples) != 1 {
			t.Fatalf("strict version %d: %v", version, err)
		}
		got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
		if err != nil || !got.Diagnosis.Admitted {
			t.Fatalf("version %d: %+v, %v", version, got.Diagnosis, err)
		}
		for _, finding := range got.Observations {
			if !finding.Finding.Admission.Admitted || finding.Finding.Class != tracecontract.FindingAssociated {
				t.Fatalf("version %d: %+v", version, finding.Finding)
			}
		}
		if version == nvtelemetry.SchemaVersionV5 && !reflect.DeepEqual(previous, got) {
			t.Fatal("version alone changed diagnosis")
		}
		previous = got
	}
}

func TestTelemetryV5ExactValues(t *testing.T) {
	r, opt := mixedV5Protocol()
	u, i := uint64(1<<53+1), int64(math.MinInt64)
	r[len(r)-3].Number = &nvtelemetry.Number{Uint: &u}
	r[len(r)-2].Number = &nvtelemetry.Number{Int: &i}
	r[len(r)-1].Number = nil
	r[len(r)-1].Evidence = nvtelemetry.EvidenceUnavailable
	r[len(r)-1].UnavailableReason = "not collected"
	raw := collectMixedVersion(t, r, opt, nvtelemetry.SchemaVersionV5)
	_, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Observations) != len(r) {
		t.Fatal("reading lost")
	}
	for j, finding := range got.Observations {
		if !reflect.DeepEqual(finding.Reading, samples[0].Readings[j]) {
			t.Fatal("exact reading changed")
		}
		if j >= len(r)-3 && finding.Finding.Value != nil {
			t.Fatal("integer/unavailable converted to float")
		}
	}
	if *got.Observations[len(r)-3].Reading.Number.Uint != u || *got.Observations[len(r)-2].Reading.Number.Int != i {
		t.Fatal("integer precision lost")
	}
	if got.Observations[len(r)-1].Finding.Admission.Admitted {
		t.Fatal("unavailable admitted")
	}
}

func v5Histogram(r []nvtelemetry.Reading, opt *cudadiag.TelemetryOptions) *nvtelemetry.Reading {
	last := &r[len(r)-1]
	n := uint64(1<<53 + 1)
	last.Number = nil
	last.Observation.Kind = "histogram"
	last.Observation.Histogram = &nvtelemetry.Histogram{Bounds: []string{"1", "+Inf"}, Counts: []uint64{0, n}, Count: &n, Temporality: "delta"}
	opt.HistogramBounds = map[string][]string{last.Name: {"1", "+Inf"}}
	return last
}

func TestTelemetryV5HistogramAdmission(t *testing.T) {
	cases := []struct {
		name, reason string
		edit         func(*nvtelemetry.Reading, *cudadiag.TelemetryOptions)
	}{
		{"delta", "", func(*nvtelemetry.Reading, *cudadiag.TelemetryOptions) {}},
		{"cumulative", "", func(r *nvtelemetry.Reading, _ *cudadiag.TelemetryOptions) {
			r.Observation.Histogram.Temporality = "cumulative"
			r.Observation.CounterEpoch = "epoch-1"
		}},
		{"cumulative-no-epoch", "histogram epoch", func(r *nvtelemetry.Reading, _ *cudadiag.TelemetryOptions) {
			r.Observation.Histogram.Temporality = "cumulative"
		}},
		{"unknown-boundaries", "histogram boundary", func(_ *nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { o.HistogramBounds = nil }},
		{"different-boundaries", "histogram boundary", func(r *nvtelemetry.Reading, o *cudadiag.TelemetryOptions) {
			o.HistogramBounds[r.Name] = []string{"2", "+Inf"}
		}},
		{"summary-only", "histogram boundary", func(r *nvtelemetry.Reading, o *cudadiag.TelemetryOptions) {
			r.Observation.Histogram.Bounds = nil
			r.Observation.Histogram.Counts = nil
			o.HistogramBounds = nil
		}},
		{"empty-delta-interval", "histogram interval", func(r *nvtelemetry.Reading, _ *cudadiag.TelemetryOptions) {
			span := *r.Observation.Interval
			span.Start = span.End
			r.Observation.Interval = &span
		}},
		{"unknown-OTLP-clock", "OTLP clock mapping", func(r *nvtelemetry.Reading, _ *cudadiag.TelemetryOptions) {
			r.Observation.OTLPClock = &nvtelemetry.OTLPClockMetadata{}
		}},
		{"asserted-OTLP-clock", "OTLP clock mapping", func(r *nvtelemetry.Reading, _ *cudadiag.TelemetryOptions) {
			r.Observation.OTLPClock = &nvtelemetry.OTLPClockMetadata{ClockDomain: "monotonic", HasClockCalibration: true}
		}},
		{"no-recorded", "unavailable reading", func(r *nvtelemetry.Reading, _ *cudadiag.TelemetryOptions) {
			z := uint64(0)
			r.Evidence = nvtelemetry.EvidenceUnavailable
			r.UnavailableReason = "no_recorded_value"
			r.Observation.Histogram = &nvtelemetry.Histogram{Count: &z, Temporality: "delta", Flags: 1}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, opt := mixedV5Protocol()
			h := v5Histogram(r, &opt)
			tc.edit(h, &opt)
			raw := collectMixedVersion(t, r, opt, nvtelemetry.SchemaVersionV5)
			if _, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0); err != nil {
				t.Fatal(err)
			}
			got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
			if err != nil {
				t.Fatal(err)
			}
			finding := got.Observations[len(r)-1]
			if !reflect.DeepEqual(finding.Reading.Observation.Histogram, h.Observation.Histogram) {
				t.Fatal("histogram changed")
			}
			if tc.reason == "" {
				if !finding.Finding.Admission.Admitted || finding.Finding.Value != nil {
					t.Fatalf("histogram not retained: %+v", finding)
				}
			} else if finding.Finding.Admission.Admitted || !strings.Contains(finding.Finding.Admission.Reason, tc.reason) {
				t.Fatalf("wrong admission: %+v", finding.Finding)
			}
			if !got.Diagnosis.Admitted {
				t.Fatal("unrelated histogram erased host/device diagnosis")
			}
		})
	}
}

func rewriteV5(t *testing.T, raw []byte, edit func(*nvtelemetry.StreamRecord)) []byte {
	t.Helper()
	var b bytes.Buffer
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		var record nvtelemetry.StreamRecord
		if err := json.Unmarshal(line, &record); err != nil {
			t.Fatal(err)
		}
		edit(&record)
		if err := json.NewEncoder(&b).Encode(record); err != nil {
			t.Fatal(err)
		}
	}
	return b.Bytes()
}

func TestTelemetryV5IndependentAdmission(t *testing.T) {
	cases := []struct {
		name, want string
		edit       func(*nvtelemetry.StreamRecord, *cudadiag.TelemetryOptions)
	}{
		{"identity", "workload identity", func(rec *nvtelemetry.StreamRecord, _ *cudadiag.TelemetryOptions) {
			if rec.Sample != nil {
				rec.Sample.Readings[0].Observation.Workload = nil
			}
		}},
		{"clock", "missing epoch", func(rec *nvtelemetry.StreamRecord, _ *cudadiag.TelemetryOptions) {
			if rec.Sample != nil {
				rec.Sample.Readings[0].Observation.ClockEpoch = ""
			}
		}},
		{"source-class", "source provenance", func(rec *nvtelemetry.StreamRecord, _ *cudadiag.TelemetryOptions) {
			if rec.Sample != nil {
				rec.Sample.Readings[0].Observation.SourceClass = "native"
			}
		}},
		{"receipt", "independent trace receipt", func(_ *nvtelemetry.StreamRecord, o *cudadiag.TelemetryOptions) { o.Receipt = nil }},
		{"raw", "corrupt raw", func(_ *nvtelemetry.StreamRecord, o *cudadiag.TelemetryOptions) {
			for k := range o.RawArtifacts {
				o.RawArtifacts[k] = []byte("corrupt")
			}
		}},
		{"loss", "unassessed", func(rec *nvtelemetry.StreamRecord, _ *cudadiag.TelemetryOptions) {
			if rec.Manifest != nil && rec.Manifest.IsComplete {
				v := rec.Manifest.LostRecordsByFamily["host"]
				v.Unknown = true
				rec.Manifest.LostRecordsByFamily["host"] = v
			}
		}},
		{"denominator", "denominator", func(rec *nvtelemetry.StreamRecord, _ *cudadiag.TelemetryOptions) {
			if rec.Sample != nil {
				rec.Sample.Readings[0].Evidence = nvtelemetry.EvidenceDerived
			}
		}},
		{"counter-epoch", "counter epoch", func(rec *nvtelemetry.StreamRecord, _ *cudadiag.TelemetryOptions) {
			if rec.Sample != nil {
				rec.Sample.Readings[0].Observation.Kind = "counter"
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, opt := mixedV5Protocol()
			raw := collectMixedVersion(t, r, opt, nvtelemetry.SchemaVersionV5)
			raw = rewriteV5(t, raw, func(rec *nvtelemetry.StreamRecord) { tc.edit(rec, &opt) })
			got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Observations) != len(r) || got.Diagnosis.Admitted || !strings.Contains(got.Observations[0].Finding.Admission.Reason, tc.want) {
				t.Fatalf("mutation admitted or lost: %+v", got)
			}
		})
	}
}

func TestTelemetryV5ReaderRefusals(t *testing.T) {
	cases := []struct {
		name string
		edit func(*nvtelemetry.Reading)
	}{
		{"uint-overflow", func(*nvtelemetry.Reading) {}},
		{"partial-clock-object", func(r *nvtelemetry.Reading) { r.Observation.OTLPClock = &nvtelemetry.OTLPClockMetadata{} }},
		{"missing-count", func(r *nvtelemetry.Reading) { r.Observation.Histogram.Count = nil }},
		{"unknown-temporality", func(r *nvtelemetry.Reading) { r.Observation.Histogram.Temporality = "unknown" }},
		{"unknown-flags", func(r *nvtelemetry.Reading) { r.Observation.Histogram.Flags = 2 }},
		{"partial-source-time", func(r *nvtelemetry.Reading) {
			r.Observation.OTLPSourceTime = &nvtelemetry.OTLPSourceTime{SourceClass: "native"}
		}},
		{"mapping-overflow", func(r *nvtelemetry.Reading) {
			r.Observation.SourceClass = "imported"
			r.Observation.ClockEpoch = "unix"
			r.TimestampMonotonic = 0
			r.TimestampUnix = 20
			r.Observation.Interval = nil
			scope := ""
			r.Observation.OTLPScopeName = &scope
			r.Observation.OTLPClock = &nvtelemetry.OTLPClockMetadata{}
			r.Observation.OTLPSourceTime = &nvtelemetry.OTLPSourceTime{SourceClass: "synthetic", SourceClockEpoch: "epoch", SourceMonotonicEndNanos: math.MaxInt64, MappingDomain: "monotonic", MappingEpoch: "epoch", MappingValidUntilNanos: math.MaxInt64, MappingUnixOffsetNanos: 1, MappingProvenance: "synthetic overflow control"}
		}},
		{"nonfinite-summary", func(r *nvtelemetry.Reading) { v := math.MaxFloat64; r.Observation.Histogram.Sum = &v }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, opt := mixedV5Protocol()
			v5Histogram(r, &opt)
			raw := collectMixedVersion(t, r, opt, nvtelemetry.SchemaVersionV5)
			raw = rewriteV5(t, raw, func(rec *nvtelemetry.StreamRecord) {
				if rec.Sample != nil {
					tc.edit(&rec.Sample.Readings[len(r)-1])
				}
			})
			if tc.name == "uint-overflow" {
				raw = bytes.ReplaceAll(raw, []byte("9007199254740993"), []byte("18446744073709551616"))
			}
			if tc.name == "partial-clock-object" {
				raw = bytes.ReplaceAll(raw, []byte(`"otlp_clock_metadata":{"clock_domain":"","has_clock_calibration":false,"clock_calibration_bounds_nanos":0}`), []byte(`"otlp_clock_metadata":{"clock_domain":""}`))
			}
			if tc.name == "nonfinite-summary" {
				raw = bytes.ReplaceAll(raw, []byte("1.7976931348623157e+308"), []byte("1e999"))
			}
			if _, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0); err == nil {
				t.Fatal("malformed V5 accepted by reader")
			}
			if _, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt); err == nil {
				t.Fatal("malformed V5 accepted by consumer")
			}
		})
	}
	r, opt := mixedV5Protocol()
	raw := collectMixedVersion(t, r, opt, nvtelemetry.SchemaVersionV5)
	future := rewriteV5(t, raw, func(rec *nvtelemetry.StreamRecord) {
		if rec.Manifest != nil {
			rec.Manifest.SchemaVersion = nvtelemetry.SchemaVersionV5 + 1
		}
	})
	for _, data := range [][]byte{future} {
		if _, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(data), 0); !errors.Is(err, nvtelemetry.ErrUnsupportedSchema) {
			t.Fatalf("future reader: %v", err)
		}
		if _, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(data), opt); !errors.Is(err, nvtelemetry.ErrUnsupportedSchema) {
			t.Fatalf("future consumer: %v", err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cudadiag.DiagnoseTelemetry(ctx, bytes.NewReader(raw), opt); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	opt.MaxBytes = int64(len(raw) - 1)
	if _, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt); !errors.Is(err, nvtelemetry.ErrBudgetExceeded) {
		t.Fatal(err)
	}
}
