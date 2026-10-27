package nvtelemetry

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func logEventFixture(name string) Reading {
	one := uint64(1)
	zero := int64(0)
	return Reading{Name: name, Unit: "code", DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, TimestampUnix: 1000, Number: &Number{Uint: &one}, Observation: &Observation{Kind: "event", SourceClass: "synthetic", Producer: "fixture", ProducerVersion: "1", ProducerSchema: "supplied.v1", ClockEpoch: "boot", Interval: &tracecontract.Interval{Clock: "monotonic", Start: 25, End: 25}, UncertaintyNS: &zero}}
}
func logEventEndFixture() Reading {
	r := logEventFixture("job_end_event")
	a := entityAssignment()
	a.Owner = EntityOwner{JobID: "job-a", AllocationID: "alloc-a", Attempt: "1"}
	a.Provenance = []EntityProvenance{{Source: "supplied", Revision: "1", StartNanos: 0, EndNanos: 100}}
	r.Observation.Assignment = &a
	r.Observation.Scope = "process"
	r.Observation.Interval.Start = 100
	r.Observation.Interval.End = 100
	r.Observation.Workload = &cudaworkload.WorkloadIdentity{Process: a.Process.Process}
	return r
}
func logEventEncoded(t *testing.T, r Reading, o ExportOptions) ([]byte, map[string]any, map[string]any) {
	t.Helper()
	var b bytes.Buffer
	if err := WriteOTLPLogs(&b, []Reading{r}, o); err != nil {
		t.Fatal(err)
	}
	groups := decodeEntityOTLP(t, b.Bytes(), "resourceLogs")
	attrs := entityResourceAttrs(t, groups[0])
	scope := groups[0].(map[string]any)["scopeLogs"].([]any)
	record := scope[0].(map[string]any)["logRecords"].([]any)[0].(map[string]any)
	return b.Bytes(), attrs, record
}
func logEventAttr(record map[string]any, key string) any {
	for _, a := range record["attributes"].([]any) {
		m := a.(map[string]any)
		if m["key"] == "nvtelemetry.event."+key {
			return m["value"]
		}
	}
	return nil
}
func TestSuppliedLogEventCategories(t *testing.T) {
	for _, name := range []string{"ecc_error_event", "job_start_event", "job_end_event", "device_reset_event"} {
		t.Run(name, func(t *testing.T) {
			r := logEventFixture(name)
			raw, _, record := logEventEncoded(t, r, ExportOptions{})
			if !bytes.Contains(raw, []byte("nvtelemetry.logs.events.v1")) || !bytes.Contains(raw, []byte(`"value_uint64"`)) {
				t.Fatal(string(raw))
			}
			if record["timeUnixNano"] != nil || record["observedTimeUnixNano"] != "1000" {
				t.Fatal(record)
			}
			for _, write := range []func(*bytes.Buffer, []Reading, ExportOptions) error{func(b *bytes.Buffer, r []Reading, o ExportOptions) error { return WriteOTLPMetrics(b, r, o) }, func(b *bytes.Buffer, r []Reading, o ExportOptions) error { return WriteExposition(b, r, o) }} {
				var b bytes.Buffer
				if err := write(&b, []Reading{r}, ExportOptions{}); err == nil || b.Len() != 0 {
					t.Fatal("event emitted as numeric metric", err)
				}
			}
		})
	}
	old := Reading{Name: MetricXIDEvent, Unit: "code", DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, IsInt: true, IntValue: 43, TimestampUnix: 1000}
	raw, _, rec := logEventEncoded(t, old, ExportOptions{})
	if !bytes.Contains(raw, []byte("nvtelemetry.logs.v1")) || rec["timeUnixNano"] != "1000" || rec["observedTimeUnixNano"] != nil {
		t.Fatal(string(raw))
	}
}
func TestSuppliedLogEventRefusals(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*Reading)
	}{
		{"wrong kind", func(r *Reading) { r.Observation.Kind = "gauge" }},
		{"unclosed name", func(r *Reading) { r.Name = MetricPowerUsageMW; r.Unit = UnitMilliwatts }},
		{"missing observation", func(r *Reading) { r.Observation = nil }},
		{"wrong unit", func(r *Reading) { r.Unit = "count" }},
		{"zero", func(r *Reading) { *r.Number.Uint = 0 }}, {"two", func(r *Reading) { *r.Number.Uint = 2 }}, {"max", func(r *Reading) { *r.Number.Uint = math.MaxUint64 }},
		{"signed", func(r *Reading) { v := int64(1); r.Number = &Number{Int: &v} }}, {"float", func(r *Reading) { v := float64(1); r.Number = &Number{Float: &v} }},
		{"two arms", func(r *Reading) { v := int64(1); r.Number.Int = &v }}, {"legacy", func(r *Reading) { r.IsInt = true; r.IntValue = 1 }},
		{"derived", func(r *Reading) { r.Evidence = EvidenceDerived }}, {"modeled", func(r *Reading) { r.Observation.SourceClass = "modeled" }},
		{"histogram", func(r *Reading) { r.Observation.Histogram = &Histogram{} }}, {"denominator", func(r *Reading) { r.Observation.Denominator = &Number{} }},
		{"counter", func(r *Reading) { r.Observation.CounterEpoch = "1" }}, {"reset", func(r *Reading) { r.Observation.CounterReset = true }},
		{"OTLP provenance", func(r *Reading) { s := "x"; r.Observation.OTLPScopeName = &s }},
		{"nonpoint", func(r *Reading) { r.Observation.Interval.End++ }}, {"negative", func(r *Reading) { r.Observation.Interval.Start = -1; r.Observation.Interval.End = -1 }},
		{"negative uncertainty", func(r *Reading) { *r.Observation.UncertaintyNS = -1 }},
		{"duplicate timestamp", func(r *Reading) {
			r.HasSourceTimestamp = true
			r.SourceTimestamp = 26
			r.SourceTimestampDomain = "monotonic"
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := logEventFixture("job_end_event")
			tc.edit(&r)
			var b bytes.Buffer
			if err := WriteOTLPLogs(&b, []Reading{r}, ExportOptions{}); err == nil || b.Len() != 0 {
				t.Fatalf("admitted malformed event: %s %v", b.String(), err)
			}
			var stream bytes.Buffer
			w := NewWriter(&stream, CollectorConfig{})
			if err := w.WriteManifest(&Manifest{SchemaVersion: 3}); err != nil {
				t.Fatal(err)
			}
			if err := w.WriteSample(&Sample{Seq: 1, Readings: []Reading{r}}); err == nil {
				t.Fatal("stream admitted mismatch")
			}
		})
	}
	// A category/kind mismatch also refuses on direct numeric paths.
	r := logEventFixture("job_end_event")
	r.Observation.Kind = "gauge"
	var b bytes.Buffer
	if err := WriteOTLPMetrics(&b, []Reading{r}, ExportOptions{}); err == nil {
		t.Fatal("metrics admitted category mismatch")
	}
	r = logEventFixture("job_end_event")
	r.Evidence = EvidenceUnavailable
	r.UnavailableReason = "denied"
	r.Number = nil
	raw, attrs, _ := logEventEncoded(t, r, ExportOptions{})
	if bytes.Contains(raw, []byte("value_uint64")) || entityAttrText(attrs, "subject.owner.job_id") != "" {
		t.Fatal(string(raw))
	}
}
func TestSuppliedLogEventStream(t *testing.T) {
	for _, version := range []int{3, 4, 5} {
		t.Run(string(rune('0'+version)), func(t *testing.T) {
			r := logEventFixture("ecc_error_event")
			if version >= 4 {
				r.Observation.Observer.SamplingCoverage = &SamplingCoverage{Status: "unknown"}
			}
			var b bytes.Buffer
			w := NewWriter(&b, CollectorConfig{})
			m := Manifest{RunID: "events", SchemaVersion: version, Devices: []DeviceMetadata{{UUID: "GPU-a"}}}
			if err := w.WriteManifest(&m); err != nil {
				t.Fatal(err)
			}
			if err := w.WriteSample(&Sample{Seq: 1, Readings: []Reading{r}}); err != nil {
				t.Fatal(err)
			}
			m.IsComplete = true
			m.TotalSamples = 1
			if err := w.WriteManifest(&m); err != nil {
				t.Fatal(err)
			}
			_, _, samples, err := ReadStreamStrict(&b, 0)
			if err != nil || len(samples) != 1 || !reflect.DeepEqual(samples[0].Readings[0].Number, r.Number) {
				t.Fatal(samples, err)
			}
		})
	}
	for _, v := range []int{1, 2} {
		r := logEventFixture("job_end_event")
		r.Number = nil
		r.Observation = nil
		var b bytes.Buffer
		w := NewWriter(&b, CollectorConfig{})
		if err := w.WriteManifest(&Manifest{SchemaVersion: v}); err != nil {
			t.Fatal(err)
		}
		if err := w.WriteSample(&Sample{Readings: []Reading{r}}); err == nil {
			t.Fatal("legacy category downgrade", v)
		}
	}
}
func TestSuppliedLogEventEndBoundary(t *testing.T) {
	r := logEventEndFixture()
	before, _ := json.Marshal(r)
	raw, attrs, _ := logEventEncoded(t, r, ExportOptions{})
	after, _ := json.Marshal(r)
	if !bytes.Equal(before, after) {
		t.Fatal("input changed")
	}
	if entityAttrText(attrs, "assignment.status") != "ended_boundary" || entityAttrText(attrs, "subject.owner.job_id") != "job-a" || entityAttrText(attrs, "owner.job_id") != "" || !bytes.Contains(raw, []byte(`"nvtelemetry.assignment.end_nanos"`)) {
		t.Fatal(string(raw))
	}
	if entityAttrText(attrs, "process.node") != "node" || attrs["nvtelemetry.assignment.provenance"] == nil {
		t.Fatal("lost original process/provenance", attrs)
	}
	rel, err := EntityRelationships([]EntityAssignment{*r.Observation.Assignment}, "GPU-a", "monotonic", "boot", 100)
	if err != nil || len(rel) != 0 {
		t.Fatal(rel, err)
	}
	ordinary := entityExportReading(12, "job-a", 1)
	ordinary.Observation.Interval.Start = 100
	ordinary.Observation.Interval.End = 100
	var b bytes.Buffer
	if err := WriteOTLPMetrics(&b, []Reading{ordinary}, ExportOptions{}); err == nil {
		t.Fatal("metric end widened")
	}
	if err := WriteOTLPLogs(&b, []Reading{ordinary}, ExportOptions{}); err == nil {
		t.Fatal("ordinary log end widened")
	}
	for _, tc := range []struct {
		name string
		edit func(*Reading)
	}{
		{"before boundary", func(r *Reading) { r.Observation.Interval.Start--; r.Observation.Interval.End-- }},
		{"after boundary", func(r *Reading) { r.Observation.Interval.Start++; r.Observation.Interval.End++ }},
		{"missing producer", func(r *Reading) { r.Observation.Producer = "" }},
		{"missing producer version", func(r *Reading) { r.Observation.ProducerVersion = "" }},
		{"missing producer schema", func(r *Reading) { r.Observation.ProducerSchema = "" }},
		{"missing source class", func(r *Reading) { r.Observation.SourceClass = "" }},
		{"unknown uncertainty", func(r *Reading) { r.Observation.UncertaintyNS = nil }}, {"nonzero uncertainty", func(r *Reading) { *r.Observation.UncertaintyNS = 1 }},
		{"incomplete process", func(r *Reading) { r.Observation.Assignment.Process.BootID = "" }}, {"unverified", func(r *Reading) { r.Observation.Assignment.Authoritative = false }},
		{"missing binding", func(r *Reading) { r.Observation.Workload = nil }}, {"missing provenance", func(r *Reading) { r.Observation.Assignment.Provenance = nil }},
		{"unknown time", func(r *Reading) { r.Observation.Interval = nil }}, {"device scope", func(r *Reading) { r.Observation.Scope = "device" }},
		{"unavailable", func(r *Reading) { r.Evidence = EvidenceUnavailable; r.UnavailableReason = "denied"; r.Number = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := logEventEndFixture()
			tc.edit(&r)
			raw, attrs, _ := logEventEncoded(t, r, ExportOptions{})
			if entityAttrText(attrs, "assignment.status") == "ended_boundary" || bytes.Contains(raw, []byte("subject.owner.")) || bytes.Contains(raw, []byte(`"nvtelemetry.owner.`)) || bytes.Contains(raw, []byte(`"nvtelemetry.process.`)) {
				t.Fatal(string(raw))
			}
		})
	}
	_, attrs, _ = logEventEncoded(t, logEventEndFixture(), ExportOptions{ClockDomain: "monotonic", HasClockCalibration: true, ClockCalibrationBoundsNanos: 1})
	if entityAttrText(attrs, "assignment.status") == "ended_boundary" {
		t.Fatal("nonzero calibration discarded")
	}
	r = logEventEndFixture()
	r.Name = "job_start_event"
	if err := WriteOTLPLogs(&b, []Reading{r}, ExportOptions{}); err == nil {
		t.Fatal("start admitted at end")
	}
}
func TestSuppliedLogEventStructuralParity(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*Reading)
	}{
		{"contradictory assignment", func(r *Reading) { r.Observation.Assignment.State = "contradictory" }},
		{"generation", func(r *Reading) { r.Observation.Workload.Process.Generation++ }}, {"UUID", func(r *Reading) { r.Observation.Assignment.Device.UUID = "GPU-other" }},
		{"clock", func(r *Reading) { r.Observation.ClockEpoch = "other" }}, {"NSPID", func(r *Reading) { r.Observation.Assignment.Process.NamespacePIDs = make([]uint32, 33) }},
		{"process text", func(r *Reading) { r.Observation.Assignment.Process.NodeID = "bad\n" }}, {"owner text", func(r *Reading) { r.Observation.Assignment.Owner.JobID = "bad\n" }},
		{"MIG contradiction", func(r *Reading) { r.Observation.Assignment.Device.MIGInstanceID = 1 }},
		{"MPS conflict", func(r *Reading) {
			a := r.Observation.Assignment
			a.Sharing = "mps"
			a.MPSRole = "client"
			p := a.Process
			p.BootID = "other"
			a.MPSServer = &p
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, withheld := range []bool{false, true} {
				r := logEventEndFixture()
				if withheld {
					r.Observation.UncertaintyNS = nil
					r.Observation.Scope = "device"
				}
				tc.edit(&r)
				var b bytes.Buffer
				if err := WriteOTLPLogs(&b, []Reading{r}, ExportOptions{}); err == nil || b.Len() != 0 {
					t.Fatalf("contradiction bypassed (withheld=%v): %s", withheld, b.String())
				}
			}
		})
	}
	// A valid current ordinary sample keeps the legacy owner and schema.
	r := entityExportReading(12, "job-a", 1)
	raw, attrs, _ := logEventEncoded(t, r, ExportOptions{})
	if entityAttrText(attrs, "owner.job_id") != "job-a" || !bytes.Contains(raw, []byte("nvtelemetry.logs.v1")) {
		t.Fatal(string(raw))
	}
}
func TestSuppliedLogEventSourceTime(t *testing.T) {
	r := logEventFixture("ecc_error_event")
	r.Observation.Interval = nil
	r.Observation.UncertaintyNS = nil
	_, _, rec := logEventEncoded(t, r, ExportOptions{})
	if rec["timeUnixNano"] != nil || rec["observedTimeUnixNano"] != "1000" || !reflect.DeepEqual(logEventAttr(rec, "source_time_known"), map[string]any{"boolValue": false}) {
		t.Fatal(rec)
	}
	r.TimestampUnix = 0
	_, _, rec = logEventEncoded(t, r, ExportOptions{})
	if rec["observedTimeUnixNano"] != nil {
		t.Fatal(rec)
	}
	r = logEventFixture("ecc_error_event")
	m := UnixTimeMapping{SourceClockDomain: "monotonic", SourceClockEpoch: "boot", SourceValidFromNanos: 0, SourceValidUntilNanos: 100, UnixOffsetNanos: 2000, Provenance: "supplied"}
	_, _, rec = logEventEncoded(t, r, ExportOptions{UnixTimeMapping: &m})
	if rec["timeUnixNano"] != "2025" || rec["observedTimeUnixNano"] != "1000" {
		t.Fatal(rec)
	}
	for _, edit := range []func(*UnixTimeMapping){func(m *UnixTimeMapping) { m.SourceClockEpoch = "wrong" }, func(m *UnixTimeMapping) { m.SourceValidFromNanos = 26 }, func(m *UnixTimeMapping) { m.Provenance = "" }, func(m *UnixTimeMapping) { m.UnixOffsetNanos = math.MaxInt64 }} {
		bad := m
		edit(&bad)
		var b bytes.Buffer
		if err := WriteOTLPLogs(&b, []Reading{r}, ExportOptions{UnixTimeMapping: &bad}); err == nil || b.Len() != 0 {
			t.Fatal("bad mapping admitted")
		}
	}
	var b bytes.Buffer
	if err := WriteOTLPLogs(&b, []Reading{r}, ExportOptions{MaxBytes: 10}); err == nil || b.Len() != 0 {
		t.Fatal("byte bound ignored")
	}
	if strings.Contains(stringMustJSON(r), "logs.events.v1") {
		t.Fatal("source was converted to export")
	}
}
func stringMustJSON(v any) string { b, _ := json.Marshal(v); return string(b) }

func logEventSupplierFixture(r Reading) Reading {
	r.Backend = "legacy-backend"
	r.Observation.SourceClass = "synthetic"
	r.Observation.Producer = "supplied-observer"
	r.Observation.ProducerVersion = "supplier-version"
	r.Observation.ProducerSchema = "supplier-schema"
	r.Observation.Observer = Observer{Population: "supplied-population", Mode: "supplied-mode", Pass: "supplied-pass", Filter: "supplied-filter", Replay: "supplied-replay"}
	return r
}

func assertLogEventSupplier(t *testing.T, attrs map[string]any, r Reading) {
	t.Helper()
	o := r.Observation
	for _, p := range [][2]string{{"scope", o.Scope}, {"source_class", o.SourceClass}, {"producer", o.Producer}, {"producer_version", o.ProducerVersion}, {"producer_schema", o.ProducerSchema}, {"population", o.Observer.Population}, {"mode", o.Observer.Mode}, {"pass", o.Observer.Pass}, {"filter", o.Observer.Filter}, {"replay", o.Observer.Replay}} {
		if got := entityAttrText(attrs, "observation."+p[0]); got != p[1] {
			t.Fatalf("lost observation supplier %s: got %q, want %q", p[0], got, p[1])
		}
	}
}

func TestSuppliedLogEventSupplierProjection(t *testing.T) {
	for _, tc := range []struct {
		name    string
		reading func() Reading
		owner   bool
	}{
		{"ended", logEventEndFixture, true},
		{"withheld", func() Reading { r := logEventEndFixture(); r.Observation.Assignment.Authoritative = false; return r }, false},
		{"absent", func() Reading { r := logEventFixture("ecc_error_event"); r.Observation.Scope = "device"; return r }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := logEventSupplierFixture(tc.reading())
			raw, attrs, _ := logEventEncoded(t, r, ExportOptions{})
			assertLogEventSupplier(t, attrs, r)
			if tc.owner {
				if entityAttrText(attrs, "assignment.status") != "ended_boundary" || entityAttrText(attrs, "subject.owner.job_id") != "job-a" {
					t.Fatal("lost historical subject", string(raw))
				}
			} else if bytes.Contains(raw, []byte(`"nvtelemetry.owner.`)) || bytes.Contains(raw, []byte(`"nvtelemetry.subject.owner.`)) || bytes.Contains(raw, []byte(`"nvtelemetry.process.`)) {
				t.Fatal("supplier labels widened ownership", string(raw))
			}
		})
	}
	t.Run("supplier-only distinction", func(t *testing.T) {
		a := logEventSupplierFixture(logEventEndFixture())
		b := cloneReadings([]Reading{a})[0]
		b.Observation.Producer = "different-observer"
		b.Observation.ProducerVersion = "different-version"
		b.Observation.ProducerSchema = "different-schema"
		first, _, _ := logEventEncoded(t, a, ExportOptions{})
		second, _, _ := logEventEncoded(t, b, ExportOptions{})
		if bytes.Equal(first, second) {
			t.Fatal("distinct supplied observation identities collapsed into identical bytes")
		}
	})
}

func TestSuppliedLogEventSupplierOrdinaryParity(t *testing.T) {
	sample, err := ReadManaged(t.Context(), strings.NewReader("# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{UUID=\"GPU-a\"} 12.345\n"), ManagedImport{SourceID: "reading-source", Producer: "reading-source-producer", ProducerVersion: "reading-source-version", NativeOrigin: "synthetic-import", Fields: DCGMExporterFields(), TimestampUnix: 1000})
	if err != nil || len(sample.Readings) != 1 {
		t.Fatal(sample, err)
	}
	r := sample.Readings[0]
	if r.IsInt {
		v := r.IntValue
		r.Number = &Number{Uint: &v}
	} else {
		v := r.Value
		r.Number = &Number{Float: &v}
	}
	r.Value = 0
	r.IntValue = 0
	r.IsInt = false
	r.Observation = &Observation{Scope: "device", Kind: "gauge"}
	r = logEventSupplierFixture(r)
	r.Observation.SourceClass = "imported"
	if r.Source.Producer == r.Observation.Producer || r.Source.ProducerVersion == r.Observation.ProducerVersion {
		t.Fatal("non-discriminating source fixture")
	}
	raw, attrs, _ := logEventEncoded(t, r, ExportOptions{})
	assertLogEventSupplier(t, attrs, r)
	if !bytes.Contains(raw, []byte("reading-source-producer")) || !bytes.Contains(raw, []byte("supplied-observer")) {
		t.Fatal("one provenance layer replaced the other", string(raw))
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != "d91b5b59f623e64fff09e9d6e4f5c84f872a45cbddb240c01651f8b6acde7b01" {
		t.Fatalf("ordinary output changed: %s", got)
	}
}
