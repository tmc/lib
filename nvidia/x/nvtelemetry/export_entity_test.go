package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func entityExportReading(pid uint32, job string, value uint64) Reading {
	a := entityAssignment()
	zero := int64(0)
	a.Process.Process = cudaworkload.ProcessGenID{PID: pid, Generation: uint64(pid) + 400}
	a.Process.ExecGeneration = 9
	a.Process.NamespacePIDs = []uint32{pid, 2}
	a.Owner = EntityOwner{JobID: job, AllocationID: "allocation-" + job, Attempt: "1"}
	return Reading{Name: MetricMemoryUsedBytes, Unit: UnitBytes, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, TimestampUnix: 1000, TimestampMonotonic: 25, Number: &Number{Uint: &value}, Observation: &Observation{Assignment: &a, UncertaintyNS: &zero, Scope: "process", Interval: &tracecontract.Interval{Start: 20, End: 30, Clock: "monotonic"}, ClockEpoch: "boot", Kind: "gauge", SourceClass: "synthetic", Producer: "fixture", ProducerVersion: "1", ProducerSchema: "test", Observer: Observer{Population: "process-memory"}}}
}

func decodeEntityOTLP(t *testing.T, data []byte, key string) []any {
	t.Helper()
	var doc map[string]any
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	if err := d.Decode(&doc); err != nil {
		t.Fatal(err)
	}
	groups, ok := doc[key].([]any)
	if !ok {
		t.Fatalf("missing %s: %s", key, data)
	}
	return groups
}
func entityResourceAttrs(t *testing.T, g any) map[string]any {
	t.Helper()
	resource := g.(map[string]any)["resource"].(map[string]any)
	attrs := resource["attributes"].([]any)
	m := map[string]any{}
	for _, v := range attrs {
		a := v.(map[string]any)
		key := a["key"].(string)
		if _, ok := m[key]; ok {
			t.Fatal("duplicate resource key", key)
		}
		m[key] = a["value"]
	}
	return m
}
func entityAttrText(attrs map[string]any, key string) string {
	v, ok := attrs["nvtelemetry."+key].(map[string]any)
	if !ok {
		return ""
	}
	s, _ := v["stringValue"].(string)
	return s
}
func entityMetricPoint(t *testing.T, g any) map[string]any {
	t.Helper()
	scope := g.(map[string]any)["scopeMetrics"].([]any)
	metrics := scope[0].(map[string]any)["metrics"].([]any)
	metric := metrics[0].(map[string]any)
	for _, kind := range []string{"gauge", "sum"} {
		if typ, ok := metric[kind].(map[string]any); ok {
			return typ["dataPoints"].([]any)[0].(map[string]any)
		}
	}
	t.Fatal("missing metric point")
	return nil
}
func entityLogFields(t *testing.T, g any, index int) map[string]string {
	t.Helper()
	scope := g.(map[string]any)["scopeLogs"].([]any)
	records := scope[0].(map[string]any)["logRecords"].([]any)
	rec := records[index].(map[string]any)
	values := rec["body"].(map[string]any)["kvlistValue"].(map[string]any)["values"].([]any)
	out := map[string]string{}
	for _, v := range values {
		field := v.(map[string]any)
		out[field["key"].(string)] = field["value"].(map[string]any)["stringValue"].(string)
	}
	return out
}

func TestEntityOTLPPopulations(t *testing.T) {
	readings := []Reading{entityExportReading(12, "job-a", 9007199254740993), entityExportReading(13, "job-b", 9007199254740995)}
	device := Reading{Name: MetricMemoryUsedBytes, Unit: UnitBytes, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, TimestampUnix: 1000, IsInt: true, IntValue: 99}
	readings = append(readings, device)
	var out bytes.Buffer
	if err := WriteOTLPMetrics(&out, readings, ExportOptions{}); err != nil {
		t.Fatal(err)
	}
	groups := decodeEntityOTLP(t, out.Bytes(), "resourceMetrics")
	if len(groups) != 3 {
		t.Fatalf("collapsed populations: %s", out.Bytes())
	}
	got := map[string]string{}
	for _, g := range groups {
		attrs := entityResourceAttrs(t, g)
		job := entityAttrText(attrs, "owner.job_id")
		point := entityMetricPoint(t, g)
		got[job] = point["asInt"].(string)
		if job != "" && entityAttrText(attrs, "assignment.status") != "assigned" {
			t.Fatal("lost attribution status", attrs)
		}
		if job == "" && entityAttrText(attrs, "assignment.status") != "absent" {
			t.Fatal("device population became assigned", attrs)
		}
	}
	if got["job-a"] != "9007199254740993" || got["job-b"] != "9007199254740995" || got[""] != "99" {
		t.Fatal("wrong resource/value pairs", got)
	}
	duplicate := append([]Reading(nil), readings...)
	duplicate = append(duplicate, readings[0])
	out.Reset()
	if err := WriteOTLPMetrics(&out, duplicate, ExportOptions{}); err == nil || out.Len() != 0 {
		t.Fatal("duplicate exact population emitted")
	}
}

func TestEntityOTLPWithheld(t *testing.T) {
	for _, tt := range []struct {
		name, status string
		modify       func(*Reading)
	}{
		{"unknown uncertainty", "unknown", func(r *Reading) { r.Observation.UncertaintyNS = nil }},
		{"unknown", "unknown", func(r *Reading) { r.Observation.Assignment.State = "unknown" }},
		{"contradictory", "contradictory", func(r *Reading) { r.Observation.Assignment.State = "contradictory" }},
		{"unverified", "unverified", func(r *Reading) { r.Observation.Assignment.Authoritative = false }},
		{"missing exec", "unknown", func(r *Reading) {
			r.Observation.Assignment.Process.HasExecGeneration = false
			r.Observation.Assignment.Process.ExecGeneration = 0
		}},
		{"device wide", "device_scope", func(r *Reading) { r.Observation.Scope = "device" }},
		{"unknown scope", "device_scope", func(r *Reading) { r.Observation.Scope = "" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := entityExportReading(12, "not-admitted", 3)
			tt.modify(&r)
			var b bytes.Buffer
			if err := WriteOTLPMetrics(&b, []Reading{r}, ExportOptions{}); err != nil {
				t.Fatal(err)
			}
			g := decodeEntityOTLP(t, b.Bytes(), "resourceMetrics")
			attrs := entityResourceAttrs(t, g[0])
			if entityAttrText(attrs, "assignment.status") != tt.status {
				t.Fatal(attrs)
			}
			if _, ok := attrs["nvtelemetry.owner.job_id"]; ok {
				t.Fatal("withheld owner promoted", attrs)
			}
		})
	}
	a, b := entityExportReading(12, "a", 3), entityExportReading(13, "b", 3)
	a.Observation.Scope = "device"
	b.Observation.Scope = "device"
	var out bytes.Buffer
	if err := WriteOTLPMetrics(&out, []Reading{a, b}, ExportOptions{}); err == nil || out.Len() != 0 {
		t.Fatal("device-wide value duplicated across jobs")
	}
}

func TestEntityOTLPRefusalBeforeWrite(t *testing.T) {
	for _, tt := range []struct {
		name   string
		modify func(*Reading, *ExportOptions)
	}{
		{"stale", func(r *Reading, _ *ExportOptions) { r.Observation.Assignment.EndNanos = 30 }},
		{"epoch", func(r *Reading, _ *ExportOptions) { r.Observation.ClockEpoch = "another-boot" }},
		{"clock", func(r *Reading, _ *ExportOptions) { r.Observation.Interval.Clock = "other-clock" }},
		{"option clock", func(_ *Reading, o *ExportOptions) { o.ClockDomain = "other-clock" }},
		{"missing interval", func(r *Reading, _ *ExportOptions) { r.Observation.Interval = nil }},
		{"wrong process", func(r *Reading, _ *ExportOptions) {
			r.Observation.Workload = &cudaworkload.WorkloadIdentity{Process: cudaworkload.ProcessGenID{PID: 12, Generation: 999}}
		}},
		{"wrong device", func(r *Reading, _ *ExportOptions) { r.Observation.Assignment.Device.UUID = "GPU-other" }},
		{"provenance", func(r *Reading, _ *ExportOptions) { r.Observation.Assignment.SourceRevision = "" }},
		{"uncertainty straddles", func(r *Reading, _ *ExportOptions) { u := int64(90); r.Observation.UncertaintyNS = &u }},
		{"too many namespaces", func(r *Reading, _ *ExportOptions) {
			r.Observation.Assignment.Process.NamespacePIDs = make([]uint32, 33)
		}},
		{"long owner", func(r *Reading, _ *ExportOptions) { r.Observation.Assignment.Owner.JobID = strings.Repeat("j", 257) }},
		{"label control", func(r *Reading, _ *ExportOptions) { r.Observation.Assignment.Process.NodeID = "node\nsecret" }},
		{"missing number", func(r *Reading, _ *ExportOptions) { r.Number = nil }},
		{"two numeric arms", func(r *Reading, _ *ExportOptions) { v := int64(1); r.Number.Int = &v }},
		{"mixed legacy", func(r *Reading, _ *ExportOptions) { r.IsInt = true; r.IntValue = 1 }},
		{"signed overflow", func(r *Reading, _ *ExportOptions) { v := uint64(math.MaxUint64); r.Number.Uint = &v }},
		{"counter epoch", func(r *Reading, _ *ExportOptions) {
			r.Name = MetricPCIERxBytes
			r.Unit = UnitBytes
			r.Observation.Kind = "counter"
		}},
		{"byte limit", func(_ *Reading, o *ExportOptions) { o.MaxBytes = 1 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := entityExportReading(12, "job", 3)
			o := ExportOptions{}
			tt.modify(&r, &o)
			var out bytes.Buffer
			if err := WriteOTLPMetrics(&out, []Reading{r}, o); err == nil || out.Len() != 0 {
				t.Fatal("invalid population escaped", err, out.String())
			}
		})
	}
}

func TestEntityOTLPLogsAndPrivacy(t *testing.T) {
	a := entityExportReading(12, "job-a", math.MaxUint64)
	a.Name = MetricClocksEventReasons
	a.Unit = UnitBitmask
	a.Observation.Workload = &cudaworkload.WorkloadIdentity{Process: a.Observation.Assignment.Process.Process, Labels: map[string]string{"env": "SECRET_ENV_TOKEN", "command": "SECRET_COMMAND", "stack": "SECRET_STACK"}}
	b := entityExportReading(13, "job-b", 0)
	b.Evidence = EvidenceUnavailable
	b.UnavailableReason = ReasonUnsupported
	b.Number = nil
	var out bytes.Buffer
	if err := WriteOTLPLogs(&out, []Reading{a, b}, ExportOptions{}); err != nil {
		t.Fatal(err)
	}
	groups := decodeEntityOTLP(t, out.Bytes(), "resourceLogs")
	if len(groups) != 2 {
		t.Fatalf("log populations collapsed: %s", out.Bytes())
	}
	if entityAttrText(entityResourceAttrs(t, groups[0]), "owner.job_id") != "job-a" || entityLogFields(t, groups[0], 0)["value_uint64"] != "18446744073709551615" {
		t.Fatal("lost exact unsigned log value")
	}
	fields := entityLogFields(t, groups[1], 0)
	if fields["status"] != "unavailable" || fields["value_uint64"] != "" || fields["value_float64"] != "" {
		t.Fatal("unavailable became numeric zero", fields)
	}
	if strings.Contains(out.String(), "SECRET_") {
		t.Fatal("raw environment/command/stack escaped")
	}
}

func TestEntityOTLPByteBoundaries(t *testing.T) {
	a, b := entityExportReading(12, "a", 3), entityExportReading(13, "b", 4)
	for _, write := range []func(io.Writer, []Reading, ExportOptions) error{WriteOTLPMetrics, WriteOTLPLogs} {
		var out bytes.Buffer
		if err := write(&out, []Reading{a, b}, ExportOptions{}); err != nil {
			t.Fatal(err)
		}
		n := out.Len()
		out.Reset()
		if err := write(&out, []Reading{a, b}, ExportOptions{MaxBytes: n}); err != nil || out.Len() != n {
			t.Fatalf("exact boundary: %v %d/%d", err, out.Len(), n)
		}
		out.Reset()
		if err := write(&out, []Reading{a, b}, ExportOptions{MaxBytes: n - 1}); err == nil || out.Len() != 0 {
			t.Fatal("one-short boundary emitted")
		}
		out.Reset()
		if err := write(&out, []Reading{a, b}, ExportOptions{MaxReadings: 1}); err == nil || out.Len() != 0 {
			t.Fatal("reading limit emitted")
		}
	}
}

func TestEntityExpositionCanonicalDeviceValues(t *testing.T) {
	exact := uint64(9007199254740993)
	r := Reading{Name: MetricMemoryUsedBytes, Unit: UnitBytes, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, TimestampUnix: 1000, Number: &Number{Uint: &exact}}
	var b bytes.Buffer
	if err := WriteExposition(&b, []Reading{r}, ExportOptions{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.String(), " 9007199254740993\n") {
		t.Fatal("canonical number became zero", b.String())
	}
	b.Reset()
	assigned := entityExportReading(12, "job", exact)
	if err := WriteExposition(&b, []Reading{assigned}, ExportOptions{}); err == nil || b.Len() != 0 {
		t.Fatal("fixed exposition profile erased attribution")
	}
}

func TestEntityOTLPRefusalDoesNotEchoMetadata(t *testing.T) {
	r := entityExportReading(12, "job", 1)
	r.Observation.Assignment.State = "SECRET_ENV_TOKEN"
	var out bytes.Buffer
	err := WriteOTLPMetrics(&out, []Reading{r}, ExportOptions{})
	if err == nil || out.Len() != 0 || strings.Contains(err.Error(), "SECRET_ENV_TOKEN") {
		t.Fatalf("invalid entity text escaped: %v", err)
	}
}

func TestEntityOTLPCanonicalManagedDeviceValues(t *testing.T) {
	sample, err := ReadManaged(t.Context(), strings.NewReader("# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{UUID=\"GPU-a\"} 12.345\n# TYPE DCGM_FI_DEV_FB_USED gauge\nDCGM_FI_DEV_FB_USED{UUID=\"GPU-a\"} 2\n"), ManagedImport{SourceID: "synthetic-import", Producer: "dcgm", ProducerVersion: "fixture", NativeOrigin: "synthetic-import", Fields: DCGMExporterFields(), TimestampUnix: 1000})
	if err != nil || len(sample.Readings) != 2 {
		t.Fatalf("fixture: %+v %v", sample, err)
	}
	for i := range sample.Readings {
		r := &sample.Readings[i]
		if r.IsInt {
			v := r.IntValue
			r.Number = &Number{Uint: &v}
		} else {
			v := r.Value
			r.Number = &Number{Float: &v}
		}
		r.IsInt = false
		r.IntValue = 0
		r.Value = 0
	}
	for _, write := range []func(io.Writer, []Reading, ExportOptions) error{WriteOTLPMetrics, WriteOTLPLogs, WriteExposition} {
		var out bytes.Buffer
		if err := write(&out, sample.Readings, ExportOptions{}); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out.String(), "12345") || !strings.Contains(out.String(), "2097152") {
			t.Fatal("managed canonical values lost", out.String())
		}
	}
}

func TestEntityOTLPCounterNegativeZero(t *testing.T) {
	zero := math.Copysign(0, -1)
	r := Reading{Name: MetricPCIERxBytes, Unit: UnitBytes, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, TimestampUnix: 1000, Number: &Number{Float: &zero}}
	opts := ExportOptions{HasCounterStart: true, CounterStartUnixNanos: 500}
	for _, write := range []func(io.Writer, []Reading, ExportOptions) error{WriteExposition, WriteOTLPMetrics, WriteOTLPLogs} {
		var b bytes.Buffer
		if err := write(&b, []Reading{r}, opts); err != nil {
			t.Fatal("negative zero is not a negative counter", err)
		}
	}
}

func TestEntityOTLPDevicePopulationIdentity(t *testing.T) {
	for _, name := range []string{"revision", "validity", "provenance"} {
		t.Run(name, func(t *testing.T) {
			a, b := entityExportReading(12, "a", 3), entityExportReading(13, "b", 4)
			a.Observation.Scope, b.Observation.Scope = "device", "device"
			switch name {
			case "revision":
				b.Observation.Assignment.SourceRevision = "other"
			case "validity":
				b.Observation.Assignment.EndNanos++
			case "provenance":
				b.Observation.Assignment.Provenance = []EntityProvenance{{Source: "fixture", Revision: "other", StartNanos: 0, EndNanos: 100}}
			}
			var out bytes.Buffer
			if err := WriteOTLPMetrics(&out, []Reading{a, b}, ExportOptions{}); err == nil || out.Len() != 0 {
				t.Fatal("assignment metadata manufactured a whole-device population", err, out.String())
			}
		})
	}
	// Independent observer populations stay distinct without any assignment.
	a, b := entityExportReading(12, "a", 3), entityExportReading(13, "b", 4)
	a.Observation.Scope, b.Observation.Scope = "device", "device"
	a.Observation.Assignment, b.Observation.Assignment = nil, nil
	a.Observation.Observer.Population, b.Observation.Observer.Population = "first", "second"
	var out bytes.Buffer
	if err := WriteOTLPMetrics(&out, []Reading{a, b}, ExportOptions{}); err != nil {
		t.Fatal(err)
	}
	groups := decodeEntityOTLP(t, out.Bytes(), "resourceMetrics")
	if len(groups) != 2 {
		t.Fatalf("observer populations collapsed: %s", out.Bytes())
	}
	for i, want := range []string{"first", "second"} {
		if got := entityAttrText(entityResourceAttrs(t, groups[i]), "observation.population"); got != want {
			t.Fatalf("population %d = %q, want %q", i, got, want)
		}
	}
}

func TestEntityOTLPProcessScopeWithoutAssignment(t *testing.T) {
	r := entityExportReading(12, "a", 3)
	r.Observation.Assignment = nil
	for _, tt := range []struct {
		name  string
		write func(io.Writer, []Reading, ExportOptions) error
	}{{"metrics", WriteOTLPMetrics}, {"logs", WriteOTLPLogs}, {"exposition", WriteExposition}} {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			if err := tt.write(&out, []Reading{r}, ExportOptions{}); err == nil || out.Len() != 0 {
				t.Fatal("missing assignment erased process scope", err, out.String())
			}
		})
	}
}
