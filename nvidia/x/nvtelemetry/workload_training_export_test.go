package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func trainingGaugeInput(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/workload/wandb01911-sdk-decoded.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != "0166da6ec1fd06277146913760c73cc38d69fbd1881b8ca5983600a1b6776bb6" {
		t.Fatal("retained SDK derivative changed")
	}
	return b
}
func trainingGaugeSource() nvtelemetry.WorkloadSource {
	return nvtelemetry.WorkloadSource{Provider: "wandb", Version: "0.19.11", RunID: "5f5i3ig2", ClockDomain: "unix_utc"}
}
func trainingGaugeOptions() nvtelemetry.TrainingWorkloadOTLPOptions {
	return nvtelemetry.TrainingWorkloadOTLPOptions{UnixClockProvenance: "caller-pinned W&B0.19.11 SDK _timestamp time.time Unix seconds; no synchronization bound"}
}
func trainingGaugeRead(t *testing.T, b []byte, s nvtelemetry.WorkloadSource) nvtelemetry.WorkloadExport {
	t.Helper()
	e, err := nvtelemetry.ReadWandBHistory(bytes.NewReader(b), s, nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

type trainingGaugeWire struct {
	ResourceMetrics []struct {
		Resource struct {
			Attributes []struct {
				Key   string
				Value struct{ StringValue string }
			}
		}
		ScopeMetrics []struct {
			Metrics []struct {
				Name, Unit string
				Gauge      *struct {
					DataPoints []struct {
						TimeUnixNano string
						AsDouble     float64
						Attributes   []struct {
							Key   string
							Value struct{ StringValue string }
						}
					}
				}
				Sum, Histogram json.RawMessage
			}
		}
	}
}

func TestWriteTrainingWorkloadOTLPActualSDK(t *testing.T) {
	input := trainingGaugeInput(t)
	e := trainingGaugeRead(t, input, trainingGaugeSource())
	var out bytes.Buffer
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, trainingGaugeOptions()); err != nil {
		t.Fatal(err)
	}
	var wire trainingGaugeWire
	if err := json.Unmarshal(out.Bytes(), &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.ResourceMetrics) != 1 || len(wire.ResourceMetrics[0].ScopeMetrics) != 1 {
		t.Fatal("scope")
	}
	r := wire.ResourceMetrics[0]
	attrs := map[string]string{}
	for _, a := range r.Resource.Attributes {
		attrs[a.Key] = a.Value.StringValue
	}
	if attrs["nvtelemetry.workload.source_report_sha256"] != "0166da6ec1fd06277146913760c73cc38d69fbd1881b8ca5983600a1b6776bb6" || attrs["nvtelemetry.workload.run_cohort"] != "5f5i3ig2" || attrs["nvtelemetry.workload.clock_basis"] != "caller_declared_unix" {
		t.Fatal(attrs)
	}
	if _, ok := attrs["nvtelemetry.workload.rank"]; ok {
		t.Fatal("absent rank became zero")
	}
	metrics := r.ScopeMetrics[0].Metrics
	if len(metrics) != 1 {
		t.Fatal(metrics)
	}
	m := metrics[0]
	if m.Name != "nvtelemetry.workload.wandb.loss" || m.Unit != "" || m.Gauge == nil || len(m.Sum) != 0 || len(m.Histogram) != 0 || len(m.Gauge.DataPoints) != 3 {
		t.Fatal(m)
	}
	// Independent literal oracle from the retained official-SDK decode receipt.
	// No production numeric/time helper derives these expected values.
	times := []string{"1793108838593494000", "1793108838593622000", "1793108838593757000"}
	values := []float64{4, 2, 1}
	steps := []string{"0", "1", "3"}
	for i, p := range m.Gauge.DataPoints {
		if p.TimeUnixNano != times[i] || p.AsDouble != values[i] {
			t.Fatal(i, p)
		}
		a := map[string]string{}
		for _, v := range p.Attributes {
			a[v.Key] = v.Value.StringValue
		}
		if a["nvtelemetry.workload.source_time_unit"] != "seconds" || a["nvtelemetry.workload.step_kind"] != "history_step" || a["nvtelemetry.workload.history_step"] != steps[i] {
			t.Fatal(a)
		}
	}
	var again bytes.Buffer
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&again, e, trainingGaugeOptions()); err != nil || !bytes.Equal(again.Bytes(), out.Bytes()) {
		t.Fatal("unstable encoding", err)
	}
}
func TestWriteTrainingWorkloadOTLPRefusals(t *testing.T) {
	for _, name := range []string{"clock declaration", "clock domain", "digest", "projection", "source identity", "missing time", "duplicate time", "duplicate step", "integer precision", "negative integer precision", "unit evidence", "unknown unit", "cardinality", "source UTF-8"} {
		t.Run(name, func(t *testing.T) {
			raw := trainingGaugeInput(t)
			s := trainingGaugeSource()
			o := trainingGaugeOptions()
			switch name {
			case "missing time":
				raw = bytes.ReplaceAll(raw, []byte(`"_timestamp":1.793108838593494e+09,`), nil)
			case "duplicate time":
				raw = append(raw, bytes.Split(raw, []byte{'\n'})[0]...)
			case "duplicate step":
				raw = bytes.Replace(raw, []byte(`"_step":1`), []byte(`"_step":0`), 1)
			case "integer precision":
				raw = bytes.Replace(raw, []byte(`"loss":4`), []byte(`"loss":9007199254740993`), 1)
			case "negative integer precision":
				raw = bytes.Replace(raw, []byte(`"loss":4`), []byte(`"loss":-9007199254740993`), 1)
			case "unit evidence":
				s.Units = map[string]string{"loss": "1"}
			case "unknown unit":
				s.Units = map[string]string{"loss": "invented"}
				o.UnitEvidence = map[string]string{"loss": "caller supplied unit document"}
			case "cardinality":
				o.Limits.MaxMetrics = 2
			}
			e := trainingGaugeRead(t, raw, s)
			switch name {
			case "clock declaration":
				o.UnixClockProvenance = ""
			case "clock domain":
				e.Source.ClockDomain = "monotonic"
			case "digest":
				e.RawSHA256 = strings.Repeat("0", 64)
			case "projection":
				e.Metrics[0].Name = "other"
			case "source identity":
				e.Source.RunID = "another"
			case "source UTF-8":
				e.Raw = append(e.Raw, 0xff)
			}
			var out bytes.Buffer
			if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, o); err == nil || out.Len() != 0 {
				t.Fatalf("%s accepted or emitted %d bytes: %v", name, out.Len(), err)
			}
		})
	}
}
func TestWriteTrainingWorkloadOTLPBoundsAndMetadata(t *testing.T) {
	raw := trainingGaugeInput(t)
	s := trainingGaugeSource()
	zero := 0
	s.Rank = &zero
	s.Units = map[string]string{"loss": "1"}
	e := trainingGaugeRead(t, raw, s)
	o := trainingGaugeOptions()
	o.UnitEvidence = map[string]string{"loss": "explicit caller scalar unit policy"}
	var out bytes.Buffer
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, o); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"unit":"1"`) || !strings.Contains(out.String(), `"key":"nvtelemetry.workload.rank","value":{"stringValue":"0"}`) {
		t.Fatal("explicit unit or rank lost")
	}
	n := out.Len()
	o.Limits.MaxBytes = int64(n)
	var boundary bytes.Buffer
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&boundary, e, o); err != nil || boundary.Len() != n {
		t.Fatal("exact byte boundary", err)
	}
	o.Limits.MaxBytes = int64(n - 1)
	var short bytes.Buffer
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&short, e, o); err == nil || short.Len() != 0 {
		t.Fatal("byte limit", err)
	}
	// Missing logging step stays absent; it is never fabricated as zero.
	raw = bytes.Replace(raw, []byte(`"_step":0,`), nil, 1)
	e = trainingGaugeRead(t, raw, trainingGaugeSource())
	out.Reset()
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, trainingGaugeOptions()); err != nil {
		t.Fatal(err)
	}
	if strings.Count(out.String(), `"key":"nvtelemetry.workload.history_step"`) != 2 {
		t.Fatal("missing history step changed")
	}
	// The safe integer endpoint is exact; larger integers are refused above.
	raw = bytes.Replace(trainingGaugeInput(t), []byte(`"loss":4`), []byte(`"loss":9007199254740992`), 1)
	e = trainingGaugeRead(t, raw, trainingGaugeSource())
	out.Reset()
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, trainingGaugeOptions()); err != nil || !strings.Contains(out.String(), `"asDouble":9007199254740992`) {
		t.Fatal("integer boundary", err)
	}
}

type trainingGaugeShortWriter struct{}

func (trainingGaugeShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }
func TestWriteTrainingWorkloadOTLPIO(t *testing.T) {
	e := trainingGaugeRead(t, trainingGaugeInput(t), trainingGaugeSource())
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(trainingGaugeShortWriter{}, e, trainingGaugeOptions()); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(nil, e, trainingGaugeOptions()); err == nil {
		t.Fatal("nil writer")
	}
}

// The legacy public reader retains caller metadata. The exporter must refuse
// invalid identity encodings before JSON can replace different bytes by U+FFFD.
func TestWriteTrainingWorkloadOTLPIdentityUTF8(t *testing.T) {
	for _, tc := range []struct {
		name, field string
		bad         byte
	}{
		{"run ff", "run", 0xff}, {"run fe", "run", 0xfe},
		{"phase", "phase", 0xff}, {"clock epoch", "epoch", 0xff},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := trainingGaugeSource()
			value := "identity-" + string([]byte{tc.bad})
			switch tc.field {
			case "run":
				source.RunID = value
			case "phase":
				source.Phase = value
			case "epoch":
				source.ClockEpochID = value
			}
			e := trainingGaugeRead(t, trainingGaugeInput(t), source)
			var out bytes.Buffer
			if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, trainingGaugeOptions()); err == nil || out.Len() != 0 {
				t.Fatalf("invalid identity encoded or conflated: error=%v bytes=%d", err, out.Len())
			}
		})
	}
	// A genuine, valid UTF-8 replacement character is a distinct supported
	// caller identity, rather than a reason to reject all non-ASCII metadata.
	source := trainingGaugeSource()
	source.RunID = "valid-\ufffd"
	e := trainingGaugeRead(t, trainingGaugeInput(t), source)
	var out bytes.Buffer
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, trainingGaugeOptions()); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "valid-\ufffd") {
		t.Fatal("valid Unicode identity changed")
	}
}
