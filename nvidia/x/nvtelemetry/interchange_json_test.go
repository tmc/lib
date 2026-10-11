package nvtelemetry_test

import (
	"bytes"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestInterchangeJSONRefusal(t *testing.T) {
	const header = `{"manifest":{"schema_version":1,"devices":[{"uuid":"GPU"}]}}` + "\n"
	tests := []struct {
		name, raw string
		want      error
	}{
		{"envelope", `{"manifest":null,"manifest":{"schema_version":1}}`, nvtelemetry.ErrMalformedStream},
		{"envelope alias", `{"manifest":null,"MANIFEST":{"schema_version":1}}`, nvtelemetry.ErrMalformedStream},
		{"version", `{"manifest":{"schema_version":99,"schema_version":1}}`, nvtelemetry.ErrMalformedStream},
		{"version alias", `{"manifest":{"SCHEMA_VERSION":99,"schema_version":1}}`, nvtelemetry.ErrMalformedStream},
		{"escaped key", `{"manifest":{"schema_version":99,"schema\u005fversion":1}}`, nvtelemetry.ErrMalformedStream},
		{"device array", `{"manifest":{"schema_version":1,"devices":[{"uuid":"wrong","uuid":"GPU"}]}}`, nvtelemetry.ErrMalformedStream},
		{"map", `{"manifest":{"schema_version":1,"devices":[{"uuid":"GPU","field_statuses":{"x":"bad","x":"good"}}]}}`, nvtelemetry.ErrMalformedStream},
		{"unknown nested", `{"manifest":{"schema_version":1,"extra":[{"x":1,"x":2}]}}`, nvtelemetry.ErrMalformedStream},
		{"reading", header + `{"sample":{"readings":[{"device_uuid":"GPU","evidence":"measured","int_value":1,"int_value":2}]}}`, nvtelemetry.ErrMalformedStream},
		{"phase", header + `{"phase":{"name":"a","NAME":"b"}}`, nvtelemetry.ErrMalformedStream},
		{"trailing", header + `{}`, nvtelemetry.ErrMalformedStream},
		{"same line trailing", strings.TrimSpace(header) + ` {}`, nvtelemetry.ErrMalformedStream},
		{"malformed", `{"manifest":]}`, nvtelemetry.ErrMalformedStream},
		{"truncated", `{"manifest":`, nvtelemetry.ErrTruncatedRecord},
		{"future", `{"manifest":{"schema_version":99}}`, nvtelemetry.ErrUnsupportedSchema},
		{"deep unknown", `{"manifest":{"schema_version":1,"extra":` + strings.Repeat("[", 130) + "0" + strings.Repeat("]", 130) + `}}`, nvtelemetry.ErrMalformedStream},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, err := nvtelemetry.ReadStream(strings.NewReader(tt.raw), 0)
			if !errors.Is(err, tt.want) {
				t.Fatalf("ReadStream error = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestInterchangeJSONCompatibility(t *testing.T) {
	for _, version := range []string{"1", "2"} {
		raw := `{"MANIFEST":{"schema_version":` + version + `,"config":null,"extra":{"x":1,"X":2},"devices":[{"uuid":"GPU","field_statuses":{"x":"bad","X":"good"}}]}}` + "\n" +
			`{"sample":{"seq":18446744073709551615,"readings":[{"device_uuid":"GPU","evidence":"measured","name":"custom","int_value":18446744073709551615,"is_int":true}]}}` + "\n"
		m, _, samples, err := nvtelemetry.ReadStream(strings.NewReader(raw), int64(len(raw)))
		if err != nil {
			t.Fatal(err)
		}
		if len(samples) != 1 || samples[0].Seq != ^uint64(0) || samples[0].Readings[0].IntValue != ^uint64(0) {
			t.Fatalf("integer precision lost: %+v", samples)
		}
		if len(m.Devices[0].FieldStatuses) != 2 {
			t.Fatal("case-sensitive map keys collapsed")
		}
		var out bytes.Buffer
		w := nvtelemetry.NewWriter(&out, nvtelemetry.CollectorConfig{})
		if err := w.WriteManifest(m); err != nil {
			t.Fatal(err)
		}
		if err := w.WriteSample(&samples[0]); err != nil {
			t.Fatal(err)
		}
		_, _, roundtrip, err := nvtelemetry.ReadStream(&out, 0)
		if err != nil || len(roundtrip) != 1 || roundtrip[0].Readings[0].IntValue != ^uint64(0) {
			t.Fatalf("writer/reader roundtrip: %v, %+v", err, roundtrip)
		}
		_, _, _, err = nvtelemetry.ReadStream(strings.NewReader(raw), int64(len(raw)-1))
		if !errors.Is(err, nvtelemetry.ErrBudgetExceeded) {
			t.Fatalf("byte bound: %v", err)
		}
	}
	for _, name := range []string{"canonical_v1.jsonl", "canonical_v2.jsonl"} {
		raw, err := os.ReadFile("testdata/" + name)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := nvtelemetry.ReadStream(bytes.NewReader(raw), 0); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
}
