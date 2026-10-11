package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"
)

func workloadOTLPFixture(t *testing.T) (WorkloadExport, WorkloadExport, WorkloadOTLPOptions) {
	t.Helper()
	source := WorkloadSource{Provider: "vllm", Version: "0.8.5", RunID: "cohort"}
	read := func(counter, bucket, count, sum string) WorkloadExport {
		raw := `# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="private-model"} ` + counter + `
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="private-model",le="0.1"} ` + bucket + `
vllm:time_to_first_token_seconds_bucket{model_name="private-model",le="+Inf"} ` + count + `
vllm:time_to_first_token_seconds_count{model_name="private-model"} ` + count + `
vllm:time_to_first_token_seconds_sum{model_name="private-model"} ` + sum + "\n"
		e, err := ReadServingExposition(context.Background(), strings.NewReader(raw), source, WorkloadLimits{})
		if err != nil {
			t.Fatal(err)
		}
		return e
	}
	before, err := StampWorkloadSnapshot(read("10", "2", "4", "0.4"), 0, "mono", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	after, err := StampWorkloadSnapshot(read("13", "4", "9", "1.1"), 10, "mono", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	return before, after, WorkloadOTLPOptions{StartNanos: 0, EndNanos: 10, ClockDomain: "mono", ClockEpochID: "epoch", HasUnixMapping: true, StartUnixNanos: 1700000000000000000, EndUnixNanos: 1700000000000000010, MappingProvenance: "calibration-receipt"}
}

// The decoding structure mirrors the public OTLP JSON request fields, without
// referring to the writer's private maps or helper functions.
type workloadDecodedOTLP struct {
	ResourceMetrics []struct {
		ScopeMetrics []struct {
			Metrics []struct {
				Name string
				Unit string
				Sum  *struct {
					AggregationTemporality int
					IsMonotonic            bool
					DataPoints             []struct {
						AsInt             string
						StartTimeUnixNano string
						TimeUnixNano      string
					}
				}
				Histogram *struct {
					AggregationTemporality int
					DataPoints             []struct {
						Count          string
						Sum            float64
						ExplicitBounds []float64
						BucketCounts   []string
					}
				}
			}
		}
	}
}

func TestWriteWorkloadOTLPDecode(t *testing.T) {
	before, after, o := workloadOTLPFixture(t)
	var buf bytes.Buffer
	if err := WriteWorkloadOTLP(&buf, before, after, o); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(buf.String(), "private-model") {
		t.Fatal("raw vendor label leaked")
	}
	var decoded workloadDecodedOTLP
	if err := json.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	metrics := decoded.ResourceMetrics[0].ScopeMetrics[0].Metrics
	if len(metrics) != 2 {
		t.Fatal(len(metrics))
	}
	for _, m := range metrics {
		if m.Sum != nil {
			if m.Sum.AggregationTemporality != 1 || !m.Sum.IsMonotonic || m.Sum.DataPoints[0].AsInt != "3" || m.Sum.DataPoints[0].StartTimeUnixNano != "1700000000000000000" {
				t.Fatalf("sum %+v", m.Sum)
			}
		} else {
			h := m.Histogram
			p := h.DataPoints[0]
			if h.AggregationTemporality != 1 || p.Count != "5" || len(p.BucketCounts) != 2 || p.BucketCounts[0] != "2" || p.BucketCounts[1] != "3" || len(p.ExplicitBounds) != 1 || p.ExplicitBounds[0] != 0.1 || p.Sum != 0.7 || m.Unit != "s" {
				t.Fatalf("histogram %+v", p)
			}
		}
	}
}
func TestWriteWorkloadOTLPRefusals(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*WorkloadExport, *WorkloadExport, *WorkloadOTLPOptions)
	}{
		{"unknown clock", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { o.HasUnixMapping = false }},
		{"mapping interval", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { o.EndUnixNanos++ }},
		{"epoch", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { o.ClockEpochID = "different" }},
		{"request label", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) {
			a.Metrics[0].Labels["request_id"] = "secret"
			b.Metrics[0].Labels["request_id"] = "secret"
		}},
		{"unit", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { b.Metrics[0].Unit = "J" }},
		{"fractional discrete count", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { b.Metrics[0].RawValue = "13.5" }},
		{"counter reset", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { b.Metrics[0].RawValue = "1" }},
		{"partial bucket reset", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { b.Metrics[1].RawValue = "8" }},
		{"count overflow", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) {
			for i := range b.Metrics {
				if b.Metrics[i].Kind == "histogram_count" || b.Metrics[i].Labels["le"] == "+Inf" {
					b.Metrics[i].RawValue = "18446744073709551621"
				}
			}
		}},
		{"byte limit", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { o.Limits.MaxBytes = 1 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, b, o := workloadOTLPFixture(t)
			tc.mutate(&a, &b, &o)
			var out bytes.Buffer
			if err := WriteWorkloadOTLP(&out, a, b, o); err == nil || out.Len() != 0 {
				t.Fatalf("admitted mutation: %v", err)
			}
		})
	}
}
func TestWriteWorkloadOTLPExactUnsignedCount(t *testing.T) {
	a, b, o := workloadOTLPFixture(t)
	for i := range a.Metrics {
		if strings.HasPrefix(a.Metrics[i].Kind, "histogram_") {
			a.Metrics[i].RawValue = "0"
			b.Metrics[i].RawValue = "9007199254740993"
		}
		if a.Metrics[i].Kind == "histogram_sum" {
			b.Metrics[i].RawValue = "1"
		}
	}
	var out bytes.Buffer
	if err := WriteWorkloadOTLP(&out, a, b, o); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"count":"9007199254740993"`) {
		t.Fatal("uint64 count rounded")
	}
}
func ExampleWriteWorkloadOTLP() {
	source := WorkloadSource{Provider: "vllm", Version: "0.8.5", RunID: "cohort"}
	read := func(value string, when int64) WorkloadExport {
		raw := "# TYPE vllm:prompt_tokens_total counter\nvllm:prompt_tokens_total{model_name=\"model\"} " + value + "\n"
		e, _ := ReadServingExposition(context.Background(), strings.NewReader(raw), source, WorkloadLimits{})
		e, _ = StampWorkloadSnapshot(e, when, "mono", "epoch")
		return e
	}
	err := WriteWorkloadOTLP(io.Discard, read("10", 0), read("13", 10), WorkloadOTLPOptions{StartNanos: 0, EndNanos: 10, ClockDomain: "mono", ClockEpochID: "epoch", HasUnixMapping: true, StartUnixNanos: 100, EndUnixNanos: 110, MappingProvenance: "receipt"})
	fmt.Println(err)
	// Output: <nil>
}
