package nvtelemetry

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

func vllmV1Source() WorkloadSource {
	return WorkloadSource{Provider: "vllm", Version: "0.8.5", Engine: "v1", RunID: "control"}
}

func vllmV1Control(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile("testdata/workload/vllm-v1-0.8.5.prom.control")
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func vllmV1Read(t *testing.T, raw string) WorkloadExport {
	t.Helper()
	e, err := ReadServingExposition(context.Background(), strings.NewReader(raw), vllmV1Source(), WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func vllmV1Window(t *testing.T) (WorkloadExport, WorkloadExport, WorkloadOTLPOptions) {
	t.Helper()
	raw := vllmV1Control(t)
	// Independent constructed second endpoint: exact deltas above double precision.
	after := strings.ReplaceAll(raw, "9007199254740993", "9007199254741000")
	after = strings.ReplaceAll(after, "9007199254741993", "9007199254742006")
	after = strings.ReplaceAll(after, "} 10\n", "} 15\n")
	after = strings.ReplaceAll(after, ",le=\"1\"} 1\n", ",le=\"1\"} 2\n")
	after = strings.ReplaceAll(after, "} 2\n", "} 4\n")
	// Restore finite buckets to 2 after the general count/sum replacement.
	after = strings.ReplaceAll(after, ",le=\"1\"} 4\n", ",le=\"1\"} 2\n")
	a, err := StampWorkloadSnapshot(vllmV1Read(t, raw), 0, "mono", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	b, err := StampWorkloadSnapshot(vllmV1Read(t, after), 10, "mono", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	return a, b, WorkloadOTLPOptions{StartNanos: 0, EndNanos: 10, ClockDomain: "mono", ClockEpochID: "epoch", HasUnixMapping: true, StartUnixNanos: 100, EndUnixNanos: 110, MappingProvenance: "constructed-clock-control"}
}

func TestVLLMV1PublicWindowOTLP(t *testing.T) {
	a, b, o := vllmV1Window(t)
	if len(a.Metrics) != 88 {
		t.Fatalf("metrics %d, want 88", len(a.Metrics))
	}
	if !reflect.DeepEqual(a.UnmappedNames, []string{"vllm:gpu_prefix_cache_queries_total", "vllm:num_requests_swapped"}) {
		t.Fatal(a.UnmappedNames)
	}
	counters, err := CounterWorkloadWindow(a, b, 0, 10, "mono", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	if len(counters) != 10 {
		t.Fatal(len(counters))
	}
	for _, p := range counters {
		want := "5"
		if p.Name == "vllm:prompt_tokens_total" {
			want = map[string]string{"0": "7", "1": "13"}[p.Labels["engine"]]
		}
		if p.Delta != want {
			t.Fatalf("%s %v delta %s want %s", p.Name, p.Labels, p.Delta, want)
		}
	}
	hist, err := HistogramWorkloadWindow(a, b, 0, 10, "mono", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	if len(hist) != 72 {
		t.Fatal(len(hist))
	}
	semantics := map[string]string{
		"vllm:request_queue_time_seconds_sum":     "aggregate_request_queue_time",
		"vllm:request_inference_time_seconds_sum": "aggregate_request_inference_time",
		"vllm:request_prefill_time_seconds_sum":   "aggregate_request_prefill_time",
		"vllm:request_decode_time_seconds_sum":    "aggregate_request_decode_time",
	}
	for _, p := range hist {
		want := "2"
		if p.Labels["le"] == "1" {
			want = "1"
		}
		if p.Delta != want {
			t.Fatalf("%s delta %s want %s", p.Name, p.Delta, want)
		}
		if s, ok := semantics[p.Name]; ok && p.Semantic != s {
			t.Fatalf("semantic %s", p.Semantic)
		}
	}
	var out bytes.Buffer
	if err := WriteWorkloadOTLP(&out, a, b, o); err != nil {
		t.Fatal(err)
	}
	// Decode public output independently of the writer's mapping and key helpers.
	var decoded struct {
		ResourceMetrics []struct {
			ScopeMetrics []struct {
				Metrics []struct {
					Name string
					Sum  *struct {
						DataPoints []struct {
							AsInt      string
							Attributes []struct {
								Key   string
								Value struct{ StringValue string }
							}
						}
					}
					Histogram *struct {
						DataPoints []struct {
							Count          string
							Sum            float64
							BucketCounts   []string
							ExplicitBounds []float64
						}
					}
				}
			}
		}
	}
	if err := json.Unmarshal(out.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	metrics := decoded.ResourceMetrics[0].ScopeMetrics[0].Metrics
	if len(metrics) != 28 {
		t.Fatalf("OTLP metrics %d", len(metrics))
	}
	hashes := map[string]bool{}
	for _, m := range metrics {
		if m.Histogram != nil {
			p := m.Histogram.DataPoints[0]
			if p.Count != "2" || p.Sum != 2 || !reflect.DeepEqual(p.BucketCounts, []string{"1", "1"}) || !reflect.DeepEqual(p.ExplicitBounds, []float64{1}) {
				t.Fatalf("histogram %+v", p)
			}
		}
		if m.Sum == nil {
			continue
		}
		p := m.Sum.DataPoints[0]
		attrs := map[string]string{}
		for _, a := range p.Attributes {
			attrs[a.Key] = a.Value.StringValue
		}
		if attrs["nvtelemetry.workload.engine_contract"] != "v1" {
			t.Fatal(attrs)
		}
		if m.Name == "nvtelemetry.workload.vllm.vllm:prompt_tokens_total" {
			for engine, want := range map[string]string{"0": "7", "1": "13"} {
				// Canonical hash input is literal and independent of workloadSeriesKey.
				h := sha256.Sum256([]byte(`"vllm:prompt_tokens_total""engine"="` + engine + `";"model_name"="m";`))
				hash := hex.EncodeToString(h[:])
				if attrs["nvtelemetry.workload.series_sha256"] == hash {
					if p.AsInt != want {
						t.Fatalf("engine %s delta %s", engine, p.AsInt)
					}
					hashes[engine] = true
				}
			}
		}
	}
	if len(hashes) != 2 {
		t.Fatalf("engine hashes %v", hashes)
	}
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	var round WorkloadExport
	if err := json.Unmarshal(raw, &round); err != nil {
		t.Fatal(err)
	}
	if round.Source.Engine != "v1" || !bytes.Equal(round.Raw, a.Raw) || round.RawSHA256 != a.RawSHA256 {
		t.Fatal("lost source/raw provenance")
	}
	if ExactWorkloadJoin(a) == nil {
		t.Fatal("accepted exact join")
	}
}

type vllmV1Transport struct {
	calls int
	raw   string
}

func (r *vllmV1Transport) RoundTrip(*http.Request) (*http.Response, error) {
	r.calls++
	return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(r.raw))}, nil
}

func TestVLLMV1HTTPRefresh(t *testing.T) {
	transport := &vllmV1Transport{raw: vllmV1Control(t)}
	client := &http.Client{Transport: transport}
	request, err := http.NewRequest("GET", "https://control.invalid/metrics", nil)
	if err != nil {
		t.Fatal(err)
	}
	fetch := func(ctx context.Context) (WorkloadHTTPResult, error) {
		return FetchWorkload(ctx, client, request, vllmV1Source(), WorkloadLimits{})
	}
	clock := NewFakeClock(1, 100)
	r, err := NewWorkloadRefresher(fetch, clock, "control-clock", "control-epoch", RefreshConfig{Interval: time.Second, Timeout: time.Second, MaxAge: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	a, fresh := r.Latest(1)
	if !fresh || a.Export.Source.Engine != "v1" || a.Export.Metrics[0].Source.Engine != "v1" || a.Export.Metrics[0].Labels["engine"] != "0" || a.Export.Metrics[0].TimeSource != "collector_observation" {
		t.Fatal("refresh lost source identity")
	}
	a.Export.Source.Engine = "changed"
	a.Export.Metrics[0].Source.Engine = "changed"
	a.Export.Metrics[0].Labels["engine"] = "9"
	a.Export.Raw[0] = '!'
	b, fresh := r.Latest(1)
	if !fresh || b.Export.Source.Engine != "v1" || b.Export.Metrics[0].Source.Engine != "v1" || b.Export.Metrics[0].Labels["engine"] != "0" || string(b.Export.Raw) != transport.raw {
		t.Fatal("cache alias")
	}
	for _, s := range []WorkloadSource{
		{Provider: "vllm", Version: "0.8.5", Engine: "V1", RunID: "control"},
		{Provider: "vllm", Version: "0.8.5", Engine: " v1", RunID: "control"},
		{Provider: "vllm", Version: "0.8.5", Engine: "v0", RunID: "control"},
		{Provider: "vllm", Version: "0.8.6", Engine: "v1", RunID: "control"},
		{Provider: "triton", Version: "25.06", Engine: "v1", RunID: "control"},
	} {
		before := transport.calls
		if _, err := FetchWorkload(context.Background(), client, request, s, WorkloadLimits{}); err == nil || transport.calls != before {
			t.Fatalf("tuple transmitted %+v: %v", s, err)
		}
	}
}

func TestVLLMV1ReaderRefusals(t *testing.T) {
	raw := "# TYPE vllm:request_success_total counter\nvllm:request_success_total{model_name=\"m\",engine=\"0\",finished_reason=\"abort\"} 10\n"
	for _, tc := range []struct{ name, old, new string }{
		{"missing engine", `,engine="0"`, ``}, {"missing model", `model_name="m",`, ``},
		{"empty model", `model_name="m"`, `model_name=""`},
		{"negative engine", `engine="0"`, `engine="-1"`}, {"noncanonical engine", `engine="0"`, `engine="01"`},
		{"fractional engine", `engine="0"`, `engine="0.1"`}, {"space engine", `engine="0"`, `engine=" 0"`},
		{"extra label", `engine="0"`, `engine="0",request_id="x"`},
		{"wrong reason", `finished_reason="abort"`, `finished_reason="success"`},
		{"missing reason", `,finished_reason="abort"`, ``},
		{"wrong type", " counter\n", " gauge\n"}, {"fraction", "} 10", "} 1.5"},
		{"negative", "} 10", "} -1"}, {"exponent", "} 10", "} 1e1001"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ReadServingExposition(context.Background(), strings.NewReader(strings.ReplaceAll(raw, tc.old, tc.new)), vllmV1Source(), WorkloadLimits{}); err == nil {
				t.Fatal("accepted invalid sample")
			}
		})
	}
	for _, name := range []string{"num_requests_running", "num_requests_waiting"} {
		input := fmt.Sprintf("# TYPE vllm:%s gauge\nvllm:%s{model_name=\"m\",engine=\"0\"} 1.5\n", name, name)
		if _, err := ReadServingExposition(context.Background(), strings.NewReader(input), vllmV1Source(), WorkloadLimits{}); err == nil {
			t.Fatal("accepted fractional request gauge")
		}
	}
	input := "# TYPE vllm:gpu_cache_usage_perc gauge\nvllm:gpu_cache_usage_perc{model_name=\"m\",engine=\"0\"} 1.01\n"
	if _, err := ReadServingExposition(context.Background(), strings.NewReader(input), vllmV1Source(), WorkloadLimits{}); err == nil {
		t.Fatal("accepted fraction >1")
	}
	s := vllmV1Source()
	s.Engine = ""
	if _, err := ReadServingExposition(context.Background(), strings.NewReader(raw), s, WorkloadLimits{}); err == nil {
		t.Fatal("accepted ambiguous legacy engine label")
	}
}

func TestVLLMV1WindowRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*WorkloadExport, *WorkloadExport, *WorkloadOTLPOptions)
	}{
		{"engine contract", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) {
			b.Source.Engine = ""
			for i := range b.Metrics {
				b.Metrics[i].Source.Engine = ""
			}
		}},
		{"point contract", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { b.Metrics[0].Source.Engine = "" }},
		{"changed engine", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { b.Metrics[0].Labels["engine"] = "2" }},
		{"missing engine endpoint", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { b.Metrics = b.Metrics[1:] }},
		{"clock", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { o.ClockDomain = "other" }},
		{"epoch", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { o.ClockEpochID = "other" }},
		{"reset", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { b.Metrics[0].RawValue = "1" }},
		{"incomplete", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { b.NextPageToken = "next" }},
		{"fake unix mapping", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) { o.HasUnixMapping = false }},
		{"forged labels", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) {
			delete(a.Metrics[0].Labels, "engine")
			delete(b.Metrics[0].Labels, "engine")
		}},
		{"forged reason", func(a, b *WorkloadExport, o *WorkloadOTLPOptions) {
			for _, e := range []*WorkloadExport{a, b} {
				for i := range e.Metrics {
					if e.Metrics[i].Name == "vllm:request_success_total" {
						e.Metrics[i].Labels["finished_reason"] = "success"
					}
				}
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, b, o := vllmV1Window(t)
			tc.mutate(&a, &b, &o)
			var out bytes.Buffer
			if err := WriteWorkloadOTLP(&out, a, b, o); err == nil || out.Len() != 0 {
				t.Fatalf("accepted mutation: %v", err)
			}
		})
	}
	raw := vllmV1Control(t)
	for _, mutated := range []string{
		strings.Replace(raw, `le="+Inf"} 2`, `le="+Inf"} 0`, 1),
		raw + "vllm:time_to_first_token_seconds_count{model_name=\"m\",engine=\"0\"} 2\n",
	} {
		if _, err := ReadServingExposition(context.Background(), strings.NewReader(mutated), vllmV1Source(), WorkloadLimits{}); err == nil {
			t.Fatal("accepted malformed/duplicate histogram")
		}
	}
}

func TestVLLMV1LegacyBytes(t *testing.T) {
	// This literal is run unchanged against the predecessor in the host batch.
	a, b, o := workloadOTLPFixture(t)
	raw, err := json.Marshal(a.Source)
	if err != nil {
		t.Fatal(err)
	}
	sourceHash := sha256.Sum256(raw)
	if hex.EncodeToString(sourceHash[:]) != "4e8367357d94baf718cc3e617064c1cfb9941d58dda5672ee96aabb20ac00b88" {
		t.Fatalf("legacy source JSON SHA256 %x", sourceHash)
	}
	if bytes.Contains(raw, []byte("Engine")) {
		t.Fatal("legacy source JSON changed")
	}
	var out bytes.Buffer
	if err := WriteWorkloadOTLP(&out, a, b, o); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out.String(), "engine_contract") {
		t.Fatal("legacy OTLP attributes changed")
	}
	h := sha256.Sum256(out.Bytes())
	// Filled from the retained predecessor control, never from V1 output.
	const want = "d50d2e3352cb852a37e81c08fb2f6a9af0e60722817dee0c43a6fcb8e4437010"
	if hex.EncodeToString(h[:]) != want {
		t.Fatalf("legacy OTLP SHA256 %x want %s", h, want)
	}
}
