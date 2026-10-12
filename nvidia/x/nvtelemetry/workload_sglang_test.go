package nvtelemetry

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestSGLangV0521ConstructedControl(t *testing.T) {
	raw := workloadFixture(t, "sglang-v0.5.21.prom.control")
	e, err := ReadServingExposition(context.Background(), bytes.NewReader(raw), WorkloadSource{Provider: "sglang", Version: "0.5.21", RunID: "control"}, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(raw)
	if !bytes.Equal(e.Raw, raw) || e.RawSHA256 != hex.EncodeToString(h[:]) {
		t.Fatal("source bytes or digest changed")
	}
	if len(e.Metrics) != 21 || len(e.UnmappedNames) != 0 {
		t.Fatalf("mapped=%d unmapped=%v", len(e.Metrics), e.UnmappedNames)
	}
	byName := make(map[string]WorkloadMetric)
	for _, p := range e.Metrics {
		byName[p.Name+"/"+p.Kind] = p
		if p.HasTime {
			t.Fatalf("scrape timestamp was invented for %s", p.Name)
		}
	}
	if p := byName["sglang:prompt_tokens_total/counter"]; p.RawValue != "9007199254740993" || p.Unit != "tokens" || p.Labels["is_streaming"] != "true" || p.Labels["dp_rank"] != "0" {
		t.Fatalf("source values/labels lost: %+v", p)
	}
	if p := byName["sglang:num_requests_total/counter"]; p.Semantic != "processed_requests" || p.Labels["model_name"] != "control-model" {
		t.Fatalf("request aggregate misclassified: %+v", p)
	}
	if p := byName["sglang:cache_hit_rate/gauge"]; p.Unit != "unknown" {
		t.Fatalf("undocumented scale inferred: %+v", p)
	}
	if p := byName["sglang:kv_transfer_speed_gb_s_sum/histogram_sum"]; p.Unit != "gigabytes_per_second" || p.RawValue != "2.5" {
		t.Fatalf("transfer histogram unit or sum lost: point=%+v metrics=%v", p, byName)
	}
	if p := byName["sglang:time_to_first_token_seconds_bucket/histogram_bucket"]; p.Name != "sglang:time_to_first_token_seconds_bucket" || p.Labels["is_streaming"] != "true" {
		t.Fatalf("histogram sample identity or family labels changed: %+v", p)
	}
	if ExactWorkloadJoin(e) == nil {
		t.Fatal("aggregate scrape admitted as exact request evidence")
	}
}

func TestServingMetricsFetchWorkloadPinnedProviders(t *testing.T) {
	for _, tc := range []struct {
		provider, version, file string
	}{
		{"sglang", "0.5.21", "sglang-v0.5.21.prom.control"},
		{"dynamo", "1.5.1", "dynamo-v1.5.1.prom.control"},
	} {
		t.Run(tc.provider, func(t *testing.T) {
			body := workloadFixture(t, tc.file)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/metrics" {
					t.Errorf("unexpected path %q", r.URL.Path)
				}
				_, _ = w.Write(body)
			}))
			defer server.Close()
			req, err := http.NewRequest(http.MethodGet, server.URL+"/metrics", nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := FetchWorkload(context.Background(), server.Client(), req, WorkloadSource{Provider: tc.provider, Version: tc.version, RunID: "control"}, WorkloadLimits{})
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got.Export.Raw, body) || len(got.Export.Metrics) == 0 {
				t.Fatal("public HTTP path changed or dropped producer bytes")
			}
		})
	}
	var called atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { called.Store(true) }))
	defer server.Close()
	req, err := http.NewRequest(http.MethodGet, server.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = FetchWorkload(ctx, server.Client(), req, WorkloadSource{Provider: "sglang", Version: "0.5.21", RunID: "control"}, WorkloadLimits{})
	if err != context.Canceled || called.Load() {
		t.Fatalf("cancelled request transmitted: err=%v called=%v", err, called.Load())
	}
}

func TestSGLangAllPinnedMappingsAndRefusals(t *testing.T) {
	for name, mapping := range sglangWorkloadMappings {
		t.Run(name, func(t *testing.T) {
			labels := sglangControlLabels(name)
			raw := oneWorkloadMetric(name, mapping, labels)
			e, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "sglang", Version: "0.5.21", RunID: "control"}, WorkloadLimits{})
			if err != nil {
				t.Fatal(err)
			}
			if len(e.Metrics) == 0 || len(e.UnmappedNames) != 0 {
				t.Fatalf("mapped=%d unmapped=%v", len(e.Metrics), e.UnmappedNames)
			}
		})
	}
	valid := string(workloadFixture(t, "sglang-v0.5.21.prom.control"))
	for _, tc := range []struct{ name, old, new string }{
		{"wrong_type", "sglang:num_running_reqs gauge", "sglang:num_running_reqs counter"},
		{"missing_identity", `,moe_ep_rank="0"`, ""},
		{"fractional_count", `sglang:num_requests_total{model_name="control-model",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0",dp_rank="0",is_streaming="true"} 3`, `sglang:num_requests_total{model_name="control-model",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0",dp_rank="0",is_streaming="true"} 3.5`},
		{"negative_count", `sglang:num_requests_total{model_name="control-model",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0",dp_rank="0",is_streaming="true"} 3`, `sglang:num_requests_total{model_name="control-model",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0",dp_rank="0",is_streaming="true"} -3`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mutated := strings.Replace(valid, tc.old, tc.new, 1)
			if mutated == valid {
				t.Fatal("mutation did not apply")
			}
			if _, err := ReadServingExposition(context.Background(), strings.NewReader(mutated), WorkloadSource{Provider: "sglang", Version: "0.5.21", RunID: "control"}, WorkloadLimits{}); err == nil {
				t.Fatal("invalid source contract accepted")
			}
		})
	}
	if _, err := ReadServingExposition(context.Background(), strings.NewReader(valid), WorkloadSource{Provider: "sglang", Version: "0.5.22", RunID: "control"}, WorkloadLimits{}); err == nil {
		t.Fatal("unknown SGLang version accepted")
	}
	if _, err := ReadServingExposition(context.Background(), strings.NewReader(valid), WorkloadSource{Provider: "sglang", Version: "0.5.21", RunID: "control"}, WorkloadLimits{MaxBytes: 32}); err == nil {
		t.Fatal("input byte limit ignored")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ReadServingExposition(ctx, strings.NewReader(valid), WorkloadSource{Provider: "sglang", Version: "0.5.21", RunID: "control"}, WorkloadLimits{}); err != context.Canceled {
		t.Fatalf("cancellation: %v", err)
	}
	unknown := strings.Replace(valid, "# TYPE sglang:num_running_reqs gauge", "# TYPE sglang:future_metric gauge\nsglang:future_metric{source_label=\"keep\"} 1\n# TYPE sglang:num_running_reqs gauge", 1)
	e, err := ReadServingExposition(context.Background(), strings.NewReader(unknown), WorkloadSource{Provider: "sglang", Version: "0.5.21", RunID: "control"}, WorkloadLimits{})
	if err != nil || len(e.UnmappedNames) != 1 || e.UnmappedNames[0] != "sglang:future_metric" || !bytes.Contains(e.Raw, []byte(`source_label="keep"`)) {
		t.Fatalf("unknown family not preserved: export=%+v err=%v", e, err)
	}
}

func sglangControlLabels(name string) string {
	labels := `model_name="m",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0"`
	switch name {
	case "sglang:num_requests_total", "sglang:prompt_tokens_total", "sglang:generation_tokens_total", "sglang:time_to_first_token_seconds", "sglang:e2e_request_latency_seconds":
		return labels + `,is_streaming="true"`
	case "sglang:realtime_tokens_total":
		return labels + `,mode="decode"`
	case "sglang:prefill_effective_tokens_total":
		return labels + `,mode="input"`
	case "sglang:cached_tokens_total":
		return labels + `,cache_source="device"`
	case "sglang:per_stage_req_latency_seconds":
		return labels + `,stage="prefill"`
	case "sglang:graph_memory_usage_gb":
		return labels + `,phase="capture"`
	case "sglang:storage_prefetch_unfulfilled_tokens_total":
		return labels + `,reason="dropped"`
	case "sglang:storage_prefetch_deferred_tokens_total":
		return labels + `,reason="host_capacity"`
	case "sglang:load_back_tokens_total", "sglang:hicache_backup_tokens_total":
		return labels + `,pool="kv"`
	case "sglang:hicache_dropped_tokens_total":
		return labels + `,reason="host_pressure",pool="kv"`
	default:
		return labels
	}
}

func TestSGLangHistogramDimensionsOnEveryComponent(t *testing.T) {
	for _, tc := range []struct{ metric, dimension string }{
		{"sglang:time_to_first_token_seconds", "is_streaming"},
		{"sglang:e2e_request_latency_seconds", "is_streaming"},
		{"sglang:per_stage_req_latency_seconds", "stage"},
	} {
		for _, component := range []string{"bucket", "sum", "count"} {
			t.Run(tc.metric+"/"+component, func(t *testing.T) {
				labels := sglangControlLabels(tc.metric)
				raw := histogramMissingDimension(tc.metric, labels, tc.dimension, component)
				_, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "sglang", Version: "0.5.21", RunID: "control"}, WorkloadLimits{})
				if err == nil || !strings.Contains(err.Error(), "missing SGLang source identity label") {
					t.Fatalf("missing %s on _%s was not refused by family validation: %v", tc.dimension, component, err)
				}
			})
		}
	}
}

func TestSGLangDimensionEnumsAndPresence(t *testing.T) {
	for _, tc := range []struct{ metric, labels, want string }{
		{"sglang:num_requests_total", `model_name="m",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0",is_streaming="maybe"`, "invalid SGLang is_streaming"},
		{"sglang:realtime_tokens_total", `model_name="m",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0",mode="input"`, "invalid SGLang mode"},
		{"sglang:prefill_effective_tokens_total", `model_name="m",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0",mode="decode"`, "invalid SGLang mode"},
		{"sglang:graph_memory_usage_gb", `model_name="m",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0"`, "missing SGLang source identity"},
	} {
		raw := oneWorkloadMetric(tc.metric, sglangWorkloadMappings[tc.metric], tc.labels)
		_, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "sglang", Version: "0.5.21", RunID: "control"}, WorkloadLimits{})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s accepted invalid dimension: %v", tc.metric, err)
		}
	}
	for _, tc := range []struct{ metric, dimension, value string }{
		{"sglang:num_requests_total", "is_streaming", "true"},
		{"sglang:num_requests_total", "is_streaming", "false"},
		{"sglang:realtime_tokens_total", "mode", "prefill_compute"},
		{"sglang:realtime_tokens_total", "mode", "prefill_cache"},
		{"sglang:realtime_tokens_total", "mode", "decode"},
		{"sglang:prefill_effective_tokens_total", "mode", "input"},
		{"sglang:prefill_effective_tokens_total", "mode", "device_hit"},
		{"sglang:prefill_effective_tokens_total", "mode", "host_hit"},
		{"sglang:prefill_effective_tokens_total", "mode", "storage_hit"},
	} {
		labels := `model_name="m",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0"` + "," + tc.dimension + `="` + tc.value + `"`
		raw := oneWorkloadMetric(tc.metric, sglangWorkloadMappings[tc.metric], labels)
		if _, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "sglang", Version: "0.5.21", RunID: "control"}, WorkloadLimits{}); err != nil {
			t.Errorf("source-defined %s=%s refused: %v", tc.dimension, tc.value, err)
		}
	}
}

func TestServingMetricsFetchWorkloadRefusesInvalidDimensions(t *testing.T) {
	for _, tc := range []struct{ provider, version, raw string }{
		{"sglang", "0.5.21", oneWorkloadMetric("sglang:per_stage_req_latency_seconds", sglangWorkloadMappings["sglang:per_stage_req_latency_seconds"], `model_name="m",engine_type="decode",tp_rank="0",pp_rank="0",moe_ep_rank="0"`)},
		{"dynamo", "1.5.1", oneWorkloadMetric("dynamo_frontend_model_migration_duration_seconds", dynamoWorkloadMappings["dynamo_frontend_model_migration_duration_seconds"], `model="m",migration_type="new_request"`)},
		{"dynamo", "1.5.1", oneWorkloadMetric("dynamo_frontend_tokenizer_cache_cached_tokens_total", dynamoWorkloadMappings["dynamo_frontend_tokenizer_cache_cached_tokens_total"], ``)},
	} {
		var calls atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			_, _ = w.Write([]byte(tc.raw))
		}))
		req, err := http.NewRequest(http.MethodGet, server.URL+"/metrics", nil)
		if err != nil {
			server.Close()
			t.Fatal(err)
		}
		_, err = FetchWorkload(context.Background(), server.Client(), req, WorkloadSource{Provider: tc.provider, Version: tc.version, RunID: "control"}, WorkloadLimits{})
		server.Close()
		if err == nil || calls.Load() != 1 {
			t.Errorf("%s invalid dimensions: err=%v transport calls=%d", tc.provider, err, calls.Load())
		}
	}
}

func histogramMissingDimension(name, labels, dimension, component string) string {
	base := withoutWorkloadLabel(labels, dimension)
	all := map[string]string{"bucket": labels, "sum": labels, "count": labels}
	all[component] = base
	return fmt.Sprintf("# TYPE %s histogram\n%s_bucket{%s,le=\"+Inf\"} 1\n%s_sum{%s} 1\n%s_count{%s} 1\n", name, name, all["bucket"], name, all["sum"], name, all["count"])
}

func withoutWorkloadLabel(labels, name string) string {
	parts := strings.Split(labels, ",")
	kept := parts[:0]
	for _, part := range parts {
		if !strings.HasPrefix(part, name+"=") {
			kept = append(kept, part)
		}
	}
	return strings.Join(kept, ",")
}

func oneWorkloadMetric(name string, mapping workloadMapping, labels string) string {
	labelSet := ""
	if labels != "" {
		labelSet = "{" + labels + "}"
	}
	if mapping.kind == "histogram" {
		bucketLabels := `le="+Inf"`
		if labels != "" {
			bucketLabels = labels + `,le="+Inf"`
		}
		return fmt.Sprintf("# TYPE %s histogram\n%s_bucket{%s} 1\n%s_sum%s 1\n%s_count%s 1\n", name, name, bucketLabels, name, labelSet, name, labelSet)
	}
	return fmt.Sprintf("# TYPE %s %s\n%s%s 1\n", name, mapping.kind, name, labelSet)
}
