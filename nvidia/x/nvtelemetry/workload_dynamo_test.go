package nvtelemetry

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"testing"
)

func TestDynamoV151ConstructedControl(t *testing.T) {
	raw := workloadFixture(t, "dynamo-v1.5.1.prom.control")
	e, err := ReadServingExposition(context.Background(), bytes.NewReader(raw), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(raw)
	if !bytes.Equal(e.Raw, raw) || e.RawSHA256 != hex.EncodeToString(h[:]) {
		t.Fatal("source bytes or digest changed")
	}
	if len(e.Metrics) != 25 || len(e.UnmappedNames) != 0 {
		t.Fatalf("mapped=%d unmapped=%v", len(e.Metrics), e.UnmappedNames)
	}
	workers := map[string]bool{}
	for _, p := range e.Metrics {
		if p.Name == "dynamo_component_requests_total" {
			workers[p.Labels["worker_id"]+"/"+p.Labels["dynamo_component"]] = true
		}
		if p.HasTime {
			t.Fatalf("scrape timestamp invented for %s", p.Name)
		}
	}
	if !workers["worker-a/prefill"] || !workers["worker-b/backend"] {
		t.Fatalf("worker roles or identities collapsed: %v", workers)
	}
	if p := findWorkloadMetric(e, "dynamo_component_gpu_cache_usage_percent", "gauge"); p.Unit != "1" || p.RawValue != "0.75" {
		t.Fatalf("source fraction was converted or mislabeled: %+v", p)
	}
	if p := findWorkloadMetric(e, "dynamo_component_total_blocks", "gauge"); p.Unit != "blocks" || p.RawValue != "128" || p.Labels["model"] != "control-model" || p.Labels["dp_rank"] != "0" {
		t.Fatalf("engine block gauge dimensions or value changed: %+v", p)
	}
	if p := findWorkloadMetric(e, "dynamo_component_model_load_time_seconds", "gauge"); p.Unit != "seconds" || p.Labels["model"] != "control-model" {
		t.Fatalf("engine model-load gauge dimensions or unit changed: %+v", p)
	}
	if p := findWorkloadMetric(e, "dynamo_frontend_time_to_first_token_seconds_bucket", "histogram_bucket"); p.Name != "dynamo_frontend_time_to_first_token_seconds_bucket" || p.Labels["model"] != "control-model" {
		t.Fatalf("histogram sample identity or family labels changed: %+v", p)
	}
	if p := findWorkloadMetric(e, "dynamo_component_cancellation_total", "counter"); p.Semantic != "worker_cancellation_signals" {
		t.Fatalf("cancellation signal became engine abort: %+v", p)
	}
	if p := findWorkloadMetric(e, "dynamo_frontend_model_cancellation_total", "counter"); p.Labels["endpoint"] != "chat_completions" || p.Labels["request_type"] != "stream" {
		t.Fatalf("frontend dimensions lost: %+v", p)
	}
	if ExactWorkloadJoin(e) == nil {
		t.Fatal("aggregate worker/frontend metrics admitted as request records")
	}
}

func TestDynamoAllPinnedMappingsAndRefusals(t *testing.T) {
	for name, mapping := range dynamoWorkloadMappings {
		t.Run(name, func(t *testing.T) {
			labels := dynamoControlLabels(name)
			raw := oneWorkloadMetric(name, mapping, labels)
			e, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{})
			if err != nil {
				t.Fatal(err)
			}
			if len(e.Metrics) == 0 || len(e.UnmappedNames) != 0 {
				t.Fatalf("mapped=%d unmapped=%v", len(e.Metrics), e.UnmappedNames)
			}
		})
	}
	valid := string(workloadFixture(t, "dynamo-v1.5.1.prom.control"))
	for _, tc := range []struct{ name, old, new string }{
		{"wrong_type", "dynamo_component_inflight_requests gauge", "dynamo_component_inflight_requests counter"},
		{"missing_hierarchy", `dynamo_namespace="control",`, ""},
		{"missing_worker_identity", `worker_id="worker-b",dp_rank="0",worker_type="decode"`, `worker_id="worker-b",dp_rank="0"`},
		{"invalid_worker_role", `worker_id="worker-b",dp_rank="0",worker_type="decode"`, `worker_id="worker-b",dp_rank="0",worker_type="unknown"`},
		{"fractional_count", `dynamo_component_requests_total{dynamo_namespace="control",dynamo_component="prefill",dynamo_endpoint="generate",worker_id="worker-a"} 7`, `dynamo_component_requests_total{dynamo_namespace="control",dynamo_component="prefill",dynamo_endpoint="generate",worker_id="worker-a"} 7.5`},
		{"fraction_not_ratio", `dynamo_component_gpu_cache_usage_percent{dynamo_namespace="control",dynamo_component="backend",dynamo_endpoint="generate",model="control-model",dp_rank="0",worker_id="worker-b"} 0.75`, `dynamo_component_gpu_cache_usage_percent{dynamo_namespace="control",dynamo_component="backend",dynamo_endpoint="generate",model="control-model",dp_rank="0",worker_id="worker-b"} 75`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mutated := strings.Replace(valid, tc.old, tc.new, 1)
			if mutated == valid {
				t.Fatal("mutation did not apply")
			}
			if _, err := ReadServingExposition(context.Background(), strings.NewReader(mutated), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{}); err == nil {
				t.Fatal("invalid source contract accepted")
			}
		})
	}
	for _, labels := range []string{`stage="unknown",phase="prefill"`, `stage="route",phase="unknown"`} {
		raw := "# TYPE dynamo_frontend_stage_requests gauge\ndynamo_frontend_stage_requests{" + labels + "} 1\n"
		if _, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{}); err == nil {
			t.Fatalf("invalid stage labels accepted: %s", labels)
		}
	}
	for _, tc := range []struct{ name, labels, value string }{
		{"dynamo_component_errors_total", `dynamo_namespace="n",dynamo_component="backend",dynamo_endpoint="generate",error_type="unknown"`, "1"},
		{"dynamo_component_router_non_max_overlap_selections_total", `dynamo_namespace="n",dynamo_component="router",dynamo_endpoint="route",worker_type="decode"`, "1"},
		{"dynamo_component_kv_cache_events_applied", `dynamo_namespace="n",dynamo_component="router",dynamo_endpoint="kv",status="unknown",event_type="stored"`, "1"},
		{"dynamo_frontend_model_ready", `model="m"`, "2"},
		{"dynamo_frontend_model_cancellation_total", `model="m",request_type="stream"`, "1"},
	} {
		mapping := dynamoWorkloadMappings[tc.name]
		raw := oneWorkloadMetric(tc.name, mapping, tc.labels)
		if mapping.kind != "histogram" {
			raw = strings.Replace(raw, " 1\n", " "+tc.value+"\n", 1)
		}
		if _, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{}); err == nil {
			t.Fatalf("invalid labels/value accepted for %s", tc.name)
		}
	}
	if _, err := ReadServingExposition(context.Background(), strings.NewReader(valid), WorkloadSource{Provider: "dynamo", Version: "1.5.2", RunID: "control"}, WorkloadLimits{}); err == nil {
		t.Fatal("unknown Dynamo version accepted")
	}
	if _, err := ReadServingExposition(context.Background(), strings.NewReader(valid), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{MaxBytes: 16}); err == nil {
		t.Fatal("input byte limit ignored")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ReadServingExposition(ctx, strings.NewReader(valid), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{}); err != context.Canceled {
		t.Fatalf("cancellation: %v", err)
	}
	unknown := strings.Replace(valid, "# TYPE dynamo_frontend_requests_total counter", "# TYPE dynamo_frontend_new_family gauge\ndynamo_frontend_new_family{custom=\"kept\"} 1\n# TYPE dynamo_frontend_requests_total counter", 1)
	e, err := ReadServingExposition(context.Background(), strings.NewReader(unknown), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{})
	if err != nil || len(e.UnmappedNames) != 1 || e.UnmappedNames[0] != "dynamo_frontend_new_family" || !bytes.Contains(e.Raw, []byte(`custom="kept"`)) {
		t.Fatalf("unknown family not preserved: export=%+v err=%v", e, err)
	}
	// An absent family is not emitted as a fabricated zero-valued metric.
	withoutCancellation := strings.Replace(valid, "# TYPE dynamo_frontend_model_cancellation_total counter\ndynamo_frontend_model_cancellation_total{model=\"control-model\",endpoint=\"chat_completions\",request_type=\"stream\"} 1\n", "", 1)
	e, err = ReadServingExposition(context.Background(), strings.NewReader(withoutCancellation), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{})
	if err != nil || findWorkloadMetric(e, "dynamo_frontend_model_cancellation_total", "counter").Name != "" {
		t.Fatalf("missing family became zero or failed parse: err=%v", err)
	}
}

func dynamoControlLabels(name string) string {
	switch {
	case name == "dynamo_component_kv_cache_events_applied":
		return `dynamo_namespace="n",dynamo_component="router",dynamo_endpoint="kv_indexer",status="ok",event_type="stored"`
	case name == "dynamo_component_errors_total":
		return `dynamo_namespace="n",dynamo_component="backend",dynamo_endpoint="generate",error_type="generate"`
	case name == "dynamo_component_router_non_max_overlap_selections_total", name == "dynamo_component_router_overlap_blocks_lost":
		return `dynamo_namespace="n",dynamo_component="router",dynamo_endpoint="route",worker_type="prefill"`
	case strings.HasPrefix(name, "dynamo_component_"):
		labels := `dynamo_namespace="n",dynamo_component="backend",dynamo_endpoint="generate"`
		if name == "dynamo_component_gpu_cache_usage_percent" || name == "dynamo_component_total_blocks" {
			labels += `,model="m",dp_rank="0"`
		} else if name == "dynamo_component_model_load_time_seconds" {
			labels += `,model="m"`
		}
		return labels
	case strings.HasPrefix(name, "dynamo_frontend_worker_"):
		return `worker_id="w",dp_rank="0",worker_type="decode"`
	case name == "dynamo_frontend_stage_requests":
		return `stage="route",phase="prefill"`
	case strings.HasPrefix(name, "dynamo_frontend_router_queue_"):
		if name == "dynamo_frontend_router_queue_backpressure_total" {
			return `model="m",worker_type="prefill",policy_class="default",reason="capacity"`
		}
		return `model="m",worker_type="prefill",policy_class="default"`
	case strings.Contains(name, "migration_duration"):
		return `model="m",migration_type="new_request",outcome="success"`
	case strings.Contains(name, "model_cancellation"):
		return `model="m",endpoint="chat_completions",request_type="stream"`
	case strings.Contains(name, "model_rejection"):
		return `model="m",endpoint="chat_completions"`
	case strings.Contains(name, "migration"):
		return `model="m",migration_type="new_request"`
	case strings.Contains(name, "tokenizer_cache_cached_tokens") || strings.Contains(name, "tokenizer_cache_uncached_tokens"):
		return `model="m"`
	case name == "dynamo_frontend_requests_total" || name == "dynamo_frontend_output_tokens_total" || name == "dynamo_frontend_request_duration_seconds" || name == "dynamo_frontend_time_to_first_token_seconds" || name == "dynamo_frontend_inter_token_latency_seconds" || name == "dynamo_frontend_input_sequence_tokens" || name == "dynamo_frontend_cached_tokens" || name == "dynamo_frontend_output_sequence_tokens":
		return `model="m"`
	case strings.HasPrefix(name, "dynamo_frontend_model_"):
		return `model="m"`
	default:
		return ""
	}
}

func TestDynamoHistogramDimensionsOnEveryComponent(t *testing.T) {
	for _, tc := range []struct{ metric, dimension string }{
		{"dynamo_frontend_model_migration_duration_seconds", "migration_type"},
		{"dynamo_frontend_model_migration_duration_seconds", "outcome"},
		{"dynamo_component_router_overlap_blocks_lost", "worker_type"},
	} {
		for _, component := range []string{"bucket", "sum", "count"} {
			t.Run(tc.metric+"/"+tc.dimension+"/"+component, func(t *testing.T) {
				labels := dynamoControlLabels(tc.metric)
				raw := histogramMissingDimension(tc.metric, labels, tc.dimension, component)
				_, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{})
				if err == nil || !strings.Contains(err.Error(), "missing Dynamo source identity label") {
					t.Fatalf("missing %s on _%s was not refused by family validation: %v", tc.dimension, component, err)
				}
			})
		}
	}
}

func TestDynamoSourceDimensionsAndEnums(t *testing.T) {
	for _, tc := range []struct{ metric, labels, want string }{
		{"dynamo_frontend_model_migration_duration_seconds", `model="m",migration_type="unknown",outcome="success"`, "invalid Dynamo migration_type"},
		{"dynamo_frontend_model_migration_duration_seconds", `model="m",migration_type="new_request",outcome="unknown"`, "invalid Dynamo outcome"},
		{"dynamo_component_router_overlap_blocks_lost", `dynamo_namespace="n",dynamo_component="router",dynamo_endpoint="route",worker_type="decode"`, "invalid Dynamo worker_type"},
		{"dynamo_frontend_tokenizer_cache_cached_tokens_total", ``, "missing Dynamo source identity"},
		{"dynamo_frontend_tokenizer_cache_uncached_tokens_total", ``, "missing Dynamo source identity"},
		{"dynamo_component_gpu_cache_usage_percent", `dynamo_namespace="n",dynamo_component="worker",dynamo_endpoint="generate",dp_rank="0"`, "missing Dynamo source identity"},
		{"dynamo_component_gpu_cache_usage_percent", `dynamo_namespace="n",dynamo_component="worker",dynamo_endpoint="generate",model="m"`, "missing Dynamo source identity"},
		{"dynamo_component_gpu_cache_usage_percent", `dynamo_namespace="n",dynamo_component="worker",dynamo_endpoint="generate",model="m",dp_rank="bad"`, "invalid Dynamo dp_rank"},
		{"dynamo_component_total_blocks", `dynamo_namespace="n",dynamo_component="worker",dynamo_endpoint="generate",model="m"`, "missing Dynamo source identity"},
		{"dynamo_component_total_blocks", `dynamo_namespace="n",dynamo_component="worker",dynamo_endpoint="generate",dp_rank="0"`, "missing Dynamo source identity"},
		{"dynamo_component_model_load_time_seconds", `dynamo_namespace="n",dynamo_component="worker",dynamo_endpoint="generate"`, "missing Dynamo source identity"},
	} {
		raw := oneWorkloadMetric(tc.metric, dynamoWorkloadMappings[tc.metric], tc.labels)
		_, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s accepted invalid dimension: %v", tc.metric, err)
		}
	}
	for _, migrationType := range []string{"new_request", "ongoing_request"} {
		for _, outcome := range []string{"success", "failure", "cancelled"} {
			labels := fmt.Sprintf(`model="m",migration_type="%s",outcome="%s"`, migrationType, outcome)
			raw := oneWorkloadMetric("dynamo_frontend_model_migration_duration_seconds", dynamoWorkloadMappings["dynamo_frontend_model_migration_duration_seconds"], labels)
			if _, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{}); err != nil {
				t.Errorf("source-defined migration dimensions %s/%s refused: %v", migrationType, outcome, err)
			}
		}
	}
	for _, value := range []string{"0", "1"} {
		raw := fmt.Sprintf("# TYPE dynamo_component_gpu_cache_usage_percent gauge\ndynamo_component_gpu_cache_usage_percent{dynamo_namespace=\"n\",dynamo_component=\"backend\",dynamo_endpoint=\"generate\",model=\"m\",dp_rank=\"0\"} %s\n", value)
		if _, err := ReadServingExposition(context.Background(), strings.NewReader(raw), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{}); err != nil {
			t.Errorf("valid cache fraction %s refused: %v", value, err)
		}
	}
	fractionalBlocks := "# TYPE dynamo_component_total_blocks gauge\ndynamo_component_total_blocks{dynamo_namespace=\"n\",dynamo_component=\"backend\",dynamo_endpoint=\"generate\",model=\"m\",dp_rank=\"0\"} 1.5\n"
	if _, err := ReadServingExposition(context.Background(), strings.NewReader(fractionalBlocks), WorkloadSource{Provider: "dynamo", Version: "1.5.1", RunID: "control"}, WorkloadLimits{}); err == nil {
		t.Fatal("fractional total KV blocks accepted")
	}
}

func findWorkloadMetric(e WorkloadExport, name, kind string) WorkloadMetric {
	for _, p := range e.Metrics {
		if p.Name == name && p.Kind == kind {
			return p
		}
	}
	return WorkloadMetric{}
}

func TestDynamoFixtureRequiresAllWorkersWithoutInventingOutcomes(t *testing.T) {
	raw := string(workloadFixture(t, "dynamo-v1.5.1.prom.control"))
	for _, role := range []string{"prefill", "backend"} {
		if !strings.Contains(raw, fmt.Sprintf(`dynamo_component="%s"`, role)) {
			t.Fatalf("constructed worker evidence missing role %s", role)
		}
	}
	if strings.Contains(raw, "request_id=") || strings.Contains(raw, "attempt=") {
		t.Fatal("aggregate control invented request identities")
	}
}
