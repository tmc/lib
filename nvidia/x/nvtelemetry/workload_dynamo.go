package nvtelemetry

import (
	"errors"
	"strconv"
	"strings"
)

// dynamoWorkloadMappings describes runtime worker/frontend metric families
// cataloged by Dynamo v1.5.1 and three source-pinned engine gauges.
var dynamoWorkloadMappings = map[string]workloadMapping{
	"dynamo_component_cancellation_total":                        {"worker_cancellation_signals", "requests", "counter"},
	"dynamo_component_errors_total":                              {"worker_errors", "errors", "counter"},
	"dynamo_component_gpu_cache_usage_percent":                   {"worker_gpu_cache_usage_fraction", "1", "gauge"},
	"dynamo_component_inflight_requests":                         {"worker_inflight_requests", "requests", "gauge"},
	"dynamo_component_model_load_time_seconds":                   {"worker_model_load_time", "seconds", "gauge"},
	"dynamo_component_request_bytes_total":                       {"worker_request_bytes", "bytes", "counter"},
	"dynamo_component_request_duration_seconds":                  {"worker_request_duration", "seconds", "histogram"},
	"dynamo_component_requests_total":                            {"worker_requests", "requests", "counter"},
	"dynamo_component_response_bytes_total":                      {"worker_response_bytes", "bytes", "counter"},
	"dynamo_component_total_blocks":                              {"worker_total_kv_blocks", "blocks", "gauge"},
	"dynamo_component_uptime_seconds":                            {"worker_runtime_uptime", "seconds", "gauge"},
	"dynamo_rejection_request_total":                             {"worker_admission_rejections", "requests", "counter"},
	"dynamo_engine_request":                                      {"engine_inflight_requests", "requests", "gauge"},
	"dynamo_request_queue":                                       {"dynamo_queued_requests", "requests", "gauge"},
	"dynamo_frontend_active_requests":                            {"frontend_active_requests", "requests", "gauge"},
	"dynamo_frontend_stage_requests":                             {"frontend_stage_requests", "requests", "gauge"},
	"dynamo_frontend_disconnected_clients":                       {"frontend_disconnected_clients", "clients", "gauge"},
	"dynamo_frontend_input_sequence_tokens":                      {"frontend_input_sequence_length", "tokens", "histogram"},
	"dynamo_frontend_cached_tokens":                              {"frontend_cached_tokens", "tokens", "histogram"},
	"dynamo_frontend_tokenizer_cache_hits_total":                 {"frontend_tokenizer_cache_hits", "operations", "counter"},
	"dynamo_frontend_tokenizer_cache_misses_total":               {"frontend_tokenizer_cache_misses", "operations", "counter"},
	"dynamo_frontend_tokenizer_cache_cached_tokens_total":        {"frontend_tokenizer_cache_tokens_hit", "tokens", "counter"},
	"dynamo_frontend_tokenizer_cache_uncached_tokens_total":      {"frontend_tokenizer_cache_tokens_encoded", "tokens", "counter"},
	"dynamo_frontend_inter_token_latency_seconds":                {"frontend_inter_token_latency", "seconds", "histogram"},
	"dynamo_frontend_output_sequence_tokens":                     {"frontend_output_sequence_length", "tokens", "histogram"},
	"dynamo_frontend_output_tokens_total":                        {"frontend_output_tokens", "tokens", "counter"},
	"dynamo_frontend_request_duration_seconds":                   {"frontend_request_duration", "seconds", "histogram"},
	"dynamo_frontend_requests_total":                             {"frontend_requests", "requests", "counter"},
	"dynamo_frontend_time_to_first_token_seconds":                {"frontend_time_to_first_token", "seconds", "histogram"},
	"dynamo_frontend_model_migration_total":                      {"frontend_request_migrations", "requests", "counter"},
	"dynamo_frontend_model_migration_duration_seconds":           {"frontend_request_migration_duration", "seconds", "histogram"},
	"dynamo_frontend_model_migration_max_seq_len_exceeded_total": {"frontend_migrations_disabled_max_sequence", "requests", "counter"},
	"dynamo_frontend_model_cancellation_total":                   {"frontend_request_cancellations", "requests", "counter"},
	"dynamo_frontend_model_rejection_total":                      {"frontend_request_rejections", "requests", "counter"},
	"dynamo_frontend_worker_active_decode_blocks":                {"frontend_worker_active_decode_blocks", "blocks", "gauge"},
	"dynamo_frontend_worker_active_prefill_tokens":               {"frontend_worker_active_prefill_tokens", "tokens", "gauge"},
	"dynamo_frontend_worker_last_time_to_first_token_seconds":    {"frontend_worker_last_time_to_first_token", "seconds", "gauge"},
	"dynamo_frontend_worker_last_input_sequence_tokens":          {"frontend_worker_last_input_sequence_length", "tokens", "gauge"},
	"dynamo_frontend_worker_last_inter_token_latency_seconds":    {"frontend_worker_last_inter_token_latency", "seconds", "gauge"},
	"dynamo_frontend_model_ready":                                {"frontend_model_ready", "boolean", "gauge"},
	"dynamo_frontend_model_total_kv_blocks":                      {"frontend_model_total_kv_blocks", "blocks", "gauge"},
	"dynamo_frontend_model_max_num_seqs":                         {"frontend_model_max_sequences", "requests", "gauge"},
	"dynamo_frontend_model_max_num_batched_tokens":               {"frontend_model_max_batched_tokens", "tokens", "gauge"},
	"dynamo_frontend_model_context_length":                       {"frontend_model_context_length", "tokens", "gauge"},
	"dynamo_frontend_model_kv_cache_block_size":                  {"frontend_model_kv_cache_block_size", "tokens", "gauge"},
	"dynamo_frontend_model_migration_limit":                      {"frontend_model_migration_limit", "requests", "gauge"},
	"dynamo_component_router_requests_started_total":             {"router_admitted_requests", "requests", "counter"},
	"dynamo_component_router_requests_total":                     {"router_processed_requests", "requests", "counter"},
	"dynamo_component_router_time_to_first_token_seconds":        {"router_time_to_first_token", "seconds", "histogram"},
	"dynamo_component_router_inter_token_latency_seconds":        {"router_inter_token_latency", "seconds", "histogram"},
	"dynamo_component_router_input_sequence_tokens":              {"router_input_sequence_length", "tokens", "histogram"},
	"dynamo_component_router_output_sequence_tokens":             {"router_output_sequence_length", "tokens", "histogram"},
	"dynamo_component_router_kv_hit_rate":                        {"router_predicted_kv_hit_rate", "1", "histogram"},
	"dynamo_component_router_non_max_overlap_selections_total":   {"router_non_max_overlap_selections", "requests", "counter"},
	"dynamo_component_router_overlap_blocks_lost":                {"router_kv_overlap_blocks_lost", "blocks", "histogram"},
	"dynamo_frontend_router_queue_pending_requests":              {"router_queue_pending_requests", "requests", "gauge"},
	"dynamo_frontend_router_queue_pending_isl_tokens":            {"router_queue_pending_input_tokens", "tokens", "gauge"},
	"dynamo_frontend_router_queue_pending_cached_tokens":         {"router_queue_pending_cached_tokens", "tokens", "gauge"},
	"dynamo_frontend_router_queue_backpressure_total":            {"router_queue_backpressure_rejections", "requests", "counter"},
	"dynamo_component_kv_cache_events_applied":                   {"router_kv_cache_events_applied", "events", "counter"},
}

var dynamoComponentRequiredLabels = []string{
	"dynamo_namespace", "dynamo_component", "dynamo_endpoint",
}

var dynamoWorkerRequiredLabels = []string{"worker_id", "dp_rank", "worker_type"}

func validateDynamoWorkloadLabels(name string, labels map[string]string) error {
	var required []string
	switch {
	case strings.HasPrefix(name, "dynamo_component_"):
		required = dynamoComponentRequiredLabels
	case strings.HasPrefix(name, "dynamo_frontend_worker_"):
		required = dynamoWorkerRequiredLabels
	case strings.HasPrefix(name, "dynamo_frontend_model_"):
		required = []string{"model"}
	case name == "dynamo_frontend_stage_requests":
		required = []string{"stage"}
	case strings.HasPrefix(name, "dynamo_frontend_router_queue_"):
		required = []string{"model", "worker_type", "policy_class"}
	case name == "dynamo_frontend_requests_total",
		name == "dynamo_frontend_output_tokens_total",
		name == "dynamo_frontend_request_duration_seconds",
		name == "dynamo_frontend_time_to_first_token_seconds",
		name == "dynamo_frontend_inter_token_latency_seconds",
		name == "dynamo_frontend_input_sequence_tokens",
		name == "dynamo_frontend_cached_tokens",
		name == "dynamo_frontend_output_sequence_tokens",
		name == "dynamo_frontend_model_cancellation_total",
		name == "dynamo_frontend_model_rejection_total",
		name == "dynamo_frontend_model_migration_total",
		name == "dynamo_frontend_model_migration_duration_seconds",
		name == "dynamo_frontend_model_migration_max_seq_len_exceeded_total":
		required = []string{"model"}
	}
	switch name {
	case "dynamo_component_gpu_cache_usage_percent", "dynamo_component_total_blocks":
		required = append(required, "model", "dp_rank")
	case "dynamo_component_model_load_time_seconds":
		required = append(required, "model")
	case "dynamo_frontend_tokenizer_cache_cached_tokens_total", "dynamo_frontend_tokenizer_cache_uncached_tokens_total":
		required = append(required, "model")
	case "dynamo_frontend_model_migration_total":
		required = append(required, "migration_type")
	case "dynamo_frontend_model_migration_duration_seconds":
		required = append(required, "migration_type", "outcome")
	case "dynamo_component_router_non_max_overlap_selections_total", "dynamo_component_router_overlap_blocks_lost":
		required = append(required, "worker_type")
	}
	for _, label := range required {
		if !hasWorkloadLabel(labels, label) {
			return errors.New("missing Dynamo source identity label")
		}
	}
	if name == "dynamo_frontend_stage_requests" {
		if _, ok := labels["phase"]; !ok {
			return errors.New("missing Dynamo source identity label")
		}
		if labels["stage"] != "preprocess" && labels["stage"] != "route" && labels["stage"] != "dispatch" {
			return errors.New("invalid Dynamo stage label")
		}
		if phase := labels["phase"]; phase != "" && phase != "prefill" && phase != "decode" && phase != "aggregated" {
			return errors.New("invalid Dynamo phase label")
		}
	}
	if strings.HasPrefix(name, "dynamo_frontend_router_queue_") {
		if labels["worker_type"] != "prefill" && labels["worker_type"] != "decode" {
			return errors.New("invalid Dynamo worker_type label")
		}
		if name == "dynamo_frontend_router_queue_backpressure_total" && labels["reason"] == "" {
			return errors.New("missing Dynamo source identity label")
		}
	}
	if name == "dynamo_component_kv_cache_events_applied" {
		validStatus := map[string]bool{"ok": true, "parent_block_not_found": true, "block_not_found": true, "invalid_block": true, "capacity_exhausted": true, "indexer_invariant_violation": true}
		if !validStatus[labels["status"]] {
			return errors.New("invalid Dynamo status label")
		}
		if labels["event_type"] != "stored" && labels["event_type"] != "removed" && labels["event_type"] != "cleared" {
			return errors.New("invalid Dynamo event_type label")
		}
	}
	if name == "dynamo_frontend_model_migration_duration_seconds" {
		if labels["migration_type"] != "new_request" && labels["migration_type"] != "ongoing_request" {
			return errors.New("invalid Dynamo migration_type label")
		}
		if labels["outcome"] != "success" && labels["outcome"] != "failure" && labels["outcome"] != "cancelled" {
			return errors.New("invalid Dynamo outcome label")
		}
	}
	if name == "dynamo_frontend_model_migration_total" && labels["migration_type"] != "new_request" && labels["migration_type"] != "ongoing_request" {
		return errors.New("invalid Dynamo migration_type label")
	}
	if name == "dynamo_frontend_model_cancellation_total" && labels["request_type"] != "unary" && labels["request_type"] != "stream" {
		return errors.New("invalid Dynamo request_type label")
	}
	if name == "dynamo_frontend_model_cancellation_total" || name == "dynamo_frontend_model_rejection_total" {
		if labels["endpoint"] == "" {
			return errors.New("missing Dynamo source identity label")
		}
	}
	if name == "dynamo_component_errors_total" {
		valid := map[string]bool{"deserialization": true, "invalid_message": true, "response_stream": true, "generate": true, "publish_response": true, "publish_final": true}
		if !valid[labels["error_type"]] {
			return errors.New("invalid Dynamo error_type label")
		}
	}
	if name == "dynamo_component_router_non_max_overlap_selections_total" || name == "dynamo_component_router_overlap_blocks_lost" {
		if labels["worker_type"] != "prefill" {
			return errors.New("invalid Dynamo worker_type label")
		}
	}
	if strings.HasPrefix(name, "dynamo_frontend_worker_") {
		if labels["worker_type"] != "prefill" && labels["worker_type"] != "decode" {
			return errors.New("invalid Dynamo worker_type label")
		}
		if _, err := strconv.ParseUint(labels["dp_rank"], 10, 32); err != nil {
			return errors.New("invalid Dynamo dp_rank label")
		}
	}
	if name == "dynamo_component_gpu_cache_usage_percent" || name == "dynamo_component_total_blocks" {
		if _, err := strconv.ParseUint(labels["dp_rank"], 10, 32); err != nil {
			return errors.New("invalid Dynamo dp_rank label")
		}
	}
	return nil
}
