package nvtelemetry

import (
	"errors"
)

// sglangWorkloadMappings describes selected aggregate metric families from
// SGLang v0.5.21. Unknown families remain in the raw export.
var sglangWorkloadMappings = map[string]workloadMapping{
	"sglang:num_requests_total":                             {"processed_requests", "requests", "counter"},
	"sglang:num_so_requests_total":                          {"structured_output_requests", "requests", "counter"},
	"sglang:num_aborted_requests_total":                     {"aborted_requests", "requests", "counter"},
	"sglang:num_retracted_requests_total":                   {"retracted_requests", "requests", "counter"},
	"sglang:num_bootstrap_failed_reqs_total":                {"bootstrap_failed_requests", "requests", "counter"},
	"sglang:num_transfer_failed_reqs_total":                 {"transfer_failed_requests", "requests", "counter"},
	"sglang:num_prefill_retries_total":                      {"prefill_retries", "requests", "counter"},
	"sglang:num_decode_host_receive_reqs_total":             {"host_receive_requests", "requests", "counter"},
	"sglang:prompt_tokens_total":                            {"processed_prefill_tokens", "tokens", "counter"},
	"sglang:generation_tokens_total":                        {"processed_generation_tokens", "tokens", "counter"},
	"sglang:cached_tokens_total":                            {"cached_tokens_by_source", "tokens", "counter"},
	"sglang:num_retracted_input_tokens_total":               {"retracted_input_tokens", "tokens", "counter"},
	"sglang:num_retracted_output_tokens_total":              {"retracted_output_tokens", "tokens", "counter"},
	"sglang:realtime_tokens_total":                          {"realtime_tokens", "tokens", "counter"},
	"sglang:prefill_effective_tokens_total":                 {"effective_prefill_tokens", "tokens", "counter"},
	"sglang:spec_verify_calls_total":                        {"speculative_verify_calls", "calls", "counter"},
	"sglang:num_running_reqs":                               {"running_requests", "requests", "gauge"},
	"sglang:num_queue_reqs":                                 {"queued_requests", "requests", "gauge"},
	"sglang:num_unique_running_routing_keys":                {"unique_running_routing_keys", "routing_keys", "gauge"},
	"sglang:decode_sum_seq_lens":                            {"decode_sequence_length_sum", "tokens", "gauge"},
	"sglang:max_running_requests_under_SLO":                 {"maximum_running_requests_under_slo", "requests", "gauge"},
	"sglang:num_grammar_queue_reqs":                         {"grammar_queued_requests", "requests", "gauge"},
	"sglang:num_prefill_bootstrap_queue_reqs":               {"prefill_bootstrap_queued_requests", "requests", "gauge"},
	"sglang:num_prefill_inflight_queue_reqs":                {"prefill_inflight_queued_requests", "requests", "gauge"},
	"sglang:num_decode_prealloc_queue_reqs":                 {"decode_prealloc_queued_requests", "requests", "gauge"},
	"sglang:num_decode_transfer_queue_reqs":                 {"decode_transfer_queued_requests", "requests", "gauge"},
	"sglang:num_decode_host_receive_queue_reqs":             {"decode_host_receive_queued_requests", "requests", "gauge"},
	"sglang:num_retracted_reqs":                             {"retracted_requests_current", "requests", "gauge"},
	"sglang:num_paused_reqs":                                {"paused_requests", "requests", "gauge"},
	"sglang:kv_available_tokens":                            {"kv_available_tokens", "tokens", "gauge"},
	"sglang:kv_evictable_tokens":                            {"kv_evictable_tokens", "tokens", "gauge"},
	"sglang:kv_used_tokens":                                 {"kv_used_tokens", "tokens", "gauge"},
	"sglang:num_used_tokens":                                {"used_tokens", "tokens", "gauge"},
	"sglang:pending_prealloc_token_usage":                   {"pending_prealloc_token_usage", "tokens", "gauge"},
	"sglang:max_total_num_tokens":                           {"max_total_tokens", "tokens", "gauge"},
	"sglang:swa_available_tokens":                           {"swa_available_tokens", "tokens", "gauge"},
	"sglang:swa_evictable_tokens":                           {"swa_evictable_tokens", "tokens", "gauge"},
	"sglang:swa_used_tokens":                                {"swa_used_tokens", "tokens", "gauge"},
	"sglang:mamba_available_tokens":                         {"mamba_available_slots", "slots", "gauge"},
	"sglang:mamba_evictable_tokens":                         {"mamba_evictable_slots", "slots", "gauge"},
	"sglang:mamba_used_tokens":                              {"mamba_used_slots", "slots", "gauge"},
	"sglang:token_usage":                                    {"token_usage", "unknown", "gauge"},
	"sglang:full_token_usage":                               {"full_attention_token_usage", "unknown", "gauge"},
	"sglang:swa_token_usage":                                {"swa_token_usage", "unknown", "gauge"},
	"sglang:mamba_usage":                                    {"mamba_usage", "unknown", "gauge"},
	"sglang:cache_hit_rate":                                 {"prefix_cache_hit_rate", "unknown", "gauge"},
	"sglang:gen_throughput":                                 {"generation_throughput", "tokens_per_second", "gauge"},
	"sglang:utilization":                                    {"utilization", "unknown", "gauge"},
	"sglang:new_token_ratio":                                {"new_token_ratio", "unknown", "gauge"},
	"sglang:time_to_first_token_seconds":                    {"time_to_first_token", "seconds", "histogram"},
	"sglang:inter_token_latency_seconds":                    {"inter_token_latency", "seconds", "histogram"},
	"sglang:e2e_request_latency_seconds":                    {"request_latency", "seconds", "histogram"},
	"sglang:queue_time_seconds":                             {"queue_time", "seconds", "histogram"},
	"sglang:per_stage_req_latency_seconds":                  {"per_stage_request_latency", "seconds", "histogram"},
	"sglang:prompt_tokens_histogram":                        {"prompt_token_length", "tokens", "histogram"},
	"sglang:uncached_prompt_tokens_histogram":               {"uncached_prompt_token_length", "tokens", "histogram"},
	"sglang:generation_tokens_histogram":                    {"generation_token_length", "tokens", "histogram"},
	"sglang:kv_transfer_speed_gb_s":                         {"kv_transfer_speed", "gigabytes_per_second", "histogram"},
	"sglang:kv_transfer_latency_ms":                         {"kv_transfer_latency", "milliseconds", "histogram"},
	"sglang:kv_transfer_bootstrap_ms":                       {"kv_transfer_bootstrap_time", "milliseconds", "histogram"},
	"sglang:kv_transfer_alloc_ms":                           {"kv_transfer_alloc_wait", "milliseconds", "histogram"},
	"sglang:kv_transfer_total_mb":                           {"kv_transfer_size", "megabytes", "histogram"},
	"sglang:hicache_host_used_tokens":                       {"host_kv_cache_used_tokens", "tokens", "gauge"},
	"sglang:hicache_host_total_tokens":                      {"host_kv_cache_capacity_tokens", "tokens", "gauge"},
	"sglang:num_streaming_sessions":                         {"streaming_sessions", "sessions", "gauge"},
	"sglang:streaming_session_held_tokens":                  {"streaming_session_held_tokens", "tokens", "gauge"},
	"sglang:kv_cache_memory_usage_gb":                       {"kv_cache_memory_usage", "gigabytes", "gauge"},
	"sglang:weight_memory_usage_gb":                         {"weight_memory_usage", "gigabytes", "gauge"},
	"sglang:graph_memory_usage_gb":                          {"graph_memory_usage", "gigabytes", "gauge"},
	"sglang:max_total_num_tokens_swa":                       {"max_swa_tokens", "tokens", "gauge"},
	"sglang:prefetched_tokens_total":                        {"storage_prefetched_tokens", "tokens", "counter"},
	"sglang:backuped_tokens_total":                          {"storage_backuped_tokens", "tokens", "counter"},
	"sglang:storage_prefetch_hit_tokens_total":              {"storage_prefetch_hit_tokens", "tokens", "counter"},
	"sglang:storage_prefetch_unfulfilled_tokens_total":      {"storage_prefetch_unfulfilled_tokens", "tokens", "counter"},
	"sglang:storage_prefetch_deferred_tokens_total":         {"storage_prefetch_deferred_tokens", "tokens", "counter"},
	"sglang:hicache_backup_dropped_tokens_total":            {"hicache_backup_dropped_tokens", "tokens", "counter"},
	"sglang:hicache_prefetch_aux_alloc_failed_tokens_total": {"hicache_prefetch_aux_alloc_failed_tokens", "tokens", "counter"},
	"sglang:evicted_tokens_total":                           {"device_kv_evicted_tokens", "tokens", "counter"},
	"sglang:load_back_tokens_total":                         {"host_to_device_loaded_tokens", "tokens", "counter"},
	"sglang:hicache_backup_bytes_total":                     {"device_to_host_backup_bytes", "bytes", "counter"},
	"sglang:load_back_bytes_total":                          {"host_to_device_load_bytes", "bytes", "counter"},
	"sglang:hicache_backup_tokens_total":                    {"device_to_host_backup_tokens", "tokens", "counter"},
	"sglang:hicache_dropped_tokens_total":                   {"device_kv_dropped_tokens", "tokens", "counter"},
	"sglang:eviction_duration_seconds":                      {"kv_eviction_duration", "seconds", "histogram"},
	"sglang:load_back_duration_seconds":                     {"host_to_device_load_duration", "seconds", "histogram"},
	"sglang:hicache_backup_duration_seconds":                {"device_to_host_backup_duration", "seconds", "histogram"},
}

var sglangRequiredLabels = []string{
	"model_name", "engine_type", "tp_rank", "pp_rank", "moe_ep_rank",
}

func validateSGLangWorkloadLabels(metric string, labels map[string]string) error {
	required := append([]string(nil), sglangRequiredLabels...)
	switch metric {
	case "sglang:num_requests_total", "sglang:prompt_tokens_total", "sglang:generation_tokens_total", "sglang:time_to_first_token_seconds", "sglang:e2e_request_latency_seconds":
		required = append(required, "is_streaming")
	case "sglang:realtime_tokens_total", "sglang:prefill_effective_tokens_total":
		required = append(required, "mode")
	case "sglang:cached_tokens_total":
		required = append(required, "cache_source")
	case "sglang:per_stage_req_latency_seconds":
		required = append(required, "stage")
	case "sglang:graph_memory_usage_gb":
		required = append(required, "phase")
	case "sglang:storage_prefetch_unfulfilled_tokens_total", "sglang:storage_prefetch_deferred_tokens_total":
		required = append(required, "reason")
	case "sglang:load_back_tokens_total", "sglang:hicache_backup_tokens_total":
		required = append(required, "pool")
	case "sglang:hicache_dropped_tokens_total":
		required = append(required, "reason", "pool")
	}
	for _, name := range required {
		if !hasWorkloadLabel(labels, name) {
			return errors.New("missing SGLang source identity label")
		}
	}
	if metric == "sglang:num_requests_total" || metric == "sglang:prompt_tokens_total" || metric == "sglang:generation_tokens_total" || metric == "sglang:time_to_first_token_seconds" || metric == "sglang:e2e_request_latency_seconds" {
		if labels["is_streaming"] != "true" && labels["is_streaming"] != "false" {
			return errors.New("invalid SGLang is_streaming label")
		}
	}
	if metric == "sglang:realtime_tokens_total" && !oneOf(labels["mode"], "prefill_compute", "prefill_cache", "decode") {
		return errors.New("invalid SGLang mode label")
	}
	if metric == "sglang:prefill_effective_tokens_total" && !oneOf(labels["mode"], "input", "device_hit", "host_hit", "storage_hit") {
		return errors.New("invalid SGLang mode label")
	}
	return nil
}

func oneOf(value string, allowed ...string) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}
