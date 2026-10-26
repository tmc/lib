package nvtelemetry

import "errors"

// The selected V1 families follow vLLM v0.8.5, commit
// ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d, vllm/v1/metrics/loggers.py.
// Other V1 families remain raw and unmapped.
var vllmV1WorkloadMappings = map[string]workloadMapping{
	"vllm:prompt_tokens_total":            {"processed_prefill_tokens", "tokens", "counter"},
	"vllm:generation_tokens_total":        {"processed_generation_tokens", "tokens", "counter"},
	"vllm:request_success_total":          {"processed_requests_by_finish_reason", "requests", "counter"},
	"vllm:num_requests_running":           {"running_requests", "requests", "gauge"},
	"vllm:num_requests_waiting":           {"waiting_requests", "requests", "gauge"},
	"vllm:gpu_cache_usage_perc":           {"gpu_kv_cache_fraction", "1", "gauge"},
	"vllm:time_to_first_token_seconds":    {"aggregate_time_to_first_token", "s", "histogram"},
	"vllm:time_per_output_token_seconds":  {"aggregate_time_per_output_token", "s", "histogram"},
	"vllm:e2e_request_latency_seconds":    {"aggregate_request_latency", "s", "histogram"},
	"vllm:request_prompt_tokens":          {"aggregate_request_prefill_tokens", "tokens", "histogram"},
	"vllm:request_generation_tokens":      {"aggregate_request_generation_tokens", "tokens", "histogram"},
	"vllm:request_queue_time_seconds":     {"aggregate_request_queue_time", "s", "histogram"},
	"vllm:request_inference_time_seconds": {"aggregate_request_inference_time", "s", "histogram"},
	"vllm:request_prefill_time_seconds":   {"aggregate_request_prefill_time", "s", "histogram"},
	"vllm:request_decode_time_seconds":    {"aggregate_request_decode_time", "s", "histogram"},
}

func validateWorkloadEngine(s WorkloadSource) error {
	if s.Engine != "" && (s.Provider != "vllm" || s.Version != "0.8.5" || s.Engine != "v1") {
		return errors.New("unsupported workload engine contract")
	}
	return nil
}

func vllmMappings(s WorkloadSource) (map[string]workloadMapping, error) {
	if err := validateWorkloadEngine(s); err != nil {
		return nil, err
	}
	if s.Provider != "vllm" || s.Version != "0.8.5" {
		return nil, errors.New("unsupported vLLM provider or version")
	}
	if s.Engine == "v1" {
		return vllmV1WorkloadMappings, nil
	}
	return vllmWorkloadMappings, nil
}

func validateVLLMLabels(s WorkloadSource, base, component string, labels map[string]string) error {
	if labels["model_name"] == "" {
		return errors.New("missing vLLM model_name label")
	}
	if s.Engine == "" {
		if hasWorkloadLabel(labels, "engine") {
			return errors.New("vLLM engine label requires explicit engine contract")
		}
		return nil
	}
	engine := labels["engine"]
	if engine == "" || len(engine) > 1 && engine[0] == '0' {
		return errors.New("invalid vLLM engine index")
	}
	for _, c := range engine {
		if c < '0' || c > '9' {
			return errors.New("invalid vLLM engine index")
		}
	}
	if base == "vllm:request_success_total" {
		switch labels["finished_reason"] {
		case "stop", "length", "abort":
		default:
			return errors.New("invalid vLLM finish reason")
		}
	}
	for k := range labels {
		if k != "model_name" && k != "engine" && !(k == "finished_reason" && base == "vllm:request_success_total") && !(k == "le" && component == "bucket") {
			return errors.New("unsupported vLLM V1 label")
		}
	}
	return nil
}
