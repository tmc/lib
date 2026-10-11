package nvtelemetry

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math/big"
	"sort"
	"strings"
)

type workloadMapping struct{ semantic, unit, kind string }

var tritonWorkloadMappings = map[string]workloadMapping{
	"nv_cache_num_hits_per_model":              {"response_cache_hits", "lookups", "counter"},
	"nv_cache_num_misses_per_model":            {"response_cache_misses", "lookups", "counter"},
	"nv_inference_first_response_histogram_ms": {"aggregate_first_response_latency", "ms", "histogram"},
	"nv_inference_request_success":             {"successful_requests", "requests", "counter"},
	"nv_inference_request_failure":             {"failed_requests", "requests", "counter"},
	"nv_inference_count":                       {"uncached_inferences", "inferences", "counter"},
	"nv_inference_exec_count":                  {"uncached_batch_executions", "executions", "counter"},
	"nv_inference_request_duration_us":         {"cumulative_request_time_including_cache", "us", "counter"},
	"nv_inference_queue_duration_us":           {"cumulative_queue_time_including_cache", "us", "counter"},
	"nv_inference_compute_input_duration_us":   {"cumulative_uncached_input_time", "us", "counter"},
	"nv_inference_compute_infer_duration_us":   {"cumulative_uncached_compute_time", "us", "counter"},
	"nv_inference_compute_output_duration_us":  {"cumulative_uncached_output_time", "us", "counter"},
	"nv_inference_pending_request_count":       {"pending_requests", "requests", "gauge"},
}

var vllmWorkloadMappings = map[string]workloadMapping{
	"vllm:prompt_tokens_total":           {"processed_prefill_tokens", "tokens", "counter"},
	"vllm:generation_tokens_total":       {"processed_generation_tokens", "tokens", "counter"},
	"vllm:request_success_total":         {"processed_requests_by_finish_reason", "requests", "counter"},
	"vllm:num_requests_running":          {"running_requests", "requests", "gauge"},
	"vllm:num_requests_waiting":          {"waiting_requests", "requests", "gauge"},
	"vllm:num_requests_swapped":          {"swapped_requests", "requests", "gauge"},
	"vllm:gpu_cache_usage_perc":          {"gpu_kv_cache_fraction", "1", "gauge"},
	"vllm:cpu_cache_usage_perc":          {"cpu_kv_cache_fraction", "1", "gauge"},
	"vllm:time_to_first_token_seconds":   {"aggregate_time_to_first_token", "s", "histogram"},
	"vllm:time_per_output_token_seconds": {"aggregate_time_per_output_token", "s", "histogram"},
	"vllm:e2e_request_latency_seconds":   {"aggregate_request_latency", "s", "histogram"},
	"vllm:request_prompt_tokens":         {"aggregate_request_prefill_tokens", "tokens", "histogram"},
	"vllm:request_generation_tokens":     {"aggregate_request_generation_tokens", "tokens", "histogram"},
}

var rayWorkloadMappings = map[string]workloadMapping{
	"ray_resources":             {"logical_resource_assignment", "resources", "gauge"},
	"ray_tasks":                 {"tasks_by_state", "tasks", "gauge"},
	"ray_actors":                {"actors_by_state", "actors", "gauge"},
	"ray_node_gpus_available":   {"node_gpu_count", "devices", "gauge"},
	"ray_node_gpus_utilization": {"node_gpu_utilization", "percent", "gauge"},
	"ray_node_gram_used":        {"node_gpu_memory_used", "bytes", "gauge"},
	"ray_node_gram_available":   {"node_gpu_memory_available", "bytes", "gauge"},
}

// ReadServingExposition reads pinned Triton 25.06, vLLM 0.8.5 or Ray 2.44.0
// Prometheus text 0.0.4 snapshots through the shared bounded exposition parser.
// Optional native timestamps are retained. A timestamp absent from the export
// stays absent; use StampWorkloadSnapshot for a separately recorded scrape time.
// Unmapped fields remain in Raw, listed in UnmappedNames.
func ReadServingExposition(ctx context.Context, r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, error) {
	if ctx == nil {
		return WorkloadExport{}, errors.New("nil workload context")
	}
	if err := ctx.Err(); err != nil {
		return WorkloadExport{}, err
	}
	var mappings map[string]workloadMapping
	switch {
	case s.Provider == "triton" && s.Version == "25.06":
		mappings = tritonWorkloadMappings
	case s.Provider == "vllm" && s.Version == "0.8.5":
		mappings = vllmWorkloadMappings
	case s.Provider == "ray" && s.Version == "2.44.0":
		mappings = rayWorkloadMappings
	default:
		return WorkloadExport{}, errors.New("unsupported serving metrics provider or version")
	}
	if r == nil {
		return WorkloadExport{}, errors.New("nil workload reader")
	}
	e, l, err := readWorkload(workloadContextReader{ctx, r}, s, l)
	if err != nil {
		return e, err
	}
	points, err := ReadExposition(ctx, bytes.NewReader(e.Raw), ExpositionOptions{Format: PrometheusText004, MaxBytes: l.MaxBytes, MaxSamples: l.MaxMetrics, MaxLabelBytes: l.MaxStringBytes, MaxLineBytes: int(min(l.MaxBytes, 1<<20))})
	if err != nil {
		return e, err
	}
	unknown := map[string]bool{}
	for _, v := range points {
		base := v.Name
		component := ""
		m, ok := mappings[base]
		if !ok {
			for _, suffix := range []string{"_bucket", "_count", "_sum"} {
				if strings.HasSuffix(base, suffix) {
					base = strings.TrimSuffix(base, suffix)
					candidate, found := mappings[base]
					if found && candidate.kind == "histogram" {
						m = candidate
						ok = true
						component = strings.TrimPrefix(suffix, "_")
						break
					}
				}
			}
		}
		if !ok {
			unknown[v.Name] = true
			continue
		}
		if v.Type != m.kind {
			return e, fmt.Errorf("workload metric %s type %q does not match pinned %s", v.Name, v.Type, m.kind)
		}
		if s.Provider == "triton" && (v.Labels["model"] == "" || v.Labels["version"] == "") {
			return e, errors.New("missing Triton model or version label")
		}
		if s.Provider == "vllm" && v.Labels["model_name"] == "" {
			return e, errors.New("missing vLLM model_name label")
		}
		if v.Name == "ray_resources" && (v.Labels["Name"] == "" || v.Labels["State"] == "") {
			return e, errors.New("missing Ray resource identity or state")
		}
		n, err := workloadNumber(v.RawValue)
		if err != nil {
			return e, err
		}
		if n.Sign() < 0 {
			return e, errors.New("negative serving metric")
		}
		if m.kind == "counter" && (m.unit == "requests" || m.unit == "tokens" || m.unit == "inferences" || m.unit == "executions" || m.unit == "lookups") && !n.IsInt() {
			return e, errors.New("fractional workload count")
		}
		p := WorkloadMetric{Source: e.Source, Name: v.Name, Semantic: m.semantic, Unit: m.unit, Kind: m.kind, Temporality: "instantaneous", Labels: v.Labels, RawValue: v.RawValue, TimeNanos: v.TimestampUnix, HasTime: v.HasTimestamp}
		if v.HasTimestamp {
			p.RawTime = v.RawTimestamp
			p.RawTimeUnit = v.TimestampUnit
			p.TimeSource = "producer"
		}
		if m.kind == "counter" || m.kind == "histogram" {
			p.Temporality = "cumulative"
		}
		if m.kind == "histogram" {
			if component == "" {
				return e, errors.New("histogram metric missing component")
			}
			p.Kind = "histogram_" + component
			if component != "sum" {
				p.Unit = "observations"
			}
		}
		if m.semantic == "gpu_kv_cache_fraction" || m.semantic == "cpu_kv_cache_fraction" {
			if n.Cmp(big.NewRat(1, 1)) > 0 {
				return e, errors.New("KV cache fraction out of range")
			}
		}
		e.Metrics = append(e.Metrics, p)
	}
	for name := range unknown {
		e.UnmappedNames = append(e.UnmappedNames, name)
	}
	sort.Strings(e.UnmappedNames)
	if err := validateWorkloadHistograms(e.Metrics); err != nil {
		return e, err
	}
	return e, nil
}

type workloadContextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r workloadContextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

// StampWorkloadSnapshot adds an explicit collector observation timestamp to a
// snapshot whose producer samples have no timestamps. It returns a copy and
// refuses to replace any native timestamp. Raw bytes and hash remain unchanged.
// This establishes a scrape observation, never a sensor or request timestamp.
func StampWorkloadSnapshot(e WorkloadExport, nanos int64, domain, epoch string) (WorkloadExport, error) {
	if domain == "" || epoch == "" {
		return WorkloadExport{}, errors.New("unknown workload observation clock")
	}
	metrics := append([]WorkloadMetric(nil), e.Metrics...)
	for i := range metrics {
		if metrics[i].HasTime {
			return WorkloadExport{}, errors.New("cannot replace native workload timestamp")
		}
		metrics[i].HasTime = true
		metrics[i].TimeSource = "collector_observation"
		metrics[i].TimeNanos = nanos
		metrics[i].Source.ClockDomain = domain
		metrics[i].Source.ClockEpochID = epoch
	}
	e.Metrics = metrics
	e.Source.ClockDomain = domain
	e.Source.ClockEpochID = epoch
	return e, nil
}

func validateWorkloadHistograms(metrics []WorkloadMetric) error {
	type bucket struct {
		bound *big.Rat
		inf   bool
		count *big.Rat
	}
	type histogram struct {
		buckets     []bucket
		count       *big.Rat
		sum         bool
		sumValue    *big.Rat
		hasTime     bool
		time        int64
		initialized bool
	}
	groups := map[string]*histogram{}
	for _, p := range metrics {
		if !strings.HasPrefix(p.Kind, "histogram_") {
			continue
		}
		component := strings.TrimPrefix(p.Kind, "histogram_")
		q := p
		q.Name = strings.TrimSuffix(p.Name, "_"+component)
		q.Labels = map[string]string{}
		for k, v := range p.Labels {
			if k != "le" {
				q.Labels[k] = v
			}
		}
		key := workloadSeriesKey(q)
		h := groups[key]
		if h == nil {
			h = &histogram{}
			groups[key] = h
		}
		if h.initialized && (h.hasTime != p.HasTime || h.time != p.TimeNanos) {
			return errors.New("histogram component timestamps disagree")
		}
		h.initialized = true
		h.hasTime = p.HasTime
		h.time = p.TimeNanos
		v, err := workloadNumber(p.RawValue)
		if err != nil {
			return err
		}
		switch component {
		case "bucket":
			if !v.IsInt() {
				return errors.New("fractional histogram count")
			}
			le, ok := p.Labels["le"]
			if !ok {
				return errors.New("histogram bucket missing bound")
			}
			b := bucket{count: v, inf: le == "+Inf"}
			if !b.inf {
				b.bound, err = workloadNumber(le)
				if err != nil {
					return err
				}
			}
			h.buckets = append(h.buckets, b)
		case "count":
			if h.count != nil || !v.IsInt() {
				return errors.New("duplicate or fractional histogram count")
			}
			h.count = v
		case "sum":
			if h.sum {
				return errors.New("duplicate histogram sum")
			}
			h.sum = true
			h.sumValue = v
		}
	}
	for _, h := range groups {
		if h.count == nil || !h.sum || len(h.buckets) == 0 {
			return errors.New("incomplete histogram components")
		}
		sort.Slice(h.buckets, func(i, j int) bool {
			a, b := h.buckets[i], h.buckets[j]
			if a.inf {
				return false
			}
			if b.inf {
				return true
			}
			return a.bound.Cmp(b.bound) < 0
		})
		for i, b := range h.buckets {
			if i > 0 {
				a := h.buckets[i-1]
				if a.inf || b.inf && a.inf || !b.inf && a.bound.Cmp(b.bound) == 0 || b.count.Cmp(a.count) < 0 {
					return errors.New("incompatible histogram bounds or cumulative counts")
				}
			}
		}
		last := h.buckets[len(h.buckets)-1]
		if !last.inf || last.count.Cmp(h.count) != 0 {
			return errors.New("histogram infinite bucket and count disagree")
		}
		if h.count.Sign() == 0 && h.sumValue.Sign() != 0 {
			return errors.New("histogram sum without observations")
		}
	}
	return nil
}
