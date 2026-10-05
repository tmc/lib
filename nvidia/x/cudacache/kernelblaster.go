package cudacache

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// KernelBlasterGroups maps the upstream sixteen strategy groups into the eight
// exploration families. The upstream known_states contains only three states;
// this mapping is a local search policy, not an assertion of schema equivalence.
// The returned map and slices are detached.
func KernelBlasterGroups() map[string][]cudatrial.BottleneckFamily {
	return map[string][]cudatrial.BottleneckFamily{
		"high_level_inefficiency":         {cudatrial.ComputeThroughputLimited, cudatrial.HybridBound},
		"memory_bandwidth_saturated":      {cudatrial.MemoryBandwidthLimited},
		"memory_latency_bound":            {cudatrial.MemoryLatencyBound},
		"memory_bank_conflicts":           {cudatrial.MemoryBankConflicts},
		"cache_inefficient":               {cudatrial.CacheInefficient},
		"compute_throughput_saturated":    {cudatrial.ComputeThroughputLimited},
		"instruction_mix_suboptimal":      {cudatrial.ComputeThroughputLimited},
		"thread_divergence_high":          {cudatrial.ComputeThroughputLimited, cudatrial.LatencyOccupancyLimited},
		"low_occupancy_register_pressure": {cudatrial.LatencyOccupancyLimited},
		"low_occupancy_shared_memory":     {cudatrial.LatencyOccupancyLimited},
		"insufficient_parallelism":        {cudatrial.LatencyOccupancyLimited},
		"hybrid_bound":                    {cudatrial.HybridBound},
		"memory_compute_balanced":         {cudatrial.HybridBound},
		"latency_memory_bound":            {cudatrial.MemoryLatencyBound, cudatrial.LatencyOccupancyLimited},
		"api_overhead_dominant":           {cudatrial.APIOverheadDominant},
		"transfer_bandwidth_limited":      {cudatrial.MemoryBandwidthLimited},
	}
}

// LoadKernelBlaster imports a bounded upstream optimization_database.json for
// one explicit target. Predicted speedups stay predictions; actual_improvement
// and actual_speedup are not admitted as hardware history without trajectories.
// All groups must be recognized. On failure the catalog is unchanged.
func (k *StrategyKnowledgeBase) LoadKernelBlaster(r io.Reader, target string) error {
	if k == nil || r == nil || cudakernel.ValidateTargetArch(target) != nil {
		return fmt.Errorf("cudacache: invalid KernelBlaster owner, reader or target")
	}
	b, err := io.ReadAll(io.LimitReader(r, strategyBytes+1))
	if err != nil {
		return err
	}
	if len(b) > strategyBytes {
		return fmt.Errorf("cudacache: KernelBlaster byte limit")
	}
	var db struct {
		Known map[string]struct {
			StateName string            `json:"state_name"`
			Primary   string            `json:"primary_bottleneck"`
			Secondary []string          `json:"secondary_characteristics"`
			Signature string            `json:"performance_signature"`
			Context   string            `json:"context_description"`
			Patterns  map[string]string `json:"relative_patterns"`
		} `json:"known_states"`
		Groups map[string]struct {
			Primary       string `json:"primary_bottleneck"`
			Secondary     string `json:"secondary_characteristics"`
			Optimizations []struct {
				Technique            string          `json:"technique"`
				Description          string          `json:"description"`
				Category             string          `json:"category"`
				Confidence           float64         `json:"confidence_score"`
				PredictedSpeedup     float64         `json:"predicted_speedup"`
				PredictedImprovement json.RawMessage `json:"predicted_improvement"`
				ActualImprovement    json.RawMessage `json:"actual_improvement"`
				ActualSpeedup        json.RawMessage `json:"actual_speedup"`
				InitialCycles        json.RawMessage `json:"initial_elapsed_cycles"`
				LastUpdated          json.RawMessage `json:"last_updated"`
				UsageCount           uint64          `json:"usage_count"`
			} `json:"optimizations"`
		} `json:"optimization_strategies"`
		Composite  map[string]json.RawMessage `json:"composite_optimizations"`
		Discovered map[string]json.RawMessage `json:"discovered_states"`
	}
	if err := memoryJSON(b); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&db); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return fmt.Errorf("cudacache: trailing KernelBlaster JSON")
	}
	if len(db.Known) != 3 || len(db.Composite) != 0 || len(db.Discovered) != 0 {
		return fmt.Errorf("cudacache: unsupported KernelBlaster states")
	}
	for name, primary := range map[string]string{"memory_bandwidth_limited": "memory_bound", "compute_throughput_limited": "compute_bound", "latency_occupancy_limited": "latency_bound"} {
		state, ok := db.Known[name]
		if !ok || state.StateName != name || state.Primary != primary || len(state.Secondary) == 0 || len(state.Patterns) == 0 || state.Signature == "" || state.Context == "" {
			return fmt.Errorf("cudacache: invalid KernelBlaster known state %q", name)
		}
	}
	mapping := KernelBlasterGroups()
	if len(db.Groups) != len(mapping) {
		return fmt.Errorf("cudacache: KernelBlaster group inventory differs")
	}
	var next StrategyKnowledgeBase
	seen := make(map[string]bool)
	for group, g := range db.Groups {
		families, ok := mapping[group]
		if !ok {
			return fmt.Errorf("cudacache: unsupported strategy group %q", group)
		}
		for _, opt := range g.Optimizations {
			key := group + "\x00" + opt.Technique
			if seen[key] {
				return fmt.Errorf("cudacache: duplicate KernelBlaster technique")
			}
			seen[key] = true
			for _, f := range families {
				if err := next.Update(Strategy{Family: f, TargetArch: target, Technique: opt.Technique, SourceGroup: group, Confidence: opt.Confidence, PredictedSpeedup: opt.PredictedSpeedup}); err != nil {
					return err
				}
			}
		}
	}
	k.mu.Lock()
	k.rows = next.rows
	k.mu.Unlock()
	return nil
}
