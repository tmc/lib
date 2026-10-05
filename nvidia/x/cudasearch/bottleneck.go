package cudasearch

import (
	"math"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// PrimaryBottleneck describes the principal resource identified by the classifier.
type PrimaryBottleneck string

// SecondaryCharacteristics are additional observations in the declared profile.
type SecondaryCharacteristics []string

// RelativePattern records qualitative observations without inventing counters.
type RelativePattern map[string]string

// BottleneckState records a classification and its evidence disposition.
// An empty StateName is unknown. ClassifyBottleneck always sets PendingHardware:
// caller-supplied metrics alone establish neither native provenance nor timing.
type BottleneckState struct {
	StateName                cudatrial.BottleneckFamily `json:"state_name"`
	PrimaryBottleneck        PrimaryBottleneck          `json:"primary_bottleneck"`
	SecondaryCharacteristics SecondaryCharacteristics   `json:"secondary_characteristics"`
	RelativePatterns         RelativePattern            `json:"relative_patterns"`
	PendingHardware          bool                       `json:"pending_hardware"`
	Reason                   string                     `json:"reason,omitempty"`
}

// ProfileMetrics uses percentages in [0,100]. Nil counters are unobserved,
// including cache hit rates; zero is a known measurement. These values are
// producer declarations, not authenticated hardware evidence.
type ProfileMetrics struct {
	MemoryThroughput, ComputeThroughput, WarpOccupancy *float64
	L1HitRate, L2HitRate, APIOverhead                  *float64
	BankConflicts                                      *uint64
}

// ClassifyBottleneck applies a fixed, deterministic exploration policy. It never
// promotes caller-supplied numbers to hardware-qualified evidence. Missing or
// invalid throughput/occupancy leaves the category unknown. Precedence is API,
// conflicts, hybrid, bandwidth, compute, cache, occupancy, then memory latency.
func ClassifyBottleneck(m ProfileMetrics) BottleneckState {
	s := BottleneckState{PendingHardware: true, Reason: "profile provenance not hardware-qualified"}
	for _, p := range []*float64{m.MemoryThroughput, m.ComputeThroughput, m.WarpOccupancy, m.L1HitRate, m.L2HitRate, m.APIOverhead} {
		if p != nil && (math.IsNaN(*p) || math.IsInf(*p, 0) || *p < 0 || *p > 100) {
			s.Reason = "invalid percentage"
			return s
		}
	}
	if m.MemoryThroughput == nil || m.ComputeThroughput == nil || m.WarpOccupancy == nil {
		s.Reason = "missing throughput or occupancy"
		return s
	}
	memory, compute, occupancy := *m.MemoryThroughput, *m.ComputeThroughput, *m.WarpOccupancy
	s.RelativePatterns = RelativePattern{}
	switch {
	case m.APIOverhead != nil && *m.APIOverhead >= 50:
		s.StateName, s.PrimaryBottleneck = cudatrial.APIOverheadDominant, "api_bound"
	case m.BankConflicts != nil && *m.BankConflicts > 0:
		s.StateName, s.PrimaryBottleneck = cudatrial.MemoryBankConflicts, "memory_bound"
	case memory >= 70 && compute >= 70:
		s.StateName, s.PrimaryBottleneck = cudatrial.HybridBound, "hybrid_bound"
	case memory >= 70:
		s.StateName, s.PrimaryBottleneck = cudatrial.MemoryBandwidthLimited, "memory_bound"
	case compute >= 70:
		s.StateName, s.PrimaryBottleneck = cudatrial.ComputeThroughputLimited, "compute_bound"
	case m.L1HitRate != nil && m.L2HitRate != nil && *m.L1HitRate < 30 && *m.L2HitRate < 30:
		s.StateName, s.PrimaryBottleneck = cudatrial.CacheInefficient, "memory_bound"
	case occupancy < 35:
		s.StateName, s.PrimaryBottleneck = cudatrial.LatencyOccupancyLimited, "latency_bound"
	default:
		s.StateName, s.PrimaryBottleneck = cudatrial.MemoryLatencyBound, "latency_bound"
	}
	if occupancy < 35 {
		s.SecondaryCharacteristics = append(s.SecondaryCharacteristics, "low_occupancy")
		s.RelativePatterns["occupancy"] = "low"
	}
	if memory >= 70 {
		s.RelativePatterns["memory_pressure"] = "high"
	}
	if compute >= 70 {
		s.RelativePatterns["compute_pressure"] = "high"
	}
	return s
}
