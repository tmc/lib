package cudaprof

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"maps"
	"math"
	"sort"
	"time"
)

// ProfileSnapshotVersion is the current schema version for ProfileSnapshot.
const ProfileSnapshotVersion = 1

// SamplingProvenance retains native PC-sampling attribution and quality totals.
// RunID and ModuleGeneration are assigned by the owner of the live CUDA context.
// They qualify native addresses/IDs; ModuleCRC alone does not identify a lifetime.
// TotalSamples and NonUserSamples are native values, not sums of reason counts.
type SamplingProvenance struct {
	RunID            string `json:"run_id"`
	ModuleGeneration uint64 `json:"module_generation"`
	ModuleCRC        uint64 `json:"module_crc"`
	KernelName       string `json:"kernel_name"`
	FunctionIndex    uint32 `json:"function_index"`
	CorrelationID    uint32 `json:"correlation_id"`
	RangeID          uint64 `json:"range_id"`
	CUPTIVersion     uint32 `json:"cupti_version"`
	CollectionMode   uint32 `json:"collection_mode"`
	TotalSamples     uint64 `json:"total_samples"`
	NonUserSamples   uint64 `json:"non_user_samples"`
}

var (
	ErrWorkloadMismatch       = errors.New("cudaprof: profile workload mismatch")
	ErrProfileTargetMismatch  = errors.New("cudaprof: profile target mismatch")
	ErrDeviceMismatch         = errors.New("cudaprof: profile device mismatch")
	ErrGuidanceUnavailable    = errors.New("cudaprof: profile guidance unavailable")
	ErrInvalidSnapshotVersion = errors.New("cudaprof: invalid profile snapshot version")
)

// ProfileSnapshot holds versioned GPU profiling observations keyed to code
// digest, region, workload, device, and collector mode. Do not mutate it while
// another goroutine is reading it. Guidance validates and copies its input.
type ProfileSnapshot struct {
	Version            int                 `json:"version"`
	Timestamp          time.Time           `json:"timestamp"`
	CodeDigest         string              `json:"code_digest"`
	Region             string              `json:"region"`
	Workload           string              `json:"workload"`
	TokensHash         string              `json:"tokens_hash,omitempty"`
	Device             string              `json:"device"`
	CollectorMode      string              `json:"collector_mode"`
	Coverage           float64             `json:"coverage"`
	ClockUncertaintyNs int64               `json:"clock_uncertainty_ns"`
	Losses             uint64              `json:"losses"`
	Sampling           *SamplingProvenance `json:"sampling,omitempty"`
	Metrics            map[string]int64    `json:"metrics"`
	// MetricUnits records producer-declared units. Missing units remain unknown.
	MetricUnits map[string]string           `json:"metric_units,omitempty"`
	PCStalls    map[uint64]map[string]int64 `json:"pc_stalls,omitempty"`
}

// NewProfileSnapshot validates a snapshot and copies its metric maps.
func NewProfileSnapshot(snap ProfileSnapshot) (*ProfileSnapshot, error) {
	if snap.Version == 0 {
		snap.Version = ProfileSnapshotVersion
	}
	if snap.Version != ProfileSnapshotVersion {
		return nil, fmt.Errorf("%w: version %d (supported %d)", ErrInvalidSnapshotVersion, snap.Version, ProfileSnapshotVersion)
	}
	if snap.CodeDigest == "" {
		return nil, errors.New("cudaprof: empty code digest")
	}
	if len(snap.CodeDigest) != 2*sha256.Size {
		return nil, fmt.Errorf("cudaprof: invalid sha256 code digest length %d", len(snap.CodeDigest))
	}
	if _, err := hex.DecodeString(snap.CodeDigest); err != nil {
		return nil, fmt.Errorf("cudaprof: malformed hex code digest: %w", err)
	}
	if snap.Region == "" {
		return nil, errors.New("cudaprof: empty region name")
	}
	if snap.Workload == "" {
		return nil, errors.New("cudaprof: empty workload identifier")
	}
	if snap.Device == "" {
		return nil, errors.New("cudaprof: empty device identifier")
	}
	if snap.CollectorMode == "" {
		return nil, errors.New("cudaprof: empty collector mode")
	}
	if math.IsNaN(snap.Coverage) || math.IsInf(snap.Coverage, 0) || snap.Coverage < 0.0 || snap.Coverage > 1.0 {
		return nil, fmt.Errorf("cudaprof: invalid coverage %f (must be between 0.0 and 1.0)", snap.Coverage)
	}
	if snap.ClockUncertaintyNs < -1 {
		return nil, fmt.Errorf("cudaprof: invalid clock uncertainty %d", snap.ClockUncertaintyNs)
	}
	var sampling *SamplingProvenance
	if p := snap.Sampling; p != nil {
		if p.RunID == "" || p.ModuleGeneration == 0 || p.KernelName == "" || p.CUPTIVersion == 0 || (p.CollectionMode != 1 && p.CollectionMode != 2) {
			return nil, errors.New("cudaprof: incomplete sampling provenance")
		}
		if p.NonUserSamples > p.TotalSamples || snap.Losses > p.TotalSamples {
			return nil, errors.New("cudaprof: sampling quality count exceeds native total")
		}
		copy := *p
		sampling = &copy
	}

	metricsCopy := make(map[string]int64, len(snap.Metrics))
	for metric, unit := range snap.MetricUnits {
		if metric == "" || unit == "" {
			return nil, errors.New("cudaprof: empty metric name or unit")
		}
	}
	for k, v := range snap.Metrics {
		if k == "" || v < 0 {
			return nil, fmt.Errorf("cudaprof: invalid metric %q count %d", k, v)
		}
		metricsCopy[k] = v
	}

	var pcStallsCopy map[uint64]map[string]int64
	if snap.PCStalls != nil {
		pcStallsCopy = make(map[uint64]map[string]int64, len(snap.PCStalls))
		for pc, m := range snap.PCStalls {
			for metric, count := range m {
				if metric == "" || count < 0 {
					return nil, fmt.Errorf("cudaprof: invalid pc %#x metric %q count %d", pc, metric, count)
				}
			}
			pcStallsCopy[pc] = maps.Clone(m)
		}
	}

	return &ProfileSnapshot{
		Version:            snap.Version,
		Timestamp:          snap.Timestamp,
		CodeDigest:         snap.CodeDigest,
		Region:             snap.Region,
		Workload:           snap.Workload,
		TokensHash:         snap.TokensHash,
		Device:             snap.Device,
		CollectorMode:      snap.CollectorMode,
		Coverage:           snap.Coverage,
		ClockUncertaintyNs: snap.ClockUncertaintyNs,
		Losses:             snap.Losses,
		Sampling:           sampling,
		Metrics:            metricsCopy,
		MetricUnits:        maps.Clone(snap.MetricUnits),
		PCStalls:           pcStallsCopy,
	}, nil
}

// SearchGuidance provides attributed profile priors and bottleneck facts to compiler search lanes
// without synthesizing artificial latency values or altering rewrite legality.
type SearchGuidance struct {
	Available         bool              `json:"available"`
	UnavailableReason string            `json:"unavailable_reason,omitempty"`
	Workload          string            `json:"workload"`
	Device            string            `json:"device"`
	StallCounts       map[string]int64  `json:"stall_counts,omitempty"` // Original metric units (warps/cycles)
	MetricUnits       map[string]string `json:"metric_units,omitempty"`
	// HotPCs ranks at most ten PCs per metric. Counts from different metrics
	// are never summed or compared; they may overlap or use different units.
	HotPCs        map[string][]uint64         `json:"hot_pcs,omitempty"`
	PCStalls      map[uint64]map[string]int64 `json:"pc_stalls,omitempty"`
	CodeDigest    string                      `json:"code_digest"`
	Region        string                      `json:"region"`
	CollectorMode string                      `json:"collector_mode"`
	TokensHash    string                      `json:"tokens_hash,omitempty"`
	Timestamp     time.Time                   `json:"timestamp"`
	Losses        uint64                      `json:"losses"`
	Sampling      *SamplingProvenance         `json:"sampling,omitempty"`
	Coverage      float64                     `json:"coverage"`
	UncertaintyNs int64                       `json:"uncertainty_ns"`
}

// ProfileTarget identifies the code and collection context a consumer expects.
// All fields except TokensHash are required. An empty TokensHash matches only
// a snapshot with no token identity, never one with an unknown or different hash.
type ProfileTarget struct {
	CodeDigest, Region, Workload, TokensHash, Device, CollectorMode string
}

// Guidance extracts observations only when every target identity matches.
// If the snapshot has zero coverage or missing metrics, ErrGuidanceUnavailable is returned.
func (s *ProfileSnapshot) Guidance(target ProfileTarget) (*SearchGuidance, error) {
	if s == nil {
		return nil, errors.New("cudaprof: nil profile snapshot")
	}
	if s.Version != ProfileSnapshotVersion {
		return nil, fmt.Errorf("%w: version %d", ErrInvalidSnapshotVersion, s.Version)
	}
	validated, err := NewProfileSnapshot(*s)
	if err != nil {
		return nil, err
	}
	s = validated
	if target.CodeDigest != s.CodeDigest || target.Region != s.Region || target.CollectorMode != s.CollectorMode {
		return nil, fmt.Errorf("%w: code digest, region or collector mode differs", ErrProfileTargetMismatch)
	}
	targetWorkload, targetTokensHash, targetDevice := target.Workload, target.TokensHash, target.Device
	if targetWorkload == "" {
		return nil, errors.New("cudaprof: empty target workload")
	}
	if targetDevice == "" {
		return nil, errors.New("cudaprof: empty target device")
	}

	if targetWorkload != s.Workload {
		return nil, fmt.Errorf("%w: target workload %q does not match snapshot %q",
			ErrWorkloadMismatch, targetWorkload, s.Workload)
	}
	if targetTokensHash != s.TokensHash {
		return nil, fmt.Errorf("%w: target tokens hash %q does not match snapshot %q",
			ErrWorkloadMismatch, targetTokensHash, s.TokensHash)
	}
	if targetDevice != s.Device {
		return nil, fmt.Errorf("%w: target device %q does not match snapshot %q",
			ErrDeviceMismatch, targetDevice, s.Device)
	}

	guidance := &SearchGuidance{Workload: s.Workload, Device: s.Device, CodeDigest: s.CodeDigest, Region: s.Region, CollectorMode: s.CollectorMode, TokensHash: s.TokensHash, Timestamp: s.Timestamp, Losses: s.Losses, Coverage: s.Coverage, UncertaintyNs: s.ClockUncertaintyNs}
	guidance.Sampling = s.Sampling
	if s.Coverage <= 0.0 || len(s.Metrics) == 0 {
		guidance.UnavailableReason = "insufficient coverage or missing telemetry metrics"
		return guidance, ErrGuidanceUnavailable
	}
	guidance.MetricUnits = s.MetricUnits
	for metric := range s.Metrics {
		if s.MetricUnits[metric] == "" {
			guidance.UnavailableReason = "missing unit for metric " + metric
			return guidance, ErrGuidanceUnavailable
		}
	}
	for _, counters := range s.PCStalls {
		for metric := range counters {
			if s.MetricUnits[metric] == "" {
				guidance.UnavailableReason = "missing unit for pc metric " + metric
				return guidance, ErrGuidanceUnavailable
			}
		}
	}
	guidance.Available = true
	guidance.StallCounts = s.Metrics
	guidance.PCStalls = s.PCStalls
	guidance.HotPCs = make(map[string][]uint64)
	// Rank each metric independently; never infer exclusivity between samples.
	type pcHit struct {
		pc    uint64
		count int64
	}
	byMetric := make(map[string][]pcHit)
	for pc, m := range s.PCStalls {
		for metric, count := range m {
			if count > 0 {
				byMetric[metric] = append(byMetric[metric], pcHit{pc: pc, count: count})
			}
		}
	}
	for metric, hits := range byMetric {
		sort.Slice(hits, func(i, j int) bool {
			if hits[i].count == hits[j].count {
				return hits[i].pc < hits[j].pc
			}
			return hits[i].count > hits[j].count
		})
		for i := 0; i < len(hits) && i < 10; i++ {
			guidance.HotPCs[metric] = append(guidance.HotPCs[metric], hits[i].pc)
		}
	}
	return guidance, nil
}
