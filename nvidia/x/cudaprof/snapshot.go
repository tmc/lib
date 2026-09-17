package cudaprof

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"maps"
	"sort"
	"time"
)

// ProfileSnapshotVersion is the current schema version for ProfileSnapshot.
const ProfileSnapshotVersion = 1

var (
	ErrWorkloadMismatch       = errors.New("cudaprof: profile workload mismatch")
	ErrDeviceMismatch         = errors.New("cudaprof: profile device mismatch")
	ErrGuidanceUnavailable    = errors.New("cudaprof: profile guidance unavailable")
	ErrInvalidSnapshotVersion = errors.New("cudaprof: invalid profile snapshot version")
)

// ProfileSnapshot is an immutable, versioned snapshot of GPU profiling observations
// keyed to code digest, region, workload, device, and collector mode.
type ProfileSnapshot struct {
	Version            int                         `json:"version"`
	Timestamp          time.Time                   `json:"timestamp"`
	CodeDigest         string                      `json:"code_digest"`
	Region             string                      `json:"region"`
	Workload           string                      `json:"workload"`
	TokensHash         string                      `json:"tokens_hash,omitempty"`
	Device             string                      `json:"device"`
	CollectorMode      string                      `json:"collector_mode"`
	Coverage           float64                     `json:"coverage"`
	ClockUncertaintyNs int64                       `json:"clock_uncertainty_ns"`
	Losses             uint64                      `json:"losses"`
	Metrics            map[string]int64            `json:"metrics"`
	PCStalls           map[uint64]map[string]int64 `json:"pc_stalls,omitempty"`
}

// NewProfileSnapshot constructs and validates an immutable ProfileSnapshot.
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
	if snap.Coverage < 0.0 || snap.Coverage > 1.0 {
		return nil, fmt.Errorf("cudaprof: invalid coverage %f (must be between 0.0 and 1.0)", snap.Coverage)
	}
	if snap.ClockUncertaintyNs < -1 {
		return nil, fmt.Errorf("cudaprof: invalid clock uncertainty %d", snap.ClockUncertaintyNs)
	}

	metricsCopy := make(map[string]int64, len(snap.Metrics))
	for k, v := range snap.Metrics {
		metricsCopy[k] = v
	}

	var pcStallsCopy map[uint64]map[string]int64
	if snap.PCStalls != nil {
		pcStallsCopy = make(map[uint64]map[string]int64, len(snap.PCStalls))
		for pc, m := range snap.PCStalls {
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
		Metrics:            metricsCopy,
		PCStalls:           pcStallsCopy,
	}, nil
}

// SearchGuidance provides attributed profile priors and bottleneck facts to compiler search lanes
// without synthesizing artificial latency values or altering rewrite legality.
type SearchGuidance struct {
	Available          bool             `json:"available"`
	UnavailableReason  string           `json:"unavailable_reason,omitempty"`
	Workload           string           `json:"workload"`
	Device             string           `json:"device"`
	PrimaryStallMetric string           `json:"primary_stall_metric,omitempty"`
	StallCounts        map[string]int64 `json:"stall_counts,omitempty"` // Original metric units (warps/cycles)
	HotPCs             []uint64         `json:"hot_pcs,omitempty"`
	MemoryBound        bool             `json:"memory_bound"`
	ScoreboardBound    bool             `json:"scoreboard_bound"`
	Coverage           float64          `json:"coverage"`
	UncertaintyNs      int64            `json:"uncertainty_ns"`
}

// Guidance extracts search guidance for the specified target workload and device.
// If the snapshot workload or device does not match the target, an error is returned.
// If the snapshot has zero coverage or missing metrics, ErrGuidanceUnavailable is returned.
func (s *ProfileSnapshot) Guidance(targetWorkload, targetTokensHash, targetDevice string) (*SearchGuidance, error) {
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
	if targetTokensHash != "" && s.TokensHash != "" && targetTokensHash != s.TokensHash {
		return nil, fmt.Errorf("%w: target tokens hash %q does not match snapshot %q",
			ErrWorkloadMismatch, targetTokensHash, s.TokensHash)
	}
	if targetDevice != s.Device {
		return nil, fmt.Errorf("%w: target device %q does not match snapshot %q",
			ErrDeviceMismatch, targetDevice, s.Device)
	}

	if s.Coverage <= 0.0 || len(s.Metrics) == 0 {
		return &SearchGuidance{
			Available:         false,
			UnavailableReason: "insufficient coverage or missing telemetry metrics",
			Workload:          s.Workload,
			Device:            s.Device,
			Coverage:          s.Coverage,
			UncertaintyNs:     s.ClockUncertaintyNs,
		}, ErrGuidanceUnavailable
	}

	// Identify primary stall bottlenecks based on hardware counters in original units
	stalls := make(map[string]int64)
	for k, v := range s.Metrics {
		stalls[k] = v
	}

	var primaryMetric string
	var maxStall int64
	var memStalls, sbStalls int64

	for k, v := range stalls {
		if v > maxStall {
			maxStall = v
			primaryMetric = k
		}
		if k == "smsp__pcsamp_warps_issue_stalled_long_scoreboard" ||
			k == "smsp__pcsamp_warps_issue_stalled_lg_throttle" ||
			k == "stall_mem" {
			memStalls += v
		}
		if k == "smsp__pcsamp_warps_issue_stalled_short_scoreboard" ||
			k == "stall_scoreboard" {
			sbStalls += v
		}
	}

	// Identify hot PCs from PCStalls
	type pcHit struct {
		pc    uint64
		count int64
	}
	var hits []pcHit
	for pc, m := range s.PCStalls {
		var tot int64
		for _, cnt := range m {
			tot += cnt
		}
		hits = append(hits, pcHit{pc: pc, count: tot})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].count == hits[j].count {
			return hits[i].pc < hits[j].pc
		}
		return hits[i].count > hits[j].count
	})
	var hotPCs []uint64
	for i := 0; i < len(hits) && i < 10; i++ {
		hotPCs = append(hotPCs, hits[i].pc)
	}

	return &SearchGuidance{
		Available:          true,
		Workload:           s.Workload,
		Device:             s.Device,
		PrimaryStallMetric: primaryMetric,
		StallCounts:        stalls,
		HotPCs:             hotPCs,
		MemoryBound:        memStalls > sbStalls && memStalls > 0,
		ScoreboardBound:    sbStalls >= memStalls && sbStalls > 0,
		Coverage:           s.Coverage,
		UncertaintyNs:      s.ClockUncertaintyNs,
	}, nil
}
