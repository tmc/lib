package cudasearch

import (
	"fmt"
	"math"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// ProfileEvidence captures hardware stall metrics and execution statistics from CUPTI/SASS profiling.
type ProfileEvidence struct {
	StallMemoryThrottle float64       // Fraction of cycles stalled on memory throttle [0, 1]
	StallScoreboard     float64       // Fraction of cycles stalled on scoreboard [0, 1]
	StallLGThrottle     float64       // Fraction of cycles stalled on local/global throttle [0, 1]
	Occupancy           float64       // Achieved SM active warp occupancy [0, 1]
	TargetOccupancy     float64       // Target threshold occupancy (e.g. 0.5)
	RegisterSpillBytes  uint64        // Bytes spilled to local memory
	CollectedAt         time.Time     // Timestamp of profile collection
	Epoch               string        // Scoring epoch when evidence was gathered
}

// IsStale reports whether the evidence was gathered under an older scoring epoch or is expired.
func (pe ProfileEvidence) IsStale(currentEpoch string) bool {
	if pe.Epoch == "" || pe.Epoch != currentEpoch {
		return true
	}
	return false
}

// ScoringModel evaluates candidate costs by combining latency measurements with profile penalties.
type ScoringModel struct {
	Epoch               string  // Monotonic version/identifier of the scoring model
	BaseLatencyWeight   float64 // Weight for measured execution latency (e.g. 1.0)
	MemoryStallWeight   float64 // Penalty weight for stall_memory_throttle (e.g. 5000.0)
	OccupancyPenalty    float64 // Penalty weight for falling below target occupancy (e.g. 2000.0)
	RegisterSpillWeight float64 // Penalty per KB of register spilling (e.g. 10.0)
}

// DefaultScoringModel returns a baseline scoring model.
func DefaultScoringModel(epoch string) ScoringModel {
	if epoch == "" {
		epoch = "epoch-v1"
	}
	return ScoringModel{
		Epoch:               epoch,
		BaseLatencyWeight:   1.0,
		MemoryStallWeight:   5000.0,
		OccupancyPenalty:    2000.0,
		RegisterSpillWeight: 10.0,
	}
}

// Score computes a composite cost for a candidate receipt and optional profile evidence.
// Lower score is better. If profile evidence is missing or stale, it falls back cleanly to timing-only.
func (sm ScoringModel) Score(receipt *cudatrial.Receipt, evidence *ProfileEvidence) (float64, error) {
	if receipt == nil {
		return math.Inf(1), fmt.Errorf("cudasearch: nil receipt")
	}
	if !receipt.Complete || receipt.Err != nil {
		return math.Inf(1), fmt.Errorf("cudasearch: incomplete receipt")
	}

	dur := receipt.Duration()
	if dur <= 0 {
		dur = time.Duration(receipt.DeviceNs)
	}
	if dur <= 0 {
		dur = time.Duration(receipt.WallNs)
	}
	if dur <= 0 {
		dur = time.Microsecond
	}

	cost := float64(dur.Nanoseconds()) * sm.BaseLatencyWeight

	// Apply profile evidence if valid and current
	if evidence != nil && !evidence.IsStale(sm.Epoch) {
		// Penalty for memory throttle stalls
		if evidence.StallMemoryThrottle > 0 {
			cost += evidence.StallMemoryThrottle * sm.MemoryStallWeight * 1000.0
		}
		// Penalty for sub-target occupancy
		target := evidence.TargetOccupancy
		if target <= 0 {
			target = 0.5
		}
		if evidence.Occupancy < target {
			deficit := target - evidence.Occupancy
			cost += deficit * sm.OccupancyPenalty * 1000.0
		}
		// Penalty for register spilling
		if evidence.RegisterSpillBytes > 0 {
			kb := float64(evidence.RegisterSpillBytes) / 1024.0
			cost += kb * sm.RegisterSpillWeight * 1000.0
		}
	}

	return cost, nil
}
