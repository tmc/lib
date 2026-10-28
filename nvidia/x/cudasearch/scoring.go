package cudasearch

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

type scoringSnapshot struct {
	owner    *MockEvaluator
	model    ScoringModel
	evidence map[string]*ProfileEvidence
}

func snapshotScoring(m *MockEvaluator) *scoringSnapshot {
	s, _ := snapshotScoringWithMemory(m, nil)
	return s
}

func snapshotScoringWithMemory(m *MockEvaluator, scope *egraphcore.MemoryScope) (*scoringSnapshot, error) {
	if err := bookkeepingSlots[scoringSnapshot](scope, 1); err != nil {
		return nil, err
	}
	evidenceMap, err := bookkeepingMap[string, *ProfileEvidence](scope)
	if err != nil {
		return nil, err
	}
	s := &scoringSnapshot{owner: m, model: m.Scoring, evidence: evidenceMap}
	if s.model == (ScoringModel{}) {
		s.model = DefaultScoringModel("")
	}
	if s.model.EvidenceTime.IsZero() {
		s.model.EvidenceTime = time.Now()
	}
	for key, evidence := range m.Evidence {
		if evidence != nil {
			if err := bookkeepingSlots[ProfileEvidence](scope, 1); err != nil {
				return nil, err
			}
			// Keys and immutable epoch strings remain borrowed. The map entry
			// and copied observation are owned by this frozen snapshot.
			if err := bookkeepingPut(scope, s.evidence, key, (*ProfileEvidence)(nil)); err != nil {
				return nil, err
			}
			copy := *evidence
			s.evidence[key] = &copy
		}
	}
	return s, nil
}

func mockScoring(ctx context.Context, m *MockEvaluator) *scoringSnapshot {
	if work, _ := ctx.Value(workBudgetKey{}).(*workBudget); work != nil && work.scoring != nil && work.scoring.owner == m {
		return work.scoring
	}
	return snapshotScoring(m)
}

// ProfileEvidence captures hardware stall metrics and execution statistics from CUPTI/SASS profiling.
type ProfileEvidence struct {
	StallMemoryThrottle float64   // Fraction of cycles stalled on memory throttle [0, 1]
	StallScoreboard     float64   // Fraction of cycles stalled on scoreboard [0, 1]
	StallLGThrottle     float64   // Fraction of cycles stalled on local/global throttle [0, 1]
	Occupancy           float64   // Achieved SM active warp occupancy [0, 1]
	TargetOccupancy     float64   // Target threshold occupancy (e.g. 0.5)
	RegisterSpillBytes  uint64    // Bytes spilled to local memory
	CollectedAt         time.Time // Timestamp of profile collection
	Epoch               string    // Scoring epoch when evidence was gathered
}

// IsStale reports whether evidence belongs to a different or unspecified epoch.
func (pe ProfileEvidence) IsStale(currentEpoch string) bool {
	if pe.Epoch == "" || pe.Epoch != currentEpoch {
		return true
	}
	return false
}

// ScoringModel evaluates latency costs. Profile observations retain their own
// units and never add a fabricated duration to the latency objective.
type ScoringModel struct {
	// EvidenceMaxAge enables timestamp checks when positive. Zero uses only
	// epoch matching. Missing, future and expired timestamps fall back to timing.
	EvidenceMaxAge time.Duration
	// EvidenceTime is the reference time for age checks. Zero uses the current
	// time; Search freezes it before evaluating any candidate.
	EvidenceTime      time.Time
	TimingKind        cudatrial.TimingKind // Device objective; empty means summed kernel time.
	Epoch             string               // Monotonic version/identifier of the scoring model
	BaseLatencyWeight float64              // Weight for measured execution latency (e.g. 1.0)
}

// DefaultScoringModel returns a baseline scoring model.
func DefaultScoringModel(epoch string) ScoringModel {
	if epoch == "" {
		epoch = "epoch-v1"
	}
	return ScoringModel{
		Epoch:             epoch,
		BaseLatencyWeight: 1.0,
	}
}

// Score computes weighted latency and validates current profile evidence.
// Lower score is better. Evidence never changes the returned latency cost.
func (sm ScoringModel) Score(receipt *cudatrial.Receipt, evidence *ProfileEvidence) (float64, error) {
	if receipt == nil {
		return math.Inf(1), fmt.Errorf("cudasearch: nil receipt")
	}
	if !receipt.Complete || receipt.Err != nil {
		return math.Inf(1), fmt.Errorf("cudasearch: incomplete receipt")
	}
	if sm.EvidenceMaxAge < 0 {
		return math.Inf(1), fmt.Errorf("cudasearch: negative evidence maximum age")
	}

	if math.IsNaN(sm.BaseLatencyWeight) || math.IsInf(sm.BaseLatencyWeight, 0) || sm.BaseLatencyWeight < 0 {
		return math.Inf(1), fmt.Errorf("cudasearch: invalid scoring weight")
	}
	if sm.BaseLatencyWeight == 0 {
		return math.Inf(1), fmt.Errorf("cudasearch: latency weight must be positive")
	}
	kind := sm.TimingKind
	if kind == "" {
		kind = cudatrial.TimingSummedKernel
	}
	if kind != cudatrial.TimingSummedKernel && kind != cudatrial.TimingGPUBusy {
		return math.Inf(1), ErrTimingKindMismatch
	}
	if receipt.TimingKind != "" && receipt.TimingKind != kind {
		return math.Inf(1), ErrTimingKindMismatch
	}
	dur, err := receipt.DurationFor(kind)
	if err != nil {
		return math.Inf(1), err
	}
	if dur <= 0 {
		return math.Inf(1), fmt.Errorf("cudasearch: nonpositive scoring duration")
	}

	cost := float64(dur.Nanoseconds()) * sm.BaseLatencyWeight

	// Apply profile evidence if valid and current
	if sm.acceptsEvidence(evidence) {
		for _, fraction := range []float64{evidence.StallMemoryThrottle, evidence.StallScoreboard, evidence.StallLGThrottle, evidence.Occupancy, evidence.TargetOccupancy} {
			if math.IsNaN(fraction) || math.IsInf(fraction, 0) || fraction < 0 || fraction > 1 {
				return math.Inf(1), fmt.Errorf("cudasearch: invalid profile fraction")
			}
		}
	}

	if math.IsNaN(cost) || math.IsInf(cost, 0) {
		return math.Inf(1), fmt.Errorf("cudasearch: nonfinite score")
	}
	return cost, nil
}

func (sm ScoringModel) acceptsEvidence(evidence *ProfileEvidence) bool {
	if evidence == nil || evidence.IsStale(sm.Epoch) {
		return false
	}
	if sm.EvidenceMaxAge == 0 {
		return true
	}
	now := sm.EvidenceTime
	if now.IsZero() {
		now = time.Now()
	}
	if evidence.CollectedAt.IsZero() || evidence.CollectedAt.After(now) {
		return false
	}
	return now.Sub(evidence.CollectedAt) <= sm.EvidenceMaxAge
}
