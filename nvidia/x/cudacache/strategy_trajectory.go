package cudacache

import (
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// RecordTrajectory updates matching catalog techniques from a qualified
// declaration. rules maps rewrite names to techniques, explicitly. A failed
// correctness check reduces confidence and never contributes a speedup.
// Confidence is updated as a running success fraction with the original prior
// given one observation of weight. At most 4096 observations per strategy and
// the most recent 256 successful speedups are retained. Failure is atomic.
// Integrity qualification is not producer authentication.
func (k *StrategyKnowledgeBase) RecordTrajectory(r cudatrial.TrajectoryReceipt, rules map[string]string) error {
	if !r.HardwareObserved() {
		return ErrUnqualifiedOutcome
	}
	return k.recordTrajectory(r, rules)
}

// RecordTrustedTrajectory updates priors from a persisted declaration after
// the caller independently trusts its provenance. Integrity is not proof of
// execution; synthetic measurements must not be imported as hardware evidence.
func (k *StrategyKnowledgeBase) RecordTrustedTrajectory(r cudatrial.TrajectoryReceipt, rules map[string]string) error {
	return k.recordTrajectory(r, rules)
}

func (k *StrategyKnowledgeBase) recordTrajectory(r cudatrial.TrajectoryReceipt, rules map[string]string) error {
	if k == nil {
		return fmt.Errorf("cudacache: nil knowledge base")
	}
	if err := r.Validate(); err != nil {
		return fmt.Errorf("%w: %w", ErrUnqualifiedOutcome, err)
	}
	if len(rules) > 4096 {
		return fmt.Errorf("cudacache: technique mapping limit")
	}
	for rule, technique := range rules {
		if len(rule) > 256 || len(technique) > 128 {
			return fmt.Errorf("cudacache: technique mapping name limit")
		}
	}
	techn := make(map[string]bool)
	for _, rule := range r.AppliedRules {
		if name := rules[rule]; name != "" {
			techn[name] = true
		}
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	rows := make([]Strategy, len(k.rows))
	matched := 0
	for i, s := range k.rows {
		s = cloneStrategy(s)
		if s.Family == r.BottleneckState && s.TargetArch == r.TargetArch && techn[s.Technique] {
			matched++
			duplicate := false
			for _, id := range s.TrajectoryIDs {
				if id == r.SHA256 {
					duplicate = true
					break
				}
			}
			if duplicate {
				rows[i] = s
				continue
			}
			if s.Observations == 4096 {
				return fmt.Errorf("cudacache: strategy observation limit")
			}
			success := 0.0
			if r.VerifiedCorrect {
				success = 1
				if len(s.HistoricalSpeedups) == 256 {
					s.HistoricalSpeedups = s.HistoricalSpeedups[1:]
				}
				s.HistoricalSpeedups = append(s.HistoricalSpeedups, r.SpeedupRatio)
			} else {
				s.Failures++
			}
			s.Confidence = (s.Confidence*float64(s.Observations+1) + success) / float64(s.Observations+2)
			s.Observations++
			s.TrajectoryIDs = append(s.TrajectoryIDs, r.SHA256)
		}
		rows[i] = s
	}
	if matched == 0 {
		return fmt.Errorf("cudacache: trajectory has no matching strategy")
	}
	b, err := json.Marshal(rows)
	if err != nil || len(b) > strategyBytes-1024 {
		return fmt.Errorf("cudacache: strategy byte limit")
	}
	k.rows = rows
	return nil
}
