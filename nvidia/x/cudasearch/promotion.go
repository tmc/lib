package cudasearch

import (
	"fmt"
	"maps"
	"math"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// ValidateGeneration checks a proposal against the execution owner's current
// generation. Owners must repeat this check at the boundary switch, under their
// own synchronization. This method neither performs nor authorizes that switch.
func (p *PromotionProposal) ValidateGeneration(current int) error {
	if p == nil || current <= 0 || p.Generation != current {
		return ErrStaleGeneration
	}
	return nil
}

// ProposePromotion returns detached qualification evidence for the execution
// owner. confirmed must come from Confirm on this result's unchanged winner.
// Both evaluations must carry the owner's current positive generation. A
// proposal records observed evidence; it does not claim a statistically proven
// speedup, retain a live executable, or change an executing graph.
func (r *SearchResult) ProposePromotion(confirmed *CandidatePlan, currentGeneration int) (*PromotionProposal, error) {
	if r == nil || r.Winner == nil || confirmed == nil {
		return nil, ErrNoAdmittedPlan
	}
	winner := r.Winner
	if currentGeneration <= 0 || winner.Generation != currentGeneration || confirmed.Generation != currentGeneration {
		return nil, ErrStaleGeneration
	}
	if !r.PolicyComplete || r.PolicyID.Strategy == "" || r.PolicyID.Strategy != r.Strategy || r.PolicyID.RuleSet == "" {
		return nil, fmt.Errorf("cudasearch: incomplete promotion policy identity")
	}
	for _, c := range []*CandidatePlan{winner, confirmed} {
		if err := promotionEvidence(c); err != nil {
			return nil, err
		}
	}
	if confirmed.ConfirmationOf != winner.ID || winner.InputKey == "" || confirmed.InputKey == "" || winner.InputKey == confirmed.InputKey || winner.WorkloadKey == confirmed.WorkloadKey {
		return nil, fmt.Errorf("cudasearch: missing independent winner confirmation")
	}
	if winner.Receipt.TimingKind != confirmed.Receipt.TimingKind {
		return nil, ErrTimingKindMismatch
	}
	if winner.VerificationPolicy != confirmed.VerificationPolicy {
		return nil, fmt.Errorf("cudasearch: confirmation numerical policy differs from screening")
	}
	policy := r.PolicyID
	policy.Options = maps.Clone(policy.Options)
	return &PromotionProposal{CandidateID: winner.ID, Generation: currentGeneration, WorkloadKey: winner.WorkloadKey,
		Receipt: cloneTrials([]cudatrial.Receipt{*winner.Receipt})[0], PolicyID: policy,
		ConfirmationID: confirmed.ID, HeldOutWorkloadKey: confirmed.WorkloadKey,
		HeldOutReceipt:  cloneTrials([]cudatrial.Receipt{*confirmed.Receipt})[0],
		ScreeningTrials: cloneTrials(winner.Trials), HeldOutTrials: cloneTrials(confirmed.Trials),
		Comparison: maps.Clone(winner.Comparison), HeldOutComparison: maps.Clone(confirmed.Comparison), VerificationPolicy: winner.VerificationPolicy}, nil
}

func promotionEvidence(c *CandidatePlan) error {
	if err := admitSearchCandidate(c); err != nil {
		return err
	}
	if c.CostSource != CostMeasured || c.ID == "" || c.WorkloadKey == "" || len(c.Comparison) == 0 || len(c.Trials) < 4 || len(c.Trials)%4 != 0 {
		return fmt.Errorf("cudasearch: incomplete executable qualification evidence")
	}
	if err := c.VerificationPolicy.validate(); err != nil {
		return err
	}
	if c.Receipt.Name != "candidate" || c.Receipt.Provenance["executable_identity"] != c.ID {
		return fmt.Errorf("cudasearch: selected receipt executable identity mismatch")
	}
	for _, comparison := range c.Comparison {
		if comparison.Elements <= 0 || comparison.MaxAbsolute < 0 || comparison.MaxRelative < 0 || math.IsNaN(comparison.MaxAbsolute) || math.IsNaN(comparison.MaxRelative) || math.IsInf(comparison.MaxAbsolute, 0) || math.IsInf(comparison.MaxRelative, 0) {
			return fmt.Errorf("cudasearch: invalid numerical comparison evidence")
		}
	}
	for i, trial := range c.Trials {
		name := "baseline"
		if i%4 == 1 || i%4 == 2 {
			name = "candidate"
		}
		if trial.Name != name || trial.Round != i/2+1 || !trial.Complete || trial.Err != nil || trial.TimingKind != c.Receipt.TimingKind {
			return fmt.Errorf("cudasearch: invalid interleaved qualification trial")
		}
		if d, err := trial.DurationFor(trial.TimingKind); err != nil || d <= 0 {
			return fmt.Errorf("cudasearch: missing qualification trial timing")
		}
		if name == "candidate" && trial.Provenance["executable_identity"] != c.ID {
			return fmt.Errorf("cudasearch: qualification executable identity mismatch")
		}
	}
	return nil
}
