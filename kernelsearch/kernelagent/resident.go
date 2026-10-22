package kernelagent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudasearch"
)

// PlanBinding associates an execution-owner supplied plan with its executable
// identity. cudaplan does not expose cudasearch's executable identity: the
// caller must establish this association from the compiled artifacts. The plan
// is borrowed and must remain alive through all outstanding completions.
type PlanBinding struct {
	CandidateID string
	Plan        *cudaplan.ExecutablePlan
}

// ErrPlanAssociationUnavailable means the public owner APIs cannot authenticate
// a search executable ID against a cudaplan.ExecutablePlan. Native promotion is
// refused until that association has an actual compiler/owner producer.
var ErrPlanAssociationUnavailable = errors.New("kernelagent: authenticated plan association unavailable")

// Resident selects a borrowed plan for subsequent launches. It never edits a
// CUDA Graph, closes a plan, or cancels an outstanding invocation. The caller
// owns both plans and completion lifetimes. The zero Resident is unusable.
type Resident struct {
	mu                sync.Mutex
	current, previous PlanBinding
	generation        int
	mode              Mode
}

// NewResident records the execution owner's positive generation and incumbent.
// HostStub permits declared fixture associations. Native promotion is refused
// until the owner provides an authenticated association API.
func NewResident(mode Mode, generation int, incumbent PlanBinding) (*Resident, error) {
	if generation <= 0 || incumbent.Plan == nil || incumbent.CandidateID == "" || (mode != HostStub && mode != Native) {
		return nil, fmt.Errorf("kernelagent: invalid resident incumbent")
	}
	return &Resident{current: incumbent, generation: generation, mode: mode}, nil
}

// Next returns the currently selected borrowed plan and its generation. A plan
// already obtained or submitted is unaffected by a later promotion or rollback.
func (r *Resident) Next() (PlanBinding, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.current, r.generation
}

// Execute submits through the selected plan's real ownership and completion
// contract. Selection and submission are serialized against boundary switches;
// the returned Completion continues to own that plan's borrowed buffers.
func (r *Resident) Execute(stream *cuda.Stream, bindings *cudaplan.Bindings) (*cudaplan.Completion, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.mode == Native && !nativeLease() {
		return nil, fmt.Errorf("kernelagent: native lease unavailable")
	}
	if r.current.Plan == nil {
		return nil, fmt.Errorf("kernelagent: no resident plan")
	}
	return r.current.Plan.Execute(stream, bindings)
}

// Promote switches only a HostStub next-launch plan after generation and identity
// admission. proposal must be produced by SearchResult.ProposePromotion;
// detached receipts are evidence, not an authenticated capability. This method
// cannot establish that a caller-supplied PlanBinding names the right artifact,
// and therefore refuses Native mode with ErrPlanAssociationUnavailable.
func (r *Resident) Promote(proposal *cudasearch.PromotionProposal, next PlanBinding) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.mode != HostStub {
		return ErrPlanAssociationUnavailable
	}
	if r.generation > math.MaxInt-2 {
		return fmt.Errorf("kernelagent: resident generation exhausted")
	}
	if err := proposal.ValidateGeneration(r.generation); err != nil {
		return err
	}
	if next.Plan == nil || next.CandidateID == "" || next.CandidateID != proposal.CandidateID || proposal.ConfirmationID == "" || len(proposal.ScreeningTrials) < 4 || len(proposal.HeldOutTrials) < 4 {
		return fmt.Errorf("kernelagent: incomplete plan promotion binding")
	}
	if r.previous.Plan != nil {
		return fmt.Errorf("kernelagent: previous promotion awaits remeasurement")
	}
	r.previous, r.current = r.current, next
	r.generation++
	return nil
}

func (r *Resident) resolve(expected int, regress bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if expected != r.generation || r.previous.Plan == nil {
		return cudasearch.ErrStaleGeneration
	}
	if regress {
		r.current = r.previous
		r.generation++
	}
	r.previous = PlanBinding{}
	return nil
}

// Remeasure confirms the unchanged search winner of result again against the
// configured borrowed baseline, through the existing evaluator and trial
// runner. result must be open: its SearchResult owns the winner's expression.
// The baseline must still be the previous incumbent. Regression or failed
// evidence restores the previous next-launch plan; outstanding launches are
// unaffected. The supplied epoch is a new archive epoch, not a candidate
// identity.
//
// On success the caller owns the returned confirmation and must Close it.
// On error the confirmation has already been closed and is not returned.
func (l *Loop) Remeasure(ctx context.Context, epoch string, resident *Resident, result *Result) (candidate *cudasearch.CandidatePlan, err error) {
	if err := l.Close(); err != nil {
		return nil, err
	}
	if resident == nil || result == nil || result.Search == nil || result.Search.Winner == nil || result.Search.Winner.Expr == nil || epoch == "" || len(epoch) > 128 {
		return nil, fmt.Errorf("kernelagent: incomplete remeasurement")
	}
	screened := result.Search.Winner
	for _, r := range l.archive.Records() {
		if r.Epoch == epoch {
			return nil, fmt.Errorf("kernelagent: epoch already recorded")
		}
	}
	resident.mu.Lock()
	current, previous, generation := resident.current, resident.previous, resident.generation
	resident.mu.Unlock()
	if current.CandidateID != screened.ID || previous.CandidateID != l.cfg.Baseline.Identity() {
		return nil, fmt.Errorf("kernelagent: remeasurement incumbent/baseline mismatch")
	}
	if l.cfg.Identity.Mode == Native && !nativeLease() {
		return nil, fmt.Errorf("kernelagent: native lease unavailable")
	}
	ctx, cancel := context.WithTimeout(ctx, l.cfg.Budget.Timeout)
	defer cancel()
	if _, err := l.archive.append(Record{Epoch: epoch, Kind: "epoch", Identity: &l.cfg.Identity, Numerical: &l.cfg.Numerical}); err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			candidate.Close()
			candidate = nil
		}
	}()
	e := l.evaluator(epoch, "remeasure", l.cfg.HeldOut, nil, generation, cancel)
	candidate, err = e.Confirm(ctx, screened)
	if cleanupErr := l.Close(); cleanupErr != nil {
		err = errors.Join(err, fmt.Errorf("kernelagent: remeasurement cleanup: %w", cleanupErr))
	}
	regress := err != nil || !improves(candidate)
	decision := "retained"
	if regress {
		decision = "rollback"
	}
	// A known regression restores the old next-launch plan even if evidence
	// storage fails. Report that failure; do not claim durable rollback evidence.
	if regress {
		if resolveErr := resident.resolve(generation, true); resolveErr != nil {
			return candidate, errors.Join(err, resolveErr)
		}
	}
	if _, archiveErr := l.archive.append(Record{Epoch: epoch, Kind: decision, Generation: generation, CandidateID: current.CandidateID, Outcome: decision, Reason: errorText(err)}); archiveErr != nil {
		return candidate, errors.Join(err, fmt.Errorf("kernelagent: %s evidence not durable: %w", decision, archiveErr))
	}
	if !regress {
		if resolveErr := resident.resolve(generation, false); resolveErr != nil {
			return candidate, resolveErr
		}
	}
	return candidate, err
}

func errorText(err error) string {
	if err != nil {
		return err.Error()
	}
	return ""
}
