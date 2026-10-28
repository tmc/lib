package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strings"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// EvaluationObserver records [ExecutableEvaluator] attempts, typically in a
// durable append-only store. It is trusted caller code called synchronously
// on the evaluating goroutine, and it must not call back into the evaluator.
// Its own storage is covered by the search's declared callback envelope, as
// for Compile; it is not a security boundary or a process-wide quota.
//
// The evaluator observes every attempt it admits through one private owner
// shared by Evaluate, Confirm and each SearchRegisters option, so each attempt
// is recorded once whichever entry point admitted it.
//
// StartEvaluation runs after configuration and lease checks and before
// extraction, confirmation expression copying, compilation, or any output,
// run or close call on an executable. An error refuses the attempt: none of
// that work happens and FinishEvaluation is not called.
//
// Each successful StartEvaluation is followed by exactly one
// FinishEvaluation, after the compiled candidate is closed, the measurement
// reuse decision is made and ConfirmationOf is set. Its context keeps the
// attempt's values but not its cancellation, so a canceled attempt can still
// be recorded. An error rejects the candidate: the entry point admits no
// candidate and returns an error joining the attempt error, if any, with the
// recording failure. A panic in a callback is recorded with an abandonment
// error before it continues to unwind. Goexit remains Goexit. An abnormal
// executable Close does not establish resource cleanup and is not retried by
// the observer; candidate publication and its source/metadata are withdrawn.
type EvaluationObserver interface {
	StartEvaluation(context.Context, EvaluationStart) error
	FinishEvaluation(context.Context, EvaluationFinish) error
}

// EvaluationKind identifies the evaluator entry point of an attempt.
type EvaluationKind string

const (
	EvaluationSearch       EvaluationKind = "search"       // Evaluate
	EvaluationConfirmation EvaluationKind = "confirmation" // Confirm
	EvaluationRegister     EvaluationKind = "register"     // one SearchRegisters option
)

// EvaluationStart describes an admitted attempt before any work. Values are
// detached copies; strings and the derivation do not alias evaluator, search
// state or screened candidate storage. Binding values are copied shallowly.
type EvaluationStart struct {
	Attempt        uint64 // Per-evaluator sequence number, starting at 1.
	Kind           EvaluationKind
	Generation     int
	WorkloadKey    string
	InputKey       string
	BaselineID     string
	ConfirmationOf string // Screened candidate ID for a confirmation.
	Policy         OutputPolicy
	Rounds         int
	TimingKind     cudatrial.TimingKind
	Derivation     []egraph.Action // Search state, or screened candidate, derivation; nil for a register option.
}

// EvaluationFinish describes an attempt's terminal outcome. Candidate is set
// only when the evaluator admits a candidate. Evidence is a detached copy of
// the attempt's identity, comparison and trials, including failed ones. Err is
// the error the attempt returns before any recording failure, including cleanup
// and measurement-memo failures; it is the same value, not a copy, and is
// charged where Search retains it as a rejection. A caller may still reject an
// admitted candidate, as Search does under its memory budget and
// SearchRegisters does for an incomparable option; that rejection appears in
// the caller's result or error, not here.
type EvaluationFinish struct {
	Attempt           uint64
	Kind              EvaluationKind
	Candidate         *ObservedCandidate
	Evidence          EvaluationEvidence
	MeasurementReused bool
	Err               error
}

// ObservedCandidate is a detached summary of an admitted candidate. Its trials
// and comparison are in [EvaluationFinish.Evidence].
type ObservedCandidate struct {
	ID, ConfirmationOf, WorkloadKey, InputKey string
	Generation                                int
	Cost                                      float64 // Median measured candidate duration in nanoseconds.
	CostSource                                CostSource
	Validated                                 bool
}

// Observation copies are reserved in the attempt scope before allocation, so
// under a capped Search they stay charged with the attempt's other retained
// evidence until the next admitted attempt or a successful Close. An observer
// that keeps them longer retains caller storage outside the ledger.
func (e *ExecutableEvaluator) evaluationStart(s *egraph.SearchState, selection *egraph.Expr, screened *CandidatePlan) (EvaluationStart, error) {
	scope := e.attemptMemory
	start := EvaluationStart{Kind: EvaluationSearch, Generation: e.Generation, Policy: e.Policy, Rounds: e.Rounds, TimingKind: e.TimingKind}
	var derivation []egraph.Action
	switch {
	case screened != nil:
		start.Kind = EvaluationConfirmation
		derivation = screened.Derivation
		start.ConfirmationOf = screened.ID
	case selection != nil:
		// A register option has no search state; record no derivation.
		start.Kind = EvaluationRegister
	default:
		derivation = s.Derivation
	}
	var err error
	if start.Derivation, err = cloneDerivationWithMemory(scope, derivation); err != nil {
		return EvaluationStart{}, err
	}
	for _, field := range []struct {
		dst *string
		src string
	}{{&start.WorkloadKey, e.Workload.Key()}, {&start.InputKey, inputKey(e.Workload)}, {&start.BaselineID, e.Baseline.Identity()}, {&start.ConfirmationOf, start.ConfirmationOf}} {
		if err := scope.Reserve(int64(len(field.src))); err != nil {
			return EvaluationStart{}, err
		}
		*field.dst = strings.Clone(field.src)
	}
	e.observations++
	start.Attempt = e.observations
	return start, nil
}

func cloneDerivationWithMemory(scope *egraphcore.MemoryScope, in []egraph.Action) ([]egraph.Action, error) {
	if len(in) == 0 {
		return nil, nil
	}
	if err := bookkeepingSlots[egraph.Action](scope, len(in)); err != nil {
		return nil, err
	}
	for _, action := range in {
		if err := reserveBookkeepingMap(scope, action.Subst); err != nil {
			return nil, err
		}
		if err := reserveBookkeepingMap(scope, action.Binding); err != nil {
			return nil, err
		}
	}
	out := make([]egraph.Action, len(in))
	copy(out, in)
	for i := range out {
		out[i].Subst = maps.Clone(out[i].Subst)
		out[i].Binding = maps.Clone(out[i].Binding)
	}
	return out, nil
}

var errEvaluationAbandoned = errors.New("cudasearch: evaluation abandoned")

// finishEvaluation records the terminal outcome and returns the attempt's
// final result. A snapshot or recording failure, or a panic in
// FinishEvaluation, closes the candidate.
func (e *ExecutableEvaluator) finishEvaluation(ctx context.Context, start EvaluationStart, candidate *CandidatePlan, failure error, reused bool) (*CandidatePlan, error) {
	if candidate == nil && failure == nil {
		// Every return supplies one or the other; only a panic or Goexit in a
		// callback unwinds with neither.
		failure = errEvaluationAbandoned
	}
	if failure != nil && candidate != nil {
		candidate.Close()
		candidate = nil
	}
	finish := EvaluationFinish{Attempt: start.Attempt, Kind: start.Kind, MeasurementReused: reused}
	if err := e.evaluationFinish(&finish, candidate); err != nil {
		finish = EvaluationFinish{Attempt: start.Attempt, Kind: start.Kind, MeasurementReused: reused}
		failure = errors.Join(failure, fmt.Errorf("cudasearch: evaluation outcome snapshot: %w", err))
	}
	finish.Err = failure
	if failure != nil {
		finish.Candidate = nil
	}
	recorded := false
	defer func() {
		if !recorded {
			candidate.Close()
		}
	}()
	if err := e.Observer.FinishEvaluation(context.WithoutCancel(ctx), finish); err != nil {
		failure = errors.Join(failure, fmt.Errorf("cudasearch: record evaluation outcome: %w", err))
	}
	recorded = true
	if failure != nil {
		candidate.Close()
		return nil, failure
	}
	return candidate, nil
}

func (e *ExecutableEvaluator) evaluationFinish(finish *EvaluationFinish, candidate *CandidatePlan) error {
	scope := e.attemptMemory
	// attemptIdentity and comparisonArm are already private attempt copies.
	finish.Evidence = EvaluationEvidence{ExecutableID: e.attemptIdentity, ComparisonArm: e.comparisonArm}
	if err := reserveBookkeepingMap(scope, e.Comparison); err != nil {
		return err
	}
	finish.Evidence.Comparison = maps.Clone(e.Comparison)
	var err error
	if finish.Evidence.Trials, err = cloneTrialsWithMemory(e.Receipts, scope); err != nil {
		return err
	}
	if candidate == nil {
		return nil
	}
	if err := bookkeepingSlots[ObservedCandidate](scope, 1); err != nil {
		return err
	}
	c := &ObservedCandidate{Generation: candidate.Generation, Cost: candidate.Cost, CostSource: candidate.CostSource, Validated: candidate.Validated}
	for _, field := range []struct {
		dst *string
		src string
	}{{&c.ID, candidate.ID}, {&c.ConfirmationOf, candidate.ConfirmationOf}, {&c.WorkloadKey, candidate.WorkloadKey}, {&c.InputKey, candidate.InputKey}} {
		if err := scope.Reserve(int64(len(field.src))); err != nil {
			return err
		}
		*field.dst = strings.Clone(field.src)
	}
	finish.Candidate = c
	return nil
}
