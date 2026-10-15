package cudasearch

import (
	"maps"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// EvaluationEvidence retains an executable attempt, including failed trials.
// ComparisonArm identifies the arm whose outputs produced Comparison.
// Empty identities mean compilation did not produce an identified executable.
type EvaluationEvidence struct {
	ExecutableID  string
	ComparisonArm string
	Comparison    map[string]OutputComparison
	Trials        []cudatrial.Receipt
}

func cloneTrials(in []cudatrial.Receipt) []cudatrial.Receipt {
	if in == nil {
		return nil
	}
	out := append([]cudatrial.Receipt(nil), in...)
	for i := range out {
		out[i].Timings = maps.Clone(in[i].Timings)
		out[i].Provenance = maps.Clone(in[i].Provenance)
	}
	return out
}

func rejection(s *egraph.SearchState, eval Evaluator, err error) CandidateRejection {
	r, _ := rejectionWithMemory(nil, s, eval, err)
	return r
}

// Strings and Binding maps remain borrowed. Branch owners already keep
// Binding aliases alive. The retained error's reachable storage is charged
// conservatively; unsupported opaque fields refuse without calling Error.
// Reserve each collection before copying it. The existing checked reflection
// walk sizes errors without traversing unsupported opaque callback captures.
func rejectionWithMemory(scope *egraphcore.MemoryScope, s *egraph.SearchState, eval Evaluator, failure error) (CandidateRejection, error) {
	if scope != nil {
		bytes, err := storageBytesWithMemory(failure, scope)
		if err != nil {
			return CandidateRejection{}, err
		}
		if err := scope.Reserve(bytes); err != nil {
			return CandidateRejection{}, err
		}
	}
	if err := bookkeepingSlots[egraph.Action](scope, len(s.Derivation)); err != nil {
		return CandidateRejection{}, err
	}
	for _, action := range s.Derivation {
		if err := reserveBookkeepingMap(scope, action.Subst); err != nil {
			return CandidateRejection{}, err
		}
	}
	if e, ok := eval.(*ExecutableEvaluator); ok {
		if err := bookkeepingSlots[EvaluationEvidence](scope, 1); err != nil {
			return CandidateRejection{}, err
		}
		if err := reserveBookkeepingMap(scope, e.Comparison); err != nil {
			return CandidateRejection{}, err
		}
		if err := bookkeepingSlots[cudatrial.Receipt](scope, len(e.Receipts)); err != nil {
			return CandidateRejection{}, err
		}
		for _, receipt := range e.Receipts {
			if err := reserveBookkeepingMap(scope, receipt.Timings); err != nil {
				return CandidateRejection{}, err
			}
			if err := reserveBookkeepingMap(scope, receipt.Provenance); err != nil {
				return CandidateRejection{}, err
			}
		}
	}
	var path []egraph.Action
	if len(s.Derivation) != 0 {
		path = make([]egraph.Action, len(s.Derivation))
	}
	copy(path, s.Derivation)
	for i := range path {
		path[i].Subst = maps.Clone(path[i].Subst)
	}
	r := CandidateRejection{Derivation: path, Err: failure}
	if e, ok := eval.(*ExecutableEvaluator); ok {
		var trials []cudatrial.Receipt
		if len(e.Receipts) != 0 {
			trials = make([]cudatrial.Receipt, len(e.Receipts))
		}
		copy(trials, e.Receipts)
		for i := range trials {
			trials[i].Timings = maps.Clone(trials[i].Timings)
			trials[i].Provenance = maps.Clone(trials[i].Provenance)
		}
		r.Evidence = &EvaluationEvidence{ExecutableID: e.attemptIdentity,
			ComparisonArm: e.comparisonArm, Comparison: maps.Clone(e.Comparison), Trials: trials}
	}
	return r, nil
}

func reserveBookkeepingMap[K comparable, V any](scope *egraphcore.MemoryScope, values map[K]V) error {
	if values == nil {
		return nil
	}
	if err := bookkeepingSlots[map[K]V](scope, 1); err != nil {
		return err
	}
	return bookkeepingSlots[struct {
		key   K
		value V
	}](scope, len(values))
}
