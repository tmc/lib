package cudasearch

import (
	"maps"

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
	path := append([]egraph.Action(nil), s.Derivation...)
	for i := range path {
		path[i].Subst = maps.Clone(path[i].Subst)
	}
	r := CandidateRejection{Derivation: path, Err: err}
	if e, ok := eval.(*ExecutableEvaluator); ok {
		r.Evidence = &EvaluationEvidence{ExecutableID: e.attemptIdentity,
			ComparisonArm: e.comparisonArm, Comparison: maps.Clone(e.Comparison), Trials: cloneTrials(e.Receipts)}
	}
	return r
}
