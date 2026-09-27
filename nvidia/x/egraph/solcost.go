package egraph

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudatensor"
)

// ErrMissingSOLGraph reports a candidate that has no caller-supplied canonical
// tensor graph for whole-candidate SOL scoring.
var ErrMissingSOLGraph = errors.New("egraph: SOL candidate has no canonical graph")

// SOLCandidate pairs an extracted expression with the canonical einsum graph
// supplied by its caller. Expr is retained for reporting; SOL never infers
// tensor sizes or graph connectivity from the expression tree.
type SOLCandidate struct {
	Name  string
	Expr  *Expr
	Graph *cudatensor.CanonicalGraph
}

// SOLCostFunc returns the intended Fused SOL duration for one whole candidate.
type SOLCostFunc func(SOLCandidate) (float64, error)

// SOLCost creates a whole-candidate scorer using cudatensor's Fused bound.
// It does not implement CostModel: CostModel.NodeCost is additive per node and
// cannot represent graph-wide I/O deduplication or roofline combination.
func SOLCost(c cudatensor.Ceiling) SOLCostFunc {
	return func(candidate SOLCandidate) (float64, error) {
		name := candidate.Name
		if name == "" {
			name = "<unnamed>"
		}
		if candidate.Graph == nil {
			if candidate.Expr == nil {
				return 0, fmt.Errorf("%w: candidate %q also has no expression", ErrMissingSOLGraph, name)
			}
			return 0, fmt.Errorf("%w: candidate %q expression %q does not carry tensor metrics or einsum edges", ErrMissingSOLGraph, name, candidate.Expr.Op)
		}
		report, err := cudatensor.ComputeSOLBounds(candidate.Graph, c)
		if err != nil {
			return 0, fmt.Errorf("egraph: score candidate %q: %w", name, err)
		}
		if math.IsNaN(report.Fused) || math.IsInf(report.Fused, 0) {
			return 0, fmt.Errorf("egraph: candidate %q has invalid Fused SOL cost", name)
		}
		return report.Fused, nil
	}
}

// RankedSOLCandidate is a candidate and its Fused SOL cost in seconds.
type RankedSOLCandidate struct {
	Candidate SOLCandidate
	Cost      float64
}

// RankSOLCandidates orders already-extracted candidates by ascending Fused
// SOL duration. Stable sorting preserves caller order when costs are equal.
func RankSOLCandidates(candidates []SOLCandidate, score SOLCostFunc) ([]RankedSOLCandidate, error) {
	if score == nil {
		return nil, errors.New("egraph: nil SOL scorer")
	}
	ranked := make([]RankedSOLCandidate, len(candidates))
	for i, candidate := range candidates {
		cost, err := score(candidate)
		if err != nil {
			return nil, err
		}
		ranked[i] = RankedSOLCandidate{Candidate: candidate, Cost: cost}
	}
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].Cost < ranked[j].Cost })
	return ranked, nil
}
