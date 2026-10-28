package cudasearch

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func inputKey(b WorkloadBindings) string {
	if len(b.InputIdentities) == 0 {
		return ""
	}
	for name, identity := range b.InputIdentities {
		if name == "" || identity == "" {
			return ""
		}
	}
	return (WorkloadBindings{Name: "held-out-inputs-v1", InputIdentities: b.InputIdentities}).Key()
}

// Confirm evaluates the screened expression on this evaluator's held-out
// workload without searching or extracting another alternative. Screening must
// be measured and declare input identities distinct from this workload's inputs.
// Callers supply independent references and truthful content identities; this
// check cannot establish statistical independence or a speedup by itself.
// The screened candidate must remain unchanged after evaluation. Operator Data
// is borrowed and must not be mutated by the compiler.
// The result records ConfirmationOf and retains numerical and ABBA evidence.
// An Observer records the start before the expression is copied and the
// outcome after ConfirmationOf is set. Confirmation does not authorize a
// promotion or mutate an executing graph.
func (e *ExecutableEvaluator) Confirm(ctx context.Context, screened *CandidatePlan) (*CandidatePlan, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if screened == nil || screened.ID == "" || screened.CostSource != CostMeasured {
		return nil, fmt.Errorf("cudasearch: confirmation requires measured screening")
	}
	if err := admitSearchCandidate(screened); err != nil {
		return nil, err
	}
	key := inputKey(e.Workload)
	if screened.WorkloadKey == "" || screened.InputKey == "" || key == "" || key == screened.InputKey || e.Workload.Key() == screened.WorkloadKey {
		return nil, fmt.Errorf("cudasearch: confirmation requires distinct declared inputs")
	}
	if screened.Receipt.TimingKind != e.TimingKind {
		return nil, ErrTimingKindMismatch
	}
	if screened.VerificationPolicy != e.Policy {
		return nil, fmt.Errorf("cudasearch: confirmation numerical policy differs from screening")
	}
	return e.evaluate(ctx, nil, 0, nil, screened)
}

// Copy the bounded tree before passing it to a compiler. Data remains borrowed,
// as in extraction; compilers must not mutate operator payloads.
func confirmationExpr(ctx context.Context, root *egraph.Expr) (*egraph.Expr, error) {
	limits := egraph.DefaultExtractionLimits()
	nodes := 0
	var copyExpr func(*egraph.Expr, int) (*egraph.Expr, error)
	copyExpr = func(expr *egraph.Expr, depth int) (*egraph.Expr, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if expr == nil || expr.Op == "" {
			return nil, fmt.Errorf("cudasearch: invalid confirmation expression")
		}
		nodes++
		if depth > limits.MaxDepth || nodes > limits.MaxNodes || len(expr.Children) > limits.MaxNodes-nodes {
			return nil, fmt.Errorf("cudasearch: confirmation expression: %w", egraph.ErrExtractionLimit)
		}
		out := &egraph.Expr{Op: expr.Op, Data: expr.Data, Children: make([]*egraph.Expr, len(expr.Children))}
		for i, child := range expr.Children {
			var err error
			out.Children[i], err = copyExpr(child, depth+1)
			if err != nil {
				return nil, err
			}
		}
		return out, nil
	}
	return copyExpr(root, 1)
}
