package metaltensor

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/egraph"
)

// LowerExecutionPlan materializes choice from this candidate space and lowers
// it into separate Metal kernel stages. The caller still owns buffer
// allocation, binding, and dispatch.
func (s *TensorCandidateSpace) LowerExecutionPlan(choice egraph.CandidateChoice) (*TensorExecutionPlan, error) {
	if s == nil || s.snapshot == nil {
		return nil, fmt.Errorf("tensor candidate: nil snapshot")
	}
	expr, err := s.snapshot.Materialize(choice)
	if err != nil {
		return nil, fmt.Errorf("tensor candidate: materialize execution choice: %w", err)
	}
	return lowerTensorCandidateExpr(expr, s.dtype.String(), nil, "tensor candidate")
}

// LowerExecutionPlan materializes choice for one concrete dimension binding
// and lowers it into separate Metal kernel stages. This validates the selected
// expression at that binding; callers must still validate the full bucket
// before promoting a candidate across its interval.
func (s *TensorBucketCandidateSpace) LowerExecutionPlan(choice egraph.CandidateChoice, dimensions map[string]int64) (*TensorExecutionPlan, error) {
	if s == nil || s.snapshot == nil {
		return nil, fmt.Errorf("tensor bucket: nil snapshot")
	}
	expr, err := s.snapshot.Materialize(choice)
	if err != nil {
		return nil, fmt.Errorf("tensor bucket: materialize execution choice: %w", err)
	}
	return lowerTensorCandidateExpr(expr, s.dtype.String(), dimensions, "tensor bucket")
}

func lowerTensorCandidateExpr(expr *egraph.Expr, expectedSort string, dimensions map[string]int64, context string) (*TensorExecutionPlan, error) {
	if expr == nil {
		return nil, fmt.Errorf("%s: nil expression", context)
	}
	if expr.Sort != expectedSort {
		return nil, fmt.Errorf("%s: root sort %q does not match tensor type %s", context, expr.Sort, expectedSort)
	}
	plan, err := LowerTensorExecutionPlan(expr, dimensions)
	if err != nil {
		return nil, fmt.Errorf("%s: lower execution plan: %w", context, err)
	}
	return plan, nil
}
