package hopper

import (
	"fmt"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// CompileWithTempBudget optimizes and lowers a term before checking its device
// temporary storage. With reuse it checks PlanArena's allocation; otherwise it
// checks aligned storage without lifetime reuse. Inputs, outputs outside Temps,
// and kernel workspaces are excluded. A zero budget preserves Optimize followed
// by Compile.
//
// If the original optimum exceeds budget, at most eight alternatives in cost
// order are considered, with bounded extraction work. This is a bounded search,
// not proof that no feasible expression exists in the graph. No CUDA compilation
// or device allocation is performed.
func CompileWithTempBudget(g *egraph.EGraph, root egraph.Id, name string, out DType, budget uint64, reuse bool) (*Plan, error) {
	e, err := Optimize(g, root)
	if err != nil {
		return nil, err
	}
	p, err := Compile(e, name, out)
	if err != nil || budget == 0 {
		return p, err
	}
	fits := func(p *Plan) (bool, error) {
		a, err := PlanArena([]*Plan{p}, 0, reuse)
		if err != nil {
			return false, err
		}
		n := a.Bytes
		return n <= budget, nil
	}
	if ok, err := fits(p); err != nil {
		return nil, err
	} else if ok {
		return p, nil
	}
	candidates, err := egraph.ExtractTopKWithBudget(g, root, 8, Cost{}, egraphcore.NewWorkBudget(100000))
	if err != nil {
		return nil, fmt.Errorf("temporary budget candidate search: %w", err)
	}
	for _, c := range candidates {
		candidate, err := Compile(c.Expr, name, out)
		if err != nil {
			continue
		}
		ok, err := fits(candidate)
		if err != nil {
			return nil, err
		}
		if ok {
			return candidate, nil
		}
	}
	return nil, fmt.Errorf("temporary budget: no plan within %d bytes among optimum and %d candidates", budget, len(candidates))
}
