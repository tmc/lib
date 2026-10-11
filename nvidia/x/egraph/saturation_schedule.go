package egraph

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/egraphcore"
)

func tupleCount(g *EGraph) int {
	if g == nil {
		return 0
	}
	_, nodes := g.Size()
	maxInt := int(^uint(0) >> 1)
	if nodes < 0 {
		return 0
	}
	facts := g.Facts()
	if facts == nil {
		return nodes
	}
	rows := facts.TupleCount()
	if rows > maxInt-nodes {
		return maxInt
	}
	return nodes + rows
}

// SaturateScheduleWithReport executes an ordered rewrite schedule. Each run
// node is a transaction: all of its rules apply to a clone, and the clone is
// committed only when the complete group fits the node and match-work limits.
// Each RHS is checked against a conservative node-growth bound before it is
// instantiated, so a run does not build an oversized trial graph first.
// Hash-consing may make actual growth smaller than the bound, so admission can
// stop while the graph is still below MaxNodes. Completed run nodes remain
// committed if a later run fails or reaches a limit; only the active run is
// discarded.
func SaturateScheduleWithReport(g *EGraph, schedule *egraphcore.Schedule[*Rewrite], opts SaturationOptions) (SaturationResult, error) {
	if g == nil {
		return SaturationResult{}, fmt.Errorf("egraph: nil graph")
	}
	if opts.MaxIterations <= 0 {
		opts.MaxIterations = defaultSaturationIterations
	}
	if opts.MaxNodes <= 0 {
		opts.MaxNodes = defaultSaturationNodes
	}
	if opts.MaxTuples <= 0 {
		opts.MaxTuples = defaultSaturationTuples
	}
	if opts.MaxMatchWork <= 0 {
		opts.MaxMatchWork = defaultSaturationMatchWork
	}
	validate := func(rule *Rewrite) error { return validateRewrite(rule) }
	if err := egraphcore.ValidateSchedule(schedule, validate); err != nil {
		return SaturationResult{}, err
	}
	_, initialNodes := g.Size()
	initialTuples := tupleCount(g)
	result := SaturationResult{Nodes: initialNodes, Tuples: initialTuples}
	if initialNodes > opts.MaxNodes {
		result.StopReason = SaturationNodeLimit
		return result, nil
	}
	if initialTuples > opts.MaxTuples {
		result.StopReason = SaturationTupleLimit
		return result, nil
	}
	budget := egraphcore.NewWorkBudget(opts.MaxMatchWork)
	var mergesTotal, factsTotal, subsumptionsTotal int
	stopReason := SaturationStopReason(0)
	searchStates := make(map[*Rewrite]*RewriteSearchState)
	groupRun := func(_ string, rules []*Rewrite) (int, bool, error) {
		_, committedNodes := g.Size()
		if !budget.TrySpend(committedNodes) {
			stopReason = SaturationMatchWorkLimit
			return 0, true, nil
		}
		pass := g.Clone()
		passSearchStates := cloneRewriteSearchStates(searchStates)
		merges, factChanges, subsumptions := 0, 0, 0
		for _, rule := range rules {
			searchState := passSearchStates[rule]
			if searchState == nil {
				searchState = &RewriteSearchState{}
				passSearchStates[rule] = searchState
			}
			m, f, s, err := rule.applyWithSearchState(pass, budget, searchState, opts.MaxNodes, false)
			if errors.Is(err, ErrMatchWorkLimit) {
				stopReason = SaturationMatchWorkLimit
				return 0, true, nil
			}
			if errors.Is(err, egraphcore.ErrRewriteNodeLimit) {
				stopReason = SaturationNodeLimit
				return 0, true, nil
			}
			if err != nil {
				return 0, true, err
			}
			if m > math.MaxInt-merges || f > math.MaxInt-factChanges || s > math.MaxInt-subsumptions {
				return 0, true, fmt.Errorf("egraph: schedule change count overflow")
			}
			merges += m
			factChanges += f
			subsumptions += s
		}
		_, nodes := pass.Size()
		if nodes > opts.MaxNodes {
			stopReason = SaturationNodeLimit
			return 0, true, nil
		}
		if tupleCount(pass) > opts.MaxTuples {
			stopReason = SaturationTupleLimit
			return 0, true, nil
		}
		if merges > math.MaxInt-factChanges || merges+factChanges > math.MaxInt-subsumptions || merges > math.MaxInt-mergesTotal || factChanges > math.MaxInt-factsTotal || subsumptions > math.MaxInt-subsumptionsTotal {
			return 0, true, fmt.Errorf("egraph: schedule change count overflow")
		}
		// Cursor progress belongs to every successfully completed run, even
		// when its rules made no graph changes. A surrounding schedule may run
		// this group again after a later group changes facts; retaining the
		// cursors avoids rescanning the old relation contents.
		searchStates = passSearchStates
		if merges != 0 || factChanges != 0 || subsumptions != 0 {
			*g = *pass
			mergesTotal += merges
			factsTotal += factChanges
			subsumptionsTotal += subsumptions
		}
		return merges + factChanges + subsumptions, false, nil
	}

	scheduled, err := egraphcore.ExecuteSchedule(schedule, opts.MaxIterations, validate, groupRun)
	_, nodes := g.Size()
	result = SaturationResult{
		Iterations:   scheduled.Runs,
		Merges:       mergesTotal,
		FactChanges:  factsTotal,
		Subsumptions: subsumptionsTotal,
		MatchWork:    budget.Used(),
		Nodes:        nodes,
		Tuples:       tupleCount(g),
		Phases:       scheduled.Reports,
	}
	if err != nil {
		return result, err
	}
	switch {
	case stopReason != 0:
		result.StopReason = stopReason
	case scheduled.RunLimitReached:
		result.StopReason = SaturationIterationLimit
	default:
		result.StopReason = SaturationScheduleDone
	}
	return result, nil
}

func cloneRewriteSearchStates(states map[*Rewrite]*RewriteSearchState) map[*Rewrite]*RewriteSearchState {
	cloned := make(map[*Rewrite]*RewriteSearchState, len(states))
	for rule, state := range states {
		if state == nil {
			continue
		}
		copy := *state
		copy.cursors = append([]egraphcore.RelationCursor(nil), state.cursors...)
		cloned[rule] = &copy
	}
	return cloned
}
