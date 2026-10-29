package egraph

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

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
	started := time.Now()
	var mergesTotal, factsTotal, subsumptionsTotal int
	var stopReason SaturationStopReason
	var stopRun, stopRule, cappedRule string
	searchStates := make(map[*Rewrite]*RewriteSearchState)
	applied := make(map[string]int) // committed matches by rule name
	var stats []RuleStats
	statIndex := make(map[[2]string]int)
	// stat returns the statistics of rule in the run group at path, keyed
	// by the group's name so repeated passes accumulate.
	stat := func(path, rule string) *RuleStats {
		run := path
		if i := strings.LastIndex(path, "/run("); i >= 0 {
			run = strings.TrimSuffix(path[i+len("/run("):], ")")
		}
		key := [2]string{run, rule}
		i, ok := statIndex[key]
		if !ok {
			i = len(stats)
			statIndex[key] = i
			stats = append(stats, RuleStats{Run: run, Name: rule})
		}
		return &stats[i]
	}
	stop := func(reason SaturationStopReason, run, rule string) (int, bool, error) {
		stopReason, stopRun, stopRule = reason, run, rule
		return 0, true, nil
	}
	groupRun := func(run string, rules []*Rewrite) (int, bool, error) {
		if opts.MaxDuration > 0 && time.Since(started) >= opts.MaxDuration {
			return stop(SaturationTimeLimit, run, "")
		}
		_, committedNodes := g.Size()
		if !budget.TrySpend(committedNodes) {
			return stop(SaturationMatchWorkLimit, run, "")
		}
		pass := g.Clone()
		passSearchStates := cloneRewriteSearchStates(searchStates)
		passApplied := make([]ruleApply, len(rules))
		merges, factChanges, subsumptions := 0, 0, 0
		for i, rule := range rules {
			limit := 0
			if n := opts.RuleMatches[rule.Name]; n > 0 {
				limit = n - applied[rule.Name]
				if limit <= 0 {
					stat(run, rule.Name).Capped = true
					if cappedRule == "" {
						cappedRule = rule.Name
					}
					continue
				}
			}
			searchState := passSearchStates[rule]
			if searchState == nil {
				searchState = &RewriteSearchState{}
				passSearchStates[rule] = searchState
			}
			ruleStarted := time.Now()
			a, err := rule.applyWithSearchState(pass, budget, searchState, opts.MaxNodes, limit, false)
			st := stat(run, rule.Name)
			st.Calls++
			st.Found += a.found
			st.Duration += time.Since(ruleStarted)
			if errors.Is(err, ErrMatchWorkLimit) {
				return stop(SaturationMatchWorkLimit, run, rule.Name)
			}
			if errors.Is(err, egraphcore.ErrRewriteNodeLimit) {
				return stop(SaturationNodeLimit, run, rule.Name)
			}
			if err != nil {
				return 0, true, fmt.Errorf("rule %q: %w", rule.Name, err)
			}
			m, f, s := a.merges, a.factChanges, a.subsumptions
			if m > math.MaxInt-merges || f > math.MaxInt-factChanges || s > math.MaxInt-subsumptions {
				return 0, true, fmt.Errorf("egraph: schedule change count overflow")
			}
			merges += m
			factChanges += f
			subsumptions += s
			passApplied[i] = a
		}
		_, nodes := pass.Size()
		if nodes > opts.MaxNodes {
			return stop(SaturationNodeLimit, run, "")
		}
		if tupleCount(pass) > opts.MaxTuples {
			return stop(SaturationTupleLimit, run, "")
		}
		if merges > math.MaxInt-factChanges || merges+factChanges > math.MaxInt-subsumptions || merges > math.MaxInt-mergesTotal || factChanges > math.MaxInt-factsTotal || subsumptions > math.MaxInt-subsumptionsTotal {
			return 0, true, fmt.Errorf("egraph: schedule change count overflow")
		}
		// Cursor progress belongs to every successfully completed run, even
		// when its rules made no graph changes. A surrounding schedule may run
		// this group again after a later group changes facts; retaining the
		// cursors avoids rescanning the old relation contents.
		searchStates = passSearchStates
		for i, rule := range rules {
			a := passApplied[i]
			if a.found == 0 {
				continue
			}
			st := stat(run, rule.Name)
			st.Applied += a.applied
			st.Merges += a.merges
			applied[rule.Name] += a.applied
			if a.applied < a.found {
				st.Capped = true
				if cappedRule == "" {
					cappedRule = rule.Name
				}
			}
		}
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
		Rules:        stats,
	}
	if err != nil {
		return result, err
	}
	switch {
	case stopReason != 0:
		result.StopReason, result.StopRun, result.StopRule = stopReason, stopRun, stopRule
	case scheduled.RunLimitReached:
		result.StopReason = SaturationIterationLimit
	case cappedRule != "":
		result.StopReason, result.StopRule = SaturationRuleMatchLimit, cappedRule
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
