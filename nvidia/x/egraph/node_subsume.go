package egraph

import "github.com/tmc/lib/egraphcore"

func subsumeMatchedNodes(g *EGraph, rule *Rewrite, matches []MatchMatch, budget *egraphcore.WorkBudget) (int, error) {
	if rule == nil || !rule.subsumesLHS {
		return 0, nil
	}
	variables, err := graphMap[string, bool](budget, 0)
	if err != nil {
		return 0, err
	}
	if err := patternVariablesWithBudget(rule.LHS, variables, budget); err != nil {
		return 0, err
	}
	changes := 0
	for _, match := range matches {
		classID := g.Find(match.ClassID)
		class := g.storage().Class(classID)
		if class == nil {
			return changes, graphErrorf(budget, "egraph: rewrite %q cannot subsume missing class %d", rule.Name, classID)
		}
		matched := false
		matchedSubsumed := false
		for i, node := range class.Nodes {
			if !budget.TrySpend(1) {
				return changes, ErrMatchWorkLimit
			}
			if !rule.LHS.matchesNode(node) {
				continue
			}
			substitutions, err := graphSlice[Subst](budget, 1, 1)
			if err != nil {
				return changes, err
			}
			initial, err := graphMap[string, Id](budget, 0)
			if err != nil {
				return changes, err
			}
			substitutions[0] = Subst(initial)
			for column, childPattern := range rule.LHS.Children {
				var next []Subst
				for _, subst := range substitutions {
					matches, err := childPattern.matchEClassBudget(g, node.Children[column], subst, budget)
					if err != nil {
						return changes, err
					}
					next, err = graphAppend(budget, next, matches...)
					if err != nil {
						return changes, err
					}
				}
				substitutions = next
				if len(substitutions) == 0 {
					break
				}
			}
			for _, subst := range substitutions {
				matches := true
				for name := range variables {
					want, wantOK := match.Subst[name]
					got, gotOK := subst[name]
					if !wantOK || !gotOK || g.Find(want) != g.Find(got) {
						matches = false
						break
					}
				}
				if !matches {
					continue
				}
				if node.Subsumed {
					matchedSubsumed = true
					continue
				}
				worklist, err := graphAppend(budget, g.worklist, classID)
				if err != nil {
					return changes, err
				}
				class.Nodes[i].Subsumed = true
				g.worklist = worklist
				g.revision++
				changes++
				matched = true
				break
			}
			if matched {
				break
			}
		}
		if !matched && !matchedSubsumed {
			return changes, graphErrorf(budget, "egraph: rewrite %q match in class %d has no visible node to subsume", rule.Name, classID)
		}
	}
	return changes, nil
}
