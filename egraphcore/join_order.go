package egraphcore

// JoinOrderCondition describes one candidate set for a conjunctive match.
// Variables must list the set's distinct binding names; Cardinality is the
// number of candidate bindings.
type JoinOrderCondition struct {
	Variables   []string
	Cardinality int
}

// PlanJoinOrder greedily orders conditions to prefer the greatest number of
// bindings shared with variables already bound, then the smallest candidate
// set. Original condition index breaks ties, so plans are deterministic. The
// input slices are not modified. A nil budget is unlimited.
func PlanJoinOrder(boundVariables []string, conditions []JoinOrderCondition, budget *WorkBudget) ([]int, error) {
	bound := make(map[string]struct{}, len(boundVariables))
	for _, name := range boundVariables {
		if !spend(budget, 1) {
			return nil, ErrMatchWorkLimit
		}
		if name == "" {
			return nil, ErrInvalidJoin
		}
		bound[name] = struct{}{}
	}
	remaining := make([]bool, len(conditions))
	for i, condition := range conditions {
		if condition.Cardinality < 0 {
			return nil, ErrInvalidJoin
		}
		remaining[i] = true
	}
	order := make([]int, 0, len(conditions))
	for len(order) < len(conditions) {
		selected := -1
		selectedShared := -1
		selectedCardinality := 0
		for i, condition := range conditions {
			if !remaining[i] {
				continue
			}
			shared := 0
			seen := make(map[string]struct{}, len(condition.Variables))
			for _, name := range condition.Variables {
				if !spend(budget, 1) {
					return nil, ErrMatchWorkLimit
				}
				if name == "" {
					return nil, ErrInvalidJoin
				}
				if _, duplicate := seen[name]; duplicate {
					continue
				}
				seen[name] = struct{}{}
				if _, ok := bound[name]; ok {
					shared++
				}
			}
			if selected < 0 || shared > selectedShared || shared == selectedShared && condition.Cardinality < selectedCardinality {
				selected = i
				selectedShared = shared
				selectedCardinality = condition.Cardinality
			}
		}
		if selected < 0 {
			return nil, ErrInvalidJoin
		}
		remaining[selected] = false
		order = append(order, selected)
		for _, name := range conditions[selected].Variables {
			bound[name] = struct{}{}
		}
	}
	return order, nil
}
