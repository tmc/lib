package egraphcore

import (
	"cmp"
	"math"
	"math/bits"
	"slices"
)

// JoinMatchesWithBindings joins structural matches with a precomputed set of
// relation bindings. anchors are the matches for the first structural
// pattern. Each structuralConditions entry is an alternative set for one
// additional condition. relationBindings is the result of a required relation
// condition; an empty slice yields no matches. To represent no relation
// condition, pass one empty binding. ClassID is taken from anchors and
// class-valued bindings are canonicalized before comparison and output.
//
// Results are deterministic and deduplicated. Work exhaustion returns no
// partial result. With no anchors, this function returns no matches. A nil
// budget or canonicalize function is invalid.
func JoinMatchesWithBindings[ID ~uint32](
	anchors []Match[ID],
	structuralConditions [][]Match[ID],
	relationBindings []Binding[ID],
	canonicalize func(ID) ID,
	budget *WorkBudget,
) (output []RuleMatch[ID], err error) {
	defer func() {
		if memoryErr := budget.MemoryError(); memoryErr != nil {
			output, err = nil, memoryErr
		}
	}()
	if budget == nil || canonicalize == nil {
		return nil, ErrInvalidJoin
	}
	if len(anchors) == 0 {
		return nil, nil
	}
	if len(relationBindings) == 0 {
		return nil, nil
	}
	if valid, complete := uniformMatchSubst(anchors, budget); !complete {
		return nil, ErrMatchWorkLimit
	} else if !valid {
		return nil, ErrInvalidJoin
	}
	type bindingSet struct {
		bindings  []Binding[ID]
		variables []string
	}
	sets, ok := memorySlice[bindingSet](budget, 0, len(structuralConditions)+1)
	if !ok {
		return nil, matchMemoryError(budget)
	}
	plan, ok := memorySlice[JoinOrderCondition](budget, 0, len(structuralConditions)+1)
	if !ok {
		return nil, matchMemoryError(budget)
	}
	for _, alternatives := range structuralConditions {
		if len(alternatives) == 0 {
			return nil, nil
		}
		if valid, complete := uniformMatchSubst(alternatives, budget); !complete {
			return nil, ErrMatchWorkLimit
		} else if !valid {
			return nil, ErrInvalidJoin
		}
		storage, ok := memorySlice[Binding[ID]](budget, len(alternatives), len(alternatives))
		if !ok {
			return nil, matchMemoryError(budget)
		}
		set := bindingSet{bindings: storage}
		for i, match := range alternatives {
			var ok bool
			set.bindings[i], ok = matchBinding(match.Subst, canonicalize, budget)
			if !ok {
				return nil, ErrMatchWorkLimit
			}
		}
		for name := range alternatives[0].Subst {
			set.variables, ok = memoryAppend(budget, set.variables, name)
			if !ok {
				return nil, matchMemoryError(budget)
			}
		}
		sets = append(sets, set)
		plan = append(plan, JoinOrderCondition{Variables: set.variables, Cardinality: len(set.bindings)})
	}
	if valid, complete := uniformBindingSchema(relationBindings, budget); !complete {
		return nil, ErrMatchWorkLimit
	} else if !valid {
		return nil, ErrInvalidJoin
	}
	set := bindingSet{bindings: relationBindings}
	for name := range relationBindings[0] {
		set.variables, ok = memoryAppend(budget, set.variables, name)
		if !ok {
			return nil, matchMemoryError(budget)
		}
	}
	sets = append(sets, set)
	plan = append(plan, JoinOrderCondition{Variables: set.variables, Cardinality: len(set.bindings)})
	bound, ok := memorySlice[string](budget, 0, len(anchors[0].Subst))
	if !ok {
		return nil, matchMemoryError(budget)
	}
	for name := range anchors[0].Subst {
		bound = append(bound, name)
	}
	order, err := PlanJoinOrder(bound, plan, budget)
	if err != nil {
		return nil, err
	}
	var results []RuleMatch[ID]
	seen, ok := memoryMap[string, struct{}](budget, 0)
	if !ok {
		return nil, matchMemoryError(budget)
	}
	for _, anchor := range anchors {
		if !spend(budget, 1) {
			return nil, ErrMatchWorkLimit
		}
		anchorID := canonicalize(anchor.ClassID)
		current, ok := matchBinding(anchor.Subst, canonicalize, budget)
		if !ok {
			return nil, ErrMatchWorkLimit
		}
		bindings, ok := memorySlice[Binding[ID]](budget, 1, 1)
		if !ok {
			return nil, matchMemoryError(budget)
		}
		bindings[0] = current
		for _, conditionIndex := range order {
			right := sets[conditionIndex].bindings
			keys, ok := sharedNames(bindings, right, budget)
			if !ok {
				return nil, ErrMatchWorkLimit
			}
			var complete bool
			var err error
			bindings, complete, err = JoinBindings(bindings, right, keys, canonicalize, budget)
			if err != nil {
				return nil, err
			}
			if !complete {
				return nil, ErrMatchWorkLimit
			}
			if len(bindings) == 0 {
				break
			}
		}
		if len(bindings) == 0 {
			continue
		}
		for _, binding := range bindings {
			if !spend(budget, 1) {
				return nil, ErrMatchWorkLimit
			}
			key, ok := anchoredBindingKey(anchorID, binding, budget)
			if !ok {
				return nil, ErrMatchWorkLimit
			}
			if _, ok := seen[key]; ok {
				continue
			}
			if !spend(budget, 1) {
				return nil, ErrMatchWorkLimit
			}
			if !memoryPut(budget, seen, key, struct{}{}) {
				return nil, matchMemoryError(budget)
			}
			copied, ok := memoryMap[string, Value[ID]](budget, len(binding))
			if !ok {
				return nil, matchMemoryError(budget)
			}
			for name, value := range binding {
				if !spend(budget, 1) {
					return nil, ErrMatchWorkLimit
				}
				if !memoryPut(budget, copied, name, value) {
					return nil, matchMemoryError(budget)
				}
			}
			results, ok = memoryAppend(budget, results, RuleMatch[ID]{ClassID: anchorID, Binding: copied})
			if !ok {
				return nil, matchMemoryError(budget)
			}
		}
	}
	if len(results) > 1 {
		levels := bits.Len(uint(len(results)))
		if len(results) > math.MaxInt/(2*levels) || !budget.TrySpend(2*len(results)*levels) {
			return nil, ErrMatchWorkLimit
		}
	}
	keys, ok := memorySlice[string](budget, len(results), len(results))
	if !ok {
		return nil, matchMemoryError(budget)
	}
	for i, result := range results {
		key, ok := anchoredBindingKey(result.ClassID, result.Binding, budget)
		if !ok {
			return nil, ErrMatchWorkLimit
		}
		keys[i] = key
	}
	resultOrder, ok := memorySlice[int](budget, len(results), len(results))
	if !ok {
		return nil, matchMemoryError(budget)
	}
	for i := range resultOrder {
		resultOrder[i] = i
	}
	slices.SortFunc(resultOrder, func(i, j int) int { return cmp.Compare(keys[i], keys[j]) })
	sorted, ok := memorySlice[RuleMatch[ID]](budget, len(results), len(results))
	if !ok {
		return nil, matchMemoryError(budget)
	}
	for i, index := range resultOrder {
		sorted[i] = results[index]
	}
	return sorted, nil
}

func uniformMatchSubst[ID ~uint32](matches []Match[ID], budget *WorkBudget) (valid, complete bool) {
	if len(matches) < 2 {
		return true, true
	}
	return uniformNameSchema(len(matches), func(i int) []string {
		names, ok := memorySlice[string](budget, 0, len(matches[i].Subst))
		if !ok {
			return nil
		}
		for name := range matches[i].Subst {
			names = append(names, name)
		}
		return names
	}, budget)
}

func uniformBindingSchema[ID ~uint32](bindings []Binding[ID], budget *WorkBudget) (valid, complete bool) {
	return uniformNameSchema(len(bindings), func(i int) []string {
		names, ok := memorySlice[string](budget, 0, len(bindings[i]))
		if !ok {
			return nil
		}
		for name := range bindings[i] {
			names = append(names, name)
		}
		return names
	}, budget)
}

func uniformNameSchema(n int, namesAt func(int) []string, budget *WorkBudget) (valid, complete bool) {
	if n < 2 {
		return true, true
	}
	first := namesAt(0)
	for i := 1; i < n; i++ {
		names := namesAt(i)
		if !spend(budget, len(first)) || !spend(budget, len(names)) {
			return false, false
		}
		if len(names) != len(first) {
			return false, true
		}
		set, ok := memoryMap[string, struct{}](budget, len(names))
		if !ok {
			return false, false
		}
		for _, name := range names {
			if !memoryPut(budget, set, name, struct{}{}) {
				return false, false
			}
		}
		for _, name := range first {
			if _, ok := set[name]; !ok {
				return false, true
			}
		}
	}
	return true, true
}

// JoinMatchesDeltas joins structural matches with bindings produced by
// JoinDeltas. Each delta identifies one relation atom changed since the
// corresponding cursor. No structural matches are returned when no delta
// bindings are available.
func JoinMatchesDeltas[ID ~uint32](
	anchors []Match[ID],
	structuralConditions [][]Match[ID],
	atoms []Atom[ID],
	deltas []DeltaAtom[ID],
	canonicalize func(ID) ID,
	budget *WorkBudget,
) ([]RuleMatch[ID], error) {
	if budget == nil || canonicalize == nil {
		return nil, ErrInvalidJoin
	}
	bindings, complete, err := JoinDeltas(atoms, deltas, canonicalize, budget)
	if err != nil {
		return nil, err
	}
	if !complete {
		return nil, ErrMatchWorkLimit
	}
	if len(bindings) == 0 {
		return nil, nil
	}
	return JoinMatchesWithBindings(anchors, structuralConditions, bindings, canonicalize, budget)
}
