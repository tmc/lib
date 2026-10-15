package egraphcore

import (
	"math"
	"math/bits"
	"sort"
)

// RuleMatch is one complete rule match. ClassID is anchored to the root of
// the first structural pattern; Binding contains values from structural and
// relation conditions.
type RuleMatch[ID ~uint32] struct {
	ClassID ID
	Binding Binding[ID]
}

// JoinMatches combines structural matches and positive relation conditions.
// anchors are the matches for the first structural LHS pattern. Each entry in
// structuralConditions is an alternative set for one additional structural
// condition. relationConditions are conjoined and joined with the structural
// bindings. The returned ClassID always comes from anchors, never a condition.
// Class-valued bindings are canonicalized before comparison and output.
// Results are deterministic and deduplicated. On work exhaustion, no partial
// result is returned. With no anchors, this function returns no matches;
// relation-only rules must provide a separate anchor policy.
func JoinMatches[ID ~uint32](
	anchors []Match[ID],
	structuralConditions [][]Match[ID],
	relationConditions []Atom[ID],
	canonicalize func(ID) ID,
	budget *WorkBudget,
) ([]RuleMatch[ID], error) {
	if budget == nil || canonicalize == nil {
		return nil, ErrInvalidJoin
	}
	relationBindings, complete, err := Join(relationConditions, canonicalize, budget)
	if err != nil {
		return nil, err
	}
	if !complete {
		return nil, ErrMatchWorkLimit
	}
	if len(relationConditions) != 0 && len(relationBindings) == 0 {
		return nil, nil
	}
	return JoinMatchesWithBindings(anchors, structuralConditions, relationBindings, canonicalize, budget)
}

func matchBinding[ID ~uint32](subst map[string]ID, canonicalize func(ID) ID, budget *WorkBudget) (Binding[ID], bool) {
	binding, ok := memoryMap[string, Value[ID]](budget, len(subst))
	if !ok {
		return nil, false
	}
	for name, id := range subst {
		if !spend(budget, 1) {
			return nil, false
		}
		if name == "" {
			return nil, false
		}
		if !memoryPut(budget, binding, name, Class(canonicalize(id))) {
			return nil, false
		}
	}
	return binding, true
}

func sharedNames[ID ~uint32](left, right []Binding[ID], budget *WorkBudget) ([]string, bool) {
	if len(left) == 0 || len(right) == 0 {
		return []string{}, true
	}
	shared, ok := memoryMap[string, struct{}](budget, 0)
	if !ok {
		return nil, false
	}
	for name := range left[0] {
		if !spend(budget, 1) {
			return nil, false
		}
		if _, ok := right[0][name]; ok {
			if !memoryPut(budget, shared, name, struct{}{}) {
				return nil, false
			}
		}
	}
	names, ok := memorySlice[string](budget, 0, len(shared))
	if !ok {
		return nil, false
	}
	for name := range shared {
		if !spend(budget, 1) {
			return nil, false
		}
		names = append(names, name)
	}
	if len(names) > 1 {
		levels := bits.Len(uint(len(names)))
		if len(names) > math.MaxInt/(2*levels) || !budget.TrySpend(2*len(names)*levels) {
			return nil, false
		}
		sort.Strings(names)
	}
	return names, true
}

func anchoredBindingKey[ID ~uint32](anchor ID, binding Binding[ID], budget *WorkBudget) (string, bool) {
	if !spend(budget, 1) {
		return "", false
	}
	key, ok := bindingKey(binding, budget)
	if !ok {
		return "", false
	}
	if len(key) > math.MaxInt-11 {
		return "", false
	}
	data, ok := memorySlice[byte](budget, 11+len(key), 11+len(key))
	if !ok {
		return "", false
	}
	number := uint32(anchor)
	for i := 9; i >= 0; i-- {
		data[i] = '0' + byte(number%10)
		number /= 10
	}
	data[10] = '|'
	copy(data[11:], key)
	if !budget.ReserveMemory(int64(len(data))) {
		return "", false
	}
	return string(data), true
}
