package egraphcore

import (
	"fmt"
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
	binding := make(Binding[ID], len(subst))
	for name, id := range subst {
		if !spend(budget, 1) {
			return nil, false
		}
		if name == "" {
			return nil, false
		}
		binding[name] = Class(canonicalize(id))
	}
	return binding, true
}

func sharedNames[ID ~uint32](left, right []Binding[ID], budget *WorkBudget) ([]string, bool) {
	if len(left) == 0 || len(right) == 0 {
		return []string{}, true
	}
	shared := make(map[string]struct{})
	for name := range left[0] {
		if !spend(budget, 1) {
			return nil, false
		}
		if _, ok := right[0][name]; ok {
			shared[name] = struct{}{}
		}
	}
	names := make([]string, 0, len(shared))
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
	return fmt.Sprintf("%010d|%s", uint32(anchor), key), true
}
