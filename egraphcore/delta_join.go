package egraphcore

import (
	"math"
	"math/bits"
	"sort"
)

// DeltaAtom selects one relation atom whose additions should seed an
// incremental join. Every atom in a rule body that has new rows should be
// included; JoinDeltas evaluates one join per selected atom and deduplicates
// matches that contain more than one new row.
type DeltaAtom[ID ~uint32] struct {
	Index  int
	Cursor RelationCursor
}

// JoinDeltas evaluates a conjunction using only newly added rows for each
// selected delta atom and the full current relations for all other atoms.
// Deletions, subsumption, and recanonicalization invalidate relation cursors;
// in that case the caller must recompute the rule from the full relations.
//
// Results are canonicalized, deduplicated, and returned in deterministic
// order. An empty delta set produces no bindings. Work exhaustion returns no
// partial result and complete=false. A nil budget is unlimited.
func JoinDeltas[ID ~uint32](
	atoms []Atom[ID],
	deltas []DeltaAtom[ID],
	canonicalize func(ID) ID,
	budget *WorkBudget,
) (results []Binding[ID], complete bool, err error) {
	if canonicalize == nil || len(atoms) == 0 && len(deltas) != 0 {
		return nil, false, ErrInvalidJoin
	}
	if len(deltas) == 0 {
		return nil, true, nil
	}
	seenAtoms := make(map[int]struct{}, len(deltas))
	for _, delta := range deltas {
		if delta.Index < 0 || delta.Index >= len(atoms) || atoms[delta.Index].Relation == nil {
			return nil, false, ErrInvalidJoin
		}
		if _, exists := seenAtoms[delta.Index]; exists {
			return nil, false, ErrInvalidJoin
		}
		seenAtoms[delta.Index] = struct{}{}
		if _, valid := atoms[delta.Index].Relation.cursorVersion(delta.Cursor); !valid {
			return nil, false, ErrRelationReset
		}
	}
	if !spend(budget, len(deltas)) {
		return nil, false, nil
	}

	seenBindings := make(map[string]Binding[ID])
	for _, delta := range deltas {
		rows, done, err := atoms[delta.Index].Relation.AddedSinceWithBudget(delta.Cursor, budget)
		if err != nil {
			return nil, false, err
		}
		if !done {
			return nil, false, nil
		}
		if len(rows) == 0 {
			continue
		}
		if !spend(budget, len(atoms)) {
			return nil, false, nil
		}
		schema := atoms[delta.Index].Relation.Schema()
		deltaRelation, err := NewRelation[ID](schema)
		if err != nil {
			return nil, false, err
		}
		for _, row := range rows {
			if !spend(budget, len(row)+1) {
				return nil, false, nil
			}
			if _, err := deltaRelation.Add(row, canonicalize); err != nil {
				return nil, false, err
			}
		}
		selected := append([]Atom[ID](nil), atoms...)
		selected[delta.Index].Relation = deltaRelation
		bindings, done, joinErr := Join(selected, canonicalize, budget)
		if joinErr != nil {
			return nil, false, joinErr
		}
		if !done {
			return nil, false, nil
		}
		for _, binding := range bindings {
			key, ok := bindingKey(binding, budget)
			if !ok {
				return nil, false, nil
			}
			if _, exists := seenBindings[key]; exists {
				continue
			}
			if !spend(budget, 1) {
				return nil, false, nil
			}
			seenBindings[key] = binding
		}
	}

	var keys []string
	for key := range seenBindings {
		if !spend(budget, 1) {
			return nil, false, nil
		}
		keys = append(keys, key)
	}
	if len(keys) > 1 {
		levels := bits.Len(uint(len(keys)))
		if len(keys) > math.MaxInt/(2*levels) || !spend(budget, 2*len(keys)*levels) {
			return nil, false, nil
		}
		sort.Strings(keys)
	}
	results = make([]Binding[ID], len(keys))
	for i, key := range keys {
		results[i] = seenBindings[key]
	}
	return results, true, nil
}
