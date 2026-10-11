package egraphcore

import (
	"errors"
	"math"
)

// FactActionKind identifies a mutation to a registered relation.
type FactActionKind uint8

const (
	FactInsert FactActionKind = iota + 1
	FactDelete
	FactSubsume
)

// FactAction is a relation mutation whose terms are constants or variables.
// Wildcards are useful in queries but cannot identify an action tuple.
type FactAction[ID ~uint32] struct {
	Kind     FactActionKind
	Relation string
	Terms    []Term[ID]
}

// ResolvedFactAction is a typed relation mutation ready to apply.
type ResolvedFactAction[ID ~uint32] struct {
	Kind     FactActionKind
	Relation string
	Values   []Value[ID]
}

// ResolveFactActions checks and evaluates every action against one binding.
// It returns no partial plan on invalid input or work exhaustion. Class values
// are canonicalized while resolving. A nil budget means unlimited work.
func ResolveFactActions[ID ~uint32](store *FactStore[ID], actions []FactAction[ID], binding Binding[ID], canonicalize func(ID) ID, budget *WorkBudget) ([]ResolvedFactAction[ID], error) {
	resolved := make([]ResolvedFactAction[ID], len(actions))
	for i, action := range actions {
		if !spend(budget, 1) {
			return nil, ErrMatchWorkLimit
		}
		if action.Kind < FactInsert || action.Kind > FactSubsume || action.Relation == "" {
			return nil, ErrInvalidFactAction
		}
		relation := store.Get(action.Relation)
		if relation == nil {
			return nil, errInvalidSchema
		}
		schema := relation.Schema()
		if len(action.Terms) != len(schema.Columns) {
			return nil, errInvalidSchema
		}
		values := make([]Value[ID], len(action.Terms))
		for column, term := range action.Terms {
			if !spend(budget, 1) {
				return nil, ErrMatchWorkLimit
			}
			switch term.Kind {
			case VariableTerm:
				if term.Name == "" {
					return nil, ErrInvalidFactAction
				}
				value, ok := binding[term.Name]
				if !ok {
					return nil, ErrInvalidFactAction
				}
				values[column] = value
			case ConstantTerm:
				values[column] = term.Value
			default:
				return nil, ErrInvalidFactAction
			}
			if values[column].Kind != schema.Columns[column].Kind {
				return nil, errInvalidSchema
			}
		}
		canonical, _, err := relation.prepare(values, canonicalize)
		if err != nil {
			return nil, err
		}
		resolved[i] = ResolvedFactAction[ID]{Kind: action.Kind, Relation: action.Relation, Values: canonical}
	}
	return resolved, nil
}

// ApplyActions applies resolved relation mutations in order. It stages every
// touched relation on a clone, then commits all changed relations together.
// Existing relation pointers remain valid. It returns the number of relations
// whose final state changed. A nil budget means unlimited work.
func (s *FactStore[ID]) ApplyActions(actions []ResolvedFactAction[ID], canonicalize func(ID) ID, budget *WorkBudget) (int, error) {
	_, changedRelations, err := s.ApplyActionsDetailed(actions, canonicalize, budget)
	return changedRelations, err
}

// ApplyActionsDetailed applies resolved relation mutations in order. It stages
// multiple actions on relation clones and commits all changed relations
// together. A singleton action uses a prepared in-place update after all
// validation and budget checks have succeeded. The returned slice has one
// entry per action and reports whether that action changed its relation at the
// point where it was applied. The second result is the number of relations
// whose final state changed. On error, no mutation is published and the action
// results are nil. A nil budget means unlimited work.
func (s *FactStore[ID]) ApplyActionsDetailed(actions []ResolvedFactAction[ID], canonicalize func(ID) ID, budget *WorkBudget) ([]bool, int, error) {
	if s == nil {
		return nil, 0, errInvalidSchema
	}
	if len(actions) == 1 {
		return s.applySingleActionDetailed(actions[0], canonicalize, budget)
	}
	actionChanges := make([]bool, len(actions))
	staged := make(map[string]*Relation[ID])
	originals := make(map[string]*Relation[ID])
	for actionIndex, action := range actions {
		if !spend(budget, 1) {
			return nil, 0, ErrMatchWorkLimit
		}
		if action.Kind < FactInsert || action.Kind > FactSubsume || action.Relation == "" {
			return nil, 0, ErrInvalidFactAction
		}
		passes := 2
		if action.Kind == FactSubsume {
			// MarkSubsumed prepares the tuple, then prepares it again through
			// Add when the tuple is not already present.
			passes++
		}
		if len(action.Values) > math.MaxInt/passes || !spend(budget, len(action.Values)*passes) {
			return nil, 0, ErrMatchWorkLimit
		}
		relation := s.Get(action.Relation)
		if relation == nil {
			return nil, 0, errInvalidSchema
		}
		copy, ok := staged[action.Relation]
		if !ok {
			schema := relation.Schema()
			rows, columns := relation.Len(), len(schema.Columns)
			if columns == math.MaxInt || rows != 0 && columns+1 > (math.MaxInt-1)/rows {
				return nil, 0, ErrMatchWorkLimit
			}
			if !spend(budget, rows*(columns+1)+1) {
				return nil, 0, ErrMatchWorkLimit
			}
			copy = relation.Clone()
			if copy == nil {
				return nil, 0, errInvalidSchema
			}
			staged[action.Relation] = copy
			originals[action.Relation] = relation
		}
		if _, _, err := relation.prepare(action.Values, canonicalize); err != nil {
			return nil, 0, err
		}
		var changed bool
		var err error
		switch action.Kind {
		case FactInsert:
			changed, err = copy.Add(action.Values, canonicalize)
		case FactDelete:
			changed, err = copy.Delete(action.Values, canonicalize)
		case FactSubsume:
			changed, err = copy.MarkSubsumed(action.Values, canonicalize)
		}
		if err != nil {
			return nil, 0, err
		}
		actionChanges[actionIndex] = changed
	}
	changedRelations := make([]string, 0, len(staged))
	for name, copy := range staged {
		original := originals[name]
		rows := original.Len()
		if copy.Len() > rows {
			rows = copy.Len()
		}
		columns := len(original.Schema().Columns)
		if columns == math.MaxInt || rows != 0 && columns+1 > (math.MaxInt-1)/rows {
			return nil, 0, ErrMatchWorkLimit
		}
		if !spend(budget, rows*(columns+1)+1) {
			return nil, 0, ErrMatchWorkLimit
		}
		if relationEqual(original, copy) {
			continue
		}
		changedRelations = append(changedRelations, name)
	}
	if !spend(budget, len(changedRelations)) {
		return nil, 0, ErrMatchWorkLimit
	}
	for _, name := range changedRelations {
		*originals[name] = *staged[name]
	}
	return actionChanges, len(changedRelations), nil
}

// applySingleActionDetailed preserves the transactional error and budget
// behavior of ApplyActionsDetailed without cloning the relation. A single
// relation mutation can be validated and fully charged before its infallible
// prepared mutation is published in place.
func (s *FactStore[ID]) applySingleActionDetailed(action ResolvedFactAction[ID], canonicalize func(ID) ID, budget *WorkBudget) ([]bool, int, error) {
	if !spend(budget, 1) {
		return nil, 0, ErrMatchWorkLimit
	}
	if action.Kind < FactInsert || action.Kind > FactSubsume || action.Relation == "" {
		return nil, 0, ErrInvalidFactAction
	}
	passes := 2
	if action.Kind == FactSubsume {
		// MarkSubsumed prepares the tuple, then prepares it again through Add
		// when the tuple is not already present.
		passes++
	}
	if len(action.Values) > math.MaxInt/passes || !spend(budget, len(action.Values)*passes) {
		return nil, 0, ErrMatchWorkLimit
	}
	relation := s.Get(action.Relation)
	if relation == nil {
		return nil, 0, errInvalidSchema
	}
	schema := relation.Schema()
	rows, columns := relation.Len(), len(schema.Columns)
	if columns == math.MaxInt || rows != 0 && columns+1 > (math.MaxInt-1)/rows {
		return nil, 0, ErrMatchWorkLimit
	}
	if !spend(budget, rows*(columns+1)+1) {
		return nil, 0, ErrMatchWorkLimit
	}
	row, key, err := relation.prepare(action.Values, canonicalize)
	if err != nil {
		return nil, 0, err
	}
	_, exists := relation.rows[key]
	_, subsumed := relation.subsumed[key]
	changed := false
	finalRows := rows
	switch action.Kind {
	case FactInsert:
		changed = !exists
		if changed {
			if finalRows == math.MaxInt {
				return nil, 0, ErrMatchWorkLimit
			}
			finalRows++
		}
	case FactDelete:
		changed = exists
		if changed {
			finalRows--
		}
	case FactSubsume:
		changed = !subsumed
		if !exists {
			if finalRows == math.MaxInt {
				return nil, 0, ErrMatchWorkLimit
			}
			finalRows++
		}
	}
	if finalRows > rows {
		rows = finalRows
	}
	if rows != 0 && columns+1 > (math.MaxInt-1)/rows {
		return nil, 0, ErrMatchWorkLimit
	}
	if !spend(budget, rows*(columns+1)+1) {
		return nil, 0, ErrMatchWorkLimit
	}
	changedRelations := 0
	if changed {
		changedRelations = 1
	}
	if !spend(budget, changedRelations) {
		return nil, 0, ErrMatchWorkLimit
	}
	switch action.Kind {
	case FactInsert:
		relation.addPrepared(row, key)
	case FactDelete:
		relation.deletePrepared(key)
	case FactSubsume:
		relation.markSubsumedPrepared(row, key)
	}
	return []bool{changed}, changedRelations, nil
}

func relationEqual[ID ~uint32](a, b *Relation[ID]) bool {
	if a == nil || b == nil || !equalSchema(a.schema, b.schema) || len(a.rows) != len(b.rows) || len(a.subsumed) != len(b.subsumed) {
		return false
	}
	for key, row := range a.rows {
		other, ok := b.rows[key]
		if !ok || len(row) != len(other) {
			return false
		}
		for i := range row {
			if row[i] != other[i] {
				return false
			}
		}
	}
	for key := range a.subsumed {
		if _, ok := b.subsumed[key]; !ok {
			return false
		}
	}
	return true
}

// ErrInvalidFactAction reports a malformed action or missing variable.
var ErrInvalidFactAction = errors.New("egraphcore: invalid fact action")
