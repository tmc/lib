package egraphcore

import "math"

// FactStore owns named typed relations. It is not safe for concurrent use.
// Relation schemas and fact values are backend-neutral; class canonicalization
// remains the backend's responsibility.
type FactStore[ID ~uint32] struct {
	relations map[string]*Relation[ID]
}

// NewFactStore creates an empty fact store.
func NewFactStore[ID ~uint32]() *FactStore[ID] {
	return &FactStore[ID]{relations: make(map[string]*Relation[ID])}
}

// Register adds schema to the store. Re-registering an identical schema is
// idempotent and returns the existing relation; a conflicting schema fails.
func (s *FactStore[ID]) Register(schema Schema) (*Relation[ID], error) {
	if s == nil {
		return nil, errInvalidSchema
	}
	if old := s.relations[schema.Name]; old != nil {
		if equalSchema(old.schema, schema) {
			return old, nil
		}
		return nil, errInvalidSchema
	}
	r, err := NewRelation[ID](schema)
	if err != nil {
		return nil, err
	}
	if s.relations == nil {
		s.relations = make(map[string]*Relation[ID])
	}
	s.relations[schema.Name] = r
	return r, nil
}

// Get returns the named relation, or nil when it is not registered.
// The returned relation is owned by the store and can be mutated directly.
func (s *FactStore[ID]) Get(name string) *Relation[ID] {
	if s == nil {
		return nil
	}
	return s.relations[name]
}

// TupleCount reports the number of stored facts across all relations,
// including facts marked subsumed. It saturates at the largest int on overflow.
func (s *FactStore[ID]) TupleCount() int {
	if s == nil {
		return 0
	}
	maxInt := int(^uint(0) >> 1)
	total := 0
	for _, relation := range s.relations {
		if relation == nil {
			continue
		}
		count := relation.Len()
		if count > maxInt-total {
			return maxInt
		}
		total += count
	}
	return total
}

// Add inserts a fact into a registered relation.
func (s *FactStore[ID]) Add(name string, values []Value[ID], canonicalize func(ID) ID) (bool, error) {
	r := s.Get(name)
	if r == nil {
		return false, errInvalidSchema
	}
	return r.Add(values, canonicalize)
}

// Delete removes a fact from a registered relation.
func (s *FactStore[ID]) Delete(name string, values []Value[ID], canonicalize func(ID) ID) (bool, error) {
	r := s.Get(name)
	if r == nil {
		return false, errInvalidSchema
	}
	return r.Delete(values, canonicalize)
}

// MarkSubsumed inserts values if absent, then marks the fact hidden from Query.
func (s *FactStore[ID]) MarkSubsumed(name string, values []Value[ID], canonicalize func(ID) ID) (bool, error) {
	r := s.Get(name)
	if r == nil {
		return false, errInvalidSchema
	}
	return r.MarkSubsumed(values, canonicalize)
}

// Recanonicalize atomically updates class-valued facts in all relations.
func (s *FactStore[ID]) Recanonicalize(canonicalize func(ID) ID) error {
	if s == nil {
		return errInvalidSchema
	}
	if canonicalize == nil {
		return ErrMissingCanonicalizer
	}
	next := make(map[string]*Relation[ID], len(s.relations))
	for name, relation := range s.relations {
		copy := relation.Clone()
		if copy == nil {
			return errInvalidSchema
		}
		if err := copy.Recanonicalize(canonicalize); err != nil {
			return err
		}
		next[name] = copy
	}
	// Commit only after every relation has been recanonicalized. Preserve the
	// registered relation pointers because callers may retain them after Get.
	for name, relation := range s.relations {
		*relation = *next[name]
	}
	return nil
}

// ReplaceFrom copies another store's relation state into s. Registered
// relation pointers remain valid. Both stores must have the same schemas. For
// each relation, insertion history is preserved only when other descends from
// s's current state; replacing it with unrelated or diverged state invalidates
// old cursors. A nil budget means unlimited work.
func (s *FactStore[ID]) ReplaceFrom(other *FactStore[ID], budget *WorkBudget) error {
	if s == nil || other == nil || len(s.relations) != len(other.relations) {
		return errInvalidSchema
	}
	if s == other {
		return nil
	}
	staged := make(map[string]*Relation[ID], len(s.relations))
	for name, current := range s.relations {
		next := other.relations[name]
		if current == nil || next == nil || !equalSchema(current.schema, next.schema) {
			return errInvalidSchema
		}
		rows, columns := next.Len(), len(next.schema.Columns)
		if columns == math.MaxInt || rows != 0 && columns+1 > (math.MaxInt-1)/rows {
			return ErrMatchWorkLimit
		}
		if !spend(budget, rows*(columns+1)+1) {
			return ErrMatchWorkLimit
		}
		copy := next.Clone()
		if copy == nil {
			return errInvalidSchema
		}
		currentCursor := RelationCursor{epoch: current.epoch, version: current.version}
		if _, descends := copy.cursorVersion(currentCursor); !descends {
			copy.resetAdditions()
		}
		staged[name] = copy
	}
	for name, relation := range s.relations {
		*relation = *staged[name]
	}
	return nil
}

// Clone returns an independent store with the same schemas, facts, indexes,
// and subsumed-fact state.
func (s *FactStore[ID]) Clone() *FactStore[ID] {
	if s == nil {
		return nil
	}
	out := NewFactStore[ID]()
	for name, relation := range s.relations {
		out.relations[name] = relation.Clone()
	}
	return out
}

func equalSchema(a, b Schema) bool {
	if a.Name != b.Name || len(a.Columns) != len(b.Columns) {
		return false
	}
	for i := range a.Columns {
		if a.Columns[i] != b.Columns[i] {
			return false
		}
	}
	return true
}
