package egraphcore

import "math"

// InsertWithBudget reserves the store's growth before publishing a new class.
// newClass owns admission of its class storage. The caller retains budget's
// scope through the last use of the store and any construction scratch.
func (s *Store[ID, Class]) InsertWithBudget(key string, newClass func(ID) (Class, error), budget *WorkBudget) (ID, error) {
	var zero ID
	if s == nil || newClass == nil || uint64(len(s.parent)) > math.MaxUint32 {
		return zero, ErrMemorySize
	}
	id := ID(len(s.parent))
	parent, ok := memoryAppend(budget, s.parent, id)
	if !ok {
		return zero, budget.MemoryError()
	}
	sizes, ok := memoryAppend(budget, s.size, uint64(1))
	if !ok {
		return zero, budget.MemoryError()
	}
	classes := s.classes
	if classes == nil {
		classes, ok = memoryMap[ID, Class](budget, 0)
		if !ok {
			return zero, budget.MemoryError()
		}
	}
	memo := s.memo
	if memo == nil {
		memo, ok = memoryMap[string, []ID](budget, 0)
		if !ok {
			return zero, budget.MemoryError()
		}
	}
	bucket, ok := memoryAppend(budget, memo[key], id)
	if !ok {
		return zero, budget.MemoryError()
	}
	// Reserve entries without publishing them. No fallible work follows commit.
	if !reserveEntry(budget, classes, id) || !reserveEntry(budget, memo, key) {
		return zero, budget.MemoryError()
	}
	class, err := newClass(id)
	if err != nil {
		return zero, err
	}
	s.parent, s.size, s.classes, s.memo = parent, sizes, classes, memo
	s.classes[id], s.memo[key] = class, bucket
	return id, nil
}

func reserveEntry[K comparable, V any](budget *WorkBudget, values map[K]V, key K) bool {
	if _, exists := values[key]; exists {
		return true
	}
	return memorySlots[struct {
		Key   K
		Value V
	}](budget, 1)
}

// AddMemoCandidateIfWithBudget is AddMemoCandidateIf with storage admission.
// same must return borrowed results without allocating unreserved storage.
func (s *Store[ID, Class]) AddMemoCandidateIfWithBudget(key string, id ID, same func(ID, ID) bool, budget *WorkBudget) (bool, error) {
	if s == nil {
		return false, ErrMemorySize
	}
	if same != nil {
		for _, candidate := range s.memo[key] {
			if same(candidate, id) {
				return false, nil
			}
		}
	}
	memo := s.memo
	if memo == nil {
		var ok bool
		memo, ok = memoryMap[string, []ID](budget, 0)
		if !ok {
			return false, budget.MemoryError()
		}
	}
	bucket, ok := memoryAppend(budget, memo[key], id)
	if !ok || !reserveEntry(budget, memo, key) {
		return false, budget.MemoryError()
	}
	s.memo = memo
	memo[key] = bucket
	return true, nil
}

func (r *Relation[ID]) prepareWithBudget(values []Value[ID], canonicalize func(ID) ID, budget *WorkBudget) ([]Value[ID], string, error) {
	if budget.MemoryScope() == nil {
		return r.prepare(values, canonicalize)
	}
	if r == nil || len(values) != len(r.schema.Columns) {
		return nil, "", errInvalidSchema
	}
	row, ok := memorySlice[Value[ID]](budget, len(values), len(values))
	if !ok {
		return nil, "", budget.MemoryError()
	}
	for i, value := range values {
		value, err := r.normalize(value, canonicalize)
		if err != nil {
			return nil, "", err
		}
		if value.Kind != r.schema.Columns[i].Kind {
			return nil, "", errInvalidSchema
		}
		row[i] = value
	}
	key, ok := tupleKeyWithBudget(row, budget)
	if !ok {
		if err := budget.MemoryError(); err != nil {
			return nil, "", err
		}
		return nil, "", ErrMemorySize
	}
	return row, key, nil
}

// Scoped fact mutation operates on unpublished staging, so an index's partial
// construction on refusal cannot corrupt a registered relation.
func (r *Relation[ID]) addPreparedWithBudget(row []Value[ID], key string, budget *WorkBudget) (bool, error) {
	if _, exists := r.rows[key]; exists {
		return false, nil
	}
	if r.version == ^uint64(0) {
		if err := r.resetAdditionsWithBudget(budget); err != nil {
			return false, err
		}
	}
	if !memoryPut(budget, r.rows, key, row) || !r.index.addWithBudget(row, key, budget) {
		return false, budget.MemoryError()
	}
	copyRow, ok := memorySlice[Value[ID]](budget, len(row), len(row))
	if !ok {
		return false, budget.MemoryError()
	}
	copy(copyRow, row)
	added, ok := memoryAppend(budget, r.added, relationAddition[ID]{version: r.version + 1, row: copyRow})
	if !ok {
		return false, budget.MemoryError()
	}
	r.version++
	r.added = added
	return true, nil
}

func (r *Relation[ID]) resetAdditionsWithBudget(budget *WorkBudget) error {
	if budget.MemoryScope() == nil {
		r.resetAdditions()
		return nil
	}
	epoch, ok := memoryObject(budget, relationEpoch{})
	if !ok {
		return budget.MemoryError()
	}
	r.epoch, r.ancestors, r.version, r.floor, r.added = epoch, nil, 0, 0, nil
	return nil
}

func (r *Relation[ID]) applyPreparedWithBudget(kind FactActionKind, row []Value[ID], key string, budget *WorkBudget) (bool, error) {
	switch kind {
	case FactInsert:
		return r.addPreparedWithBudget(row, key, budget)
	case FactDelete:
		if _, exists := r.rows[key]; !exists {
			return false, nil
		}
		if err := r.resetAdditionsWithBudget(budget); err != nil {
			return false, err
		}
		r.index.Remove(r.rows[key], key)
		delete(r.rows, key)
		delete(r.subsumed, key)
		return true, nil
	case FactSubsume:
		if _, hidden := r.subsumed[key]; hidden {
			return false, nil
		}
		if _, err := r.addPreparedWithBudget(row, key, budget); err != nil {
			return false, err
		}
		if !memoryPut(budget, r.subsumed, key, struct{}{}) {
			return false, budget.MemoryError()
		}
		if err := r.resetAdditionsWithBudget(budget); err != nil {
			return false, err
		}
		return true, nil
	}
	return false, ErrInvalidFactAction
}

// RecanonicalizeWithBudget stages canonical relations with allocation-site
// admission, preserving registered pointers and invalidating old delta cursors.
// A nil memory scope retains ordinary Recanonicalize behavior.
func (s *FactStore[ID]) RecanonicalizeWithBudget(canonicalize func(ID) ID, budget *WorkBudget) error {
	if budget.MemoryScope() == nil {
		return s.Recanonicalize(canonicalize)
	}
	if s == nil {
		return errInvalidSchema
	}
	if canonicalize == nil {
		return ErrMissingCanonicalizer
	}
	staged, err := s.CanonicalCopyWithBudget(canonicalize, budget)
	if err != nil {
		return err
	}
	for name, relation := range s.relations {
		*relation = *staged.relations[name]
	}
	return nil
}
