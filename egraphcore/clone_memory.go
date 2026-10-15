package egraphcore

// CloneWithBudget copies the store's collections, reserving each object,
// backing array and map entry before allocation. The caller keeps budget's
// MemoryScope open until the copy is no longer used. Strings remain borrowed.
// cloneClass must copy backend-owned mutable data using the supplied budget;
// a nil callback borrows class values, as Clone does.
func (s *Store[ID, Class]) CloneWithBudget(cloneClass func(Class, *WorkBudget) (Class, error), budget *WorkBudget) (*Store[ID, Class], error) {
	if s == nil {
		return nil, nil
	}
	parent, ok := memorySlice[ID](budget, len(s.parent), len(s.parent))
	if !ok {
		return nil, budget.MemoryError()
	}
	copy(parent, s.parent)
	if len(parent) == 0 {
		parent = nil
	}
	sizes, ok := memorySlice[uint64](budget, len(s.size), len(s.size))
	if !ok {
		return nil, budget.MemoryError()
	}
	copy(sizes, s.size)
	if len(sizes) == 0 {
		sizes = nil
	}
	classes, ok := memoryMap[ID, Class](budget, len(s.classes))
	if !ok {
		return nil, budget.MemoryError()
	}
	memo, ok := memoryMap[string, []ID](budget, len(s.memo))
	if !ok {
		return nil, budget.MemoryError()
	}
	out, ok := memoryObject(budget, Store[ID, Class]{parent: parent, size: sizes, classes: classes, memo: memo})
	if !ok {
		return nil, budget.MemoryError()
	}
	for id, class := range s.classes {
		if cloneClass != nil {
			var err error
			class, err = cloneClass(class, budget)
			if err != nil {
				return nil, err
			}
		}
		if !memoryPut(budget, classes, id, class) {
			return nil, budget.MemoryError()
		}
	}
	for key, source := range s.memo {
		ids, ok := memorySlice[ID](budget, len(source), len(source))
		if !ok {
			return nil, budget.MemoryError()
		}
		copy(ids, source)
		if len(ids) == 0 {
			ids = nil
		}
		if !memoryPut(budget, memo, key, ids) {
			return nil, budget.MemoryError()
		}
	}
	return out, nil
}

// CloneWithBudget copies schemas, facts, indexes and delta history, reserving
// storage before allocation. The caller owns budget's MemoryScope and keeps it
// open through the copy's last use. Strings and immutable ancestor epoch tokens
// remain borrowed. Unlike a canonical copy, this preserves cursor lineage.
func (s *FactStore[ID]) CloneWithBudget(budget *WorkBudget) (*FactStore[ID], error) {
	if s == nil {
		return nil, nil
	}
	relations, ok := memoryMap[string, *Relation[ID]](budget, len(s.relations))
	if !ok {
		return nil, budget.MemoryError()
	}
	out, ok := memoryObject(budget, FactStore[ID]{relations: relations})
	if !ok {
		return nil, budget.MemoryError()
	}
	for name, relation := range s.relations {
		cloned, err := relation.cloneWithBudget(budget)
		if err != nil {
			return nil, err
		}
		if !memoryPut(budget, relations, name, cloned) {
			return nil, budget.MemoryError()
		}
	}
	return out, nil
}

func (r *Relation[ID]) cloneWithBudget(budget *WorkBudget) (*Relation[ID], error) {
	if r == nil {
		return nil, nil
	}
	columns, ok := memorySlice[Column](budget, len(r.schema.Columns), len(r.schema.Columns))
	if !ok {
		return nil, budget.MemoryError()
	}
	copy(columns, r.schema.Columns)
	if len(columns) == 0 {
		columns = nil
	}
	rows, ok := memoryMap[string, []Value[ID]](budget, len(r.rows))
	if !ok {
		return nil, budget.MemoryError()
	}
	subsumed, ok := memoryMap[string, struct{}](budget, len(r.subsumed))
	if !ok {
		return nil, budget.MemoryError()
	}
	epoch, ok := memoryObject(budget, relationEpoch{})
	if !ok {
		return nil, budget.MemoryError()
	}
	// Allocate the final bounded history directly. Clone's append/trim
	// intermediates are unnecessary, and the parent epoch must remain last.
	count := len(r.ancestors)
	if count >= relationCursorLineageLimit {
		count = relationCursorLineageLimit - 1
	}
	ancestors, ok := memorySlice[relationAncestor](budget, count+1, count+1)
	if !ok {
		return nil, budget.MemoryError()
	}
	copy(ancestors, r.ancestors[len(r.ancestors)-count:])
	ancestors[count] = relationAncestor{epoch: r.epoch, maxVersion: r.version}
	added, ok := memorySlice[relationAddition[ID]](budget, len(r.added), len(r.added))
	if !ok {
		return nil, budget.MemoryError()
	}
	out, ok := memoryObject(budget, Relation[ID]{schema: Schema{Name: r.schema.Name, Columns: columns}, rows: rows, subsumed: subsumed, epoch: epoch, ancestors: ancestors, version: r.version, floor: r.floor, added: added})
	if !ok {
		return nil, budget.MemoryError()
	}
	for key, source := range r.rows {
		row, ok := memorySlice[Value[ID]](budget, len(source), len(source))
		if !ok {
			return nil, budget.MemoryError()
		}
		copy(row, source)
		if len(row) == 0 {
			row = nil
		}
		if !memoryPut(budget, rows, key, row) {
			return nil, budget.MemoryError()
		}
	}
	for key := range r.subsumed {
		if !memoryPut(budget, subsumed, key, struct{}{}) {
			return nil, budget.MemoryError()
		}
	}
	for i, source := range r.added {
		row, ok := memorySlice[Value[ID]](budget, len(source.row), len(source.row))
		if !ok {
			return nil, budget.MemoryError()
		}
		copy(row, source.row)
		if len(row) == 0 {
			row = nil
		}
		added[i] = relationAddition[ID]{version: source.version, row: row}
	}
	index, err := r.index.cloneWithBudget(budget)
	if err != nil {
		return nil, err
	}
	out.index = index
	return out, nil
}

func (index *ValueIndex[ID]) cloneWithBudget(budget *WorkBudget) (*ValueIndex[ID], error) {
	if index == nil {
		return nil, nil
	}
	out, ok := newValueIndexWithBudget[ID](len(index.columns), budget)
	if !ok {
		return nil, budget.MemoryError()
	}
	for column, values := range index.columns {
		// The ordinary clone leaves an empty column nil, even if the source
		// held an empty map. Preserve that observable internal invariant.
		if len(values) == 0 {
			continue
		}
		copied, ok := memoryMap[Value[ID], map[string]struct{}](budget, len(values))
		if !ok {
			return nil, budget.MemoryError()
		}
		out.columns[column] = copied
		for value, keys := range values {
			copyKeys, ok := memoryMap[string, struct{}](budget, len(keys))
			if !ok {
				return nil, budget.MemoryError()
			}
			for key := range keys {
				if !memoryPut(budget, copyKeys, key, struct{}{}) {
					return nil, budget.MemoryError()
				}
			}
			if !memoryPut(budget, copied, value, copyKeys) {
				return nil, budget.MemoryError()
			}
		}
	}
	return out, nil
}
