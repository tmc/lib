package egraphcore

// ValueIndex maps each value in a row column to the keys of rows containing
// that value. It is not safe for concurrent use. Rows remain owned by the
// caller; the index stores only their comparable values and row keys.
type ValueIndex[ID ~uint32] struct {
	columns []map[Value[ID]]map[string]struct{}
}

// NewValueIndex creates an index with columnCount columns.
func NewValueIndex[ID ~uint32](columnCount int) *ValueIndex[ID] {
	if columnCount < 0 {
		columnCount = 0
	}
	return &ValueIndex[ID]{columns: make([]map[Value[ID]]map[string]struct{}, columnCount)}
}

func newValueIndexWithBudget[ID ~uint32](columnCount int, budget *WorkBudget) (*ValueIndex[ID], bool) {
	columns, ok := memorySlice[map[Value[ID]]map[string]struct{}](budget, columnCount, columnCount)
	if !ok {
		return nil, false
	}
	return memoryObject(budget, ValueIndex[ID]{columns: columns})
}

func (index *ValueIndex[ID]) addWithBudget(row []Value[ID], key string, budget *WorkBudget) bool {
	if index == nil || len(row) != len(index.columns) {
		return false
	}
	for column, value := range row {
		if index.columns[column] == nil {
			values, ok := memoryMap[Value[ID], map[string]struct{}](budget, 0)
			if !ok {
				return false
			}
			index.columns[column] = values
		}
		keys := index.columns[column][value]
		if keys == nil {
			var ok bool
			keys, ok = memoryMap[string, struct{}](budget, 0)
			if !ok || !memoryPut(budget, index.columns[column], value, keys) {
				return false
			}
		}
		if !memoryPut(budget, keys, key, struct{}{}) {
			return false
		}
	}
	return true
}

// Add records key in the index for every value in row. It returns false when
// the row has the wrong number of columns or the index is nil.
func (index *ValueIndex[ID]) Add(row []Value[ID], key string) bool {
	if index == nil || len(row) != len(index.columns) {
		return false
	}
	for column, value := range row {
		if index.columns[column] == nil {
			index.columns[column] = make(map[Value[ID]]map[string]struct{})
		}
		keys := index.columns[column][value]
		if keys == nil {
			keys = make(map[string]struct{})
			index.columns[column][value] = keys
		}
		keys[key] = struct{}{}
	}
	return true
}

// Remove deletes key from every value index for row.
func (index *ValueIndex[ID]) Remove(row []Value[ID], key string) bool {
	if index == nil || len(row) != len(index.columns) {
		return false
	}
	for column, value := range row {
		keys := index.columns[column][value]
		delete(keys, key)
		if len(keys) == 0 {
			delete(index.columns[column], value)
		}
	}
	return true
}

// Count reports the number of row keys indexed for column and value.
func (index *ValueIndex[ID]) Count(column int, value Value[ID]) int {
	if index == nil || column < 0 || column >= len(index.columns) {
		return 0
	}
	return len(index.columns[column][value])
}

// Contains reports whether key is indexed for column and value.
func (index *ValueIndex[ID]) Contains(column int, value Value[ID], key string) bool {
	if index == nil || column < 0 || column >= len(index.columns) {
		return false
	}
	_, ok := index.columns[column][value][key]
	return ok
}

// Range calls visit for each row key indexed for column and value. Iteration
// order is unspecified. It stops and returns false when visit returns false.
func (index *ValueIndex[ID]) Range(column int, value Value[ID], visit func(string) bool) bool {
	if index == nil || column < 0 || column >= len(index.columns) || visit == nil {
		return false
	}
	for key := range index.columns[column][value] {
		if !visit(key) {
			return false
		}
	}
	return true
}

// Clone returns an independent copy of the index.
func (index *ValueIndex[ID]) Clone() *ValueIndex[ID] {
	if index == nil {
		return nil
	}
	clone := NewValueIndex[ID](len(index.columns))
	for column, values := range index.columns {
		for value, keys := range values {
			copyKeys := make(map[string]struct{}, len(keys))
			for key := range keys {
				copyKeys[key] = struct{}{}
			}
			if clone.columns[column] == nil {
				clone.columns[column] = make(map[Value[ID]]map[string]struct{})
			}
			clone.columns[column][value] = copyKeys
		}
	}
	return clone
}
