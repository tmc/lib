package egraphcore

import "sort"

// OrderedRow is a copied row returned by OrderedTable. Values is a copy of the
// indexed tuple; Payload is copied by assignment and must not contain mutable
// data that callers change. Order is stable insertion order; Subsumed rows
// remain inspectable.
type OrderedRow[ID ~uint32, Row any] struct {
	Key      string
	Values   []Value[ID]
	Payload  Row
	Order    uint64
	Subsumed bool
}

type orderedRow[ID ~uint32, Row any] struct {
	values   []Value[ID]
	payload  Row
	order    uint64
	subsumed bool
}

// OrderedTable owns keyed rows, value indexes, insertion order, and
// subsumption state. It supports backend-specific schemas and canonical keys;
// callers validate values and choose keys before mutation. It is not
// concurrency-safe.
type OrderedTable[ID ~uint32, Row any] struct {
	rows      map[string]orderedRow[ID, Row]
	index     *ValueIndex[ID]
	width     int
	nextOrder uint64
	nextFresh uint64
	cloneRow  func(Row) Row
}

// NewOrderedTable creates an empty table indexed by columnCount values. The
// zero value is also usable as a table with no indexed columns.
func NewOrderedTable[ID ~uint32, Row any](columnCount int) *OrderedTable[ID, Row] {
	return NewOrderedTableWithClone[ID, Row](columnCount, nil)
}

// NewOrderedTableWithClone creates a table whose row payloads are copied with
// clone when inserted, returned, and cloned. A nil clone treats payloads as
// immutable values copied by assignment.
func NewOrderedTableWithClone[ID ~uint32, Row any](columnCount int, clone func(Row) Row) *OrderedTable[ID, Row] {
	if columnCount < 0 {
		columnCount = 0
	}
	return &OrderedTable[ID, Row]{rows: make(map[string]orderedRow[ID, Row]), index: NewValueIndex[ID](columnCount), width: columnCount, cloneRow: clone}
}

// Len returns the number of rows, including subsumed rows.
func (t *OrderedTable[ID, Row]) Len() int {
	if t == nil {
		return 0
	}
	return len(t.rows)
}

// Get returns one copied row.
func (t *OrderedTable[ID, Row]) Get(key string) (OrderedRow[ID, Row], bool) {
	if t == nil {
		return OrderedRow[ID, Row]{}, false
	}
	r, ok := t.rows[key]
	return OrderedRow[ID, Row]{Key: key, Values: append([]Value[ID](nil), r.values...), Payload: t.clonePayload(r.payload), Order: r.order, Subsumed: r.subsumed}, ok
}

// Put inserts or replaces a row. New rows receive the next stable order;
// replacements retain their existing order and subsumption state.
func (t *OrderedTable[ID, Row]) Put(key string, values []Value[ID], payload Row) (inserted bool, ok bool) {
	if t == nil || len(values) != t.indexWidth() {
		return false, false
	}
	t.initialize()
	if current, exists := t.rows[key]; exists {
		t.index.Remove(current.values, key)
		current.values = append([]Value[ID](nil), values...)
		current.payload = t.clonePayload(payload)
		t.rows[key] = current
		t.index.Add(current.values, key)
		return false, true
	}
	order := t.nextOrder
	t.nextOrder++
	row := orderedRow[ID, Row]{values: append([]Value[ID](nil), values...), payload: t.clonePayload(payload), order: order}
	t.rows[key] = row
	t.index.Add(row.values, key)
	return true, true
}

// PutAt inserts or replaces a row with an explicit order. It is used when
// folding rows whose keys collide after backend canonicalization.
func (t *OrderedTable[ID, Row]) PutAt(key string, values []Value[ID], payload Row, order uint64, subsumed bool) bool {
	if t == nil || len(values) != t.indexWidth() {
		return false
	}
	t.initialize()
	if current, exists := t.rows[key]; exists {
		t.index.Remove(current.values, key)
	}
	t.rows[key] = orderedRow[ID, Row]{values: append([]Value[ID](nil), values...), payload: t.clonePayload(payload), order: order, subsumed: subsumed}
	t.index.Add(values, key)
	if order >= t.nextOrder {
		t.nextOrder = order + 1
	}
	return true
}

func (t *OrderedTable[ID, Row]) indexWidth() int { return t.width }

func (t *OrderedTable[ID, Row]) initialize() {
	if t.rows == nil {
		t.rows = make(map[string]orderedRow[ID, Row])
	}
	if t.index == nil {
		t.index = NewValueIndex[ID](t.width)
	}
}

func (t *OrderedTable[ID, Row]) clonePayload(row Row) Row {
	if t.cloneRow != nil {
		return t.cloneRow(row)
	}
	return row
}

// Delete removes a row and all of its index entries.
func (t *OrderedTable[ID, Row]) Delete(key string) bool {
	if t == nil {
		return false
	}
	r, ok := t.rows[key]
	if !ok {
		return false
	}
	t.index.Remove(r.values, key)
	delete(t.rows, key)
	return true
}

// MarkSubsumed marks an existing row hidden from matching. It returns false
// when the row is absent or already subsumed.
func (t *OrderedTable[ID, Row]) MarkSubsumed(key string) bool {
	if t == nil {
		return false
	}
	r, ok := t.rows[key]
	if !ok || r.subsumed {
		return false
	}
	r.subsumed = true
	t.rows[key] = r
	return true
}

// Contains reports whether key is indexed for column and value.
func (t *OrderedTable[ID, Row]) Contains(column int, value Value[ID], key string) bool {
	return t != nil && t.index.Contains(column, value, key)
}

// Range visits keys indexed for column and value. Iteration order is
// unspecified. It stops and returns false when visit returns false.
func (t *OrderedTable[ID, Row]) Range(column int, value Value[ID], visit func(string) bool) bool {
	return t != nil && t.index.Range(column, value, visit)
}

// Rows returns copied rows in insertion order, with key as a tie breaker.
func (t *OrderedTable[ID, Row]) Rows() []OrderedRow[ID, Row] {
	if t == nil {
		return nil
	}
	out := make([]OrderedRow[ID, Row], 0, len(t.rows))
	for key, row := range t.rows {
		out = append(out, OrderedRow[ID, Row]{Key: key, Values: append([]Value[ID](nil), row.values...), Payload: t.clonePayload(row.payload), Order: row.order, Subsumed: row.subsumed})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Order != out[j].Order {
			return out[i].Order < out[j].Order
		}
		return out[i].Key < out[j].Key
	})
	return out
}

// Counters returns the next stable row order and fresh-value ordinal.
func (t *OrderedTable[ID, Row]) Counters() (order, fresh uint64) {
	if t == nil {
		return 0, 0
	}
	return t.nextOrder, t.nextFresh
}

// SetCounters advances table counters without moving them backwards.
func (t *OrderedTable[ID, Row]) SetCounters(order, fresh uint64) {
	if t == nil {
		return
	}
	if order > t.nextOrder {
		t.nextOrder = order
	}
	if fresh > t.nextFresh {
		t.nextFresh = fresh
	}
}

// Clone returns independent rows and indexes.
func (t *OrderedTable[ID, Row]) Clone() *OrderedTable[ID, Row] {
	if t == nil {
		return nil
	}
	out := NewOrderedTableWithClone[ID, Row](t.indexWidth(), t.cloneRow)
	out.nextOrder, out.nextFresh = t.nextOrder, t.nextFresh
	if t.index == nil {
		out.index = NewValueIndex[ID](t.width)
	}
	for key, row := range t.rows {
		copy := orderedRow[ID, Row]{values: append([]Value[ID](nil), row.values...), payload: t.clonePayload(row.payload), order: row.order, subsumed: row.subsumed}
		out.rows[key] = copy
		out.index.Add(copy.values, key)
	}
	return out
}

// ReplaceFrom atomically replaces row state from a table with equal index
// width. The receiver retains its identity for outstanding references.
func (t *OrderedTable[ID, Row]) ReplaceFrom(other *OrderedTable[ID, Row]) bool {
	if t == nil || other == nil || t.indexWidth() != other.indexWidth() {
		return false
	}
	copy := other.Clone()
	*t = *copy
	return true
}
