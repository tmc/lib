package egraphcore

import (
	"encoding/binary"
	"errors"
	"math"
	"sort"
)

var errInvalidSchema = errors.New("egraphcore: invalid relation schema or tuple")

// ErrMissingCanonicalizer reports an operation that requires class
// canonicalization but received no canonicalizer.
var ErrMissingCanonicalizer = errors.New("egraphcore: missing class canonicalizer")

// ErrRelationReset reports that a relation cursor predates a mutation that
// invalidated incremental evaluation. The caller must recompute from the
// current relation and take a new snapshot.
var ErrRelationReset = errors.New("egraphcore: relation cursor invalidated; recompute required")

type relationEpoch struct{ marker byte }

type relationAncestor struct {
	epoch      *relationEpoch
	maxVersion uint64
}

const relationCursorLineageLimit = 16

// RelationCursor identifies a point in one relation's insertion history.
// Its fields are private; obtain cursors with Relation.Snapshot.
type RelationCursor struct {
	epoch   *relationEpoch
	version uint64
}

type relationAddition[ID ~uint32] struct {
	version uint64
	row     []Value[ID]
}

// ValueKind identifies the representation of a relation value.
type ValueKind uint8

const (
	ClassValue ValueKind = iota + 1
	Int64Value
	BoolValue
	Float64Value
	StringValue
	UnitValue
	BigIntValue
	BigRatValue
)

// Value is a comparable typed value suitable for a relation column. Float64
// values are represented by their IEEE bits, so NaNs and signed zero remain
// distinct facts.
type Value[ID ~uint32] struct {
	Kind   ValueKind
	Class  ID
	Int64  int64
	Bool   bool
	Float  uint64
	String string
}

func Class[ID ~uint32](id ID) Value[ID]   { return Value[ID]{Kind: ClassValue, Class: id} }
func Int64[ID ~uint32](v int64) Value[ID] { return Value[ID]{Kind: Int64Value, Int64: v} }
func Bool[ID ~uint32](v bool) Value[ID]   { return Value[ID]{Kind: BoolValue, Bool: v} }
func Float64[ID ~uint32](v float64) Value[ID] {
	return Value[ID]{Kind: Float64Value, Float: math.Float64bits(v)}
}
func String[ID ~uint32](v string) Value[ID] { return Value[ID]{Kind: StringValue, String: v} }

// Unit constructs the sole value of a unit column.
func Unit[ID ~uint32]() Value[ID] { return Value[ID]{Kind: UnitValue} }

// BigInt and BigRat store already-canonical decimal and rational strings.
func BigInt[ID ~uint32](v string) Value[ID] { return Value[ID]{Kind: BigIntValue, String: v} }
func BigRat[ID ~uint32](v string) Value[ID] { return Value[ID]{Kind: BigRatValue, String: v} }

// Column describes the type of one relation column. Sort is meaningful only
// for class columns; backends may use it to enforce their own sort policy.
type Column struct {
	Kind ValueKind
	Sort string
}

// Schema declares a named relation. Its column slice is copied by NewRelation.
type Schema struct {
	Name    string
	Columns []Column
}

// Filter constrains one column during a relation query.
type Filter[ID ~uint32] struct {
	Column int
	Value  Value[ID]
}

// Relation is a set of typed tuples with one equality index per column.
// It is not safe for concurrent use. A backend supplies canonicalize to map
// class-valued cells to current representatives.
type Relation[ID ~uint32] struct {
	schema    Schema
	rows      map[string][]Value[ID]
	subsumed  map[string]struct{}
	index     *ValueIndex[ID]
	epoch     *relationEpoch
	ancestors []relationAncestor
	version   uint64
	floor     uint64
	added     []relationAddition[ID]
}

// NewRelation creates an empty relation and validates its schema.
func NewRelation[ID ~uint32](schema Schema) (*Relation[ID], error) {
	if schema.Name == "" {
		return nil, errInvalidSchema
	}
	cols := append([]Column(nil), schema.Columns...)
	for _, c := range cols {
		if c.Kind < ClassValue || c.Kind > BigRatValue || (c.Kind != ClassValue && c.Sort != "") {
			return nil, errInvalidSchema
		}
	}
	return &Relation[ID]{schema: Schema{Name: schema.Name, Columns: cols}, rows: make(map[string][]Value[ID]), subsumed: make(map[string]struct{}), index: NewValueIndex[ID](len(cols)), epoch: &relationEpoch{}}, nil
}

// MarkSubsumed inserts values if absent and hides the resulting fact from
// Query while retaining it in Rows. It reports whether the fact was newly
// marked subsumed. A successful mark invalidates existing cursors because it
// changes the query-visible relation; callers must recompute and take a new
// snapshot.
func (r *Relation[ID]) MarkSubsumed(values []Value[ID], canonicalize func(ID) ID) (bool, error) {
	row, key, err := r.prepare(values, canonicalize)
	if err != nil {
		return false, err
	}
	return r.markSubsumedPrepared(row, key), nil
}

func (r *Relation[ID]) markSubsumedPrepared(row []Value[ID], key string) bool {
	changed, _ := r.applyPreparedWithBudget(FactSubsume, row, key, nil)
	return changed
}

// Snapshot returns a cursor at the current insertion history position.
// Cursors are relation-state-specific and become invalid after deletion,
// subsumption, or recanonicalization. A source cursor can be used with a
// descendant clone while its epoch remains in the bounded clone lineage.
func (r *Relation[ID]) Snapshot() RelationCursor {
	if r == nil {
		return RelationCursor{}
	}
	return RelationCursor{epoch: r.epoch, version: r.version}
}

// AddedSince returns facts added after cursor in insertion order. Additions
// are reported once; duplicate Add calls do not advance the version. A
// successful Delete, MarkSubsumed, or Recanonicalize invalidates old cursors;
// cursors used with unrelated, diverged, or old clones return ErrRelationReset. In
// either case the caller must recompute the current relation.
func (r *Relation[ID]) AddedSince(cursor RelationCursor) ([][]Value[ID], error) {
	rows, complete, err := r.AddedSinceWithBudget(cursor, nil)
	if err != nil {
		return nil, err
	}
	if !complete {
		return nil, ErrMatchWorkLimit
	}
	return rows, nil
}

// AddedSinceWithBudget is AddedSince with bounded history lookup and row-copy
// work. complete is false when budget runs out; no partial rows are returned.
// A nil budget means unlimited work.
func (r *Relation[ID]) AddedSinceWithBudget(cursor RelationCursor, budget *WorkBudget) (rows [][]Value[ID], complete bool, err error) {
	defer func() {
		if memoryErr := budget.MemoryError(); memoryErr != nil {
			rows, complete, err = nil, false, memoryErr
		}
	}()
	if _, ok := r.cursorVersion(cursor); !ok {
		return nil, false, ErrRelationReset
	}
	lo, hi := 0, len(r.added)
	for lo < hi {
		if !spend(budget, 1) {
			return nil, false, nil
		}
		mid := lo + (hi-lo)/2
		if r.added[mid].version <= cursor.version {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if !spend(budget, 1) {
		return nil, false, nil
	}
	var out [][]Value[ID]
	for _, addition := range r.added[lo:] {
		if !spend(budget, len(addition.row)+1) {
			return nil, false, nil
		}
		copied, ok := memorySlice[Value[ID]](budget, len(addition.row), len(addition.row))
		if !ok {
			return nil, false, budget.MemoryError()
		}
		copy(copied, addition.row)
		out, ok = memoryAppend(budget, out, copied)
		if !ok {
			return nil, false, budget.MemoryError()
		}
	}
	return out, true, nil
}

func (r *Relation[ID]) cursorVersion(cursor RelationCursor) (uint64, bool) {
	if r == nil || cursor.epoch == nil || cursor.version < r.floor {
		return 0, false
	}
	if cursor.epoch == r.epoch {
		return cursor.version, cursor.version <= r.version
	}
	for _, ancestor := range r.ancestors {
		if cursor.epoch == ancestor.epoch {
			return cursor.version, cursor.version <= ancestor.maxVersion
		}
	}
	return 0, false
}

// Compact discards additions at or before cursor to bound retained delta
// history. Cursors older than cursor become invalid and require recomputation.
// Callers with multiple consumers must pass the oldest cursor they still need.
func (r *Relation[ID]) Compact(cursor RelationCursor) error {
	version, ok := r.cursorVersion(cursor)
	if !ok {
		return ErrRelationReset
	}
	n := 0
	for _, addition := range r.added {
		if addition.version > version {
			r.added[n] = addition
			n++
		}
	}
	clear(r.added[n:])
	r.added = r.added[:n]
	r.floor = version
	return nil
}

// Schema returns a defensive copy of the relation schema.
func (r *Relation[ID]) Schema() Schema {
	if r == nil {
		return Schema{}
	}
	return Schema{Name: r.schema.Name, Columns: append([]Column(nil), r.schema.Columns...)}
}

// SchemaWithBudget returns a defensive schema copy after reserving its column
// backing storage. The budget's scope remains open through the copy's last use.
func (r *Relation[ID]) SchemaWithBudget(budget *WorkBudget) (Schema, error) {
	if r == nil {
		return Schema{}, nil
	}
	columns, ok := memorySlice[Column](budget, len(r.schema.Columns), len(r.schema.Columns))
	if !ok {
		return Schema{}, budget.MemoryError()
	}
	copy(columns, r.schema.Columns)
	return Schema{Name: r.schema.Name, Columns: columns}, nil
}

// Clone returns an independent copy of r, including its indexes, delta
// history, and subsumed fact state. Cursors from r remain valid while their
// epochs remain in the bounded clone lineage. The clone has its own epoch so
// cursors from diverged clones cannot be mixed. The returned relation is not
// safe for concurrent use.
func (r *Relation[ID]) Clone() *Relation[ID] {
	if r == nil {
		return nil
	}
	out, err := NewRelation[ID](r.schema)
	if err != nil {
		return nil // Relations can only be constructed with a valid schema.
	}
	for key, row := range r.rows {
		copyRow := append([]Value[ID](nil), row...)
		out.rows[key] = copyRow
	}
	for key := range r.subsumed {
		out.subsumed[key] = struct{}{}
	}
	out.ancestors = append(out.ancestors, r.ancestors...)
	out.ancestors = append(out.ancestors, relationAncestor{epoch: r.epoch, maxVersion: r.version})
	if len(out.ancestors) > relationCursorLineageLimit {
		out.ancestors = append([]relationAncestor(nil), out.ancestors[len(out.ancestors)-relationCursorLineageLimit:]...)
	}
	out.version = r.version
	out.floor = r.floor
	out.added = make([]relationAddition[ID], len(r.added))
	for i, addition := range r.added {
		out.added[i] = relationAddition[ID]{version: addition.version, row: append([]Value[ID](nil), addition.row...)}
	}
	out.index = r.index.Clone()
	return out
}

// Len reports the number of stored facts, including facts marked subsumed.
func (r *Relation[ID]) Len() int {
	if r == nil {
		return 0
	}
	return len(r.rows)
}

// Add inserts values after canonicalizing class values. It reports whether a
// new tuple was added. Invalid tuples are rejected without mutation.
func (r *Relation[ID]) Add(values []Value[ID], canonicalize func(ID) ID) (bool, error) {
	row, key, err := r.prepare(values, canonicalize)
	if err != nil {
		return false, err
	}
	return r.addPrepared(row, key), nil
}

func (r *Relation[ID]) addPrepared(row []Value[ID], key string) bool {
	changed, _ := r.addPreparedWithBudget(row, key, nil)
	return changed
}

// Delete removes a tuple after canonicalizing class values. A successful
// deletion invalidates existing cursors because insertion-only deltas cannot
// represent retraction; callers must recompute and take a new snapshot.
func (r *Relation[ID]) Delete(values []Value[ID], canonicalize func(ID) ID) (bool, error) {
	_, key, err := r.prepare(values, canonicalize)
	if err != nil {
		return false, err
	}
	return r.deletePrepared(key), nil
}

func (r *Relation[ID]) deletePrepared(key string) bool {
	changed, _ := r.applyPreparedWithBudget(FactDelete, nil, key, nil)
	return changed
}

func (r *Relation[ID]) resetAdditions() {
	r.epoch = &relationEpoch{}
	r.ancestors = nil
	r.version = 0
	r.floor = 0
	r.added = nil
}

// Query returns matching rows in deterministic tuple-key order. Each
// candidate examined spends one unit; on exhaustion it returns no partial
// result and complete=false.
func (r *Relation[ID]) Query(filters []Filter[ID], canonicalize func(ID) ID, budget *WorkBudget) (rows [][]Value[ID], complete bool, err error) {
	defer func() {
		if memoryErr := budget.MemoryError(); memoryErr != nil {
			rows, complete, err = nil, false, memoryErr
		}
	}()
	if r == nil {
		return nil, false, errInvalidSchema
	}
	var normalized []Value[ID]
	for _, f := range filters {
		if budget != nil && !budget.TrySpend(1) {
			return nil, false, nil
		}
		if f.Column < 0 || f.Column >= len(r.schema.Columns) {
			return nil, false, errInvalidSchema
		}
		v, e := r.normalize(f.Value, canonicalize)
		if e != nil {
			return nil, false, e
		}
		if r.schema.Columns[f.Column].Kind != v.Kind {
			return nil, false, errInvalidSchema
		}
		var ok bool
		normalized, ok = memoryAppend(budget, normalized, v)
		if !ok {
			return nil, false, budget.MemoryError()
		}
	}
	var candidateColumn = -1
	var candidateValue Value[ID]
	candidateCount := math.MaxInt
	for i, f := range filters {
		v := normalized[i]
		count := r.index.Count(f.Column, v)
		if count == 0 {
			return nil, true, nil
		}
		if count < candidateCount {
			candidateColumn, candidateValue, candidateCount = f.Column, v, count
		}
	}
	var keys []string
	if candidateColumn < 0 {
		for k := range r.rows {
			if budget != nil && !budget.TrySpend(1) {
				return nil, false, nil
			}
			var ok bool
			keys, ok = memoryAppend(budget, keys, k)
			if !ok {
				return nil, false, budget.MemoryError()
			}
		}
	} else {
		complete := r.index.Range(candidateColumn, candidateValue, func(k string) bool {
			if budget != nil && !budget.TrySpend(1) {
				return false
			}
			var ok bool
			keys, ok = memoryAppend(budget, keys, k)
			return ok
		})
		if !complete {
			return nil, false, budget.MemoryError()
		}
	}
	if !sortKeys(keys, budget) {
		return nil, false, nil
	}
	out, ok := memorySlice[[]Value[ID]](budget, 0, len(keys))
	if !ok {
		return nil, false, budget.MemoryError()
	}
	for _, key := range keys {
		if budget != nil && !budget.TrySpend(1) {
			return nil, false, nil
		}
		if _, hidden := r.subsumed[key]; hidden {
			continue
		}
		row := r.rows[key]
		match := true
		for i, f := range filters {
			if budget != nil && !budget.TrySpend(1) {
				return nil, false, nil
			}
			if row[f.Column] != normalized[i] {
				match = false
				break
			}
		}
		if match {
			copied, ok := memorySlice[Value[ID]](budget, len(row), len(row))
			if !ok {
				return nil, false, budget.MemoryError()
			}
			copy(copied, row)
			out, ok = memoryAppend(budget, out, copied)
			if !ok {
				return nil, false, budget.MemoryError()
			}
		}
	}
	return out, true, nil
}

// Rows returns a defensive copy of all rows in deterministic order.
func (r *Relation[ID]) Rows() [][]Value[ID] {
	if r == nil {
		return nil
	}
	keys := make([]string, 0, len(r.rows))
	for k := range r.rows {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([][]Value[ID], 0, len(keys))
	for _, k := range keys {
		out = append(out, append([]Value[ID](nil), r.rows[k]...))
	}
	return out
}

// Recanonicalize atomically replaces class IDs with current representatives,
// merges duplicate rows, and rebuilds all indexes. It starts a new delta epoch;
// callers must recompute the resulting rows and take a new snapshot.
func (r *Relation[ID]) Recanonicalize(canonicalize func(ID) ID) error {
	if r == nil {
		return errInvalidSchema
	}
	if canonicalize == nil {
		return ErrMissingCanonicalizer
	}
	next, err := NewRelation[ID](r.schema)
	if err != nil {
		return err
	}
	for _, row := range r.Rows() {
		oldKey := tupleKey(row)
		if _, hidden := r.subsumed[oldKey]; hidden {
			if _, err := next.MarkSubsumed(row, canonicalize); err != nil {
				return err
			}
		} else if _, err := next.Add(row, canonicalize); err != nil {
			return err
		}
	}
	*r = *next
	// Recanonicalization can merge facts and changes the identity of rows.
	// Treat the rebuilt relation as a new baseline for incremental consumers.
	r.resetAdditions()
	return nil
}

func (r *Relation[ID]) canonicalCopyWithBudget(canonicalize func(ID) ID, budget *WorkBudget) (*Relation[ID], error) {
	if r == nil {
		return nil, errInvalidSchema
	}
	if !spend(budget, len(r.schema.Columns)) || !spend(budget, len(r.rows)) {
		return nil, ErrMatchWorkLimit
	}
	columns, ok := memorySlice[Column](budget, len(r.schema.Columns), len(r.schema.Columns))
	if !ok {
		return nil, budget.MemoryError()
	}
	copy(columns, r.schema.Columns)
	rows, ok := memoryMap[string, []Value[ID]](budget, len(r.rows))
	if !ok {
		return nil, budget.MemoryError()
	}
	subsumed, ok := memoryMap[string, struct{}](budget, len(r.subsumed))
	if !ok {
		return nil, budget.MemoryError()
	}
	index, ok := newValueIndexWithBudget[ID](len(columns), budget)
	if !ok {
		return nil, budget.MemoryError()
	}
	epoch, ok := memoryObject(budget, relationEpoch{})
	if !ok {
		return nil, budget.MemoryError()
	}
	result, ok := memoryObject(budget, Relation[ID]{schema: Schema{Name: r.schema.Name, Columns: columns}, rows: rows, subsumed: subsumed, index: index, epoch: epoch})
	if !ok {
		return nil, budget.MemoryError()
	}
	for oldKey, source := range r.rows {
		if !spend(budget, len(source)) {
			return nil, ErrMatchWorkLimit
		}
		row, ok := memorySlice[Value[ID]](budget, len(source), len(source))
		if !ok {
			return nil, budget.MemoryError()
		}
		for i, value := range source {
			value, err := r.normalize(value, canonicalize)
			if err != nil {
				return nil, err
			}
			row[i] = value
		}
		key, ok := tupleKeyWithBudget(row, budget)
		if !ok {
			return nil, budget.MemoryError()
		}
		if _, exists := rows[key]; !exists {
			if !memoryPut(budget, rows, key, row) || !index.addWithBudget(row, key, budget) {
				return nil, budget.MemoryError()
			}
		}
		if _, hidden := r.subsumed[oldKey]; hidden {
			if !memoryPut(budget, subsumed, key, struct{}{}) {
				return nil, budget.MemoryError()
			}
		}
	}
	return result, nil
}

func (r *Relation[ID]) prepare(values []Value[ID], canonicalize func(ID) ID) ([]Value[ID], string, error) {
	if r == nil || len(values) != len(r.schema.Columns) {
		return nil, "", errInvalidSchema
	}
	row := append([]Value[ID](nil), values...)
	for i, v := range row {
		if r.schema.Columns[i].Kind != v.Kind {
			return nil, "", errInvalidSchema
		}
		nv, err := r.normalize(v, canonicalize)
		if err != nil {
			return nil, "", err
		}
		row[i] = nv
	}
	return row, tupleKey(row), nil
}

func (r *Relation[ID]) normalize(v Value[ID], canonicalize func(ID) ID) (Value[ID], error) {
	if r == nil || v.Kind < ClassValue || v.Kind > BigRatValue {
		return Value[ID]{}, errInvalidSchema
	}
	var clean Value[ID]
	clean.Kind = v.Kind
	switch v.Kind {
	case ClassValue:
		if canonicalize == nil {
			return Value[ID]{}, ErrMissingCanonicalizer
		}
		clean.Class = v.Class
	case Int64Value:
		clean.Int64 = v.Int64
	case BoolValue:
		clean.Bool = v.Bool
	case Float64Value:
		clean.Float = v.Float
	case StringValue, BigIntValue, BigRatValue:
		clean.String = v.String
	case UnitValue:
	}
	if clean != v {
		return Value[ID]{}, errInvalidSchema
	}
	if clean.Kind == ClassValue {
		clean.Class = canonicalize(clean.Class)
	}
	return clean, nil
}

// sortKeys uses bottom-up merge sort so every comparison and output move is
// charged. Enumeration is charged by Query before this function is called.
func sortKeys(keys []string, budget *WorkBudget) bool {
	n := len(keys)
	if n < 2 {
		return true
	}
	tmp, ok := memorySlice[string](budget, n, n)
	if !ok {
		return false
	}
	for width := 1; width < n; width *= 2 {
		for lo := 0; lo < n; lo += 2 * width {
			mid, hi := lo+width, lo+2*width
			if mid > n {
				mid = n
			}
			if hi > n {
				hi = n
			}
			i, j, out := lo, mid, lo
			for i < mid && j < hi {
				if budget != nil && !budget.TrySpend(1) {
					return false
				}
				if budget != nil && !budget.TrySpend(1) {
					return false
				}
				if keys[i] <= keys[j] {
					tmp[out] = keys[i]
					i++
				} else {
					tmp[out] = keys[j]
					j++
				}
				out++
			}
			for i < mid {
				if budget != nil && !budget.TrySpend(1) {
					return false
				}
				tmp[out] = keys[i]
				i++
				out++
			}
			for j < hi {
				if budget != nil && !budget.TrySpend(1) {
					return false
				}
				tmp[out] = keys[j]
				j++
				out++
			}
			if budget != nil && !budget.TrySpend(hi-lo) {
				return false
			}
			copy(keys[lo:hi], tmp[lo:hi])
		}
		if width > n/2 {
			break
		}
	}
	return true
}

func tupleKey[ID ~uint32](row []Value[ID]) string {
	b := make([]byte, 0, len(row)*16)
	var buf [8]byte
	for _, v := range row {
		b = append(b, byte(v.Kind))
		switch v.Kind {
		case ClassValue:
			binary.LittleEndian.PutUint64(buf[:], uint64(v.Class))
			b = append(b, buf[:]...)
		case Int64Value:
			binary.LittleEndian.PutUint64(buf[:], uint64(v.Int64))
			b = append(b, buf[:]...)
		case BoolValue:
			if v.Bool {
				b = append(b, 1)
			} else {
				b = append(b, 0)
			}
		case Float64Value:
			binary.LittleEndian.PutUint64(buf[:], v.Float)
			b = append(b, buf[:]...)
		case StringValue:
			binary.LittleEndian.PutUint64(buf[:], uint64(len(v.String)))
			b = append(b, buf[:]...)
			b = append(b, v.String...)
		case UnitValue:
		case BigIntValue, BigRatValue:
			binary.LittleEndian.PutUint64(buf[:], uint64(len(v.String)))
			b = append(b, buf[:]...)
			b = append(b, v.String...)
		}
	}
	return string(b)
}

// tupleKeyWithBudget computes the exact encoded size before creating either
// the byte backing or the returned string. Values and their strings are borrowed.
func tupleKeyWithBudget[ID ~uint32](row []Value[ID], budget *WorkBudget) (string, bool) {
	size := 0
	for _, value := range row {
		cell := 1
		switch value.Kind {
		case ClassValue, Int64Value, Float64Value:
			cell += 8
		case BoolValue:
			cell++
		case StringValue, BigIntValue, BigRatValue:
			if len(value.String) > math.MaxInt-9 {
				return "", false
			}
			cell += 8 + len(value.String)
		}
		if cell > math.MaxInt-size {
			return "", false
		}
		size += cell
	}
	data, ok := memorySlice[byte](budget, size, size)
	if !ok {
		return "", false
	}
	offset := 0
	for _, value := range row {
		data[offset] = byte(value.Kind)
		offset++
		switch value.Kind {
		case ClassValue:
			binary.LittleEndian.PutUint64(data[offset:], uint64(value.Class))
			offset += 8
		case Int64Value:
			binary.LittleEndian.PutUint64(data[offset:], uint64(value.Int64))
			offset += 8
		case Float64Value:
			binary.LittleEndian.PutUint64(data[offset:], value.Float)
			offset += 8
		case BoolValue:
			if value.Bool {
				data[offset] = 1
			}
			offset++
		case StringValue, BigIntValue, BigRatValue:
			binary.LittleEndian.PutUint64(data[offset:], uint64(len(value.String)))
			offset += 8
			offset += copy(data[offset:], value.String)
		}
	}
	if !budget.ReserveMemory(int64(size)) {
		return "", false
	}
	return string(data), true
}
