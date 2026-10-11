package egraphcore

import "errors"

// ErrInvalidJoin reports an invalid atom or binding join.
var ErrInvalidJoin = errors.New("egraphcore: invalid relation join")

// TermKind identifies a column term in a relational atom.
type TermKind uint8

const (
	// WildcardTerm accepts any value in a relation column.
	WildcardTerm TermKind = iota
	// VariableTerm binds a column value to a named variable.
	VariableTerm
	// ConstantTerm requires the column to equal Value.
	ConstantTerm
)

// Term is a wildcard, a named variable, or a typed constant.
type Term[ID ~uint32] struct {
	Kind  TermKind
	Name  string
	Value Value[ID]
}

// Wildcard returns a term that accepts any column value.
func Wildcard[ID ~uint32]() Term[ID] { return Term[ID]{Kind: WildcardTerm} }

// Variable returns a term that binds a column value to name.
func Variable[ID ~uint32](name string) Term[ID] {
	return Term[ID]{Kind: VariableTerm, Name: name}
}

// Constant returns a term that requires an exact typed value.
func Constant[ID ~uint32](value Value[ID]) Term[ID] {
	return Term[ID]{Kind: ConstantTerm, Value: value}
}

// Atom applies terms to the columns of one relation. Repeated variable names
// within or across atoms impose equality. Variable names occupy one shared
// namespace; callers that distinguish backend namespaces should qualify names.
type Atom[ID ~uint32] struct {
	Relation *Relation[ID]
	Terms    []Term[ID]
}

// Binding is one variable assignment produced by Join.
type Binding[ID ~uint32] map[string]Value[ID]

// Join evaluates a conjunction of relation atoms. At each step it probes the
// remaining atoms with current bindings and selects the one with the fewest
// rows. It uses Relation.Query indexes for constants and bound variables, and
// returns deterministic bindings. canonicalize is required to normalize
// class-valued columns. If work runs out, it returns no partial bindings and
// complete=false. A nil budget is unlimited.
func Join[ID ~uint32](atoms []Atom[ID], canonicalize func(ID) ID, budget *WorkBudget) (results []Binding[ID], complete bool, err error) {
	if canonicalize == nil {
		return nil, false, ErrInvalidJoin
	}
	for _, atom := range atoms {
		if !spend(budget, 1) {
			return nil, false, nil
		}
		for range atom.Terms {
			if !spend(budget, 1) {
				return nil, false, nil
			}
		}
	}
	if !validJoin(atoms) {
		return nil, false, ErrInvalidJoin
	}
	ordered := append([]Atom[ID](nil), atoms...)
	// Use a budgeted stable insertion sort. Stable order preserves source order
	// for equally sized relations; a dynamic selective order is chosen below.
	for i := 1; i < len(ordered); i++ {
		item := ordered[i]
		j := i
		for j > 0 {
			if !spend(budget, 1) {
				return nil, false, nil
			}
			if ordered[j-1].Relation.Len() <= item.Relation.Len() {
				break
			}
			if !spend(budget, 1) {
				return nil, false, nil
			}
			ordered[j] = ordered[j-1]
			j--
		}
		if j != i {
			if !spend(budget, 1) {
				return nil, false, nil
			}
			ordered[j] = item
		}
	}
	bindings := make([]Binding[ID], 0)
	seen := make(map[string]struct{})
	var visit func([]Atom[ID], Binding[ID]) bool
	visit = func(remaining []Atom[ID], binding Binding[ID]) bool {
		if len(remaining) == 0 {
			if !spend(budget, 1) {
				return false
			}
			key, ok := bindingKey(binding, budget)
			if !ok {
				return false
			}
			if _, ok := seen[key]; ok {
				return true
			}
			if !spend(budget, 1) {
				return false
			}
			seen[key] = struct{}{}
			copy := make(Binding[ID], len(binding))
			for name, value := range binding {
				copy[name] = value
			}
			bindings = append(bindings, copy)
			return true
		}
		best := 0
		var rows [][]Value[ID]
		for i, candidate := range remaining {
			candidateRows, complete, queryErr := queryAtom(candidate, binding, canonicalize, budget)
			if queryErr != nil {
				err = queryErr
				return false
			}
			if !complete {
				return false
			}
			if len(candidateRows) == 0 {
				return true
			}
			if i == 0 || len(candidateRows) < len(rows) {
				best, rows = i, candidateRows
			}
		}
		atom := remaining[best]
		if !spend(budget, len(remaining)-1) {
			return false
		}
		nextAtoms := make([]Atom[ID], 0, len(remaining)-1)
		nextAtoms = append(nextAtoms, remaining[:best]...)
		nextAtoms = append(nextAtoms, remaining[best+1:]...)
		for _, row := range rows {
			if !spend(budget, len(row)+1) {
				return false
			}
			next := make(Binding[ID], len(binding)+len(atom.Terms))
			for name, value := range binding {
				if !spend(budget, 1) {
					return false
				}
				next[name] = value
			}
			match := true
			for column, term := range atom.Terms {
				if term.Kind != VariableTerm {
					continue
				}
				if previous, exists := next[term.Name]; exists {
					if previous != row[column] {
						match = false
						break
					}
				} else {
					next[term.Name] = row[column]
				}
			}
			if match && !visit(nextAtoms, next) {
				return false
			}
		}
		return true
	}
	if !visit(ordered, make(Binding[ID])) {
		if err != nil {
			return nil, false, err
		}
		return nil, false, nil
	}
	return bindings, true, nil
}

// JoinBindings joins two materialized binding sets on shared variable names.
// Every binding must contain every name in rightKeys; incomplete key plans
// return ErrInvalidJoin. It uses a hash index on rightKeys and also checks
// equality for every other variable present in both bindings. Class values
// are canonicalized before comparison and in returned bindings. Output
// bindings are deterministic and deduplicated. canonicalize is required. On
// budget exhaustion it returns no partial output and complete=false. A nil
// budget is unlimited.
func JoinBindings[ID ~uint32](left, right []Binding[ID], rightKeys []string, canonicalize func(ID) ID, budget *WorkBudget) (results []Binding[ID], complete bool, err error) {
	if canonicalize == nil {
		return nil, false, ErrInvalidJoin
	}
	keys, ok := normalizeJoinKeys(rightKeys, budget)
	if !ok {
		return nil, false, nil
	}
	if keys == nil {
		return nil, false, ErrInvalidJoin
	}
	leftRows, ok, prepareErr := prepareBindings(left, canonicalize, budget)
	if !ok {
		if prepareErr != nil {
			return nil, false, prepareErr
		}
		return nil, false, nil
	}
	rightRows, ok, prepareErr := prepareBindings(right, canonicalize, budget)
	if !ok {
		if prepareErr != nil {
			return nil, false, prepareErr
		}
		return nil, false, nil
	}
	if !sortBindingRows(leftRows, budget) || !sortBindingRows(rightRows, budget) {
		return nil, false, nil
	}
	index := make(map[string][]bindingRow[ID])
	for _, row := range rightRows {
		key, present, workOK := selectedBindingKey(row.binding, keys, budget)
		if !workOK {
			return nil, false, nil
		}
		if !present {
			return nil, false, ErrInvalidJoin
		}
		if !spend(budget, 1) {
			return nil, false, nil
		}
		index[key] = append(index[key], row)
	}
	seen := make(map[string]struct{})
	for _, lrow := range leftRows {
		key, present, workOK := selectedBindingKey(lrow.binding, keys, budget)
		if !workOK {
			return nil, false, nil
		}
		if !present {
			return nil, false, ErrInvalidJoin
		}
		for _, rrow := range index[key] {
			if !spend(budget, 1) {
				return nil, false, nil
			}
			merged, matches, workOK := mergeBindings(lrow.binding, rrow.binding, canonicalize, budget)
			if !workOK {
				return nil, false, nil
			}
			if !matches {
				continue
			}
			mergeKey, workOK := bindingKey(merged, budget)
			if !workOK {
				return nil, false, nil
			}
			if _, exists := seen[mergeKey]; exists {
				continue
			}
			if !spend(budget, 1) {
				return nil, false, nil
			}
			seen[mergeKey] = struct{}{}
			results = append(results, merged)
		}
	}
	resultRows := make([]bindingRow[ID], len(results))
	for i, binding := range results {
		key, workOK := bindingKey(binding, budget)
		if !workOK {
			return nil, false, nil
		}
		resultRows[i] = bindingRow[ID]{binding: binding, key: key}
	}
	if !sortBindingRows(resultRows, budget) {
		return nil, false, nil
	}
	results = make([]Binding[ID], len(resultRows))
	for i, row := range resultRows {
		if !spend(budget, 1) {
			return nil, false, nil
		}
		results[i] = row.binding
	}
	return results, true, nil
}

type bindingRow[ID ~uint32] struct {
	binding Binding[ID]
	key     string
}

func normalizeJoinKeys(keys []string, budget *WorkBudget) ([]string, bool) {
	out := make([]string, len(keys))
	copy(out, keys)
	for _, key := range out {
		if !spend(budget, 1) {
			return nil, false
		}
		if key == "" {
			return nil, true
		}
	}
	if !sortKeys(out, budget) {
		return nil, false
	}
	for i := 1; i < len(out); i++ {
		if out[i] == out[i-1] {
			return nil, true
		}
	}
	return out, true
}

func prepareBindings[ID ~uint32](input []Binding[ID], canonicalize func(ID) ID, budget *WorkBudget) ([]bindingRow[ID], bool, error) {
	rows := make([]bindingRow[ID], 0, len(input))
	for _, binding := range input {
		if !spend(budget, 1) {
			return nil, false, nil
		}
		normalized := make(Binding[ID], len(binding))
		for name, value := range binding {
			if !spend(budget, 1) {
				return nil, false, nil
			}
			if name == "" {
				return nil, false, ErrInvalidJoin
			}
			value, ok := normalizeBindingValue(value, canonicalize)
			if !ok {
				return nil, false, ErrInvalidJoin
			}
			normalized[name] = value
		}
		key, ok := bindingKey(normalized, budget)
		if !ok {
			return nil, false, nil
		}
		rows = append(rows, bindingRow[ID]{binding: normalized, key: key})
	}
	return rows, true, nil
}

func normalizeBindingValue[ID ~uint32](value Value[ID], canonicalize func(ID) ID) (Value[ID], bool) {
	if value.Kind < ClassValue || value.Kind > BigRatValue {
		return Value[ID]{}, false
	}
	normalized := Value[ID]{Kind: value.Kind}
	switch value.Kind {
	case ClassValue:
		normalized.Class = value.Class
	case Int64Value:
		normalized.Int64 = value.Int64
	case BoolValue:
		normalized.Bool = value.Bool
	case Float64Value:
		normalized.Float = value.Float
	case StringValue, BigIntValue, BigRatValue:
		normalized.String = value.String
	}
	if normalized != value {
		return Value[ID]{}, false
	}
	if normalized.Kind == ClassValue && canonicalize != nil {
		normalized.Class = canonicalize(normalized.Class)
	}
	return normalized, true
}

func sortBindingRows[ID ~uint32](rows []bindingRow[ID], budget *WorkBudget) bool {
	n := len(rows)
	if n < 2 {
		return true
	}
	tmp := make([]bindingRow[ID], n)
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
				if !spend(budget, 1) {
					return false
				}
				if rows[i].key <= rows[j].key {
					tmp[out] = rows[i]
					i++
				} else {
					tmp[out] = rows[j]
					j++
				}
				out++
			}
			for i < mid {
				if !spend(budget, 1) {
					return false
				}
				tmp[out] = rows[i]
				i++
				out++
			}
			for j < hi {
				if !spend(budget, 1) {
					return false
				}
				tmp[out] = rows[j]
				j++
				out++
			}
			if !spend(budget, hi-lo) {
				return false
			}
			copy(rows[lo:hi], tmp[lo:hi])
		}
		if width > n/2 {
			break
		}
	}
	return true
}

func selectedBindingKey[ID ~uint32](binding Binding[ID], keys []string, budget *WorkBudget) (string, bool, bool) {
	values := make([]Value[ID], 0, len(keys)*2)
	for _, name := range keys {
		if !spend(budget, 1) {
			return "", false, false
		}
		value, ok := binding[name]
		if !ok {
			return "", false, true
		}
		values = append(values, String[ID](name), value)
	}
	if !spend(budget, len(values)) {
		return "", false, false
	}
	return tupleKey(values), true, true
}

func mergeBindings[ID ~uint32](left, right Binding[ID], canonicalize func(ID) ID, budget *WorkBudget) (Binding[ID], bool, bool) {
	merged := make(Binding[ID], len(left)+len(right))
	for name, value := range left {
		if !spend(budget, 1) {
			return nil, false, false
		}
		merged[name] = value
	}
	for name, value := range right {
		if !spend(budget, 1) {
			return nil, false, false
		}
		value, ok := normalizeBindingValue(value, canonicalize)
		if !ok {
			return nil, false, false
		}
		if previous, exists := merged[name]; exists {
			if previous != value {
				return nil, false, true
			}
		} else {
			merged[name] = value
		}
	}
	return merged, true, true
}

func validJoin[ID ~uint32](atoms []Atom[ID]) bool {
	type variableType struct {
		kind ValueKind
		sort string
	}
	variableTypes := make(map[string]variableType)
	for _, atom := range atoms {
		if atom.Relation == nil || len(atom.Terms) != len(atom.Relation.schema.Columns) {
			return false
		}
		for column, term := range atom.Terms {
			switch term.Kind {
			case WildcardTerm:
				if term.Name != "" || term.Value != (Value[ID]{}) {
					return false
				}
			case VariableTerm:
				if term.Name == "" || term.Value != (Value[ID]{}) {
					return false
				}
				columnType := atom.Relation.schema.Columns[column]
				if previous, ok := variableTypes[term.Name]; ok {
					if previous.kind != columnType.Kind || (previous.sort != "" && columnType.Sort != "" && previous.sort != columnType.Sort) {
						return false
					}
					if previous.sort == "" {
						previous.sort = columnType.Sort
					}
					variableTypes[term.Name] = previous
				} else {
					variableTypes[term.Name] = variableType{kind: columnType.Kind, sort: columnType.Sort}
				}
			case ConstantTerm:
				if term.Name != "" || term.Value.Kind != atom.Relation.schema.Columns[column].Kind {
					return false
				}
			default:
				return false
			}
		}
	}
	return true
}

func queryAtom[ID ~uint32](atom Atom[ID], binding Binding[ID], canonicalize func(ID) ID, budget *WorkBudget) ([][]Value[ID], bool, error) {
	filters := make([]Filter[ID], 0, len(atom.Terms))
	for column, term := range atom.Terms {
		if !spend(budget, 1) {
			return nil, false, nil
		}
		var value Value[ID]
		switch term.Kind {
		case ConstantTerm:
			value = term.Value
		case VariableTerm:
			if bound, ok := binding[term.Name]; ok {
				value = bound
			} else {
				continue
			}
		case WildcardTerm:
			continue
		}
		if !spend(budget, 1) {
			return nil, false, nil
		}
		filters = append(filters, Filter[ID]{Column: column, Value: value})
	}
	rows, complete, err := atom.Relation.Query(filters, canonicalize, budget)
	if err != nil || !complete {
		return nil, complete, err
	}
	for i := range rows {
		if !spend(budget, len(rows[i])) {
			return nil, false, nil
		}
		for column, value := range rows[i] {
			normalized, ok := normalizeBindingValue(value, canonicalize)
			if !ok {
				return nil, false, ErrInvalidJoin
			}
			rows[i][column] = normalized
		}
	}
	return rows, true, nil
}

func bindingKey[ID ~uint32](binding Binding[ID], budget *WorkBudget) (string, bool) {
	names := make([]string, 0, len(binding))
	for name := range binding {
		if !spend(budget, 1) {
			return "", false
		}
		names = append(names, name)
	}
	if !sortKeys(names, budget) {
		return "", false
	}
	values := make([]Value[ID], 0, len(names))
	for _, name := range names {
		// Names are length-prefixed to avoid ambiguous concatenations.
		if !spend(budget, 2) {
			return "", false
		}
		values = append(values, String[ID](name))
		values = append(values, binding[name])
	}
	if !spend(budget, len(values)) {
		return "", false
	}
	return tupleKey(values), true
}

func spend(budget *WorkBudget, units int) bool {
	return budget == nil || budget.TrySpend(units)
}
