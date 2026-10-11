package egraph

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/tmc/lib/egraphcore"
)

// RelationSchema declares a set of typed tuples. An empty class-column sort
// accepts any e-class sort.
type RelationSchema struct {
	Name    string
	Sorts   []string // Compatibility view of class columns.
	Columns []FunctionColumnType
}

type relationTable struct {
	schema RelationSchema
	core   *egraphcore.Relation[Id]
	// The maps below are compatibility views for Apple's existing matcher.
	// The shared relation owns rows, indexes, and subsumption state.
	rows     map[string][]FunctionValue
	subsumed map[string]struct{}
	index    functionValueIndex
}

// TypedRelationRow is a copy of one canonical mixed relation row.
type TypedRelationRow struct {
	Columns  []FunctionValue
	Subsumed bool
}

type rowIndex []map[Id]map[string]struct{}

func newRowIndex(columns int) rowIndex {
	return make(rowIndex, columns)
}

func addIndexedRow(index rowIndex, tuple []Id, key string) {
	for i, id := range tuple {
		if index[i] == nil {
			index[i] = make(map[Id]map[string]struct{})
		}
		if index[i][id] == nil {
			index[i][id] = make(map[string]struct{})
		}
		index[i][id][key] = struct{}{}
	}
}

func removeIndexedRow(index rowIndex, tuple []Id, key string) {
	for i, id := range tuple {
		if index[i] == nil || index[i][id] == nil {
			continue
		}
		delete(index[i][id], key)
		if len(index[i][id]) == 0 {
			delete(index[i], id)
		}
	}
}

// RegisterRelation declares a predicate relation with one sort per column.
// Re-registering an identical schema is a no-op.
func (g *EGraph) RegisterRelation(name string, sorts ...string) error {
	if g == nil {
		return fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.relations == nil {
		return fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return fmt.Errorf("egraph: rebuild before relation registration: %w", err)
	}
	if name == "" {
		return fmt.Errorf("egraph: empty relation name")
	}
	if owner, ok := g.variantOwners[name]; ok {
		return fmt.Errorf("egraph: relation %q is the constructor of datatype %q", name, owner)
	}
	if g.functions[name] != nil {
		return fmt.Errorf("egraph: name %q is already registered as a function", name)
	}
	columns := make([]FunctionColumnType, len(sorts))
	for i, sort := range sorts {
		columns[i] = ClassColumn(sort)
	}
	return g.registerRelation(name, columns, sorts)
}

// RegisterTypedRelation declares a predicate relation with class or primitive columns.
func (g *EGraph) RegisterTypedRelation(name string, columns ...FunctionColumnType) error {
	if g == nil {
		return fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.relations == nil {
		return fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return fmt.Errorf("egraph: rebuild before relation registration: %w", err)
	}
	if name == "" {
		return fmt.Errorf("egraph: empty relation name")
	}
	if owner, ok := g.variantOwners[name]; ok {
		return fmt.Errorf("egraph: relation %q is the constructor of datatype %q", name, owner)
	}
	if g.functions[name] != nil {
		return fmt.Errorf("egraph: name %q is already registered as a function", name)
	}
	for i, typ := range columns {
		if typ.Kind < FunctionClassColumn || typ.Kind > FunctionBigRatColumn {
			return fmt.Errorf("egraph: relation %q column %d has invalid kind %d", name, i, typ.Kind)
		}
		if typ.Kind != FunctionClassColumn && typ.Sort != "" {
			return fmt.Errorf("egraph: relation %q primitive column %d has an e-class sort", name, i)
		}
	}
	sorts := make([]string, len(columns))
	for i, typ := range columns {
		sorts[i] = typ.Sort
	}
	return g.registerRelation(name, columns, sorts)
}

func (g *EGraph) registerRelation(name string, columns []FunctionColumnType, sorts []string) error {
	if g.declaredSortsStrict {
		for i, column := range columns {
			if err := validateDeclaredClassColumn(g, column, fmt.Sprintf("relation %q column %d", name, i)); err != nil {
				return fmt.Errorf("egraph: %w", err)
			}
		}
	}
	if current := g.relations[name]; current != nil {
		if !equalStrings(current.schema.Sorts, sorts) || !equalFunctionColumnTypes(current.schema.Columns, columns) {
			return fmt.Errorf("egraph: relation %q is already registered with a different schema", name)
		}
		return nil
	}
	coreColumns := make([]egraphcore.Column, len(columns))
	for i, column := range columns {
		coreColumns[i] = egraphcore.Column{Kind: coreValueKind(column.Kind), Sort: column.Sort}
	}
	if g.facts == nil {
		g.facts = egraphcore.NewFactStore[Id]()
	}
	core, err := g.facts.Register(egraphcore.Schema{Name: name, Columns: coreColumns})
	if err != nil {
		return fmt.Errorf("egraph: create relation %q: %w", name, err)
	}
	g.relations[name] = &relationTable{
		schema:   RelationSchema{Name: name, Sorts: append([]string(nil), sorts...), Columns: append([]FunctionColumnType(nil), columns...)},
		core:     core,
		rows:     make(map[string][]FunctionValue),
		subsumed: make(map[string]struct{}),
		index:    make(functionValueIndex, len(columns)),
	}
	g.revision++
	return nil
}

// SubsumeRelation inserts a tuple if needed, then hides it from relation
// pattern matching while retaining it in the table for inspection. It returns
// false if the tuple is already subsumed.
func (g *EGraph) SubsumeRelation(name string, tuple ...Id) (bool, error) {
	return g.mutateRelationValues(name, classFunctionValues(tuple), RelationSubsume)
}

// AddRelation inserts a tuple. IDs are canonicalized before it is stored.
// It returns false when the tuple is already present.
func (g *EGraph) AddRelation(name string, tuple ...Id) (bool, error) {
	return g.mutateRelationValues(name, classFunctionValues(tuple), RelationInsert)
}

// DeleteRelation removes a tuple. It returns false when the tuple is absent.
func (g *EGraph) DeleteRelation(name string, tuple ...Id) (bool, error) {
	return g.mutateRelationValues(name, classFunctionValues(tuple), RelationDelete)
}

// RelationRows returns a sorted copy of all canonical tuples, including
// subsumed rows.
func (g *EGraph) RelationRows(name string) ([][]Id, error) {
	if g == nil {
		return nil, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.relations == nil {
		return nil, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("egraph: rebuild before reading relation rows: %w", err)
	}
	table := g.relations[name]
	if table == nil {
		return nil, fmt.Errorf("egraph: unknown relation %q", name)
	}
	typedRows := make([][]FunctionValue, 0)
	for _, row := range table.core.Rows() {
		typedRows = append(typedRows, fromCoreValues(row))
	}
	sort.Slice(typedRows, func(i, j int) bool { return typedFunctionKey(typedRows[i]) < typedFunctionKey(typedRows[j]) })
	rows := make([][]Id, 0, len(typedRows))
	for _, values := range typedRows {
		row := make([]Id, len(values))
		for i, value := range values {
			if value.Kind != FunctionClassColumn {
				return nil, fmt.Errorf("egraph: relation %q contains primitive column %d; use RelationValues", name, i)
			}
			row[i] = value.ID
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func (g *EGraph) relation(name string, arity int) (*relationTable, error) {
	table := g.relations[name]
	if table == nil {
		return nil, fmt.Errorf("egraph: unknown relation %q", name)
	}
	if len(table.schema.Columns) != arity {
		return nil, fmt.Errorf("egraph: relation %q expects %d columns, got %d", name, len(table.schema.Columns), arity)
	}
	return table, nil
}

func (g *EGraph) canonicalTuple(table *relationTable, tuple []Id) ([]FunctionValue, error) {
	values := make([]FunctionValue, len(tuple))
	for i, id := range tuple {
		values[i] = ClassValue(id)
	}
	return g.canonicalRelationValues(table, values)
}

func (g *EGraph) canonicalRelationValues(table *relationTable, values []FunctionValue) ([]FunctionValue, error) {
	if len(values) != len(table.schema.Columns) {
		return nil, fmt.Errorf("egraph: relation %q expects %d columns, got %d", table.schema.Name, len(table.schema.Columns), len(values))
	}
	canonical := make([]FunctionValue, len(values))
	for i, value := range values {
		var err error
		canonical[i], err = g.canonicalRelationValue(table.schema.Columns[i], value, fmt.Sprintf("relation %q column %d", table.schema.Name, i))
		if err != nil {
			return nil, err
		}
	}
	return canonical, nil
}

func (g *EGraph) canonicalRelationValue(typ FunctionColumnType, value FunctionValue, where string) (FunctionValue, error) {
	if typ.Kind == FunctionFloat64Column {
		// Relations use egraphcore's IEEE-bit identity for floats.
		if value.Kind != typ.Kind {
			return FunctionValue{}, fmt.Errorf("egraph: %s requires kind %d, got %d", where, typ.Kind, value.Kind)
		}
		return Float64Value(value.Float64), nil
	}
	return g.canonicalTypedValue(typ, value, where)
}

func (g *EGraph) canonicalizeRelations() error {
	if g.facts != nil {
		if err := g.facts.Recanonicalize(g.Find); err != nil {
			return fmt.Errorf("egraph: recanonicalize relations: %w", err)
		}
	}
	for _, table := range g.relations {
		subsumed := make(map[string]struct{}, len(table.subsumed))
		for key := range table.subsumed {
			if row, ok := table.rows[key]; ok {
				canonical := append([]FunctionValue(nil), row...)
				for i, value := range canonical {
					if value.Kind == FunctionClassColumn {
						canonical[i] = ClassValue(g.Find(value.ID))
					}
				}
				subsumed[typedFunctionKey(canonical)] = struct{}{}
			}
		}
		table.subsumed = subsumed
		g.syncRelationView(table)
	}
	return nil
}

// AddRelationValues inserts a mixed typed tuple.
func (g *EGraph) AddRelationValues(name string, values ...FunctionValue) (bool, error) {
	return g.mutateRelationValues(name, values, RelationInsert)
}

// DeleteRelationValues removes a mixed typed tuple.
func (g *EGraph) DeleteRelationValues(name string, values ...FunctionValue) (bool, error) {
	return g.mutateRelationValues(name, values, RelationDelete)
}

// SubsumeRelationValues inserts and hides a mixed typed tuple.
func (g *EGraph) SubsumeRelationValues(name string, values ...FunctionValue) (bool, error) {
	return g.mutateRelationValues(name, values, RelationSubsume)
}

func (g *EGraph) mutateRelationValues(name string, values []FunctionValue, kind RelationActionKind) (bool, error) {
	if g == nil {
		return false, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.relations == nil {
		return false, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return false, fmt.Errorf("egraph: rebuild before relation mutation: %w", err)
	}
	table, err := g.relation(name, len(values))
	if err != nil {
		return false, err
	}
	canonical, err := g.canonicalRelationValues(table, values)
	if err != nil {
		return false, err
	}
	key := typedFunctionKey(canonical)
	coreKind, err := coreFactActionKind(kind)
	if err != nil {
		return false, err
	}
	changed, _, err := g.facts.ApplyActionsDetailed([]egraphcore.ResolvedFactAction[Id]{
		{Kind: coreKind, Relation: name, Values: toCoreValues(canonical)},
	}, g.Find, nil)
	if err != nil {
		return false, fmt.Errorf("egraph: apply relation %q action: %w", name, err)
	}
	if len(changed) == 0 || !changed[0] {
		return false, nil
	}
	switch kind {
	case RelationInsert:
		g.logRelationInsertion(name, key)
	case RelationDelete:
		delete(table.subsumed, key)
	case RelationSubsume:
		table.subsumed[key] = struct{}{}
	}
	g.syncRelationView(table)
	g.revision++
	return true, nil
}

func classFunctionValues(ids []Id) []FunctionValue {
	values := make([]FunctionValue, len(ids))
	for i, id := range ids {
		values[i] = ClassValue(id)
	}
	return values
}

func coreFactActionKind(kind RelationActionKind) (egraphcore.FactActionKind, error) {
	switch kind {
	case RelationInsert:
		return egraphcore.FactInsert, nil
	case RelationDelete:
		return egraphcore.FactDelete, nil
	case RelationSubsume:
		return egraphcore.FactSubsume, nil
	default:
		return 0, fmt.Errorf("egraph: invalid relation action kind %d", kind)
	}
}

func coreValueKind(kind FunctionColumnKind) egraphcore.ValueKind { return egraphcore.ValueKind(kind) }

func toCoreValues(values []FunctionValue) []egraphcore.Value[Id] {
	out := make([]egraphcore.Value[Id], len(values))
	for i, value := range values {
		out[i] = toCoreValue(value)
	}
	return out
}

func toCoreValue(value FunctionValue) egraphcore.Value[Id] {
	switch value.Kind {
	case FunctionClassColumn:
		return egraphcore.Class(value.ID)
	case FunctionInt64Column:
		return egraphcore.Int64[Id](value.Int64)
	case FunctionBoolColumn:
		return egraphcore.Bool[Id](value.Bool)
	case FunctionFloat64Column:
		return egraphcore.Float64[Id](value.Float64)
	case FunctionStringColumn:
		return egraphcore.String[Id](value.String)
	case FunctionUnitColumn:
		return egraphcore.Unit[Id]()
	case FunctionBigIntColumn:
		return egraphcore.BigInt[Id](value.BigInt)
	case FunctionBigRatColumn:
		return egraphcore.BigRat[Id](value.BigRat)
	default:
		return egraphcore.Value[Id]{}
	}
}

func fromCoreValues(values []egraphcore.Value[Id]) []FunctionValue {
	out := make([]FunctionValue, len(values))
	for i, value := range values {
		switch value.Kind {
		case egraphcore.ClassValue:
			out[i] = ClassValue(value.Class)
		case egraphcore.Int64Value:
			out[i] = Int64Value(value.Int64)
		case egraphcore.BoolValue:
			out[i] = BoolValue(value.Bool)
		case egraphcore.Float64Value:
			out[i] = Float64Value(math.Float64frombits(value.Float))
		case egraphcore.StringValue:
			out[i] = StringValue(value.String)
		case egraphcore.UnitValue:
			out[i] = UnitValue()
		case egraphcore.BigIntValue:
			out[i] = FunctionValue{Kind: FunctionBigIntColumn, BigInt: value.String}
		case egraphcore.BigRatValue:
			out[i] = FunctionValue{Kind: FunctionBigRatColumn, BigRat: value.String}
		}
	}
	return out
}

// syncRelationView keeps Apple's existing matcher inputs derived from the
// shared relation after each mutation or class canonicalization.
func (g *EGraph) syncRelationView(table *relationTable) {
	table.rows = make(map[string][]FunctionValue)
	table.index = make(functionValueIndex, len(table.schema.Columns))
	for _, row := range table.core.Rows() {
		values := fromCoreValues(row)
		key := typedFunctionKey(values)
		table.rows[key] = values
		indexTypedRelationRow(table, values, key)
	}
}

// logRelationInsertion records successful insertions only while a saturation
// schedule is collecting deltas. The log is scoped to that schedule.
func (g *EGraph) logRelationInsertion(name, key string) {
	if !g.relationLogsEnabled {
		return
	}
	if g.relationInsertions == nil {
		g.relationInsertions = make(map[string][]string)
	}
	g.relationInsertions[name] = append(g.relationInsertions[name], key)
}

func indexTypedRelationRow(table *relationTable, row []FunctionValue, key string) {
	for i, value := range row {
		if table.index[i] == nil {
			table.index[i] = make(map[egraphcore.Value[Id]]map[string]struct{})
		}
		indexValue := toCoreValue(value)
		if table.index[i][indexValue] == nil {
			table.index[i][indexValue] = make(map[string]struct{})
		}
		table.index[i][indexValue][key] = struct{}{}
	}
}

// RelationValues returns sorted copies of all canonical mixed rows.
func (g *EGraph) RelationValues(name string) ([]TypedRelationRow, error) {
	if g == nil {
		return nil, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.relations == nil {
		return nil, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("egraph: rebuild before reading relation rows: %w", err)
	}
	table := g.relations[name]
	if table == nil {
		return nil, fmt.Errorf("egraph: unknown relation %q", name)
	}
	typedRows := make([][]FunctionValue, 0)
	for _, row := range table.core.Rows() {
		typedRows = append(typedRows, fromCoreValues(row))
	}
	sort.Slice(typedRows, func(i, j int) bool { return typedFunctionKey(typedRows[i]) < typedFunctionKey(typedRows[j]) })
	rows := make([]TypedRelationRow, 0, len(typedRows))
	for _, values := range typedRows {
		key := typedFunctionKey(values)
		rows = append(rows, TypedRelationRow{Columns: values, Subsumed: hasRelationSubsumed(table, key)})
	}
	return rows, nil
}
func hasRelationSubsumed(table *relationTable, key string) bool {
	_, ok := table.subsumed[key]
	return ok
}

func relationKey(tuple []Id) string {
	var key strings.Builder
	for _, id := range tuple {
		fmt.Fprintf(&key, "%d;", id)
	}
	return key.String()
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
