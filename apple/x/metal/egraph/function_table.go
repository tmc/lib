package egraph

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"sort"
	"strings"

	"github.com/tmc/lib/egraphcore"
)

// FunctionMergePolicy resolves writes with the same key and rows that collide
// after e-class IDs are canonicalized.
type FunctionMergePolicy uint8

const (
	FunctionKeepOld FunctionMergePolicy = iota + 1
	FunctionUseNew
	FunctionUnion
	FunctionAssertEqual
	FunctionConst
	FunctionMinInt64
	FunctionMaxInt64
	FunctionMergeExpression
)

// FunctionMergeExprKind identifies a node in a function merge expression.
type FunctionMergeExprKind uint8

const (
	FunctionMergeOldValue FunctionMergeExprKind = iota + 1
	FunctionMergeNewValue
	FunctionMergeConstant
	FunctionMergePrimitiveExpr
	FunctionMergeLookupExpr
	FunctionMergeCallExpr
)

// FunctionMergeOp identifies a pure built-in operation in a merge, action,
// computed leaf, or primitive condition expression.
type FunctionMergeOp uint8

const (
	FunctionMergeInt64Add FunctionMergeOp = iota + 1
	FunctionMergeInt64Sub
	FunctionMergeInt64Mul
	FunctionMergeInt64Div
	FunctionMergeInt64Mod
	FunctionMergeInt64And
	FunctionMergeInt64Min
	FunctionMergeInt64Max
	FunctionMergeInt64Equal
	FunctionMergeInt64NotEqual
	FunctionMergeInt64Less
	FunctionMergeInt64LessEqual
	FunctionMergeInt64Greater
	FunctionMergeInt64GreaterEqual
	FunctionMergeBoolAnd
	FunctionMergeBoolOr
	FunctionMergeBoolNot
	FunctionMergeFloat64Add
	FunctionMergeFloat64Sub
	FunctionMergeFloat64Mul
	FunctionMergeFloat64Div
	FunctionMergeFloat64Equal
	FunctionMergeFloat64NotEqual
	FunctionMergeFloat64Less
	FunctionMergeFloat64LessEqual
	FunctionMergeFloat64Greater
	FunctionMergeFloat64GreaterEqual
	FunctionMergeValueEqual
	FunctionMergeValueNotEqual
	FunctionMergeInt64Or
	FunctionMergeInt64Xor
	FunctionMergeInt64ShiftLeft
	FunctionMergeInt64ShiftRight
	FunctionMergeInt64Not
	FunctionMergeInt64Log2
	FunctionMergeInt64ToFloat64
	FunctionMergeInt64ToString
	FunctionMergeStringCountMatches
	FunctionMergeBoolXor
	FunctionMergeBoolImplies
	FunctionMergeFloat64Mod
	FunctionMergeFloat64Pow
	FunctionMergeFloat64Neg
	FunctionMergeFloat64Abs
	FunctionMergeFloat64Min
	FunctionMergeFloat64Max
	FunctionMergeFloat64ToInt64
	FunctionMergeStringConcat
	FunctionMergeStringReplace
)

// FunctionMergeExpr is an immutable expression evaluated when a function row
// conflicts with an existing row. Its fields are private; use the constructors
// below to build an expression tree.
type FunctionMergeExpr struct {
	kind  FunctionMergeExprKind
	value FunctionValue
	op    FunctionMergeOp
	name  string
	args  []FunctionMergeExpr
}

// FunctionMergeOld returns the existing result value.
func FunctionMergeOld() FunctionMergeExpr {
	return FunctionMergeExpr{kind: FunctionMergeOldValue}
}

// FunctionMergeNew returns the incoming result value.
func FunctionMergeNew() FunctionMergeExpr {
	return FunctionMergeExpr{kind: FunctionMergeNewValue}
}

// FunctionMergeConst returns value as a constant expression.
func FunctionMergeConst(value FunctionValue) FunctionMergeExpr {
	return FunctionMergeExpr{kind: FunctionMergeConstant, value: value}
}

// FunctionMergePrimitive builds a pure built-in operation. It copies args so
// later changes to the caller's slice cannot change the expression.
func FunctionMergePrimitive(op FunctionMergeOp, args ...FunctionMergeExpr) FunctionMergeExpr {
	return FunctionMergeExpr{kind: FunctionMergePrimitiveExpr, op: op, args: cloneFunctionMergeExprs(args)}
}

// FunctionMergeLookup reads a result from another registered function table.
func FunctionMergeLookup(target string, args ...FunctionMergeExpr) FunctionMergeExpr {
	return FunctionMergeExpr{kind: FunctionMergeLookupExpr, name: target, args: cloneFunctionMergeExprs(args)}
}

// FunctionMergeCall builds a call to a registered pure typed merge operator.
func FunctionMergeCall(name string, args ...FunctionMergeExpr) FunctionMergeExpr {
	return FunctionMergeExpr{kind: FunctionMergeCallExpr, name: name, args: cloneFunctionMergeExprs(args)}
}

// FunctionMergeEvaluator evaluates one registered pure typed merge operator.
// It must be deterministic, must not mutate the graph or other state, and must
// return a value matching the operator's declared result type.
type FunctionMergeEvaluator func(args []FunctionValue) (FunctionValue, error)

// FunctionMergeOperator declares a pure typed operation available to function
// merge expressions. The graph copies Arguments during registration.
type FunctionMergeOperator struct {
	Arguments []FunctionColumnType
	Result    FunctionColumnType
	Evaluate  FunctionMergeEvaluator
}

// RegisterFunctionMergeOperator registers a named pure operation for typed
// function merge expressions. Operators are immutable after registration.
func (g *EGraph) RegisterFunctionMergeOperator(name string, operator FunctionMergeOperator) error {
	if g == nil {
		return fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.functionMergeOps == nil {
		return fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if name == "" {
		return fmt.Errorf("egraph: function merge operator name must be non-empty")
	}
	if operator.Evaluate == nil {
		return fmt.Errorf("egraph: function merge operator %q has no evaluator", name)
	}
	if operator.Result.Kind < FunctionClassColumn || operator.Result.Kind > FunctionBigRatColumn {
		return fmt.Errorf("egraph: function merge operator %q has invalid result kind %d", name, operator.Result.Kind)
	}
	if err := validateSortExpression(operator.Result.Sort); err != nil {
		return fmt.Errorf("egraph: function merge operator %q result sort: %w", name, err)
	}
	if operator.Result.Kind != FunctionClassColumn && operator.Result.Sort != "" {
		return fmt.Errorf("egraph: function merge operator %q primitive result has an e-class sort", name)
	}
	for i, argument := range operator.Arguments {
		if argument.Kind < FunctionClassColumn || argument.Kind > FunctionBigRatColumn {
			return fmt.Errorf("egraph: function merge operator %q argument %d has invalid kind %d", name, i, argument.Kind)
		}
		if err := validateSortExpression(argument.Sort); err != nil {
			return fmt.Errorf("egraph: function merge operator %q argument %d sort: %w", name, i, err)
		}
		if argument.Kind != FunctionClassColumn && argument.Sort != "" {
			return fmt.Errorf("egraph: function merge operator %q primitive argument %d has an e-class sort", name, i)
		}
	}
	if g.declaredSortsStrict {
		if err := validateDeclaredClassColumn(g, operator.Result, fmt.Sprintf("function merge operator %q result", name)); err != nil {
			return fmt.Errorf("egraph: %w", err)
		}
		for i, argument := range operator.Arguments {
			if err := validateDeclaredClassColumn(g, argument, fmt.Sprintf("function merge operator %q argument %d", name, i)); err != nil {
				return fmt.Errorf("egraph: %w", err)
			}
		}
	}
	if _, exists := g.functionMergeOps[name]; exists {
		return fmt.Errorf("egraph: function merge operator %q is already registered", name)
	}
	operator.Arguments = append([]FunctionColumnType(nil), operator.Arguments...)
	g.functionMergeOps[name] = operator
	return nil
}

func cloneFunctionMergeExprs(exprs []FunctionMergeExpr) []FunctionMergeExpr {
	if exprs == nil {
		return nil
	}
	cloned := make([]FunctionMergeExpr, len(exprs))
	for i, expr := range exprs {
		expr.args = cloneFunctionMergeExprs(expr.args)
		cloned[i] = expr
	}
	return cloned
}

func equalFunctionMergeExpr(a, b *FunctionMergeExpr) bool {
	if a == nil || b == nil {
		return a == b
	}
	if a.kind != b.kind || a.value != b.value || a.op != b.op || a.name != b.name || len(a.args) != len(b.args) {
		return false
	}
	for i := range a.args {
		if !equalFunctionMergeExpr(&a.args[i], &b.args[i]) {
			return false
		}
	}
	return true
}

// FunctionDefaultKind selects what LookupFunction does for a missing key.
type FunctionDefaultKind uint8

const (
	FunctionNoDefault FunctionDefaultKind = iota
	FunctionDefaultFail
	FunctionDefaultConst
	FunctionDefaultFreshID
)

// FunctionDefault configures lookup behavior for absent keys. Const uses Value;
// FreshID creates a unique leaf with the function's ValueSort.
type FunctionDefault struct {
	Kind       FunctionDefaultKind
	Value      Id
	TypedValue FunctionValue
}

// FunctionColumnKind identifies an e-class or primitive function-table column.
type FunctionColumnKind uint8

const (
	FunctionClassColumn FunctionColumnKind = iota + 1
	FunctionInt64Column
	FunctionBoolColumn
	FunctionFloat64Column
	FunctionStringColumn
	FunctionUnitColumn
	FunctionBigIntColumn
	FunctionBigRatColumn
)

// FunctionColumnType describes one function key or result column. Class columns
// use Sort; primitive columns use their corresponding FunctionValue field.
type FunctionColumnType struct {
	Kind FunctionColumnKind
	Sort string
}

// FunctionValue is one typed function-table cell. Only the field selected by
// Kind is meaningful. BigInt and BigRat contain canonical decimal and
// numerator/denominator text, respectively, so values remain comparable.
type FunctionValue struct {
	Kind    FunctionColumnKind
	ID      Id
	Int64   int64
	Bool    bool
	Float64 float64
	String  string
	BigInt  string
	BigRat  string
}

// ClassValue returns a function value containing an e-class ID.
func ClassValue(id Id) FunctionValue { return FunctionValue{Kind: FunctionClassColumn, ID: id} }

// Int64Value returns a function value containing a primitive i64.
func Int64Value(value int64) FunctionValue {
	return FunctionValue{Kind: FunctionInt64Column, Int64: value}
}

// BoolValue returns a function value containing a primitive bool.
func BoolValue(value bool) FunctionValue { return FunctionValue{Kind: FunctionBoolColumn, Bool: value} }

// Float64Value returns a function value containing a primitive float64.
func Float64Value(value float64) FunctionValue {
	return FunctionValue{Kind: FunctionFloat64Column, Float64: value}
}

// StringValue returns a function value containing a primitive string.
func StringValue(value string) FunctionValue {
	return FunctionValue{Kind: FunctionStringColumn, String: value}
}

// UnitValue returns a function value containing the unit value.
func UnitValue() FunctionValue { return FunctionValue{Kind: FunctionUnitColumn} }

// BigIntValue returns an immutable copy of value as an exact integer value.
// A nil value produces an invalid value that typed-column validation rejects.
func BigIntValue(value *big.Int) FunctionValue {
	if value == nil {
		return FunctionValue{Kind: FunctionBigIntColumn}
	}
	return FunctionValue{Kind: FunctionBigIntColumn, BigInt: value.String()}
}

// BigRatValue returns an immutable copy of value as an exact rational value.
// A nil value produces an invalid value that typed-column validation rejects.
func BigRatValue(value *big.Rat) FunctionValue {
	if value == nil {
		return FunctionValue{Kind: FunctionBigRatColumn}
	}
	return FunctionValue{Kind: FunctionBigRatColumn, BigRat: value.RatString()}
}

// ClassColumn declares an e-class column with an optional required sort.
func ClassColumn(sort string) FunctionColumnType {
	return FunctionColumnType{Kind: FunctionClassColumn, Sort: sort}
}

// Int64Column declares a primitive i64 column.
func Int64Column() FunctionColumnType { return FunctionColumnType{Kind: FunctionInt64Column} }

// BoolColumn declares a primitive bool column.
func BoolColumn() FunctionColumnType { return FunctionColumnType{Kind: FunctionBoolColumn} }

// Float64Column declares a primitive float64 column.
func Float64Column() FunctionColumnType { return FunctionColumnType{Kind: FunctionFloat64Column} }

// StringColumn declares a primitive string column.
func StringColumn() FunctionColumnType { return FunctionColumnType{Kind: FunctionStringColumn} }

// UnitColumn declares a unit-valued column.
func UnitColumn() FunctionColumnType { return FunctionColumnType{Kind: FunctionUnitColumn} }

// BigIntColumn declares an exact arbitrary-precision integer column.
func BigIntColumn() FunctionColumnType { return FunctionColumnType{Kind: FunctionBigIntColumn} }

// BigRatColumn declares an exact arbitrary-precision rational column.
func BigRatColumn() FunctionColumnType { return FunctionColumnType{Kind: FunctionBigRatColumn} }

// ErrFunctionDefault reports a missing key in a function configured to fail.
var ErrFunctionDefault = errors.New("egraph: function has no default")

// FunctionSchema declares a keyed function table. KeySorts and ValueSort
// declare e-class columns; KeyTypes and ValueType can instead declare typed
// class, i64, bool, float64, string, unit, BigInt, and BigRat columns. Empty
// sorts accept any e-class sort.
// FunctionKeepOld retains the earliest
// write; FunctionUseNew keeps the latest; FunctionUnion merges result
// e-classes; FunctionAssertEqual errors when values differ. Write order also
// resolves KeepOld and UseNew rows whose keys collide after canonicalization.
// FunctionConst selects MergeValue on conflicts. FunctionMinInt64 and
// FunctionMaxInt64 select the smaller or larger i64 leaf value.
// FunctionMergeExpression evaluates MergeExpr, a typed expression built with
// FunctionMergeOld, FunctionMergeNew, FunctionMergeConst, and the pure
// FunctionMergePrimitive operations or registered FunctionMergeCall
// operators. Union requires a non-empty ValueSort.
type FunctionSchema struct {
	Name       string
	KeySorts   []string
	ValueSort  string
	Merge      FunctionMergePolicy
	MergeValue Id
	Default    FunctionDefault
	CanSubsume bool
	// KeyTypes and ValueType opt into typed columns. When omitted, KeySorts and
	// ValueSort retain their original all-e-class meaning.
	KeyTypes  []FunctionColumnType
	ValueType FunctionColumnType
	// MergeTypedValue is used by FunctionConst for a typed result column.
	MergeTypedValue FunctionValue
	// MergeExpr is evaluated on conflicting values when Merge is
	// FunctionMergeExpression. The expression is copied during registration.
	MergeExpr *FunctionMergeExpr
	typed     bool
}

type functionTable struct {
	schema FunctionSchema
	rows   *egraphcore.OrderedTable[Id, functionTableRow]
}

type functionTableRow struct {
	legacy functionRow
	typed  typedFunctionRow
}

func cloneFunctionTableRow(row functionTableRow) functionTableRow {
	row.legacy.keys = append([]Id(nil), row.legacy.keys...)
	row.typed.keys = append([]FunctionValue(nil), row.typed.keys...)
	return row
}

func newFunctionTable(schema FunctionSchema) *functionTable {
	width := len(schema.KeySorts) + 1
	if schema.typed {
		width = len(schema.KeyTypes) + 1
	}
	return &functionTable{schema: schema, rows: egraphcore.NewOrderedTableWithClone[Id, functionTableRow](width, cloneFunctionTableRow)}
}

func functionCoreIDs(ids []Id) []egraphcore.Value[Id] {
	values := make([]egraphcore.Value[Id], len(ids))
	for i, id := range ids {
		values[i] = egraphcore.Class(id)
	}
	return values
}

func (t *functionTable) getLegacy(key string) (functionRow, bool) {
	row, ok := t.rows.Get(key)
	row.Payload.legacy.order = row.Order
	row.Payload.legacy.subsumed = row.Subsumed
	return row.Payload.legacy, ok && !t.schema.typed
}

func (t *functionTable) getTyped(key string) (typedFunctionRow, bool) {
	row, ok := t.rows.Get(key)
	row.Payload.typed.order = row.Order
	row.Payload.typed.subsumed = row.Subsumed
	return row.Payload.typed, ok && t.schema.typed
}

func (t *functionTable) putLegacy(key string, row functionRow) {
	subsumed := row.subsumed
	if current, ok := t.rows.Get(key); ok {
		subsumed = subsumed || current.Subsumed
	}
	order := row.order
	row.order = 0
	row.subsumed = false
	t.rows.PutAt(key, functionCoreIDs(functionRowTuple(row)), functionTableRow{legacy: row}, order, subsumed)
}

func (t *functionTable) putTyped(key string, row typedFunctionRow) {
	subsumed := row.subsumed
	if current, ok := t.rows.Get(key); ok {
		subsumed = subsumed || current.Subsumed
	}
	order := row.order
	row.order = 0
	row.subsumed = false
	t.rows.PutAt(key, toCoreValues(typedFunctionTuple(row)), functionTableRow{typed: row}, order, subsumed)
}

func (t *functionTable) counters() (uint64, uint64) { return t.rows.Counters() }

func (t *functionTable) setCounters(order, fresh uint64) { t.rows.SetCounters(order, fresh) }

func (t *functionTable) takeOrder() uint64 {
	order, fresh := t.rows.Counters()
	t.rows.SetCounters(order+1, fresh)
	return order
}

func (t *functionTable) freshIndex() uint64 {
	_, fresh := t.rows.Counters()
	return fresh
}

func (t *functionTable) advanceFresh() {
	order, fresh := t.rows.Counters()
	t.rows.SetCounters(order, fresh+1)
}

type typedFunctionRow struct {
	keys     []FunctionValue
	value    FunctionValue
	order    uint64
	subsumed bool
}

// functionValueIndex remains the compatibility index for relation snapshots.
// Keyed function rows use egraphcore.ValueIndex directly.
type functionValueIndex []map[egraphcore.Value[Id]]map[string]struct{}

type functionRow struct {
	keys     []Id
	value    Id
	order    uint64
	subsumed bool
}

// FunctionRow is a copy of one canonical function-table row.
type FunctionRow struct {
	Keys     []Id
	Value    Id
	Subsumed bool
}

func normalizeFunctionSchema(schema FunctionSchema) (FunctionSchema, error) {
	typed := schema.KeyTypes != nil || schema.ValueType.Kind != 0 || schema.MergeTypedValue.Kind != 0 || schema.Default.TypedValue.Kind != 0
	if schema.MergeExpr != nil {
		cloned := *schema.MergeExpr
		cloned.args = cloneFunctionMergeExprs(cloned.args)
		schema.MergeExpr = &cloned
	}
	if !typed {
		schema.KeyTypes = make([]FunctionColumnType, len(schema.KeySorts))
		for i, sort := range schema.KeySorts {
			schema.KeyTypes[i] = ClassColumn(sort)
		}
		schema.ValueType = ClassColumn(schema.ValueSort)
		schema.typed = false
		return schema, nil
	}
	if schema.KeyTypes == nil {
		schema.KeyTypes = make([]FunctionColumnType, len(schema.KeySorts))
		for i, sort := range schema.KeySorts {
			schema.KeyTypes[i] = ClassColumn(sort)
		}
	}
	if len(schema.KeySorts) != 0 && len(schema.KeySorts) != len(schema.KeyTypes) {
		return schema, fmt.Errorf("egraph: function %q key type count does not match key sort count", schema.Name)
	}
	keySorts := make([]string, len(schema.KeyTypes))
	for i, typ := range schema.KeyTypes {
		if typ.Kind < FunctionClassColumn || typ.Kind > FunctionBigRatColumn {
			return schema, fmt.Errorf("egraph: function %q key %d has invalid column kind %d", schema.Name, i, typ.Kind)
		}
		if typ.Kind == FunctionClassColumn {
			if len(schema.KeySorts) != 0 && schema.KeySorts[i] != "" && schema.KeySorts[i] != typ.Sort {
				return schema, fmt.Errorf("egraph: function %q key %d has conflicting sort declarations", schema.Name, i)
			}
			keySorts[i] = typ.Sort
		} else if len(schema.KeySorts) != 0 && schema.KeySorts[i] != "" {
			return schema, fmt.Errorf("egraph: function %q primitive key %d cannot have an e-class sort", schema.Name, i)
		}
	}
	schema.KeySorts = keySorts
	if schema.ValueType.Kind == 0 {
		schema.ValueType = ClassColumn(schema.ValueSort)
	}
	if schema.ValueType.Kind < FunctionClassColumn || schema.ValueType.Kind > FunctionBigRatColumn {
		return schema, fmt.Errorf("egraph: function %q has invalid result column kind %d", schema.Name, schema.ValueType.Kind)
	}
	if schema.ValueType.Kind == FunctionClassColumn {
		if schema.ValueSort != "" && schema.ValueSort != schema.ValueType.Sort {
			return schema, fmt.Errorf("egraph: function %q has conflicting result sort declarations", schema.Name)
		}
		schema.ValueSort = schema.ValueType.Sort
	} else if schema.ValueSort != "" {
		return schema, fmt.Errorf("egraph: function %q primitive result cannot have an e-class sort", schema.Name)
	}
	schema.typed = schema.ValueType.Kind != FunctionClassColumn
	for _, typ := range schema.KeyTypes {
		schema.typed = schema.typed || typ.Kind != FunctionClassColumn
	}
	return schema, nil
}

func equalFunctionColumnTypes(a, b []FunctionColumnType) bool {
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

func (g *EGraph) validateFunctionMergeExpr(expr *FunctionMergeExpr, resultType FunctionColumnType) error {
	typ, err := g.functionMergeExprType(expr, resultType)
	if err != nil {
		return err
	}
	if typ.Kind != resultType.Kind || typ.Kind == FunctionClassColumn && resultType.Sort != "" && typ.Sort != resultType.Sort {
		return fmt.Errorf("result type %v does not match function result type %v", typ, resultType)
	}
	return nil
}

func (g *EGraph) functionMergeExprType(expr *FunctionMergeExpr, resultType FunctionColumnType) (FunctionColumnType, error) {
	if expr == nil {
		return FunctionColumnType{}, fmt.Errorf("nil expression")
	}
	switch expr.kind {
	case FunctionMergeOldValue, FunctionMergeNewValue:
		return resultType, nil
	case FunctionMergeConstant:
		constantType := FunctionColumnType{Kind: expr.value.Kind}
		if constantType.Kind == FunctionClassColumn {
			root, err := g.FindChecked(expr.value.ID)
			if err != nil {
				return FunctionColumnType{}, fmt.Errorf("merge expression constant: %w", err)
			}
			class := g.store.Class(root)
			if class == nil {
				return FunctionColumnType{}, fmt.Errorf("merge expression constant references missing class %d", root)
			}
			constantType.Sort = class.Sort
		}
		if _, err := g.canonicalTypedValue(constantType, expr.value, "merge expression constant"); err != nil {
			return FunctionColumnType{}, err
		}
		return constantType, nil
	case FunctionMergeLookupExpr:
		target := g.functions[expr.name]
		if target == nil {
			return FunctionColumnType{}, fmt.Errorf("lookup target %q is not registered", expr.name)
		}
		if len(expr.args) != len(target.schema.KeyTypes) {
			return FunctionColumnType{}, fmt.Errorf("lookup target %q expects %d key columns, got %d", expr.name, len(target.schema.KeyTypes), len(expr.args))
		}
		for i := range expr.args {
			argType, err := g.functionMergeExprType(&expr.args[i], resultType)
			if err != nil {
				return FunctionColumnType{}, fmt.Errorf("lookup target %q argument %d: %w", expr.name, i, err)
			}
			want := target.schema.KeyTypes[i]
			if argType.Kind != want.Kind || want.Kind == FunctionClassColumn && argType.Sort != "" && want.Sort != "" && argType.Sort != want.Sort {
				return FunctionColumnType{}, fmt.Errorf("lookup target %q argument %d has type %v, want %v", expr.name, i, argType, want)
			}
		}
		return target.schema.ValueType, nil
	case FunctionMergeCallExpr:
		operator, ok := g.functionMergeOps[expr.name]
		if !ok {
			return FunctionColumnType{}, fmt.Errorf("merge operator %q is not registered", expr.name)
		}
		if len(expr.args) != len(operator.Arguments) {
			return FunctionColumnType{}, fmt.Errorf("merge operator %q expects %d arguments, got %d", expr.name, len(operator.Arguments), len(expr.args))
		}
		for i, argument := range expr.args {
			got, err := g.functionMergeExprType(&argument, resultType)
			if err != nil {
				return FunctionColumnType{}, fmt.Errorf("merge operator %q argument %d: %w", expr.name, i, err)
			}
			want := operator.Arguments[i]
			if got.Kind != want.Kind || want.Kind == FunctionClassColumn && got.Sort != "" && want.Sort != "" && got.Sort != want.Sort {
				return FunctionColumnType{}, fmt.Errorf("merge operator %q argument %d has type %v, want %v", expr.name, i, got, want)
			}
		}
		return operator.Result, nil
	case FunctionMergePrimitiveExpr:
		argument, resultKind, arity, err := functionPrimitiveSignature(expr.op)
		if err != nil {
			return FunctionColumnType{}, err
		}
		result := FunctionColumnType{Kind: resultKind}
		if arity >= 0 && len(expr.args) != arity {
			return FunctionColumnType{}, fmt.Errorf("primitive operation %d expects %d arguments, got %d", expr.op, arity, len(expr.args))
		}
		if expr.op == FunctionMergeValueEqual || expr.op == FunctionMergeValueNotEqual {
			leftType, err := g.functionMergeExprType(&expr.args[0], resultType)
			if err != nil {
				return FunctionColumnType{}, fmt.Errorf("argument 0: %w", err)
			}
			rightType, err := g.functionMergeExprType(&expr.args[1], resultType)
			if err != nil {
				return FunctionColumnType{}, fmt.Errorf("argument 1: %w", err)
			}
			if !primitiveEqualityKind(leftType.Kind) || leftType.Kind != rightType.Kind || leftType.Kind == FunctionClassColumn && leftType.Sort != rightType.Sort {
				return FunctionColumnType{}, fmt.Errorf("value equality requires arguments with the same primitive type, got %v and %v", leftType, rightType)
			}
			return result, nil
		}
		for i := range expr.args {
			typ, err := g.functionMergeExprType(&expr.args[i], resultType)
			if err != nil {
				return FunctionColumnType{}, fmt.Errorf("argument %d: %w", i, err)
			}
			if argument != 0 && typ.Kind != argument {
				return FunctionColumnType{}, fmt.Errorf("argument %d requires column kind %d, got %d", i, argument, typ.Kind)
			}
		}
		return result, nil
	default:
		return FunctionColumnType{}, fmt.Errorf("invalid expression kind %d", expr.kind)
	}
}

// TypedFunctionRow is a copy of one canonical typed function-table row.
type TypedFunctionRow struct {
	Keys     []FunctionValue
	Value    FunctionValue
	Subsumed bool
}

// FunctionChange reports the effect of inserting or merging a function row.
type FunctionChange uint8

const (
	FunctionUnchanged FunctionChange = iota
	FunctionInserted
	FunctionUpdated
)

// RegisterFunction declares a keyed function table. Re-registering an
// identical schema is a no-op.
func (g *EGraph) RegisterFunction(schema FunctionSchema) error {
	if g == nil {
		return fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.functions == nil {
		return fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return fmt.Errorf("egraph: rebuild before function registration: %w", err)
	}
	if schema.Name == "" {
		return fmt.Errorf("egraph: empty function name")
	}
	if owner, ok := g.variantOwners[schema.Name]; ok {
		return fmt.Errorf("egraph: function %q is the constructor of datatype %q", schema.Name, owner)
	}
	var err error
	schema, err = normalizeFunctionSchema(schema)
	if err != nil {
		return err
	}
	if g.declaredSortsStrict {
		for i, typ := range schema.KeyTypes {
			if err := validateDeclaredClassColumn(g, typ, fmt.Sprintf("function %q key %d", schema.Name, i)); err != nil {
				return fmt.Errorf("egraph: %w", err)
			}
		}
		if err := validateDeclaredClassColumn(g, schema.ValueType, fmt.Sprintf("function %q value", schema.Name)); err != nil {
			return fmt.Errorf("egraph: %w", err)
		}
	}
	if g.relations[schema.Name] != nil {
		return fmt.Errorf("egraph: name %q is already registered as a relation", schema.Name)
	}
	if schema.Merge != FunctionKeepOld && schema.Merge != FunctionUseNew && schema.Merge != FunctionUnion && schema.Merge != FunctionAssertEqual && schema.Merge != FunctionConst && schema.Merge != FunctionMinInt64 && schema.Merge != FunctionMaxInt64 && schema.Merge != FunctionMergeExpression {
		return fmt.Errorf("egraph: function %q has invalid merge policy", schema.Name)
	}
	if schema.Merge == FunctionMergeExpression {
		if schema.MergeExpr == nil {
			return fmt.Errorf("egraph: function %q merge expression is missing", schema.Name)
		}
		var targets = make(map[string]struct{})
		collectFunctionMergeLookups(schema.MergeExpr, targets)
		for targetName := range targets {
			if g.functions[targetName] == nil {
				return fmt.Errorf("egraph: function %q merge lookup target %q is not registered", schema.Name, targetName)
			}
		}
		if err := g.validateFunctionMergeExpr(schema.MergeExpr, schema.ValueType); err != nil {
			return fmt.Errorf("egraph: function %q merge expression: %w", schema.Name, err)
		}
	} else if schema.MergeExpr != nil {
		return fmt.Errorf("egraph: function %q has a merge expression without the expression merge policy", schema.Name)
	}
	if schema.Merge == FunctionUnion && schema.ValueType.Kind != FunctionClassColumn {
		return fmt.Errorf("egraph: function %q union merge requires a value sort", schema.Name)
	}
	if (schema.Merge == FunctionMinInt64 || schema.Merge == FunctionMaxInt64) && schema.ValueType.Kind != FunctionInt64Column && (schema.ValueType.Kind != FunctionClassColumn || schema.ValueSort != "i64") {
		return fmt.Errorf("egraph: function %q integer min/max merge requires an i64 result", schema.Name)
	}
	if schema.Merge == FunctionConst {
		if schema.typed {
			if _, err := g.canonicalTypedValue(schema.ValueType, schema.MergeTypedValue, "constant merge"); err != nil {
				return err
			}
		} else {
			if schema.MergeValue == InvalidId {
				return fmt.Errorf("egraph: function %q constant merge has invalid class ID", schema.Name)
			}
			if _, _, err := g.canonicalFunctionRow(&functionTable{schema: schema}, nil, schema.MergeValue); err != nil {
				return fmt.Errorf("egraph: function %q has invalid constant merge value: %w", schema.Name, err)
			}
		}
	}
	switch schema.Default.Kind {
	case FunctionNoDefault, FunctionDefaultFail:
	case FunctionDefaultConst:
		if schema.typed {
			if _, err := g.canonicalTypedValue(schema.ValueType, schema.Default.TypedValue, "constant default"); err != nil {
				return err
			}
		} else {
			if schema.Default.Value == InvalidId {
				return fmt.Errorf("egraph: function %q constant default has invalid class ID", schema.Name)
			}
			if _, _, err := g.canonicalFunctionRow(&functionTable{schema: schema}, nil, schema.Default.Value); err != nil {
				return fmt.Errorf("egraph: function %q has invalid constant default: %w", schema.Name, err)
			}
		}
	case FunctionDefaultFreshID:
		if schema.ValueType.Kind != FunctionClassColumn || schema.ValueSort == "" {
			return fmt.Errorf("egraph: function %q fresh default requires a value sort", schema.Name)
		}
		if schema.Merge == FunctionMinInt64 || schema.Merge == FunctionMaxInt64 {
			return fmt.Errorf("egraph: function %q integer min/max merge cannot use a fresh-ID default", schema.Name)
		}
	default:
		return fmt.Errorf("egraph: function %q has invalid default kind %d", schema.Name, schema.Default.Kind)
	}
	if current := g.functions[schema.Name]; current != nil {
		if !equalStrings(current.schema.KeySorts, schema.KeySorts) || current.schema.ValueSort != schema.ValueSort || current.schema.Merge != schema.Merge || current.schema.MergeValue != schema.MergeValue || current.schema.Default != schema.Default || current.schema.CanSubsume != schema.CanSubsume || !equalFunctionColumnTypes(current.schema.KeyTypes, schema.KeyTypes) || current.schema.ValueType != schema.ValueType || current.schema.MergeTypedValue != schema.MergeTypedValue || !equalFunctionMergeExpr(current.schema.MergeExpr, schema.MergeExpr) {
			return fmt.Errorf("egraph: function %q is already registered with a different schema", schema.Name)
		}
		return nil
	}
	schema.KeySorts = append([]string(nil), schema.KeySorts...)
	schema.KeyTypes = append([]FunctionColumnType(nil), schema.KeyTypes...)
	g.functions[schema.Name] = newFunctionTable(schema)
	g.revision++
	return nil
}

// SubsumeFunction inserts the default row if needed, then hides it from
// function-pattern matching while retaining it for inspection. The function
// schema must permit subsumption. It returns false if the row is already hidden.
func (g *EGraph) SubsumeFunction(name string, keys ...Id) (bool, error) {
	if g == nil {
		return false, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.functions == nil {
		return false, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return false, fmt.Errorf("egraph: rebuild before function subsumption: %w", err)
	}
	table, err := g.function(name, len(keys))
	if err != nil {
		return false, err
	}
	if table.schema.typed {
		values := make([]FunctionValue, len(keys))
		for i, id := range keys {
			values[i] = ClassValue(id)
		}
		return g.subsumeTypedFunction(name, values)
	}
	if !table.schema.CanSubsume {
		return false, fmt.Errorf("egraph: function %q does not permit subsumption", name)
	}
	canonical, _, err := g.canonicalFunctionRow(table, keys, InvalidId)
	if err != nil {
		return false, err
	}
	key := relationKey(canonical)
	if row, ok := table.getLegacy(key); ok && row.subsumed {
		return false, nil
	}
	if _, ok := table.getLegacy(key); !ok {
		_, _, err := g.LookupFunction(name, canonical...)
		if err != nil {
			return false, err
		}
	}
	if _, ok := table.getLegacy(key); !ok {
		return false, fmt.Errorf("egraph: function %q default did not insert a row for subsumption", name)
	}
	row, _ := table.getLegacy(key)
	row.order = table.takeOrder()
	row.subsumed = true
	table.putLegacy(key, row)
	table.rows.MarkSubsumed(key)
	g.revision++
	return true, nil
}

func (g *EGraph) subsumeFunctionForRewrite(name string, keys ...Id) (bool, error) {
	return g.SubsumeFunction(name, keys...)
}

func (g *EGraph) subsumeTypedFunctionForRewrite(name string, keys []FunctionValue) (bool, error) {
	return g.SubsumeFunctionValues(name, keys...)
}

// SubsumeFunctionValues inserts a default row if needed, then hides a typed
// function row from pattern matching while retaining its value for inspection.
func (g *EGraph) SubsumeFunctionValues(name string, keys ...FunctionValue) (bool, error) {
	return g.subsumeTypedFunction(name, keys)
}

func (g *EGraph) subsumeTypedFunction(name string, keys []FunctionValue) (bool, error) {
	if g == nil || !g.store.Initialized() || g.functions == nil {
		return false, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return false, fmt.Errorf("egraph: rebuild before function subsumption: %w", err)
	}
	table, err := g.function(name, len(keys))
	if err != nil {
		return false, err
	}
	if !table.schema.typed {
		return false, fmt.Errorf("egraph: function %q has only e-class columns; use SubsumeFunction", name)
	}
	if !table.schema.CanSubsume {
		return false, fmt.Errorf("egraph: function %q does not permit subsumption", name)
	}
	canonical, err := g.canonicalTypedKeys(table, keys)
	if err != nil {
		return false, err
	}
	key := typedFunctionKey(canonical)
	if row, ok := table.getTyped(key); ok && row.subsumed {
		return false, nil
	}
	if _, ok := table.getTyped(key); !ok {
		if _, _, err := g.LookupFunctionValues(name, canonical...); err != nil {
			return false, err
		}
	}
	row, ok := table.getTyped(key)
	if !ok {
		return false, fmt.Errorf("egraph: function %q default did not insert a row for subsumption", name)
	}
	row.subsumed = true
	row.order = table.takeOrder()
	table.putTyped(key, row)
	table.rows.MarkSubsumed(key)
	g.revision++
	return true, nil
}

// SetFunction inserts a key/value row or resolves an existing row according
// to the table's merge policy. IDs are canonicalized before storage. Under
// FunctionUnion, the result union and row update may be committed even when
// the subsequent rebuild reports a conflict in another function table; the
// returned change and error report that partial success.
func (g *EGraph) SetFunction(name string, keys []Id, value Id) (FunctionChange, error) {
	if g == nil {
		return FunctionUnchanged, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.functions == nil {
		return FunctionUnchanged, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return FunctionUnchanged, fmt.Errorf("egraph: rebuild before function set: %w", err)
	}
	table, err := g.function(name, len(keys))
	if err != nil {
		return FunctionUnchanged, err
	}
	if table.schema.typed {
		return FunctionUnchanged, fmt.Errorf("egraph: function %q has primitive columns; use SetFunctionValues", name)
	}
	canonical, result, err := g.canonicalFunctionRow(table, keys, value)
	if err != nil {
		return FunctionUnchanged, err
	}
	key := relationKey(canonical)
	old, ok := table.getLegacy(key)
	if !ok {
		row := functionRow{keys: canonical, value: result, order: table.takeOrder()}
		table.putLegacy(key, row)
		g.revision++
		return FunctionInserted, nil
	}
	if old.value == result && table.schema.Merge != FunctionMergeExpression {
		return FunctionUnchanged, nil
	}
	if table.schema.Merge == FunctionMergeExpression {
		merged, err := g.evaluateFunctionMergeExpr(table, ClassValue(old.value), ClassValue(result))
		if err != nil {
			return FunctionUnchanged, fmt.Errorf("egraph: function %q merge expression: %w", name, err)
		}
		merged, err = g.canonicalTypedValue(table.schema.ValueType, merged, "merge expression result")
		if err != nil {
			return FunctionUnchanged, err
		}
		if merged.ID == old.value {
			return FunctionUnchanged, nil
		}
		row := functionRow{keys: canonical, value: merged.ID, order: table.takeOrder(), subsumed: old.subsumed}
		table.putLegacy(key, row)
		g.revision++
		return FunctionUpdated, nil
	}
	if table.schema.Merge == FunctionAssertEqual {
		return FunctionUnchanged, fmt.Errorf("egraph: function %q key already has a different result", name)
	}
	if table.schema.Merge == FunctionConst {
		constant := g.Find(table.schema.MergeValue)
		if old.value == constant {
			return FunctionUnchanged, nil
		}
		row := functionRow{keys: canonical, value: constant, order: old.order}
		table.putLegacy(key, row)
		g.revision++
		return FunctionUpdated, nil
	}
	if table.schema.Merge == FunctionUnion {
		_, merged, err := g.UnionChecked(old.value, result)
		if err != nil {
			return FunctionUnchanged, fmt.Errorf("egraph: merge function %q result: %w", name, err)
		}
		if !merged {
			return FunctionUnchanged, nil
		}
		table.putLegacy(key, functionRow{keys: canonical, value: g.Find(old.value), order: old.order})
		g.revision++
		if err := g.RebuildChecked(); err != nil {
			return FunctionUpdated, fmt.Errorf("egraph: rebuild after function union: %w", err)
		}
		return FunctionUpdated, nil
	}
	if table.schema.Merge == FunctionMinInt64 || table.schema.Merge == FunctionMaxInt64 {
		merged, err := g.mergeInt64FunctionValues(table, old.value, result)
		if err != nil {
			return FunctionUnchanged, err
		}
		if merged == old.value {
			return FunctionUnchanged, nil
		}
		row := functionRow{keys: canonical, value: merged, order: old.order}
		table.putLegacy(key, row)
		g.revision++
		return FunctionUpdated, nil
	}
	if table.schema.Merge == FunctionKeepOld {
		return FunctionUnchanged, nil
	}
	row := functionRow{keys: canonical, value: result, order: table.takeOrder()}
	table.putLegacy(key, row)
	g.revision++
	return FunctionUpdated, nil
}

// DeleteFunction removes the row with keys. Deleting an absent row is harmless.
func (g *EGraph) DeleteFunction(name string, keys ...Id) (bool, error) {
	return g.deleteFunction(name, false, keys...)
}

func (g *EGraph) deleteFunctionForRewrite(name string, keys ...Id) (bool, error) {
	return g.deleteFunction(name, false, keys...)
}

func (g *EGraph) deleteFunction(name string, preserveSubsumption bool, keys ...Id) (bool, error) {
	if g == nil {
		return false, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.functions == nil {
		return false, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	table, err := g.function(name, len(keys))
	if err != nil {
		return false, err
	}
	if table.schema.typed {
		return false, fmt.Errorf("egraph: function %q has primitive columns; use DeleteFunctionValues", name)
	}
	rebuildErr := g.RebuildChecked()
	canonical, _, err := g.canonicalFunctionRow(table, keys, InvalidId)
	if err != nil {
		return false, err
	}
	key := relationKey(canonical)
	deleted := false
	for _, stored := range table.rows.Rows() {
		storedKey, row := stored.Key, stored.Payload.legacy
		storedKeys := make([]Id, len(row.keys))
		for i, id := range row.keys {
			storedKeys[i] = g.Find(id)
		}
		if relationKey(storedKeys) == key {
			table.rows.Delete(storedKey)
			deleted = true
		}
	}
	if deleted {
		g.revision++
		if rebuildErr != nil {
			if err := g.RebuildChecked(); err != nil {
				return true, fmt.Errorf("egraph: delete applied but rebuild still fails: %w", err)
			}
		}
		return true, nil
	}
	if rebuildErr != nil {
		return false, fmt.Errorf("egraph: rebuild before function delete: %w", rebuildErr)
	}
	return false, nil
}

// LookupFunction returns the value for keys. On a missing key, a configured
// constant or fresh-ID default inserts a row; a fail default returns
// ErrFunctionDefault. With no configured default it returns InvalidId, false,
// nil without changing the table.
func (g *EGraph) LookupFunction(name string, keys ...Id) (Id, bool, error) {
	if g == nil {
		return InvalidId, false, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.functions == nil {
		return InvalidId, false, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return InvalidId, false, fmt.Errorf("egraph: rebuild before function lookup: %w", err)
	}
	table, err := g.function(name, len(keys))
	if err != nil {
		return InvalidId, false, err
	}
	if table.schema.typed {
		return InvalidId, false, fmt.Errorf("egraph: function %q has primitive columns; use LookupFunctionValues", name)
	}
	canonical, _, err := g.canonicalFunctionRow(table, keys, InvalidId)
	if err != nil {
		return InvalidId, false, err
	}
	row, ok := table.getLegacy(relationKey(canonical))
	if ok {
		return g.Find(row.value), true, nil
	}
	switch table.schema.Default.Kind {
	case FunctionNoDefault:
		return InvalidId, false, nil
	case FunctionDefaultFail:
		return InvalidId, false, fmt.Errorf("egraph: function %q: %w", name, ErrFunctionDefault)
	case FunctionDefaultConst:
		if _, err := g.SetFunction(name, canonical, table.schema.Default.Value); err != nil {
			return InvalidId, false, err
		}
	case FunctionDefaultFreshID:
		value := g.add(ENode{
			Op:   "egraph.function-default",
			Sort: table.schema.ValueSort,
			Data: functionDefaultNode{name: name, index: table.freshIndex()},
		})
		if value == InvalidId {
			return InvalidId, false, fmt.Errorf("egraph: create fresh default for function %q: class ID space exhausted", name)
		}
		table.advanceFresh()
		if _, err := g.SetFunction(name, canonical, value); err != nil {
			return InvalidId, false, err
		}
	default:
		return InvalidId, false, fmt.Errorf("egraph: function %q has invalid default kind %d", name, table.schema.Default.Kind)
	}
	value, ok, err := g.LookupFunction(name, canonical...)
	return value, ok, err
}

type functionDefaultNode struct {
	name  string
	index uint64
}

// FunctionRows returns all rows, including subsumed rows, sorted by their
// canonical key tuple. If rebuild fails on an AssertEqual conflict, it returns
// the current rows together with the error so callers can identify rows to delete.
func (g *EGraph) FunctionRows(name string) ([][]Id, error) {
	if g == nil {
		return nil, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.functions == nil {
		return nil, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	table := g.functions[name]
	if table == nil {
		return nil, fmt.Errorf("egraph: unknown function %q", name)
	}
	if table.schema.typed {
		return nil, fmt.Errorf("egraph: function %q has primitive columns; use FunctionValues", name)
	}
	if err := g.RebuildChecked(); err != nil {
		return functionRowsSnapshot(g, table), fmt.Errorf("egraph: rebuild before reading function rows: %w", err)
	}
	return functionRowsSnapshot(g, table), nil
}

// FunctionRowsWithStatus returns canonical function rows, including whether
// each row is subsumed. Rows are sorted by canonical key and then value.
func (g *EGraph) FunctionRowsWithStatus(name string) ([]FunctionRow, error) {
	if g == nil {
		return nil, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.functions == nil {
		return nil, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	table := g.functions[name]
	if table == nil {
		return nil, fmt.Errorf("egraph: unknown function %q", name)
	}
	if table.schema.typed {
		return nil, fmt.Errorf("egraph: function %q has primitive columns; use FunctionValues", name)
	}
	if err := g.RebuildChecked(); err != nil {
		return functionRowsWithStatusSnapshot(g, table), fmt.Errorf("egraph: rebuild before reading function rows: %w", err)
	}
	return functionRowsWithStatusSnapshot(g, table), nil
}

func functionRowsWithStatusSnapshot(g *EGraph, table *functionTable) []FunctionRow {
	storedRows := table.rows.Rows()
	keys := make([]string, 0, len(storedRows))
	rowByKey := make(map[string]functionRow, len(storedRows))
	for _, stored := range storedRows {
		keys = append(keys, stored.Key)
		row := stored.Payload.legacy
		row.subsumed = stored.Subsumed
		rowByKey[stored.Key] = row
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := rowByKey[keys[i]], rowByKey[keys[j]]
		for k := range a.keys {
			ak, bk := g.Find(a.keys[k]), g.Find(b.keys[k])
			if ak != bk {
				return ak < bk
			}
		}
		return g.Find(a.value) < g.Find(b.value)
	})
	rows := make([]FunctionRow, 0, len(keys))
	for _, key := range keys {
		row := rowByKey[key]
		canonicalKeys := make([]Id, len(row.keys))
		for i, id := range row.keys {
			canonicalKeys[i] = g.Find(id)
		}
		rows = append(rows, FunctionRow{Keys: canonicalKeys, Value: g.Find(row.value), Subsumed: row.subsumed})
	}
	return rows
}

func functionRowsSnapshot(g *EGraph, table *functionTable) [][]Id {
	keys := make([]string, 0, table.rows.Len())
	rowKeys := make(map[string]string, table.rows.Len())
	rowValues := make(map[string]functionRow, table.rows.Len())
	for _, stored := range table.rows.Rows() {
		storedKey, row := stored.Key, stored.Payload.legacy
		rowValues[storedKey] = row
		canonical := make([]Id, len(row.keys))
		for i, id := range row.keys {
			canonical[i] = g.Find(id)
		}
		key := relationKey(canonical)
		keys = append(keys, storedKey)
		rowKeys[storedKey] = key
	}
	sort.Slice(keys, func(i, j int) bool {
		ki, kj := rowKeys[keys[i]], rowKeys[keys[j]]
		if ki != kj {
			return ki < kj
		}
		return g.Find(rowValues[keys[i]].value) < g.Find(rowValues[keys[j]].value)
	})
	rows := make([][]Id, 0, len(keys))
	for _, storedKey := range keys {
		row := rowValues[storedKey]
		values := make([]Id, len(row.keys)+1)
		for i, id := range row.keys {
			values[i] = g.Find(id)
		}
		values[len(row.keys)] = g.Find(row.value)
		rows = append(rows, values)
	}
	return rows
}

func (g *EGraph) function(name string, arity int) (*functionTable, error) {
	table := g.functions[name]
	if table == nil {
		return nil, fmt.Errorf("egraph: unknown function %q", name)
	}
	if len(table.schema.KeySorts) != arity {
		return nil, fmt.Errorf("egraph: function %q expects %d key columns, got %d", name, len(table.schema.KeySorts), arity)
	}
	return table, nil
}

func (g *EGraph) canonicalFunctionRow(table *functionTable, keys []Id, value Id) ([]Id, Id, error) {
	if len(keys) != len(table.schema.KeySorts) {
		return nil, InvalidId, fmt.Errorf("egraph: function %q expects %d key columns, got %d", table.schema.Name, len(table.schema.KeySorts), len(keys))
	}
	canonical := make([]Id, len(keys))
	for i, id := range keys {
		root, err := g.FindChecked(id)
		if err != nil {
			return nil, InvalidId, fmt.Errorf("egraph: function %q key %d: %w", table.schema.Name, i, err)
		}
		class := g.store.Class(root)
		if class == nil {
			return nil, InvalidId, fmt.Errorf("egraph: function %q key %d references missing class %d", table.schema.Name, i, root)
		}
		if table.schema.KeySorts[i] != "" && class.Sort != table.schema.KeySorts[i] {
			return nil, InvalidId, fmt.Errorf("egraph: function %q key %d requires sort %q, got %q", table.schema.Name, i, table.schema.KeySorts[i], class.Sort)
		}
		canonical[i] = root
	}
	if value == InvalidId {
		return canonical, value, nil
	}
	root, err := g.FindChecked(value)
	if err != nil {
		return nil, InvalidId, fmt.Errorf("egraph: function %q result: %w", table.schema.Name, err)
	}
	class := g.store.Class(root)
	if class == nil {
		return nil, InvalidId, fmt.Errorf("egraph: function %q result references missing class %d", table.schema.Name, root)
	}
	if table.schema.ValueSort != "" && class.Sort != table.schema.ValueSort {
		return nil, InvalidId, fmt.Errorf("egraph: function %q result requires sort %q, got %q", table.schema.Name, table.schema.ValueSort, class.Sort)
	}
	if table.schema.Merge == FunctionMinInt64 || table.schema.Merge == FunctionMaxInt64 {
		if _, err := g.int64FunctionValue(table, root); err != nil {
			return nil, InvalidId, err
		}
	}
	return canonical, root, nil
}

func (g *EGraph) canonicalizeFunctions() (bool, error) {
	names := make([]string, 0, len(g.functions))
	for name := range g.functions {
		names = append(names, name)
	}
	sort.Strings(names)
	stageOrder, err := g.functionStageOrder(names)
	if err != nil {
		return false, err
	}
	// Apply all FunctionUnion constraints before checking AssertEqual tables.
	// A result union can make a separate assertion succeed; returning an
	// assertion error first would depend on table-name ordering.
	for _, name := range names {
		table := g.functions[name]
		if table.schema.typed {
			continue
		}
		if table.schema.Merge != FunctionUnion {
			continue
		}
		rows := canonicalRows(g, table)
		keys := make([]string, 0, len(rows))
		for key := range rows {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			group := rows[key]
			sort.Slice(group, func(i, j int) bool { return group[i].order < group[j].order })
			for i := 1; i < len(group); i++ {
				if group[0].value != group[i].value {
					if _, merged, err := g.UnionChecked(group[0].value, group[i].value); err != nil {
						return false, fmt.Errorf("egraph: union function %q results: %w", name, err)
					} else if merged {
						return true, nil
					}
				}
			}
		}
	}
	mergedTyped, err := g.unionTypedFunctionResults(names)
	if err != nil {
		return false, err
	}
	if mergedTyped {
		return true, nil
	}
	staged := make(map[*functionTable]stagedFunctionRows, len(names))
	view := functionMergeView{tables: make(map[string]stagedFunctionRows, len(names)), fresh: newFunctionFreshStage()}
	for _, name := range stageOrder {
		table := g.functions[name]
		if table.schema.typed {
			rows, err := g.canonicalTypedFunctionRowsWithView(table, view)
			if err != nil {
				return false, err
			}
			result := stagedFunctionRows{rows: rows, typed: true, privateRows: true}
			staged[table] = result
			view.tables[name] = result
			continue
		}
		canonical := canonicalRows(g, table)
		resultTable := newFunctionTable(table.schema)
		keys := make([]string, 0, len(canonical))
		for key := range canonical {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			group := canonical[key]
			sort.Slice(group, func(i, j int) bool { return group[i].order < group[j].order })
			hidden := false
			for _, row := range group {
				hidden = hidden || row.subsumed
			}
			var current functionRow
			haveCurrent := false
			for _, row := range group {
				if haveCurrent {
					merged, unionChanged, err := g.mergeCanonicalFunctionRowsWithView(table, current, row, view)
					if err != nil {
						return false, fmt.Errorf("egraph: function %q key %s: %w", name, key, err)
					}
					if unionChanged {
						return true, nil
					}
					current = merged
					continue
				}
				current = row
				haveCurrent = true
			}
			current.subsumed = hidden
			resultTable.putLegacy(key, current)
			if hidden {
				resultTable.rows.MarkSubsumed(key)
			}
		}
		order, fresh := table.counters()
		resultTable.setCounters(order, fresh)
		result := stagedFunctionRows{rows: resultTable.rows, privateRows: true}
		staged[table] = result
		view.tables[name] = result
	}
	// Lookup defaults update the staged view while dependent tables evaluate.
	// Row ownership and counters travel together in each staged OrderedTable.
	for _, name := range stageOrder {
		result := staged[g.functions[name]]
		viewResult := view.tables[name]
		result.rows = viewResult.rows
		result.privateRows = viewResult.privateRows
		result.defaultsInserted = viewResult.defaultsInserted
		staged[g.functions[name]] = result
	}
	if err := g.commitFunctionFreshStage(view.fresh); err != nil {
		return false, err
	}
	var defaultsInserted uint64
	for table, result := range staged {
		defaultsInserted += result.defaultsInserted
		table.rows.ReplaceFrom(result.rows)
	}
	if defaultsInserted != 0 {
		g.revision += defaultsInserted
	}
	return defaultsInserted != 0 || len(view.fresh.nodes) != 0, nil
}

type stagedFunctionRows struct {
	rows             *egraphcore.OrderedTable[Id, functionTableRow]
	typed            bool
	privateRows      bool
	defaultsInserted uint64
}

func (s stagedFunctionRows) getLegacy(key string) (functionRow, bool) {
	row, ok := s.rows.Get(key)
	row.Payload.legacy.order = row.Order
	row.Payload.legacy.subsumed = row.Subsumed
	return row.Payload.legacy, ok && !s.typed
}

func (s stagedFunctionRows) getTyped(key string) (typedFunctionRow, bool) {
	row, ok := s.rows.Get(key)
	row.Payload.typed.order = row.Order
	row.Payload.typed.subsumed = row.Subsumed
	return row.Payload.typed, ok && s.typed
}

func (s stagedFunctionRows) putLegacy(key string, row functionRow) {
	subsumed := row.subsumed
	if current, ok := s.rows.Get(key); ok {
		subsumed = subsumed || current.Subsumed
	}
	order := row.order
	row.order = 0
	row.subsumed = false
	s.rows.PutAt(key, functionCoreIDs(functionRowTuple(row)), functionTableRow{legacy: row}, order, subsumed)
}

func (s stagedFunctionRows) putTyped(key string, row typedFunctionRow) {
	subsumed := row.subsumed
	if current, ok := s.rows.Get(key); ok {
		subsumed = subsumed || current.Subsumed
	}
	order := row.order
	row.order = 0
	row.subsumed = false
	s.rows.PutAt(key, toCoreValues(typedFunctionTuple(row)), functionTableRow{typed: row}, order, subsumed)
}

func (s stagedFunctionRows) counters() (uint64, uint64) { return s.rows.Counters() }

func (s stagedFunctionRows) takeOrder() uint64 {
	order, fresh := s.rows.Counters()
	s.rows.SetCounters(order+1, fresh)
	return order
}

func (s stagedFunctionRows) takeFresh() uint64 {
	order, fresh := s.rows.Counters()
	s.rows.SetCounters(order, fresh+1)
	return fresh
}

type functionMergeView struct {
	tables map[string]stagedFunctionRows
	fresh  *functionFreshStage
}

// functionActionBatch buffers function mutations produced by one rule set.
// Lookups used to materialize subsumption defaults read from view, which is a
// snapshot of the function tables taken before the batch starts.
type functionActionBatch struct {
	view             functionMergeView
	ops              []functionActionOp
	predicted        map[string]FunctionValue
	prepared         bool
	deletesCommitted bool
	deleted          int
}

type functionActionOp struct {
	kind        FunctionActionKind
	name        string
	keys        []FunctionValue
	value       FunctionValue
	row         FunctionValue
	hasRow      bool
	rowSubsumed bool
}

func classValues(ids []Id) []FunctionValue {
	values := make([]FunctionValue, len(ids))
	for i, id := range ids {
		values[i] = ClassValue(id)
	}
	return values
}

func ptrFunctionValue(value FunctionValue) *FunctionValue { return &value }

func newFunctionActionBatch(g *EGraph) *functionActionBatch {
	return &functionActionBatch{view: newFunctionMergeView(g), predicted: make(map[string]FunctionValue)}
}

func (b *functionActionBatch) stage(g *EGraph, kind FunctionActionKind, name string, keys []FunctionValue, value *FunctionValue) error {
	table := g.functions[name]
	if table == nil {
		return fmt.Errorf("egraph: unknown function %q", name)
	}
	canonical, err := g.canonicalTypedKeysWithView(b.view, table, keys)
	if err != nil {
		return err
	}
	op := functionActionOp{kind: kind, name: name, keys: canonical}
	if value != nil {
		op.value, err = g.canonicalTypedValueWithView(b.view, table.schema.ValueType, *value, "result")
		if err != nil {
			return err
		}
	}
	if kind == FunctionDeleteAction || kind == FunctionSubsumeAction {
		staged := b.view.tables[name]
		if !table.schema.typed {
			key := relationKey(idsFromFunctionValues(canonical))
			if row, ok := staged.getLegacy(key); ok {
				op.row, op.hasRow = ClassValue(row.value), true
				op.rowSubsumed = row.subsumed
			}
		} else {
			key := typedFunctionKey(canonical)
			if row, ok := staged.getTyped(key); ok {
				op.row, op.hasRow, op.rowSubsumed = row.value, true, row.subsumed
			}
		}
	}
	if kind == FunctionSubsumeAction {
		if !table.schema.CanSubsume {
			return fmt.Errorf("egraph: function %q does not permit subsumption", name)
		}
		if op.rowSubsumed {
			return nil
		}
		if !op.hasRow {
			predictionKey := name + "\x00" + typedFunctionKey(canonical)
			if predicted, ok := b.predicted[predictionKey]; ok {
				op.row = predicted
				op.hasRow = true
			} else {
				// Egglog's execution state predicts one missing-key default per
				// table and key, even when several buffered actions use it.
				switch table.schema.Default.Kind {
				case FunctionDefaultFreshID:
					staged := b.view.tables[name]
					op.row, err = g.freshFunctionDefaultWithView(b.view, table, &staged)
					if err == nil {
						b.view.tables[name] = staged
					}
				case FunctionDefaultConst:
					if table.schema.typed {
						op.row, err = g.canonicalTypedValueWithView(b.view, table.schema.ValueType, table.schema.Default.TypedValue, "lookup default")
					} else {
						op.row, err = g.canonicalTypedValueWithView(b.view, table.schema.ValueType, ClassValue(table.schema.Default.Value), "lookup default")
					}
				case FunctionDefaultFail:
					err = fmt.Errorf("%w", ErrFunctionDefault)
				case FunctionNoDefault:
					err = errPrimitiveNoValue
				default:
					err = fmt.Errorf("unsupported default kind %d", table.schema.Default.Kind)
				}
				if err != nil {
					return fmt.Errorf("egraph: function %q default for subsumption: %w", name, err)
				}
				op.hasRow = true
				b.predicted[predictionKey] = op.row
			}
		}
	}
	b.ops = append(b.ops, op)
	return nil
}

func (b *functionActionBatch) commit(g *EGraph) (inserts, updates, deletes, subsumptions int, err error) {
	if err := g.commitFunctionFreshStage(b.view.fresh); err != nil {
		return 0, 0, 0, 0, err
	}
	if err := b.commitDeletes(g); err != nil {
		return 0, 0, b.deleted, 0, err
	}
	deletes = b.deleted
	if err := b.commitDefaultRows(g); err != nil {
		return inserts, updates, deletes, subsumptions, err
	}
	for _, op := range b.ops {
		if op.kind == FunctionDeleteAction {
			continue
		}
		table := g.functions[op.name]
		var change FunctionChange
		if table.schema.typed {
			value := op.value
			if op.kind == FunctionSubsumeAction {
				value = op.row
			}
			change, err = g.SetFunctionValues(op.name, op.keys, value)
			if err == nil {
				keys, keyErr := g.canonicalTypedKeys(table, op.keys)
				if keyErr != nil {
					err = keyErr
				} else {
					key := typedFunctionKey(keys)
					row, _ := table.getTyped(key)
					if op.kind == FunctionSubsumeAction {
						if !row.subsumed {
							subsumptions++
							g.revision++
						}
						row.subsumed = true
						table.rows.MarkSubsumed(key)
					}
					table.putTyped(key, row)
				}
			}
		} else {
			value := op.value
			if op.kind == FunctionSubsumeAction {
				value = op.row
			}
			change, err = g.SetFunction(op.name, idsFromFunctionValues(op.keys), value.ID)
			if err == nil {
				keys, canonicalValue, keyErr := g.canonicalFunctionRow(table, idsFromFunctionValues(op.keys), value.ID)
				if keyErr != nil {
					err = keyErr
				} else {
					key := relationKey(keys)
					row, _ := table.getLegacy(key)
					if op.kind == FunctionSubsumeAction {
						if !row.subsumed {
							subsumptions++
							g.revision++
						}
						row.subsumed = true
						table.rows.MarkSubsumed(key)
					}
					row.value = canonicalValue
					table.putLegacy(key, row)
				}
			}
		}
		if err != nil {
			return inserts, updates, deletes, subsumptions, err
		}
		switch change {
		case FunctionInserted:
			inserts++
		case FunctionUpdated:
			updates++
		}
	}
	return inserts, updates, deletes, subsumptions, nil
}

// commitDeletes applies captured keys before unions can canonicalize them.
func (b *functionActionBatch) commitDeletes(g *EGraph) error {
	if b.deletesCommitted {
		return nil
	}
	for _, op := range b.ops {
		if op.kind != FunctionDeleteAction {
			continue
		}
		table := g.functions[op.name]
		if table.schema.typed {
			key := typedFunctionKey(op.keys)
			_, ok := table.getTyped(key)
			if !ok {
				continue
			}
			table.rows.Delete(key)
			b.deleted++
		} else {
			keys := idsFromFunctionValues(op.keys)
			key := relationKey(keys)
			_, ok := table.getLegacy(key)
			if !ok {
				continue
			}
			table.rows.Delete(key)
			b.deleted++
		}
		g.revision++
	}
	b.deletesCommitted = true
	return nil
}

func (b *functionActionBatch) commitDefaultRows(g *EGraph) error {
	for name, staged := range b.view.tables {
		if staged.defaultsInserted == 0 {
			continue
		}
		table := g.functions[name]
		if staged.typed {
			for _, stored := range staged.rows.Rows() {
				key, row := stored.Key, stored.Payload.typed
				row.order = stored.Order
				row.subsumed = stored.Subsumed
				if _, exists := table.getTyped(key); exists {
					continue
				}
				table.putTyped(key, row)
				if stored.Subsumed {
					table.rows.MarkSubsumed(key)
				}
				g.revision++
			}
		} else {
			for _, stored := range staged.rows.Rows() {
				key, row := stored.Key, stored.Payload.legacy
				row.order = stored.Order
				row.subsumed = stored.Subsumed
				if _, exists := table.getLegacy(key); exists {
					continue
				}
				table.putLegacy(key, row)
				if stored.Subsumed {
					table.rows.MarkSubsumed(key)
				}
				g.revision++
			}
		}
		order, fresh := staged.rows.Counters()
		table.setCounters(order, fresh)
	}
	return nil
}

func newFunctionMergeView(g *EGraph) functionMergeView {
	view := functionMergeView{tables: make(map[string]stagedFunctionRows, len(g.functions)), fresh: newFunctionFreshStage()}
	for name, target := range g.functions {
		staged := stagedFunctionRows{rows: target.rows.Clone(), typed: target.schema.typed, privateRows: true}
		view.tables[name] = staged
	}
	return view
}

func cloneFunctionMergeView(view functionMergeView) functionMergeView {
	cloned := functionMergeView{tables: make(map[string]stagedFunctionRows, len(view.tables)), fresh: newFunctionFreshStage()}
	if view.fresh != nil {
		cloned.fresh.nodes = append(cloned.fresh.nodes, view.fresh.nodes...)
		for id, node := range view.fresh.provisional {
			cloned.fresh.provisional[id] = node
		}
	}
	for name, staged := range view.tables {
		copyStaged := staged
		if staged.rows != nil {
			copyStaged.rows = staged.rows.Clone()
		}
		copyStaged.privateRows = true
		cloned.tables[name] = copyStaged
	}
	return cloned
}

func makeStagedFunctionRowsPrivate(staged *stagedFunctionRows) {
	if staged.privateRows {
		return
	}
	if staged.rows != nil {
		staged.rows = staged.rows.Clone()
	}
	staged.privateRows = true
}

func (g *EGraph) commitFunctionMergeView(view functionMergeView) error {
	if err := g.commitFunctionFreshStage(view.fresh); err != nil {
		return err
	}
	var inserted uint64
	for name, staged := range view.tables {
		if staged.defaultsInserted == 0 {
			continue
		}
		target := g.functions[name]
		target.rows.ReplaceFrom(staged.rows)
		inserted += staged.defaultsInserted
	}
	g.revision += inserted
	return nil
}

type functionFreshStage struct {
	nodes       []ENode
	provisional map[Id]ENode
}

func newFunctionFreshStage() *functionFreshStage {
	return &functionFreshStage{provisional: make(map[Id]ENode)}
}

func (s *functionFreshStage) reserve(g *EGraph, node ENode) (Id, error) {
	if uint64(g.store.Len())+uint64(len(s.nodes)) >= uint64(InvalidId) {
		return InvalidId, fmt.Errorf("egraph: class ID space exhausted")
	}
	id := Id(g.store.Len() + len(s.nodes))
	if g.hasFunctionFreshNode(node) {
		return InvalidId, fmt.Errorf("egraph: fresh default node unexpectedly already exists")
	}
	for _, existing := range s.nodes {
		if existing.Equal(node) {
			return InvalidId, fmt.Errorf("egraph: fresh default node unexpectedly duplicated")
		}
	}
	s.nodes = append(s.nodes, node)
	s.provisional[id] = node
	return id, nil
}

func (g *EGraph) hasFunctionFreshNode(node ENode) bool {
	canonical := g.canonicalize(node)
	found := false
	g.store.VisitMemoCandidates(canonical.hashKey(), func(id Id) bool {
		root := g.Find(id)
		class := g.store.Class(root)
		if class == nil {
			return true
		}
		for _, existing := range class.Nodes {
			if g.canonicalize(existing).Equal(canonical) {
				found = true
				return false
			}
		}
		return true
	})
	return found
}

func (g *EGraph) commitFunctionFreshStage(stage *functionFreshStage) error {
	if stage == nil || len(stage.nodes) == 0 {
		return nil
	}
	if uint64(g.store.Len())+uint64(len(stage.nodes)) > uint64(InvalidId) {
		return fmt.Errorf("egraph: class ID space exhausted")
	}
	keys := make([]string, len(stage.nodes))
	for i, node := range stage.nodes {
		plannedID := Id(g.store.Len() + i)
		if provisional, ok := stage.provisional[plannedID]; !ok || !provisional.Equal(node) {
			return fmt.Errorf("egraph: invalid provisional fresh class ID %d", plannedID)
		}
		canon := g.canonicalize(node)
		key := canon.hashKey()
		if g.hasFunctionFreshNode(canon) {
			return fmt.Errorf("egraph: fresh default node changed before commit")
		}
		keys[i] = key
	}
	for i, node := range stage.nodes {
		canon := g.canonicalize(node)
		key := keys[i]
		analysisData := g.makeAnalysisData(canon)
		id, ok := g.store.Insert(key, func(id Id) *EClass {
			return &EClass{
				ID:              id,
				Sort:            canon.Sort,
				Nodes:           []ENode{canon},
				AnalysisData:    analysisData,
				analysisDataSet: analysisData != nil,
				nodesByOpArity:  map[nodeIndexKey][]int{{op: canon.Op, arity: len(canon.Children)}: {0}},
			}
		})
		if !ok {
			return fmt.Errorf("egraph: class ID space exhausted")
		}
		g.revision++
		g.nodeCount++
		opKey := nodeIndexKey{op: canon.Op, arity: len(canon.Children)}
		g.opIndex[opKey] = append(g.opIndex[opKey], id)
	}
	return nil
}

func (g *EGraph) functionStageOrder(names []string) ([]string, error) {
	baseline := make([]string, 0, len(names))
	for _, name := range names {
		if !g.functions[name].schema.typed {
			baseline = append(baseline, name)
		}
	}
	for _, name := range names {
		if g.functions[name].schema.typed {
			baseline = append(baseline, name)
		}
	}
	position := make(map[string]int, len(names))
	for i, name := range baseline {
		position[name] = i
	}
	dependencies := make(map[string]map[string]struct{}, len(names))
	for _, name := range names {
		table := g.functions[name]
		if table.schema.Merge == FunctionMergeExpression {
			collectFunctionMergeLookups(table.schema.MergeExpr, dependenciesFor(dependencies, name))
		}
	}
	for dependent, targets := range dependencies {
		for target := range targets {
			if _, ok := position[target]; !ok {
				return nil, fmt.Errorf("egraph: function %q merge lookup target %q is not registered", dependent, target)
			}
		}
	}
	order := make([]string, 0, len(names))
	placed := make(map[string]bool, len(names))
	for len(order) < len(names) {
		selected := ""
		for _, candidate := range baseline {
			if placed[candidate] {
				continue
			}
			ready := true
			for target := range dependencies[candidate] {
				if !placed[target] {
					ready = false
					break
				}
			}
			if ready {
				selected = candidate
				break
			}
		}
		if selected == "" {
			return nil, fmt.Errorf("egraph: function merge lookup dependency cycle")
		}
		placed[selected] = true
		order = append(order, selected)
	}
	return order, nil
}

func dependenciesFor(all map[string]map[string]struct{}, name string) map[string]struct{} {
	if all[name] == nil {
		all[name] = make(map[string]struct{})
	}
	return all[name]
}

func collectFunctionMergeLookups(expr *FunctionMergeExpr, targets map[string]struct{}) {
	if expr == nil {
		return
	}
	if expr.kind == FunctionMergeLookupExpr {
		targets[expr.name] = struct{}{}
	}
	for i := range expr.args {
		collectFunctionMergeLookups(&expr.args[i], targets)
	}
}

func (g *EGraph) unionTypedFunctionResults(names []string) (bool, error) {
	for _, name := range names {
		table := g.functions[name]
		if !table.schema.typed || table.schema.Merge != FunctionUnion {
			continue
		}
		groups := make(map[string][]typedFunctionRow, table.rows.Len())
		for _, stored := range table.rows.Rows() {
			row := stored.Payload.typed
			row.order = stored.Order
			row.subsumed = stored.Subsumed
			keys, err := g.canonicalTypedKeys(table, row.keys)
			if err != nil {
				return false, err
			}
			value, err := g.canonicalTypedValue(table.schema.ValueType, row.value, "result")
			if err != nil {
				return false, err
			}
			row.keys, row.value = keys, value
			groups[typedFunctionKey(keys)] = append(groups[typedFunctionKey(keys)], row)
		}
		keys := make([]string, 0, len(groups))
		for key := range groups {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			group := groups[key]
			sort.Slice(group, func(i, j int) bool { return group[i].order < group[j].order })
			for i := 1; i < len(group); i++ {
				left, right := group[0].value, group[i].value
				if left.ID == right.ID {
					continue
				}
				if _, merged, err := g.UnionChecked(left.ID, right.ID); err != nil {
					return false, fmt.Errorf("egraph: function %q key %s: %w", name, key, err)
				} else if merged {
					return true, nil
				}
			}
		}
	}
	return false, nil
}

func (g *EGraph) mergeCanonicalFunctionRows(table *functionTable, current, incoming functionRow) (functionRow, bool, error) {
	return g.mergeCanonicalFunctionRowsWithView(table, current, incoming, functionMergeView{})
}

func (g *EGraph) mergeCanonicalFunctionRowsWithView(table *functionTable, current, incoming functionRow, view functionMergeView) (functionRow, bool, error) {
	if table.schema.Merge == FunctionMinInt64 || table.schema.Merge == FunctionMaxInt64 {
		merged, err := g.mergeInt64FunctionValues(table, current.value, incoming.value)
		if err != nil {
			return functionRow{}, false, err
		}
		return mergedFunctionRow(current, incoming, merged), false, nil
	}
	policy := egraphcore.FunctionMergePolicy(0)
	var constant Id
	var custom func(egraphcore.Value[Id], egraphcore.Value[Id]) (egraphcore.Value[Id], error)
	switch table.schema.Merge {
	case FunctionKeepOld:
		policy = egraphcore.FunctionMergeKeepOld
	case FunctionUseNew:
		policy = egraphcore.FunctionMergeUseNew
	case FunctionUnion:
		policy = egraphcore.FunctionMergeUnionClass
	case FunctionAssertEqual:
		policy = egraphcore.FunctionMergeAssertEqual
	case FunctionConst:
		policy = egraphcore.FunctionMergeConstant
		constant = g.Find(table.schema.MergeValue)
	case FunctionMergeExpression:
		policy = egraphcore.FunctionMergeCustom
		custom = func(old, incoming egraphcore.Value[Id]) (egraphcore.Value[Id], error) {
			merged, err := g.evaluateFunctionMergeExprWithView(table, fromCoreValues([]egraphcore.Value[Id]{old})[0], fromCoreValues([]egraphcore.Value[Id]{incoming})[0], view)
			if err != nil {
				return egraphcore.Value[Id]{}, err
			}
			merged, err = g.canonicalTypedValueWithView(view, table.schema.ValueType, merged, "merge expression result")
			if err != nil {
				return egraphcore.Value[Id]{}, err
			}
			return egraphcore.Class(merged.ID), nil
		}
	default:
		return functionRow{}, false, fmt.Errorf("invalid merge policy %d", table.schema.Merge)
	}
	var unionChanged bool
	merged, err := egraphcore.MergeFunctionValue(policy, egraphcore.Class[Id](current.value), egraphcore.Class[Id](incoming.value), egraphcore.Class[Id](constant), egraphcore.FunctionMergeCallbacks[Id]{
		Canonicalize: g.Find,
		Union: func(left, right Id) (Id, error) {
			root, changed, err := g.UnionChecked(left, right)
			unionChanged = changed
			return root, err
		},
		Custom: custom,
	})
	if err != nil {
		if errors.Is(err, egraphcore.ErrConflictingFunctionResults) {
			return functionRow{}, false, fmt.Errorf("conflicting results after canonicalization")
		}
		return functionRow{}, false, err
	}
	if unionChanged {
		return functionRow{}, true, nil
	}
	return mergedFunctionRow(current, incoming, merged.Class), false, nil
}

func mergedFunctionRow(current, incoming functionRow, value Id) functionRow {
	newSubsumed := current.subsumed || incoming.subsumed
	changed := value != current.value || newSubsumed != current.subsumed
	result := current
	result.value = value
	result.subsumed = newSubsumed
	if changed {
		result.order = incoming.order
	}
	return result
}

func functionRowTuple(row functionRow) []Id {
	tuple := make([]Id, len(row.keys)+1)
	copy(tuple, row.keys)
	tuple[len(row.keys)] = row.value
	return tuple
}

func (g *EGraph) mergeInt64FunctionValues(table *functionTable, old, new Id) (Id, error) {
	oldValue, err := g.int64FunctionValue(table, old)
	if err != nil {
		return InvalidId, err
	}
	newValue, err := g.int64FunctionValue(table, new)
	if err != nil {
		return InvalidId, err
	}
	if table.schema.Merge == FunctionMinInt64 && oldValue <= newValue || table.schema.Merge == FunctionMaxInt64 && oldValue >= newValue {
		return old, nil
	}
	return new, nil
}

func (g *EGraph) int64FunctionValue(table *functionTable, id Id) (int64, error) {
	root, err := g.FindChecked(id)
	if err != nil {
		return 0, fmt.Errorf("egraph: function %q merge value: %w", table.schema.Name, err)
	}
	class := g.store.Class(root)
	if class == nil || class.Sort != "i64" {
		return 0, fmt.Errorf("egraph: function %q merge value %d is not an i64 class", table.schema.Name, id)
	}
	var value int64
	found := false
	for _, node := range class.Nodes {
		if len(node.Children) != 0 {
			return 0, fmt.Errorf("egraph: function %q merge value %d is not a literal i64", table.schema.Name, id)
		}
		literal, ok := node.Data.(int64)
		if !ok {
			return 0, fmt.Errorf("egraph: function %q merge value %d is not a literal i64", table.schema.Name, id)
		}
		if found && literal != value {
			return 0, fmt.Errorf("egraph: function %q merge value %d has conflicting i64 literals", table.schema.Name, id)
		}
		value = literal
		found = true
	}
	if !found {
		return 0, fmt.Errorf("egraph: function %q merge value %d has no i64 literal", table.schema.Name, id)
	}
	return value, nil
}

func canonicalRows(g *EGraph, table *functionTable) map[string][]functionRow {
	rows := make(map[string][]functionRow, table.rows.Len())
	for _, stored := range table.rows.Rows() {
		original := stored.Payload.legacy
		original.order = stored.Order
		row := functionRow{keys: append([]Id(nil), original.keys...), value: g.Find(original.value), order: original.order, subsumed: stored.Subsumed}
		for i, id := range row.keys {
			row.keys[i] = g.Find(id)
		}
		key := relationKey(row.keys)
		rows[key] = append(rows[key], row)
	}
	return rows
}

// SetFunctionValues inserts or merges a row in a function with primitive
// columns. Class-valued columns are canonicalized before storage; primitive
// values are compared exactly.
func (g *EGraph) SetFunctionValues(name string, keys []FunctionValue, value FunctionValue) (FunctionChange, error) {
	if g == nil || !g.store.Initialized() || g.functions == nil {
		return FunctionUnchanged, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return FunctionUnchanged, fmt.Errorf("egraph: rebuild before function set: %w", err)
	}
	table, err := g.function(name, len(keys))
	if err != nil {
		return FunctionUnchanged, err
	}
	if !table.schema.typed {
		return FunctionUnchanged, fmt.Errorf("egraph: function %q has only e-class columns; use SetFunction", name)
	}
	if err := g.canonicalizeTypedFunctionTable(table); err != nil {
		return FunctionUnchanged, err
	}
	canonicalKeys, err := g.canonicalTypedKeys(table, keys)
	if err != nil {
		return FunctionUnchanged, err
	}
	canonicalValue, err := g.canonicalTypedValue(table.schema.ValueType, value, "result")
	if err != nil {
		return FunctionUnchanged, err
	}
	key := typedFunctionKey(canonicalKeys)
	old, exists := table.getTyped(key)
	if !exists {
		row := typedFunctionRow{keys: canonicalKeys, value: canonicalValue, order: table.takeOrder()}
		table.putTyped(key, row)
		g.revision++
		return FunctionInserted, nil
	}
	merged, err := g.mergeTypedFunctionValue(table, old.value, canonicalValue)
	if err != nil {
		return FunctionUnchanged, err
	}
	if merged == old.value {
		return FunctionUnchanged, nil
	}
	row := old
	row.value = merged
	if table.schema.Merge == FunctionUseNew || table.schema.Merge == FunctionMergeExpression {
		row.order = table.takeOrder()
	}
	table.putTyped(key, row)
	g.revision++
	if table.schema.Merge == FunctionUnion {
		if err := g.RebuildChecked(); err != nil {
			return FunctionUpdated, fmt.Errorf("egraph: rebuild after function union: %w", err)
		}
	}
	return FunctionUpdated, nil
}

// LookupFunctionValues looks up a typed function row. Its default behavior
// matches LookupFunction: absent rows return not-found, fail, or insert the
// configured constant/fresh class value.
func (g *EGraph) LookupFunctionValues(name string, keys ...FunctionValue) (FunctionValue, bool, error) {
	if g == nil || !g.store.Initialized() || g.functions == nil {
		return FunctionValue{}, false, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if err := g.RebuildChecked(); err != nil {
		return FunctionValue{}, false, fmt.Errorf("egraph: rebuild before function lookup: %w", err)
	}
	table, err := g.function(name, len(keys))
	if err != nil {
		return FunctionValue{}, false, err
	}
	if !table.schema.typed {
		return nilFunctionValue(), false, fmt.Errorf("egraph: function %q has only e-class columns; use LookupFunction", name)
	}
	if err := g.canonicalizeTypedFunctionTable(table); err != nil {
		return FunctionValue{}, false, err
	}
	canonical, err := g.canonicalTypedKeys(table, keys)
	if err != nil {
		return FunctionValue{}, false, err
	}
	if row, ok := table.getTyped(typedFunctionKey(canonical)); ok {
		value, err := g.canonicalTypedValue(table.schema.ValueType, row.value, "result")
		return value, true, err
	}
	switch table.schema.Default.Kind {
	case FunctionNoDefault:
		return FunctionValue{}, false, nil
	case FunctionDefaultFail:
		return FunctionValue{}, false, fmt.Errorf("egraph: function %q: %w", name, ErrFunctionDefault)
	case FunctionDefaultConst:
		_, err = g.SetFunctionValues(name, canonical, table.schema.Default.TypedValue)
	case FunctionDefaultFreshID:
		value := g.add(ENode{Op: "egraph.function-default", Sort: table.schema.ValueSort, Data: functionDefaultNode{name: name, index: table.freshIndex()}})
		if value == InvalidId {
			return FunctionValue{}, false, fmt.Errorf("egraph: create fresh default for function %q: class ID space exhausted", name)
		}
		table.advanceFresh()
		_, err = g.SetFunctionValues(name, canonical, ClassValue(value))
	default:
		return FunctionValue{}, false, fmt.Errorf("egraph: function %q has invalid default kind %d", name, table.schema.Default.Kind)
	}
	if err != nil {
		return FunctionValue{}, false, err
	}
	return g.LookupFunctionValues(name, canonical...)
}

// DeleteFunctionValues deletes the row for keys after canonicalizing only
// e-class key columns. Deleting an absent row is harmless.
func (g *EGraph) DeleteFunctionValues(name string, keys ...FunctionValue) (bool, error) {
	return g.deleteFunctionValues(name, false, keys...)
}

func (g *EGraph) deleteFunctionValuesForRewrite(name string, keys ...FunctionValue) (bool, error) {
	return g.deleteFunctionValues(name, false, keys...)
}

func (g *EGraph) deleteFunctionValues(name string, preserveSubsumption bool, keys ...FunctionValue) (bool, error) {
	if g == nil || !g.store.Initialized() || g.functions == nil {
		return false, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	table, err := g.function(name, len(keys))
	if err != nil {
		return false, err
	}
	if !table.schema.typed {
		return false, fmt.Errorf("egraph: function %q has only e-class columns; use DeleteFunction", name)
	}
	rebuildErr := g.RebuildChecked()
	canonical, err := g.canonicalTypedKeys(table, keys)
	if err != nil {
		return false, err
	}
	key := typedFunctionKey(canonical)
	var deleted bool
	for _, stored := range table.rows.Rows() {
		storedKey, row := stored.Key, stored.Payload.typed
		storedCanonical, err := g.canonicalTypedKeys(table, row.keys)
		if err != nil {
			return false, err
		}
		if typedFunctionKey(storedCanonical) != key {
			continue
		}
		table.rows.Delete(storedKey)
		deleted = true
	}
	if !deleted {
		if rebuildErr != nil {
			return false, fmt.Errorf("egraph: rebuild before function delete: %w", rebuildErr)
		}
		return false, nil
	}
	g.revision++
	if rebuildErr != nil {
		if err := g.RebuildChecked(); err != nil {
			return true, fmt.Errorf("egraph: delete applied but rebuild still fails: %w", err)
		}
	}
	return true, nil
}

// FunctionValues returns canonical typed rows sorted by their typed key and
// result. Rows include subsumed entries.
func (g *EGraph) FunctionValues(name string) ([]TypedFunctionRow, error) {
	if g == nil || !g.store.Initialized() || g.functions == nil {
		return nil, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	table := g.functions[name]
	if table == nil {
		return nil, fmt.Errorf("egraph: unknown function %q", name)
	}
	if !table.schema.typed {
		return nil, fmt.Errorf("egraph: function %q has only e-class columns; use FunctionRows", name)
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, err
	}
	if err := g.canonicalizeTypedFunctionTable(table); err != nil {
		return nil, err
	}
	storedRows := table.rows.Rows()
	keys := make([]string, 0, len(storedRows))
	rowByKey := make(map[string]typedFunctionRow, len(storedRows))
	for _, stored := range storedRows {
		keys = append(keys, stored.Key)
		row := stored.Payload.typed
		row.subsumed = stored.Subsumed
		rowByKey[stored.Key] = row
	}
	sort.Strings(keys)
	rows := make([]TypedFunctionRow, 0, len(keys))
	for _, key := range keys {
		row := rowByKey[key]
		hidden := row.subsumed
		rows = append(rows, TypedFunctionRow{Keys: append([]FunctionValue(nil), row.keys...), Value: row.value, Subsumed: hidden})
	}
	return rows, nil
}

func nilFunctionValue() FunctionValue { return FunctionValue{} }

func (g *EGraph) canonicalTypedKeys(table *functionTable, keys []FunctionValue) ([]FunctionValue, error) {
	if len(keys) != len(table.schema.KeyTypes) {
		return nil, fmt.Errorf("egraph: function %q expects %d key columns, got %d", table.schema.Name, len(table.schema.KeyTypes), len(keys))
	}
	canonical := make([]FunctionValue, len(keys))
	for i, value := range keys {
		column, err := g.canonicalTypedValue(table.schema.KeyTypes[i], value, fmt.Sprintf("key %d", i))
		if err != nil {
			return nil, err
		}
		canonical[i] = column
	}
	return canonical, nil
}

func (g *EGraph) canonicalTypedValue(typ FunctionColumnType, value FunctionValue, where string) (FunctionValue, error) {
	if value.Kind != typ.Kind {
		return FunctionValue{}, fmt.Errorf("egraph: function column requires kind %d, got %d", typ.Kind, value.Kind)
	}
	switch typ.Kind {
	case FunctionClassColumn:
		root, err := g.FindChecked(value.ID)
		if err != nil {
			return FunctionValue{}, fmt.Errorf("egraph: function %s: %w", where, err)
		}
		class := g.store.Class(root)
		if class == nil {
			return FunctionValue{}, fmt.Errorf("egraph: function %s references missing class %d", where, root)
		}
		if typ.Sort != "" && class.Sort != typ.Sort {
			return FunctionValue{}, fmt.Errorf("egraph: function %s requires sort %q, got %q", where, typ.Sort, class.Sort)
		}
		return ClassValue(root), nil
	case FunctionInt64Column:
		return Int64Value(value.Int64), nil
	case FunctionBoolColumn:
		return BoolValue(value.Bool), nil
	case FunctionFloat64Column:
		if math.IsNaN(value.Float64) {
			return FunctionValue{}, fmt.Errorf("egraph: function %s rejects NaN", where)
		}
		if value.Float64 == 0 {
			return Float64Value(0), nil
		}
		return Float64Value(value.Float64), nil
	case FunctionStringColumn:
		return StringValue(value.String), nil
	case FunctionBigIntColumn:
		integer, ok := new(big.Int).SetString(value.BigInt, 10)
		if !ok {
			return FunctionValue{}, fmt.Errorf("egraph: function %s has invalid BigInt value %q", where, value.BigInt)
		}
		return BigIntValue(integer), nil
	case FunctionBigRatColumn:
		rational, ok := new(big.Rat).SetString(value.BigRat)
		if !ok {
			return FunctionValue{}, fmt.Errorf("egraph: function %s has invalid BigRat value %q", where, value.BigRat)
		}
		return BigRatValue(rational), nil
	case FunctionUnitColumn:
		return UnitValue(), nil
	default:
		return FunctionValue{}, fmt.Errorf("egraph: function has invalid column kind %d", typ.Kind)
	}
}

func typedFunctionKey(values []FunctionValue) string {
	var b strings.Builder
	for _, value := range values {
		fmt.Fprintf(&b, "%d:", value.Kind)
		switch value.Kind {
		case FunctionClassColumn:
			fmt.Fprintf(&b, "%d;", value.ID)
		case FunctionInt64Column:
			fmt.Fprintf(&b, "%d;", value.Int64)
		case FunctionBoolColumn:
			if value.Bool {
				b.WriteString("1;")
			} else {
				b.WriteString("0;")
			}
		case FunctionFloat64Column:
			fmt.Fprintf(&b, "%016x;", math.Float64bits(value.Float64))
		case FunctionStringColumn:
			fmt.Fprintf(&b, "%d:%s;", len(value.String), value.String)
		case FunctionBigIntColumn:
			fmt.Fprintf(&b, "%d:%s;", len(value.BigInt), value.BigInt)
		case FunctionBigRatColumn:
			fmt.Fprintf(&b, "%d:%s;", len(value.BigRat), value.BigRat)
		case FunctionUnitColumn:
			b.WriteString("unit;")
		}
	}
	return b.String()
}

func functionValuesEqual(a, b FunctionValue) bool {
	return toCoreValue(a) == toCoreValue(b)
}

func idsFromFunctionValues(values []FunctionValue) []Id {
	ids := make([]Id, len(values))
	for i, value := range values {
		ids[i] = value.ID
	}
	return ids
}

func typedFunctionTuple(row typedFunctionRow) []FunctionValue {
	values := make([]FunctionValue, len(row.keys)+1)
	copy(values, row.keys)
	values[len(row.keys)] = row.value
	return values
}

func (g *EGraph) mergeTypedFunctionValue(table *functionTable, old, new FunctionValue) (FunctionValue, error) {
	if old == new && table.schema.Merge != FunctionMergeExpression {
		return old, nil
	}
	if table.schema.ValueType.Kind == FunctionClassColumn &&
		(table.schema.Merge == FunctionMinInt64 || table.schema.Merge == FunctionMaxInt64) {
		oldValue, err := g.int64FunctionValue(table, old.ID)
		if err != nil {
			return FunctionValue{}, err
		}
		newValue, err := g.int64FunctionValue(table, new.ID)
		if err != nil {
			return FunctionValue{}, err
		}
		if table.schema.Merge == FunctionMinInt64 && oldValue <= newValue ||
			table.schema.Merge == FunctionMaxInt64 && oldValue >= newValue {
			return old, nil
		}
		return new, nil
	}
	policy := egraphcore.FunctionMergePolicy(0)
	var constant FunctionValue
	var custom func(egraphcore.Value[Id], egraphcore.Value[Id]) (egraphcore.Value[Id], error)
	switch table.schema.Merge {
	case FunctionKeepOld:
		policy = egraphcore.FunctionMergeKeepOld
	case FunctionUseNew:
		policy = egraphcore.FunctionMergeUseNew
	case FunctionAssertEqual:
		policy = egraphcore.FunctionMergeAssertEqual
	case FunctionConst:
		policy = egraphcore.FunctionMergeConstant
		var err error
		constant, err = g.canonicalTypedValue(table.schema.ValueType, table.schema.MergeTypedValue, "constant merge")
		if err != nil {
			return FunctionValue{}, err
		}
	case FunctionMinInt64:
		policy = egraphcore.FunctionMergeMinInt64
	case FunctionMaxInt64:
		policy = egraphcore.FunctionMergeMaxInt64
	case FunctionMergeExpression:
		policy = egraphcore.FunctionMergeCustom
		custom = func(old, incoming egraphcore.Value[Id]) (egraphcore.Value[Id], error) {
			value, err := g.evaluateFunctionMergeExpr(table, fromCoreValues([]egraphcore.Value[Id]{old})[0], fromCoreValues([]egraphcore.Value[Id]{incoming})[0])
			if err != nil {
				return egraphcore.Value[Id]{}, err
			}
			return toCoreValue(value), nil
		}
	case FunctionUnion:
		policy = egraphcore.FunctionMergeUnionClass
	default:
		return FunctionValue{}, fmt.Errorf("egraph: function %q has invalid merge policy", table.schema.Name)
	}
	merged, err := egraphcore.MergeFunctionValue(policy, toCoreValue(old), toCoreValue(new), toCoreValue(constant), egraphcore.FunctionMergeCallbacks[Id]{
		Canonicalize: g.Find,
		Union: func(left, right Id) (Id, error) {
			root, _, err := g.UnionChecked(left, right)
			return root, err
		},
		Custom: custom,
	})
	if err != nil {
		if policy == egraphcore.FunctionMergeAssertEqual {
			return FunctionValue{}, fmt.Errorf("egraph: function %q key already has a different result: %w", table.schema.Name, err)
		}
		return FunctionValue{}, fmt.Errorf("egraph: merge function %q result: %w", table.schema.Name, err)
	}
	return fromCoreValues([]egraphcore.Value[Id]{merged})[0], nil
}

func (g *EGraph) evaluateFunctionMergeExpr(table *functionTable, old, new FunctionValue) (FunctionValue, error) {
	view := newFunctionMergeView(g)
	value, err := g.evaluateFunctionMergeExprWithView(table, old, new, view)
	if err != nil {
		return FunctionValue{}, err
	}
	value, err = g.canonicalTypedValueWithView(view, table.schema.ValueType, value, "merge expression result")
	if err != nil {
		return FunctionValue{}, err
	}
	if err := g.commitFunctionMergeView(view); err != nil {
		return FunctionValue{}, err
	}
	return value, nil
}

func (g *EGraph) evaluateFunctionMergeExprWithView(table *functionTable, old, new FunctionValue, view functionMergeView) (FunctionValue, error) {
	var eval func(*FunctionMergeExpr) (FunctionValue, error)
	eval = func(expr *FunctionMergeExpr) (FunctionValue, error) {
		if expr == nil {
			return FunctionValue{}, fmt.Errorf("nil expression")
		}
		switch expr.kind {
		case FunctionMergeOldValue:
			return old, nil
		case FunctionMergeNewValue:
			return new, nil
		case FunctionMergeConstant:
			return g.canonicalTypedValue(table.schema.ValueType, expr.value, "merge expression constant")
		case FunctionMergeLookupExpr:
			target := g.functions[expr.name]
			if target == nil {
				return FunctionValue{}, fmt.Errorf("lookup target %q is not registered", expr.name)
			}
			keys := make([]FunctionValue, len(expr.args))
			for i := range expr.args {
				value, err := eval(&expr.args[i])
				if err != nil {
					return FunctionValue{}, fmt.Errorf("lookup target %q argument %d: %w", expr.name, i, err)
				}
				keys[i] = value
			}
			canonical, err := g.canonicalTypedKeysWithView(view, target, keys)
			if err != nil {
				return FunctionValue{}, err
			}
			key := typedFunctionKey(canonical)
			if target.schema.typed {
				staged, ok := view.tables[expr.name]
				if !ok {
					return FunctionValue{}, fmt.Errorf("lookup target %q has not been staged", expr.name)
				}
				row, ok := staged.getTyped(key)
				if !ok {
					switch target.schema.Default.Kind {
					case FunctionDefaultFail:
						return FunctionValue{}, fmt.Errorf("lookup target %q: %w", expr.name, ErrFunctionDefault)
					case FunctionDefaultConst:
						value, err := g.canonicalTypedValueWithView(view, target.schema.ValueType, target.schema.Default.TypedValue, "lookup default")
						if err != nil {
							return FunctionValue{}, err
						}
						makeStagedFunctionRowsPrivate(&staged)
						row = typedFunctionRow{keys: canonical, value: value, order: staged.takeOrder()}
						staged.defaultsInserted++
						staged.putTyped(key, row)
						view.tables[expr.name] = staged
					case FunctionDefaultFreshID:
						value, err := g.freshFunctionDefaultWithView(view, target, &staged)
						if err != nil {
							return FunctionValue{}, err
						}
						makeStagedFunctionRowsPrivate(&staged)
						row = typedFunctionRow{keys: canonical, value: value, order: staged.takeOrder()}
						staged.defaultsInserted++
						staged.putTyped(key, row)
						view.tables[expr.name] = staged
					case FunctionNoDefault:
						return FunctionValue{}, fmt.Errorf("lookup target %q has no row for key", expr.name)
					default:
						return FunctionValue{}, fmt.Errorf("lookup target %q fresh-ID default is unavailable during evaluation", expr.name)
					}
				}
				return g.canonicalTypedValueWithView(view, target.schema.ValueType, row.value, "lookup result")
			}
			legacyKey := relationKey(idsFromFunctionValues(canonical))
			staged, ok := view.tables[expr.name]
			if !ok {
				return FunctionValue{}, fmt.Errorf("lookup target %q has not been staged", expr.name)
			}
			row, ok := staged.getLegacy(legacyKey)
			if !ok {
				switch target.schema.Default.Kind {
				case FunctionDefaultFail:
					return FunctionValue{}, fmt.Errorf("lookup target %q: %w", expr.name, ErrFunctionDefault)
				case FunctionDefaultConst:
					value, err := g.canonicalTypedValueWithView(view, target.schema.ValueType, ClassValue(target.schema.Default.Value), "lookup default")
					if err != nil {
						return FunctionValue{}, err
					}
					makeStagedFunctionRowsPrivate(&staged)
					row = functionRow{keys: idsFromFunctionValues(canonical), value: value.ID, order: staged.takeOrder()}
					staged.defaultsInserted++
					staged.putLegacy(legacyKey, row)
					view.tables[expr.name] = staged
				case FunctionDefaultFreshID:
					value, err := g.freshFunctionDefaultWithView(view, target, &staged)
					if err != nil {
						return FunctionValue{}, err
					}
					makeStagedFunctionRowsPrivate(&staged)
					row = functionRow{keys: idsFromFunctionValues(canonical), value: value.ID, order: staged.takeOrder()}
					staged.defaultsInserted++
					staged.putLegacy(legacyKey, row)
					view.tables[expr.name] = staged
				case FunctionNoDefault:
					return FunctionValue{}, fmt.Errorf("lookup target %q has no row for key", expr.name)
				default:
					return FunctionValue{}, fmt.Errorf("lookup target %q fresh-ID default is unavailable during evaluation", expr.name)
				}
			}
			return g.canonicalTypedValueWithView(view, target.schema.ValueType, ClassValue(row.value), "lookup result")
		case FunctionMergeCallExpr:
			operator, ok := g.functionMergeOps[expr.name]
			if !ok {
				return FunctionValue{}, fmt.Errorf("merge operator %q is not registered", expr.name)
			}
			if len(expr.args) != len(operator.Arguments) {
				return FunctionValue{}, fmt.Errorf("merge operator %q expects %d arguments, got %d", expr.name, len(operator.Arguments), len(expr.args))
			}
			args := make([]FunctionValue, len(expr.args))
			for i := range expr.args {
				value, err := eval(&expr.args[i])
				if err != nil {
					return FunctionValue{}, fmt.Errorf("merge operator %q argument %d: %w", expr.name, i, err)
				}
				args[i], err = g.canonicalTypedValueWithView(view, operator.Arguments[i], value, "merge operator argument")
				if err != nil {
					return FunctionValue{}, fmt.Errorf("merge operator %q argument %d: %w", expr.name, i, err)
				}
			}
			value, err := operator.Evaluate(args)
			if err != nil {
				return FunctionValue{}, fmt.Errorf("merge operator %q: %w", expr.name, err)
			}
			return g.canonicalTypedValueWithView(view, operator.Result, value, "merge operator result")
		case FunctionMergePrimitiveExpr:
			args := make([]FunctionValue, len(expr.args))
			for i := range expr.args {
				value, err := eval(&expr.args[i])
				if err != nil {
					return FunctionValue{}, fmt.Errorf("argument %d: %w", i, err)
				}
				args[i] = value
			}
			value, err := evaluateFunctionPrimitive(g, expr.op, args)
			if err != nil {
				return FunctionValue{}, err
			}
			return value, nil
		default:
			return FunctionValue{}, fmt.Errorf("invalid expression kind %d", expr.kind)
		}
	}
	return eval(table.schema.MergeExpr)
}

func (g *EGraph) lookupFunctionWithView(name string, keys []FunctionValue, view functionMergeView) (FunctionValue, error) {
	target := g.functions[name]
	if target == nil {
		return FunctionValue{}, fmt.Errorf("lookup target %q is not registered", name)
	}
	canonical, err := g.canonicalTypedKeysWithView(view, target, keys)
	if err != nil {
		return FunctionValue{}, err
	}
	if target.schema.typed {
		staged, ok := view.tables[name]
		if !ok {
			return FunctionValue{}, fmt.Errorf("lookup target %q has not been staged", name)
		}
		key := typedFunctionKey(canonical)
		row, ok := staged.getTyped(key)
		if !ok {
			switch target.schema.Default.Kind {
			case FunctionDefaultFail:
				return FunctionValue{}, fmt.Errorf("lookup target %q: %w", name, ErrFunctionDefault)
			case FunctionDefaultConst:
				value, err := g.canonicalTypedValueWithView(view, target.schema.ValueType, target.schema.Default.TypedValue, "lookup default")
				if err != nil {
					return FunctionValue{}, err
				}
				makeStagedFunctionRowsPrivate(&staged)
				row = typedFunctionRow{keys: canonical, value: value, order: staged.takeOrder()}
				staged.defaultsInserted++
				staged.putTyped(key, row)
				view.tables[name] = staged
			case FunctionDefaultFreshID:
				value, err := g.freshFunctionDefaultWithView(view, target, &staged)
				if err != nil {
					return FunctionValue{}, err
				}
				makeStagedFunctionRowsPrivate(&staged)
				row = typedFunctionRow{keys: canonical, value: value, order: staged.takeOrder()}
				staged.defaultsInserted++
				staged.putTyped(key, row)
				view.tables[name] = staged
			case FunctionNoDefault:
				return FunctionValue{}, fmt.Errorf("lookup target %q has no row for key: %w", name, errPrimitiveNoValue)
			default:
				return FunctionValue{}, fmt.Errorf("lookup target %q fresh-ID default is unavailable during evaluation", name)
			}
		}
		return g.canonicalTypedValueWithView(view, target.schema.ValueType, row.value, "lookup result")
	}
	legacyKey := relationKey(idsFromFunctionValues(canonical))
	staged, ok := view.tables[name]
	if !ok {
		return FunctionValue{}, fmt.Errorf("lookup target %q has not been staged", name)
	}
	row, ok := staged.getLegacy(legacyKey)
	if !ok {
		switch target.schema.Default.Kind {
		case FunctionDefaultFail:
			return FunctionValue{}, fmt.Errorf("lookup target %q: %w", name, ErrFunctionDefault)
		case FunctionDefaultConst:
			value, err := g.canonicalTypedValueWithView(view, target.schema.ValueType, ClassValue(target.schema.Default.Value), "lookup default")
			if err != nil {
				return FunctionValue{}, err
			}
			makeStagedFunctionRowsPrivate(&staged)
			row = functionRow{keys: idsFromFunctionValues(canonical), value: value.ID, order: staged.takeOrder()}
			staged.defaultsInserted++
			staged.putLegacy(legacyKey, row)
			view.tables[name] = staged
		case FunctionDefaultFreshID:
			value, err := g.freshFunctionDefaultWithView(view, target, &staged)
			if err != nil {
				return FunctionValue{}, err
			}
			makeStagedFunctionRowsPrivate(&staged)
			row = functionRow{keys: idsFromFunctionValues(canonical), value: value.ID, order: staged.takeOrder()}
			staged.defaultsInserted++
			staged.putLegacy(legacyKey, row)
			view.tables[name] = staged
		case FunctionNoDefault:
			return FunctionValue{}, fmt.Errorf("lookup target %q has no row for key: %w", name, errPrimitiveNoValue)
		default:
			return FunctionValue{}, fmt.Errorf("lookup target %q fresh-ID default is unavailable during evaluation", name)
		}
	}
	return g.canonicalTypedValueWithView(view, target.schema.ValueType, ClassValue(row.value), "lookup result")
}

func (g *EGraph) freshFunctionDefaultWithView(view functionMergeView, table *functionTable, staged *stagedFunctionRows) (FunctionValue, error) {
	_, fresh := staged.counters()
	if fresh == ^uint64(0) {
		return FunctionValue{}, fmt.Errorf("egraph: create fresh default for function %q: fresh ID space exhausted", table.schema.Name)
	}
	node := ENode{
		Op:   "egraph.function-default",
		Sort: table.schema.ValueSort,
		Data: functionDefaultNode{name: table.schema.Name, index: fresh},
	}
	id, err := view.fresh.reserve(g, node)
	if err != nil {
		return FunctionValue{}, fmt.Errorf("egraph: create fresh default for function %q: %w", table.schema.Name, err)
	}
	makeStagedFunctionRowsPrivate(staged)
	staged.takeFresh()
	return ClassValue(id), nil
}

func (g *EGraph) canonicalTypedKeysWithView(view functionMergeView, table *functionTable, keys []FunctionValue) ([]FunctionValue, error) {
	if len(keys) != len(table.schema.KeyTypes) {
		return nil, fmt.Errorf("egraph: function %q expects %d key columns, got %d", table.schema.Name, len(table.schema.KeyTypes), len(keys))
	}
	canonical := make([]FunctionValue, len(keys))
	for i, value := range keys {
		var err error
		canonical[i], err = g.canonicalTypedValueWithView(view, table.schema.KeyTypes[i], value, fmt.Sprintf("key %d", i))
		if err != nil {
			return nil, err
		}
	}
	return canonical, nil
}

func (g *EGraph) canonicalTypedValueWithView(view functionMergeView, typ FunctionColumnType, value FunctionValue, where string) (FunctionValue, error) {
	if value.Kind != typ.Kind || typ.Kind != FunctionClassColumn {
		return g.canonicalTypedValue(typ, value, where)
	}
	if view.fresh != nil {
		if node, ok := view.fresh.provisional[value.ID]; ok {
			if typ.Sort != "" && node.Sort != typ.Sort {
				return FunctionValue{}, fmt.Errorf("egraph: function %s requires sort %q, got %q", where, typ.Sort, node.Sort)
			}
			return ClassValue(value.ID), nil
		}
	}
	return g.canonicalTypedValue(typ, value, where)
}

func (g *EGraph) canonicalizeTypedFunctionTable(table *functionTable) error {
	rows, err := g.canonicalTypedFunctionRows(table)
	if err != nil {
		return err
	}
	table.rows.ReplaceFrom(rows)
	return nil
}

func (g *EGraph) canonicalTypedFunctionRows(table *functionTable) (*egraphcore.OrderedTable[Id, functionTableRow], error) {
	return g.canonicalTypedFunctionRowsWithView(table, functionMergeView{})
}

func (g *EGraph) canonicalTypedFunctionRowsWithView(table *functionTable, view functionMergeView) (*egraphcore.OrderedTable[Id, functionTableRow], error) {
	groups := make(map[string][]typedFunctionRow, table.rows.Len())
	for _, stored := range table.rows.Rows() {
		row := stored.Payload.typed
		row.order = stored.Order
		row.subsumed = stored.Subsumed
		keys, err := g.canonicalTypedKeys(table, row.keys)
		if err != nil {
			return nil, err
		}
		value, err := g.canonicalTypedValue(table.schema.ValueType, row.value, "result")
		if err != nil {
			return nil, err
		}
		row.keys, row.value = keys, value
		groups[typedFunctionKey(keys)] = append(groups[typedFunctionKey(keys)], row)
	}
	result := newFunctionTable(table.schema)
	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		group := groups[key]
		sort.Slice(group, func(i, j int) bool { return group[i].order < group[j].order })
		row := group[0]
		for _, old := range group {
			if old.subsumed {
				row.subsumed = true
			}
		}
		for _, next := range group[1:] {
			merged, err := g.mergeCanonicalTypedFunctionRowsWithView(table, row, next, view)
			if err != nil {
				return nil, fmt.Errorf("egraph: function %q key %s: %w", table.schema.Name, key, err)
			}
			row = merged
		}
		result.putTyped(key, row)
		if row.subsumed {
			result.rows.MarkSubsumed(key)
		}
	}
	order, fresh := table.counters()
	result.setCounters(order, fresh)
	return result.rows, nil
}

func (g *EGraph) mergeCanonicalTypedFunctionRows(table *functionTable, current, incoming typedFunctionRow) (typedFunctionRow, error) {
	return g.mergeCanonicalTypedFunctionRowsWithView(table, current, incoming, functionMergeView{})
}

func (g *EGraph) mergeCanonicalTypedFunctionRowsWithView(table *functionTable, current, incoming typedFunctionRow, view functionMergeView) (typedFunctionRow, error) {
	var value FunctionValue
	var err error
	if table.schema.Merge == FunctionMergeExpression {
		value, err = g.evaluateFunctionMergeExprWithView(table, current.value, incoming.value, view)
	} else {
		value, err = g.mergeTypedFunctionValue(table, current.value, incoming.value)
	}
	if err != nil {
		return typedFunctionRow{}, err
	}
	newSubsumed := current.subsumed || incoming.subsumed
	changed := value != current.value || newSubsumed != current.subsumed
	result := current
	result.value = value
	result.subsumed = newSubsumed
	if changed {
		result.order = incoming.order
	}
	return result, nil
}
