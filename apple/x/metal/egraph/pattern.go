package egraph

import (
	"fmt"
	"math"
	"math/big"
	"math/bits"
	"reflect"
	"sort"
	"strings"

	"github.com/tmc/lib/egraphcore"
)

type matchWorkBudget struct {
	core *egraphcore.WorkBudget
}

func (b *matchWorkBudget) spend() bool {
	if b == nil {
		return true
	}
	return b.core != nil && b.core.TrySpend(1)
}

func (b *matchWorkBudget) spendCount(count int) bool {
	if b == nil {
		return true
	}
	return b.core != nil && b.core.TrySpend(count)
}

// spendSort reserves comparison work before sorting, so an exhausted search
// does not begin an uncharged table-sized sort.
func (b *matchWorkBudget) spendSort(length int) bool {
	if length < 2 {
		return true
	}
	levels := bits.Len(uint(length))
	maxInt := int(^uint(0) >> 1)
	if length > maxInt/(2*levels) {
		return false
	}
	return b.spendCount(2 * length * levels)
}

func (b *matchWorkBudget) spendPattern(pattern *Pattern) bool {
	stack := []*Pattern{pattern}
	seen := make(map[*Pattern]bool)
	for len(stack) > 0 {
		last := len(stack) - 1
		current := stack[last]
		stack = stack[:last]
		if current == nil || seen[current] {
			continue
		}
		if !b.spend() {
			return false
		}
		seen[current] = true
		for _, child := range current.Children {
			stack = append(stack, child)
		}
		for _, column := range current.FunctionColumns {
			stack = append(stack, column.Term)
		}
		for _, column := range current.RelationColumns {
			stack = append(stack, column.Term)
		}
	}
	return true
}

func patternInstantiationWork(pattern *Pattern) (int, error) {
	work := 0
	seen := make(map[*Pattern]bool)
	stack := []*Pattern{pattern}
	for len(stack) > 0 {
		last := len(stack) - 1
		current := stack[last]
		stack = stack[:last]
		if work == math.MaxInt {
			return 0, fmt.Errorf("pattern instantiation work count overflow")
		}
		work++
		if current == nil || seen[current] {
			continue
		}
		seen[current] = true
		stack = append(stack, current.Children...)
	}
	return work, nil
}

// Subst maps pattern variable names (e.g. "?x") to canonical E-Class IDs.
type Subst map[string]Id

// Clone returns a shallow copy of the substitution map.
func (s Subst) Clone() Subst {
	c := make(Subst, len(s))
	for k, v := range s {
		c[k] = v
	}
	return c
}

// Pattern represents a structural tree pattern containing operators, literals, and variables (?var).
type Pattern struct {
	Op              string
	Sort            string // Optional required result sort.
	IsVar           bool
	RelationName    string // Set for a relation atom; relation patterns are query-only.
	FunctionName    string // Set for a function row; function patterns are query-only.
	Data            any
	HasData         bool
	int64Expr       *FunctionActionExpr // Computed i64 leaf; valid only when instantiating a RHS.
	float64Expr     *FunctionActionExpr // Computed f64 leaf; valid only when instantiating a RHS.
	Children        []*Pattern
	FunctionColumns []FunctionColumnPattern
	RelationColumns []FunctionColumnPattern
}

// FunctionColumnPattern constrains one typed function-table column. Term is
// used for an e-class column; Variable or Value is used for primitive columns.
type FunctionColumnPattern struct {
	Kind     FunctionColumnKind
	Term     *Pattern
	Variable string
	Value    FunctionValue
	HasValue bool
	Expr     *FunctionActionExpr
}

// FunctionActionExpr is a typed value expression evaluated from bindings in a
// rewrite match. Expressions are immutable after construction.
type FunctionActionExpr struct {
	kind     functionActionExprKind
	value    FunctionValue
	variable string
	name     string
	op       FunctionMergeOp
	args     []FunctionActionExpr
}

type functionActionExprKind uint8

const (
	functionActionVariable functionActionExprKind = iota + 1
	functionActionConstant
	functionActionPrimitive
	functionActionClassInt64
	functionActionClassFloat64
	functionActionClassRef
	functionActionLookup
)

const functionActionDynamicColumn FunctionColumnKind = ^FunctionColumnKind(0)

// FunctionActionVar refers to a primitive variable bound by a typed function
// or relation pattern on the left-hand side.
func FunctionActionVar(name string) FunctionActionExpr {
	return FunctionActionExpr{kind: functionActionVariable, variable: patternVariableName(name)}
}

// FunctionActionInt64FromClass projects an i64 value from an e-class variable
// bound by a term, function, or relation pattern. The class must contain one
// unambiguous i64 leaf.
func FunctionActionInt64FromClass(name string) FunctionActionExpr {
	return FunctionActionExpr{kind: functionActionClassInt64, variable: patternVariableName(name)}
}

// FunctionActionFloat64FromClass projects an f64 value from an e-class
// variable. The class must contain one unambiguous f64 leaf.
func FunctionActionFloat64FromClass(name string) FunctionActionExpr {
	return FunctionActionExpr{kind: functionActionClassFloat64, variable: patternVariableName(name)}
}

// FunctionActionClass refers to an e-class variable bound by a term, function,
// or relation pattern. It can be used as a key to a function lookup.
func FunctionActionClass(name string) FunctionActionExpr {
	return FunctionActionExpr{kind: functionActionClassRef, variable: patternVariableName(name)}
}

// FunctionActionConst returns a typed value constant expression.
func FunctionActionConst(value FunctionValue) FunctionActionExpr {
	return FunctionActionExpr{kind: functionActionConstant, value: value}
}

// FunctionActionOp builds a checked primitive operation expression. It copies
// args so later caller mutations cannot change the expression.
func FunctionActionOp(op FunctionMergeOp, args ...FunctionActionExpr) FunctionActionExpr {
	return FunctionActionExpr{kind: functionActionPrimitive, op: op, args: cloneFunctionActionExprs(args)}
}

// FunctionActionLookup looks up a registered function using values computed by
// args. The lookup result can be used as a typed action key or value.
func FunctionActionLookup(name string, args ...FunctionActionExpr) FunctionActionExpr {
	return FunctionActionExpr{kind: functionActionLookup, name: name, args: cloneFunctionActionExprs(args)}
}

// PatInt64Expr creates a computed i64 leaf pattern for a RHS term. It can be
// embedded as a child of an ordinary constructor pattern.
func PatInt64Expr(expr FunctionActionExpr) *Pattern {
	cloned := cloneFunctionActionExprs([]FunctionActionExpr{expr})[0]
	return &Pattern{Op: "$computed-i64", Sort: "i64", int64Expr: &cloned}
}

// PatFloat64Expr creates a computed f64 leaf pattern for a RHS term.
func PatFloat64Expr(expr FunctionActionExpr) *Pattern {
	cloned := cloneFunctionActionExprs([]FunctionActionExpr{expr})[0]
	return &Pattern{Op: "$computed-f64", Sort: "f64", float64Expr: &cloned}
}

func cloneFunctionActionExprs(exprs []FunctionActionExpr) []FunctionActionExpr {
	cloned := make([]FunctionActionExpr, len(exprs))
	for i, expr := range exprs {
		expr.args = cloneFunctionActionExprs(expr.args)
		cloned[i] = expr
	}
	return cloned
}

// FunctionActionColumnExpr assigns an expression to a typed action column.
// Use FunctionActionClass or FunctionActionLookup for class-valued expressions.
func FunctionActionColumnExpr(kind FunctionColumnKind, expr FunctionActionExpr) FunctionColumnPattern {
	cloned := cloneFunctionActionExprs([]FunctionActionExpr{expr})[0]
	return FunctionColumnPattern{Kind: kind, Expr: &cloned}
}

// FunctionClassPattern matches an e-class function column with a term pattern.
func FunctionClassPattern(term *Pattern) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionClassColumn, Term: term}
}

// FunctionInt64Var matches and binds a primitive i64 function column.
func FunctionInt64Var(name string) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionInt64Column, Variable: patternVariableName(name)}
}

// FunctionInt64Literal matches a primitive i64 function column exactly.
func FunctionInt64Literal(value int64) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionInt64Column, Value: Int64Value(value), HasValue: true}
}

// FunctionBigIntVar matches and binds an arbitrary-precision integer column.
func FunctionBigIntVar(name string) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionBigIntColumn, Variable: patternVariableName(name)}
}

// FunctionBigIntLiteral matches an arbitrary-precision integer column exactly.
func FunctionBigIntLiteral(value *big.Int) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionBigIntColumn, Value: BigIntValue(value), HasValue: true}
}

// FunctionBigRatVar matches and binds an arbitrary-precision rational column.
func FunctionBigRatVar(name string) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionBigRatColumn, Variable: patternVariableName(name)}
}

// FunctionBigRatLiteral matches an arbitrary-precision rational column exactly.
func FunctionBigRatLiteral(value *big.Rat) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionBigRatColumn, Value: BigRatValue(value), HasValue: true}
}

// FunctionBoolVar matches and binds a primitive bool function column.
func FunctionBoolVar(name string) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionBoolColumn, Variable: patternVariableName(name)}
}

// FunctionBoolLiteral matches a primitive bool function column exactly.
func FunctionBoolLiteral(value bool) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionBoolColumn, Value: BoolValue(value), HasValue: true}
}

// FunctionFloat64Var matches and binds a primitive float64 function column.
func FunctionFloat64Var(name string) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionFloat64Column, Variable: patternVariableName(name)}
}

// FunctionFloat64Literal matches a primitive float64 function column exactly.
func FunctionFloat64Literal(value float64) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionFloat64Column, Value: Float64Value(value), HasValue: true}
}

// FunctionStringVar matches and binds a primitive string function column.
func FunctionStringVar(name string) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionStringColumn, Variable: patternVariableName(name)}
}

// FunctionStringLiteral matches a primitive string function column exactly.
func FunctionStringLiteral(value string) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionStringColumn, Value: StringValue(value), HasValue: true}
}

// FunctionUnit matches the unit value in a function column.
func FunctionUnit() FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionUnitColumn, Value: UnitValue(), HasValue: true}
}

// FunctionUnitVar matches and binds a unit-valued function column.
func FunctionUnitVar(name string) FunctionColumnPattern {
	return FunctionColumnPattern{Kind: FunctionUnitColumn, Variable: patternVariableName(name)}
}

func patternVariableName(name string) string {
	if !strings.HasPrefix(name, "?") {
		return "?" + name
	}
	return name
}

// RelationPattern creates a query-only pattern for a registered predicate
// relation. Its children are matched against the relation's columns.
func RelationPattern(name string, columns ...*Pattern) *Pattern {
	return &Pattern{Op: name, RelationName: name, Children: columns}
}

// FunctionPattern creates a query-only pattern for a function table. Its
// children match the key columns followed by the result column.
func FunctionPattern(name string, columns ...*Pattern) *Pattern {
	return &Pattern{Op: name, FunctionName: name, Children: columns}
}

// TypedFunctionPattern creates a query-only function pattern with typed base
// columns in addition to ordinary e-class term columns.
func TypedFunctionPattern(name string, columns ...FunctionColumnPattern) *Pattern {
	return &Pattern{Op: name, FunctionName: name, FunctionColumns: append([]FunctionColumnPattern(nil), columns...)}
}

// TypedRelationPattern creates a query pattern over class and primitive relation columns.
func TypedRelationPattern(name string, columns ...FunctionColumnPattern) *Pattern {
	return &Pattern{Op: name, RelationName: name, RelationColumns: append([]FunctionColumnPattern(nil), columns...)}
}

// Var creates a variable pattern node (e.g. "?a", "?x").
func Var(name string) *Pattern {
	if !strings.HasPrefix(name, "?") {
		name = "?" + name
	}
	return &Pattern{
		Op:    name,
		IsVar: true,
	}
}

// VarSorted creates a variable pattern constrained to one result sort.
func VarSorted(sort, name string) *Pattern {
	p := Var(name)
	p.Sort = sort
	return p
}

// Pat creates an operator pattern node with child patterns.
func Pat(op string, children ...*Pattern) *Pattern {
	return &Pattern{
		Op:       op,
		IsVar:    false,
		Children: children,
	}
}

// PatSorted creates an operator pattern constrained to one result sort.
func PatSorted(sort, op string, children ...*Pattern) *Pattern {
	return &Pattern{Op: op, Sort: sort, Children: children}
}

// PatWithData creates an operator pattern that also requires an exact payload
// match. Use it when a rewrite depends on literal, shape, or type metadata.
func PatWithData(op string, data any, children ...*Pattern) *Pattern {
	return &Pattern{
		Op:       op,
		Data:     data,
		HasData:  true,
		Children: children,
	}
}

// PatWithDataSorted creates an operator pattern constrained by result sort and payload.
func PatWithDataSorted(sort, op string, data any, children ...*Pattern) *Pattern {
	return &Pattern{Op: op, Sort: sort, Data: data, HasData: true, Children: children}
}

// ParsePattern parses an S-expression pattern string like "(+ ?a ?b)" or "(gemv (rmsnorm ?x ?w) ?wq)".
func ParsePattern(s string) (*Pattern, error) {
	core, err := egraphcore.ParsePattern(s)
	if err != nil {
		return nil, fmt.Errorf("egraph: %w", err)
	}
	return &Pattern{Op: core.Op, IsVar: core.Variable, Children: convertCorePatternChildren(core.Children)}, nil
}

func convertCorePatternChildren(children []*egraphcore.Pattern) []*Pattern {
	type pair struct {
		core  *egraphcore.Pattern
		local *Pattern
	}
	converted := make([]*Pattern, len(children))
	stack := make([]pair, 0, len(children))
	for i, child := range children {
		converted[i] = &Pattern{Op: child.Op, IsVar: child.Variable}
		stack = append(stack, pair{core: child, local: converted[i]})
	}
	for len(stack) != 0 {
		last := len(stack) - 1
		current := stack[last]
		stack = stack[:last]
		current.local.Children = make([]*Pattern, len(current.core.Children))
		for i, child := range current.core.Children {
			current.local.Children[i] = &Pattern{Op: child.Op, IsVar: child.Variable}
			stack = append(stack, pair{core: child, local: current.local.Children[i]})
		}
	}
	return converted
}

// Match captures the binding of a term or relation pattern. ClassID is
// InvalidId when the pattern matches relation rows rather than an e-class.
type Match struct {
	ClassID Id
	Subst   Subst
	// BaseSubst contains primitive bindings from typed function patterns.
	BaseSubst      map[string]FunctionValue
	lookupView     *functionMergeView
	computedValues map[*Pattern]FunctionValue
}

// Search returns all e-classes matching this pattern. It returns nil for an
// invalid pattern or graph; use SearchChecked to distinguish errors from no matches.
func (p *Pattern) Search(g *EGraph) []Match {
	matches, _ := p.SearchChecked(g)
	return matches
}

// SearchChecked rebuilds pending unions, then returns term or relation matches.
// Relation matches have ClassID set to InvalidId.
func (p *Pattern) SearchChecked(g *EGraph) ([]Match, error) {
	matches, _, err := p.searchChecked(g, nil)
	return matches, err
}

func (p *Pattern) searchChecked(g *EGraph, budget *matchWorkBudget) ([]Match, bool, error) {
	if p == nil {
		return nil, false, fmt.Errorf("egraph: nil pattern")
	}
	if g == nil {
		return nil, false, fmt.Errorf("egraph: nil graph")
	}
	if err := validatePattern(p); err != nil {
		return nil, false, fmt.Errorf("egraph: invalid pattern: %w", err)
	}
	if !g.store.Initialized() || g.opIndex == nil {
		return nil, false, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if !budget.spendPattern(p) {
		return nil, false, nil
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, false, fmt.Errorf("egraph: rebuild before search: %w", err)
	}
	variables := make(map[string]bool)
	collectVariables(p, variables)
	var names []string
	for name := range variables {
		names = append(names, name)
	}
	if !budget.spendSort(len(names)) {
		return nil, false, nil
	}
	sort.Strings(names)
	if p.RelationName != "" {
		return p.searchRelationKeysWithBudget(g, names, nil, budget)
	}
	if p.FunctionName != "" {
		return p.searchFunctionWithBudget(g, names, budget)
	}

	var classes []*EClass
	if p.IsVar {
		if budget == nil {
			classes = g.classesList()
		} else {
			// Class IDs are assigned monotonically, so walking unionFind keeps
			// the same stable order without first allocating and sorting every
			// live class.
			for rawID := 0; rawID < g.store.Len(); rawID++ {
				if !budget.spend() {
					return nil, false, nil
				}
				if class := g.store.Class(Id(rawID)); class != nil {
					classes = append(classes, class)
				}
			}
		}
	} else {
		seenClasses := make(map[Id]bool)
		key := nodeIndexKey{op: p.Op, arity: len(p.Children)}
		for _, classID := range g.opIndex[key] {
			if budget != nil && !budget.spend() {
				return nil, false, nil
			}
			root := g.Find(classID)
			if seenClasses[root] {
				continue
			}
			if class := g.store.Class(root); class != nil {
				classes = append(classes, class)
				seenClasses[root] = true
			}
		}
		if len(classes) == 0 {
			return nil, true, nil
		}
	}

	var matches []Match
	for _, class := range classes {
		if budget != nil && !budget.spend() {
			return nil, false, nil
		}
		var substs []Subst
		if budget == nil {
			substs = p.matchClass(g, class.ID, make(Subst))
		} else {
			var complete bool
			substs, complete = p.matchClassWithBudget(g, class.ID, make(Subst), budget)
			if !complete {
				return nil, false, nil
			}
		}
		seen := make(map[string]bool)
		for _, s := range substs {
			var key strings.Builder
			for _, name := range names {
				fmt.Fprintf(&key, "%d:%s=%d;", len(name), name, g.Find(s[name]))
			}
			matchKey := key.String()
			if seen[matchKey] {
				continue
			}
			seen[matchKey] = true
			if budget != nil && !budget.spend() {
				return nil, false, nil
			}
			matches = append(matches, Match{
				ClassID: g.Find(class.ID),
				Subst:   s,
			})
		}
	}
	return matches, true, nil
}

func (p *Pattern) searchClassesChecked(g *EGraph, classIDs []Id, budget *matchWorkBudget) ([]Match, bool, error) {
	if p == nil {
		return nil, false, fmt.Errorf("egraph: nil pattern")
	}
	if g == nil || !g.store.Initialized() || g.opIndex == nil {
		return nil, false, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if g.dirty {
		return nil, false, fmt.Errorf("egraph: graph must be rebuilt before searching selected classes")
	}
	if err := validatePattern(p); err != nil {
		return nil, false, fmt.Errorf("egraph: invalid pattern: %w", err)
	}
	if p.RelationName != "" || p.FunctionName != "" {
		return nil, false, fmt.Errorf("egraph: selected-class search requires a term pattern")
	}
	if !budget.spendPattern(p) {
		return nil, false, nil
	}
	variables := make(map[string]bool)
	collectVariables(p, variables)
	names := make([]string, 0, len(variables))
	for name := range variables {
		names = append(names, name)
	}
	if !budget.spendSort(len(names)) {
		return nil, false, nil
	}
	sort.Strings(names)
	seenClasses := make(map[Id]bool)
	var matches []Match
	for _, classID := range classIDs {
		if !budget.spend() {
			return nil, false, nil
		}
		root := g.Find(classID)
		if root == InvalidId {
			continue
		}
		class := g.store.Class(root)
		if class == nil || seenClasses[root] {
			continue
		}
		seenClasses[root] = true
		substs, complete := p.matchClassWithBudget(g, root, make(Subst), budget)
		if !complete {
			return nil, false, nil
		}
		seenMatches := make(map[string]bool)
		for _, subst := range substs {
			var key strings.Builder
			for _, name := range names {
				fmt.Fprintf(&key, "%d:%s=%d;", len(name), name, g.Find(subst[name]))
			}
			matchKey := key.String()
			if seenMatches[matchKey] {
				continue
			}
			if !budget.spend() {
				return nil, false, nil
			}
			seenMatches[matchKey] = true
			matches = append(matches, Match{ClassID: root, Subst: subst})
		}
	}
	return matches, true, nil
}

func (p *Pattern) searchFunction(g *EGraph, names []string) ([]Match, error) {
	arity := len(p.Children) - 1
	if len(p.FunctionColumns) > 0 {
		arity = len(p.FunctionColumns) - 1
	}
	table, err := g.function(p.FunctionName, arity)
	if err != nil {
		return nil, err
	}
	if table.schema.typed {
		return p.searchTypedFunction(g, table, names)
	}
	if len(p.FunctionColumns) > 0 {
		return p.searchAllClassTypedFunction(g, table, names)
	}
	if len(p.Children) != len(table.schema.KeySorts)+1 {
		return nil, fmt.Errorf("egraph: function pattern %q expects %d columns, got %d", p.FunctionName, len(table.schema.KeySorts)+1, len(p.Children))
	}
	storedRows := table.rows.Rows()
	allKeys := make([]string, 0, len(storedRows))
	for _, row := range storedRows {
		allKeys = append(allKeys, row.Key)
	}
	keys, err := indexedPatternKeys(g, p.Children, table.rows, allKeys)
	if err != nil {
		return nil, err
	}
	var matches []Match
	seen := make(map[string]bool)
	for _, key := range keys {
		row, ok := table.getLegacy(key)
		if !ok || row.subsumed {
			continue
		}
		columns := functionRowTuple(row)
		substs := []Subst{make(Subst)}
		for i, pattern := range p.Children {
			var next []Subst
			for _, subst := range substs {
				next = append(next, pattern.matchClass(g, columns[i], subst)...)
			}
			substs = next
			if len(substs) == 0 {
				break
			}
		}
		for _, subst := range substs {
			var matchKey strings.Builder
			for _, name := range names {
				fmt.Fprintf(&matchKey, "%d:%s=%d;", len(name), name, g.Find(subst[name]))
			}
			key := matchKey.String()
			if seen[key] {
				continue
			}
			seen[key] = true
			matches = append(matches, Match{ClassID: InvalidId, Subst: subst})
		}
	}
	return matches, nil
}

func (p *Pattern) searchFunctionWithBudget(g *EGraph, names []string, budget *matchWorkBudget) ([]Match, bool, error) {
	if budget == nil {
		matches, err := p.searchFunction(g, names)
		return matches, true, err
	}
	arity := len(p.Children) - 1
	if len(p.FunctionColumns) > 0 {
		arity = len(p.FunctionColumns) - 1
	}
	table, err := g.function(p.FunctionName, arity)
	if err != nil {
		return nil, false, err
	}
	if table.schema.typed {
		return p.searchTypedFunctionWithBudget(g, table, names, budget)
	}
	if len(p.FunctionColumns) > 0 {
		return p.searchAllClassTypedFunctionWithBudget(g, table, names, budget)
	}
	if len(p.Children) != len(table.schema.KeySorts)+1 {
		return nil, false, fmt.Errorf("egraph: function pattern %q expects %d columns, got %d", p.FunctionName, len(table.schema.KeySorts)+1, len(p.Children))
	}
	storedRows := table.rows.Rows()
	allKeys := make([]string, 0, len(storedRows))
	for _, row := range storedRows {
		if !budget.spend() {
			return nil, false, nil
		}
		allKeys = append(allKeys, row.Key)
	}
	keys, complete, err := indexedPatternKeysWithBudget(g, p.Children, table.rows, allKeys, budget)
	if err != nil || !complete {
		return nil, complete, err
	}
	var matches []Match
	seen := make(map[string]bool)
	for _, key := range keys {
		if !budget.spend() {
			return nil, false, nil
		}
		row, ok := table.getLegacy(key)
		if !ok || row.subsumed {
			continue
		}
		columns := functionRowTuple(row)
		substs := []Subst{make(Subst)}
		for i, pattern := range p.Children {
			var next []Subst
			for _, subst := range substs {
				if !budget.spend() {
					return nil, false, nil
				}
				found, complete := pattern.matchClassWithBudget(g, columns[i], subst, budget)
				if !complete {
					return nil, false, nil
				}
				next = append(next, found...)
			}
			substs = next
			if len(substs) == 0 {
				break
			}
		}
		for _, subst := range substs {
			var key strings.Builder
			for _, name := range names {
				fmt.Fprintf(&key, "%d:%s=%d;", len(name), name, g.Find(subst[name]))
			}
			matchKey := key.String()
			if seen[matchKey] {
				continue
			}
			if !budget.spend() {
				return nil, false, nil
			}
			seen[matchKey] = true
			matches = append(matches, Match{ClassID: InvalidId, Subst: subst})
		}
	}
	return matches, true, nil
}

func (p *Pattern) searchAllClassTypedFunctionWithBudget(g *EGraph, table *functionTable, names []string, budget *matchWorkBudget) ([]Match, bool, error) {
	columns := p.FunctionColumns
	if len(columns) != len(table.schema.KeySorts)+1 {
		return nil, false, fmt.Errorf("egraph: function pattern %q expects %d columns, got %d", p.FunctionName, len(table.schema.KeySorts)+1, len(columns))
	}
	storedRows := table.rows.Rows()
	rows := make(map[string]functionRow, len(storedRows))
	keys := make([]string, 0, len(storedRows))
	for _, stored := range storedRows {
		if !budget.spend() {
			return nil, false, nil
		}
		keys = append(keys, stored.Key)
		row := stored.Payload.legacy
		row.subsumed = stored.Subsumed
		rows[stored.Key] = row
	}
	if !budget.spendSort(len(keys)) {
		return nil, false, nil
	}
	sort.Slice(keys, func(i, j int) bool {
		left, right := rows[keys[i]], rows[keys[j]]
		for k := range left.keys {
			l, r := g.Find(left.keys[k]), g.Find(right.keys[k])
			if l != r {
				return l < r
			}
		}
		return g.Find(left.value) < g.Find(right.value)
	})
	var matches []Match
	seen := make(map[string]bool)
	for _, key := range keys {
		if !budget.spend() {
			return nil, false, nil
		}
		row := rows[key]
		if row.subsumed {
			continue
		}
		values := append(append([]Id(nil), row.keys...), row.value)
		states := []Match{{ClassID: InvalidId, Subst: make(Subst), BaseSubst: make(map[string]FunctionValue)}}
		for i, column := range columns {
			if column.Kind != 0 && column.Kind != FunctionClassColumn {
				return nil, false, fmt.Errorf("egraph: function pattern %q column %d requires an e-class pattern", p.FunctionName, i)
			}
			if column.Term == nil {
				return nil, false, fmt.Errorf("egraph: function pattern %q column %d requires a term pattern", p.FunctionName, i)
			}
			var next []Match
			for _, state := range states {
				if !budget.spend() {
					return nil, false, nil
				}
				substs, complete := column.Term.matchClassWithBudget(g, values[i], state.Subst, budget)
				if !complete {
					return nil, false, nil
				}
				for _, subst := range substs {
					if !budget.spend() {
						return nil, false, nil
					}
					next = append(next, Match{ClassID: InvalidId, Subst: subst, BaseSubst: cloneBaseSubst(state.BaseSubst)})
				}
			}
			states = next
			if len(states) == 0 {
				break
			}
		}
		for _, state := range states {
			matchKey := typedSubstitutionKey(state.Subst, state.BaseSubst, names, g)
			if seen[matchKey] {
				continue
			}
			if !budget.spend() {
				return nil, false, nil
			}
			seen[matchKey] = true
			matches = append(matches, state)
		}
	}
	return matches, true, nil
}

func (p *Pattern) searchTypedFunctionWithBudget(g *EGraph, table *functionTable, names []string, budget *matchWorkBudget) ([]Match, bool, error) {
	columns := p.FunctionColumns
	if len(columns) == 0 && len(p.Children) > 0 {
		columns = make([]FunctionColumnPattern, len(p.Children))
		for i, child := range p.Children {
			columns[i] = FunctionClassPattern(child)
		}
	}
	if len(columns) != len(table.schema.KeyTypes)+1 {
		return nil, false, fmt.Errorf("egraph: function pattern %q expects %d columns, got %d", p.FunctionName, len(table.schema.KeyTypes)+1, len(columns))
	}
	for i := 0; i < 6; i++ {
		if !budget.spendCount(table.rows.Len()) {
			return nil, false, nil
		}
	}
	for i := 0; i < 3; i++ {
		if !budget.spendSort(table.rows.Len()) {
			return nil, false, nil
		}
	}
	if err := g.canonicalizeTypedFunctionTable(table); err != nil {
		return nil, false, err
	}
	candidates := map[string]struct{}(nil)
	canonicalLiterals := make([]*FunctionValue, len(columns))
	for i, column := range columns {
		typ := table.schema.ValueType
		if i < len(table.schema.KeyTypes) {
			typ = table.schema.KeyTypes[i]
		}
		if column.Kind != 0 && column.Kind != typ.Kind {
			return nil, false, fmt.Errorf("egraph: function pattern %q column %d has kind %d, want %d", p.FunctionName, i, column.Kind, typ.Kind)
		}
		var values []FunctionValue
		switch {
		case column.Term != nil:
			if typ.Kind != FunctionClassColumn {
				return nil, false, fmt.Errorf("egraph: function pattern %q column %d requires a primitive pattern", p.FunctionName, i)
			}
			found, complete, err := column.Term.searchChecked(g, budget)
			if err != nil || !complete {
				return nil, complete, err
			}
			seen := make(map[Id]bool)
			for _, match := range found {
				if !budget.spend() {
					return nil, false, nil
				}
				id := g.Find(match.ClassID)
				if !seen[id] {
					values = append(values, ClassValue(id))
					seen[id] = true
				}
			}
		case column.HasValue:
			canonical, err := g.canonicalTypedValue(typ, column.Value, fmt.Sprintf("pattern column %d", i))
			if err != nil {
				return nil, false, err
			}
			canonicalLiterals[i] = &canonical
			values = append(values, canonical)
		case column.Variable != "":
			continue
		default:
			return nil, false, fmt.Errorf("egraph: function pattern %q column %d has no constraint", p.FunctionName, i)
		}
		columnRows := make(map[string]struct{})
		for _, value := range values {
			complete := table.rows.Range(i, toCoreValue(value), func(key string) bool {
				if !budget.spend() {
					return false
				}
				columnRows[key] = struct{}{}
				return true
			})
			if !complete {
				return nil, false, nil
			}
		}
		if candidates == nil {
			candidates = columnRows
		} else {
			for key := range candidates {
				if !budget.spend() {
					return nil, false, nil
				}
				if _, ok := columnRows[key]; !ok {
					delete(candidates, key)
				}
			}
		}
		if len(candidates) == 0 {
			return nil, true, nil
		}
	}
	var keys []string
	if candidates == nil {
		for _, row := range table.rows.Rows() {
			if !budget.spend() {
				return nil, false, nil
			}
			keys = append(keys, row.Key)
		}
	} else {
		for key := range candidates {
			if !budget.spend() {
				return nil, false, nil
			}
			keys = append(keys, key)
		}
	}
	if !budget.spendSort(len(keys)) {
		return nil, false, nil
	}
	sort.Strings(keys)
	var matches []Match
	seen := make(map[string]bool)
	termVariables := make([]map[string]bool, len(columns))
	for i, column := range columns {
		if column.Term == nil {
			continue
		}
		if !budget.spendPattern(column.Term) {
			return nil, false, nil
		}
		termVariables[i] = make(map[string]bool)
		collectVariables(column.Term, termVariables[i])
	}
	for _, key := range keys {
		if !budget.spend() {
			return nil, false, nil
		}
		row, ok := table.getTyped(key)
		if !ok || row.subsumed {
			continue
		}
		values := typedFunctionTuple(row)
		states := []Match{{Subst: make(Subst), BaseSubst: make(map[string]FunctionValue)}}
		for i, column := range columns {
			var next []Match
			for _, state := range states {
				if !budget.spend() {
					return nil, false, nil
				}
				value := values[i]
				if column.Term != nil {
					if value.Kind != FunctionClassColumn {
						continue
					}
					crossKind := false
					for name := range termVariables[i] {
						if _, bound := state.BaseSubst[name]; bound {
							crossKind = true
							break
						}
					}
					if crossKind {
						continue
					}
					substs, complete := column.Term.matchClassWithBudget(g, value.ID, state.Subst, budget)
					if !complete {
						return nil, false, nil
					}
					for _, subst := range substs {
						if !budget.spend() {
							return nil, false, nil
						}
						next = append(next, Match{Subst: subst, BaseSubst: cloneBaseSubst(state.BaseSubst)})
					}
					continue
				}
				if column.Kind != value.Kind {
					continue
				}
				if column.HasValue {
					if canonicalLiterals[i] != nil && *canonicalLiterals[i] == value {
						next = append(next, state)
					}
					continue
				}
				if column.Variable == "" {
					continue
				}
				if _, bound := state.Subst[column.Variable]; bound {
					continue
				}
				if old, bound := state.BaseSubst[column.Variable]; bound {
					if old == value {
						next = append(next, state)
					}
					continue
				}
				bases := cloneBaseSubst(state.BaseSubst)
				bases[column.Variable] = value
				next = append(next, Match{Subst: state.Subst, BaseSubst: bases})
			}
			states = next
			if len(states) == 0 {
				break
			}
		}
		for _, state := range states {
			matchKey := typedSubstitutionKey(state.Subst, state.BaseSubst, names, g)
			if seen[matchKey] {
				continue
			}
			if !budget.spend() {
				return nil, false, nil
			}
			seen[matchKey] = true
			matches = append(matches, Match{ClassID: InvalidId, Subst: state.Subst, BaseSubst: state.BaseSubst})
		}
	}
	return matches, true, nil
}

func (p *Pattern) searchAllClassTypedFunction(g *EGraph, table *functionTable, names []string) ([]Match, error) {
	columns := p.FunctionColumns
	if len(columns) != len(table.schema.KeySorts)+1 {
		return nil, fmt.Errorf("egraph: function pattern %q expects %d columns, got %d", p.FunctionName, len(table.schema.KeySorts)+1, len(columns))
	}
	rows := functionRowsWithStatusSnapshot(g, table)
	var matches []Match
	seen := make(map[string]bool)
	for _, row := range rows {
		if row.Subsumed {
			continue
		}
		values := append(append([]Id(nil), row.Keys...), row.Value)
		states := []Match{{ClassID: InvalidId, Subst: make(Subst), BaseSubst: make(map[string]FunctionValue)}}
		for i, column := range columns {
			if column.Kind != 0 && column.Kind != FunctionClassColumn {
				return nil, fmt.Errorf("egraph: function pattern %q column %d requires an e-class pattern", p.FunctionName, i)
			}
			if column.Term == nil {
				return nil, fmt.Errorf("egraph: function pattern %q column %d requires a term pattern", p.FunctionName, i)
			}
			var next []Match
			for _, state := range states {
				for _, subst := range column.Term.matchClass(g, values[i], state.Subst) {
					next = append(next, Match{ClassID: InvalidId, Subst: subst, BaseSubst: cloneBaseSubst(state.BaseSubst)})
				}
			}
			states = next
			if len(states) == 0 {
				break
			}
		}
		for _, state := range states {
			key := typedSubstitutionKey(state.Subst, state.BaseSubst, names, g)
			if !seen[key] {
				seen[key] = true
				matches = append(matches, state)
			}
		}
	}
	return matches, nil
}

func (p *Pattern) searchTypedFunction(g *EGraph, table *functionTable, names []string) ([]Match, error) {
	columns := p.FunctionColumns
	if len(columns) == 0 && len(p.Children) > 0 {
		columns = make([]FunctionColumnPattern, len(p.Children))
		for i, child := range p.Children {
			columns[i] = FunctionClassPattern(child)
		}
	}
	if len(columns) != len(table.schema.KeyTypes)+1 {
		return nil, fmt.Errorf("egraph: function pattern %q expects %d columns, got %d", p.FunctionName, len(table.schema.KeyTypes)+1, len(columns))
	}
	if err := g.canonicalizeTypedFunctionTable(table); err != nil {
		return nil, err
	}
	candidates := map[string]struct{}(nil)
	canonicalLiterals := make([]*FunctionValue, len(columns))
	for i, column := range columns {
		typ := table.schema.ValueType
		if i < len(table.schema.KeyTypes) {
			typ = table.schema.KeyTypes[i]
		}
		if column.Kind != 0 && column.Kind != typ.Kind {
			return nil, fmt.Errorf("egraph: function pattern %q column %d has kind %d, want %d", p.FunctionName, i, column.Kind, typ.Kind)
		}
		var values []FunctionValue
		switch {
		case column.Term != nil:
			if typ.Kind != FunctionClassColumn {
				return nil, fmt.Errorf("egraph: function pattern %q column %d requires a primitive pattern", p.FunctionName, i)
			}
			found, err := column.Term.SearchChecked(g)
			if err != nil {
				return nil, err
			}
			seen := make(map[Id]bool)
			for _, match := range found {
				id := g.Find(match.ClassID)
				if !seen[id] {
					values = append(values, ClassValue(id))
					seen[id] = true
				}
			}
		case column.HasValue:
			canonical, err := g.canonicalTypedValue(typ, column.Value, fmt.Sprintf("pattern column %d", i))
			if err != nil {
				return nil, err
			}
			canonicalLiterals[i] = &canonical
			values = append(values, canonical)
		case column.Variable != "":
			continue
		default:
			return nil, fmt.Errorf("egraph: function pattern %q column %d has no constraint", p.FunctionName, i)
		}
		columnRows := make(map[string]struct{})
		for _, value := range values {
			if !table.rows.Range(i, toCoreValue(value), func(key string) bool {
				columnRows[key] = struct{}{}
				return true
			}) {
				return nil, fmt.Errorf("egraph: invalid typed function index")
			}
		}
		if candidates == nil {
			candidates = columnRows
			continue
		}
		for key := range candidates {
			if _, ok := columnRows[key]; !ok {
				delete(candidates, key)
			}
		}
		if len(candidates) == 0 {
			return nil, nil
		}
	}
	var keys []string
	if candidates == nil {
		for _, row := range table.rows.Rows() {
			keys = append(keys, row.Key)
		}
	} else {
		for key := range candidates {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	var matches []Match
	seenMatches := make(map[string]bool)
	for _, key := range keys {
		row, ok := table.getTyped(key)
		if !ok || row.subsumed {
			continue
		}
		values := typedFunctionTuple(row)
		states := []Match{{Subst: make(Subst), BaseSubst: make(map[string]FunctionValue)}}
		for i, pattern := range columns {
			next := make([]Match, 0, len(states))
			for _, state := range states {
				value := values[i]
				if pattern.Term != nil {
					termVariables := make(map[string]bool)
					collectVariables(pattern.Term, termVariables)
					crossKind := false
					for name := range termVariables {
						if _, bound := state.BaseSubst[name]; bound {
							crossKind = true
							break
						}
					}
					if crossKind {
						continue
					}
					for _, subst := range pattern.Term.matchClass(g, value.ID, state.Subst) {
						bound := cloneBaseSubst(state.BaseSubst)
						next = append(next, Match{Subst: subst, BaseSubst: bound})
					}
					continue
				}
				if pattern.Kind != value.Kind {
					continue
				}
				if pattern.HasValue {
					if canonicalLiterals[i] == nil || *canonicalLiterals[i] != value {
						continue
					}
					next = append(next, state)
					continue
				}
				if pattern.Variable == "" {
					continue
				}
				if _, bound := state.Subst[pattern.Variable]; bound {
					continue
				}
				if old, bound := state.BaseSubst[pattern.Variable]; bound {
					if old == value {
						next = append(next, state)
					}
					continue
				}
				bases := cloneBaseSubst(state.BaseSubst)
				bases[pattern.Variable] = value
				next = append(next, Match{Subst: state.Subst, BaseSubst: bases})
			}
			states = next
			if len(states) == 0 {
				break
			}
		}
		for _, state := range states {
			matchKey := typedSubstitutionKey(state.Subst, state.BaseSubst, names, g)
			if seenMatches[matchKey] {
				continue
			}
			seenMatches[matchKey] = true
			matches = append(matches, Match{ClassID: InvalidId, Subst: state.Subst, BaseSubst: state.BaseSubst})
		}
	}
	return matches, nil
}

func cloneBaseSubst(src map[string]FunctionValue) map[string]FunctionValue {
	dst := make(map[string]FunctionValue, len(src))
	for name, value := range src {
		dst[name] = value
	}
	return dst
}

func typedSubstitutionKey(ids Subst, bases map[string]FunctionValue, names []string, g *EGraph) string {
	var key strings.Builder
	for _, name := range names {
		if id, ok := ids[name]; ok {
			fmt.Fprintf(&key, "%d:%s=i%d;", len(name), name, g.Find(id))
			continue
		}
		if value, ok := bases[name]; ok {
			fmt.Fprintf(&key, "%d:%s=v%s;", len(name), name, typedFunctionKey([]FunctionValue{value}))
		}
	}
	return key.String()
}

func (p *Pattern) searchRelation(g *EGraph, names []string) ([]Match, error) {
	return p.searchRelationKeys(g, names, nil)
}

// searchRelationKeys matches only the supplied canonical row keys when keys is
// non-nil. A nil key slice retains the full-search behavior used by the public
// search path.
func (p *Pattern) searchRelationKeys(g *EGraph, names []string, rowKeys []string) ([]Match, error) {
	matches, _, err := p.searchRelationKeysWithBudget(g, names, rowKeys, nil)
	return matches, err
}

func (p *Pattern) searchRelationKeysWithBudget(g *EGraph, names []string, rowKeys []string, budget *matchWorkBudget) ([]Match, bool, error) {
	if err := validatePattern(p); err != nil {
		return nil, false, fmt.Errorf("egraph: invalid pattern: %w", err)
	}
	if g == nil || !g.store.Initialized() || g.relations == nil {
		return nil, false, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	columns := p.RelationColumns
	if len(columns) == 0 && len(p.Children) > 0 {
		columns = make([]FunctionColumnPattern, len(p.Children))
		for i, child := range p.Children {
			columns[i] = FunctionClassPattern(child)
		}
	}
	table, err := g.relation(p.RelationName, len(columns))
	if err != nil {
		return nil, false, err
	}
	canonicalLiterals := make([]*FunctionValue, len(columns))
	var candidates map[string]struct{}
	for i, pattern := range columns {
		typ := table.schema.Columns[i]
		if pattern.Kind != 0 && pattern.Kind != typ.Kind {
			return nil, false, fmt.Errorf("egraph: relation pattern %q column %d has kind %d, want %d", p.RelationName, i, pattern.Kind, typ.Kind)
		}
		if pattern.Term != nil && typ.Kind != FunctionClassColumn {
			return nil, false, fmt.Errorf("egraph: relation pattern %q column %d requires a class pattern", p.RelationName, i)
		}
		if typ.Kind == FunctionClassColumn && pattern.Term == nil && !pattern.HasValue {
			return nil, false, fmt.Errorf("egraph: relation pattern %q column %d requires a class pattern", p.RelationName, i)
		}
		if typ.Kind != FunctionClassColumn && pattern.Term != nil {
			return nil, false, fmt.Errorf("egraph: relation pattern %q column %d requires a primitive pattern", p.RelationName, i)
		}
		if pattern.HasValue {
			value, err := g.canonicalRelationValue(typ, pattern.Value, fmt.Sprintf("relation pattern %q column %d", p.RelationName, i))
			if err != nil {
				return nil, false, err
			}
			canonicalLiterals[i] = &value
			if rowKeys != nil {
				columnRows := make(map[string]struct{})
				if i < len(table.index) {
					for key := range table.index[i][toCoreValue(value)] {
						if !budget.spend() {
							return nil, false, nil
						}
						columnRows[key] = struct{}{}
					}
				}
				if candidates == nil {
					candidates = columnRows
				} else {
					for key := range candidates {
						if !budget.spend() {
							return nil, false, nil
						}
						if _, ok := columnRows[key]; !ok {
							delete(candidates, key)
						}
					}
				}
			}
		}
	}
	if rowKeys != nil && candidates != nil && len(candidates) == 0 {
		return nil, true, nil
	}
	termVariables := make([]map[string]bool, len(columns))
	for i, pattern := range columns {
		if pattern.Term == nil {
			continue
		}
		if !budget.spendPattern(pattern.Term) {
			return nil, false, nil
		}
		termVariables[i] = make(map[string]bool)
		collectVariables(pattern.Term, termVariables[i])
	}
	var keys []string
	if rowKeys == nil {
		filters := make([]egraphcore.Filter[Id], 0, len(columns))
		for i, value := range canonicalLiterals {
			if value == nil {
				continue
			}
			coreValue, ok := functionValueToCore(*value)
			if !ok {
				return nil, false, fmt.Errorf("egraph: relation pattern %q column %d has unsupported literal", p.RelationName, i)
			}
			filters = append(filters, egraphcore.Filter[Id]{Column: i, Value: coreValue})
		}
		rows, complete, err := table.core.Query(filters, g.Find, coreWorkBudget(budget))
		if err != nil {
			return nil, false, fmt.Errorf("egraph: query relation pattern %q: %w", p.RelationName, err)
		}
		if !complete {
			return nil, false, nil
		}
		keys = make([]string, 0, len(rows))
		for _, coreRow := range rows {
			if !budget.spend() {
				return nil, false, nil
			}
			row := make([]FunctionValue, len(coreRow))
			for i, coreValue := range coreRow {
				value, ok := functionValueFromCore(coreValue)
				if !ok {
					return nil, false, fmt.Errorf("egraph: relation %q column %d has unsupported stored value", p.RelationName, i)
				}
				row[i] = value
			}
			keys = append(keys, typedFunctionKey(row))
		}
	} else {
		keys = rowKeys
	}
	if rowKeys != nil && candidates != nil {
		sourceKeys := keys
		keys = make([]string, 0, len(candidates))
		for _, key := range sourceKeys {
			if !budget.spend() {
				return nil, false, nil
			}
			if _, ok := candidates[key]; ok {
				keys = append(keys, key)
			}
		}
	}
	if !budget.spendSort(len(keys)) {
		return nil, false, nil
	}
	sort.Strings(keys)
	var matches []Match
	seen := make(map[string]bool)
	for _, key := range keys {
		if !budget.spend() {
			return nil, false, nil
		}
		if _, hidden := table.subsumed[key]; hidden {
			continue
		}
		row := table.rows[key]
		states := []Match{{Subst: make(Subst), BaseSubst: make(map[string]FunctionValue)}}
		for i, pattern := range columns {
			typ, value := table.schema.Columns[i], row[i]
			var next []Match
			for _, state := range states {
				if !budget.spend() {
					return nil, false, nil
				}
				if pattern.Term != nil {
					if typ.Kind != FunctionClassColumn {
						continue
					}
					crossKind := false
					for name := range termVariables[i] {
						if _, ok := state.BaseSubst[name]; ok {
							crossKind = true
							break
						}
					}
					if crossKind {
						continue
					}
					var substs []Subst
					complete := true
					substs, complete = pattern.Term.matchClassWithBudget(g, value.ID, state.Subst, budget)
					if !complete {
						return nil, false, nil
					}
					for _, subst := range substs {
						if !budget.spend() {
							return nil, false, nil
						}
						next = append(next, Match{Subst: subst, BaseSubst: cloneBaseSubst(state.BaseSubst)})
					}
					continue
				}
				if value.Kind != typ.Kind {
					continue
				}
				if pattern.HasValue {
					if canonicalLiterals[i] != nil && functionValuesEqual(*canonicalLiterals[i], value) {
						next = append(next, state)
					}
					continue
				}
				if pattern.Variable == "" {
					continue
				}
				if _, ok := state.Subst[pattern.Variable]; ok {
					continue
				}
				if old, ok := state.BaseSubst[pattern.Variable]; ok {
					if functionValuesEqual(old, value) {
						next = append(next, state)
					}
					continue
				}
				bases := cloneBaseSubst(state.BaseSubst)
				if !budget.spend() {
					return nil, false, nil
				}
				bases[pattern.Variable] = value
				next = append(next, Match{Subst: state.Subst, BaseSubst: bases})
			}
			states = next
			if len(states) == 0 {
				break
			}
		}
		for _, state := range states {
			matchKey := typedSubstitutionKey(state.Subst, state.BaseSubst, names, g)
			if seen[matchKey] {
				continue
			}
			seen[matchKey] = true
			if !budget.spend() {
				return nil, false, nil
			}
			matches = append(matches, Match{ClassID: InvalidId, Subst: state.Subst, BaseSubst: state.BaseSubst})
		}
	}
	return matches, true, nil
}

type functionValueRange interface {
	Range(int, egraphcore.Value[Id], func(string) bool) bool
}

func indexedPatternKeys(g *EGraph, patterns []*Pattern, index functionValueRange, allKeys []string) ([]string, error) {
	keys, _, err := indexedPatternKeysWithBudget(g, patterns, index, allKeys, nil)
	return keys, err
}

func indexedPatternKeysWithBudget(g *EGraph, patterns []*Pattern, index functionValueRange, allKeys []string, budget *matchWorkBudget) ([]string, bool, error) {
	var candidates map[string]struct{}
	for column, pattern := range patterns {
		if pattern.IsVar && pattern.Sort == "" {
			continue
		}
		matches, complete, err := pattern.searchChecked(g, budget)
		if err != nil {
			return nil, false, fmt.Errorf("egraph: search indexed column %d: %w", column, err)
		}
		if !complete {
			return nil, false, nil
		}
		columnRows := make(map[string]struct{})
		for _, match := range matches {
			if !budget.spend() {
				return nil, false, nil
			}
			id := g.Find(match.ClassID)
			stopped := false
			index.Range(column, egraphcore.Class(id), func(key string) bool {
				if !budget.spend() {
					stopped = true
					return false
				}
				columnRows[key] = struct{}{}
				return true
			})
			if stopped {
				return nil, false, nil
			}
		}
		if candidates == nil {
			candidates = columnRows
			continue
		}
		for key := range candidates {
			if !budget.spend() {
				return nil, false, nil
			}
			if _, ok := columnRows[key]; !ok {
				delete(candidates, key)
			}
		}
		if len(candidates) == 0 {
			return nil, true, nil
		}
	}
	if candidates == nil {
		for range allKeys {
			if !budget.spend() {
				return nil, false, nil
			}
		}
		keys := append([]string(nil), allKeys...)
		if !budget.spendSort(len(keys)) {
			return nil, false, nil
		}
		sort.Strings(keys)
		return keys, true, nil
	}
	keys := make([]string, 0, len(candidates))
	for key := range candidates {
		if !budget.spend() {
			return nil, false, nil
		}
		keys = append(keys, key)
	}
	if !budget.spendSort(len(keys)) {
		return nil, false, nil
	}
	sort.Strings(keys)
	return keys, true, nil
}

func (p *Pattern) matchClass(g *EGraph, classID Id, subst Subst) []Subst {
	results, _ := p.matchClassWithBudget(g, classID, subst, nil)
	return results
}

func (p *Pattern) matchClassWithBudget(g *EGraph, classID Id, subst Subst, budget *matchWorkBudget) ([]Subst, bool) {
	workBudget := coreWorkBudget(budget)
	if workBudget == nil {
		workBudget = egraphcore.NewWorkBudget(0)
	}
	results, err := egraphcore.MatchEClassWithMatcher(p, classID, subst, workBudget, egraphcore.MatcherCallbacks[Id, ENode, *Pattern]{
		Find:            g.Find,
		ValidPattern:    func(pattern *Pattern) bool { return pattern != nil },
		IsVariable:      func(pattern *Pattern) (string, bool) { return pattern.Op, pattern.IsVar },
		PatternChildren: func(pattern *Pattern) []*Pattern { return pattern.Children },
		ClassEligible: func(id Id, pattern *Pattern) bool {
			class := g.store.Class(id)
			return class != nil && (!pattern.IsVar || pattern.Sort == "" || class.Sort == pattern.Sort)
		},
		Candidates: func(id Id, pattern *Pattern) []ENode {
			class := g.store.Class(id)
			if class == nil {
				return nil
			}
			indexes := class.nodesByOpArity[nodeIndexKey{op: pattern.Op, arity: len(pattern.Children)}]
			candidates := make([]ENode, 0, len(indexes))
			for _, index := range indexes {
				if index >= 0 && index < len(class.Nodes) {
					candidates = append(candidates, class.Nodes[index])
				}
			}
			return candidates
		},
		NodeChildren: func(node ENode) []Id { return node.Children },
		NodeEligible: func(node ENode, pattern *Pattern) bool { return !node.Subsumed && pattern.matchesNode(node) },
	})
	if err != nil {
		return nil, false
	}
	converted := make([]Subst, len(results))
	for i, result := range results {
		converted[i] = Subst(result)
	}
	return converted, true
}

func (p *Pattern) matchesNode(node ENode) bool {
	if p.int64Expr != nil || p.float64Expr != nil || p.IsVar || node.Op != p.Op || len(node.Children) != len(p.Children) {
		return false
	}
	if p.Sort != "" && node.Sort != p.Sort {
		return false
	}
	if p.HasData {
		return reflect.DeepEqual(p.Data, node.Data)
	}
	if node.Data == nil {
		return true
	}
	// The textual pattern language cannot name arbitrary ENode payloads. Only
	// string leaves whose payload is exactly the printed symbol are literals.
	value, ok := node.Data.(string)
	return ok && len(node.Children) == 0 && value == node.Op
}

func (g *EGraph) subsumeMatch(pattern *Pattern, match Match) (bool, error) {
	classID, err := g.FindChecked(match.ClassID)
	if err != nil {
		return false, err
	}
	class := g.store.Class(classID)
	if class == nil {
		return false, fmt.Errorf("egraph: cannot subsume missing class %d", classID)
	}
	variables := make(map[string]bool)
	collectVariables(pattern, variables)
	matchedSubsumed := false
	for i, node := range class.Nodes {
		if !pattern.matchesNode(node) {
			continue
		}
		substs := []Subst{make(Subst)}
		for column, child := range pattern.Children {
			var next []Subst
			for _, subst := range substs {
				next = append(next, child.matchClass(g, node.Children[column], subst)...)
			}
			substs = next
			if len(substs) == 0 {
				break
			}
		}
		for _, subst := range substs {
			matches := true
			for name := range variables {
				want, wantOK := match.Subst[name]
				got, gotOK := subst[name]
				if !wantOK || !gotOK || g.Find(want) != g.Find(got) {
					matches = false
					break
				}
			}
			if !matches {
				continue
			}
			if node.Subsumed {
				matchedSubsumed = true
				continue
			}
			class.Nodes[i].Subsumed = true
			g.revision++
			g.recordTermChangeAndParents(classID)
			return true, nil
		}
	}
	if matchedSubsumed {
		return false, nil
	}
	return false, fmt.Errorf("egraph: rewrite match in class %d has no matching node to subsume", classID)
}

// Instantiate creates corresponding concrete nodes in the E-Graph using bindings in subst.
func (p *Pattern) Instantiate(g *EGraph, subst Subst) Id {
	id, err := p.InstantiateChecked(g, subst)
	if err != nil {
		return InvalidId
	}
	return id
}

// InstantiateChecked instantiates a pattern and reports unbound variables or
// invalid class IDs instead of panicking.
func (p *Pattern) InstantiateChecked(g *EGraph, subst Subst) (Id, error) {
	return p.InstantiateWithMatchChecked(g, Match{Subst: subst})
}

// InstantiateWithMatchChecked instantiates a pattern using both class and
// primitive bindings. It is required for patterns containing PatInt64Expr.
func (p *Pattern) InstantiateWithMatchChecked(g *EGraph, match Match) (Id, error) {
	if g == nil {
		return InvalidId, fmt.Errorf("egraph: nil graph")
	}
	if err := validatePattern(p); err != nil {
		return InvalidId, fmt.Errorf("egraph: invalid pattern: %w", err)
	}
	if p.RelationName != "" || p.FunctionName != "" {
		name := p.RelationName
		kind := "relation"
		if p.FunctionName != "" {
			name = p.FunctionName
			kind = "function"
		}
		return InvalidId, fmt.Errorf("egraph: %s pattern %q cannot be instantiated as an e-node", kind, name)
	}
	type frame struct {
		pattern *Pattern
		visited bool
	}
	ids := make(map[*Pattern]Id)
	stack := []frame{{pattern: p}}
	for len(stack) > 0 {
		last := len(stack) - 1
		current := stack[last]
		stack = stack[:last]
		if id, ok := ids[current.pattern]; ok {
			if current.pattern == p && len(stack) == 0 {
				return id, nil
			}
			continue
		}
		if current.pattern.int64Expr != nil || current.pattern.float64Expr != nil {
			expr := current.pattern.int64Expr
			sort := "i64"
			if current.pattern.float64Expr != nil {
				expr = current.pattern.float64Expr
				sort = "f64"
			}
			value, ok := match.computedValues[current.pattern]
			if !ok {
				var err error
				value, err = evaluateFunctionActionExpr(g, expr, match)
				if err != nil {
					return InvalidId, fmt.Errorf("egraph: computed %s leaf: %w", sort, err)
				}
			}
			want := FunctionInt64Column
			if sort == "f64" {
				want = FunctionFloat64Column
			}
			if value.Kind != want {
				return InvalidId, fmt.Errorf("egraph: computed leaf has kind %d, want %s", value.Kind, sort)
			}
			payload := any(value.Int64)
			if sort == "f64" {
				if math.IsNaN(value.Float64) {
					return InvalidId, fmt.Errorf("egraph: computed f64 leaf rejects NaN")
				}
				floatValue := value.Float64
				if floatValue == 0 {
					floatValue = 0
				}
				payload = floatValue
			}
			id, err := g.AddChecked(NewSortedLeaf(sort, payload))
			if err != nil {
				return InvalidId, err
			}
			ids[current.pattern] = id
			continue
		}
		if current.pattern.IsVar {
			id, ok := match.Subst[current.pattern.Op]
			if !ok {
				return InvalidId, fmt.Errorf("egraph: unbound pattern variable %s during instantiation", current.pattern.Op)
			}
			canonical, err := g.FindChecked(id)
			if err != nil {
				return InvalidId, err
			}
			class := g.store.Class(canonical)
			if class == nil || (current.pattern.Sort != "" && class.Sort != current.pattern.Sort) {
				return InvalidId, fmt.Errorf("egraph: pattern variable %s requires sort %q", current.pattern.Op, current.pattern.Sort)
			}
			ids[current.pattern] = canonical
			continue
		}
		if !current.visited {
			stack = append(stack, frame{pattern: current.pattern, visited: true})
			for i := len(current.pattern.Children) - 1; i >= 0; i-- {
				stack = append(stack, frame{pattern: current.pattern.Children[i]})
			}
			continue
		}
		childIDs := make([]Id, len(current.pattern.Children))
		for i, child := range current.pattern.Children {
			id, ok := ids[child]
			if !ok {
				return InvalidId, fmt.Errorf("egraph: pattern child was not instantiated")
			}
			childIDs[i] = id
		}
		data := current.pattern.Data
		if !current.pattern.HasData {
			data = nil
		}
		id, err := g.AddChecked(ENode{Op: current.pattern.Op, Sort: current.pattern.Sort, Data: data, Children: childIDs})
		if err != nil {
			return InvalidId, err
		}
		ids[current.pattern] = id
	}
	return ids[p], nil
}
