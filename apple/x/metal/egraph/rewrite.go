package egraph

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/egraphcore"
)

// Rewrite defines a rule that joins left-hand patterns and applies union or
// analysis merge actions.
type Rewrite struct {
	Name                   string
	LHS                    *Pattern
	RHS                    *Pattern
	Guard                  RewriteGuard
	conditions             []*Pattern
	actions                []*Pattern
	unions                 []UnionAction
	facts                  []AnalysisAction
	relations              []RelationAction
	functions              []FunctionAction
	primitiveConditions    []FunctionActionExpr
	classConditions        []ClassCondition
	classPatternConditions []classPatternCondition
	primitiveBindings      []PrimitiveBinding
	nodeDeletes            []NodeDeleteAction
	subsumesLHS            bool
}

// PrimitiveBinding names the result of a computed primitive expression in a
// rewrite. Bindings are evaluated in slice order before conditions and actions.
type PrimitiveBinding struct {
	Name string
	Expr FunctionActionExpr
}

// AnalysisAction merges a fact into the analysis value of a matched e-class.
// Fact must satisfy the graph Analysis implementation's immutability contract.
type AnalysisAction struct {
	Variable string
	Fact     any
}

// UnionAction merges the classes matched by Left and Right.
type UnionAction struct {
	Left  *Pattern
	Right *Pattern
}

// ClassCondition requires two LHS-bound e-class variables to be either in the
// same equivalence class or in different equivalence classes, according to
// Equal. The comparison uses current canonical representatives at match time.
type ClassCondition struct {
	Left  string
	Right string
	Equal bool
}

type classPatternCondition struct {
	Variable string
	Pattern  *Pattern
	Equal    bool
}

// RelationActionKind identifies a mutation to a predicate relation.
type RelationActionKind uint8

const (
	RelationInsert RelationActionKind = iota + 1
	RelationDelete
	RelationSubsume
)

// RelationAction inserts, deletes, or subsumes a tuple in a registered
// predicate relation. Columns are term patterns whose variables must be bound
// by lhs.
type RelationAction struct {
	Kind         RelationActionKind
	Name         string
	Columns      []*Pattern
	Typed        bool
	TypedColumns []FunctionColumnPattern
}

// FunctionActionKind identifies a function-table mutation.
type FunctionActionKind uint8

const (
	FunctionSetAction FunctionActionKind = iota + 1
	FunctionDeleteAction
	FunctionSubsumeAction
)

// FunctionAction sets, deletes, or subsumes a row. Keys and Value are term
// patterns whose variables must be bound by a left-hand pattern. Delete and
// subsume require Value to be nil.
type FunctionAction struct {
	Kind       FunctionActionKind
	Name       string
	Keys       []*Pattern
	Value      *Pattern
	Typed      bool // Marks a typed action with zero key columns.
	TypedKeys  []FunctionColumnPattern
	TypedValue *FunctionColumnPattern
}

// NewAnalysisAction creates an action that merges fact into a matched class.
func NewAnalysisAction(variable string, fact any) AnalysisAction {
	if !strings.HasPrefix(variable, "?") {
		variable = "?" + variable
	}
	return AnalysisAction{Variable: variable, Fact: fact}
}

// RewriteGuard decides whether a structural match is valid for a rewrite.
// The match is a copy; guards must not mutate the graph.
type RewriteGuard func(g *EGraph, match Match) (bool, error)

// WithClassCondition adds an equality or inequality check over two e-class
// variables bound by the rewrite's LHS patterns. Names may be written with or
// without a leading '?'. Conditions are evaluated against canonical class
// representatives after matching and before any rewrite mutation.
func (r *Rewrite) WithClassCondition(leftVar, rightVar string, equal bool) error {
	if r == nil {
		return fmt.Errorf("egraph: nil rewrite")
	}
	if !strings.HasPrefix(leftVar, "?") {
		leftVar = "?" + leftVar
	}
	if !strings.HasPrefix(rightVar, "?") {
		rightVar = "?" + rightVar
	}
	if err := r.validate(); err != nil {
		return err
	}
	classes := functionActionClassBindings(r.leftPatterns())
	if !classes[leftVar] {
		return fmt.Errorf("egraph: rewrite %q class condition variable %s is not class-bound by lhs", r.Name, leftVar)
	}
	if !classes[rightVar] {
		return fmt.Errorf("egraph: rewrite %q class condition variable %s is not class-bound by lhs", r.Name, rightVar)
	}
	r.classConditions = append(r.classConditions, ClassCondition{Left: leftVar, Right: rightVar, Equal: equal})
	return nil
}

// WithClassPatternCondition adds an equality or inequality check between an
// LHS-bound e-class and a ground term pattern. Equality requires the bound
// class to match the pattern; inequality requires it not to match. If the
// pattern has no matching class, equality is false and inequality is true.
func (r *Rewrite) WithClassPatternCondition(variable string, pattern *Pattern, equal bool) error {
	if r == nil {
		return fmt.Errorf("egraph: nil rewrite")
	}
	if !strings.HasPrefix(variable, "?") {
		variable = "?" + variable
	}
	if err := r.validate(); err != nil {
		return err
	}
	classes := functionActionClassBindings(r.leftPatterns())
	if !classes[variable] {
		return fmt.Errorf("egraph: rewrite %q class pattern condition variable %s is not class-bound by lhs", r.Name, variable)
	}
	if err := validateTermPattern(pattern); err != nil {
		return fmt.Errorf("egraph: rewrite %q class pattern condition: %w", r.Name, err)
	}
	if pattern.IsVar || patternHasComputedLeaf(pattern) {
		return fmt.Errorf("egraph: rewrite %q class pattern condition requires a ground term pattern", r.Name)
	}
	variables := make(map[string]bool)
	collectVariables(pattern, variables)
	if len(variables) != 0 {
		return fmt.Errorf("egraph: rewrite %q class pattern condition requires a ground term pattern", r.Name)
	}
	r.classPatternConditions = append(r.classPatternConditions, classPatternCondition{Variable: variable, Pattern: pattern, Equal: equal})
	return nil
}

// WithPrimitiveCondition adds a typed boolean condition evaluated for each
// joined LHS match before any rewrite action is applied. Expressions may read
// primitive function/relation bindings with FunctionActionVar and may project
// i64 literals from matched e-classes with FunctionActionInt64FromClass.
func (r *Rewrite) WithPrimitiveCondition(expr FunctionActionExpr) error {
	if r == nil {
		return fmt.Errorf("egraph: nil rewrite")
	}
	if err := r.validate(); err != nil {
		return err
	}
	bindings, err := functionActionBindingTypes(r.leftPatterns())
	if err != nil {
		return err
	}
	classes := functionActionClassBindings(r.leftPatterns())
	for _, binding := range r.primitiveBindings {
		kind, err := functionActionExpressionTypeWithClasses(&binding.Expr, bindings, classes)
		if err != nil {
			return fmt.Errorf("egraph: rewrite %q primitive binding %s: %w", r.Name, binding.Name, err)
		}
		bindings[binding.Name] = kind
	}
	kind, err := functionActionExpressionTypeWithClasses(&expr, bindings, classes)
	if err != nil {
		return fmt.Errorf("egraph: rewrite %q primitive condition: %w", r.Name, err)
	}
	if kind != FunctionBoolColumn {
		return fmt.Errorf("egraph: rewrite %q primitive condition has kind %d, want bool", r.Name, kind)
	}
	cloned := cloneFunctionActionExprs([]FunctionActionExpr{expr})[0]
	r.primitiveConditions = append(r.primitiveConditions, cloned)
	return nil
}

// NewPrimitivePatternRewrite creates a term rewrite with computed primitive
// query bindings. Each binding is evaluated in order for every LHS match and
// can be used by later bindings, conditions, and RHS/action expressions.
func NewPrimitivePatternRewrite(name string, lhs, rhs *Pattern, bindings []PrimitiveBinding) (*Rewrite, error) {
	return NewPrimitiveRule(name, []*Pattern{lhs}, []*Pattern{rhs}, nil, nil, nil, nil, bindings)
}

// NewPrimitiveRule creates a joined rewrite with computed primitive query
// bindings and union, analysis, relation, or function actions. Bindings are
// evaluated in declaration order before conditions and actions. The first
// left-hand pattern must be a term pattern when rhs is non-empty.
func NewPrimitiveRule(name string, lhs, rhs []*Pattern, unions []UnionAction, facts []AnalysisAction, relations []RelationAction, functions []FunctionAction, bindings []PrimitiveBinding) (*Rewrite, error) {
	r, err := newActionRewriteWithPrimitiveBindings(name, lhs, rhs, unions, facts, relations, functions, nil, bindings)
	if err != nil {
		return nil, err
	}
	return r, nil
}

// NewRewrite parses left-hand and right-hand pattern strings and creates a Rewrite rule.
func NewRewrite(name, lhsStr, rhsStr string) (*Rewrite, error) {
	lhs, err := ParsePattern(lhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse lhs %q: %w", lhsStr, err)
	}
	rhs, err := ParsePattern(rhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse rhs %q: %w", rhsStr, err)
	}
	return NewPatternRewrite(name, lhs, rhs)
}

// NewSubsumingRewrite parses patterns and creates a rewrite that subsumes its
// matched left-hand node after applying the right-hand side.
func NewSubsumingRewrite(name, lhsStr, rhsStr string) (*Rewrite, error) {
	lhs, err := ParsePattern(lhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse lhs %q: %w", lhsStr, err)
	}
	rhs, err := ParsePattern(rhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse rhs %q: %w", rhsStr, err)
	}
	return NewSubsumingPatternRewrite(name, lhs, rhs)
}

// NewGuardedRewrite parses string patterns and creates a rewrite that applies
// only when guard accepts each structural match. The guard must not mutate
// the graph.
func NewGuardedRewrite(name, lhsStr, rhsStr string, guard RewriteGuard) (*Rewrite, error) {
	lhs, err := ParsePattern(lhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse lhs %q: %w", lhsStr, err)
	}
	rhs, err := ParsePattern(rhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse rhs %q: %w", rhsStr, err)
	}
	return NewGuardedPatternRewrite(name, lhs, rhs, guard)
}

// NewPatternRewrite creates a rewrite from patterns built programmatically.
// Use it with PatWithData when rules depend on typed or literal payloads.
func NewPatternRewrite(name string, lhs, rhs *Pattern) (*Rewrite, error) {
	return newPatternRewrite(name, lhs, rhs, nil)
}

// NewSubsumingPatternRewrite creates a term rewrite that hides its matched
// left-hand node from later rule matching and extraction after applying RHS.
func NewSubsumingPatternRewrite(name string, lhs, rhs *Pattern) (*Rewrite, error) {
	r, err := newPatternRewrite(name, lhs, rhs, nil)
	if err != nil {
		return nil, err
	}
	r.subsumesLHS = true
	if err := r.validate(); err != nil {
		return nil, err
	}
	return r, nil
}

// NewMultiPatternRewrite creates a rewrite whose left-hand patterns join on
// shared variables. The first pattern identifies the class unified with RHS.
func NewMultiPatternRewrite(name string, lhs []*Pattern, rhs *Pattern) (*Rewrite, error) {
	return newMultiPatternRewrite(name, lhs, rhs, nil)
}

// NewMultiUnionRewrite creates a rule that joins its left-hand patterns and
// unions every right-hand pattern with the class matched by the first LHS.
func NewMultiUnionRewrite(name string, lhs, rhs []*Pattern) (*Rewrite, error) {
	return newMultiUnionRewrite(name, lhs, rhs, nil)
}

// NewMultiActionRewrite creates a joined rewrite with union and analysis
// merge actions. Every action variable must be bound by a left-hand pattern.
func NewMultiActionRewrite(name string, lhs, rhs []*Pattern, facts []AnalysisAction) (*Rewrite, error) {
	return newMultiActionRewrite(name, lhs, rhs, nil, facts, nil)
}

// NewRule creates a rule with joined left-hand patterns and arbitrary union
// and analysis merge actions. Every action variable must be bound by lhs.
func NewRule(name string, lhs []*Pattern, unions []UnionAction, facts []AnalysisAction) (*Rewrite, error) {
	return newMultiActionRewrite(name, lhs, nil, unions, facts, nil)
}

// NewGuardedRule creates a rule whose actions run only when guard accepts the
// joined match. The guard must not mutate the graph.
func NewGuardedRule(name string, lhs []*Pattern, unions []UnionAction, facts []AnalysisAction, guard RewriteGuard) (*Rewrite, error) {
	if guard == nil {
		return nil, fmt.Errorf("egraph: nil rewrite guard")
	}
	return newActionRewrite(name, lhs, nil, unions, facts, nil, nil, guard)
}

// NewRelationRule creates a rule with union, analysis, and predicate relation
// insert/delete/subsume actions.
func NewRelationRule(name string, lhs []*Pattern, unions []UnionAction, relations []RelationAction, facts []AnalysisAction) (*Rewrite, error) {
	return newActionRewrite(name, lhs, nil, unions, facts, relations, nil, nil)
}

// NewGuardedRelationRule creates a guarded rule with predicate relation
// insert/delete/subsume actions.
func NewGuardedRelationRule(name string, lhs []*Pattern, unions []UnionAction, relations []RelationAction, facts []AnalysisAction, guard RewriteGuard) (*Rewrite, error) {
	if guard == nil {
		return nil, fmt.Errorf("egraph: nil rewrite guard")
	}
	return newActionRewrite(name, lhs, nil, unions, facts, relations, nil, guard)
}

// NewFunctionRule creates a rule with function table set/delete/subsume actions.
func NewFunctionRule(name string, lhs []*Pattern, unions []UnionAction, functions []FunctionAction, facts []AnalysisAction) (*Rewrite, error) {
	return newActionRewrite(name, lhs, nil, unions, facts, nil, functions, nil)
}

// NewGuardedFunctionRule creates a guarded rule with function table
// set/delete/subsume actions.
func NewGuardedFunctionRule(name string, lhs []*Pattern, unions []UnionAction, functions []FunctionAction, facts []AnalysisAction, guard RewriteGuard) (*Rewrite, error) {
	if guard == nil {
		return nil, fmt.Errorf("egraph: nil rewrite guard")
	}
	return newActionRewrite(name, lhs, nil, unions, facts, nil, functions, guard)
}

// NewGuardedPatternRewrite creates a rewrite that applies only when guard
// accepts each structural match. The guard must not mutate the graph.
func NewGuardedPatternRewrite(name string, lhs, rhs *Pattern, guard RewriteGuard) (*Rewrite, error) {
	if guard == nil {
		return nil, fmt.Errorf("egraph: nil rewrite guard")
	}
	return newPatternRewrite(name, lhs, rhs, guard)
}

// NewGuardedMultiPatternRewrite creates a guarded rewrite from joined
// left-hand patterns. The first pattern identifies the class unified with RHS.
func NewGuardedMultiPatternRewrite(name string, lhs []*Pattern, rhs *Pattern, guard RewriteGuard) (*Rewrite, error) {
	if guard == nil {
		return nil, fmt.Errorf("egraph: nil rewrite guard")
	}
	return newMultiPatternRewrite(name, lhs, rhs, guard)
}

// NewGuardedMultiUnionRewrite creates a guarded multi-pattern rule with
// multiple union actions.
func NewGuardedMultiUnionRewrite(name string, lhs, rhs []*Pattern, guard RewriteGuard) (*Rewrite, error) {
	if guard == nil {
		return nil, fmt.Errorf("egraph: nil rewrite guard")
	}
	return newMultiUnionRewrite(name, lhs, rhs, guard)
}

// NewGuardedMultiActionRewrite creates a guarded joined rewrite with union
// and analysis merge actions.
func NewGuardedMultiActionRewrite(name string, lhs, rhs []*Pattern, facts []AnalysisAction, guard RewriteGuard) (*Rewrite, error) {
	if guard == nil {
		return nil, fmt.Errorf("egraph: nil rewrite guard")
	}
	return newMultiActionRewrite(name, lhs, rhs, nil, facts, guard)
}

func newPatternRewrite(name string, lhs, rhs *Pattern, guard RewriteGuard) (*Rewrite, error) {
	return newMultiPatternRewrite(name, []*Pattern{lhs}, rhs, guard)
}

func newMultiPatternRewrite(name string, lhs []*Pattern, rhs *Pattern, guard RewriteGuard) (*Rewrite, error) {
	return newMultiUnionRewrite(name, lhs, []*Pattern{rhs}, guard)
}

func newMultiUnionRewrite(name string, lhs, rhs []*Pattern, guard RewriteGuard) (*Rewrite, error) {
	return newMultiActionRewrite(name, lhs, rhs, nil, nil, guard)
}

func newMultiActionRewrite(name string, lhs, rhs []*Pattern, unions []UnionAction, facts []AnalysisAction, guard RewriteGuard) (*Rewrite, error) {
	return newActionRewrite(name, lhs, rhs, unions, facts, nil, nil, guard)
}

func newActionRewrite(name string, lhs, rhs []*Pattern, unions []UnionAction, facts []AnalysisAction, relations []RelationAction, functions []FunctionAction, guard RewriteGuard) (*Rewrite, error) {
	return newActionRewriteWithPrimitiveBindings(name, lhs, rhs, unions, facts, relations, functions, guard, nil)
}

func newActionRewriteWithPrimitiveBindings(name string, lhs, rhs []*Pattern, unions []UnionAction, facts []AnalysisAction, relations []RelationAction, functions []FunctionAction, guard RewriteGuard, primitiveBindings []PrimitiveBinding) (*Rewrite, error) {
	if len(lhs) == 0 || len(rhs)+len(unions)+len(facts)+len(relations)+len(functions) == 0 {
		return nil, fmt.Errorf("egraph: rewrite needs a left-hand pattern and at least one action")
	}
	if len(rhs) > 0 && lhs[0] != nil && (lhs[0].RelationName != "" || lhs[0].FunctionName != "") {
		return nil, fmt.Errorf("egraph: RHS rewrites require a term pattern as the first LHS")
	}
	bound := make(map[string]bool)
	for i, pattern := range lhs {
		if err := validatePattern(pattern); err != nil {
			return nil, fmt.Errorf("egraph: invalid lhs pattern %d: %w", i, err)
		}
		if patternHasComputedLeaf(pattern) {
			return nil, fmt.Errorf("egraph: lhs pattern %d contains a computed i64 leaf", i)
		}
		collectVariables(pattern, bound)
	}
	actionBindings, err := functionActionBindingTypes(lhs)
	if err != nil {
		return nil, fmt.Errorf("egraph: invalid primitive action bindings: %w", err)
	}
	classBindings := functionActionClassBindings(lhs)
	bindings := make([]PrimitiveBinding, len(primitiveBindings))
	for i, binding := range primitiveBindings {
		binding.Name = patternVariableName(binding.Name)
		if len(binding.Name) < 2 || binding.Expr.kind == 0 {
			return nil, fmt.Errorf("egraph: primitive binding %d requires a name and expression", i)
		}
		if classBindings[binding.Name] {
			return nil, fmt.Errorf("egraph: primitive binding %s conflicts with an e-class binding", binding.Name)
		}
		kind, err := functionActionExpressionTypeWithClasses(&binding.Expr, actionBindings, classBindings)
		if err != nil {
			return nil, fmt.Errorf("egraph: primitive binding %s: %w", binding.Name, err)
		}
		if old, ok := actionBindings[binding.Name]; ok && old != kind {
			return nil, fmt.Errorf("egraph: primitive binding %s has kind %d, existing binding has kind %d", binding.Name, kind, old)
		}
		bindings[i] = PrimitiveBinding{Name: binding.Name, Expr: cloneFunctionActionExprs([]FunctionActionExpr{binding.Expr})[0]}
		actionBindings[binding.Name] = kind
		bound[binding.Name] = true
	}
	var unbound []string
	for i, action := range rhs {
		if err := validateTermPattern(action); err != nil {
			return nil, fmt.Errorf("egraph: invalid rhs pattern %d: %w", i, err)
		}
		rhsVars := make(map[string]bool)
		collectVariables(action, rhsVars)
		for name := range rhsVars {
			if !bound[name] {
				unbound = append(unbound, name)
			}
		}
		if err := validateComputedPatterns(action, actionBindings, classBindings); err != nil {
			return nil, fmt.Errorf("egraph: invalid rhs pattern %d: %w", i, err)
		}
	}
	for i, action := range unions {
		for _, pattern := range []*Pattern{action.Left, action.Right} {
			if err := validateTermPattern(pattern); err != nil {
				return nil, fmt.Errorf("egraph: invalid union action %d: %w", i, err)
			}
			variables := make(map[string]bool)
			collectVariables(pattern, variables)
			for name := range variables {
				if !bound[name] {
					unbound = append(unbound, name)
				}
			}
		}
	}
	for i, action := range relations {
		if action.Kind != RelationInsert && action.Kind != RelationDelete && action.Kind != RelationSubsume {
			return nil, fmt.Errorf("egraph: relation action %d has invalid kind %d", i, action.Kind)
		}
		if action.Name == "" {
			return nil, fmt.Errorf("egraph: relation action %d has empty relation name", i)
		}
		if action.Typed || action.TypedColumns != nil {
			if len(action.Columns) != 0 {
				return nil, fmt.Errorf("egraph: relation action %d has both class and typed columns", i)
			}
			for _, column := range action.TypedColumns {
				if err := validateFunctionActionColumn(column); err != nil {
					return nil, fmt.Errorf("egraph: invalid relation action %d: %w", i, err)
				}
				if column.Variable != "" && !bound[column.Variable] {
					unbound = append(unbound, column.Variable)
				}
				if column.Variable != "" && column.Kind != FunctionClassColumn && actionBindings[column.Variable] != column.Kind {
					return nil, fmt.Errorf("egraph: relation action %d variable %s has no matching primitive binding", i, column.Variable)
				}
				if column.Expr != nil {
					kind, err := functionActionExpressionTypeWithClasses(column.Expr, actionBindings, classBindings)
					if err != nil {
						return nil, fmt.Errorf("egraph: relation action %d expression: %w", i, err)
					}
					if kind != functionActionDynamicColumn && kind != column.Kind {
						return nil, fmt.Errorf("egraph: relation action %d expression kind %d does not match column kind %d", i, kind, column.Kind)
					}
					vars := make(map[string]bool)
					collectFunctionActionExprVars(column.Expr, vars)
					for name := range vars {
						if !bound[name] {
							unbound = append(unbound, name)
						}
					}
				}
			}
			continue
		}
		for _, pattern := range action.Columns {
			if err := validateTermPattern(pattern); err != nil {
				return nil, fmt.Errorf("egraph: invalid relation action %d: %w", i, err)
			}
			variables := make(map[string]bool)
			collectVariables(pattern, variables)
			for name := range variables {
				if !bound[name] {
					unbound = append(unbound, name)
				}
			}
		}
	}
	for i, action := range functions {
		if action.Kind != FunctionSetAction && action.Kind != FunctionDeleteAction && action.Kind != FunctionSubsumeAction {
			return nil, fmt.Errorf("egraph: function action %d has invalid kind %d", i, action.Kind)
		}
		if action.Name == "" {
			return nil, fmt.Errorf("egraph: function action %d has empty function name", i)
		}
		typedAction := action.Typed || action.TypedKeys != nil || action.TypedValue != nil
		if typedAction && (len(action.Keys) != 0 || action.Value != nil) {
			return nil, fmt.Errorf("egraph: function action %d mixes typed and e-class columns", i)
		}
		if action.Kind == FunctionSetAction && action.Value == nil && action.TypedValue == nil {
			return nil, fmt.Errorf("egraph: function set action %d has no value", i)
		}
		if action.Kind != FunctionSetAction && (action.Value != nil || action.TypedValue != nil) {
			return nil, fmt.Errorf("egraph: function mutation action %d has a value", i)
		}
		patterns := append([]*Pattern(nil), action.Keys...)
		if action.Value != nil {
			patterns = append(patterns, action.Value)
		}
		for _, pattern := range patterns {
			if err := validateTermPattern(pattern); err != nil {
				return nil, fmt.Errorf("egraph: invalid function action %d: %w", i, err)
			}
			variables := make(map[string]bool)
			collectVariables(pattern, variables)
			for name := range variables {
				if !bound[name] {
					unbound = append(unbound, name)
				}
			}
		}
		for _, column := range action.TypedKeys {
			if err := validateFunctionActionColumn(column); err != nil {
				return nil, fmt.Errorf("egraph: invalid function action %d: %w", i, err)
			}
			if column.Expr != nil {
				typ, err := functionActionExpressionTypeWithClasses(column.Expr, actionBindings, classBindings)
				if err != nil {
					return nil, fmt.Errorf("egraph: invalid function action %d key expression: %w", i, err)
				}
				if typ != functionActionDynamicColumn && typ != column.Kind {
					return nil, fmt.Errorf("egraph: function action %d key expression has kind %d, column requires kind %d", i, typ, column.Kind)
				}
			}
			if column.Term != nil {
				variables := make(map[string]bool)
				collectVariables(column.Term, variables)
				for name := range variables {
					if !bound[name] {
						unbound = append(unbound, name)
					}
				}
			}
			if column.Variable != "" && !bound[column.Variable] {
				unbound = append(unbound, column.Variable)
			}
		}
		if action.TypedValue != nil {
			if err := validateFunctionActionColumn(*action.TypedValue); err != nil {
				return nil, fmt.Errorf("egraph: invalid function action %d value: %w", i, err)
			}
			if action.TypedValue.Expr != nil {
				typ, err := functionActionExpressionTypeWithClasses(action.TypedValue.Expr, actionBindings, classBindings)
				if err != nil {
					return nil, fmt.Errorf("egraph: invalid function action %d value expression: %w", i, err)
				}
				if typ != functionActionDynamicColumn && typ != action.TypedValue.Kind {
					return nil, fmt.Errorf("egraph: function action %d value expression has kind %d, column requires kind %d", i, typ, action.TypedValue.Kind)
				}
			}
			if action.TypedValue.Term != nil {
				variables := make(map[string]bool)
				collectVariables(action.TypedValue.Term, variables)
				for name := range variables {
					if !bound[name] {
						unbound = append(unbound, name)
					}
				}
			}
			if action.TypedValue.Variable != "" && !bound[action.TypedValue.Variable] {
				unbound = append(unbound, action.TypedValue.Variable)
			}
		}
	}
	for _, action := range facts {
		if len(action.Variable) < 2 || !strings.HasPrefix(action.Variable, "?") {
			return nil, fmt.Errorf("egraph: analysis action variable %q must start with ? and have a name", action.Variable)
		}
		if !bound[action.Variable] {
			unbound = append(unbound, action.Variable)
		}
	}
	if len(unbound) > 0 {
		sort.Strings(unbound)
		return nil, fmt.Errorf("egraph: action variables not bound by lhs: %s", strings.Join(unbound, ", "))
	}
	patterns := append([]*Pattern(nil), lhs...)
	actions := append([]*Pattern(nil), rhs...)
	factActions := append([]AnalysisAction(nil), facts...)
	unionActions := append([]UnionAction(nil), unions...)
	relationActions := cloneRelationActions(relations)
	functionActions := cloneFunctionActions(functions)
	var firstRHS *Pattern
	if len(actions) > 0 {
		firstRHS = actions[0]
	}
	return &Rewrite{
		Name:              name,
		LHS:               patterns[0],
		RHS:               firstRHS,
		Guard:             guard,
		conditions:        patterns[1:],
		actions:           actions[1:],
		unions:            unionActions,
		facts:             factActions,
		relations:         relationActions,
		functions:         functionActions,
		primitiveBindings: bindings,
	}, nil
}

func patternHasComputedLeaf(pattern *Pattern) bool {
	stack := []*Pattern{pattern}
	seen := make(map[*Pattern]bool)
	for len(stack) != 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if current == nil || seen[current] {
			continue
		}
		seen[current] = true
		if current.int64Expr != nil || current.float64Expr != nil {
			return true
		}
		stack = append(stack, current.Children...)
		for _, columns := range [][]FunctionColumnPattern{current.FunctionColumns, current.RelationColumns} {
			for _, column := range columns {
				stack = append(stack, column.Term)
			}
		}
	}
	return false
}

func cloneFunctionActions(actions []FunctionAction) []FunctionAction {
	cloned := make([]FunctionAction, len(actions))
	for i, action := range actions {
		cloned[i] = action
		cloned[i].Keys = append([]*Pattern(nil), action.Keys...)
		cloned[i].TypedKeys = append([]FunctionColumnPattern(nil), action.TypedKeys...)
		for j := range cloned[i].TypedKeys {
			cloned[i].TypedKeys[j].Expr = cloneFunctionActionExprPtr(action.TypedKeys[j].Expr)
		}
		if action.TypedValue != nil {
			value := *action.TypedValue
			value.Expr = cloneFunctionActionExprPtr(action.TypedValue.Expr)
			cloned[i].TypedValue = &value
		}
	}
	return cloned
}

func cloneFunctionActionExprPtr(expr *FunctionActionExpr) *FunctionActionExpr {
	if expr == nil {
		return nil
	}
	cloned := cloneFunctionActionExprs([]FunctionActionExpr{*expr})[0]
	return &cloned
}

func cloneRelationActions(actions []RelationAction) []RelationAction {
	cloned := make([]RelationAction, len(actions))
	for i, action := range actions {
		cloned[i] = action
		cloned[i].Columns = append([]*Pattern(nil), action.Columns...)
		cloned[i].TypedColumns = append([]FunctionColumnPattern(nil), action.TypedColumns...)
		for j := range cloned[i].TypedColumns {
			cloned[i].TypedColumns[j].Expr = cloneFunctionActionExprPtr(action.TypedColumns[j].Expr)
		}
	}
	return cloned
}

func validateTermPattern(pattern *Pattern) error {
	if err := validatePattern(pattern); err != nil {
		return err
	}
	stack := []*Pattern{pattern}
	for len(stack) > 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if current.RelationName != "" || current.FunctionName != "" {
			return fmt.Errorf("relation or function pattern %q cannot be used as a term", current.Op)
		}
		stack = append(stack, current.Children...)
	}
	return nil
}

func validateFunctionActionColumn(column FunctionColumnPattern) error {
	if column.Expr != nil {
		if column.Term != nil || column.Variable != "" || column.HasValue {
			return fmt.Errorf("column has more than one value expression")
		}
		if column.Kind < FunctionInt64Column || column.Kind > FunctionBigRatColumn {
			return fmt.Errorf("invalid expression column kind %d", column.Kind)
		}
		return nil
	}
	if column.Term != nil {
		if column.Kind != 0 && column.Kind != FunctionClassColumn {
			return fmt.Errorf("term expression has primitive kind")
		}
		return validateTermPattern(column.Term)
	}
	if column.Kind < FunctionInt64Column || column.Kind > FunctionBigRatColumn {
		return fmt.Errorf("invalid primitive kind %d", column.Kind)
	}
	if column.HasValue && column.Variable != "" {
		return fmt.Errorf("column has both literal and variable")
	}
	if !column.HasValue && (len(column.Variable) < 2 || !strings.HasPrefix(column.Variable, "?")) {
		return fmt.Errorf("column requires a variable or literal")
	}
	return nil
}

func functionActionBindingTypes(lhs []*Pattern) (map[string]FunctionColumnKind, error) {
	bindings := make(map[string]FunctionColumnKind)
	for _, root := range lhs {
		stack := []*Pattern{root}
		seen := make(map[*Pattern]bool)
		for len(stack) > 0 {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if p == nil || seen[p] {
				continue
			}
			seen[p] = true
			for _, column := range p.FunctionColumns {
				if column.Variable == "" || column.Kind == 0 || column.Kind == FunctionClassColumn {
					continue
				}
				if old, ok := bindings[column.Variable]; ok && old != column.Kind {
					return nil, fmt.Errorf("variable %s is bound with conflicting primitive kinds %d and %d", column.Variable, old, column.Kind)
				}
				bindings[column.Variable] = column.Kind
				if column.Term != nil {
					stack = append(stack, column.Term)
				}
			}
			for _, column := range p.RelationColumns {
				if column.Variable == "" || column.Kind == 0 || column.Kind == FunctionClassColumn {
					continue
				}
				if old, ok := bindings[column.Variable]; ok && old != column.Kind {
					return nil, fmt.Errorf("variable %s is bound with conflicting primitive kinds %d and %d", column.Variable, old, column.Kind)
				}
				bindings[column.Variable] = column.Kind
			}
			stack = append(stack, p.Children...)
		}
	}
	return bindings, nil
}

func functionActionClassBindings(lhs []*Pattern) map[string]bool {
	bindings := make(map[string]bool)
	stack := append([]*Pattern(nil), lhs...)
	seen := make(map[*Pattern]bool)
	for len(stack) != 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if p == nil || seen[p] {
			continue
		}
		seen[p] = true
		if p.IsVar {
			bindings[p.Op] = true
			continue
		}
		for _, columns := range [][]FunctionColumnPattern{p.FunctionColumns, p.RelationColumns} {
			for _, column := range columns {
				if column.Kind == FunctionClassColumn && column.Variable != "" {
					bindings[column.Variable] = true
				}
				if column.Term != nil {
					stack = append(stack, column.Term)
				}
			}
		}
		stack = append(stack, p.Children...)
	}
	return bindings
}

func functionActionExpressionTypeWithClasses(expr *FunctionActionExpr, bindings map[string]FunctionColumnKind, classes map[string]bool) (FunctionColumnKind, error) {
	if expr == nil {
		return 0, fmt.Errorf("nil expression")
	}
	switch expr.kind {
	case functionActionVariable:
		kind, ok := bindings[expr.variable]
		if !ok {
			return 0, fmt.Errorf("variable %s is not bound by a primitive function pattern", expr.variable)
		}
		return kind, nil
	case functionActionClassInt64:
		if !classes[expr.variable] || bindings[expr.variable] != 0 {
			return 0, fmt.Errorf("variable %s is not bound as an e-class", expr.variable)
		}
		return FunctionInt64Column, nil
	case functionActionClassFloat64:
		if !classes[expr.variable] || bindings[expr.variable] != 0 {
			return 0, fmt.Errorf("variable %s is not bound as an e-class", expr.variable)
		}
		return FunctionFloat64Column, nil
	case functionActionClassRef:
		if !classes[expr.variable] || bindings[expr.variable] != 0 {
			return 0, fmt.Errorf("variable %s is not bound as an e-class", expr.variable)
		}
		return FunctionClassColumn, nil
	case functionActionConstant:
		if expr.value.Kind < FunctionClassColumn || expr.value.Kind > FunctionBigRatColumn {
			return 0, fmt.Errorf("invalid primitive constant kind %d", expr.value.Kind)
		}
		if expr.value.Kind == FunctionFloat64Column && math.IsNaN(expr.value.Float64) {
			return 0, fmt.Errorf("f64 primitive constant rejects NaN")
		}
		return expr.value.Kind, nil
	case functionActionLookup:
		if expr.name == "" {
			return 0, fmt.Errorf("function lookup has empty name")
		}
		for i := range expr.args {
			if _, err := functionActionExpressionTypeWithClasses(&expr.args[i], bindings, classes); err != nil {
				return 0, fmt.Errorf("lookup argument %d: %w", i, err)
			}
		}
		return functionActionDynamicColumn, nil
	case functionActionPrimitive:
		want, result, arity, err := functionPrimitiveSignature(expr.op)
		if err != nil {
			return 0, err
		}
		if arity >= 0 && len(expr.args) != arity {
			return 0, fmt.Errorf("operation %d requires %d arguments, got %d", expr.op, arity, len(expr.args))
		}
		if want == 0 {
			left, err := functionActionExpressionTypeWithClasses(&expr.args[0], bindings, classes)
			if err != nil {
				return 0, fmt.Errorf("argument 0: %w", err)
			}
			right, err := functionActionExpressionTypeWithClasses(&expr.args[1], bindings, classes)
			if err != nil {
				return 0, fmt.Errorf("argument 1: %w", err)
			}
			if left != functionActionDynamicColumn && right != functionActionDynamicColumn && (!primitiveEqualityKind(left) || left != right) {
				return 0, fmt.Errorf("value equality requires arguments with the same primitive kind, got %d and %d", left, right)
			}
			return result, nil
		}
		for i := range expr.args {
			kind, err := functionActionExpressionTypeWithClasses(&expr.args[i], bindings, classes)
			if err != nil {
				return 0, fmt.Errorf("argument %d: %w", i, err)
			}
			if kind != functionActionDynamicColumn && kind != want {
				return 0, fmt.Errorf("argument %d has kind %d, want %d", i, kind, want)
			}
		}
		return result, nil
	default:
		return 0, fmt.Errorf("invalid expression kind %d", expr.kind)
	}
}

func functionActionExpressionTypeForGraph(g *EGraph, expr *FunctionActionExpr, bindings map[string]FunctionColumnKind, classes map[string]bool) (FunctionColumnType, error) {
	if expr == nil {
		return FunctionColumnType{}, fmt.Errorf("nil expression")
	}
	switch expr.kind {
	case functionActionVariable:
		kind, ok := bindings[expr.variable]
		if !ok {
			return FunctionColumnType{}, fmt.Errorf("variable %s is not bound by a primitive function pattern", expr.variable)
		}
		if kind == functionActionDynamicColumn {
			return FunctionColumnType{}, fmt.Errorf("variable %s has unresolved function lookup type", expr.variable)
		}
		return FunctionColumnType{Kind: kind}, nil
	case functionActionClassInt64:
		if !classes[expr.variable] || bindings[expr.variable] != 0 {
			return FunctionColumnType{}, fmt.Errorf("variable %s is not bound as an e-class", expr.variable)
		}
		return FunctionColumnType{Kind: FunctionInt64Column}, nil
	case functionActionClassFloat64:
		if !classes[expr.variable] || bindings[expr.variable] != 0 {
			return FunctionColumnType{}, fmt.Errorf("variable %s is not bound as an e-class", expr.variable)
		}
		return FunctionColumnType{Kind: FunctionFloat64Column}, nil
	case functionActionClassRef:
		if !classes[expr.variable] || bindings[expr.variable] != 0 {
			return FunctionColumnType{}, fmt.Errorf("variable %s is not bound as an e-class", expr.variable)
		}
		return FunctionColumnType{Kind: FunctionClassColumn}, nil
	case functionActionConstant:
		typ := FunctionColumnType{Kind: expr.value.Kind}
		if typ.Kind == FunctionClassColumn {
			root, err := g.FindChecked(expr.value.ID)
			if err != nil {
				return FunctionColumnType{}, err
			}
			class := g.store.Class(root)
			if class == nil {
				return FunctionColumnType{}, fmt.Errorf("class constant references missing class %d", root)
			}
			typ.Sort = class.Sort
			return typ, nil
		}
		if _, err := g.canonicalTypedValue(typ, expr.value, "function action constant"); err != nil {
			return FunctionColumnType{}, err
		}
		return typ, nil
	case functionActionLookup:
		target := g.functions[expr.name]
		if target == nil {
			return FunctionColumnType{}, fmt.Errorf("lookup target %q is not registered", expr.name)
		}
		if len(expr.args) != len(target.schema.KeyTypes) {
			return FunctionColumnType{}, fmt.Errorf("lookup target %q expects %d key columns, got %d", expr.name, len(target.schema.KeyTypes), len(expr.args))
		}
		for i := range expr.args {
			argType, err := functionActionExpressionTypeForGraph(g, &expr.args[i], bindings, classes)
			if err != nil {
				return FunctionColumnType{}, fmt.Errorf("lookup target %q argument %d: %w", expr.name, i, err)
			}
			want := target.schema.KeyTypes[i]
			if argType.Kind != want.Kind || want.Kind == FunctionClassColumn && argType.Sort != "" && want.Sort != "" && argType.Sort != want.Sort {
				return FunctionColumnType{}, fmt.Errorf("lookup target %q argument %d has type %v, want %v", expr.name, i, argType, want)
			}
		}
		return target.schema.ValueType, nil
	case functionActionPrimitive:
		want, result, arity, err := functionPrimitiveSignature(expr.op)
		if err != nil {
			return FunctionColumnType{}, err
		}
		if arity >= 0 && len(expr.args) != arity {
			return FunctionColumnType{}, fmt.Errorf("operation %d requires %d arguments, got %d", expr.op, arity, len(expr.args))
		}
		var left FunctionColumnType
		for i := range expr.args {
			got, err := functionActionExpressionTypeForGraph(g, &expr.args[i], bindings, classes)
			if err != nil {
				return FunctionColumnType{}, fmt.Errorf("argument %d: %w", i, err)
			}
			if i == 0 {
				left = got
			}
			if want == 0 {
				if !primitiveEqualityKind(got.Kind) || i == 1 && got.Kind != left.Kind {
					return FunctionColumnType{}, fmt.Errorf("value equality requires arguments with the same primitive kind")
				}
			} else if got.Kind != want {
				return FunctionColumnType{}, fmt.Errorf("argument %d has kind %d, want %d", i, got.Kind, want)
			}
		}
		return FunctionColumnType{Kind: result}, nil
	default:
		return FunctionColumnType{}, fmt.Errorf("invalid function action expression kind %d", expr.kind)
	}
}

func validateActionExpressionType(g *EGraph, expr *FunctionActionExpr, bindings map[string]FunctionColumnKind, classes map[string]bool, want FunctionColumnType) error {
	got, err := functionActionExpressionTypeForGraph(g, expr, bindings, classes)
	if err != nil {
		return err
	}
	if got.Kind != want.Kind || got.Kind == FunctionClassColumn && got.Sort != "" && want.Sort != "" && got.Sort != want.Sort {
		return fmt.Errorf("expression has type %v, want %v", got, want)
	}
	return nil
}

func (r *Rewrite) validateFunctionActionExpressions(g *EGraph) error {
	bindings, err := functionActionBindingTypes(r.leftPatterns())
	if err != nil {
		return err
	}
	classes := functionActionClassBindings(r.leftPatterns())
	provisionalClassBindings := make(map[string]bool)
	for _, binding := range r.primitiveBindings {
		typ, err := functionActionExpressionTypeForGraph(g, &binding.Expr, bindings, classes)
		if err != nil {
			return fmt.Errorf("egraph: rewrite %q primitive binding %s: %w", r.Name, binding.Name, err)
		}
		if r.Guard != nil && functionActionMayReturnFreshClass(g, &binding.Expr, provisionalClassBindings) {
			return fmt.Errorf("egraph: rewrite %q primitive binding %s may return a provisional class that its guard cannot inspect", r.Name, binding.Name)
		}
		provisionalClassBindings[binding.Name] = functionActionMayReturnFreshClass(g, &binding.Expr, provisionalClassBindings)
		bindings[binding.Name] = typ.Kind
	}
	for i := range r.primitiveConditions {
		if err := validateActionExpressionType(g, &r.primitiveConditions[i], bindings, classes, FunctionColumnType{Kind: FunctionBoolColumn}); err != nil {
			return fmt.Errorf("egraph: rewrite %q primitive condition %d: %w", r.Name, i, err)
		}
	}
	for _, pattern := range r.actionTermPatterns() {
		stack := []*Pattern{pattern}
		seen := make(map[*Pattern]bool)
		for len(stack) != 0 {
			current := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if current == nil || seen[current] {
				continue
			}
			seen[current] = true
			if current.int64Expr != nil {
				if err := validateActionExpressionType(g, current.int64Expr, bindings, classes, FunctionColumnType{Kind: FunctionInt64Column}); err != nil {
					return fmt.Errorf("egraph: rewrite %q computed i64 leaf: %w", r.Name, err)
				}
			}
			if current.float64Expr != nil {
				if err := validateActionExpressionType(g, current.float64Expr, bindings, classes, FunctionColumnType{Kind: FunctionFloat64Column}); err != nil {
					return fmt.Errorf("egraph: rewrite %q computed f64 leaf: %w", r.Name, err)
				}
			}
			stack = append(stack, current.Children...)
		}
	}
	for _, action := range r.relations {
		if !(action.Typed || action.TypedColumns != nil) {
			continue
		}
		table, err := g.relation(action.Name, len(action.TypedColumns))
		if err != nil {
			return err
		}
		for i, column := range action.TypedColumns {
			if column.Expr != nil {
				if err := validateActionExpressionType(g, column.Expr, bindings, classes, table.schema.Columns[i]); err != nil {
					return fmt.Errorf("egraph: rewrite %q relation action %q column %d: %w", r.Name, action.Name, i, err)
				}
			}
		}
	}
	for _, action := range r.functions {
		if !(action.Typed || action.TypedKeys != nil || action.TypedValue != nil) {
			continue
		}
		table, err := g.function(action.Name, len(action.TypedKeys))
		if err != nil {
			return err
		}
		for i, column := range action.TypedKeys {
			if column.Expr != nil {
				if err := validateActionExpressionType(g, column.Expr, bindings, classes, table.schema.KeyTypes[i]); err != nil {
					return fmt.Errorf("egraph: rewrite %q function action %q key %d: %w", r.Name, action.Name, i, err)
				}
			}
		}
		if action.TypedValue != nil && action.TypedValue.Expr != nil {
			if err := validateActionExpressionType(g, action.TypedValue.Expr, bindings, classes, table.schema.ValueType); err != nil {
				return fmt.Errorf("egraph: rewrite %q function action %q value: %w", r.Name, action.Name, err)
			}
		}
	}
	return nil
}

func functionActionMayReturnFreshClass(g *EGraph, expr *FunctionActionExpr, bindings map[string]bool) bool {
	if expr == nil {
		return false
	}
	switch expr.kind {
	case functionActionVariable:
		return bindings[expr.variable]
	case functionActionLookup:
		target := g.functions[expr.name]
		return target != nil && target.schema.ValueType.Kind == FunctionClassColumn && target.schema.Default.Kind == FunctionDefaultFreshID
	default:
		return false
	}
}

func collectFunctionActionExprVars(expr *FunctionActionExpr, variables map[string]bool) {
	if expr == nil {
		return
	}
	if expr.kind == functionActionVariable || expr.kind == functionActionClassInt64 || expr.kind == functionActionClassFloat64 || expr.kind == functionActionClassRef {
		variables[expr.variable] = true
	}
	for i := range expr.args {
		collectFunctionActionExprVars(&expr.args[i], variables)
	}
}

func functionActionExprHasLookup(expr *FunctionActionExpr) bool {
	if expr == nil {
		return false
	}
	if expr.kind == functionActionLookup {
		return true
	}
	for i := range expr.args {
		if functionActionExprHasLookup(&expr.args[i]) {
			return true
		}
	}
	return false
}

func (r *Rewrite) hasFunctionActionLookup() bool {
	for _, pattern := range r.actionTermPatterns() {
		stack := []*Pattern{pattern}
		seen := make(map[*Pattern]bool)
		for len(stack) > 0 {
			current := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if current == nil || seen[current] {
				continue
			}
			seen[current] = true
			if functionActionExprHasLookup(current.int64Expr) || functionActionExprHasLookup(current.float64Expr) {
				return true
			}
			stack = append(stack, current.Children...)
		}
	}
	for _, action := range r.relations {
		for i := range action.TypedColumns {
			if functionActionExprHasLookup(action.TypedColumns[i].Expr) {
				return true
			}
		}
	}
	for _, action := range r.functions {
		for i := range action.TypedKeys {
			if functionActionExprHasLookup(action.TypedKeys[i].Expr) {
				return true
			}
		}
		if action.TypedValue != nil && functionActionExprHasLookup(action.TypedValue.Expr) {
			return true
		}
	}
	return false
}

func (r *Rewrite) hasFilterFunctionActionLookup() bool {
	for i := range r.primitiveBindings {
		if functionActionExprHasLookup(&r.primitiveBindings[i].Expr) {
			return true
		}
	}
	for i := range r.primitiveConditions {
		if functionActionExprHasLookup(&r.primitiveConditions[i]) {
			return true
		}
	}
	return false
}

func validatePattern(p *Pattern) error {
	if p == nil {
		return fmt.Errorf("nil pattern")
	}
	type frame struct {
		pattern *Pattern
		child   int
	}
	state := map[*Pattern]uint8{p: 1}
	if err := validatePatternNode(p); err != nil {
		return err
	}
	stack := []frame{{pattern: p}}
	for len(stack) > 0 {
		top := &stack[len(stack)-1]
		if top.pattern != p && (top.pattern.RelationName != "" || top.pattern.FunctionName != "") {
			return fmt.Errorf("relation and function patterns cannot be nested")
		}
		if top.child == len(top.pattern.Children) {
			state[top.pattern] = 2
			stack = stack[:len(stack)-1]
			continue
		}
		index := top.child
		top.child++
		child := top.pattern.Children[index]
		if child == nil {
			return fmt.Errorf("child %d: nil child pattern", index)
		}
		switch state[child] {
		case 1:
			return fmt.Errorf("cyclic pattern at operator %q", child.Op)
		case 2:
			continue
		}
		if err := validatePatternNode(child); err != nil {
			return fmt.Errorf("child %d: %w", index, err)
		}
		state[child] = 1
		stack = append(stack, frame{pattern: child})
	}
	return nil
}

func validatePatternNode(p *Pattern) error {
	if p.int64Expr != nil || p.float64Expr != nil {
		if p.int64Expr != nil && p.float64Expr != nil {
			return fmt.Errorf("pattern has multiple computed leaf expressions")
		}
		if p.float64Expr != nil {
			if p.Op != "$computed-f64" || p.Sort != "f64" || p.IsVar || p.RelationName != "" || p.FunctionName != "" || p.Data != nil || p.HasData || len(p.Children) != 0 {
				return fmt.Errorf("invalid computed f64 leaf pattern")
			}
			return nil
		}
		if p.Op != "$computed-i64" || p.Sort != "i64" || p.IsVar || p.RelationName != "" || p.FunctionName != "" || p.Data != nil || p.HasData || len(p.Children) != 0 {
			return fmt.Errorf("invalid computed i64 leaf pattern")
		}
		return nil
	}
	if p.FunctionName != "" {
		if p.FunctionName != p.Op || p.RelationName != "" || len(p.RelationColumns) != 0 || p.IsVar || p.Sort != "" || p.HasData || p.Data != nil || (len(p.Children) == 0 && len(p.FunctionColumns) == 0) || (len(p.Children) > 0 && len(p.FunctionColumns) > 0) {
			return fmt.Errorf("invalid function pattern %q", p.FunctionName)
		}
		for i, column := range p.FunctionColumns {
			if column.Expr != nil {
				return fmt.Errorf("function pattern column %d cannot contain an action expression", i)
			}
			if column.Term != nil {
				if column.Kind != 0 && column.Kind != FunctionClassColumn {
					return fmt.Errorf("function column %d has a term and primitive kind", i)
				}
				if err := validateTermPattern(column.Term); err != nil {
					return fmt.Errorf("function column %d: %w", i, err)
				}
				continue
			}
			if column.Kind < FunctionInt64Column || column.Kind > FunctionBigRatColumn {
				return fmt.Errorf("function column %d has invalid primitive kind %d", i, column.Kind)
			}
			if column.HasValue && column.Variable != "" {
				return fmt.Errorf("function column %d has both literal and variable", i)
			}
			if !column.HasValue && (len(column.Variable) < 2 || !strings.HasPrefix(column.Variable, "?")) {
				return fmt.Errorf("function column %d requires a variable or literal", i)
			}
		}
		return nil
	}
	if p.RelationName != "" {
		if p.RelationName != p.Op || len(p.FunctionColumns) != 0 || p.IsVar || p.Sort != "" || p.HasData || p.Data != nil || (len(p.Children) > 0 && len(p.RelationColumns) > 0) {
			return fmt.Errorf("invalid relation pattern %q", p.RelationName)
		}
		for i, column := range p.RelationColumns {
			if column.Expr != nil {
				return fmt.Errorf("relation pattern column %d cannot contain an action expression", i)
			}
			if column.Term != nil {
				if column.Kind != 0 && column.Kind != FunctionClassColumn {
					return fmt.Errorf("relation column %d has a term and primitive kind", i)
				}
				if err := validateTermPattern(column.Term); err != nil {
					return fmt.Errorf("relation column %d: %w", i, err)
				}
				continue
			}
			if column.HasValue && (column.Kind == 0 || column.Kind == FunctionClassColumn) {
				if column.Variable != "" {
					return fmt.Errorf("relation column %d has both literal and variable", i)
				}
				continue
			}
			if column.Kind == FunctionClassColumn {
				return fmt.Errorf("relation column %d requires a class term or literal", i)
			}
			if column.Kind < FunctionInt64Column || column.Kind > FunctionBigRatColumn {
				return fmt.Errorf("relation column %d has invalid primitive kind %d", i, column.Kind)
			}
			if column.HasValue && column.Variable != "" {
				return fmt.Errorf("relation column %d has both literal and variable", i)
			}
			if !column.HasValue && (len(column.Variable) < 2 || !strings.HasPrefix(column.Variable, "?")) {
				return fmt.Errorf("relation column %d requires a variable or literal", i)
			}
		}
		return nil
	}
	if p.Op == "" {
		return fmt.Errorf("empty operator")
	}
	if p.IsVar {
		if len(p.Op) < 2 || !strings.HasPrefix(p.Op, "?") {
			return fmt.Errorf("variable %q must have a non-empty name prefixed with ?", p.Op)
		}
		if len(p.Children) != 0 || p.HasData {
			return fmt.Errorf("variable %q cannot have data or children", p.Op)
		}
	}
	return nil
}

func collectVariables(p *Pattern, variables map[string]bool) {
	stack := []*Pattern{p}
	seen := make(map[*Pattern]bool)
	for len(stack) > 0 {
		last := len(stack) - 1
		current := stack[last]
		stack = stack[:last]
		if seen[current] {
			continue
		}
		seen[current] = true
		if current.IsVar {
			variables[current.Op] = true
			continue
		}
		if current.int64Expr != nil {
			collectFunctionActionExprVars(current.int64Expr, variables)
		}
		if current.float64Expr != nil {
			collectFunctionActionExprVars(current.float64Expr, variables)
		}
		for _, column := range current.FunctionColumns {
			if column.Term != nil {
				stack = append(stack, column.Term)
			}
			if column.Variable != "" {
				variables[column.Variable] = true
			}
		}
		for _, column := range current.RelationColumns {
			if column.Term != nil {
				stack = append(stack, column.Term)
			}
			if column.Variable != "" {
				variables[column.Variable] = true
			}
		}
		for i := len(current.Children) - 1; i >= 0; i-- {
			stack = append(stack, current.Children[i])
		}
	}
}

func validateComputedPatterns(pattern *Pattern, primitiveBindings map[string]FunctionColumnKind, classBindings map[string]bool) error {
	stack := []*Pattern{pattern}
	seen := make(map[*Pattern]bool)
	for len(stack) != 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if current == nil || seen[current] {
			continue
		}
		seen[current] = true
		expr, want := current.int64Expr, FunctionInt64Column
		if current.float64Expr != nil {
			expr, want = current.float64Expr, FunctionFloat64Column
		}
		if expr != nil {
			kind, err := functionActionExpressionTypeWithClasses(expr, primitiveBindings, classBindings)
			if err != nil {
				return fmt.Errorf("computed leaf expression: %w", err)
			}
			if kind != want {
				return fmt.Errorf("computed leaf expression has kind %d, want %d", kind, want)
			}
		}
		stack = append(stack, current.Children...)
	}
	return nil
}

func instantiateFunctionColumn(g *EGraph, column FunctionColumnPattern, match Match) (FunctionValue, error) {
	if column.Expr != nil {
		return evaluateFunctionActionExpr(g, column.Expr, match)
	}
	if column.Term != nil {
		id, err := column.Term.InstantiateWithMatchChecked(g, match)
		if err != nil {
			return FunctionValue{}, err
		}
		return ClassValue(id), nil
	}
	if column.HasValue {
		return column.Value, nil
	}
	if id, ok := match.Subst[column.Variable]; ok {
		if column.Kind != FunctionClassColumn {
			return FunctionValue{}, fmt.Errorf("egraph: function variable %s is an e-class, not a primitive", column.Variable)
		}
		return ClassValue(g.Find(id)), nil
	}
	value, ok := match.BaseSubst[column.Variable]
	if !ok {
		return FunctionValue{}, fmt.Errorf("egraph: unbound function value %s", column.Variable)
	}
	if value.Kind != column.Kind {
		return FunctionValue{}, fmt.Errorf("egraph: function value %s has kind %d, want %d", column.Variable, value.Kind, column.Kind)
	}
	return value, nil
}

func evaluateFunctionActionExpr(g *EGraph, expr *FunctionActionExpr, match Match) (FunctionValue, error) {
	if expr == nil {
		return FunctionValue{}, fmt.Errorf("egraph: nil function action expression")
	}
	switch expr.kind {
	case functionActionVariable:
		value, ok := match.BaseSubst[expr.variable]
		if !ok {
			return FunctionValue{}, fmt.Errorf("egraph: unbound primitive function value %s", expr.variable)
		}
		return value, nil
	case functionActionClassInt64:
		id, ok := match.Subst[expr.variable]
		if !ok {
			return FunctionValue{}, fmt.Errorf("egraph: unbound e-class variable %s", expr.variable)
		}
		value, err := eclassInt64Value(g, id)
		if err != nil {
			return FunctionValue{}, fmt.Errorf("egraph: project %s as i64: %w", expr.variable, err)
		}
		return Int64Value(value), nil
	case functionActionClassFloat64:
		id, ok := match.Subst[expr.variable]
		if !ok {
			return FunctionValue{}, fmt.Errorf("egraph: unbound e-class variable %s", expr.variable)
		}
		value, err := eclassFloat64Value(g, id)
		if err != nil {
			return FunctionValue{}, fmt.Errorf("egraph: project %s as f64: %w", expr.variable, err)
		}
		return Float64Value(value), nil
	case functionActionClassRef:
		id, ok := match.Subst[expr.variable]
		if !ok {
			return FunctionValue{}, fmt.Errorf("egraph: unbound e-class variable %s", expr.variable)
		}
		root, err := g.FindChecked(id)
		if err != nil {
			return FunctionValue{}, err
		}
		return ClassValue(root), nil
	case functionActionConstant:
		return expr.value, nil
	case functionActionLookup:
		if g == nil {
			return FunctionValue{}, fmt.Errorf("egraph: nil graph")
		}
		args := make([]FunctionValue, len(expr.args))
		for i := range expr.args {
			value, err := evaluateFunctionActionExpr(g, &expr.args[i], match)
			if err != nil {
				return FunctionValue{}, fmt.Errorf("egraph: function lookup %q argument %d: %w", expr.name, i, err)
			}
			args[i] = value
		}
		if match.lookupView != nil {
			return g.lookupFunctionWithView(expr.name, args, *match.lookupView)
		}
		table := g.functions[expr.name]
		if table == nil {
			return FunctionValue{}, fmt.Errorf("egraph: lookup target %q is not registered", expr.name)
		}
		if len(args) != len(table.schema.KeyTypes) {
			return FunctionValue{}, fmt.Errorf("egraph: lookup target %q expects %d key columns, got %d", expr.name, len(table.schema.KeyTypes), len(args))
		}
		if table.schema.typed {
			value, ok, err := g.LookupFunctionValues(expr.name, args...)
			if err != nil {
				return FunctionValue{}, err
			}
			if !ok {
				return FunctionValue{}, fmt.Errorf("egraph: lookup target %q has no row for key: %w", expr.name, errPrimitiveNoValue)
			}
			return value, nil
		}
		keys := make([]Id, len(args))
		for i, arg := range args {
			if arg.Kind != FunctionClassColumn {
				return FunctionValue{}, fmt.Errorf("egraph: lookup target %q key %d has kind %d, want class", expr.name, i, arg.Kind)
			}
			keys[i] = arg.ID
		}
		value, ok, err := g.LookupFunction(expr.name, keys...)
		if err != nil {
			return FunctionValue{}, err
		}
		if !ok {
			return FunctionValue{}, fmt.Errorf("egraph: lookup target %q has no row for key: %w", expr.name, errPrimitiveNoValue)
		}
		return ClassValue(value), nil
	case functionActionPrimitive:
		args := make([]FunctionValue, len(expr.args))
		for i := range expr.args {
			value, err := evaluateFunctionActionExpr(g, &expr.args[i], match)
			if err != nil {
				return FunctionValue{}, fmt.Errorf("egraph: function action expression argument %d: %w", i, err)
			}
			args[i] = value
		}
		value, err := evaluateFunctionPrimitive(g, expr.op, args)
		if err != nil {
			return FunctionValue{}, fmt.Errorf("egraph: function action primitive: %w", err)
		}
		return value, nil
	default:
		return FunctionValue{}, fmt.Errorf("egraph: invalid function action expression kind %d", expr.kind)
	}
}

func eclassInt64Value(g *EGraph, id Id) (int64, error) {
	if g == nil {
		return 0, fmt.Errorf("nil graph")
	}
	root, err := g.FindChecked(id)
	if err != nil {
		return 0, err
	}
	class := g.store.Class(root)
	if class == nil {
		return 0, fmt.Errorf("e-class %d is missing", root)
	}
	if class.Sort != "i64" {
		return 0, fmt.Errorf("e-class %d has sort %q, want i64", root, class.Sort)
	}
	var value int64
	found := false
	for _, node := range class.Nodes {
		if len(node.Children) != 0 || node.Sort != "i64" {
			continue
		}
		candidate, ok := node.Data.(int64)
		if !ok {
			continue
		}
		if found && candidate != value {
			return 0, fmt.Errorf("e-class %d contains conflicting i64 leaves %d and %d", root, value, candidate)
		}
		value, found = candidate, true
	}
	if !found {
		return 0, fmt.Errorf("e-class %d has no i64 leaf payload", root)
	}
	return value, nil
}

func eclassFloat64Value(g *EGraph, id Id) (float64, error) {
	if g == nil {
		return 0, fmt.Errorf("nil graph")
	}
	root, err := g.FindChecked(id)
	if err != nil {
		return 0, err
	}
	class := g.store.Class(root)
	if class == nil {
		return 0, fmt.Errorf("e-class %d is missing", root)
	}
	if class.Sort != "f64" {
		return 0, fmt.Errorf("e-class %d has sort %q, want f64", root, class.Sort)
	}
	var value float64
	found := false
	for _, node := range class.Nodes {
		if len(node.Children) != 0 || node.Sort != "f64" {
			continue
		}
		candidate, ok := node.Data.(float64)
		if !ok {
			continue
		}
		if math.IsNaN(candidate) {
			return 0, fmt.Errorf("e-class %d contains an f64 NaN literal", root)
		}
		if candidate == 0 {
			candidate = 0
		}
		if found && candidate != value {
			return 0, fmt.Errorf("e-class %d contains conflicting f64 leaves %v and %v", root, value, candidate)
		}
		value, found = candidate, true
	}
	if !found {
		return 0, fmt.Errorf("e-class %d contains no f64 literal", root)
	}
	return value, nil
}

// MustRewrite creates a Rewrite or panics if parsing fails.
func MustRewrite(name, lhsStr, rhsStr string) *Rewrite {
	r, err := NewRewrite(name, lhsStr, rhsStr)
	if err != nil {
		panic(err)
	}
	return r
}

// Apply searches the graph for left-hand matches, applies the rule's actions,
// and returns the number of class unions.
func (r *Rewrite) Apply(g *EGraph) int {
	merges, _ := r.ApplyChecked(g)
	return merges
}

// ApplyChecked applies a rewrite and reports invalid patterns, graph IDs, or
// action bindings. Graph changes are published only when the full rewrite
// succeeds. Its count includes class unions, not analysis or relation changes.
func (r *Rewrite) ApplyChecked(g *EGraph) (int, error) {
	if g == nil {
		return 0, fmt.Errorf("egraph: nil graph")
	}
	staged := g.cloneForScheduleRun()
	merges, _, _, _, _, _, _, _, _, err := r.applyChecked(staged, 0)
	if err != nil {
		return 0, err
	}
	*g = *staged
	return merges, nil
}

func (r *Rewrite) applyChecked(g *EGraph, maxNodes int) (merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges int, nodeLimit bool, err error) {
	return r.applyCheckedMatches(g, maxNodes, nil, false)
}

func (r *Rewrite) applyCheckedMatches(g *EGraph, maxNodes int, suppliedMatches []Match, hasSuppliedMatches bool) (merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges int, nodeLimit bool, err error) {
	return r.applyCheckedMatchesWithBudget(g, maxNodes, suppliedMatches, hasSuppliedMatches, nil)
}

func (r *Rewrite) filterClassMatchConditions(g *EGraph, matches []Match, budget *matchWorkBudget) ([]Match, error) {
	if len(r.classConditions) != 0 {
		filtered := matches[:0]
		for matchIndex, match := range matches {
			accepted := true
			for conditionIndex, condition := range r.classConditions {
				if !budget.spend() {
					return nil, errSaturationMatchWorkLimit
				}
				left, leftOK := match.Subst[condition.Left]
				right, rightOK := match.Subst[condition.Right]
				if !leftOK || !rightOK {
					return nil, fmt.Errorf("egraph: rewrite %q match %d class condition %d is missing a bound variable", r.Name, matchIndex, conditionIndex)
				}
				if (g.Find(left) == g.Find(right)) != condition.Equal {
					accepted = false
					break
				}
			}
			if accepted {
				filtered = append(filtered, match)
			}
		}
		matches = filtered
	}
	if len(r.classPatternConditions) != 0 {
		targets := make([]map[Id]struct{}, len(r.classPatternConditions))
		for conditionIndex, condition := range r.classPatternConditions {
			found, complete, err := condition.Pattern.searchChecked(g, budget)
			if err != nil {
				return nil, fmt.Errorf("egraph: search class pattern condition %d for rewrite %q: %w", conditionIndex, r.Name, err)
			}
			if !complete {
				return nil, errSaturationMatchWorkLimit
			}
			targets[conditionIndex] = make(map[Id]struct{}, len(found))
			for _, target := range found {
				if target.ClassID != InvalidId {
					targets[conditionIndex][g.Find(target.ClassID)] = struct{}{}
				}
			}
		}
		filtered := matches[:0]
		for matchIndex, match := range matches {
			accepted := true
			for conditionIndex, condition := range r.classPatternConditions {
				if !budget.spend() {
					return nil, errSaturationMatchWorkLimit
				}
				classID, ok := match.Subst[condition.Variable]
				if !ok {
					return nil, fmt.Errorf("egraph: rewrite %q match %d class pattern condition %d is missing bound variable %s", r.Name, matchIndex, conditionIndex, condition.Variable)
				}
				_, equal := targets[conditionIndex][g.Find(classID)]
				if equal != condition.Equal {
					accepted = false
					break
				}
			}
			if accepted {
				filtered = append(filtered, match)
			}
		}
		matches = filtered
	}
	return matches, nil
}

func (r *Rewrite) prepareQueryMatches(g *EGraph, matches []Match, budget *matchWorkBudget, passView *functionMergeView) ([]Match, error) {
	var err error
	matches, err = r.filterClassMatchConditions(g, matches, budget)
	if err != nil {
		return nil, err
	}
	filterLookups := r.hasFilterFunctionActionLookup()
	if filterLookups && passView == nil {
		return nil, fmt.Errorf("egraph: rewrite %q requires a function lookup view", r.Name)
	}
	acceptedMatches := make([]Match, 0, len(matches))
	for matchIndex, match := range matches {
		var candidateView functionMergeView
		if filterLookups {
			candidateView = cloneFunctionMergeView(*passView)
			match.lookupView = &candidateView
		}
		match.BaseSubst = cloneBaseSubst(match.BaseSubst)
		accepted := true
		for bindingIndex := range r.primitiveBindings {
			if !budget.spend() {
				return nil, errSaturationMatchWorkLimit
			}
			binding := &r.primitiveBindings[bindingIndex]
			value, evalErr := evaluateFunctionActionExpr(g, &binding.Expr, match)
			if evalErr != nil {
				if errors.Is(evalErr, errPrimitiveNoValue) {
					accepted = false
					break
				}
				return nil, fmt.Errorf("egraph: rewrite %q match %d primitive binding %s: %w", r.Name, matchIndex, binding.Name, evalErr)
			}
			if old, exists := match.BaseSubst[binding.Name]; exists {
				if old != value {
					accepted = false
					break
				}
			} else {
				match.BaseSubst[binding.Name] = value
			}
		}
		if accepted {
			for conditionIndex := range r.primitiveConditions {
				if !budget.spend() {
					return nil, errSaturationMatchWorkLimit
				}
				value, evalErr := evaluateFunctionActionExpr(g, &r.primitiveConditions[conditionIndex], match)
				if evalErr != nil {
					if errors.Is(evalErr, errPrimitiveNoValue) {
						accepted = false
						break
					}
					return nil, fmt.Errorf("egraph: rewrite %q match %d condition %d: %w", r.Name, matchIndex, conditionIndex, evalErr)
				}
				if value.Kind != FunctionBoolColumn {
					return nil, fmt.Errorf("egraph: rewrite %q condition %d returned kind %d, want bool", r.Name, conditionIndex, value.Kind)
				}
				if !value.Bool {
					accepted = false
					break
				}
			}
		}
		if accepted && r.Guard != nil {
			guardMatch := Match{ClassID: match.ClassID, Subst: match.Subst.Clone(), BaseSubst: cloneBaseSubst(match.BaseSubst)}
			guardRevision := g.revision
			apply, guardErr := r.Guard(g, guardMatch)
			if g.revision != guardRevision {
				rebuildErr := g.RebuildChecked()
				if guardErr != nil {
					return nil, errors.Join(fmt.Errorf("egraph: rewrite %q guard mutated the graph: %w", r.Name, guardErr), rebuildErr)
				}
				return nil, errors.Join(fmt.Errorf("egraph: rewrite %q guard mutated the graph", r.Name), rebuildErr)
			}
			if guardErr != nil {
				return nil, fmt.Errorf("egraph: rewrite %q guard: %w", r.Name, guardErr)
			}
			accepted = apply
		}
		if accepted {
			if filterLookups {
				*passView = candidateView
			}
			match.lookupView = nil
			acceptedMatches = append(acceptedMatches, match)
		}
	}
	return acceptedMatches, nil
}

func (r *Rewrite) applyCheckedMatchesWithBudget(g *EGraph, maxNodes int, suppliedMatches []Match, hasSuppliedMatches bool, budget *matchWorkBudget) (merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges int, nodeLimit bool, err error) {
	return r.applyCheckedMatchesWithPreparedConditions(g, maxNodes, suppliedMatches, hasSuppliedMatches, false, budget, nil, true)
}

func (r *Rewrite) applyCapturedMatchesWithBudget(g *EGraph, maxNodes int, matches []Match, budget *matchWorkBudget) (merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges int, nodeLimit bool, err error) {
	return r.applyCheckedMatchesWithPreparedConditions(g, maxNodes, matches, true, true, budget, nil, true)
}

func (r *Rewrite) applyCapturedMatchesWithFunctionBatch(g *EGraph, maxNodes int, matches []Match, budget *matchWorkBudget, batch *functionActionBatch) (merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges int, nodeLimit bool, err error) {
	return r.applyCheckedMatchesWithPreparedConditions(g, maxNodes, matches, true, true, budget, batch, false)
}

func (r *Rewrite) applyCheckedMatchesWithPreparedConditions(g *EGraph, maxNodes int, suppliedMatches []Match, hasSuppliedMatches, matchesPrepared bool, budget *matchWorkBudget, functionBatch *functionActionBatch, commitFunctionBatch bool) (merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges int, nodeLimit bool, err error) {
	if g == nil {
		return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: nil graph")
	}
	if err := r.validate(); err != nil {
		return 0, 0, 0, 0, 0, 0, 0, 0, false, err
	}
	if err := r.validateFunctionActionExpressions(g); err != nil {
		return 0, 0, 0, 0, 0, 0, 0, 0, false, err
	}
	if len(r.facts) > 0 && g.analysis == nil {
		return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: rewrite %q has analysis actions but graph has no analysis", r.Name)
	}
	for _, action := range r.relations {
		arity := len(action.Columns)
		if action.Typed || action.TypedColumns != nil {
			arity = len(action.TypedColumns)
		}
		table, err := g.relation(action.Name, arity)
		if err != nil {
			return 0, 0, 0, 0, 0, 0, 0, 0, false, err
		}
		if action.Typed || action.TypedColumns != nil {
			for i, column := range action.TypedColumns {
				if err := validateActionColumnSchema(table.schema.Columns[i], column); err != nil {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: relation action %q column %d: %w", action.Name, i, err)
				}
				if column.HasValue {
					if _, err := g.canonicalTypedValue(table.schema.Columns[i], column.Value, fmt.Sprintf("relation action %q column %d", action.Name, i)); err != nil {
						return 0, 0, 0, 0, 0, 0, 0, 0, false, err
					}
				}
			}
		} else {
			for i, column := range action.Columns {
				if table.schema.Columns[i].Kind != FunctionClassColumn {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: relation action %q column %d requires typed columns", action.Name, i)
				}
				if err := validateActionTermSort(table.schema.Columns[i].Sort, column); err != nil {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, err
				}
			}
		}
	}
	for _, action := range r.functions {
		keyCount := len(action.Keys)
		if action.Typed || action.TypedKeys != nil {
			keyCount = len(action.TypedKeys)
		}
		table, err := g.function(action.Name, keyCount)
		if err != nil {
			return 0, 0, 0, 0, 0, 0, 0, 0, false, err
		}
		if action.Typed || action.TypedKeys != nil || action.TypedValue != nil {
			if !table.schema.typed || len(table.schema.KeyTypes) != len(action.TypedKeys) {
				return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: function action %q does not match a typed function schema", action.Name)
			}
			for i, column := range action.TypedKeys {
				if err := validateActionColumnSchema(table.schema.KeyTypes[i], column); err != nil {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: function action %q key %d: %w", action.Name, i, err)
				}
				if column.HasValue {
					if _, err := g.canonicalTypedValue(table.schema.KeyTypes[i], column.Value, fmt.Sprintf("function action %q key %d", action.Name, i)); err != nil {
						return 0, 0, 0, 0, 0, 0, 0, 0, false, err
					}
				}
			}
			if action.Kind == FunctionSetAction {
				if action.TypedValue == nil {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: function set action %q has no typed value", action.Name)
				}
				if err := validateActionColumnSchema(table.schema.ValueType, *action.TypedValue); err != nil {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: function action %q value: %w", action.Name, err)
				}
				if action.TypedValue.HasValue {
					if _, err := g.canonicalTypedValue(table.schema.ValueType, action.TypedValue.Value, fmt.Sprintf("function action %q value", action.Name)); err != nil {
						return 0, 0, 0, 0, 0, 0, 0, 0, false, err
					}
				}
			}
		} else {
			if table.schema.typed || len(table.schema.KeySorts) != len(action.Keys) {
				return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: function action %q does not match an e-class function schema", action.Name)
			}
			for i, term := range action.Keys {
				if err := validateActionTermSort(table.schema.KeySorts[i], term); err != nil {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: function action %q key %d: %w", action.Name, i, err)
				}
			}
			if action.Value != nil {
				if err := validateActionTermSort(table.schema.ValueSort, action.Value); err != nil {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: function action %q value: %w", action.Name, err)
				}
			}
		}
	}
	matches := suppliedMatches
	if !hasSuppliedMatches {
		if budget != nil {
			var complete bool
			matches, complete, err = r.searchMatchesWithBudget(g, budget)
			if err == nil && !complete {
				return 0, 0, 0, 0, 0, 0, 0, 0, false, errSaturationMatchWorkLimit
			}
		} else {
			matches, err = r.searchMatches(g)
		}
		if err != nil {
			return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: search rewrite %q: %w", r.Name, err)
		}
	}
	if functionBatch == nil {
		functionBatch = newFunctionActionBatch(g)
	}
	var lookupView *functionMergeView
	if !matchesPrepared {
		matches, err = r.filterClassMatchConditions(g, matches, budget)
		if err != nil {
			return 0, 0, 0, 0, 0, 0, 0, 0, false, err
		}
	}
	filterLookups := r.hasFilterFunctionActionLookup()
	guardsApplied := matchesPrepared
	if filterLookups && !matchesPrepared {
		view := cloneFunctionMergeView(functionBatch.view)
		filtered := matches[:0]
		for matchIndex, match := range matches {
			candidateView := cloneFunctionMergeView(view)
			match.lookupView = &candidateView
			match.BaseSubst = cloneBaseSubst(match.BaseSubst)
			accepted := true
			for bindingIndex := range r.primitiveBindings {
				if !budget.spend() {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, errSaturationMatchWorkLimit
				}
				binding := &r.primitiveBindings[bindingIndex]
				value, evalErr := evaluateFunctionActionExpr(g, &binding.Expr, match)
				if evalErr != nil {
					if errors.Is(evalErr, errPrimitiveNoValue) {
						accepted = false
						break
					}
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: rewrite %q match %d primitive binding %s: %w", r.Name, matchIndex, binding.Name, evalErr)
				}
				if old, exists := match.BaseSubst[binding.Name]; exists {
					if old != value {
						accepted = false
						break
					}
				} else {
					match.BaseSubst[binding.Name] = value
				}
			}
			if accepted {
				for conditionIndex := range r.primitiveConditions {
					if !budget.spend() {
						return 0, 0, 0, 0, 0, 0, 0, 0, false, errSaturationMatchWorkLimit
					}
					value, evalErr := evaluateFunctionActionExpr(g, &r.primitiveConditions[conditionIndex], match)
					if evalErr != nil {
						if errors.Is(evalErr, errPrimitiveNoValue) {
							accepted = false
							break
						}
						return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: rewrite %q match %d condition %d: %w", r.Name, matchIndex, conditionIndex, evalErr)
					}
					if value.Kind != FunctionBoolColumn {
						return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: rewrite %q condition %d returned kind %d, want bool", r.Name, conditionIndex, value.Kind)
					}
					if !value.Bool {
						accepted = false
						break
					}
				}
			}
			if accepted && r.Guard != nil {
				guardMatch := Match{ClassID: match.ClassID, Subst: match.Subst.Clone(), BaseSubst: cloneBaseSubst(match.BaseSubst)}
				guardRevision := g.revision
				apply, guardErr := r.Guard(g, guardMatch)
				if g.revision != guardRevision {
					rebuildErr := g.RebuildChecked()
					if guardErr != nil {
						return 0, 0, 0, 0, 0, 0, 0, 0, false, errors.Join(fmt.Errorf("egraph: rewrite %q guard mutated the graph: %w", r.Name, guardErr), rebuildErr)
					}
					return 0, 0, 0, 0, 0, 0, 0, 0, false, errors.Join(fmt.Errorf("egraph: rewrite %q guard mutated the graph", r.Name), rebuildErr)
				}
				if guardErr != nil {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: rewrite %q guard: %w", r.Name, guardErr)
				}
				accepted = apply
			}
			if accepted {
				match.lookupView = nil
				filtered = append(filtered, match)
				view = candidateView
			}
		}
		matches = filtered
		lookupView = &view
		for i := range matches {
			matches[i].lookupView = lookupView
		}
		guardsApplied = true
	} else if len(r.primitiveBindings) != 0 && !matchesPrepared {
		filtered := matches[:0]
		for matchIndex := range matches {
			match := matches[matchIndex]
			// SearchChecked leaves BaseSubst nil for ordinary term matches.
			// Give each candidate a private map before adding computed bindings.
			match.BaseSubst = cloneBaseSubst(match.BaseSubst)
			accepted := true
			for bindingIndex := range r.primitiveBindings {
				if !budget.spend() {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, errSaturationMatchWorkLimit
				}
				binding := &r.primitiveBindings[bindingIndex]
				value, evalErr := evaluateFunctionActionExpr(g, &binding.Expr, match)
				if evalErr != nil {
					if errors.Is(evalErr, errPrimitiveNoValue) {
						accepted = false
						break
					}
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: rewrite %q match %d primitive binding %s: %w", r.Name, matchIndex, binding.Name, evalErr)
				}
				if old, exists := match.BaseSubst[binding.Name]; exists {
					if old != value {
						accepted = false
						break
					}
				} else {
					match.BaseSubst[binding.Name] = value
				}
			}
			if accepted {
				filtered = append(filtered, match)
			}
		}
		matches = filtered
	}
	if !filterLookups && !matchesPrepared && len(r.primitiveConditions) != 0 {
		filtered := matches[:0]
		for matchIndex, match := range matches {
			accepted := true
			for conditionIndex := range r.primitiveConditions {
				if !budget.spend() {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, errSaturationMatchWorkLimit
				}
				value, err := evaluateFunctionActionExpr(g, &r.primitiveConditions[conditionIndex], match)
				if err != nil {
					if errors.Is(err, errPrimitiveNoValue) {
						accepted = false
						break
					}
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: rewrite %q match %d condition %d: %w", r.Name, matchIndex, conditionIndex, err)
				}
				if value.Kind != FunctionBoolColumn {
					return 0, 0, 0, 0, 0, 0, 0, 0, false, fmt.Errorf("egraph: rewrite %q condition %d returned kind %d, want bool", r.Name, conditionIndex, value.Kind)
				}
				if !value.Bool {
					accepted = false
					break
				}
			}
			if accepted {
				filtered = append(filtered, match)
			}
		}
		matches = filtered
	}
	defer func() {
		if merges > 0 || nodeDeletions > 0 {
			if rebuildErr := g.RebuildChecked(); rebuildErr != nil {
				err = errors.Join(err, fmt.Errorf("egraph: rebuild after rewrite %q: %w", r.Name, rebuildErr))
			}
		}
	}()
	acceptedMatches := make([]Match, 0, len(matches))
	for _, match := range matches {
		if !guardsApplied && r.Guard != nil {
			guardMatch := Match{ClassID: match.ClassID, Subst: match.Subst.Clone(), BaseSubst: cloneBaseSubst(match.BaseSubst)}
			guardRevision := g.revision
			apply, err := r.Guard(g, guardMatch)
			if g.revision != guardRevision {
				rebuildErr := g.RebuildChecked()
				if err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, errors.Join(fmt.Errorf("egraph: rewrite %q guard mutated the graph: %w", r.Name, err), rebuildErr)
				}
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, errors.Join(fmt.Errorf("egraph: rewrite %q guard mutated the graph", r.Name), rebuildErr)
			}
			if err != nil {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: rewrite %q guard: %w", r.Name, err)
			}
			if !apply {
				continue
			}
		}
		acceptedMatches = append(acceptedMatches, match)
	}
	if lookupView == nil && r.hasFunctionActionLookup() && !functionBatch.prepared {
		view := cloneFunctionMergeView(functionBatch.view)
		lookupView = &view
		for i := range acceptedMatches {
			acceptedMatches[i].lookupView = lookupView
		}
	}
	rightPatterns := r.rightPatterns()
	rightPatternWork := make([]int, len(rightPatterns))
	if len(acceptedMatches) > 0 {
		for i, rhs := range rightPatterns {
			rightPatternWork[i], err = patternInstantiationWork(rhs)
			if err != nil {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: rewrite %q rhs: %w", r.Name, err)
			}
		}
	}
	var preparedExpressions [][]preparedFunctionActionExpressions
	if !functionBatch.prepared {
		preparedExpressions, err = prepareFunctionActionExpressions(g, r.Name, r.functions, acceptedMatches)
		if err != nil {
			return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
		}
	}
	preparedRelationExpressions := make([][][]*FunctionValue, len(acceptedMatches))
	for mi, match := range acceptedMatches {
		preparedRelationExpressions[mi] = make([][]*FunctionValue, len(r.relations))
		for ai, action := range r.relations {
			if !(action.Typed || action.TypedColumns != nil) {
				continue
			}
			preparedRelationExpressions[mi][ai] = make([]*FunctionValue, len(action.TypedColumns))
			for ci, column := range action.TypedColumns {
				if column.Expr == nil {
					continue
				}
				value, err := evaluateFunctionActionExpr(g, column.Expr, match)
				if err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: rewrite %q relation action %q column %d: %w", r.Name, action.Name, ci, err)
				}
				canonical, err := canonicalFunctionActionValue(g, match, g.relations[action.Name].schema.Columns[ci], value, fmt.Sprintf("relation action %q column %d", action.Name, ci))
				if err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
				}
				preparedRelationExpressions[mi][ai][ci] = &canonical
			}
		}
	}
	if !functionBatch.prepared {
		if err := prepareComputedActionExpressions(g, r.Name, r.actionTermPatterns(), acceptedMatches); err != nil {
			return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
		}
	}
	if lookupView != nil {
		functionBatch.view = *lookupView
	}
	if maxNodes > 0 && lookupView != nil {
		_, nodes := graphSize(g)
		projected := uint64(nodes) + uint64(len(lookupView.fresh.nodes))
		if projected > uint64(maxNodes) {
			return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, true, nil
		}
	}
	// Capture only the prefix of matches that the node limit permits. Function
	// actions are resolved before unions, so staging later matches would retain
	// intents for work the match loop must skip.
	if commitFunctionBatch && maxNodes > 0 {
		_, nodes := graphSize(g)
		reserved := uint64(nodes)
		if lookupView != nil {
			reserved += uint64(len(lookupView.fresh.nodes))
		}
		limit := uint64(maxNodes)
		perMatch := rewriteActionTermNodeUpperBound(r)
		accepted := 0
		for accepted < len(acceptedMatches) && reserved <= limit {
			remaining := limit - reserved
			if perMatch > 0 && uint64(accepted+1) > remaining/perMatch {
				break
			}
			termReserve := perMatch * uint64(accepted+1)
			freshReserve := rewriteFreshDefaultNodeUpperBound(g, []capturedRewriteMatches{{rule: r, matches: acceptedMatches[:accepted+1]}}, lookupView)
			if freshReserve > remaining-termReserve {
				break
			}
			accepted++
		}
		if accepted < len(acceptedMatches) {
			nodeLimit = true
		}
		acceptedMatches = acceptedMatches[:accepted]
	}
	// Egglog applies table removals before the buffered insert actions, even
	// when a delete appears later in the rule's action list.
	for matchIndex, match := range acceptedMatches {
		for actionIndex, action := range r.relations {
			if action.Kind != RelationDelete {
				continue
			}
			if action.Typed || action.TypedColumns != nil {
				values := make([]FunctionValue, len(action.TypedColumns))
				for i, column := range action.TypedColumns {
					if preparedRelationExpressions[matchIndex][actionIndex][i] != nil {
						values[i] = *preparedRelationExpressions[matchIndex][actionIndex][i]
						continue
					}
					value, err := instantiateFunctionColumn(g, column, match)
					if err != nil {
						return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
					}
					values[i] = value
				}
				changed, err := g.DeleteRelationValues(action.Name, values...)
				if err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
				}
				if changed {
					relationChanges++
				}
				continue
			}
			tuple := make([]Id, len(action.Columns))
			for i, column := range action.Columns {
				id, err := column.InstantiateWithMatchChecked(g, match)
				if err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
				}
				tuple[i] = id
			}
			changed, err := g.DeleteRelation(action.Name, tuple...)
			if err != nil {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
			}
			if changed {
				relationChanges++
			}
		}
	}
	if !functionBatch.prepared && commitFunctionBatch {
		if err := r.stageCapturedFunctionActions(g, acceptedMatches, functionBatch); err != nil {
			return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
		}
	}
	if commitFunctionBatch {
		if err := functionBatch.commitDeletes(g); err != nil {
			return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
		}
	}
matchesLoop:
	for matchIndex, m := range acceptedMatches {
		if maxNodes > 0 {
			_, nodes := graphSize(g)
			projected := uint64(nodes) + patternNodeUpperBound(r.actionTermPatterns())
			if projected > uint64(maxNodes) {
				nodeLimit = true
				break matchesLoop
			}
		}
		subsumeCurrentMatch := func() error {
			if !r.subsumesLHS {
				return nil
			}
			changed, err := g.subsumeMatch(r.LHS, m)
			if changed {
				subsumptionChanges++
			}
			return err
		}
		var lhsClass *EClass
		if len(rightPatterns) > 0 {
			classID := g.Find(m.ClassID)
			if classID == InvalidId || int(classID) >= g.store.ClassCount() {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: rewrite %q matched a missing class", r.Name)
			}
			lhsClass = g.store.Class(classID)
			if lhsClass == nil {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: rewrite %q matched a missing class", r.Name)
			}
		}
		var targetSort string
		if lhsClass != nil {
			targetSort = lhsClass.Sort
		}
		for _, rhs := range rightPatterns {
			rhsSort := rhs.Sort
			if rhs.IsVar {
				id, ok := m.Subst[rhs.Op]
				if !ok {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: unbound pattern variable %s during instantiation", rhs.Op)
				}
				rhsClass := g.store.Class(g.Find(id))
				if rhsClass == nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: invalid e-class %d for pattern variable %s", id, rhs.Op)
				}
				if rhs.Sort != "" && rhs.Sort != rhsClass.Sort {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: pattern variable %s requires sort %q", rhs.Op, rhs.Sort)
				}
				rhsSort = rhsClass.Sort
			}
			if targetSort != rhsSort {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: rewrite %q changes sort from %q to %q", r.Name, targetSort, rhsSort)
			}
		}
		for rhsIndex, rhs := range rightPatterns {
			rhsWork := rightPatternWork[rhsIndex]
			report, err := egraphcore.ApplyMatches[Id, ENode, Match](
				g,
				[]Match{m},
				func(match Match) Id { return match.ClassID },
				func(match Match) (Id, error) { return rhs.InstantiateWithMatchChecked(g, match) },
				func(Match) int { return rhsWork },
				nil,
				coreWorkBudget(budget),
				false,
			)
			if err != nil {
				if errors.Is(err, egraphcore.ErrRewriteWorkBudget) {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, errSaturationMatchWorkLimit
				}
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
			}
			merges += report.Merges
			if maxNodes > 0 {
				_, nodes := graphSize(g)
				if nodes > maxNodes {
					nodeLimit = true
					if err := subsumeCurrentMatch(); err != nil {
						return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
					}
					break matchesLoop
				}
			}
		}
		for _, action := range r.unions {
			if action.Left.Sort != "" && action.Right.Sort != "" && action.Left.Sort != action.Right.Sort {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: union action changes sort from %q to %q", action.Left.Sort, action.Right.Sort)
			}
			leftID, err := action.Left.InstantiateWithMatchChecked(g, m)
			if err != nil {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
			}
			rightID, err := action.Right.InstantiateWithMatchChecked(g, m)
			if err != nil {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
			}
			if _, merged, err := g.UnionChecked(leftID, rightID); err != nil {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
			} else if merged {
				merges++
			}
			if maxNodes > 0 {
				_, nodes := graphSize(g)
				if nodes > maxNodes {
					nodeLimit = true
					if err := subsumeCurrentMatch(); err != nil {
						return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
					}
					break matchesLoop
				}
			}
		}
		for _, action := range r.facts {
			id := m.Subst[action.Variable]
			changed, err := g.MergeAnalysisData(id, action.Fact)
			if err != nil {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
			}
			if changed {
				analysisChanges++
			}
		}
		for actionIndex, action := range r.relations {
			if action.Kind == RelationDelete {
				continue
			}
			if action.Typed || action.TypedColumns != nil {
				values := make([]FunctionValue, len(action.TypedColumns))
				for i, column := range action.TypedColumns {
					if preparedRelationExpressions[matchIndex][actionIndex][i] != nil {
						values[i] = *preparedRelationExpressions[matchIndex][actionIndex][i]
						continue
					}
					value, err := instantiateFunctionColumn(g, column, m)
					if err != nil {
						return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
					}
					values[i] = value
				}
				var changed bool
				var err error
				switch action.Kind {
				case RelationInsert:
					changed, err = g.AddRelationValues(action.Name, values...)
				case RelationDelete:
					changed, err = g.DeleteRelationValues(action.Name, values...)
				case RelationSubsume:
					changed, err = g.SubsumeRelationValues(action.Name, values...)
				}
				if err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
				}
				if changed {
					if action.Kind == RelationSubsume {
						subsumptionChanges++
					} else {
						relationChanges++
					}
				}
				continue
			}
			tuple := make([]Id, len(action.Columns))
			for i, column := range action.Columns {
				id, err := column.InstantiateWithMatchChecked(g, m)
				if err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
				}
				tuple[i] = id
			}
			var changed bool
			var err error
			switch action.Kind {
			case RelationInsert:
				changed, err = g.AddRelation(action.Name, tuple...)
			case RelationDelete:
				changed, err = g.DeleteRelation(action.Name, tuple...)
			case RelationSubsume:
				changed, err = g.SubsumeRelation(action.Name, tuple...)
			}
			if err != nil {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
			}
			if changed {
				if action.Kind == RelationSubsume {
					subsumptionChanges++
				} else {
					relationChanges++
				}
			}
			if maxNodes > 0 {
				_, nodes := graphSize(g)
				if nodes > maxNodes {
					nodeLimit = true
					if err := subsumeCurrentMatch(); err != nil {
						return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
					}
					break matchesLoop
				}
			}
		}
		for actionIndex, action := range r.functions {
			if functionBatch.prepared {
				break
			}
			if action.Typed || action.TypedKeys != nil || action.TypedValue != nil {
				keys := make([]FunctionValue, len(action.TypedKeys))
				for i, column := range action.TypedKeys {
					var value FunctionValue
					if preparedExpressions[matchIndex][actionIndex].keys[i] != nil {
						value = *preparedExpressions[matchIndex][actionIndex].keys[i]
					} else {
						var err error
						value, err = instantiateFunctionColumn(g, column, m)
						if err != nil {
							return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
						}
					}
					keys[i] = value
				}
				switch action.Kind {
				case FunctionSetAction:
					var value FunctionValue
					if preparedExpressions[matchIndex][actionIndex].value != nil {
						value = *preparedExpressions[matchIndex][actionIndex].value
					} else {
						var err error
						value, err = instantiateFunctionColumn(g, *action.TypedValue, m)
						if err != nil {
							return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
						}
					}
					if err := functionBatch.stage(g, action.Kind, action.Name, keys, &value); err != nil {
						return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
					}
				case FunctionDeleteAction:
					if err := functionBatch.stage(g, action.Kind, action.Name, keys, nil); err != nil {
						return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
					}
				case FunctionSubsumeAction:
					if err := functionBatch.stage(g, action.Kind, action.Name, keys, nil); err != nil {
						return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
					}
				}
				if maxNodes > 0 {
					_, nodes := graphSize(g)
					if nodes > maxNodes {
						nodeLimit = true
						if err := subsumeCurrentMatch(); err != nil {
							return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
						}
						break matchesLoop
					}
				}
				continue
			}
			keys := make([]Id, len(action.Keys))
			for i, pattern := range action.Keys {
				id, err := pattern.InstantiateWithMatchChecked(g, m)
				if err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
				}
				keys[i] = id
			}
			switch action.Kind {
			case FunctionSetAction:
				value, err := action.Value.InstantiateWithMatchChecked(g, m)
				if err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
				}
				if err := functionBatch.stage(g, action.Kind, action.Name, classValues(keys), ptrFunctionValue(ClassValue(value))); err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
				}
			case FunctionDeleteAction:
				if err := functionBatch.stage(g, action.Kind, action.Name, classValues(keys), nil); err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
				}
			case FunctionSubsumeAction:
				if err := functionBatch.stage(g, action.Kind, action.Name, classValues(keys), nil); err != nil {
					return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
				}
			}
			if maxNodes > 0 {
				_, nodes := graphSize(g)
				if nodes > maxNodes {
					nodeLimit = true
					if err := subsumeCurrentMatch(); err != nil {
						return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
					}
					break matchesLoop

				}
			}
		}
		for actionIndex, action := range r.nodeDeletes {
			deleted, err := g.deleteNodeForMatch(action.Term, m)
			if err != nil {
				return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: rewrite %q node delete %d: %w", r.Name, actionIndex, err)
			}
			if deleted {
				nodeDeletions++
			}
		}
		if err := subsumeCurrentMatch(); err != nil {
			return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, err
		}
	}

	if commitFunctionBatch {
		inserts, updates, deletes, subsumptions, commitErr := functionBatch.commit(g)
		if commitErr != nil {
			return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, false, fmt.Errorf("egraph: commit rewrite %q function actions: %w", r.Name, commitErr)
		}
		functionInserts += inserts
		functionUpdates += updates
		functionDeletes += deletes
		subsumptionChanges += subsumptions
	}
	return merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, nodeLimit, nil
}

func validateActionColumnSchema(typ FunctionColumnType, column FunctionColumnPattern) error {
	kind := column.Kind
	if kind == 0 && column.Term != nil {
		kind = FunctionClassColumn
	}
	if kind != typ.Kind {
		return fmt.Errorf("column kind %d does not match schema kind %d", kind, typ.Kind)
	}
	if typ.Kind == FunctionClassColumn && column.Term != nil && column.Term.Sort != "" && typ.Sort != "" && column.Term.Sort != typ.Sort {
		return fmt.Errorf("term requires sort %q, schema requires sort %q", column.Term.Sort, typ.Sort)
	}
	return nil
}

func canonicalFunctionActionValue(g *EGraph, match Match, typ FunctionColumnType, value FunctionValue, where string) (FunctionValue, error) {
	if match.lookupView != nil {
		return g.canonicalTypedValueWithView(*match.lookupView, typ, value, where)
	}
	return g.canonicalTypedValue(typ, value, where)
}

func validateActionTermSort(want string, term *Pattern) error {
	if term != nil && want != "" && term.Sort != "" && term.Sort != want {
		return fmt.Errorf("term requires sort %q, schema requires sort %q", term.Sort, want)
	}
	return nil
}

type preparedFunctionActionExpressions struct {
	keys  []*FunctionValue
	value *FunctionValue
}

func (r *Rewrite) actionTermPatterns() []*Pattern {
	patterns := append([]*Pattern(nil), r.rightPatterns()...)
	for _, action := range r.unions {
		patterns = append(patterns, action.Left, action.Right)
	}
	for _, action := range r.relations {
		patterns = append(patterns, action.Columns...)
		for _, column := range action.TypedColumns {
			patterns = append(patterns, column.Term)
		}
	}
	for _, action := range r.functions {
		patterns = append(patterns, action.Keys...)
		if action.Value != nil {
			patterns = append(patterns, action.Value)
		}
		for _, column := range action.TypedKeys {
			patterns = append(patterns, column.Term)
		}
		if action.TypedValue != nil {
			patterns = append(patterns, action.TypedValue.Term)
		}
	}
	return patterns
}

func rewriteActionTermNodeUpperBound(rule *Rewrite) uint64 {
	patterns := rule.actionTermPatterns()
	for _, action := range rule.nodeDeletes {
		patterns = append(patterns, action.Term)
	}
	return patternNodeUpperBound(patterns)
}

func rewriteFreshDefaultNodeUpperBound(g *EGraph, entries []capturedRewriteMatches, view *functionMergeView) uint64 {
	if view == nil {
		created := newFunctionMergeView(g)
		view = &created
	}
	seen := make(map[string]struct{})
	var bound uint64
	for entryIndex, entry := range entries {
		for actionIndex, action := range entry.rule.functions {
			if action.Kind != FunctionSubsumeAction {
				continue
			}
			table := g.functions[action.Name]
			if table == nil || table.schema.Default.Kind != FunctionDefaultFreshID {
				continue
			}
			for matchIndex, match := range entry.matches {
				var keys []FunctionValue
				known := true
				if action.Typed || action.TypedKeys != nil || action.TypedValue != nil {
					keys = make([]FunctionValue, len(action.TypedKeys))
					for i, column := range action.TypedKeys {
						value, ok := capturedFunctionColumnValue(g, column, match)
						if !ok {
							known = false
							break
						}
						keys[i] = value
					}
				} else {
					keys = make([]FunctionValue, len(action.Keys))
					for i, pattern := range action.Keys {
						if pattern == nil || !pattern.IsVar {
							known = false
							break
						}
						id, ok := match.Subst[pattern.Op]
						if !ok {
							known = false
							break
						}
						keys[i] = ClassValue(g.Find(id))
					}
				}
				var key string
				if known {
					canonical, err := g.canonicalTypedKeysWithView(*view, table, keys)
					if err != nil {
						known = false
					} else {
						key = typedFunctionKey(canonical)
						staged := view.tables[action.Name]
						if table.schema.typed {
							if _, exists := staged.getTyped(key); exists {
								continue
							}
						} else if _, exists := staged.getLegacy(relationKey(idsFromFunctionValues(canonical))); exists {
							continue
						}
					}
				}
				if !known {
					key = fmt.Sprintf("unknown:%d:%d:%d", entryIndex, actionIndex, matchIndex)
				}
				identity := action.Name + "\x00" + key
				if _, duplicate := seen[identity]; duplicate {
					continue
				}
				seen[identity] = struct{}{}
				bound++
			}
		}
	}
	return bound
}

func capturedFunctionColumnValue(g *EGraph, column FunctionColumnPattern, match Match) (FunctionValue, bool) {
	if column.Expr != nil {
		if column.Expr.kind != functionActionVariable {
			return FunctionValue{}, false
		}
		value, ok := match.BaseSubst[column.Expr.variable]
		return value, ok && value.Kind == column.Kind
	}
	if column.Term != nil {
		if !column.Term.IsVar {
			return FunctionValue{}, false
		}
		id, ok := match.Subst[column.Term.Op]
		return ClassValue(g.Find(id)), ok
	}
	if column.Variable != "" {
		if id, ok := match.Subst[column.Variable]; ok {
			return ClassValue(g.Find(id)), column.Kind == FunctionClassColumn
		}
		value, ok := match.BaseSubst[column.Variable]
		return value, ok && value.Kind == column.Kind
	}
	if column.HasValue {
		return column.Value, true
	}
	return FunctionValue{}, false
}

// patternNodeUpperBound counts the maximum number of ENode additions across
// separate pattern instantiations. Pattern identity matches each instantiator's
// cache, so shared subpatterns within one root are counted once. Hash-consing
// can make the actual count smaller.
func patternNodeUpperBound(patterns []*Pattern) uint64 {
	var count uint64
	for _, root := range patterns {
		seen := make(map[*Pattern]bool)
		stack := []*Pattern{root}
		for len(stack) != 0 {
			last := len(stack) - 1
			pattern := stack[last]
			stack = stack[:last]
			if pattern == nil || seen[pattern] {
				continue
			}
			seen[pattern] = true
			if pattern.RelationName == "" && pattern.FunctionName == "" && !pattern.IsVar {
				count++
			}
			stack = append(stack, pattern.Children...)
		}
	}
	return count
}

func prepareComputedActionExpressions(g *EGraph, rewriteName string, patterns []*Pattern, matches []Match) error {
	for matchIndex := range matches {
		match := &matches[matchIndex]
		match.computedValues = make(map[*Pattern]FunctionValue)
		seen := make(map[*Pattern]bool)
		stack := append([]*Pattern(nil), patterns...)
		for len(stack) != 0 {
			pattern := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if pattern == nil || seen[pattern] {
				continue
			}
			seen[pattern] = true
			if pattern.int64Expr != nil && pattern.float64Expr != nil {
				return fmt.Errorf("egraph: rewrite %q match %d computed leaf has conflicting types", rewriteName, matchIndex)
			}
			expr := pattern.int64Expr
			want := FunctionInt64Column
			if pattern.float64Expr != nil {
				expr = pattern.float64Expr
				want = FunctionFloat64Column
			}
			if expr != nil {
				value, err := evaluateFunctionActionExpr(g, expr, *match)
				if err != nil {
					return fmt.Errorf("egraph: rewrite %q match %d computed leaf: %w", rewriteName, matchIndex, err)
				}
				if value.Kind != want {
					return fmt.Errorf("egraph: rewrite %q match %d computed leaf has kind %d, want %d", rewriteName, matchIndex, value.Kind, want)
				}
				if want == FunctionFloat64Column && math.IsNaN(value.Float64) {
					return fmt.Errorf("egraph: rewrite %q match %d computed f64 leaf rejects NaN", rewriteName, matchIndex)
				}
				match.computedValues[pattern] = value
			}
			stack = append(stack, pattern.Children...)
		}
	}
	return nil
}

func prepareFunctionActionExpressions(g *EGraph, rewriteName string, actions []FunctionAction, matches []Match) ([][]preparedFunctionActionExpressions, error) {
	prepared := make([][]preparedFunctionActionExpressions, len(matches))
	for matchIndex, match := range matches {
		prepared[matchIndex] = make([]preparedFunctionActionExpressions, len(actions))
		for actionIndex, action := range actions {
			if !(action.Typed || action.TypedKeys != nil || action.TypedValue != nil) {
				continue
			}
			table := g.functions[action.Name]
			if table == nil || !table.schema.typed {
				return nil, fmt.Errorf("egraph: rewrite %q function action %q has no typed schema", rewriteName, action.Name)
			}
			cached := &prepared[matchIndex][actionIndex]
			cached.keys = make([]*FunctionValue, len(action.TypedKeys))
			for columnIndex, column := range action.TypedKeys {
				if column.Expr == nil {
					continue
				}
				value, err := evaluateFunctionActionExpr(g, column.Expr, match)
				if err != nil {
					return nil, fmt.Errorf("egraph: rewrite %q match %d function action %q key %d: %w", rewriteName, matchIndex, action.Name, columnIndex, err)
				}
				value, err = canonicalFunctionActionValue(g, match, table.schema.KeyTypes[columnIndex], value, fmt.Sprintf("function action %q key %d", action.Name, columnIndex))
				if err != nil {
					return nil, fmt.Errorf("egraph: rewrite %q match %d: %w", rewriteName, matchIndex, err)
				}
				cached.keys[columnIndex] = &value
			}
			if action.TypedValue != nil && action.TypedValue.Expr != nil {
				value, err := evaluateFunctionActionExpr(g, action.TypedValue.Expr, match)
				if err != nil {
					return nil, fmt.Errorf("egraph: rewrite %q match %d function action %q value: %w", rewriteName, matchIndex, action.Name, err)
				}
				value, err = canonicalFunctionActionValue(g, match, table.schema.ValueType, value, fmt.Sprintf("function action %q value", action.Name))
				if err != nil {
					return nil, fmt.Errorf("egraph: rewrite %q match %d: %w", rewriteName, matchIndex, err)
				}
				cached.value = &value
			}
		}
	}
	return prepared, nil
}

// stageCapturedFunctionActions resolves every function action while matches
// still describe the pre-action graph, then appends its write to batch.
func (r *Rewrite) stageCapturedFunctionActions(g *EGraph, matches []Match, batch *functionActionBatch) error {
	for i := range matches {
		matches[i].lookupView = &batch.view
	}
	prepared, err := prepareFunctionActionExpressions(g, r.Name, r.functions, matches)
	if err != nil {
		return err
	}
	if err := prepareComputedActionExpressions(g, r.Name, r.actionTermPatterns(), matches); err != nil {
		return err
	}
	for matchIndex, match := range matches {
		for actionIndex, action := range r.functions {
			if action.Typed || action.TypedKeys != nil || action.TypedValue != nil {
				keys := make([]FunctionValue, len(action.TypedKeys))
				for i, column := range action.TypedKeys {
					if prepared[matchIndex][actionIndex].keys[i] != nil {
						keys[i] = *prepared[matchIndex][actionIndex].keys[i]
						continue
					}
					value, err := instantiateFunctionColumn(g, column, match)
					if err != nil {
						return fmt.Errorf("egraph: rewrite %q match %d function action %q key %d: %w", r.Name, matchIndex, action.Name, i, err)
					}
					keys[i] = value
				}
				var value *FunctionValue
				if action.Kind == FunctionSetAction {
					if prepared[matchIndex][actionIndex].value != nil {
						copyValue := *prepared[matchIndex][actionIndex].value
						value = &copyValue
					} else {
						instantiated, err := instantiateFunctionColumn(g, *action.TypedValue, match)
						if err != nil {
							return fmt.Errorf("egraph: rewrite %q match %d function action %q value: %w", r.Name, matchIndex, action.Name, err)
						}
						value = &instantiated
					}
				}
				if err := batch.stage(g, action.Kind, action.Name, keys, value); err != nil {
					return fmt.Errorf("egraph: rewrite %q match %d function action %q: %w", r.Name, matchIndex, action.Name, err)
				}
				continue
			}
			keys := make([]Id, len(action.Keys))
			for i, pattern := range action.Keys {
				id, err := pattern.InstantiateWithMatchChecked(g, match)
				if err != nil {
					return fmt.Errorf("egraph: rewrite %q match %d function action %q key %d: %w", r.Name, matchIndex, action.Name, i, err)
				}
				keys[i] = id
			}
			var value *FunctionValue
			if action.Kind == FunctionSetAction {
				id, err := action.Value.InstantiateWithMatchChecked(g, match)
				if err != nil {
					return fmt.Errorf("egraph: rewrite %q match %d function action %q value: %w", r.Name, matchIndex, action.Name, err)
				}
				classValue := ClassValue(id)
				value = &classValue
			}
			if err := batch.stage(g, action.Kind, action.Name, classValues(keys), value); err != nil {
				return fmt.Errorf("egraph: rewrite %q match %d function action %q: %w", r.Name, matchIndex, action.Name, err)
			}
		}
	}
	batch.prepared = true
	return nil
}

func prepareCapturedActionLookups(g *EGraph, entries []capturedRewriteMatches, view *functionMergeView) error {
	for _, entry := range entries {
		for i := range entry.matches {
			match := &entry.matches[i]
			match.lookupView = view
			if _, err := prepareFunctionActionExpressions(g, entry.rule.Name, entry.rule.functions, []Match{*match}); err != nil {
				return err
			}
			if err := prepareComputedActionExpressions(g, entry.rule.Name, entry.rule.actionTermPatterns(), []Match{*match}); err != nil {
				return err
			}
			for _, action := range entry.rule.relations {
				for _, column := range action.TypedColumns {
					if column.Expr == nil {
						continue
					}
					if _, err := evaluateFunctionActionExpr(g, column.Expr, *match); err != nil {
						return fmt.Errorf("rewrite %q relation action %q lookup: %w", entry.rule.Name, action.Name, err)
					}
				}
			}
		}
	}
	return nil
}

func (r *Rewrite) validate() error {
	if r == nil || r.LHS == nil || r.RHS == nil && len(r.actions) == 0 && len(r.unions) == 0 && len(r.facts) == 0 && len(r.relations) == 0 && len(r.functions) == 0 && len(r.nodeDeletes) == 0 {
		return fmt.Errorf("egraph: invalid rewrite")
	}
	validationRHS := r.rightPatterns()
	if len(validationRHS) == 0 && len(r.nodeDeletes) != 0 {
		// Validate the joined LHS and action bindings through the shared
		// constructor checks without inventing a graph mutation.
		validationRHS = []*Pattern{r.LHS}
	}
	if _, err := newActionRewriteWithPrimitiveBindings(r.Name, r.leftPatterns(), validationRHS, r.unions, r.facts, r.relations, r.functions, r.Guard, r.primitiveBindings); err != nil {
		return err
	}
	bindings, err := functionActionBindingTypes(r.leftPatterns())
	if err != nil {
		return err
	}
	classes := functionActionClassBindings(r.leftPatterns())
	for i, condition := range r.classConditions {
		if !classes[condition.Left] || !classes[condition.Right] {
			return fmt.Errorf("egraph: rewrite %q class condition %d requires class-bound lhs variables %q and %q", r.Name, i, condition.Left, condition.Right)
		}
	}
	for i, condition := range r.classPatternConditions {
		if !classes[condition.Variable] {
			return fmt.Errorf("egraph: rewrite %q class pattern condition %d requires class-bound lhs variable %q", r.Name, i, condition.Variable)
		}
		if err := validateTermPattern(condition.Pattern); err != nil || condition.Pattern.IsVar || patternHasComputedLeaf(condition.Pattern) {
			return fmt.Errorf("egraph: rewrite %q class pattern condition %d requires a ground term pattern", r.Name, i)
		}
		variables := make(map[string]bool)
		collectVariables(condition.Pattern, variables)
		if len(variables) != 0 {
			return fmt.Errorf("egraph: rewrite %q class pattern condition %d requires a ground term pattern", r.Name, i)
		}
	}
	for _, binding := range r.primitiveBindings {
		kind, err := functionActionExpressionTypeWithClasses(&binding.Expr, bindings, classes)
		if err != nil {
			return fmt.Errorf("egraph: rewrite %q primitive binding %s: %w", r.Name, binding.Name, err)
		}
		bindings[binding.Name] = kind
	}
	for i := range r.primitiveConditions {
		kind, err := functionActionExpressionTypeWithClasses(&r.primitiveConditions[i], bindings, classes)
		if err != nil {
			return fmt.Errorf("egraph: rewrite %q primitive condition %d: %w", r.Name, i, err)
		}
		if kind != FunctionBoolColumn {
			return fmt.Errorf("egraph: rewrite %q primitive condition %d has kind %d, want bool", r.Name, i, kind)
		}
	}
	for _, pattern := range r.rightPatterns() {
		if err := validateComputedPatterns(pattern, bindings, classes); err != nil {
			return fmt.Errorf("egraph: rewrite %q computed RHS: %w", r.Name, err)
		}
	}
	if r.subsumesLHS {
		if err := validateTermPattern(r.LHS); err != nil || r.LHS.IsVar {
			return fmt.Errorf("egraph: rewrite %q can only subsume a concrete term pattern", r.Name)
		}
	}
	bound := make(map[string]bool)
	for _, pattern := range r.leftPatterns() {
		collectVariables(pattern, bound)
	}
	for i, action := range r.nodeDeletes {
		if err := validateTermPattern(action.Term); err != nil || action.Term.IsVar || patternHasComputedLeaf(action.Term) {
			return fmt.Errorf("egraph: rewrite %q node delete %d requires a concrete term pattern", r.Name, i)
		}
		if r.LHS.RelationName != "" || r.LHS.FunctionName != "" {
			return fmt.Errorf("egraph: rewrite %q node delete requires a term-pattern lhs", r.Name)
		}
		variables := make(map[string]bool)
		collectVariables(action.Term, variables)
		for variable := range variables {
			if !bound[variable] {
				return fmt.Errorf("egraph: rewrite %q node delete variable %s is not bound by lhs", r.Name, variable)
			}
		}
	}
	return nil
}

func (r *Rewrite) rightPatterns() []*Pattern {
	patterns := make([]*Pattern, 0, 1+len(r.actions))
	if r.RHS != nil {
		patterns = append(patterns, r.RHS)
	}
	return append(patterns, r.actions...)
}

func (r *Rewrite) leftPatterns() []*Pattern {
	patterns := make([]*Pattern, 1, 1+len(r.conditions))
	patterns[0] = r.LHS
	return append(patterns, r.conditions...)
}

// joinMatchSets joins two independently searched pattern result sets. It is
// shared by the full search and the relation delta path so both paths use the
// same canonical class and typed-binding compatibility rules.
func joinMatchSets(g *EGraph, matches, candidates []Match, bound, candidateVariables map[string]bool) ([]Match, error) {
	joined, _, err := joinMatchSetsWithBudget(g, matches, candidates, bound, candidateVariables, nil)
	return joined, err
}

func joinMatchSetsWithBudget(g *EGraph, matches, candidates []Match, bound, candidateVariables map[string]bool, budget *matchWorkBudget) ([]Match, bool, error) {
	var shared []string
	for name := range candidateVariables {
		if bound[name] {
			shared = append(shared, name)
		}
	}
	if !budget.spendSort(len(shared)) {
		return nil, false, nil
	}
	sort.Strings(shared)
	leftBindings, leftNaNs, complete := encodeCoreBindings(g, matches, true, 1, budget)
	if !complete {
		return nil, false, nil
	}
	rightBindings, rightNaNs, complete := encodeCoreBindings(g, candidates, false, 1<<32, budget)
	if !complete {
		return nil, false, nil
	}
	coreKeys := make([]string, 0, len(shared))
	for _, name := range shared {
		if bound[name] && candidateVariables[name] {
			coreKeys = append(coreKeys, "join:"+name)
		}
	}
	coreMatches, complete, err := egraphcore.JoinBindings(leftBindings, rightBindings, coreKeys, func(id Id) Id { return g.Find(id) }, coreWorkBudget(budget))
	if err != nil || !complete {
		return nil, complete, err
	}
	allNames := make(map[string]bool, len(bound)+len(candidateVariables))
	for name := range bound {
		allNames[name] = true
	}
	for name := range candidateVariables {
		allNames[name] = true
	}
	orderedNames := make([]string, 0, len(allNames))
	for name := range allNames {
		orderedNames = append(orderedNames, name)
	}
	if !budget.spendSort(len(orderedNames)) {
		return nil, false, nil
	}
	sort.Strings(orderedNames)
	joined := make([]Match, 0, len(coreMatches))
	seen := make(map[string]bool)
	for _, binding := range coreMatches {
		if !budget.spend() {
			return nil, false, nil
		}
		subst := make(Subst)
		bases := make(map[string]FunctionValue)
		for key, value := range binding {
			switch {
			case strings.HasPrefix(key, "class:"):
				subst[strings.TrimPrefix(key, "class:")] = value.Class
			case strings.HasPrefix(key, "base:"):
				original, ok := leftNaNs[value.Float]
				if !ok {
					original, ok = rightNaNs[value.Float]
				}
				if ok && value.Kind == egraphcore.Float64Value {
					bases[strings.TrimPrefix(key, "base:")] = original
				} else {
					converted, valid := functionValueFromCore(value)
					if !valid {
						return nil, false, fmt.Errorf("egraph: invalid core binding value")
					}
					bases[strings.TrimPrefix(key, "base:")] = converted
				}
			}
		}
		classID := InvalidId
		if value, ok := binding["anchor:class-id"]; ok {
			classID = Id(value.Int64)
		}
		key := fmt.Sprintf("%d:%s", classID, typedSubstitutionKey(subst, bases, orderedNames, g))
		if seen[key] {
			continue
		}
		seen[key] = true
		joined = append(joined, Match{ClassID: classID, Subst: subst, BaseSubst: bases})
	}
	return joined, true, nil
}

func coreWorkBudget(budget *matchWorkBudget) *egraphcore.WorkBudget {
	if budget == nil {
		return nil
	}
	return budget.core
}

func encodeCoreBindings(g *EGraph, matches []Match, keepAnchor bool, nanStart uint64, budget *matchWorkBudget) ([]egraphcore.Binding[Id], map[uint64]FunctionValue, bool) {
	bindings := make([]egraphcore.Binding[Id], len(matches))
	nans := make(map[uint64]FunctionValue)
	nanSequence := nanStart
	for i, match := range matches {
		if !budget.spend() {
			return nil, nil, false
		}
		binding := make(egraphcore.Binding[Id], len(match.Subst)+len(match.BaseSubst)*2+1)
		for name := range match.Subst {
			if _, collision := match.BaseSubst[name]; collision {
				bindings[i] = nil
				break
			}
			binding["join:"+name] = egraphcore.Int64[Id](1)
		}
		if bindings[i] == nil {
			continue
		}
		for name, id := range match.Subst {
			if !budget.spend() {
				return nil, nil, false
			}
			binding["class:"+name] = egraphcore.Class(Id(id))
		}
		for name, value := range match.BaseSubst {
			if !budget.spend() {
				return nil, nil, false
			}
			converted, valid := functionValueToCore(value)
			if !valid {
				return nil, nil, false
			}
			if value.Kind == FunctionFloat64Column && math.IsNaN(value.Float64) {
				bits := uint64(0x7ff8000000000000) + nanSequence
				nanSequence++
				converted.Float = bits
				nans[bits] = value
			}
			binding["base:"+name] = converted
			binding["join:"+name] = egraphcore.Int64[Id](2)
		}
		if keepAnchor {
			anchor := match.ClassID
			if anchor != InvalidId {
				anchor = g.Find(anchor)
			}
			binding["anchor:class-id"] = egraphcore.Int64[Id](int64(anchor))
		}
		bindings[i] = binding
	}
	valid := bindings[:0]
	for _, binding := range bindings {
		if !budget.spend() {
			return nil, nil, false
		}
		if binding != nil {
			valid = append(valid, binding)
		}
	}
	return valid, nans, true
}

func functionValueToCore(value FunctionValue) (egraphcore.Value[Id], bool) {
	switch value.Kind {
	case FunctionClassColumn:
		return egraphcore.Class(Id(value.ID)), true
	case FunctionInt64Column:
		return egraphcore.Int64[Id](value.Int64), true
	case FunctionBoolColumn:
		return egraphcore.Bool[Id](value.Bool), true
	case FunctionFloat64Column:
		return egraphcore.Float64[Id](value.Float64), true
	case FunctionStringColumn:
		return egraphcore.String[Id](value.String), true
	case FunctionUnitColumn:
		return egraphcore.Unit[Id](), true
	case FunctionBigIntColumn:
		return egraphcore.BigInt[Id](value.BigInt), true
	case FunctionBigRatColumn:
		return egraphcore.BigRat[Id](value.BigRat), true
	default:
		return egraphcore.Value[Id]{}, false
	}
}

func functionValueFromCore(value egraphcore.Value[Id]) (FunctionValue, bool) {
	switch value.Kind {
	case egraphcore.ClassValue:
		return ClassValue(Id(value.Class)), true
	case egraphcore.Int64Value:
		return Int64Value(value.Int64), true
	case egraphcore.BoolValue:
		return BoolValue(value.Bool), true
	case egraphcore.Float64Value:
		return Float64Value(math.Float64frombits(value.Float)), true
	case egraphcore.StringValue:
		return StringValue(value.String), true
	case egraphcore.UnitValue:
		return UnitValue(), true
	case egraphcore.BigIntValue:
		return FunctionValue{Kind: FunctionBigIntColumn, BigInt: value.String}, true
	case egraphcore.BigRatValue:
		return FunctionValue{Kind: FunctionBigRatColumn, BigRat: value.String}, true
	default:
		return FunctionValue{}, false
	}
}

func (r *Rewrite) searchMatches(g *EGraph) ([]Match, error) {
	matches, _, err := r.searchMatchesWithBudget(g, nil)
	return matches, err
}

func (r *Rewrite) searchMatchesWithBudget(g *EGraph, budget *matchWorkBudget) ([]Match, bool, error) {
	patterns := r.leftPatterns()
	matches, complete, err := patterns[0].searchChecked(g, budget)
	if err != nil || len(matches) == 0 {
		return matches, complete, err
	}
	if !complete {
		return nil, false, nil
	}
	bound := make(map[string]bool)
	collectVariables(patterns[0], bound)
	type condition struct {
		matches []Match
		vars    map[string]bool
	}
	pending := make([]condition, 0, len(patterns)-1)
	for _, pattern := range patterns[1:] {
		conditionMatches, complete, err := pattern.searchChecked(g, budget)
		if err != nil {
			return nil, false, err
		}
		if !complete {
			return nil, false, nil
		}
		if len(conditionMatches) == 0 {
			return nil, true, nil
		}
		variables := make(map[string]bool)
		collectVariables(pattern, variables)
		pending = append(pending, condition{matches: conditionMatches, vars: variables})
	}
	boundNames := make([]string, 0, len(bound))
	for name := range bound {
		boundNames = append(boundNames, name)
	}
	planConditions := make([]egraphcore.JoinOrderCondition, len(pending))
	for i, candidate := range pending {
		variables := make([]string, 0, len(candidate.vars))
		for name := range candidate.vars {
			variables = append(variables, name)
		}
		planConditions[i] = egraphcore.JoinOrderCondition{Variables: variables, Cardinality: len(candidate.matches)}
	}
	order, err := egraphcore.PlanJoinOrder(boundNames, planConditions, coreWorkBudget(budget))
	if err != nil {
		if errors.Is(err, egraphcore.ErrMatchWorkLimit) {
			return nil, false, nil
		}
		return nil, false, err
	}
	for _, conditionIndex := range order {
		currentCondition := pending[conditionIndex]
		conditionVariables := currentCondition.vars
		matches, complete, err = joinMatchSetsWithBudget(g, matches, currentCondition.matches, bound, conditionVariables, budget)
		if err != nil {
			return nil, false, err
		}
		if !complete {
			return nil, false, nil
		}
		if len(matches) == 0 {
			return nil, true, nil
		}
		for name := range conditionVariables {
			bound[name] = true
		}
	}
	return matches, true, nil
}

type relationMatchTracker struct {
	previous    [][]Match
	keys        []map[string]struct{}
	cursors     []int
	coreCursors []egraphcore.RelationCursor
	revision    uint64
	ready       bool
}

type termMatchTracker struct {
	matches []map[Id]map[string]Match
	cursor  int
	ready   bool
}

func semiNaiveTermEligible(r *Rewrite) bool {
	if r == nil || r.LHS == nil || r.RHS == nil || r.Guard != nil || len(r.conditions) != 0 || len(r.actions) != 0 || len(r.unions) != 0 || len(r.facts) != 0 || len(r.relations) != 0 || len(r.functions) != 0 || len(r.primitiveConditions) != 0 || len(r.classConditions) != 0 || len(r.classPatternConditions) != 0 || len(r.primitiveBindings) != 0 || len(r.nodeDeletes) != 0 || r.subsumesLHS {
		return false
	}
	var ordinaryTerm func(*Pattern) bool
	ordinaryTerm = func(p *Pattern) bool {
		if p == nil || p.RelationName != "" || p.FunctionName != "" {
			return false
		}
		for _, child := range p.Children {
			if !ordinaryTerm(child) {
				return false
			}
		}
		return true
	}
	for _, p := range r.leftPatterns() {
		if !ordinaryTerm(p) {
			return false
		}
	}
	return true
}

func (r *Rewrite) semiNaiveTermMatches(g *EGraph, tracker *termMatchTracker, budget *matchWorkBudget) ([]Match, error) {
	patterns := r.leftPatterns()
	variables := make([]map[string]bool, len(patterns))
	names := make([][]string, len(patterns))
	for i, pattern := range patterns {
		variables[i] = make(map[string]bool)
		collectVariables(pattern, variables[i])
		names[i] = sortedVariableNames(variables[i])
	}
	if !tracker.ready || tracker.cursor > len(g.termChanges) || g.dirty || len(tracker.matches) != len(patterns) {
		current := make([]map[Id]map[string]Match, len(patterns))
		sets := make([][]Match, len(patterns))
		for i, pattern := range patterns {
			matches, complete, err := pattern.searchChecked(g, budget)
			if err != nil {
				return nil, err
			}
			if !complete {
				return nil, errSaturationMatchWorkLimit
			}
			sets[i] = matches
			current[i] = cacheTermMatches(matches, names[i], g, budget)
			if current[i] == nil {
				return nil, errSaturationMatchWorkLimit
			}
		}
		matches, err := joinTermPatternMatches(g, sets, variables, budget)
		if err != nil {
			return nil, err
		}
		tracker.matches, tracker.cursor, tracker.ready = current, len(g.termChanges), true
		return matches, nil
	}
	changes := g.termChanges[tracker.cursor:]
	if len(changes) == 0 {
		return nil, nil
	}
	frontier, roots, err := termChangeFrontier(g, changes, budget)
	if err != nil {
		return nil, err
	}
	previous := tracker.matches
	current := make([]map[Id]map[string]Match, len(patterns))
	deltas := make([][]Match, len(patterns))
	for i, pattern := range patterns {
		updated := make(map[Id]map[string]Match, len(tracker.matches[i]))
		for id, group := range tracker.matches[i] {
			updated[id] = group
		}
		previousAffected := make(map[Id]map[string]bool)
		affectedCacheIDs := make(map[Id]bool)
		for _, id := range frontier {
			root := g.Find(id)
			for _, cacheID := range []Id{id, root} {
				if cacheID == InvalidId || affectedCacheIDs[cacheID] {
					continue
				}
				affectedCacheIDs[cacheID] = true
				for _, oldMatch := range updated[cacheID] {
					if !budget.spend() {
						return nil, errSaturationMatchWorkLimit
					}
					if previousAffected[root] == nil {
						previousAffected[root] = make(map[string]bool)
					}
					previousAffected[root][typedSubstitutionKey(oldMatch.Subst, oldMatch.BaseSubst, names[i], g)] = true
				}
			}
		}
		for cacheID := range affectedCacheIDs {
			delete(updated, cacheID)
		}
		matches, complete, err := pattern.searchClassesChecked(g, roots, budget)
		if err != nil {
			return nil, err
		}
		if !complete {
			return nil, errSaturationMatchWorkLimit
		}
		for _, match := range matches {
			if !budget.spend() {
				return nil, errSaturationMatchWorkLimit
			}
			root := g.Find(match.ClassID)
			key := typedSubstitutionKey(match.Subst, match.BaseSubst, names[i], g)
			if !previousAffected[root][key] {
				deltas[i] = append(deltas[i], match)
			}
			if updated[root] == nil {
				updated[root] = make(map[string]Match)
			}
			updated[root][key] = match
		}
		current[i] = updated
	}
	matches, err := joinTermMatchDeltas(g, previous, current, deltas, variables, budget)
	if err != nil {
		return nil, err
	}
	tracker.matches, tracker.cursor = current, len(g.termChanges)
	return matches, nil
}

func cacheTermMatches(matches []Match, names []string, g *EGraph, budget *matchWorkBudget) map[Id]map[string]Match {
	cache := make(map[Id]map[string]Match)
	for _, match := range matches {
		if !budget.spend() {
			return nil
		}
		root := g.Find(match.ClassID)
		if cache[root] == nil {
			cache[root] = make(map[string]Match)
		}
		cache[root][typedSubstitutionKey(match.Subst, match.BaseSubst, names, g)] = match
	}
	return cache
}

func termChangeFrontier(g *EGraph, changes []Id, budget *matchWorkBudget) ([]Id, []Id, error) {
	frontier := make([]Id, 0, len(changes))
	seen := make(map[Id]bool)
	queue := append([]Id(nil), changes...)
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		if !budget.spend() {
			return nil, nil, errSaturationMatchWorkLimit
		}
		if !seen[id] {
			seen[id] = true
			frontier = append(frontier, id)
		}
		root := g.Find(id)
		if root != InvalidId && !seen[root] {
			queue = append(queue, root)
		}
		for _, parent := range g.parentIndex[id] {
			if !budget.spend() {
				return nil, nil, errSaturationMatchWorkLimit
			}
			queue = append(queue, parent)
		}
		if root != InvalidId && root != id {
			for _, parent := range g.parentIndex[root] {
				if !budget.spend() {
					return nil, nil, errSaturationMatchWorkLimit
				}
				queue = append(queue, parent)
			}
		}
	}
	var roots []Id
	rootSeen := make(map[Id]bool)
	for _, id := range frontier {
		root := g.Find(id)
		if root != InvalidId && g.store.Class(root) != nil && !rootSeen[root] {
			rootSeen[root] = true
			roots = append(roots, root)
		}
	}
	return frontier, roots, nil
}

func joinTermPatternMatches(g *EGraph, sets [][]Match, variables []map[string]bool, budget *matchWorkBudget) ([]Match, error) {
	if len(sets) == 0 || len(sets[0]) == 0 {
		return nil, nil
	}
	matches := sets[0]
	bound := make(map[string]bool)
	for name := range variables[0] {
		bound[name] = true
	}
	pending := make([]int, 0, len(sets)-1)
	for i := 1; i < len(sets); i++ {
		if len(sets[i]) == 0 {
			return nil, nil
		}
		pending = append(pending, i)
	}
	for len(pending) > 0 {
		selected := 0
		selectedShared := -1
		for i, candidate := range pending {
			shared := 0
			for name := range variables[candidate] {
				if bound[name] {
					shared++
				}
			}
			if shared > selectedShared || shared == selectedShared && len(sets[candidate]) < len(sets[pending[selected]]) {
				selected, selectedShared = i, shared
			}
		}
		source := pending[selected]
		pending = append(pending[:selected], pending[selected+1:]...)
		var complete bool
		var err error
		matches, complete, err = joinMatchSetsWithBudget(g, matches, sets[source], bound, variables[source], budget)
		if err != nil {
			return nil, err
		}
		if !complete {
			return nil, errSaturationMatchWorkLimit
		}
		for name := range variables[source] {
			bound[name] = true
		}
		if len(matches) == 0 {
			return nil, nil
		}
	}
	return matches, nil
}

func joinTermMatchDeltas(g *EGraph, previous, current []map[Id]map[string]Match, deltas [][]Match, variables []map[string]bool, budget *matchWorkBudget) ([]Match, error) {
	var results []Match
	for pivot := range deltas {
		if len(deltas[pivot]) == 0 {
			continue
		}
		joined := []Match{{ClassID: InvalidId, Subst: make(Subst), BaseSubst: make(map[string]FunctionValue)}}
		bound := make(map[string]bool)
		for source := range current {
			cache := current[source]
			if source < pivot {
				// Matches introduced in earlier sources belong to an earlier pivot.
				cache = previous[source]
			}
			set, flattenErr := flattenTermMatches(cache, budget)
			if flattenErr != nil {
				return nil, flattenErr
			}
			if source == pivot {
				set = deltas[source]
			}
			var complete bool
			var err error
			joined, complete, err = joinMatchSetsWithBudget(g, joined, set, bound, variables[source], budget)
			if err != nil {
				return nil, err
			}
			if !complete {
				return nil, errSaturationMatchWorkLimit
			}
			for name := range variables[source] {
				bound[name] = true
			}
			if len(joined) == 0 {
				break
			}
		}
		results = append(results, joined...)
	}
	allVars := make(map[string]bool)
	for _, set := range variables {
		for name := range set {
			allVars[name] = true
		}
	}
	names := sortedVariableNames(allVars)
	seen := make(map[string]bool)
	dedup := results[:0]
	for _, match := range results {
		if !budget.spend() {
			return nil, errSaturationMatchWorkLimit
		}
		classID := g.Find(match.ClassID)
		key := fmt.Sprintf("%d:%s", classID, typedSubstitutionKey(match.Subst, match.BaseSubst, names, g))
		if !seen[key] {
			seen[key] = true
			dedup = append(dedup, match)
		}
	}
	return dedup, nil
}

func flattenTermMatches(cache map[Id]map[string]Match, budget *matchWorkBudget) ([]Match, error) {
	var matches []Match
	for _, group := range cache {
		for _, match := range group {
			if !budget.spend() {
				return nil, errSaturationMatchWorkLimit
			}
			matches = append(matches, match)
		}
	}
	return matches, nil
}

// semiNaiveRelationMatches returns matches whose joined tuple was introduced
// by at least one relation source since the previous call. It is deliberately
// restricted to insertion-only rules; callers retain the full-search fallback
// for every other rewrite shape.
func (r *Rewrite) semiNaiveRelationMatches(g *EGraph, tracker *relationMatchTracker, budget *matchWorkBudget) ([]Match, int, error) {
	if semiNaiveCoreRelationMatchable(r, g) {
		return r.semiNaiveCoreRelationMatches(g, tracker, budget)
	}
	patterns := r.leftPatterns()
	previous := tracker.previous
	wasReady := tracker.ready
	current := make([][]Match, len(patterns))
	variables := make([]map[string]bool, len(patterns))
	keys := make([]map[string]struct{}, len(patterns))
	deltas := make([][]Match, len(patterns))
	cursors := make([]int, len(patterns))
	for i, pattern := range patterns {
		variables[i] = make(map[string]bool)
		collectVariables(pattern, variables[i])
		names := sortedVariableNames(variables[i])
		keys[i] = make(map[string]struct{})
		old := map[string]struct{}(nil)
		if tracker.ready && i < len(tracker.keys) {
			old = tracker.keys[i]
			for key := range old {
				if !budget.spend() {
					return nil, 0, errSaturationMatchWorkLimit
				}
				keys[i][key] = struct{}{}
			}
		}
		var matches []Match
		if !tracker.ready || i >= len(tracker.cursors) {
			var err error
			var complete bool
			matches, complete, err = pattern.searchChecked(g, budget)
			if err != nil {
				return nil, 0, err
			}
			if !complete {
				return nil, 0, errSaturationMatchWorkLimit
			}
			cursors[i] = len(g.relationInsertions[pattern.RelationName])
		} else {
			log := g.relationInsertions[pattern.RelationName]
			cursor := tracker.cursors[i]
			if cursor > len(log) {
				return nil, 0, fmt.Errorf("egraph: relation insertion log cursor %d exceeds %d rows for %q", cursor, len(log), pattern.RelationName)
			}
			for range log[cursor:] {
				if !budget.spend() {
					return nil, 0, errSaturationMatchWorkLimit
				}
			}
			rowKeys := append([]string{}, log[cursor:]...)
			var err error
			var complete bool
			matches, complete, err = pattern.searchRelationKeysWithBudget(g, names, rowKeys, budget)
			if err != nil {
				return nil, 0, err
			}
			if !complete {
				return nil, 0, errSaturationMatchWorkLimit
			}
			cursors[i] = len(log)
		}
		previousMatches := previousMatches(tracker, i)
		for range previousMatches {
			if !budget.spend() {
				return nil, 0, errSaturationMatchWorkLimit
			}
		}
		current[i] = append([]Match(nil), previousMatches...)
		for _, match := range matches {
			if !budget.spend() {
				return nil, 0, errSaturationMatchWorkLimit
			}
			key := typedSubstitutionKey(match.Subst, match.BaseSubst, names, g)
			if _, exists := old[key]; !exists {
				deltas[i] = append(deltas[i], match)
			}
			if _, exists := keys[i][key]; !exists {
				keys[i][key] = struct{}{}
				current[i] = append(current[i], match)
			}
		}
	}
	if len(patterns) == 0 {
		return nil, 0, nil
	}
	var results []Match
	for pivot := range patterns {
		if len(deltas[pivot]) == 0 {
			continue
		}
		joined := []Match{{ClassID: InvalidId, Subst: make(Subst), BaseSubst: make(map[string]FunctionValue)}}
		bound := make(map[string]bool)
		for source := range patterns {
			set := current[source]
			if source < pivot {
				if wasReady {
					set = previous[source]
				} else {
					set = nil
				}
			}
			if source == pivot {
				set = deltas[source]
			}
			var err error
			var complete bool
			joined, complete, err = joinMatchSetsWithBudget(g, joined, set, bound, variables[source], budget)
			if err != nil {
				return nil, 0, err
			}
			if !complete {
				return nil, 0, errSaturationMatchWorkLimit
			}
			for name := range variables[source] {
				bound[name] = true
			}
			if len(joined) == 0 {
				break
			}
		}
		for _, match := range joined {
			if !budget.spend() {
				return nil, 0, errSaturationMatchWorkLimit
			}
			results = append(results, match)
		}
	}
	allVars := make(map[string]bool)
	for _, set := range variables {
		for name := range set {
			allVars[name] = true
		}
	}
	names := sortedVariableNames(allVars)
	seen := make(map[string]struct{}, len(results))
	dedup := results[:0]
	for _, match := range results {
		if !budget.spend() {
			return nil, 0, errSaturationMatchWorkLimit
		}
		key := typedSubstitutionKey(match.Subst, match.BaseSubst, names, g)
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		dedup = append(dedup, match)
	}
	tracker.previous, tracker.keys, tracker.cursors, tracker.revision, tracker.ready = current, keys, cursors, g.revision, true
	return dedup, len(patterns), nil
}

// semiNaiveCoreRelationMatchable limits the shared relation join to patterns
// whose semantics are represented exactly by egraphcore terms. In particular,
// class-column structural patterns can match several nodes in one e-class and
// must continue through Apple's pattern matcher.
func semiNaiveCoreRelationMatchable(r *Rewrite, g *EGraph) bool {
	if r == nil || g == nil {
		return false
	}
	classVars := make(map[string]bool)
	primitiveVars := make(map[string]bool)
	for _, pattern := range r.leftPatterns() {
		if pattern == nil || pattern.RelationName == "" || pattern.FunctionName != "" {
			return false
		}
		columns := pattern.RelationColumns
		if len(columns) == 0 && len(pattern.Children) > 0 {
			columns = make([]FunctionColumnPattern, len(pattern.Children))
			for i, child := range pattern.Children {
				columns[i] = FunctionClassPattern(child)
			}
		}
		table := g.relations[pattern.RelationName]
		if table == nil || len(columns) != len(table.schema.Columns) {
			return false
		}
		for i, column := range columns {
			typ := table.schema.Columns[i]
			if column.Kind != 0 && column.Kind != typ.Kind || column.Expr != nil {
				return false
			}
			if typ.Kind == FunctionClassColumn {
				if column.Term != nil {
					if !column.Term.IsVar || column.Term.Sort != "" || len(column.Term.Children) != 0 || column.Term.RelationName != "" || column.Term.FunctionName != "" || column.Term.HasData || column.Term.Data != nil {
						return false
					}
					classVars[column.Term.Op] = true
				} else if !column.HasValue {
					return false
				} else if _, err := g.canonicalTypedValue(typ, column.Value, fmt.Sprintf("relation pattern %q column %d", pattern.RelationName, i)); err != nil {
					return false
				}
				continue
			}
			if column.Term != nil {
				return false
			}
			if column.HasValue {
				if _, err := g.canonicalTypedValue(typ, column.Value, fmt.Sprintf("relation pattern %q column %d", pattern.RelationName, i)); err != nil {
					return false
				}
			} else if column.Variable != "" {
				primitiveVars[patternVariableName(column.Variable)] = true
			} else {
				return false
			}
		}
	}
	for name := range classVars {
		if primitiveVars[name] {
			return false
		}
	}
	return true
}

func (r *Rewrite) semiNaiveCoreRelationMatches(g *EGraph, tracker *relationMatchTracker, budget *matchWorkBudget) ([]Match, int, error) {
	patterns := r.leftPatterns()
	atoms := make([]egraphcore.Atom[Id], len(patterns))
	for i, pattern := range patterns {
		table := g.relations[pattern.RelationName]
		columns := pattern.RelationColumns
		if len(columns) == 0 && len(pattern.Children) > 0 {
			columns = make([]FunctionColumnPattern, len(pattern.Children))
			for j, child := range pattern.Children {
				columns[j] = FunctionClassPattern(child)
			}
		}
		terms := make([]egraphcore.Term[Id], len(columns))
		for j, column := range columns {
			if table.schema.Columns[j].Kind == FunctionClassColumn {
				if column.Term != nil {
					terms[j] = egraphcore.Variable[Id](column.Term.Op)
				} else {
					value, _ := functionValueToCore(column.Value)
					terms[j] = egraphcore.Constant(value)
				}
			} else if column.HasValue {
				value, _ := functionValueToCore(column.Value)
				terms[j] = egraphcore.Constant(value)
			} else {
				terms[j] = egraphcore.Variable[Id](patternVariableName(column.Variable))
			}
		}
		atoms[i] = egraphcore.Atom[Id]{Relation: table.core, Terms: terms}
	}

	var bindings []egraphcore.Binding[Id]
	fullSearch := !tracker.ready || len(tracker.coreCursors) != len(atoms)
	if fullSearch {
		var complete bool
		var err error
		bindings, complete, err = egraphcore.Join(atoms, g.Find, coreWorkBudget(budget))
		if err != nil {
			return nil, 0, err
		}
		if !complete {
			return nil, 0, errSaturationMatchWorkLimit
		}
	} else {
		deltas := make([]egraphcore.DeltaAtom[Id], len(atoms))
		for i := range atoms {
			deltas[i] = egraphcore.DeltaAtom[Id]{Index: i, Cursor: tracker.coreCursors[i]}
		}
		var complete bool
		var err error
		bindings, complete, err = egraphcore.JoinDeltas(atoms, deltas, g.Find, coreWorkBudget(budget))
		if errors.Is(err, egraphcore.ErrRelationReset) {
			fullSearch = true
			bindings, complete, err = egraphcore.Join(atoms, g.Find, coreWorkBudget(budget))
		}
		if err != nil {
			return nil, 0, err
		}
		if !complete {
			return nil, 0, errSaturationMatchWorkLimit
		}
	}
	matches := make([]Match, len(bindings))
	for i, binding := range bindings {
		match := Match{ClassID: InvalidId, Subst: make(Subst), BaseSubst: make(map[string]FunctionValue)}
		for name, value := range binding {
			switch value.Kind {
			case egraphcore.ClassValue:
				match.Subst[name] = value.Class
			default:
				converted, ok := functionValueFromCore(value)
				if !ok {
					return nil, 0, fmt.Errorf("egraph: unsupported relation binding value kind %d", value.Kind)
				}
				match.BaseSubst[name] = converted
			}
		}
		matches[i] = match
	}
	tracker.coreCursors = make([]egraphcore.RelationCursor, len(atoms))
	for i, atom := range atoms {
		tracker.coreCursors[i] = atom.Relation.Snapshot()
	}
	tracker.ready = true
	tracker.revision = g.revision
	return matches, len(patterns), nil
}

func previousMatches(tracker *relationMatchTracker, source int) []Match {
	if tracker == nil || source < 0 || source >= len(tracker.previous) {
		return nil
	}
	return tracker.previous[source]
}

func sortedVariableNames(variables map[string]bool) []string {
	names := make([]string, 0, len(variables))
	for name := range variables {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func semiNaiveRelationEligible(r *Rewrite) bool {
	if r == nil || len(r.leftPatterns()) < 2 || r.Guard != nil || len(r.rightPatterns()) != 0 || len(r.unions) != 0 || len(r.facts) != 0 || len(r.functions) != 0 || len(r.nodeDeletes) != 0 || len(r.classConditions) != 0 || len(r.classPatternConditions) != 0 || len(r.primitiveConditions) != 0 || len(r.primitiveBindings) != 0 || r.subsumesLHS {
		return false
	}
	bound := make(map[string]bool)
	for _, pattern := range r.leftPatterns() {
		if pattern == nil || pattern.RelationName == "" || pattern.FunctionName != "" {
			return false
		}
		collectVariables(pattern, bound)
	}
	if len(r.relations) == 0 {
		return false
	}
	for _, action := range r.relations {
		if action.Kind != RelationInsert {
			return false
		}
		if action.Typed || action.TypedColumns != nil {
			if len(action.Columns) != 0 || len(action.TypedColumns) == 0 {
				return false
			}
			for _, column := range action.TypedColumns {
				if column.Expr != nil || column.HasValue {
					return false
				}
				if column.Kind == FunctionClassColumn {
					if column.Variable != "" || column.Term == nil || !column.Term.IsVar || !bound[column.Term.Op] {
						return false
					}
				} else if column.Term != nil || column.Variable == "" || !bound[column.Variable] {
					return false
				}
			}
			continue
		}
		if len(action.Columns) == 0 {
			return false
		}
		for _, column := range action.Columns {
			if column == nil || !column.IsVar || !bound[column.Op] {
				return false
			}
		}
	}
	return true
}

func substitutionKey(subst Subst, names []string, g *EGraph) string {
	var key strings.Builder
	for _, name := range names {
		id, ok := subst[name]
		if !ok {
			continue
		}
		fmt.Fprintf(&key, "%d:%s=%d;", len(name), name, g.Find(id))
	}
	return key.String()
}

func combinedSubstitutionKey(match Match, names []string, g *EGraph) string {
	return typedSubstitutionKey(match.Subst, match.BaseSubst, names, g)
}

// SaturationStopReason describes why equality saturation stopped.
type SaturationStopReason string

const (
	SaturationFixedPoint     SaturationStopReason = "fixed_point"
	SaturationIterationLimit SaturationStopReason = "iteration_limit"
	SaturationNodeLimit      SaturationStopReason = "node_limit"
	SaturationTupleLimit     SaturationStopReason = "tuple_limit"
	SaturationMatchWorkLimit SaturationStopReason = "match_work_limit"
	SaturationRunComplete    SaturationStopReason = "run_complete"
	SaturationScheduleDone   SaturationStopReason = "schedule_complete"
)

const defaultSaturationIterations = 256
const defaultSaturationNodes = 10_000_000
const defaultSaturationTuples = 10_000_000
const defaultSaturationMatchWork = 100_000_000

var errSaturationMatchWorkLimit = errors.New("egraph: saturation match work limit reached")

// SaturationOptions bounds equality saturation. A non-positive limit uses
// the package default. MaxTuples defaults to ten million and includes e-nodes,
// stored facts, and function rows, including rows marked subsumed. MaxMatchWork
// bounds search work for rewrites whose LHS
// patterns are term, relation, or function patterns, plus estimated traversal
// work for direct RHS e-node instantiation. It counts matcher state visits,
// table-row checks and normalization, sort work, intermediate match-join work,
// and class or primitive condition evaluation. Graph rebuilds, guards, union
// actions, relation or function actions, and node-deletion actions are not
// included.
type SaturationOptions struct {
	MaxIterations int `json:"max_iterations"`
	MaxNodes      int `json:"max_nodes"`
	MaxTuples     int `json:"max_tuples"`
	MaxMatchWork  int `json:"max_match_work"`
}

// ScheduleNodeKind identifies one operation in a composable rewrite schedule.
type ScheduleNodeKind uint8

const (
	ScheduleRunKind ScheduleNodeKind = iota + 1
	ScheduleSaturateKind
	ScheduleSequenceKind
	ScheduleRepeatKind
)

// RewriteSchedule describes an ordered rewrite program. Run applies each rule
// once; Saturate repeats its child schedule to a fixed point; Sequence runs
// children in order; Repeat executes its child up to a fixed limit, stopping
// early when an execution makes no changes.
type RewriteSchedule struct {
	Kind        ScheduleNodeKind
	Name        string
	Rules       []*Rewrite `json:"-"`
	Children    []*RewriteSchedule
	Repetitions int
}

// NewRunSchedule creates a single-pass rule node.
func NewRunSchedule(name string, rules ...*Rewrite) *RewriteSchedule {
	return &RewriteSchedule{Kind: ScheduleRunKind, Name: name, Rules: rules}
}

// NewSaturateSchedule creates a fixed-point rule node.
func NewSaturateSchedule(name string, rules ...*Rewrite) *RewriteSchedule {
	body := NewRunSchedule(name, rules...)
	return &RewriteSchedule{Kind: ScheduleSaturateKind, Name: name, Children: []*RewriteSchedule{body}}
}

// NewFixedPointSchedule creates a fixed-point node around any schedule node.
func NewFixedPointSchedule(name string, child *RewriteSchedule) *RewriteSchedule {
	return &RewriteSchedule{Kind: ScheduleSaturateKind, Name: name, Children: []*RewriteSchedule{child}}
}

// NewSequenceSchedule creates a schedule that executes children in order.
func NewSequenceSchedule(children ...*RewriteSchedule) *RewriteSchedule {
	return &RewriteSchedule{Kind: ScheduleSequenceKind, Children: children}
}

// NewRepeatSchedule creates a schedule that executes child count times.
func NewRepeatSchedule(count int, child *RewriteSchedule) *RewriteSchedule {
	return &RewriteSchedule{Kind: ScheduleRepeatKind, Repetitions: count, Children: []*RewriteSchedule{child}}
}

// SaturationPhase is a named rewrite stage. Each phase reaches a fixed point
// before the next phase begins. The scheduler repeats the ordered phase list
// until a full cycle produces no graph, analysis, relation, function, or structural-node changes.
type SaturationPhase struct {
	Name  string     `json:"name"`
	Rules []*Rewrite `json:"-"`
}

// SaturationPhaseResult reports the work performed by one schedule phase.
type SaturationPhaseResult struct {
	Name               string               `json:"name"`
	Iterations         int                  `json:"iterations"`
	Merges             int                  `json:"merges"`
	SubsumptionChanges int                  `json:"subsumption_changes"`
	AnalysisChanges    int                  `json:"analysis_changes"`
	RelationChanges    int                  `json:"relation_changes"`
	FunctionInserts    int                  `json:"function_inserts"`
	FunctionUpdates    int                  `json:"function_updates"`
	FunctionDeletes    int                  `json:"function_deletes"`
	NodeDeletions      int                  `json:"node_deletions"`
	NodesBefore        int                  `json:"nodes_before"`
	Classes            int                  `json:"classes"`
	Nodes              int                  `json:"nodes"`
	Tuples             int                  `json:"tuples"`
	NodeDelta          int                  `json:"node_delta"`
	DurationNS         int64                `json:"duration_ns"`
	Rebuilds           int                  `json:"rebuilds"`
	RebuildPasses      int                  `json:"rebuild_passes"`
	RebuildClasses     int                  `json:"rebuild_classes_visited"`
	RebuildNodes       int                  `json:"rebuild_nodes_visited"`
	RebuildCanon       int                  `json:"rebuild_canonicalizations"`
	RebuildMerges      int                  `json:"rebuild_class_merges"`
	RebuildTablePasses int                  `json:"rebuild_function_table_passes"`
	RebuildMergeRounds int                  `json:"rebuild_function_merge_rounds"`
	RebuildDurationNS  int64                `json:"rebuild_duration_ns"`
	MatchWork          int                  `json:"match_work"`
	StopReason         SaturationStopReason `json:"stop_reason"`
}

// SaturationResult summarizes one equality-saturation run.
type SaturationResult struct {
	Iterations         int                     `json:"iterations"`
	Merges             int                     `json:"merges"`
	SubsumptionChanges int                     `json:"subsumption_changes"`
	AnalysisChanges    int                     `json:"analysis_changes"`
	RelationChanges    int                     `json:"relation_changes"`
	FunctionInserts    int                     `json:"function_inserts"`
	FunctionUpdates    int                     `json:"function_updates"`
	FunctionDeletes    int                     `json:"function_deletes"`
	NodeDeletions      int                     `json:"node_deletions"`
	Classes            int                     `json:"classes"`
	Nodes              int                     `json:"nodes"`
	Tuples             int                     `json:"tuples"`
	MatchWork          int                     `json:"match_work"`
	Phases             []SaturationPhaseResult `json:"phases"`
	StopReason         SaturationStopReason    `json:"stop_reason"`
}

// SaturateWithReport repeatedly applies rewrite rules and reports whether it
// reached a fixed point or exhausted an iteration, node, tuple, or match-work
// limit.
func SaturateWithReport(g *EGraph, rules []*Rewrite, maxIterations int) (SaturationResult, error) {
	return SaturateScheduleWithReport(g, []SaturationPhase{{Name: "main", Rules: rules}}, maxIterations)
}

// SaturateChecked applies rules and returns an error unless the graph reaches a
// fixed point. Use this when extraction or execution requires a complete
// rewrite search space.
func SaturateChecked(g *EGraph, rules []*Rewrite, opts SaturationOptions) (SaturationResult, error) {
	result, err := SaturateSchedule(g, []SaturationPhase{{Name: "main", Rules: rules}}, opts)
	if err != nil {
		return result, err
	}
	if result.StopReason != SaturationFixedPoint && result.StopReason != SaturationScheduleDone {
		return result, fmt.Errorf("egraph: saturation stopped before fixed point: %s", result.StopReason)
	}
	return result, nil
}

// SaturateScheduleWithReport runs named rewrite phases in order. Each phase
// repeats its rules until they produce no merges, analysis, relation, function, or
// structural-node changes. maxIterations limits total rounds across all phases;
// non-positive values use the default of 256. The default node limit is 10
// million e-nodes. MaxMatchWork limits matching, table-row checks and
// normalization, sorting, joins, class or primitive condition evaluation, and
// estimated traversal of direct RHS e-node patterns for rewrites whose LHS
// patterns are terms, relations, or function tables. Graph rebuilds, guards,
// union actions, relation or function actions, and node-deletion actions are
// not charged to that budget. Each iteration sweep is applied to a trial graph
// and committed as one unit. If a later sweep fails or reaches a limit, earlier
// completed sweeps remain in g.
func SaturateScheduleWithReport(g *EGraph, phases []SaturationPhase, maxIterations int) (SaturationResult, error) {
	return SaturateSchedule(g, phases, SaturationOptions{MaxIterations: maxIterations})
}

// SaturateSchedule rebuilds the graph, then runs named rewrite phases in order
// until a full phase cycle produces no graph, analysis, relation, function, or
// structural-node changes or a limit is reached. The iteration limit counts
// rounds across all phases and cycles. Before applying a match, saturation
// reserves a conservative upper bound for its term instantiations. Hash
// consing may make the actual increase smaller than this bound. Each iteration
// sweep is committed only when all its rules succeed and remain within limits;
// earlier committed sweeps remain in g if a later sweep fails or stops.
func SaturateSchedule(g *EGraph, phases []SaturationPhase, opts SaturationOptions) (SaturationResult, error) {
	if g == nil {
		return SaturationResult{}, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.opIndex == nil {
		return SaturationResult{}, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	for phaseIndex, phase := range phases {
		for ruleIndex, rule := range phase.Rules {
			if err := rule.validate(); err != nil {
				return SaturationResult{}, fmt.Errorf("egraph: invalid rewrite in phase %d (%q), rule %d: %w", phaseIndex, phase.Name, ruleIndex, err)
			}
		}
	}
	if err := g.RebuildChecked(); err != nil {
		return SaturationResult{}, fmt.Errorf("egraph: rebuild before saturation: %w", err)
	}
	g.termChanges = nil
	g.termChangesEnabled = true
	defer func() {
		g.termChangesEnabled = false
		g.termChanges = nil
	}()
	// Relation insertion logs are private to this schedule. No operation before
	// the initial rebuild contributes to its source-side deltas.
	g.relationInsertions = make(map[string][]string)
	g.relationLogsEnabled = true
	defer func() {
		g.relationLogsEnabled = false
		g.relationInsertions = nil
	}()
	if opts.MaxIterations <= 0 {
		opts.MaxIterations = defaultSaturationIterations
	}
	if opts.MaxNodes <= 0 {
		opts.MaxNodes = defaultSaturationNodes
	}
	if opts.MaxTuples <= 0 {
		opts.MaxTuples = defaultSaturationTuples
	}
	if opts.MaxMatchWork <= 0 {
		opts.MaxMatchWork = defaultSaturationMatchWork
	}
	matchBudget := &matchWorkBudget{core: egraphcore.NewWorkBudget(opts.MaxMatchWork)}
	result := SaturationResult{StopReason: SaturationIterationLimit}
	result.Classes, result.Nodes = graphSize(g)
	result.Tuples = tupleCount(g)
	if result.Nodes > opts.MaxNodes {
		result.StopReason = SaturationNodeLimit
		return result, nil
	}
	if result.Tuples > opts.MaxTuples {
		result.StopReason = SaturationTupleLimit
		return result, nil
	}
	if len(phases) == 0 {
		result.StopReason = SaturationFixedPoint
		return result, nil
	}
	phaseResults := make([]SaturationPhaseResult, len(phases))
	var activePhaseResult *SaturationPhaseResult
	var activeRebuildEvents *[]RebuildStats
	priorRebuildObserver := g.rebuildObserver
	g.rebuildObserver = func(stats RebuildStats) {
		if activeRebuildEvents != nil {
			*activeRebuildEvents = append(*activeRebuildEvents, stats)
			return
		}
		if activePhaseResult != nil {
			activePhaseResult.Rebuilds++
			activePhaseResult.RebuildPasses += stats.Passes
			activePhaseResult.RebuildClasses += stats.ClassesVisited
			activePhaseResult.RebuildNodes += stats.NodesVisited
			activePhaseResult.RebuildCanon += stats.Canonicalizations
			activePhaseResult.RebuildMerges += stats.ClassMerges
			activePhaseResult.RebuildTablePasses += stats.FunctionTablePasses
			activePhaseResult.RebuildMergeRounds += stats.FunctionMergeRounds
			activePhaseResult.RebuildDurationNS += stats.Duration.Nanoseconds()
		}
		if priorRebuildObserver != nil {
			priorRebuildObserver(stats)
		}
	}
	defer func() { g.rebuildObserver = priorRebuildObserver }()
	for phaseIndex, phase := range phases {
		name := phase.Name
		if name == "" {
			name = fmt.Sprintf("phase-%d", phaseIndex+1)
		}
		phaseResults[phaseIndex].Name = name
	}
	phasesRun := 0
	// These snapshots live only for this SaturateSchedule invocation. Any
	// rule outside the insertion-only relation subset invalidates them.
	relationTrackers := make(map[*Rewrite]*relationMatchTracker)
	termTrackers := make(map[*Rewrite]*termMatchTracker)
	for result.Iterations < opts.MaxIterations {
		cycleChanges := 0
		stopped := false
		for phaseIndex, phase := range phases {
			phasesRun = phaseIndex + 1
			phaseStart := time.Now()
			phaseResult := &phaseResults[phaseIndex]
			activePhaseResult = phaseResult
			if phaseResult.Iterations == 0 {
				_, phaseResult.NodesBefore = graphSize(g)
			}
			phaseResult.StopReason = SaturationIterationLimit
			phaseFixedPoint := false
			for result.Iterations < opts.MaxIterations {
				result.Iterations++
				phaseResult.Iterations++
				// Rules in a sweep share this trial graph, preserving the legacy
				// sequential behavior while making the sweep atomic at its limit.
				trial := g.cloneForScheduleRun()
				trialTermTrackers := cloneTermMatchTrackers(termTrackers)
				trialRelationTrackers := cloneRelationMatchTrackers(relationTrackers)
				var rebuildEvents []RebuildStats
				activeRebuildEvents = &rebuildEvents
				iterationChanges := 0
				var (
					mergesTotal, analysisTotal, relationTotal int
					insertsTotal, updatesTotal, deletesTotal  int
					nodeDeletionsTotal, subsumptionsTotal     int
					iterationStop                             SaturationStopReason
					iterationErr                              error
					failingRule                               *Rewrite
				)
				for _, rule := range phase.Rules {
					var (
						merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges int
						overNodeLimit                                                                                                                  bool
						matchWorkLimit                                                                                                                 bool
						err                                                                                                                            error
					)
					matchWorkBefore := matchBudget.core.Used()
					if semiNaiveTermEligible(rule) {
						tracker := trialTermTrackers[rule]
						if tracker == nil {
							tracker = &termMatchTracker{}
							trialTermTrackers[rule] = tracker
						}
						matches, searchErr := rule.semiNaiveTermMatches(trial, tracker, matchBudget)
						if searchErr != nil {
							err = fmt.Errorf("egraph: search rewrite %q: %w", rule.Name, searchErr)
						} else {
							beforeRevision := trial.revision
							merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, overNodeLimit, err = rule.applyCheckedMatchesWithBudget(trial, opts.MaxNodes, matches, true, matchBudget)
							if trial.revision != beforeRevision {
								clear(trialRelationTrackers)
							} else {
								for _, current := range trialRelationTrackers {
									if current.ready {
										current.revision = trial.revision
									}
								}
							}
						}
					} else if semiNaiveRelationEligible(rule) {
						tracker := trialRelationTrackers[rule]
						if tracker == nil {
							tracker = &relationMatchTracker{}
							trialRelationTrackers[rule] = tracker
						}
						matches, _, searchErr := rule.semiNaiveRelationMatches(trial, tracker, matchBudget)
						if searchErr != nil {
							err = fmt.Errorf("egraph: search rewrite %q: %w", rule.Name, searchErr)
						} else {
							beforeRevision := trial.revision
							merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, overNodeLimit, err = rule.applyCheckedMatchesWithBudget(trial, opts.MaxNodes, matches, true, matchBudget)
							if trial.revision-beforeRevision != uint64(relationChanges) {
								clear(trialRelationTrackers)
							} else {
								for _, current := range trialRelationTrackers {
									if current.ready {
										current.revision = trial.revision
									}
								}
							}
						}
					} else {
						clear(trialRelationTrackers)
						merges, analysisChanges, relationChanges, functionInserts, functionUpdates, functionDeletes, nodeDeletions, subsumptionChanges, overNodeLimit, err = rule.applyCheckedMatchesWithBudget(trial, opts.MaxNodes, nil, false, matchBudget)
					}
					work := matchBudget.core.Used() - matchWorkBefore
					result.MatchWork += work
					phaseResult.MatchWork += work
					if errors.Is(err, errSaturationMatchWorkLimit) {
						matchWorkLimit = true
						err = nil
					}
					if err != nil {
						iterationErr, failingRule = err, rule
						break
					}
					iterationChanges += merges + subsumptionChanges + analysisChanges + relationChanges + functionInserts + functionUpdates + functionDeletes + nodeDeletions
					mergesTotal += merges
					subsumptionsTotal += subsumptionChanges
					analysisTotal += analysisChanges
					relationTotal += relationChanges
					insertsTotal += functionInserts
					updatesTotal += functionUpdates
					deletesTotal += functionDeletes
					nodeDeletionsTotal += nodeDeletions
					if matchWorkLimit {
						iterationStop = SaturationMatchWorkLimit
						break
					}
					if overNodeLimit {
						iterationStop = SaturationNodeLimit
						break
					}
				}
				activeRebuildEvents = nil
				if iterationErr != nil {
					phaseResult.Classes, phaseResult.Nodes = graphSize(g)
					phaseResult.NodeDelta = phaseResult.Nodes - phaseResult.NodesBefore
					phaseResult.DurationNS += time.Since(phaseStart).Nanoseconds()
					result.Classes, result.Nodes = graphSize(g)
					result.Phases = phaseResults[:phasesRun]
					if failingRule != nil {
						return result, fmt.Errorf("egraph: apply rewrite %q in phase %q: %w", failingRule.Name, phaseResult.Name, iterationErr)
					}
					return result, fmt.Errorf("egraph: apply rewrite in phase %q: %w", phaseResult.Name, iterationErr)
				}
				if iterationStop == "" {
					_, trialNodes := graphSize(trial)
					if trialNodes > opts.MaxNodes {
						iterationStop = SaturationNodeLimit
					}
				}
				if iterationStop == "" && tupleCount(trial) > opts.MaxTuples {
					iterationStop = SaturationTupleLimit
				}
				if iterationStop == "" {
					*g = *trial
					termTrackers = trialTermTrackers
					relationTrackers = trialRelationTrackers
					result.Merges += mergesTotal
					result.SubsumptionChanges += subsumptionsTotal
					result.AnalysisChanges += analysisTotal
					result.RelationChanges += relationTotal
					result.FunctionInserts += insertsTotal
					result.FunctionUpdates += updatesTotal
					result.FunctionDeletes += deletesTotal
					result.NodeDeletions += nodeDeletionsTotal
					phaseResult.Merges += mergesTotal
					phaseResult.SubsumptionChanges += subsumptionsTotal
					phaseResult.AnalysisChanges += analysisTotal
					phaseResult.RelationChanges += relationTotal
					phaseResult.FunctionInserts += insertsTotal
					phaseResult.FunctionUpdates += updatesTotal
					phaseResult.FunctionDeletes += deletesTotal
					phaseResult.NodeDeletions += nodeDeletionsTotal
					for _, stats := range rebuildEvents {
						if g.rebuildObserver != nil {
							g.rebuildObserver(stats)
						}
					}
				} else {
					phaseResult.StopReason = iterationStop
				}
				phaseResult.Classes, phaseResult.Nodes = graphSize(g)
				phaseResult.Tuples = tupleCount(g)
				if iterationStop != "" {
					break
				}
				iterationChanges = mergesTotal + subsumptionsTotal + analysisTotal + relationTotal + insertsTotal + updatesTotal + deletesTotal + nodeDeletionsTotal
				if iterationChanges == 0 {
					phaseResult.StopReason = SaturationFixedPoint
					phaseFixedPoint = true
					break
				}
				cycleChanges += iterationChanges
			}
			phaseResult.Classes, phaseResult.Nodes = graphSize(g)
			phaseResult.NodeDelta = phaseResult.Nodes - phaseResult.NodesBefore
			phaseResult.DurationNS += time.Since(phaseStart).Nanoseconds()
			if phaseResult.StopReason == SaturationNodeLimit {
				result.StopReason = SaturationNodeLimit
				stopped = true
				break
			}
			if phaseResult.StopReason == SaturationMatchWorkLimit {
				result.StopReason = SaturationMatchWorkLimit
				stopped = true
				break
			}
			if phaseResult.StopReason == SaturationTupleLimit {
				result.StopReason = SaturationTupleLimit
				stopped = true
				break
			}
			if !phaseFixedPoint {
				result.StopReason = SaturationIterationLimit
				stopped = true
				break
			}
		}
		if stopped {
			break
		}
		if cycleChanges == 0 {
			result.StopReason = SaturationFixedPoint
			break
		}
		if result.Iterations >= opts.MaxIterations {
			result.StopReason = SaturationIterationLimit
			break
		}
	}
	result.Phases = phaseResults[:phasesRun]
	result.Classes, result.Nodes = graphSize(g)
	result.Tuples = tupleCount(g)
	return result, nil
}

// ExecuteSchedule evaluates a composable rewrite schedule. MaxIterations
// counts one pass over a Run or Saturate rule list. MaxNodes is checked before
// execution and after each applied match, like SaturateSchedule. MaxMatchWork
// bounds matching, table-row checks and normalization, sorting, joins, and
// class or primitive condition evaluation for rewrites whose LHS patterns are
// terms, relations, or function tables. Graph rebuilds, guards, and RHS
// actions are not charged.
func ExecuteSchedule(g *EGraph, schedule *RewriteSchedule, opts SaturationOptions) (SaturationResult, error) {
	if g == nil {
		return SaturationResult{}, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.opIndex == nil {
		return SaturationResult{}, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	coreSchedule, err := compileRewriteSchedule(schedule)
	if err != nil {
		return SaturationResult{}, fmt.Errorf("egraph: invalid schedule: %w", err)
	}
	if err := egraphcore.ValidateSchedule(coreSchedule, func(rule *Rewrite) error { return rule.validate() }); err != nil {
		return SaturationResult{}, fmt.Errorf("egraph: invalid schedule: %w", err)
	}
	if err := g.RebuildChecked(); err != nil {
		return SaturationResult{}, fmt.Errorf("egraph: rebuild before schedule: %w", err)
	}
	g.termChanges = nil
	g.termChangesEnabled = true
	defer func() {
		g.termChangesEnabled = false
		g.termChanges = nil
	}()
	g.relationInsertions = make(map[string][]string)
	g.relationLogsEnabled = true
	defer func() {
		g.relationLogsEnabled = false
		g.relationInsertions = nil
	}()
	if opts.MaxIterations <= 0 {
		opts.MaxIterations = defaultSaturationIterations
	}
	if opts.MaxNodes <= 0 {
		opts.MaxNodes = defaultSaturationNodes
	}
	if opts.MaxTuples <= 0 {
		opts.MaxTuples = defaultSaturationTuples
	}
	if opts.MaxMatchWork <= 0 {
		opts.MaxMatchWork = defaultSaturationMatchWork
	}
	matchBudget := &matchWorkBudget{core: egraphcore.NewWorkBudget(opts.MaxMatchWork)}
	result := SaturationResult{StopReason: SaturationScheduleDone}
	result.Classes, result.Nodes = graphSize(g)
	result.Tuples = tupleCount(g)
	if result.Nodes > opts.MaxNodes {
		result.StopReason = SaturationNodeLimit
		return result, nil
	}
	if result.Tuples > opts.MaxTuples {
		result.StopReason = SaturationTupleLimit
		return result, nil
	}
	termTrackers := make(map[*Rewrite]*termMatchTracker)
	relationTrackers := make(map[*Rewrite]*relationMatchTracker)
	scheduleResult, err := egraphcore.ExecuteSchedule(coreSchedule, opts.MaxIterations, func(rule *Rewrite) error {
		return rule.validate()
	}, func(name string, rules []*Rewrite) (int, bool, error) {
		pass := g.cloneForScheduleRun()
		observer := pass.rebuildObserver
		var rebuildEvents []RebuildStats
		if observer != nil {
			pass.rebuildObserver = func(stats RebuildStats) {
				rebuildEvents = append(rebuildEvents, stats)
			}
		}
		passResult := result
		passResult.Phases = append([]SaturationPhaseResult(nil), result.Phases...)
		passTerms := cloneTermMatchTrackers(termTrackers)
		passRelations := cloneRelationMatchTrackers(relationTrackers)
		changed, err := executeRulePass(pass, rules, name, opts, matchBudget, passTerms, passRelations, &passResult)
		if err == nil {
			passResult.Tuples = tupleCount(pass)
			if n := len(passResult.Phases); n > 0 {
				passResult.Phases[n-1].Tuples = passResult.Tuples
			}
		}
		if err == nil && passResult.Tuples > opts.MaxTuples {
			passResult.StopReason = SaturationTupleLimit
			if n := len(passResult.Phases); n > 0 {
				passResult.Phases[n-1].StopReason = SaturationTupleLimit
			}
		}
		if err != nil {
			result.Iterations = passResult.Iterations
			result.MatchWork = passResult.MatchWork
			if len(passResult.Phases) > len(result.Phases) {
				result.Phases = append(result.Phases, passResult.Phases[len(result.Phases):]...)
			}
			return 0, true, err
		}
		if passResult.StopReason == SaturationNodeLimit || passResult.StopReason == SaturationMatchWorkLimit || passResult.StopReason == SaturationTupleLimit {
			// Keep the consumed work in the report, but discard every graph and
			// rewrite-result mutation made by this run group.
			result.Iterations = passResult.Iterations
			result.MatchWork = passResult.MatchWork
			if len(passResult.Phases) > len(result.Phases) {
				result.Phases = append(result.Phases, passResult.Phases[len(result.Phases):]...)
			}
			result.StopReason = passResult.StopReason
			result.Classes, result.Nodes = graphSize(g)
			result.Tuples = tupleCount(g)
			return 0, true, nil
		}
		pass.rebuildObserver = observer
		*g = *pass
		termTrackers = passTerms
		relationTrackers = passRelations
		result = passResult
		for _, stats := range rebuildEvents {
			observer(stats)
		}
		return boolInt(changed), false, nil
	})
	if err != nil {
		result.Classes, result.Nodes = graphSize(g)
		result.Tuples = tupleCount(g)
		return result, err
	}
	if scheduleResult.RunLimitReached {
		result.StopReason = SaturationIterationLimit
	}
	if result.StopReason != SaturationIterationLimit && result.StopReason != SaturationNodeLimit && result.StopReason != SaturationMatchWorkLimit && result.StopReason != SaturationTupleLimit {
		result.StopReason = SaturationScheduleDone
	}
	result.Classes, result.Nodes = graphSize(g)
	result.Tuples = tupleCount(g)
	return result, nil
}

func validateSchedule(schedule *RewriteSchedule) error {
	coreSchedule, err := compileRewriteSchedule(schedule)
	if err != nil {
		return err
	}
	return egraphcore.ValidateSchedule(coreSchedule, func(rule *Rewrite) error { return rule.validate() })
}

// compileRewriteSchedule adapts the public Metal schedule value to the shared
// schedule engine. Core execution names are intentionally retained in phase
// metrics so every backend reports the same hierarchical schedule path.
func compileRewriteSchedule(schedule *RewriteSchedule) (*egraphcore.Schedule[*Rewrite], error) {
	state := make(map[*RewriteSchedule]uint8)
	var compile func(*RewriteSchedule) (*egraphcore.Schedule[*Rewrite], error)
	compile = func(node *RewriteSchedule) (*egraphcore.Schedule[*Rewrite], error) {
		if node == nil {
			return nil, fmt.Errorf("nil schedule node")
		}
		if state[node] == 1 {
			return nil, fmt.Errorf("cyclic schedule")
		}
		state[node] = 1
		defer func() { state[node] = 2 }()
		switch node.Kind {
		case ScheduleRunKind:
			if len(node.Children) != 0 || node.Repetitions != 0 {
				return nil, fmt.Errorf("rule schedule %q has children or repetitions", node.Name)
			}
			name := node.Name
			if name == "" {
				name = stringScheduleKind(node.Kind)
			}
			return egraphcore.Run(name, node.Rules...), nil
		case ScheduleSaturateKind:
			if len(node.Rules) != 0 || len(node.Children) != 1 || node.Repetitions != 0 {
				return nil, fmt.Errorf("saturate requires one child schedule")
			}
			child, err := compile(node.Children[0])
			if err != nil {
				return nil, err
			}
			return egraphcore.SaturateNamed(node.Name, child), nil
		case ScheduleSequenceKind:
			if len(node.Rules) != 0 || node.Repetitions != 0 {
				return nil, fmt.Errorf("sequence has rules or repetitions")
			}
			children := make([]*egraphcore.Schedule[*Rewrite], len(node.Children))
			for i, child := range node.Children {
				compiled, err := compile(child)
				if err != nil {
					return nil, err
				}
				children[i] = compiled
			}
			return egraphcore.SequenceNamed(node.Name, children...), nil
		case ScheduleRepeatKind:
			if len(node.Rules) != 0 || len(node.Children) != 1 || node.Repetitions < 0 {
				return nil, fmt.Errorf("repeat requires one child and a non-negative repetition limit")
			}
			child, err := compile(node.Children[0])
			if err != nil {
				return nil, err
			}
			return egraphcore.RepeatNamed(node.Name, node.Repetitions, child), nil
		default:
			return nil, fmt.Errorf("unknown schedule kind %d", node.Kind)
		}
	}
	coreSchedule, err := compile(schedule)
	if err != nil {
		return nil, err
	}
	return coreSchedule, nil
}

func boolInt(changed bool) int {
	if changed {
		return 1
	}
	return 0
}

func scheduleChangeCount(result *SaturationResult) int {
	return result.Merges + result.SubsumptionChanges + result.AnalysisChanges + result.RelationChanges + result.FunctionInserts + result.FunctionUpdates + result.FunctionDeletes + result.NodeDeletions
}

type capturedRewriteMatches struct {
	rule    *Rewrite
	matches []Match
	mode    uint8
}

func splitRewriteDeletes(rule *Rewrite) (deletions, remaining *Rewrite) {
	// Function deletes are staged with sets and subsumptions. The function
	// transaction captures their old tuples and applies those removals before
	// its ordered insert buffer.
	deletedFunctions := []FunctionAction(nil)
	otherFunctions := rule.functions
	deletedRelations := make([]RelationAction, 0, len(rule.relations))
	otherRelations := make([]RelationAction, 0, len(rule.relations))
	for _, action := range rule.relations {
		switch action.Kind {
		case RelationDelete:
			deletedRelations = append(deletedRelations, action)
		default:
			otherRelations = append(otherRelations, action)
		}
	}
	if len(deletedFunctions)+len(deletedRelations) == 0 {
		return nil, rule
	}
	makeDeleteRule := func(functions []FunctionAction, relations []RelationAction) *Rewrite {
		cleanupRule := *rule
		cleanupRule.RHS = nil
		cleanupRule.actions = nil
		cleanupRule.unions = nil
		cleanupRule.facts = nil
		cleanupRule.relations = relations
		cleanupRule.functions = functions
		cleanupRule.nodeDeletes = nil
		cleanupRule.subsumesLHS = false
		return &cleanupRule
	}
	deletions = makeDeleteRule(deletedFunctions, deletedRelations)
	remainingRule := *rule
	remainingRule.relations = otherRelations
	remainingRule.functions = otherFunctions
	if remainingRule.RHS == nil && len(remainingRule.actions)+len(remainingRule.unions)+len(remainingRule.facts)+len(remainingRule.relations)+len(remainingRule.functions)+len(remainingRule.nodeDeletes) == 0 && !remainingRule.subsumesLHS {
		return deletions, nil
	}
	return deletions, &remainingRule
}

func executeRulePass(g *EGraph, rules []*Rewrite, name string, opts SaturationOptions, matchBudget *matchWorkBudget, termTrackers map[*Rewrite]*termMatchTracker, relationTrackers map[*Rewrite]*relationMatchTracker, result *SaturationResult) (bool, error) {
	start := time.Now()
	phase := SaturationPhaseResult{Name: name, Iterations: 1, StopReason: SaturationRunComplete}
	_, phase.NodesBefore = graphSize(g)
	result.Iterations++
	originalTermTrackers := termTrackers
	originalRelationTrackers := relationTrackers
	stagedTermTrackers := make(map[*Rewrite]*termMatchTracker, len(termTrackers))
	for rule, tracker := range termTrackers {
		if tracker == nil {
			continue
		}
		copyTracker := *tracker
		stagedTermTrackers[rule] = &copyTracker
	}
	stagedRelationTrackers := make(map[*Rewrite]*relationMatchTracker, len(relationTrackers))
	for rule, tracker := range relationTrackers {
		if tracker == nil {
			continue
		}
		copyTracker := *tracker
		stagedRelationTrackers[rule] = &copyTracker
	}
	termTrackers = stagedTermTrackers
	relationTrackers = stagedRelationTrackers
	captured := make([]capturedRewriteMatches, 0, len(rules))
	var passView *functionMergeView
	for _, rule := range rules {
		if rule.hasFunctionActionLookup() || rule.hasFilterFunctionActionLookup() {
			view := newFunctionMergeView(g)
			passView = &view
			break
		}
	}
	for _, rule := range rules {
		workBefore := matchBudget.core.Used()
		entry := capturedRewriteMatches{rule: rule}
		var err error
		if err = rule.validate(); err != nil {
			phase.Classes, phase.Nodes = graphSize(g)
			phase.NodeDelta = phase.Nodes - phase.NodesBefore
			phase.DurationNS = time.Since(start).Nanoseconds()
			result.Phases = append(result.Phases, phase)
			return false, fmt.Errorf("egraph: validate rewrite %q in schedule %q: %w", rule.Name, name, err)
		}
		if err = rule.validateFunctionActionExpressions(g); err != nil {
			phase.Classes, phase.Nodes = graphSize(g)
			phase.NodeDelta = phase.Nodes - phase.NodesBefore
			phase.DurationNS = time.Since(start).Nanoseconds()
			result.Phases = append(result.Phases, phase)
			return false, fmt.Errorf("egraph: validate rewrite %q in schedule %q: %w", rule.Name, name, err)
		}
		if semiNaiveTermEligible(rule) {
			entry.mode = 1
			tracker := termTrackers[rule]
			if tracker == nil {
				tracker = &termMatchTracker{}
				termTrackers[rule] = tracker
			}
			entry.matches, err = rule.semiNaiveTermMatches(g, tracker, matchBudget)
		} else if semiNaiveRelationEligible(rule) {
			entry.mode = 2
			tracker := relationTrackers[rule]
			if tracker == nil {
				tracker = &relationMatchTracker{}
				relationTrackers[rule] = tracker
			}
			entry.matches, _, err = rule.semiNaiveRelationMatches(g, tracker, matchBudget)
		} else {
			var complete bool
			entry.matches, complete, err = rule.searchMatchesWithBudget(g, matchBudget)
			if err == nil && !complete {
				err = errSaturationMatchWorkLimit
			}
		}
		if err == nil {
			entry.matches, err = rule.prepareQueryMatches(g, entry.matches, matchBudget, passView)
		}
		work := matchBudget.core.Used() - workBefore
		result.MatchWork += work
		phase.MatchWork += work
		if err != nil {
			if errors.Is(err, errSaturationMatchWorkLimit) {
				phase.StopReason = SaturationMatchWorkLimit
				break
			}
			phase.Classes, phase.Nodes = graphSize(g)
			phase.NodeDelta = phase.Nodes - phase.NodesBefore
			phase.DurationNS = time.Since(start).Nanoseconds()
			result.Phases = append(result.Phases, phase)
			return false, fmt.Errorf("egraph: search rewrite %q in schedule %q: %w", rule.Name, name, err)
		}
		captured = append(captured, entry)
	}
	if phase.StopReason == SaturationMatchWorkLimit {
		phase.Classes, phase.Nodes = graphSize(g)
		phase.NodeDelta = phase.Nodes - phase.NodesBefore
		phase.DurationNS = time.Since(start).Nanoseconds()
		result.Phases = append(result.Phases, phase)
		result.StopReason = phase.StopReason
		return false, nil
	}
	if passView != nil {
		view := cloneFunctionMergeView(*passView)
		if err := prepareCapturedActionLookups(g, captured, &view); err != nil {
			phase.Classes, phase.Nodes = graphSize(g)
			phase.NodeDelta = phase.Nodes - phase.NodesBefore
			phase.DurationNS = time.Since(start).Nanoseconds()
			result.Phases = append(result.Phases, phase)
			return false, fmt.Errorf("egraph: prepare action lookups in schedule %q: %w", name, err)
		}
		passView = &view
	}
	queryFunctionInserts := 0
	// Reserve every captured action before committing any of them. Otherwise a
	// node limit can stop one wave before the remaining actions run, leaving the
	// pass in a state Egglog's buffered rule set cannot produce.
	nodesAvailable := uint64(0)
	_, nodesBeforeApply := graphSize(g)
	if opts.MaxNodes > 0 && nodesBeforeApply <= opts.MaxNodes {
		nodesAvailable = uint64(opts.MaxNodes - nodesBeforeApply)
	}
	reservedNodes := uint64(0)
	if passView != nil {
		reservedNodes = uint64(len(passView.fresh.nodes))
	}
	reservedNodes += rewriteFreshDefaultNodeUpperBound(g, captured, passView)
	if opts.MaxNodes > 0 && reservedNodes > nodesAvailable {
		reservedNodes = nodesAvailable + 1
	}
	if opts.MaxNodes > 0 && reservedNodes <= nodesAvailable {
		for _, entry := range captured {
			perMatch := rewriteActionTermNodeUpperBound(entry.rule)
			if perMatch == 0 {
				continue
			}
			remaining := nodesAvailable - reservedNodes
			if uint64(len(entry.matches)) > remaining/perMatch {
				reservedNodes = nodesAvailable + 1
				break
			}
			reservedNodes += perMatch * uint64(len(entry.matches))
		}
	}
	if opts.MaxNodes > 0 && (nodesBeforeApply > opts.MaxNodes || reservedNodes > nodesAvailable) {
		phase.StopReason = SaturationNodeLimit
		phase.Classes, phase.Nodes = graphSize(g)
		phase.NodeDelta = phase.Nodes - phase.NodesBefore
		phase.DurationNS = time.Since(start).Nanoseconds()
		result.Phases = append(result.Phases, phase)
		result.StopReason = phase.StopReason
		return false, nil
	}
	functionBatch := newFunctionActionBatch(g)
	if passView != nil {
		functionBatch.view = cloneFunctionMergeView(*passView)
		for _, staged := range passView.tables {
			queryFunctionInserts += int(staged.defaultsInserted)
		}
	}
	for _, entry := range captured {
		if err := entry.rule.stageCapturedFunctionActions(g, entry.matches, functionBatch); err != nil {
			phase.Classes, phase.Nodes = graphSize(g)
			phase.NodeDelta = phase.Nodes - phase.NodesBefore
			phase.DurationNS = time.Since(start).Nanoseconds()
			result.Phases = append(result.Phases, phase)
			return false, fmt.Errorf("egraph: prepare function actions for rewrite %q in schedule %q: %w", entry.rule.Name, name, err)
		}
	}
	if err := functionBatch.commitDeletes(g); err != nil {
		phase.Classes, phase.Nodes = graphSize(g)
		phase.NodeDelta = phase.Nodes - phase.NodesBefore
		phase.DurationNS = time.Since(start).Nanoseconds()
		result.Phases = append(result.Phases, phase)
		return false, fmt.Errorf("egraph: apply function deletes in schedule %q: %w", name, err)
	}
	for rule, tracker := range stagedTermTrackers {
		originalTermTrackers[rule] = tracker
	}
	for rule, tracker := range stagedRelationTrackers {
		originalRelationTrackers[rule] = tracker
	}
	termTrackers = originalTermTrackers
	relationTrackers = originalRelationTrackers
	phase.FunctionInserts += queryFunctionInserts
	result.FunctionInserts += queryFunctionInserts
	changed := queryFunctionInserts != 0
	applyEntry := func(entry capturedRewriteMatches) (bool, error) {
		rule := entry.rule
		workBefore := matchBudget.core.Used()
		var (
			merges, analysis, relations, inserts, updates, deletes, nodeDeletions, subsumptions int
			nodeLimit                                                                           bool
			err                                                                                 error
		)
		if entry.mode == 0 {
			clear(relationTrackers)
		}
		beforeRevision := g.revision
		merges, analysis, relations, inserts, updates, deletes, nodeDeletions, subsumptions, nodeLimit, err = rule.applyCapturedMatchesWithFunctionBatch(g, opts.MaxNodes, entry.matches, matchBudget, functionBatch)
		if entry.mode == 1 {
			if g.revision != beforeRevision {
				clear(relationTrackers)
			} else {
				for _, current := range relationTrackers {
					if current.ready {
						current.revision = g.revision
					}
				}
			}
		} else if entry.mode == 2 {
			if g.revision-beforeRevision != uint64(relations) {
				clear(relationTrackers)
			} else {
				for _, current := range relationTrackers {
					if current.ready {
						current.revision = g.revision
					}
				}
			}
		}
		work := matchBudget.core.Used() - workBefore
		result.MatchWork += work
		phase.MatchWork += work
		matchLimit := errors.Is(err, errSaturationMatchWorkLimit)
		if matchLimit {
			err = nil
		}
		phase.Merges += merges
		phase.SubsumptionChanges += subsumptions
		phase.AnalysisChanges += analysis
		phase.RelationChanges += relations
		phase.FunctionInserts += inserts
		phase.FunctionUpdates += updates
		phase.FunctionDeletes += deletes
		phase.NodeDeletions += nodeDeletions
		result.Merges += merges
		result.SubsumptionChanges += subsumptions
		result.AnalysisChanges += analysis
		result.RelationChanges += relations
		result.FunctionInserts += inserts
		result.FunctionUpdates += updates
		result.FunctionDeletes += deletes
		result.NodeDeletions += nodeDeletions
		changed = changed || merges+subsumptions+analysis+relations+inserts+updates+deletes+nodeDeletions != 0
		if err != nil {
			return false, fmt.Errorf("egraph: apply rewrite %q in schedule %q: %w", rule.Name, name, err)
		}
		if nodeLimit {
			phase.StopReason = SaturationNodeLimit
			return true, nil
		}
		if matchLimit {
			phase.StopReason = SaturationMatchWorkLimit
			return true, nil
		}
		_, nodes := graphSize(g)
		if nodes > opts.MaxNodes {
			phase.StopReason = SaturationNodeLimit
			return true, nil
		}
		return false, nil
	}
	actions := make([]capturedRewriteMatches, 0, len(captured))
	deletions := make([]capturedRewriteMatches, 0, len(captured))
	for _, entry := range captured {
		deleteRule, remainingRule := splitRewriteDeletes(entry.rule)
		if deleteRule != nil {
			deletions = append(deletions, capturedRewriteMatches{rule: deleteRule, matches: entry.matches})
		}
		if remainingRule != nil {
			entry.rule = remainingRule
			actions = append(actions, entry)
		}
	}
	// Egglog buffers deletes separately and applies them before inserts. Keep
	// every non-delete action in rule order because function merge policies can
	// distinguish the order of writes and subsumptions.
	for _, entries := range [][]capturedRewriteMatches{deletions, actions} {
		for _, entry := range entries {
			stop, err := applyEntry(entry)
			if err != nil {
				phase.Classes, phase.Nodes = graphSize(g)
				phase.NodeDelta = phase.Nodes - phase.NodesBefore
				phase.DurationNS = time.Since(start).Nanoseconds()
				result.Phases = append(result.Phases, phase)
				return changed, err
			}
			if stop {
				break
			}
		}
		if phase.StopReason == SaturationNodeLimit || phase.StopReason == SaturationMatchWorkLimit {
			break
		}
	}
	inserts, updates, deletes, subsumptions, commitErr := functionBatch.commit(g)
	if commitErr != nil {
		phase.Classes, phase.Nodes = graphSize(g)
		phase.NodeDelta = phase.Nodes - phase.NodesBefore
		phase.DurationNS = time.Since(start).Nanoseconds()
		result.Phases = append(result.Phases, phase)
		return changed, fmt.Errorf("egraph: commit function actions in schedule %q: %w", name, commitErr)
	}
	phase.FunctionInserts += inserts
	phase.FunctionUpdates += updates
	phase.FunctionDeletes += deletes
	phase.SubsumptionChanges += subsumptions
	result.FunctionInserts += inserts
	result.FunctionUpdates += updates
	result.FunctionDeletes += deletes
	result.SubsumptionChanges += subsumptions
	changed = changed || inserts+updates+deletes+subsumptions != 0
	phase.Classes, phase.Nodes = graphSize(g)
	phase.Tuples = tupleCount(g)
	phase.NodeDelta = phase.Nodes - phase.NodesBefore
	phase.DurationNS = time.Since(start).Nanoseconds()
	result.Phases = append(result.Phases, phase)
	if phase.StopReason == SaturationNodeLimit || phase.StopReason == SaturationMatchWorkLimit {
		result.StopReason = phase.StopReason
	}
	return changed, nil
}

func stringScheduleKind(kind ScheduleNodeKind) string {
	switch kind {
	case ScheduleRunKind:
		return "run"
	case ScheduleSaturateKind:
		return "saturate"
	case ScheduleSequenceKind:
		return "sequence"
	case ScheduleRepeatKind:
		return "repeat"
	default:
		return "unknown"
	}
}

func graphSize(g *EGraph) (classes, nodes int) {
	if g == nil {
		return 0, 0
	}
	return g.store.ClassCount(), g.nodeCount
}

func tupleCount(g *EGraph) int {
	if g == nil {
		return 0
	}
	maxInt := int(^uint(0) >> 1)
	total := g.nodeCount
	add := func(count int) {
		if count > maxInt-total {
			total = maxInt
			return
		}
		total += count
	}
	if g.facts != nil {
		add(g.facts.TupleCount())
	}
	for _, table := range g.functions {
		if table != nil {
			add(table.rows.Len())
		}
	}
	return total
}

// Saturate repeatedly applies rewrite rules until they produce no unions,
// analysis, relation, function, or structural-node changes, or maxIterations is
// reached. It returns union merges only; the report API exposes other changes
// and stop reasons.
//
// Deprecated: Saturate discards errors and stop reasons. Use SaturateWithReport.
func Saturate(g *EGraph, rules []*Rewrite, maxIterations int) (iterations int, totalMerges int) {
	result, _ := SaturateWithReport(g, rules, maxIterations)
	return result.Iterations, result.Merges
}
