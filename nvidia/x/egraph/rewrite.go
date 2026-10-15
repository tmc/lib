package egraph

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"

	"github.com/tmc/lib/egraphcore"
)

// Rewrite defines a pattern rewrite with optional positive pattern conditions.
type Rewrite struct {
	Name string
	LHS  *Pattern
	RHS  *Pattern
	// Conditions are positive patterns whose substitutions join with the LHS.
	Conditions []*Pattern
	// RelationConditions are positive typed fact atoms joined with the LHS and
	// Conditions. Names identify relations in the graph's FactStore.
	RelationConditions []RelationCondition
	// RelationActions mutate typed facts for each completed rule match.
	RelationActions []egraphcore.FactAction[Id]
	subsumesLHS     bool
}

// RewriteSearchState retains relation cursors for incremental search. It is
// tied to one Rewrite and one graph revision; reset it after changing either.
type RewriteSearchState struct {
	rule     *Rewrite
	identity string
	revision uint64
	cursors  []egraphcore.RelationCursor
	primed   bool
}

// Reset discards the retained cursors so the next search scans full relations.
func (s *RewriteSearchState) Reset() {
	if s == nil {
		return
	}
	s.rule = nil
	s.identity = ""
	s.revision = 0
	s.cursors = nil
	s.primed = false
}

// SearchWithDelta searches a rewrite incrementally across calls. The first
// call, a graph revision change, or an explicit Reset performs a full relation
// join. Later calls use relation additions since the prior search. A stale
// relation cursor invalidates the saved state and falls back to a full join.
func (r *Rewrite) SearchWithDelta(g *EGraph, state *RewriteSearchState, budget *egraphcore.WorkBudget) ([]MatchMatch, error) {
	if r == nil || g == nil || state == nil || budget == nil {
		return nil, fmt.Errorf("egraph: nil rewrite, graph, state, or work budget")
	}
	atoms, err := relationConditionAtoms(g, r)
	if err != nil {
		return nil, err
	}
	identity := r.Identity()
	useDeltas := len(atoms) != 0 && state.primed && state.rule == r && state.identity == identity && state.revision == g.revision && len(state.cursors) == len(atoms)
	var cursors []egraphcore.RelationCursor
	if useDeltas {
		cursors = state.cursors
	}
	matches, err := searchRewriteMatchesWithDeltaCursors(g, r, budget, cursors, useDeltas)
	if errors.Is(err, egraphcore.ErrRelationReset) && useDeltas {
		state.Reset()
		matches, err = searchRewriteMatchesWithBudget(g, r, budget)
	}
	if err != nil {
		return nil, err
	}
	current := make([]egraphcore.RelationCursor, len(atoms))
	for i, atom := range atoms {
		current[i] = atom.Relation.Snapshot()
	}
	state.rule = r
	state.identity = identity
	state.revision = g.revision
	state.cursors = current
	state.primed = true
	return matches, nil
}

// RelationCondition is one positive atom over a named graph fact relation.
// Terms have one entry per relation column.
type RelationCondition struct {
	Name  string
	Terms []egraphcore.Term[Id]
}

// Identity identifies the complete rule definition, including its conditions.
// Names alone are intentionally insufficient for replay across rule sets.
func (r *Rewrite) Identity() string {
	if r == nil {
		return "<nil>"
	}
	lhs, rhs := "<nil>", "<nil>"
	if r.LHS != nil {
		lhs = r.LHS.String()
	}
	if r.RHS != nil {
		rhs = r.RHS.String()
	}
	identity := r.Name + "|" + lhs + "=>" + rhs
	if r.subsumesLHS {
		identity += " subsume-lhs"
	}
	if len(r.Conditions) != 0 {
		conditions := make([]string, len(r.Conditions))
		for i, condition := range r.Conditions {
			if condition == nil {
				conditions[i] = "<nil>"
			} else {
				conditions[i] = condition.String()
			}
		}
		identity += " where " + strings.Join(conditions, " & ")
	}
	if len(r.RelationConditions) != 0 {
		conditions := make([]string, len(r.RelationConditions))
		for i, condition := range r.RelationConditions {
			conditions[i] = relationConditionIdentity(condition)
		}
		identity += " facts " + strings.Join(conditions, " & ")
	}
	if len(r.RelationActions) != 0 {
		actions := make([]string, len(r.RelationActions))
		for i, action := range r.RelationActions {
			actions[i] = factActionIdentity(action)
		}
		identity += " do " + strings.Join(actions, " & ")
	}
	return identity
}

func relationConditionIdentity(condition RelationCondition) string {
	var b strings.Builder
	b.WriteString(condition.Name)
	b.WriteByte('(')
	for i, term := range condition.Terms {
		if i != 0 {
			b.WriteByte(',')
		}
		switch term.Kind {
		case egraphcore.WildcardTerm:
			b.WriteByte('_')
		case egraphcore.VariableTerm:
			b.WriteString("?")
			b.WriteString(term.Name)
		case egraphcore.ConstantTerm:
			fmt.Fprintf(&b, "%d:%d:%d:%t:%d:%q", term.Value.Kind, uint32(term.Value.Class), term.Value.Int64, term.Value.Bool, term.Value.Float, term.Value.String)
		default:
			fmt.Fprintf(&b, "invalid:%d", term.Kind)
		}
	}
	b.WriteByte(')')
	return b.String()
}

// NewRewrite parses left-hand and right-hand pattern strings and creates a Rewrite rule.
func NewRewrite(name, lhsStr, rhsStr string) (*Rewrite, error) {
	return NewRewriteWithConditions(name, lhsStr, rhsStr)
}

// NewSubsumingRewrite creates a rewrite that hides its matched left-hand node
// from later matching and extraction after applying the right-hand side.
func NewSubsumingRewrite(name, lhsStr, rhsStr string) (*Rewrite, error) {
	r, err := NewRewrite(name, lhsStr, rhsStr)
	if err != nil {
		return nil, err
	}
	r.subsumesLHS = true
	if err := validateRewrite(r); err != nil {
		return nil, err
	}
	return r, nil
}

// NewSubsumingPatternRewrite creates a pattern rewrite that hides its matched
// left-hand node from later matching and extraction after applying the RHS.
func NewSubsumingPatternRewrite(name string, lhs, rhs *Pattern) (*Rewrite, error) {
	r := &Rewrite{Name: name, LHS: lhs, RHS: rhs, subsumesLHS: true}
	if err := validateRewrite(r); err != nil {
		return nil, err
	}
	return r, nil
}

// NewRewriteWithConditions parses the rewrite patterns and positive condition
// patterns. Variables in the RHS may be bound by either the LHS or a condition.
func NewRewriteWithConditions(name, lhsStr, rhsStr string, conditionStrs ...string) (*Rewrite, error) {
	lhs, err := ParsePattern(lhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse lhs %q: %w", lhsStr, err)
	}
	rhs, err := ParsePattern(rhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse rhs %q: %w", rhsStr, err)
	}
	conditions := make([]*Pattern, len(conditionStrs))
	for i, conditionStr := range conditionStrs {
		conditions[i], err = ParsePattern(conditionStr)
		if err != nil {
			return nil, fmt.Errorf("egraph: parse condition %d %q: %w", i, conditionStr, err)
		}
	}
	rule := &Rewrite{
		Name:       name,
		LHS:        lhs,
		RHS:        rhs,
		Conditions: conditions,
	}
	if err := validateRewrite(rule); err != nil {
		return nil, err
	}
	return rule, nil
}

// NewRewriteWithRelationConditions parses a rewrite and attaches positive
// named-relation atoms. Relation schemas are resolved against the EGraph when
// the rule is searched or applied.
func NewRewriteWithRelationConditions(name, lhsStr, rhsStr string, conditions ...RelationCondition) (*Rewrite, error) {
	r, err := NewRewrite(name, lhsStr, rhsStr)
	if err != nil {
		return nil, err
	}
	r.RelationConditions = cloneRelationConditions(conditions)
	if err := validateRewrite(r); err != nil {
		return nil, err
	}
	return r, nil
}

// NewRewriteWithFactActions parses a rewrite and attaches named relation
// mutations. Relation schemas are resolved against the graph when the rule is
// searched or applied.
func NewRewriteWithFactActions(name, lhsStr, rhsStr string, actions ...egraphcore.FactAction[Id]) (*Rewrite, error) {
	lhs, err := ParsePattern(lhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse lhs %q: %w", lhsStr, err)
	}
	var rhs *Pattern
	if rhsStr != "" {
		rhs, err = ParsePattern(rhsStr)
		if err != nil {
			return nil, fmt.Errorf("egraph: parse rhs %q: %w", rhsStr, err)
		}
	}
	r := &Rewrite{Name: name, LHS: lhs, RHS: rhs}
	r.RelationActions = cloneFactActions(actions)
	if err := validateRewrite(r); err != nil {
		return nil, err
	}
	return r, nil
}

func cloneRelationConditions(conditions []RelationCondition) []RelationCondition {
	out := make([]RelationCondition, len(conditions))
	for i, condition := range conditions {
		out[i] = RelationCondition{Name: condition.Name, Terms: append([]egraphcore.Term[Id](nil), condition.Terms...)}
	}
	return out
}

func cloneFactActions(actions []egraphcore.FactAction[Id]) []egraphcore.FactAction[Id] {
	out := make([]egraphcore.FactAction[Id], len(actions))
	for i, action := range actions {
		out[i] = egraphcore.FactAction[Id]{Kind: action.Kind, Relation: action.Relation, Terms: append([]egraphcore.Term[Id](nil), action.Terms...)}
	}
	return out
}

func factActionIdentity(action egraphcore.FactAction[Id]) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d:%s(", action.Kind, action.Relation)
	for i, term := range action.Terms {
		if i != 0 {
			b.WriteByte(',')
		}
		switch term.Kind {
		case egraphcore.VariableTerm:
			fmt.Fprintf(&b, "?%s", term.Name)
		case egraphcore.ConstantTerm:
			fmt.Fprintf(&b, "=%#v", term.Value)
		default:
			fmt.Fprintf(&b, "invalid:%d", term.Kind)
		}
	}
	b.WriteByte(')')
	return b.String()
}

// MustRewrite creates a Rewrite or panics if parsing fails (useful for table-defined constants).
func MustRewrite(name, lhsStr, rhsStr string) *Rewrite {
	r, err := NewRewrite(name, lhsStr, rhsStr)
	if err != nil {
		panic(err)
	}
	return r
}

// Apply searches the E-Graph for matches of LHS and unifies them with RHS instantiations.
func (r *Rewrite) Apply(g *EGraph) int {
	merges, _ := r.ApplyWithBudget(g, egraphcore.NewWorkBudget(0))
	return merges
}

// ApplyWithBudget searches completely and charges matching and RHS traversal
// before mutating g. If the budget is exhausted, it returns
// ErrMatchWorkLimit without applying any matches.
func (r *Rewrite) ApplyWithBudget(g *EGraph, budget *egraphcore.WorkBudget) (int, error) {
	merges, _, _, err := r.applyWithBudget(g, budget)
	return merges, err
}

func (r *Rewrite) applyWithBudget(g *EGraph, budget *egraphcore.WorkBudget) (int, int, int, error) {
	return r.applyWithSearchState(g, budget, nil, 0, true)
}

func (r *Rewrite) applyWithSearchState(g *EGraph, budget *egraphcore.WorkBudget, searchState *RewriteSearchState, maxNodes int, outerTransaction bool) (int, int, int, error) {
	if err := validateRewrite(r); err != nil {
		return 0, 0, 0, err
	}
	if g == nil || budget == nil {
		return 0, 0, 0, fmt.Errorf("egraph: nil graph or work budget")
	}
	// Apply has no typed proof state.  Hardware-sensitive rules must go
	// through SearchState.ApplyAction, where their operand contracts are
	// checked, so the legacy entry point always fails closed.
	if isMandatoryRule(r.Name) || isMandatoryRewrite(r) {
		return 0, 0, 0, nil
	}
	var err error
	var matches []MatchMatch
	if searchState == nil {
		matches, err = searchRewriteMatchesWithBudget(g, r, budget)
	} else {
		matches, err = r.SearchWithDelta(g, searchState, budget)
	}
	if err != nil {
		return 0, 0, 0, err
	}
	rhsWork := 0
	if r.RHS != nil {
		rhsWork, err = patternWork(r.RHS)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("egraph: rewrite %q rhs: %w", r.Name, err)
		}
	}
	applyWork := rhsWork
	if maxNodes > 0 {
		if applyWork > math.MaxInt-applyWork {
			return 0, 0, 0, fmt.Errorf("egraph: rewrite %q rhs work count overflow", r.Name)
		}
		applyWork += applyWork
	}
	resolved, err := resolveFactActions(g, r, matches, budget)
	if err != nil {
		return 0, 0, 0, err
	}
	if len(resolved) != 0 {
		staged := g
		if outerTransaction {
			_, cloneWork := g.Size()
			if !budget.TrySpend(cloneWork) {
				return 0, 0, 0, ErrMatchWorkLimit
			}
			staged = g.Clone()
		}
		merges := 0
		if r.RHS != nil {
			report, err := egraphcore.ApplyMatches[Id, ENode, MatchMatch](
				staged,
				matches,
				func(match MatchMatch) Id { return match.ClassID },
				func(match MatchMatch) (Id, error) { return r.RHS.instantiateFrom(staged, match.Subst, r.Name), nil },
				func(MatchMatch) int { return applyWork },
				patternGrowthAdmission(staged, maxNodes, r.RHS),
				budget,
				true,
			)
			if errors.Is(err, egraphcore.ErrRewriteWorkBudget) {
				return 0, 0, 0, ErrMatchWorkLimit
			}
			if err != nil {
				return 0, 0, 0, fmt.Errorf("egraph: rewrite %q apply: %w", r.Name, err)
			}
			merges = report.Merges
		}
		factChanges, err := staged.Facts().ApplyActions(resolved, staged.Find, budget)
		if err != nil {
			if errors.Is(err, egraphcore.ErrMatchWorkLimit) {
				return 0, 0, 0, ErrMatchWorkLimit
			}
			return 0, 0, 0, err
		}
		subsumptions, err := subsumeMatchedNodes(staged, r, matches, budget)
		if err != nil {
			return 0, 0, 0, err
		}
		if outerTransaction {
			facts := g.Facts()
			if err := facts.ReplaceFrom(staged.Facts(), budget); err != nil {
				if errors.Is(err, egraphcore.ErrMatchWorkLimit) {
					return 0, 0, 0, ErrMatchWorkLimit
				}
				return 0, 0, 0, err
			}
			staged.facts = facts
			*g = *staged
		}
		return merges, factChanges, subsumptions, nil
	}

	merges := 0
	if r.RHS != nil {
		staged := g
		if outerTransaction {
			_, cloneWork := g.Size()
			if !budget.TrySpend(cloneWork) {
				return 0, 0, 0, ErrMatchWorkLimit
			}
			staged = g.Clone()
		}
		report, err := egraphcore.ApplyMatches[Id, ENode, MatchMatch](
			staged,
			matches,
			func(match MatchMatch) Id { return match.ClassID },
			func(match MatchMatch) (Id, error) { return r.RHS.instantiateFrom(staged, match.Subst, r.Name), nil },
			func(MatchMatch) int { return applyWork },
			patternGrowthAdmission(staged, maxNodes, r.RHS),
			budget,
			true,
		)
		if errors.Is(err, egraphcore.ErrRewriteWorkBudget) {
			return 0, 0, 0, ErrMatchWorkLimit
		}
		if err != nil {
			return 0, 0, 0, fmt.Errorf("egraph: rewrite %q apply: %w", r.Name, err)
		}
		merges = report.Merges
		subsumptions, err := subsumeMatchedNodes(staged, r, matches, budget)
		if err != nil {
			return 0, 0, 0, err
		}
		if outerTransaction {
			*g = *staged
		}
		return merges, 0, subsumptions, nil
	}
	return merges, 0, 0, nil
}

func resolveFactActions(g *EGraph, r *Rewrite, matches []MatchMatch, budget *egraphcore.WorkBudget) ([]egraphcore.ResolvedFactAction[Id], error) {
	var resolved []egraphcore.ResolvedFactAction[Id]
	for _, match := range matches {
		binding := match.Binding
		if binding == nil {
			binding = make(egraphcore.Binding[Id], len(match.Subst))
			for name, id := range match.Subst {
				if !budget.TrySpend(1) {
					return nil, ErrMatchWorkLimit
				}
				binding[name] = egraphcore.Class(id)
			}
		}
		actions, err := egraphcore.ResolveFactActions(g.Facts(), r.RelationActions, binding, g.Find, budget)
		if err != nil {
			if errors.Is(err, egraphcore.ErrMatchWorkLimit) {
				return nil, ErrMatchWorkLimit
			}
			return nil, err
		}
		resolved = append(resolved, actions...)
	}
	return resolved, nil
}

func validateRewrite(r *Rewrite) error {
	return validateRewriteWithBudget(r, nil)
}

func validateRewriteWithBudget(r *Rewrite, budget *egraphcore.WorkBudget) error {
	if r == nil || r.LHS == nil || r.RHS == nil && len(r.RelationActions) == 0 {
		return fmt.Errorf("egraph: rewrite and patterns must be non-nil")
	}
	if err := validatePatternTreeWithBudget(r.LHS, budget); err != nil {
		return fmt.Errorf("egraph: invalid lhs: %w", err)
	}
	if r.RHS != nil {
		if err := validatePatternTreeWithBudget(r.RHS, budget); err != nil {
			return fmt.Errorf("egraph: invalid rhs: %w", err)
		}
	}
	if r.subsumesLHS && (r.LHS.IsVar || r.RHS == nil) {
		return fmt.Errorf("egraph: rewrite %q can only subsume a concrete term pattern with an rhs", r.Name)
	}
	bound, err := graphMap[string, bool](budget, 0)
	if err != nil {
		return err
	}
	if err := patternVariablesWithBudget(r.LHS, bound, budget); err != nil {
		return err
	}
	for i, condition := range r.Conditions {
		if err := validatePatternTreeWithBudget(condition, budget); err != nil {
			return fmt.Errorf("invalid condition %d: %w", i, err)
		}
		if err := patternVariablesWithBudget(condition, bound, budget); err != nil {
			return err
		}
	}
	for i, condition := range r.RelationConditions {
		if condition.Name == "" {
			return fmt.Errorf("invalid relation condition %d: empty relation name", i)
		}
		for j, term := range condition.Terms {
			switch term.Kind {
			case egraphcore.WildcardTerm:
			case egraphcore.VariableTerm:
				if term.Name == "" {
					return fmt.Errorf("invalid relation condition %d column %d: empty variable", i, j)
				}
				if err := graphPut(budget, bound, term.Name, true); err != nil {
					return err
				}
			case egraphcore.ConstantTerm:
				if term.Value.Kind < egraphcore.ClassValue || term.Value.Kind > egraphcore.BigRatValue {
					return fmt.Errorf("invalid relation condition %d column %d: invalid constant kind", i, j)
				}
			default:
				return fmt.Errorf("invalid relation condition %d column %d: invalid term kind", i, j)
			}
		}
	}
	for i, action := range r.RelationActions {
		if action.Relation == "" {
			return fmt.Errorf("invalid relation action %d: empty relation name", i)
		}
		if action.Kind < egraphcore.FactInsert || action.Kind > egraphcore.FactSubsume {
			return fmt.Errorf("invalid relation action %d: invalid action kind", i)
		}
		for j, term := range action.Terms {
			switch term.Kind {
			case egraphcore.VariableTerm:
				if term.Name == "" {
					return fmt.Errorf("invalid relation action %d column %d: empty variable", i, j)
				}
				if !bound[term.Name] {
					return fmt.Errorf("relation action %d variable %q is not bound by lhs or conditions", i, term.Name)
				}
			case egraphcore.ConstantTerm:
				if term.Name != "" || term.Value.Kind < egraphcore.ClassValue || term.Value.Kind > egraphcore.BigRatValue {
					return fmt.Errorf("invalid relation action %d column %d: invalid constant", i, j)
				}
			default:
				return fmt.Errorf("invalid relation action %d column %d: only variables and constants are allowed", i, j)
			}
		}
	}
	var unbound []string
	rhsVariables, err := graphMap[string, bool](budget, 0)
	if err != nil {
		return err
	}
	if r.RHS != nil {
		if err := patternVariablesWithBudget(r.RHS, rhsVariables, budget); err != nil {
			return err
		}
	}
	for name := range rhsVariables {
		if !bound[name] {
			unbound, err = graphAppend(budget, unbound, name)
			if err != nil {
				return err
			}
		}
	}
	if len(unbound) != 0 {
		sort.Strings(unbound)
		source := "lhs"
		if len(r.Conditions) != 0 {
			source = "lhs or conditions"
		}
		return fmt.Errorf("rhs variables not bound by %s: %s", source, strings.Join(unbound, ", "))
	}
	return nil
}

// searchRewriteMatchesWithBudget matches the LHS, then joins each positive
// pattern condition with the current variable bindings. ClassID remains the
// LHS root match anchor; condition roots only contribute variable bindings.
func searchRewriteMatchesWithBudget(g *EGraph, r *Rewrite, budget *egraphcore.WorkBudget) ([]MatchMatch, error) {
	return searchRewriteMatchesWithDeltaCursors(g, r, budget, nil, false)
}

func searchRewriteMatchesWithDeltaCursors(g *EGraph, r *Rewrite, budget *egraphcore.WorkBudget, cursors []egraphcore.RelationCursor, useDeltas bool) ([]MatchMatch, error) {
	if err := validateRewriteWithBudget(r, budget); err != nil {
		return nil, err
	}
	view, err := matchingView(g, budget)
	if err != nil {
		return nil, err
	}
	g = view
	if useDeltas && budget.MemoryScope() != nil {
		return nil, ErrMemoryOperation
	}
	atoms, err := relationConditionAtomsWithBudget(g, r, budget)
	if err != nil {
		return nil, err
	}
	if err := validateRelationConditionTypesWithBudget(g, r, atoms, budget); err != nil {
		return nil, err
	}
	lhsMatches, err := r.LHS.SearchWithBudget(g, budget)
	if err != nil || len(lhsMatches) == 0 {
		return lhsMatches, err
	}
	if len(r.Conditions) == 0 && len(r.RelationConditions) == 0 {
		return lhsMatches, nil
	}
	conditionMatches, err := graphSlice[[]MatchMatch](budget, len(r.Conditions), len(r.Conditions))
	if err != nil {
		return nil, err
	}
	for i, condition := range r.Conditions {
		conditionMatches[i], err = condition.SearchWithBudget(g, budget)
		if err != nil {
			return nil, err
		}
		if len(conditionMatches[i]) == 0 {
			return nil, nil
		}
	}
	anchors, err := graphSlice[egraphcore.Match[Id]](budget, len(lhsMatches), len(lhsMatches))
	if err != nil {
		return nil, err
	}
	for i, match := range lhsMatches {
		subst, err := graphMap[string, Id](budget, len(match.Subst))
		if err != nil {
			return nil, err
		}
		anchors[i] = egraphcore.Match[Id]{ClassID: match.ClassID, Subst: subst}
		for name, id := range match.Subst {
			if !budget.TrySpend(1) {
				return nil, matchingError(budget)
			}
			if err := graphPut(budget, anchors[i].Subst, name, id); err != nil {
				return nil, err
			}
		}
	}
	structural, err := graphSlice[[]egraphcore.Match[Id]](budget, len(conditionMatches), len(conditionMatches))
	if err != nil {
		return nil, err
	}
	for i, matches := range conditionMatches {
		structural[i], err = graphSlice[egraphcore.Match[Id]](budget, len(matches), len(matches))
		if err != nil {
			return nil, err
		}
		for j, match := range matches {
			subst, err := graphMap[string, Id](budget, len(match.Subst))
			if err != nil {
				return nil, err
			}
			structural[i][j] = egraphcore.Match[Id]{ClassID: match.ClassID, Subst: subst}
			for name, id := range match.Subst {
				if !budget.TrySpend(1) {
					return nil, matchingError(budget)
				}
				if err := graphPut(budget, structural[i][j].Subst, name, id); err != nil {
					return nil, err
				}
			}
		}
	}
	var joined []egraphcore.RuleMatch[Id]
	if useDeltas {
		if len(cursors) != len(atoms) {
			return nil, egraphcore.ErrInvalidJoin
		}
		deltas := make([]egraphcore.DeltaAtom[Id], len(atoms))
		for i, cursor := range cursors {
			deltas[i] = egraphcore.DeltaAtom[Id]{Index: i, Cursor: cursor}
		}
		joined, err = egraphcore.JoinMatchesDeltas(anchors, structural, atoms, deltas, g.Find, budget)
	} else {
		joined, err = egraphcore.JoinMatches(anchors, structural, atoms, g.Find, budget)
	}
	if err != nil {
		if errors.Is(err, egraphcore.ErrMatchWorkLimit) {
			return nil, matchingError(budget)
		}
		return nil, err
	}
	results, err := graphSlice[MatchMatch](budget, len(joined), len(joined))
	if err != nil {
		return nil, err
	}
	for i, match := range joined {
		subst, err := graphMap[string, Id](budget, len(match.Binding))
		if err != nil {
			return nil, err
		}
		binding, err := graphMap[string, egraphcore.Value[Id]](budget, len(match.Binding))
		if err != nil {
			return nil, err
		}
		for name, value := range match.Binding {
			if !budget.TrySpend(1) {
				return nil, matchingError(budget)
			}
			if err := graphPut(budget, binding, name, value); err != nil {
				return nil, err
			}
			if value.Kind == egraphcore.ClassValue {
				if err := graphPut(budget, subst, name, g.Find(value.Class)); err != nil {
					return nil, err
				}
			}
		}
		results[i] = MatchMatch{ClassID: g.Find(match.ClassID), Subst: subst, Binding: binding}
	}
	return results, nil
}

func relationConditionAtoms(g *EGraph, r *Rewrite) ([]egraphcore.Atom[Id], error) {
	return relationConditionAtomsWithBudget(g, r, nil)
}

func relationConditionAtomsWithBudget(g *EGraph, r *Rewrite, budget *egraphcore.WorkBudget) ([]egraphcore.Atom[Id], error) {
	atoms, err := graphSlice[egraphcore.Atom[Id]](budget, len(r.RelationConditions), len(r.RelationConditions))
	if err != nil {
		return nil, err
	}
	for i, condition := range r.RelationConditions {
		relation := g.Facts().Get(condition.Name)
		if relation == nil {
			return nil, fmt.Errorf("egraph: rewrite %q references unknown relation %q", r.Name, condition.Name)
		}
		terms, err := graphSlice[egraphcore.Term[Id]](budget, len(condition.Terms), len(condition.Terms))
		if err != nil {
			return nil, err
		}
		copy(terms, condition.Terms)
		atoms[i] = egraphcore.Atom[Id]{Relation: relation, Terms: terms}
	}
	return atoms, nil
}

func validateRelationConditionTypes(g *EGraph, r *Rewrite, atoms []egraphcore.Atom[Id]) error {
	return validateRelationConditionTypesWithBudget(g, r, atoms, nil)
}

func validateRelationConditionTypesWithBudget(g *EGraph, r *Rewrite, atoms []egraphcore.Atom[Id], budget *egraphcore.WorkBudget) error {
	kinds, err := graphMap[string, egraphcore.ValueKind](budget, 0)
	if err != nil {
		return err
	}
	for _, condition := range r.Conditions {
		vars, err := graphMap[string, bool](budget, 0)
		if err != nil {
			return err
		}
		if err := patternVariablesWithBudget(condition, vars, budget); err != nil {
			return err
		}
		for name := range vars {
			if old, ok := kinds[name]; ok && old != egraphcore.ClassValue {
				return fmt.Errorf("egraph: rewrite %q variable %q has conflicting condition types", r.Name, name)
			}
			if err := graphPut(budget, kinds, name, egraphcore.ClassValue); err != nil {
				return err
			}
		}
	}
	if r.LHS != nil {
		vars, err := graphMap[string, bool](budget, 0)
		if err != nil {
			return err
		}
		if err := patternVariablesWithBudget(r.LHS, vars, budget); err != nil {
			return err
		}
		for name := range vars {
			if err := graphPut(budget, kinds, name, egraphcore.ClassValue); err != nil {
				return err
			}
		}
	}
	for i, atom := range atoms {
		schema, err := atom.Relation.SchemaWithBudget(budget)
		if err != nil {
			return err
		}
		if len(atom.Terms) != len(schema.Columns) {
			return fmt.Errorf("egraph: rewrite %q relation condition %d has %d terms for %d columns", r.Name, i, len(atom.Terms), len(schema.Columns))
		}
		for column, term := range atom.Terms {
			kind := schema.Columns[column].Kind
			switch term.Kind {
			case egraphcore.ConstantTerm:
				if term.Value.Kind != kind {
					return fmt.Errorf("egraph: rewrite %q relation condition %d column %d has constant kind %d, want %d", r.Name, i, column, term.Value.Kind, kind)
				}
			case egraphcore.VariableTerm:
				if old, ok := kinds[term.Name]; ok && old != kind {
					return fmt.Errorf("egraph: rewrite %q variable %q has conflicting condition types", r.Name, term.Name)
				}
				if err := graphPut(budget, kinds, term.Name, kind); err != nil {
					return err
				}
			}
		}
	}
	for i, action := range r.RelationActions {
		relation := g.Facts().Get(action.Relation)
		if relation == nil {
			return fmt.Errorf("egraph: rewrite %q references unknown relation action target %q", r.Name, action.Relation)
		}
		schema, err := relation.SchemaWithBudget(budget)
		if err != nil {
			return err
		}
		if len(action.Terms) != len(schema.Columns) {
			return fmt.Errorf("egraph: rewrite %q relation action %d has %d terms for %d columns", r.Name, i, len(action.Terms), len(schema.Columns))
		}
		for column, term := range action.Terms {
			kind := schema.Columns[column].Kind
			switch term.Kind {
			case egraphcore.ConstantTerm:
				if term.Value.Kind != kind {
					return fmt.Errorf("egraph: rewrite %q relation action %d column %d has constant kind %d, want %d", r.Name, i, column, term.Value.Kind, kind)
				}
			case egraphcore.VariableTerm:
				boundKind, ok := kinds[term.Name]
				if !ok {
					return fmt.Errorf("egraph: rewrite %q relation action variable %q is not bound", r.Name, term.Name)
				}
				if boundKind != kind {
					return fmt.Errorf("egraph: rewrite %q relation action %d column %d variable %q has kind %d, want %d", r.Name, i, column, term.Name, boundKind, kind)
				}
			}
		}
	}
	rhs, err := graphMap[string, bool](budget, 0)
	if err != nil {
		return err
	}
	if r.RHS != nil {
		if err := patternVariablesWithBudget(r.RHS, rhs, budget); err != nil {
			return err
		}
	}
	for name := range rhs {
		if kinds[name] != egraphcore.ClassValue {
			return fmt.Errorf("egraph: rewrite %q rhs variable %q is not bound to an e-class", r.Name, name)
		}
	}
	return nil
}

func substBinding(subst Subst, budget *egraphcore.WorkBudget) (egraphcore.Binding[Id], bool) {
	binding := make(egraphcore.Binding[Id], len(subst))
	for name, id := range subst {
		if !budget.TrySpend(1) {
			return nil, false
		}
		binding[name] = egraphcore.Class[Id](id)
	}
	return binding, true
}

func sharedBindingNames(left, right []egraphcore.Binding[Id], budget *egraphcore.WorkBudget) ([]string, bool) {
	if len(left) == 0 || len(right) == 0 {
		return nil, true
	}
	shared := make(map[string]bool)
	for name := range left[0] {
		if !budget.TrySpend(1) {
			return nil, false
		}
		if _, ok := right[0][name]; ok {
			shared[name] = true
		}
	}
	keys := make([]string, 0, len(shared))
	for name := range shared {
		if !budget.TrySpend(1) {
			return nil, false
		}
		keys = append(keys, name)
	}
	return keys, true
}

func validatePatternTree(root *Pattern) error {
	return validatePatternTreeWithBudget(root, nil)
}

func validatePatternTreeWithBudget(root *Pattern, budget *egraphcore.WorkBudget) error {
	type entry struct {
		pattern *Pattern
		exit    bool
	}
	state, err := graphMap[*Pattern, uint8](budget, 0)
	if err != nil {
		return err
	}
	stack, err := graphSlice[entry](budget, 1, 1)
	if err != nil {
		return err
	}
	stack[0] = entry{pattern: root}
	for len(stack) > 0 {
		last := len(stack) - 1
		current := stack[last]
		stack = stack[:last]
		if current.pattern == nil {
			return fmt.Errorf("nil pattern node")
		}
		if budget.MemoryScope() != nil && current.pattern.HasData && dataComparisonAllocates(reflect.TypeOf(current.pattern.Data)) {
			return fmt.Errorf("%w: reference payload comparison", ErrMemoryOperation)
		}
		if current.exit {
			if err := graphPut(budget, state, current.pattern, uint8(2)); err != nil {
				return err
			}
			continue
		}
		switch state[current.pattern] {
		case 1:
			return fmt.Errorf("cyclic pattern")
		case 2:
			continue
		}
		if err := graphPut(budget, state, current.pattern, uint8(1)); err != nil {
			return err
		}
		if current.pattern.IsVar && len(current.pattern.Children) != 0 {
			return fmt.Errorf("variable pattern %q has children", current.pattern.Op)
		}
		stack, err = graphAppend(budget, stack, entry{pattern: current.pattern, exit: true})
		if err != nil {
			return err
		}
		for i := len(current.pattern.Children) - 1; i >= 0; i-- {
			stack, err = graphAppend(budget, stack, entry{pattern: current.pattern.Children[i]})
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func patternVariables(root *Pattern, variables map[string]bool) {
	_ = patternVariablesWithBudget(root, variables, nil)
}

func patternVariablesWithBudget(root *Pattern, variables map[string]bool, budget *egraphcore.WorkBudget) error {
	stack, err := graphSlice[*Pattern](budget, 1, 1)
	if err != nil {
		return err
	}
	stack[0] = root
	for len(stack) > 0 {
		last := len(stack) - 1
		current := stack[last]
		stack = stack[:last]
		if current.IsVar {
			if err := graphPut(budget, variables, current.Op, true); err != nil {
				return err
			}
			continue
		}
		stack, err = graphAppend(budget, stack, current.Children...)
		if err != nil {
			return err
		}
	}
	return nil
}

func patternWork(root *Pattern) (int, error) {
	work := 0
	stack := []*Pattern{root}
	for len(stack) > 0 {
		last := len(stack) - 1
		current := stack[last]
		stack = stack[:last]
		if work == math.MaxInt {
			return 0, fmt.Errorf("pattern work count overflow")
		}
		work++
		stack = append(stack, current.Children...)
	}
	return work, nil
}

func patternGrowthAdmission(g *EGraph, maxNodes int, pattern *Pattern) func(MatchMatch) error {
	if maxNodes <= 0 {
		return nil
	}
	_, reservedNodes := g.Size()
	return func(match MatchMatch) error {
		if reservedNodes > maxNodes {
			return egraphcore.ErrRewriteNodeLimit
		}
		growth, err := patternNodeGrowthUpperBound(pattern, match.Subst)
		if err != nil {
			return err
		}
		if growth > maxNodes-reservedNodes {
			return egraphcore.ErrRewriteNodeLimit
		}
		reservedNodes += growth
		return nil
	}
}

func patternNodeGrowthUpperBound(root *Pattern, subst Subst) (int, error) {
	var growth int
	stack := []*Pattern{root}
	for len(stack) > 0 {
		last := len(stack) - 1
		pattern := stack[last]
		stack = stack[:last]
		if pattern.IsVar {
			if _, bound := subst[pattern.Op]; bound {
				continue
			}
		}
		if growth == math.MaxInt {
			return 0, fmt.Errorf("pattern node growth bound overflow")
		}
		growth++
		stack = append(stack, pattern.Children...)
	}
	return growth, nil
}

func isMandatoryRule(name string) bool {
	switch name {
	case "hoist_q2k_scales_weights", "hoist_iq2xxs_scales_weights", "fuse_rmsnorm_quant_q8", "fuse_rmsnorm_quant_q8_default_eps", "fuse_swiglu_quant_q8_1", "vectorize_moe_gather_w16", "vectorize_moe_gather_w32", "prefill_group_expand_w256":
		return true
	default:
		return false
	}
}

func isMandatoryRewrite(r *Rewrite) bool {
	if r == nil || r.LHS == nil || r.RHS == nil {
		return false
	}
	sig := r.LHS.String() + "=>" + r.RHS.String()
	for _, known := range MandatoryRewriteRules() {
		if known.LHS.String()+"=>"+known.RHS.String() == sig {
			return true
		}
	}
	return false
}

const defaultSaturationIterations = 256
const defaultSaturationMatchWork = 100_000_000
const defaultSaturationNodes = 10_000_000
const defaultSaturationTuples = 10_000_000

// SaturationOptions bounds one rewrite saturation. Non-positive limits use
// defaults: 256 iterations, ten million nodes, ten million tuples, and 100
// million search, graph-copy, RHS, and fact-action work units. MaxTuples counts
// e-nodes and stored facts, including facts marked subsumed. For
// SaturateScheduleWithReport, MaxIterations caps Run-group callback
// invocations; SaturateWithReport uses a single Run group inside a fixed-point
// schedule, so the same limit counts fixed-point passes.
type SaturationOptions struct {
	MaxIterations int
	MaxNodes      int
	MaxMatchWork  int
	MaxTuples     int
}

type SaturationStopReason = egraphcore.SaturationStopReason

const (
	SaturationFixedPoint     = egraphcore.SaturationFixedPoint
	SaturationIterationLimit = egraphcore.SaturationIterationLimit
	SaturationMatchWorkLimit = egraphcore.SaturationWorkLimit
	SaturationNodeLimit      = egraphcore.SaturationNodeLimit
	SaturationTupleLimit     = egraphcore.SaturationTupleLimit
	SaturationScheduleDone   = egraphcore.SaturationScheduleDone
)

// SaturationResult reports pass invocations, successful class merges, work,
// the graph's committed node count, and why saturation stopped.
type SaturationResult struct {
	Iterations   int
	Merges       int
	FactChanges  int
	Subsumptions int
	MatchWork    int
	Nodes        int
	Tuples       int
	Phases       []egraphcore.ScheduleRunReport
	StopReason   SaturationStopReason
}

// SaturateWithReport repeatedly applies rules until they make no changes or a
// configured node, iteration, or work limit is reached. Each pass is
// transactional: if a rule cannot finish within the shared work budget or
// exceeds the node limit, that pass is discarded. Malformed rewrites and graph
// failures are returned as errors.
func SaturateWithReport(g *EGraph, rules []*Rewrite, opts SaturationOptions) (SaturationResult, error) {
	result, err := SaturateScheduleWithReport(g, egraphcore.Saturate(egraphcore.Run("main", rules...)), opts)
	if result.StopReason == SaturationScheduleDone {
		result.StopReason = SaturationFixedPoint
	}
	return result, err
}

// SaturateChecked applies rules and returns an error unless the graph reaches a
// fixed point. Callers that extract or execute a result should use this when a
// bounded, incomplete search space is not valid input.
func SaturateChecked(g *EGraph, rules []*Rewrite, opts SaturationOptions) (SaturationResult, error) {
	result, err := SaturateWithReport(g, rules, opts)
	if err != nil {
		return result, err
	}
	if result.StopReason != SaturationFixedPoint {
		return result, fmt.Errorf("egraph: saturation stopped before fixed point: %d", result.StopReason)
	}
	return result, nil
}

// Saturate repeatedly applies rules until no more unions occur or the
// iteration or default matching-work limit is reached. A non-positive
// maxIterations uses the default limit of 256. Expected limit stops are
// available through SaturateWithReport.
func Saturate(g *EGraph, rules []*Rewrite, maxIterations int) (iterations int, totalMerges int) {
	result, _ := SaturateWithReport(g, rules, SaturationOptions{MaxIterations: maxIterations})
	return result.Iterations, result.Merges
}
