package egraph

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"reflect"
	"strings"

	"github.com/tmc/lib/egraphcore"
)

var ErrMatchWorkLimit = errors.New("egraph: pattern match work limit exceeded")

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
	Op       string
	IsVar    bool
	Children []*Pattern
	// HasData matches Data exactly. Without HasData, Data is unconstrained.
	HasData bool
	Data    any
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

// Pat creates an operator pattern node with child patterns.
func Pat(op string, children ...*Pattern) *Pattern {
	return &Pattern{
		Op:       op,
		IsVar:    false,
		Children: children,
	}
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

// MatchMatch represents a matched pattern instance at a specific EClass.
type MatchMatch struct {
	ClassID Id
	Subst   Subst
	Binding egraphcore.Binding[Id]
}

// Search rebuilds pending unions, then searches for all matches across the
// entire E-Graph. It returns nil if matching or rebuilding fails.
func (p *Pattern) Search(g *EGraph) []MatchMatch {
	if p == nil || g == nil {
		return nil
	}
	if err := validatePatternTree(p); err != nil {
		return nil
	}
	if err := g.RebuildChecked(); err != nil {
		return nil
	}
	classes := g.classes()
	roots := make([]Id, len(classes))
	for i, class := range classes {
		roots[i] = class.ID
	}
	const perClassWork = 100000
	maxInt := int(^uint(0) >> 1)
	workLimit := maxInt
	if len(roots) <= maxInt/perClassWork {
		workLimit = len(roots) * perClassWork
	}
	matches, _, err := p.searchCore(g, roots, egraphcore.NewWorkBudget(workLimit), 0)
	if err != nil {
		return nil
	}
	return matches
}

// SearchWithBudget rebuilds pending unions, then returns every match if the
// search completes. It charges class enumeration and match emission to budget
// and returns no partial matches when work is exhausted.
func (p *Pattern) SearchWithBudget(g *EGraph, budget *egraphcore.WorkBudget) ([]MatchMatch, error) {
	if p == nil || g == nil || budget == nil {
		return nil, egraphcore.ErrInvalidPattern
	}
	if err := validatePatternTreeWithBudget(p, budget); err != nil {
		return nil, fmt.Errorf("egraph: invalid pattern: %w", err)
	}
	view, err := matchingView(g, budget)
	if err != nil {
		return nil, fmt.Errorf("egraph: rebuild before search: %w", err)
	}
	g = view
	classCount := g.storage().ClassCount()
	sortWork := classCount
	if classCount > 1 {
		levels := bits.Len(uint(classCount))
		if classCount > math.MaxInt/(2*levels) {
			return nil, matchingError(budget)
		}
		sortWork = 2 * classCount * levels
	}
	if !budget.TrySpend(sortWork) {
		return nil, matchingError(budget)
	}
	roots, err := matchingRoots(g, budget)
	if err != nil {
		return nil, err
	}
	maxMatches := 0
	if remaining := budget.Remaining(); remaining < int(^uint(0)>>1) {
		maxMatches = remaining + 1
	}
	matches, limited, err := p.searchCore(g, roots, budget, maxMatches)
	if errors.Is(err, egraphcore.ErrMatchWorkLimit) {
		return nil, matchingError(budget)
	}
	if err != nil {
		return nil, err
	}
	if limited {
		return nil, matchingError(budget)
	}
	if !budget.TrySpend(len(matches)) {
		return nil, matchingError(budget)
	}
	return matches, nil
}

// SearchBounded searches in class order and returns at most limit matches.
// Matching has a separate 100000-step allowance for calls and node visits;
// a small output limit does not prohibit nested patterns. The bool reports reaching either
// allowance, so it does not imply that another match exists.
func (p *Pattern) SearchBounded(g *EGraph, limit int) ([]MatchMatch, bool) {
	if limit <= 0 || p == nil || g == nil {
		return nil, false
	}
	matches, truncated, err := p.SearchBoundedChecked(g, limit)
	if err != nil {
		return nil, true
	}
	return matches, truncated || len(matches) >= limit
}

// SearchBoundedChecked returns at most limit matches and reports whether more
// matches were available. Unlike SearchBounded, it distinguishes work-limit
// exhaustion and invalid patterns from an output limit.
func (p *Pattern) SearchBoundedChecked(g *EGraph, limit int) ([]MatchMatch, bool, error) {
	return p.SearchBoundedWithBudget(g, limit, egraphcore.NewWorkBudget(100000))
}

// SearchBoundedWithBudget rebuilds pending unions, then returns at most limit
// matches and reports whether more matches were available. It shares budget
// with the caller.
func (p *Pattern) SearchBoundedWithBudget(g *EGraph, limit int, budget *egraphcore.WorkBudget) ([]MatchMatch, bool, error) {
	if limit <= 0 {
		return nil, false, nil
	}
	if p == nil || g == nil || budget == nil {
		return nil, false, egraphcore.ErrInvalidPattern
	}
	if err := validatePatternTreeWithBudget(p, budget); err != nil {
		return nil, false, err
	}
	view, err := matchingView(g, budget)
	if err != nil {
		return nil, false, fmt.Errorf("egraph: rebuild before bounded search: %w", err)
	}
	g = view
	classCount := g.storage().ClassCount()
	sortWork := classCount
	if classCount > 1 {
		levels := bits.Len(uint(classCount))
		if classCount > math.MaxInt/(2*levels) {
			return nil, false, matchingError(budget)
		}
		sortWork = 2 * classCount * levels
	}
	if !budget.TrySpend(sortWork) {
		return nil, false, matchingError(budget)
	}
	roots, err := matchingRoots(g, budget)
	if err != nil {
		return nil, false, err
	}
	maxMatches := limit
	if limit < math.MaxInt {
		maxMatches++
	} else {
		maxMatches = 0
	}
	matches, limited, err := p.searchCore(g, roots, budget, maxMatches)
	if err != nil {
		if errors.Is(err, egraphcore.ErrMatchWorkLimit) {
			return nil, false, matchingError(budget)
		}
		return nil, false, err
	}
	if !budget.TrySpend(len(matches)) {
		return nil, false, matchingError(budget)
	}
	if limited {
		return matches[:limit], true, nil
	}
	return matches, false, nil
}

func (p *Pattern) searchCore(g *EGraph, roots []Id, budget *egraphcore.WorkBudget, maxMatches int) ([]MatchMatch, bool, error) {
	matches, limited, err := egraphcore.SearchWithMatcher(
		p, roots, budget, maxMatches, matcherCallbacks(g),
	)
	if err != nil {
		return nil, false, err
	}
	converted, err := graphSlice[MatchMatch](budget, len(matches), len(matches))
	if err != nil {
		return nil, false, err
	}
	for i, match := range matches {
		binding, err := graphMap[string, egraphcore.Value[Id]](budget, len(match.Subst))
		if err != nil {
			return nil, false, err
		}
		for name, id := range match.Subst {
			if err := graphPut(budget, binding, name, egraphcore.Class(id)); err != nil {
				return nil, false, err
			}
		}
		converted[i] = MatchMatch{ClassID: match.ClassID, Subst: Subst(match.Subst), Binding: binding}
	}
	return converted, limited, nil
}

// MatchEClass rebuilds pending unions, then checks whether this pattern
// matches a specific e-class. It returns nil if matching or rebuilding fails.
func (p *Pattern) MatchEClass(g *EGraph, classID Id, current Subst) []Subst {
	if p == nil || g == nil {
		return nil
	}
	if err := validatePatternTree(p); err != nil {
		return nil
	}
	if g.RebuildChecked() != nil {
		return nil
	}
	result, _ := p.matchEClassBudget(g, classID, current, egraphcore.NewWorkBudget(100000))
	return result
}

// MatchEClassBounded rebuilds pending unions, then matches with a finite budget
// for recursive calls and e-node visits, including nodes that do not match.
// Exhaustion returns no matches and ErrMatchWorkLimit.
func (p *Pattern) MatchEClassBounded(g *EGraph, classID Id, current Subst, limit int) ([]Subst, error) {
	if limit <= 0 {
		return nil, ErrMatchWorkLimit
	}
	if p == nil || g == nil {
		return nil, egraphcore.ErrInvalidPattern
	}
	if err := validatePatternTree(p); err != nil {
		return nil, err
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("egraph: rebuild before class match: %w", err)
	}
	result, err := p.matchEClassBudget(g, classID, current, egraphcore.NewWorkBudget(limit))
	if errors.Is(err, egraphcore.ErrMatchWorkLimit) {
		return nil, ErrMatchWorkLimit
	}
	return result, err
}

func (p *Pattern) matchEClassBudget(g *EGraph, classID Id, current Subst, budget *egraphcore.WorkBudget) ([]Subst, error) {
	if p == nil || g == nil {
		return nil, egraphcore.ErrInvalidPattern
	}
	results, err := egraphcore.MatchEClassWithMatcher(p, classID, current, budget, matcherCallbacks(g))
	if err != nil {
		return nil, err
	}
	converted := make([]Subst, len(results))
	for i, subst := range results {
		converted[i] = Subst(subst)
	}
	return converted, nil
}

func matcherCallbacks(g *EGraph) egraphcore.MatcherCallbacks[Id, ENode, *Pattern] {
	return egraphcore.MatcherCallbacks[Id, ENode, *Pattern]{
		Find:            g.Find,
		ValidPattern:    func(pattern *Pattern) bool { return pattern != nil },
		IsVariable:      func(pattern *Pattern) (string, bool) { return pattern.Op, pattern.IsVar },
		PatternChildren: func(pattern *Pattern) []*Pattern { return pattern.Children },
		Candidates: func(id Id, _ *Pattern) []ENode {
			class := g.class(id)
			if class == nil {
				return nil
			}
			return class.Nodes
		},
		NodeEligible: func(node ENode, pattern *Pattern) bool {
			return !node.Subsumed && pattern.matchesNode(node)
		},
		NodeChildren: func(node ENode) []Id { return node.Children },
	}
}

func (p *Pattern) matchesNode(node ENode) bool {
	if p == nil || p.IsVar || node.Op != p.Op || len(node.Children) != len(p.Children) {
		return false
	}
	if p.HasData {
		return reflect.DeepEqual(p.Data, node.Data)
	}
	return true
}

// Instantiate creates new ENodes in the E-Graph according to this pattern and the given substitution.
func (p *Pattern) Instantiate(g *EGraph, s Subst) Id {
	if p.IsVar {
		if id, ok := s[p.Op]; ok {
			return g.Find(id)
		}
		// Unbound variable becomes a leaf
		return g.Add(NewLeaf(p.Op))
	}

	childIDs := make([]Id, len(p.Children))
	for i, c := range p.Children {
		childIDs[i] = c.Instantiate(g, s)
	}

	return g.Add(NewNode(p.Op, childIDs...))
}

func (p *Pattern) String() string {
	if p.IsVar {
		return p.Op
	}
	if len(p.Children) == 0 {
		return p.Op
	}
	var sb strings.Builder
	sb.WriteString("(")
	sb.WriteString(p.Op)
	for _, c := range p.Children {
		sb.WriteString(" ")
		sb.WriteString(c.String())
	}
	sb.WriteString(")")
	return sb.String()
}
