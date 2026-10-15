package egraph

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// ErrActionEnumerationLimit reports that action enumeration stopped at its
// requested output limit, so the returned candidates may not be complete.
var ErrActionEnumerationLimit = errors.New("egraph: action enumeration limit reached")

// Action represents a single, deterministic rewrite transition applied to a specific e-class match.
type Action struct {
	RuleName  string                 // Identifier of the rewrite rule
	ClassID   Id                     // Target e-class ID matched
	RHSNode   string                 // Serialized representation of RHS target instantiation
	StepIndex int                    // Monotonic step index in search derivation
	Subst     Subst                  // Exact variable-to-class bindings used by the match
	Binding   egraphcore.Binding[Id] // Complete typed binding, including primitive facts.
	Source    string                 // Fingerprint of the matched class before transition
	RuleID    string                 // Full rule identity used for replay validation
}

// Key returns a canonical string representation of the action for deduplication and sorting.
func (a Action) Key() string {
	return fmt.Sprintf("%s@%d[%s]->%s", a.RuleName, a.ClassID, a.Source, a.RHSNode)
}

func classFingerprint(g *EGraph, id Id) string {
	c := g.class(id)
	if c == nil {
		return "<missing>"
	}
	parts := make([]string, 0, len(c.Nodes))
	for _, n := range c.Nodes {
		parts = append(parts, n.Key())
	}
	sort.Strings(parts)
	return strings.Join(parts, "|")
}

// SearchState encapsulates a snapshot of an e-graph during heuristic exploration (Beam / MCTS),
// along with its recorded derivation history and root mapping.
type SearchState struct {
	Graph      *EGraph
	Roots      map[string]Id
	Derivation []Action
	Depth      int
	Metadata   map[Id]TypedMetadata
}

// NewSearchState initializes a fresh SearchState wrapping the given graph and root mappings.
func NewSearchState(g *EGraph, roots map[string]Id) *SearchState {
	if g == nil {
		return nil
	}
	copiedRoots := make(map[string]Id, len(roots))
	for k, v := range roots {
		copiedRoots[k] = g.Find(v)
	}
	return &SearchState{
		Graph:      g.Clone(),
		Roots:      copiedRoots,
		Derivation: nil,
		Depth:      0,
		Metadata:   make(map[Id]TypedMetadata),
	}
}

// Clone creates an isolated, deep copy of the search state for branching.
func (s *SearchState) Clone() *SearchState {
	if s == nil || s.Graph == nil {
		return nil
	}
	copiedRoots := make(map[string]Id, len(s.Roots))
	for k, v := range s.Roots {
		copiedRoots[k] = s.Graph.Find(v)
	}
	copiedDerivation := make([]Action, len(s.Derivation))
	for i, action := range s.Derivation {
		copiedDerivation[i] = action
		copiedDerivation[i].Subst = action.Subst.Clone()
		copiedDerivation[i].Binding = cloneBinding(action.Binding)
	}

	return &SearchState{
		Graph:      s.Graph.Clone(),
		Roots:      copiedRoots,
		Derivation: copiedDerivation,
		Depth:      s.Depth,
		Metadata:   cloneMetadata(s.Metadata),
	}
}

func cloneMetadata(src map[Id]TypedMetadata) map[Id]TypedMetadata {
	dst := make(map[Id]TypedMetadata, len(src))
	for id, meta := range src {
		dst[id] = meta.Clone()
	}
	return dst
}

func cloneBinding(src egraphcore.Binding[Id]) egraphcore.Binding[Id] {
	dst := make(egraphcore.Binding[Id], len(src))
	for name, value := range src {
		dst[name] = value
	}
	return dst
}

// SetMetadata attaches the checked semantic contract used by mandatory rules.
func (s *SearchState) SetMetadata(id Id, meta TypedMetadata) {
	if s == nil || s.Graph == nil {
		return
	}
	if s.Metadata == nil {
		s.Metadata = make(map[Id]TypedMetadata)
	}
	s.Metadata[s.Graph.Find(id)] = meta.Clone()
}

// CanonicalRoot returns the current canonical ID of the named root.
func (s *SearchState) CanonicalRoot(name string) (Id, bool) {
	if s == nil || s.Graph == nil {
		return 0, false
	}
	id, ok := s.Roots[name]
	if !ok {
		return 0, false
	}
	return s.Graph.Find(id), true
}

// ActionCandidate bundles a concrete rewrite rule and its specific match for execution.
type ActionCandidate struct {
	Rule  *Rewrite
	Match MatchMatch
}

// ActionSet owns the storage created while enumerating its Actions. Truncated
// reports that more matches exist. Close follows the last use of the actions;
// their rules and graph nodes remain borrowed from the caller.
type ActionSet struct {
	Actions   []ActionCandidate
	Truncated bool
	memory    *egraphcore.MemoryScope
}

// Close releases the action set's reservations once. It does not close or
// mutate the borrowed source graph or rules.
func (s *ActionSet) Close() {
	if s == nil {
		return
	}
	s.Actions = nil
	s.memory.Close()
}

// EnumerateActionsWithMemory reserves supported owned matching storage before
// allocation. The caller provides a clean graph and keeps the returned owner
// open while using its actions. Memory units and callback obligations follow
// egraphcore.MemoryScope; this is not a process or compiler memory limit.
func EnumerateActionsWithMemory(g *EGraph, rules []*Rewrite, limit int, memory egraphcore.Memory) (*ActionSet, error) {
	scope, err := egraphcore.NewMemoryScope(memory)
	if err != nil {
		return nil, err
	}
	budget, err := egraphcore.NewWorkBudgetWithMemory(100000, scope)
	if err != nil {
		scope.Close()
		return nil, err
	}
	if err := graphSlots[ActionSet](budget, 1); err != nil {
		scope.Close()
		return nil, err
	}
	result := &ActionSet{memory: scope}
	actions, truncated, err := enumerateActionsWithBudget(g, rules, limit, budget)
	if err != nil {
		result.Close()
		return nil, err
	}
	result.Actions, result.Truncated = actions, truncated
	return result, nil
}

// ActionKey returns a deterministic string key for ordering candidates.
func (c ActionCandidate) ActionKey() string {
	if c.Rule == nil {
		return ""
	}
	var sb strings.Builder
	sb.WriteString(c.Rule.Name)
	sb.WriteString(fmt.Sprintf("@%d[", c.Match.ClassID))
	keys := make([]string, 0, len(c.Match.Subst))
	for k := range c.Match.Subst {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for i, k := range keys {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(fmt.Sprintf("%s:%d", k, c.Match.Subst[k]))
	}
	binding := c.Match.Binding
	if binding == nil {
		binding = bindingFromSubst(c.Match.Subst)
	}
	keys = keys[:0]
	for k := range binding {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		sb.WriteString(fmt.Sprintf("|%s:%#v", k, binding[k]))
	}
	sb.WriteString("]")
	return sb.String()
}

// EnumerateActions returns up to the default limit of legal rewrite actions in
// deterministic order. Use EnumerateActionsChecked to detect errors or
// truncation.
func EnumerateActions(g *EGraph, rules []*Rewrite) []ActionCandidate {
	return EnumerateActionsBounded(g, rules, 10000)
}

// EnumerateActionsBounded deterministically enumerates at most limit matches.
// It returns nil when matching fails; use EnumerateActionsBoundedChecked to
// distinguish errors from an empty result or a truncated result.
func EnumerateActionsBounded(g *EGraph, rules []*Rewrite, limit int) []ActionCandidate {
	candidates, _, err := EnumerateActionsBoundedChecked(g, rules, limit)
	if err != nil {
		return nil
	}
	return candidates
}

// EnumerateActionsChecked enumerates actions with the default output limit and
// reports errors or truncation instead of presenting an incomplete set as
// complete.
func EnumerateActionsChecked(g *EGraph, rules []*Rewrite) ([]ActionCandidate, error) {
	candidates, truncated, err := EnumerateActionsBoundedChecked(g, rules, 10000)
	if err != nil {
		return nil, err
	}
	if truncated {
		return nil, ErrActionEnumerationLimit
	}
	return candidates, nil
}

// EnumerateActionsBoundedChecked deterministically enumerates at most limit
// matches. truncated is true when more matches exist than fit in the result.
func EnumerateActionsBoundedChecked(g *EGraph, rules []*Rewrite, limit int) ([]ActionCandidate, bool, error) {
	return enumerateActionsWithBudget(g, rules, limit, egraphcore.NewWorkBudget(100000))
}

func enumerateActionsWithBudget(g *EGraph, rules []*Rewrite, limit int, budget *egraphcore.WorkBudget) ([]ActionCandidate, bool, error) {
	if limit <= 0 {
		return nil, false, nil
	}
	if g == nil {
		return nil, false, fmt.Errorf("egraph: nil graph")
	}
	var candidates []ActionCandidate
	if budget.MemoryScope() != nil {
		view, err := matchingView(g, budget)
		if err != nil {
			return nil, false, err
		}
		g = view
	}
	truncated := false
	for _, rule := range rules {
		if truncated {
			break
		}
		if rule == nil {
			continue
		}
		probing := len(candidates) == limit
		remaining := 1
		if !probing {
			remaining = limit - len(candidates)
		}
		var matches []MatchMatch
		var err error
		if len(rule.Conditions) == 0 && len(rule.RelationConditions) == 0 {
			var limited bool
			matches, limited, err = rule.LHS.SearchBoundedWithBudget(g, remaining, budget)
			if err != nil {
				return nil, false, fmt.Errorf("egraph: enumerate rewrite %q: %w", rule.Name, err)
			}
			truncated = truncated || limited
		} else {
			matches, err = searchRewriteMatchesWithBudget(g, rule, budget)
			if err != nil {
				return nil, false, fmt.Errorf("egraph: enumerate rewrite %q: %w", rule.Name, err)
			}
			if len(matches) > remaining {
				matches = matches[:remaining]
				truncated = true
			}
		}
		if probing {
			if len(matches) != 0 {
				truncated = true
				break
			}
			continue
		}
		for _, m := range matches {
			candidate := ActionCandidate{
				Rule:  rule,
				Match: m,
			}
			var err error
			candidates, err = graphAppend(budget, candidates, candidate)
			if err != nil {
				return nil, false, err
			}
		}
	}

	// Compute no-op status once per candidate; sort comparisons must remain
	// pure and bounded rather than cloning the graph repeatedly.
	type ordered struct {
		candidate ActionCandidate
		noop      bool
		key       string
	}
	items, err := graphSlice[ordered](budget, len(candidates), len(candidates))
	if err != nil {
		return nil, false, err
	}
	for i, candidate := range candidates {
		noop := actionIsNoOpWithBudget(g, candidate, budget)
		if err := budget.MemoryError(); err != nil {
			return nil, false, err
		}
		key, err := actionKeyWithBudget(candidate, budget)
		if err != nil {
			return nil, false, err
		}
		items[i] = ordered{candidate, noop, key}
	}
	slices.SortStableFunc(items, func(a, b ordered) int {
		ni, nj := a.noop, b.noop
		if ni != nj {
			if ni {
				return 1
			}
			return -1
		}
		return cmp.Compare(a.key, b.key)
	})
	for i := range items {
		candidates[i] = items[i].candidate
	}
	return candidates, truncated, nil
}

func actionIsNoOp(g *EGraph, c ActionCandidate) bool {
	return actionIsNoOpWithBudget(g, c, nil)
}

func actionIsNoOpWithBudget(g *EGraph, c ActionCandidate, budget *egraphcore.WorkBudget) bool {
	if c.Rule == nil {
		return false
	}
	if len(c.Rule.RelationActions) != 0 || c.Rule.RHS == nil {
		return false
	}
	return patternIsExistingClassWithBudget(g, c.Rule.RHS, c.Match.Subst, c.Match.ClassID, budget)
}

func bindingFromSubst(subst Subst) egraphcore.Binding[Id] {
	binding := make(egraphcore.Binding[Id], len(subst))
	for name, id := range subst {
		binding[name] = egraphcore.Class(id)
	}
	return binding
}

func matchBindingsEqual(g *EGraph, left, right MatchMatch) bool {
	a, b := left.Binding, right.Binding
	if a == nil {
		a = bindingFromSubst(left.Subst)
	}
	if b == nil {
		b = bindingFromSubst(right.Subst)
	}
	if len(a) != len(b) {
		return false
	}
	for name, value := range a {
		other, ok := b[name]
		if !ok {
			return false
		}
		if value.Kind == egraphcore.ClassValue {
			value.Class = g.Find(value.Class)
		}
		if other.Kind == egraphcore.ClassValue {
			other.Class = g.Find(other.Class)
		}
		if value != other {
			return false
		}
	}
	return true
}

// TransitionResult details the outcome of applying an action to a search state.
type TransitionResult struct {
	State        *SearchState
	Merged       bool
	FactChanges  int
	Subsumptions int
	Action       Action
	NoOp         bool
}

// ApplyAction applies a single ActionCandidate to the search state, returning a new branched SearchState
// and indicating whether a non-trivial union occurred.
func (s *SearchState) ApplyAction(candidate ActionCandidate) (*TransitionResult, error) {
	return s.applyAction(candidate, nil, nil)
}

// A supplied nextState is already detached and is also the receiver during
// clone-budgeted transitions. Ordinary ApplyAction clones after validation.
func (s *SearchState) applyAction(candidate ActionCandidate, nextState *SearchState, cloneBudget *egraphcore.WorkBudget) (*TransitionResult, error) {
	if s == nil || s.Graph == nil {
		return nil, fmt.Errorf("egraph: nil search state or graph")
	}
	if candidate.Rule == nil {
		return nil, fmt.Errorf("egraph: nil rewrite rule in candidate")
	}
	target := s.Graph.Find(candidate.Match.ClassID)
	if s.Graph.class(target) == nil {
		return nil, fmt.Errorf("egraph: match class %d is missing", candidate.Match.ClassID)
	}
	transitionBudget := egraphcore.NewWorkBudget(100000)
	valid := false
	matches, matchErr := searchRewriteMatchesWithBudget(s.Graph, candidate.Rule, transitionBudget)
	if matchErr != nil {
		return nil, matchErr
	}
	for _, match := range matches {
		if s.Graph.Find(match.ClassID) == target && substEqual(match.Subst, candidate.Match.Subst) && matchBindingsEqual(s.Graph, match, candidate.Match) {
			valid = true
			break
		}
	}
	if !valid {
		return nil, fmt.Errorf("egraph: candidate match is stale or forged")
	}
	if err := checkMandatoryTransition(s, candidate.Rule, target, candidate.Match.Subst); err != nil {
		return nil, err
	}

	// Create branched graph
	resolved, err := resolveFactActions(s.Graph, candidate.Rule, []MatchMatch{candidate.Match}, transitionBudget)
	if err != nil {
		return nil, err
	}
	if nextState == nil {
		nextState = s.Clone()
	}
	g := nextState.Graph
	// The in-place executor must freeze the pre-transition fingerprint before
	// RHS instantiation changes its working graph. Ordinary execution retains
	// the original timing and source graph of this computation.
	source := ""
	if nextState == s {
		source = classFingerprint(s.Graph, candidate.Match.ClassID)
	}

	canonTarget := g.Find(candidate.Match.ClassID)
	canonRHS := canonTarget
	if candidate.Rule.RHS != nil {
		rhsID := candidate.Rule.RHS.Instantiate(g, candidate.Match.Subst)
		canonRHS = g.Find(rhsID)
	}

	if nextState != s {
		source = classFingerprint(s.Graph, candidate.Match.ClassID)
	}
	action := Action{
		RuleName:  candidate.Rule.Name,
		ClassID:   canonTarget,
		RHSNode:   fmt.Sprintf("%d", canonRHS),
		StepIndex: len(s.Derivation),
		Subst:     candidate.Match.Subst.Clone(),
		Binding:   cloneBinding(candidate.Match.Binding),
		Source:    source,
		RuleID:    candidate.Rule.Identity(),
	}

	if canonTarget == canonRHS && len(resolved) == 0 && !candidate.Rule.subsumesLHS {
		// No-op transition: already in same e-class
		return transitionWithBudget(TransitionResult{
			State:  nextState,
			Merged: false,
			Action: action,
			NoOp:   true,
		}, cloneBudget)
	}

	merged := false
	if canonTarget != canonRHS {
		_, merged, err = g.UnionChecked(canonTarget, canonRHS)
		if err != nil {
			return nil, fmt.Errorf("egraph: apply rewrite %q union: %w", candidate.Rule.Name, err)
		}
	}
	if merged {
		if err := g.RebuildChecked(); err != nil {
			return nil, fmt.Errorf("egraph: apply rewrite %q rebuild: %w", candidate.Rule.Name, err)
		}
	}
	factChanges, err := g.Facts().ApplyActions(resolved, g.Find, transitionBudget)
	if err != nil {
		return nil, fmt.Errorf("egraph: apply relation actions: %w", err)
	}
	subsumptions, err := subsumeMatchedNodes(g, candidate.Rule, []MatchMatch{candidate.Match}, transitionBudget)
	if err != nil {
		return nil, fmt.Errorf("egraph: apply rewrite %q subsumption: %w", candidate.Rule.Name, err)
	}
	if !merged && factChanges == 0 && subsumptions == 0 {
		return transitionWithBudget(TransitionResult{State: nextState, Merged: false, Action: action, NoOp: true}, cloneBudget)
	}

	// Remap roots
	for k, v := range nextState.Roots {
		nextState.Roots[k] = g.Find(v)
	}

	nextState.Derivation = append(nextState.Derivation, action)
	nextState.Depth++

	return transitionWithBudget(TransitionResult{
		State:        nextState,
		Merged:       merged,
		FactChanges:  factChanges,
		Subsumptions: subsumptions,
		Action:       action,
		NoOp:         !merged && factChanges == 0 && subsumptions == 0,
	}, cloneBudget)
}

func checkMandatoryTransition(s *SearchState, rule *Rewrite, target Id, subst Subst) error {
	if rule == nil || !isMandatoryRewrite(rule) {
		return nil
	}
	name := rule.Name
	meta, ok := metadataFor(s, target)
	if !ok {
		return ErrMissingTypedMetadata
	}
	if meta.Effect != cudair.EffectReadOnly {
		return ErrEffectBoundaryViolation
	}
	if strings.Contains(name, "vectorize_moe") {
		if meta.Alignment < 16 || meta.Alignment%16 != 0 {
			return ErrUnalignedVectorAccess
		}
		tokens, err := operandMetadata(s, subst, "?tokens")
		if err != nil {
			return err
		}
		indices, err := operandMetadata(s, subst, "?indices")
		if err != nil {
			return err
		}
		if tokens.Effect != cudair.EffectReadOnly || indices.Effect != cudair.EffectReadOnly {
			return ErrEffectBoundaryViolation
		}
		if !integerType(indices.DType) {
			return fmt.Errorf("%w: gather indices must be integer", ErrIncompatibleDType)
		}
		// An aligned result does not establish alignment of vector loads from
		// the token rows. Require both the base and every row stride to preserve
		// the 16-byte source alignment; other layouts need a separate proof.
		if tokens.Alignment < 16 || tokens.Alignment%16 != 0 {
			return ErrUnalignedVectorAccess
		}
		if tokens.Shape.Rank() != 2 || tokens.Layout != cudashape.LayoutRowMajor {
			return ErrInvalidRewriteProof
		}
		if tokens.DType < cudair.Float32 || tokens.DType > cudair.Int8 {
			return ErrIncompatibleDType
		}
		if tokens.DType != meta.DType {
			return ErrIncompatibleDType
		}
		if indices.Shape.Rank() != 1 || indices.Layout != cudashape.LayoutRowMajor ||
			meta.Shape.Rank() != 2 || meta.Layout != cudashape.LayoutRowMajor ||
			meta.Shape.Dim(0) != indices.Shape.Dim(0) || meta.Shape.Dim(1) != tokens.Shape.Dim(1) {
			return ErrInvalidRewriteProof
		}
		width := tokens.Shape.Dim(1)
		if width < 1 || width%(16/tokens.DType.SizeBytes()) != 0 {
			return ErrUnalignedVectorAccess
		}
		return nil
	}
	if strings.Contains(name, "prefill_group") {
		if meta.Shape.Rank() == 0 || meta.Shape.Dim(meta.Shape.Rank()-1) != 256 || !meta.LoopInvariant || meta.NumericalPolicy == "" {
			return ErrInvalidRewriteProof
		}
		return requireReadonlyOperands(s, subst, "?x", "?w")
	}
	if strings.Contains(name, "hoist_") {
		if !meta.LoopInvariant || meta.NumericalPolicy == "" {
			return ErrInvalidRewriteProof
		}
		return requireReadonlyOperands(s, subst, "?w", "?scales", "?x")
	}
	if strings.Contains(name, "fuse_rmsnorm") {
		if !meta.LoopInvariant || meta.NumericalPolicy != "rmsnorm-q8" {
			return ErrInvalidRewriteProof
		}
		return requireFloatReadonlyOperands(s, subst, "?x", "?w_norm")
	}
	if strings.Contains(name, "fuse_swiglu") {
		if !meta.LoopInvariant || meta.NumericalPolicy != "swiglu-q8_1" {
			return ErrInvalidRewriteProof
		}
		return requireFloatReadonlyOperands(s, subst, "?x", "?w_g", "?w_u")
	}
	return ErrInvalidRewriteProof
}

func metadataFor(s *SearchState, id Id) (TypedMetadata, bool) {
	m, ok := s.Metadata[s.Graph.Find(id)]
	return m, ok
}

func operandMetadata(s *SearchState, subst Subst, variable string) (TypedMetadata, error) {
	id, ok := subst[variable]
	if !ok {
		return TypedMetadata{}, fmt.Errorf("%w: operand %s", ErrMissingTypedMetadata, variable)
	}
	m, ok := metadataFor(s, id)
	if !ok {
		return TypedMetadata{}, fmt.Errorf("%w: operand %s", ErrMissingTypedMetadata, variable)
	}
	return m, nil
}

func requireReadonlyOperands(s *SearchState, subst Subst, variables ...string) error {
	for _, variable := range variables {
		m, err := operandMetadata(s, subst, variable)
		if err != nil {
			return err
		}
		if m.Effect != cudair.EffectReadOnly {
			return ErrEffectBoundaryViolation
		}
	}
	return nil
}

func requireFloatReadonlyOperands(s *SearchState, subst Subst, variables ...string) error {
	for _, variable := range variables {
		m, err := operandMetadata(s, subst, variable)
		if err != nil {
			return err
		}
		if m.Effect != cudair.EffectReadOnly {
			return ErrEffectBoundaryViolation
		}
		if !floatType(m.DType) {
			return fmt.Errorf("%w: operand %s must be floating point", ErrIncompatibleDType, variable)
		}
	}
	return nil
}

func floatType(t cudair.DataType) bool {
	return t == cudair.Float16 || t == cudair.BFloat16 || t == cudair.Float32
}
func integerType(t cudair.DataType) bool {
	return t == cudair.Int8 || t == cudair.Int32 || t == cudair.Uint8
}

// ReplaySearchState replays a derivation from a typed starting state. The
// older ReplayDerivation wrapper intentionally has no metadata and therefore
// cannot replay mandatory typed rules.
func ReplaySearchState(initial *SearchState, rules []*Rewrite, trace []Action) (*SearchState, error) {
	if initial == nil || initial.Graph == nil {
		return nil, fmt.Errorf("egraph: nil replay state")
	}
	state := initial.Clone()
	ruleMap := make(map[string]*Rewrite, len(rules))
	for _, r := range rules {
		if r != nil {
			ruleMap[r.Name] = r
		}
	}
	for step, act := range trace {
		rule, ok := ruleMap[act.RuleName]
		if !ok {
			return nil, fmt.Errorf("egraph: replay step %d: rule %q not found", step, act.RuleName)
		}
		if act.RuleID != "" && act.RuleID != rule.Identity() {
			return nil, fmt.Errorf("egraph: replay step %d: rule identity mismatch", step)
		}
		candidates, truncated, err := EnumerateActionsBoundedChecked(state.Graph, []*Rewrite{rule}, 10000)
		if err != nil {
			return nil, fmt.Errorf("egraph: replay step %d: enumerate rule %q: %w", step, act.RuleName, err)
		}
		if truncated {
			return nil, fmt.Errorf("egraph: replay step %d: rule %q: %w", step, act.RuleName, ErrActionEnumerationLimit)
		}
		var matched *ActionCandidate
		for i := range candidates {
			c := &candidates[i]
			if state.Graph.Find(c.Match.ClassID) == state.Graph.Find(act.ClassID) && classFingerprint(state.Graph, c.Match.ClassID) == act.Source && substEqual(c.Match.Subst, act.Subst) && (act.Binding == nil || matchBindingsEqual(state.Graph, c.Match, MatchMatch{Subst: act.Subst, Binding: act.Binding})) {
				matched = c
				break
			}
		}
		if matched == nil {
			return nil, fmt.Errorf("egraph: replay step %d: no valid match for rule %q", step, act.RuleName)
		}
		res, err := state.ApplyAction(*matched)
		if err != nil {
			return nil, fmt.Errorf("egraph: replay step %d: apply failed: %w", step, err)
		}
		state = res.State
	}
	return state, nil
}

// ReplayDerivation applies a recorded sequence of actions onto an initial EGraph and root set,
// validating that the identical end-state is reached deterministically.
func ReplayDerivation(initial *EGraph, initialRoots map[string]Id, rules []*Rewrite, trace []Action) (*SearchState, error) {
	if initial == nil {
		return nil, fmt.Errorf("egraph: nil initial graph")
	}
	for _, act := range trace {
		if isMandatoryRule(act.RuleName) {
			return nil, ErrMissingTypedMetadata
		}
	}
	ruleMap := make(map[string]*Rewrite, len(rules))
	for _, r := range rules {
		if r != nil {
			ruleMap[r.Name] = r
		}
	}

	state := NewSearchState(initial, initialRoots)

	for step, act := range trace {
		rule, ok := ruleMap[act.RuleName]
		if !ok {
			return nil, fmt.Errorf("egraph: replay step %d: rule %q not found", step, act.RuleName)
		}
		if isMandatoryRewrite(rule) {
			return nil, ErrMissingTypedMetadata
		}
		if act.RuleID != "" && act.RuleID != rule.Identity() {
			return nil, fmt.Errorf("egraph: replay step %d: rule identity mismatch", step)
		}

		// Enumerate actions in current state and find the matching action key
		candidates, truncated, err := EnumerateActionsBoundedChecked(state.Graph, []*Rewrite{rule}, 10000)
		if err != nil {
			return nil, fmt.Errorf("egraph: replay step %d: enumerate rule %q: %w", step, act.RuleName, err)
		}
		if truncated {
			return nil, fmt.Errorf("egraph: replay step %d: rule %q: %w", step, act.RuleName, ErrActionEnumerationLimit)
		}
		var matchedCand *ActionCandidate
		for i := range candidates {
			c := &candidates[i]
			if state.Graph.Find(c.Match.ClassID) == state.Graph.Find(act.ClassID) && classFingerprint(state.Graph, c.Match.ClassID) == act.Source && substEqual(c.Match.Subst, act.Subst) && (act.Binding == nil || matchBindingsEqual(state.Graph, c.Match, MatchMatch{Subst: act.Subst, Binding: act.Binding})) {
				matchedCand = c
				break
			}
		}

		if matchedCand == nil {
			return nil, fmt.Errorf("egraph: replay step %d: no valid match for rule %q on class %d", step, act.RuleName, act.ClassID)
		}

		res, err := state.ApplyAction(*matchedCand)
		if err != nil {
			return nil, fmt.Errorf("egraph: replay step %d: apply failed: %w", step, err)
		}
		state = res.State
	}

	return state, nil
}

func substEqual(a, b Subst) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}
