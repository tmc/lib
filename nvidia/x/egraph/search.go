package egraph

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// Action represents a single, deterministic rewrite transition applied to a specific e-class match.
type Action struct {
	RuleName  string // Identifier of the rewrite rule
	ClassID   Id     // Target e-class ID matched
	RHSNode   string // Serialized representation of RHS target instantiation
	StepIndex int    // Monotonic step index in search derivation
	Subst     Subst  // Exact variable-to-class bindings used by the match
	Source    string // Fingerprint of the matched class before transition
	RuleID    string // Full rule identity used for replay validation
}

// Key returns a canonical string representation of the action for deduplication and sorting.
func (a Action) Key() string {
	return fmt.Sprintf("%s@%d[%s]->%s", a.RuleName, a.ClassID, a.Source, a.RHSNode)
}

func classFingerprint(g *EGraph, id Id) string {
	c := g.Class(id)
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
	copiedRoots := make(map[string]Id, len(s.Roots))
	for k, v := range s.Roots {
		copiedRoots[k] = s.Graph.Find(v)
	}
	copiedDerivation := make([]Action, len(s.Derivation))
	for i, action := range s.Derivation {
		copiedDerivation[i] = action
		copiedDerivation[i].Subst = action.Subst.Clone()
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

// SetMetadata attaches the checked semantic contract used by mandatory rules.
func (s *SearchState) SetMetadata(id Id, meta TypedMetadata) {
	if s.Metadata == nil {
		s.Metadata = make(map[Id]TypedMetadata)
	}
	s.Metadata[s.Graph.Find(id)] = meta.Clone()
}

// CanonicalRoot returns the current canonical ID of the named root.
func (s *SearchState) CanonicalRoot(name string) (Id, bool) {
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

// ActionKey returns a deterministic string key for ordering candidates.
func (c ActionCandidate) ActionKey() string {
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
	sb.WriteString("]")
	return sb.String()
}

// EnumerateActions finds all legal applicable rewrite transitions across the rules,
// returning them in deterministic sorted order.
func EnumerateActions(g *EGraph, rules []*Rewrite) []ActionCandidate {
	return EnumerateActionsBounded(g, rules, 10000)
}

// EnumerateActionsBounded deterministically enumerates at most limit matches.
func EnumerateActionsBounded(g *EGraph, rules []*Rewrite, limit int) []ActionCandidate {
	if limit <= 0 {
		return nil
	}
	var candidates []ActionCandidate

	for _, rule := range rules {
		if len(candidates) == limit {
			break
		}
		matches, _ := rule.LHS.SearchBounded(g, limit-len(candidates))
		for _, m := range matches {
			candidate := ActionCandidate{
				Rule:  rule,
				Match: m,
			}
			candidates = append(candidates, candidate)
		}
	}

	// Compute no-op status once per candidate; sort comparisons must remain
	// pure and bounded rather than cloning the graph repeatedly.
	type ordered struct {
		candidate ActionCandidate
		noop      bool
	}
	items := make([]ordered, len(candidates))
	for i, candidate := range candidates {
		items[i] = ordered{candidate, actionIsNoOp(g, candidate)}
	}
	sort.SliceStable(items, func(i, j int) bool {
		ni, nj := items[i].noop, items[j].noop
		if ni != nj {
			return !ni
		}
		return items[i].candidate.ActionKey() < items[j].candidate.ActionKey()
	})
	for i := range items {
		candidates[i] = items[i].candidate
	}
	return candidates
}

func actionIsNoOp(g *EGraph, c ActionCandidate) bool {
	if c.Rule == nil {
		return false
	}
	return patternIsExistingClass(g, c.Rule.RHS, c.Match.Subst, c.Match.ClassID)
}

// TransitionResult details the outcome of applying an action to a search state.
type TransitionResult struct {
	State  *SearchState
	Merged bool
	Action Action
	NoOp   bool
}

// ApplyAction applies a single ActionCandidate to the search state, returning a new branched SearchState
// and indicating whether a non-trivial union occurred.
func (s *SearchState) ApplyAction(candidate ActionCandidate) (*TransitionResult, error) {
	if candidate.Rule == nil {
		return nil, fmt.Errorf("egraph: nil rewrite rule in candidate")
	}
	target := s.Graph.Find(candidate.Match.ClassID)
	if s.Graph.Class(target) == nil {
		return nil, fmt.Errorf("egraph: match class %d is missing", candidate.Match.ClassID)
	}
	valid := false
	substs, matchErr := candidate.Rule.LHS.MatchEClassBounded(s.Graph, target, make(Subst), 100000)
	if matchErr != nil {
		return nil, matchErr
	}
	for _, subst := range substs {
		if substEqual(subst, candidate.Match.Subst) {
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
	nextState := s.Clone()
	g := nextState.Graph

	// Instantiate RHS
	rhsID := candidate.Rule.RHS.Instantiate(g, candidate.Match.Subst)
	canonTarget := g.Find(candidate.Match.ClassID)
	canonRHS := g.Find(rhsID)

	action := Action{
		RuleName:  candidate.Rule.Name,
		ClassID:   canonTarget,
		RHSNode:   fmt.Sprintf("%d", canonRHS),
		StepIndex: len(s.Derivation),
		Subst:     candidate.Match.Subst.Clone(),
		Source:    classFingerprint(s.Graph, candidate.Match.ClassID),
		RuleID:    candidate.Rule.Identity(),
	}

	if canonTarget == canonRHS {
		// No-op transition: already in same e-class
		return &TransitionResult{
			State:  nextState,
			Merged: false,
			Action: action,
			NoOp:   true,
		}, nil
	}

	_, merged := g.Union(canonTarget, canonRHS)
	if merged {
		g.Rebuild()
	}

	// Remap roots
	for k, v := range nextState.Roots {
		nextState.Roots[k] = g.Find(v)
	}

	nextState.Derivation = append(nextState.Derivation, action)
	nextState.Depth++

	return &TransitionResult{
		State:  nextState,
		Merged: merged,
		Action: action,
		NoOp:   !merged,
	}, nil
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
		if meta.Alignment < 16 {
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
		candidates := EnumerateActions(state.Graph, []*Rewrite{rule})
		var matched *ActionCandidate
		for i := range candidates {
			c := &candidates[i]
			if state.Graph.Find(c.Match.ClassID) == state.Graph.Find(act.ClassID) && classFingerprint(state.Graph, c.Match.ClassID) == act.Source && substEqual(c.Match.Subst, act.Subst) {
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
	for _, act := range trace {
		if isMandatoryRule(act.RuleName) {
			return nil, ErrMissingTypedMetadata
		}
	}
	ruleMap := make(map[string]*Rewrite, len(rules))
	for _, r := range rules {
		ruleMap[r.Name] = r
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
		candidates := EnumerateActions(state.Graph, []*Rewrite{rule})
		var matchedCand *ActionCandidate
		for i := range candidates {
			c := &candidates[i]
			if state.Graph.Find(c.Match.ClassID) == state.Graph.Find(act.ClassID) && classFingerprint(state.Graph, c.Match.ClassID) == act.Source && substEqual(c.Match.Subst, act.Subst) {
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
