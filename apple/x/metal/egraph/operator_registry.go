package egraph

import (
	"fmt"
	"sort"
	"sync"
)

// MSLGenerator generates the MSL source for one extracted operator expression.
// The callback must treat expr and its children as immutable. A registry may
// invoke callbacks concurrently; callbacks must be safe for concurrent use.
type MSLGenerator func(expr *Expr, opts CodegenOptions) (string, error)

// RewriteFactory creates one rewrite owned by an operator definition. The
// registry calls factories outside its lock; each call should return fresh
// rewrite state because Rewrite has mutable configuration methods.
type RewriteFactory func() (*Rewrite, error)

// OperatorRewrite binds a rewrite factory to the saturation phase in which
// the rewrite runs. Order controls phase order; rewrites within a phase are
// ordered by owning operator name and then declaration order.
type OperatorRewrite struct {
	Phase   string
	Order   int
	Factory RewriteFactory
}

// OperatorSchedule describes a typed schedule over rewrite phases registered
// with an OperatorRegistry. Run nodes name a phase; other nodes compose runs
// with fixed-point, sequence, and bounded-repeat behavior.
type OperatorSchedule struct {
	Kind        ScheduleNodeKind
	Name        string
	Phase       string
	Children    []*OperatorSchedule
	Repetitions int
}

// RunOperators creates a one-pass schedule node for a registered rewrite phase.
func RunOperators(phase string) *OperatorSchedule {
	return &OperatorSchedule{Kind: ScheduleRunKind, Name: phase, Phase: phase}
}

// FixedPointOperators repeats child until it makes no reported changes.
func FixedPointOperators(name string, child *OperatorSchedule) *OperatorSchedule {
	return &OperatorSchedule{Kind: ScheduleSaturateKind, Name: name, Children: []*OperatorSchedule{child}}
}

// SequenceOperators creates a schedule that runs its children in order.
func SequenceOperators(children ...*OperatorSchedule) *OperatorSchedule {
	return &OperatorSchedule{Kind: ScheduleSequenceKind, Children: children}
}

// RepeatOperators runs child up to count times, stopping early when a pass
// reports no changes.
func RepeatOperators(count int, child *OperatorSchedule) *OperatorSchedule {
	return &OperatorSchedule{Kind: ScheduleRepeatKind, Repetitions: count, Children: []*OperatorSchedule{child}}
}

// OperatorDefinition describes the signatures, optional MSL generator, and
// operator-owned rewrite factories with their saturation phases. The
// registry copies all supplied slices when the definition is registered.
type OperatorDefinition struct {
	Signatures []OperatorSignature
	MSL        MSLGenerator
	Rewrites   []OperatorRewrite
}

// OperatorRegistry maps operator names to typed signatures, code generators,
// and phase-aware rewrite factories. Registration and lookup are safe to use
// concurrently.
// Callback state remains the caller's responsibility and must be immutable or
// concurrency-safe.
type OperatorRegistry struct {
	mu          sync.RWMutex
	definitions map[string]OperatorDefinition
}

// NewOperatorRegistry returns an empty operator registry.
func NewOperatorRegistry() *OperatorRegistry {
	return &OperatorRegistry{definitions: make(map[string]OperatorDefinition)}
}

// Register adds an operator definition. Re-registering a name is an error.
// Signatures must be non-empty and mutually unambiguous. The registry copies
// signature slice state, so callers may reuse or modify their input slices.
func (r *OperatorRegistry) Register(name string, definition OperatorDefinition) error {
	if r == nil {
		return fmt.Errorf("egraph: nil operator registry")
	}
	if name == "" {
		return fmt.Errorf("egraph: operator name must be non-empty")
	}
	if len(definition.Signatures) == 0 {
		return fmt.Errorf("egraph: operator %q must have at least one signature", name)
	}

	copyDef := OperatorDefinition{
		MSL:        definition.MSL,
		Signatures: make([]OperatorSignature, len(definition.Signatures)),
		Rewrites:   append([]OperatorRewrite(nil), definition.Rewrites...),
	}
	for i, rewrite := range copyDef.Rewrites {
		if rewrite.Phase == "" {
			return fmt.Errorf("egraph: operator %q rewrite %d has empty saturation phase", name, i)
		}
		if rewrite.Factory == nil {
			return fmt.Errorf("egraph: operator %q rewrite factory %d is nil", name, i)
		}
	}
	checker := New()
	for i, signature := range definition.Signatures {
		if signature.Result == "" {
			return fmt.Errorf("egraph: operator %q signature %d has empty result sort", name, i)
		}
		copyDef.Signatures[i] = OperatorSignature{
			Result:   signature.Result,
			Children: append([]string(nil), signature.Children...),
		}
		if err := checker.RegisterOperator(name, copyDef.Signatures[i]); err != nil {
			return fmt.Errorf("egraph: invalid signature %d for operator %q: %w", i, name, err)
		}
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.definitions == nil {
		r.definitions = make(map[string]OperatorDefinition)
	}
	if _, exists := r.definitions[name]; exists {
		return fmt.Errorf("egraph: operator %q is already registered", name)
	}
	r.definitions[name] = copyDef
	return nil
}

// AddRewrites appends operator-owned rewrites to an existing definition.
// Rewrite slices are copied, so callers may reuse or modify their input slice.
// Phase order consistency is checked when NewSaturationSchedule is built.
func (r *OperatorRegistry) AddRewrites(name string, rewrites ...OperatorRewrite) error {
	if r == nil {
		return fmt.Errorf("egraph: nil operator registry")
	}
	if name == "" {
		return fmt.Errorf("egraph: operator name must be non-empty")
	}
	if len(rewrites) == 0 {
		return fmt.Errorf("egraph: operator %q must add at least one rewrite", name)
	}
	for i, rewrite := range rewrites {
		if rewrite.Phase == "" {
			return fmt.Errorf("egraph: operator %q rewrite %d has empty saturation phase", name, i)
		}
		if rewrite.Factory == nil {
			return fmt.Errorf("egraph: operator %q rewrite factory %d is nil", name, i)
		}
	}

	added := append([]OperatorRewrite(nil), rewrites...)
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.definitions == nil {
		return fmt.Errorf("egraph: unknown operator %q", name)
	}
	definition, ok := r.definitions[name]
	if !ok {
		return fmt.Errorf("egraph: unknown operator %q", name)
	}
	definition.Rewrites = append(append([]OperatorRewrite(nil), definition.Rewrites...), added...)
	r.definitions[name] = definition
	return nil
}

// NewSaturationSchedule calls registered operator-owned rewrite factories
// and returns detached phase and rule slices. Phases are ordered by Order,
// then phase name; rules within each phase are ordered by operator name and
// declaration order. Factories must return fresh rewrites.
func (r *OperatorRegistry) NewSaturationSchedule() ([]SaturationPhase, error) {
	if r == nil {
		return nil, fmt.Errorf("egraph: nil operator registry")
	}
	r.mu.RLock()
	names := make([]string, 0, len(r.definitions))
	definitions := make(map[string]OperatorDefinition, len(r.definitions))
	for name, definition := range r.definitions {
		names = append(names, name)
		definitions[name] = definition
	}
	r.mu.RUnlock()
	sort.Strings(names)
	type scheduledRewrite struct {
		owner string
		index int
		rule  *Rewrite
	}
	byPhase := make(map[string][]scheduledRewrite)
	phaseOrder := make(map[string]int)
	for _, name := range names {
		for index, definition := range definitions[name].Rewrites {
			rule, err := definition.Factory()
			if err != nil {
				return nil, fmt.Errorf("egraph: create rewrite %d for operator %q: %w", index, name, err)
			}
			if rule == nil {
				return nil, fmt.Errorf("egraph: rewrite factory %d for operator %q returned nil", index, name)
			}
			if err := rule.validate(); err != nil {
				return nil, fmt.Errorf("egraph: invalid rewrite %d for operator %q: %w", index, name, err)
			}
			if old, ok := phaseOrder[definition.Phase]; ok && old != definition.Order {
				return nil, fmt.Errorf("egraph: saturation phase %q has conflicting order values %d and %d", definition.Phase, old, definition.Order)
			}
			phaseOrder[definition.Phase] = definition.Order
			byPhase[definition.Phase] = append(byPhase[definition.Phase], scheduledRewrite{
				owner: name, index: index, rule: rule,
			})
		}
	}
	phases := make([]SaturationPhase, 0, len(byPhase))
	for name, rewrites := range byPhase {
		sort.SliceStable(rewrites, func(i, j int) bool {
			if rewrites[i].owner != rewrites[j].owner {
				return rewrites[i].owner < rewrites[j].owner
			}
			return rewrites[i].index < rewrites[j].index
		})
		rules := make([]*Rewrite, len(rewrites))
		for i := range rewrites {
			rules[i] = rewrites[i].rule
		}
		phases = append(phases, SaturationPhase{Name: name, Rules: rules})
	}
	sort.Slice(phases, func(i, j int) bool {
		if phaseOrder[phases[i].Name] != phaseOrder[phases[j].Name] {
			return phaseOrder[phases[i].Name] < phaseOrder[phases[j].Name]
		}
		return phases[i].Name < phases[j].Name
	})
	return phases, nil
}

// NewRewriteSchedule resolves phase references in spec to fresh rewrite rules
// and returns a detached executable schedule. A phase may appear more than
// once in spec; its factories are called once and the resulting rules are
// reused at each reference.
func (r *OperatorRegistry) NewRewriteSchedule(spec *OperatorSchedule) (*RewriteSchedule, error) {
	phases, err := r.NewSaturationSchedule()
	if err != nil {
		return nil, err
	}
	byName := make(map[string][]*Rewrite, len(phases))
	for _, phase := range phases {
		byName[phase.Name] = phase.Rules
	}
	active := make(map[*OperatorSchedule]bool)
	var resolve func(*OperatorSchedule) (*RewriteSchedule, error)
	resolve = func(node *OperatorSchedule) (*RewriteSchedule, error) {
		if node == nil {
			return nil, fmt.Errorf("egraph: nil operator schedule node")
		}
		if active[node] {
			return nil, fmt.Errorf("egraph: cyclic operator schedule")
		}
		active[node] = true
		defer delete(active, node)
		children := make([]*RewriteSchedule, len(node.Children))
		for i, child := range node.Children {
			resolved, err := resolve(child)
			if err != nil {
				return nil, fmt.Errorf("egraph: operator schedule child %d: %w", i, err)
			}
			children[i] = resolved
		}
		var resolved *RewriteSchedule
		switch node.Kind {
		case ScheduleRunKind:
			if node.Phase == "" || len(children) != 0 || node.Repetitions != 0 {
				return nil, fmt.Errorf("egraph: operator run requires a phase and no children or repetitions")
			}
			rules, ok := byName[node.Phase]
			if !ok {
				return nil, fmt.Errorf("egraph: unknown operator rewrite phase %q", node.Phase)
			}
			name := node.Name
			if name == "" {
				name = node.Phase
			}
			resolved = NewRunSchedule(name, rules...)
		case ScheduleSaturateKind:
			if node.Phase != "" || len(children) != 1 || node.Repetitions != 0 {
				return nil, fmt.Errorf("egraph: operator fixed-point schedule requires one child")
			}
			resolved = NewFixedPointSchedule(node.Name, children[0])
		case ScheduleSequenceKind:
			if node.Phase != "" || node.Repetitions != 0 {
				return nil, fmt.Errorf("egraph: operator sequence cannot name a phase or repetition count")
			}
			resolved = NewSequenceSchedule(children...)
		case ScheduleRepeatKind:
			if node.Phase != "" || len(children) != 1 || node.Repetitions < 0 {
				return nil, fmt.Errorf("egraph: operator repeat requires one child and a non-negative count")
			}
			resolved = NewRepeatSchedule(node.Repetitions, children[0])
		default:
			return nil, fmt.Errorf("egraph: unknown operator schedule kind %d", node.Kind)
		}
		if node.Name != "" {
			resolved.Name = node.Name
		}
		return resolved, nil
	}
	schedule, err := resolve(spec)
	if err != nil {
		return nil, err
	}
	if err := validateSchedule(schedule); err != nil {
		return nil, fmt.Errorf("egraph: invalid operator schedule: %w", err)
	}
	return schedule, nil
}

// NewSaturationRewriteSchedule converts the registry's ordered phases into a
// composable schedule. Each phase reaches a fixed point before the next phase,
// and the ordered phase sequence repeats until an entire cycle makes no
// changes, matching SaturateSchedule's legacy behavior.
func (r *OperatorRegistry) NewSaturationRewriteSchedule() (*RewriteSchedule, error) {
	phases, err := r.NewSaturationSchedule()
	if err != nil {
		return nil, err
	}
	phaseSchedules := make([]*RewriteSchedule, len(phases))
	for i, phase := range phases {
		phaseSchedules[i] = NewFixedPointSchedule(phase.Name, NewRunSchedule(phase.Name, phase.Rules...))
	}
	schedule := NewFixedPointSchedule("operator-saturation", NewSequenceSchedule(phaseSchedules...))
	if err := validateSchedule(schedule); err != nil {
		return nil, fmt.Errorf("egraph: invalid registry saturation schedule: %w", err)
	}
	return schedule, nil
}

// NewEGraph creates a strict e-graph with every registered operator signature.
// Later registrations do not alter the returned graph.
func (r *OperatorRegistry) NewEGraph() (*EGraph, error) {
	return r.NewEGraphWithAnalysis(nil)
}

// NewEGraphWithAnalysis creates a strict e-graph with every registered
// operator signature and the supplied e-class analysis. Later registrations
// do not alter the returned graph.
func (r *OperatorRegistry) NewEGraphWithAnalysis(analysis Analysis) (*EGraph, error) {
	if r == nil {
		return nil, fmt.Errorf("egraph: nil operator registry")
	}
	g := NewWithAnalysis(analysis)
	if err := r.RegisterOperators(g); err != nil {
		return nil, err
	}
	return g, nil
}

// RegisterOperators installs all registered signatures into an empty graph.
// It is useful when a graph supplies a specialized e-class analysis.
func (r *OperatorRegistry) RegisterOperators(g *EGraph) error {
	if r == nil {
		return fmt.Errorf("egraph: nil operator registry")
	}
	if g == nil {
		return fmt.Errorf("egraph: nil target graph")
	}
	if g.store.ClassCount() != 0 {
		return fmt.Errorf("egraph: target graph must be empty")
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.definitions))
	for name := range r.definitions {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		definition := r.definitions[name]
		for _, signature := range definition.Signatures {
			if err := g.RegisterOperator(name, signature); err != nil {
				return fmt.Errorf("egraph: register operator %q: %w", name, err)
			}
		}
	}
	return g.RequireOperatorSignatures()
}

func (r *OperatorRegistry) lookup(name string) (OperatorDefinition, bool) {
	if r == nil {
		return OperatorDefinition{}, false
	}
	r.mu.RLock()
	definition, ok := r.definitions[name]
	r.mu.RUnlock()
	return definition, ok
}
