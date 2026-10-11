package egraph

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/egraphcore"
)

// ErrAnalysisWorkLimit reports that analysis propagation exceeded its
// per-graph callback evaluation limit.
var ErrAnalysisWorkLimit = errors.New(analysisWorkLimitMessage)

// AnalysisWorkLimitError reports the configured callback evaluation limit.
type AnalysisWorkLimitError struct {
	// Limit is the maximum Make and Merge callback evaluations for the operation.
	Limit int
}

func (e *AnalysisWorkLimitError) Error() string {
	if e == nil || e.Limit <= 0 {
		return analysisWorkLimitMessage
	}
	return fmt.Sprintf("%s (%d)", analysisWorkLimitMessage, e.Limit)
}

func (e *AnalysisWorkLimitError) Unwrap() error { return ErrAnalysisWorkLimit }

const analysisWorkLimitMessage = "egraph: analysis work limit reached"

type analysisWorkBudget struct {
	limit int
	used  int
}

func (b *analysisWorkBudget) spend() error {
	if b.used >= b.limit {
		return &AnalysisWorkLimitError{Limit: b.limit}
	}
	b.used++
	return nil
}

// Analysis derives immutable facts for equivalence classes. Make derives a
// fact for one node using the current child-class facts; nil means that the
// node contributes no fact. Merge joins two present facts and reports whether
// the stored value changed. Presence is tracked separately, so an explicit
// nil fact added with MergeAnalysisData remains distinct from no fact.
//
// Merge must be associative, commutative, and idempotent, and repeated Make and
// Merge calls must reach a fixed point. Analysis methods must not mutate the
// graph or mutate values previously returned by Make or Merge. Merge must
// support nil operands if callers add nil facts or a previous merge returns
// nil.
type Analysis interface {
	Make(g *EGraph, node ENode) any
	Merge(a, b any) (merged any, changed bool)
}

// AnalysisData returns the current analysis value for a valid e-class. The
// boolean is false when the graph has no analysis or id is invalid. Use
// AnalysisFact when the caller must distinguish an absent fact from an
// explicitly stored nil value.
func (g *EGraph) AnalysisData(id Id) (any, bool) {
	data, ok, err := g.AnalysisDataChecked(id)
	if err != nil {
		return nil, false
	}
	return data, ok
}

// AnalysisDataChecked returns the current analysis value and reports rebuild
// errors before reading class data. Its boolean reports whether id names a
// valid e-class when an analysis is installed. Use AnalysisFactChecked to
// inspect fact presence.
func (g *EGraph) AnalysisDataChecked(id Id) (any, bool, error) {
	data, valid, _, err := g.analysisFactChecked(id)
	return data, valid, err
}

// AnalysisFact returns the current analysis fact and whether one is present.
// It returns false when the graph has no analysis, the class is invalid, or
// the class has no fact. A nil value with true presence is an explicit nil
// fact.
func (g *EGraph) AnalysisFact(id Id) (any, bool) {
	data, present, err := g.AnalysisFactChecked(id)
	if err != nil {
		return nil, false
	}
	return data, present
}

// AnalysisFactChecked returns the current fact, its presence, and any rebuild
// error. Presence is independent of the fact value, which may be nil.
func (g *EGraph) AnalysisFactChecked(id Id) (any, bool, error) {
	data, _, present, err := g.analysisFactChecked(id)
	return data, present, err
}

func (g *EGraph) analysisFactChecked(id Id) (data any, valid, present bool, err error) {
	if g == nil || g.analysis == nil {
		return nil, false, false, nil
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, false, false, err
	}
	root, err := g.FindChecked(id)
	if err != nil {
		return nil, false, false, err
	}
	class := g.store.Class(root)
	if class == nil {
		return nil, false, false, nil
	}
	return class.AnalysisData, true, class.analysisDataSet, nil
}

// MergeAnalysisData joins fact into an e-class analysis value and propagates
// the change to parent classes. Analysis.Merge must preserve its documented
// semilattice contract. The fact and merged values must remain immutable.
// A nil fact is an explicit value; Merge must handle nil operands. It returns
// false when the analysis reports no change.
// If propagation exhausts its work limit after storing the fact, it returns
// true with ErrAnalysisWorkLimit and leaves the graph dirty for a checked retry.
func (g *EGraph) MergeAnalysisData(id Id, fact any) (bool, error) {
	if g == nil {
		return false, fmt.Errorf("egraph: nil graph")
	}
	if g.analysis == nil {
		return false, fmt.Errorf("egraph: graph has no analysis")
	}
	if err := g.RebuildChecked(); err != nil {
		return false, fmt.Errorf("egraph: rebuild before merging analysis data: %w", err)
	}
	root, err := g.FindChecked(id)
	if err != nil {
		return false, err
	}
	class := g.store.Class(root)
	if class == nil {
		return false, fmt.Errorf("egraph: missing class %d", root)
	}
	budget := g.newAnalysisWorkBudget()
	base, baseChanged, err := mergeAnalysisValuesWithBudget(g.analysis, class.analysisFacts, class.analysisFactsSet, fact, true, budget)
	if err != nil {
		return false, err
	}
	if !baseChanged {
		return false, nil
	}
	merged, changed, err := mergeAnalysisValuesWithBudget(g.analysis, class.AnalysisData, class.analysisDataSet, fact, true, budget)
	if err != nil {
		return false, err
	}
	class.analysisFacts = base
	class.analysisFactsSet = true
	if changed {
		class.AnalysisData = merged
		class.analysisDataSet = true
	}
	g.revision++
	// RebuildChecked above leaves parentIndex keyed by current canonical
	// children. Seed this propagation with the changed class and its immediate
	// parents: the fact itself changed the class even if Make does not add
	// another fact when it is recomputed below.
	seeds := append([]Id{root}, g.parentIndex[root]...)
	if err := g.propagateAnalysis(seeds, func(id Id, visit func(Id)) {
		for _, parent := range g.parentIndex[id] {
			visit(parent)
		}
	}, budget); err != nil {
		g.dirty = true
		return true, err
	}
	return true, nil
}

func (g *EGraph) makeAnalysisData(node ENode) any {
	if g.analysis == nil {
		return nil
	}
	node.Children = append([]Id(nil), node.Children...)
	return g.analysis.Make(g, node)
}

func (g *EGraph) rebuildAnalysis(budget *analysisWorkBudget) error {
	if g.analysis == nil {
		return nil
	}
	classes := g.classesList()
	parents := make(map[Id]map[Id]struct{}, len(classes))
	for _, class := range classes {
		parent := g.Find(class.ID)
		for _, node := range class.Nodes {
			for _, child := range node.Children {
				child = g.Find(child)
				if parents[child] == nil {
					parents[child] = make(map[Id]struct{})
				}
				parents[child][parent] = struct{}{}
			}
		}
	}

	seeds := make([]Id, len(classes))
	for i, class := range classes {
		root := g.Find(class.ID)
		seeds[i] = root
		if canonical := g.store.Class(root); canonical != nil {
			canonical.AnalysisData = canonical.analysisFacts
			canonical.analysisDataSet = canonical.analysisFactsSet
		}
	}
	return g.propagateAnalysis(seeds, func(id Id, visit func(Id)) {
		for parent := range parents[id] {
			visit(parent)
		}
	}, budget)
}

func (g *EGraph) propagateAnalysis(seeds []Id, visitParents func(Id, func(Id)), budget *analysisWorkBudget) error {
	if g.analysis == nil {
		return nil
	}
	return egraphcore.PropagateAnalysis(seeds, egraphcore.AnalysisPropagation[Id, ENode]{
		Canonical: g.Find,
		Exists: func(id Id) bool {
			return id != InvalidId && g.store.Class(id) != nil
		},
		Nodes: func(id Id) []ENode {
			class := g.store.Class(id)
			if class == nil {
				return nil
			}
			return class.Nodes
		},
		Data: func(id Id) (any, bool) {
			class := g.store.Class(id)
			if class == nil {
				return nil, false
			}
			return class.AnalysisData, class.analysisDataSet
		},
		SetData: func(id Id, data any, present bool) {
			if class := g.store.Class(id); class != nil {
				class.AnalysisData = data
				class.analysisDataSet = present
			}
		},
		Make: func(node ENode) (any, bool, error) {
			made := g.analysis.Make(g, g.canonicalize(node))
			return made, made != nil, nil
		},
		Merge: func(left, right any) (any, bool, error) {
			merged, changed := g.analysis.Merge(left, right)
			return merged, changed, nil
		},
		VisitParents: visitParents,
		Spend:        budget.spend,
	})
}

func mergeAnalysisValuesWithBudget(analysis Analysis, left any, leftSet bool, right any, rightSet bool, budget *analysisWorkBudget) (any, bool, error) {
	if !leftSet {
		return right, rightSet, nil
	}
	if !rightSet {
		return left, false, nil
	}
	if err := budget.spend(); err != nil {
		return nil, false, err
	}
	merged, changed := analysis.Merge(left, right)
	return merged, changed, nil
}

// SetAnalysisWorkLimit sets the maximum Analysis.Make and Analysis.Merge
// callback evaluations allowed in one propagation operation. limit must be
// positive. RebuildChecked applies one budget across all analysis work in that
// rebuild; MergeAnalysisData applies one budget across its merge and propagation.
func (g *EGraph) SetAnalysisWorkLimit(limit int) error {
	if g == nil {
		return fmt.Errorf("egraph: nil graph")
	}
	if limit <= 0 {
		return fmt.Errorf("egraph: analysis work limit must be positive")
	}
	g.analysisWorkLimit = limit
	return nil
}

// AnalysisWorkLimit returns the current per-operation analysis callback limit.
func (g *EGraph) AnalysisWorkLimit() int {
	if g == nil || g.analysisWorkLimit <= 0 {
		return defaultAnalysisWorkLimit
	}
	return g.analysisWorkLimit
}

func (g *EGraph) newAnalysisWorkBudget() *analysisWorkBudget {
	return &analysisWorkBudget{limit: g.AnalysisWorkLimit()}
}

func mergeAnalysisValues(analysis Analysis, left any, leftSet bool, right any, rightSet bool) (any, bool) {
	if !leftSet {
		return right, rightSet
	}
	if !rightSet {
		return left, false
	}
	return analysis.Merge(left, right)
}
