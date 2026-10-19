package egraph

import (
	"context"

	"github.com/tmc/lib/egraphcore"
)

// CloneWithBudget copies the graph with allocation-site reservations. The
// caller keeps budget's MemoryScope open through the copy's last use, including
// aliases. Scratch remains charged until that scope closes. Immutable strings
// and historical relation epoch tokens remain borrowed. Mutable payloads use
// extraction's existing checked clone policy; unsupported ownership returns
// ErrMemoryOperation. A nil memory scope delegates to ordinary Clone, including
// its legacy cyclic-payload behavior. The source is never canonicalized.
func (g *EGraph) CloneWithBudget(ctx context.Context, budget *egraphcore.WorkBudget) (*EGraph, error) {
	if ctx == nil {
		return nil, graphErrorf(budget, "egraph: nil clone context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if g == nil {
		return nil, nil
	}
	if budget.MemoryScope() == nil {
		return g.Clone(), nil
	}
	if err := graphSlots[EGraph](budget, 1); err != nil {
		return nil, err
	}
	out := &EGraph{revision: g.revision, insertionOrigin: g.insertionOrigin, insertionOriginSet: g.insertionOriginSet}
	var err error
	out.worklist, err = graphSlice[Id](budget, len(g.worklist), len(g.worklist))
	if err != nil {
		return nil, err
	}
	copy(out.worklist, g.worklist)
	out.origins, err = graphMap[string, string](budget, len(g.origins))
	if err != nil {
		return nil, err
	}
	for key, origin := range g.origins {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := graphPut(budget, out.origins, key, origin); err != nil {
			return nil, err
		}
	}
	facts := g.facts
	if facts == nil {
		if err := graphSlots[egraphcore.FactStore[Id]](budget, 1); err != nil {
			return nil, err
		}
		facts = new(egraphcore.FactStore[Id])
	}
	out.facts, err = facts.CloneWithBudget(budget)
	if err != nil {
		return nil, err
	}
	if err := graphSlots[extractionBudget](budget, 1); err != nil {
		return nil, err
	}
	payload := &extractionBudget{ctx: ctx}
	cloneNode := func(node ENode) (ENode, error) {
		if err := ctx.Err(); err != nil {
			return ENode{}, err
		}
		children, err := graphSlice[Id](budget, len(node.Children), len(node.Children))
		if err != nil {
			return ENode{}, err
		}
		copy(children, node.Children)
		if len(children) == 0 {
			children = nil
		}
		data, err := cloneDataWithMemory(node.Data, payload, budget, budget)
		if err != nil {
			return ENode{}, err
		}
		return ENode{Op: node.Op, Children: children, Data: data, Subsumed: node.Subsumed}, nil
	}
	store := g.store
	if store == nil {
		if err := graphSlots[egraphcore.Store[Id, *EClass]](budget, 1); err != nil {
			return nil, err
		}
		store = new(egraphcore.Store[Id, *EClass])
	}
	out.store, err = store.CloneWithBudget(func(class *EClass, _ *egraphcore.WorkBudget) (*EClass, error) {
		if class == nil {
			return nil, nil
		}
		nodes, err := graphSlice[ENode](budget, len(class.Nodes), len(class.Nodes))
		if err != nil {
			return nil, err
		}
		for i, node := range class.Nodes {
			nodes[i], err = cloneNode(node)
			if err != nil {
				return nil, err
			}
		}
		parents, err := graphSlice[parentRef](budget, len(class.parents), len(class.parents))
		if err != nil {
			return nil, err
		}
		for i, parent := range class.parents {
			node, err := cloneNode(parent.node)
			if err != nil {
				return nil, err
			}
			parents[i] = parentRef{node: node, classID: parent.classID}
		}
		if err := graphSlots[EClass](budget, 1); err != nil {
			return nil, err
		}
		return &EClass{ID: class.ID, Nodes: nodes, parents: parents}, nil
	}, budget)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// CloneWithBudget copies the graph, roots, derivation maps and metadata with
// preallocation reservations. The caller owns budget's MemoryScope. Closing it
// invalidates the copy and aliases. TypedMetadata's immutable Shape remains
// borrowed, matching TypedMetadata.Clone. A nil scope uses ordinary Clone.
func (s *SearchState) CloneWithBudget(ctx context.Context, budget *egraphcore.WorkBudget) (*SearchState, error) {
	if ctx == nil {
		return nil, graphErrorf(budget, "egraph: nil clone context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s == nil || s.Graph == nil {
		return nil, nil
	}
	if budget.MemoryScope() == nil {
		return s.Clone(), nil
	}
	if err := graphSlots[SearchState](budget, 1); err != nil {
		return nil, err
	}
	out := &SearchState{Depth: s.Depth}
	var err error
	out.Graph, err = s.Graph.CloneWithBudget(ctx, budget)
	if err != nil {
		return nil, err
	}
	out.Roots, err = graphMap[string, Id](budget, len(s.Roots))
	if err != nil {
		return nil, err
	}
	for name, id := range s.Roots {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := graphPut(budget, out.Roots, name, out.Graph.Find(id)); err != nil {
			return nil, err
		}
	}
	out.Derivation, err = graphSlice[Action](budget, len(s.Derivation), len(s.Derivation))
	if err != nil {
		return nil, err
	}
	for i, action := range s.Derivation {
		out.Derivation[i] = action
		subst, err := graphMap[string, Id](budget, len(action.Subst))
		if err != nil {
			return nil, err
		}
		for name, id := range action.Subst {
			if err := graphPut(budget, subst, name, id); err != nil {
				return nil, err
			}
		}
		binding, err := graphMap[string, egraphcore.Value[Id]](budget, len(action.Binding))
		if err != nil {
			return nil, err
		}
		for name, value := range action.Binding {
			if err := graphPut(budget, binding, name, value); err != nil {
				return nil, err
			}
		}
		out.Derivation[i].Subst, out.Derivation[i].Binding = subst, binding
	}
	out.Metadata, err = graphMap[Id, TypedMetadata](budget, len(s.Metadata))
	if err != nil {
		return nil, err
	}
	for id, metadata := range s.Metadata {
		if err := graphPut(budget, out.Metadata, id, metadata.Clone()); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// ApplyActionWithCloneBudget admits one detached clone before validation can
// canonicalize or rebuild a graph, then runs the existing checked executor on
// that clone without a second copy. Matching, fact staging, graph mutation and
// retained action storage share the clone's allocation owner. Temporary and
// replaced storage remains charged through the last use of all results.
// This includes returned validation errors: callers inspect or retain an error
// before closing the scope. Callback errors propagate without invoking Error.
// Unsupported payload operations return ErrMemoryOperation. The caller owns
// budget's MemoryScope, including on failure. A nil scope uses ApplyAction.
func (s *SearchState) ApplyActionWithCloneBudget(ctx context.Context, candidate ActionCandidate, budget *egraphcore.WorkBudget) (*TransitionResult, error) {
	if budget.MemoryScope() == nil {
		return s.ApplyAction(candidate)
	}
	cloned, err := s.CloneWithBudget(ctx, budget)
	if err != nil {
		return nil, err
	}
	if cloned == nil {
		return nil, graphErrorf(budget, "egraph: nil search state or graph")
	}
	return cloned.applyAction(ctx, candidate, cloned, budget)
}

func transitionWithBudget(result TransitionResult, budget *egraphcore.WorkBudget) (*TransitionResult, error) {
	if err := graphSlots[TransitionResult](budget, 1); err != nil {
		return nil, err
	}
	out := new(TransitionResult)
	*out = result
	return out, nil
}
