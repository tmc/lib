package egraph

import "github.com/tmc/lib/egraphcore"

// nodeOrigin returns the first insertion's origin. Reusing an existing node
// never rewrites its history merely because another rule found it.
func (g *EGraph) nodeOrigin(node ENode) string {
	if origin, ok := g.origins[node.Key()]; ok {
		return origin
	}
	if origin, ok := g.origins[g.Canonicalize(node).Key()]; ok {
		return origin
	}
	return "input"
}

func (g *EGraph) preserveOrigin(before, after ENode) {
	_ = g.preserveOriginWithBudget(before, after, nil)
}

func (g *EGraph) preserveOriginWithBudget(before, after ENode, budget *egraphcore.WorkBudget) error {
	var err error
	if g.origins == nil {
		g.origins, err = graphMap[string, string](budget, 0)
		if err != nil {
			return err
		}
	}
	key, err := lookupNodeKey(after, budget)
	if err != nil {
		return err
	}
	if _, ok := g.origins[key]; !ok {
		origin := "input"
		beforeKey, err := lookupNodeKey(before, budget)
		if err != nil {
			return err
		}
		if value, ok := g.origins[beforeKey]; ok {
			origin = value
		} else {
			canon, err := g.canonicalizeWithBudget(before, budget)
			if err != nil {
				return err
			}
			canonKey, err := lookupNodeKey(canon, budget)
			if err != nil {
				return err
			}
			if value, ok := g.origins[canonKey]; ok {
				origin = value
			}
		}
		return graphPut(budget, g.origins, key, origin)
	}
	return nil
}

func (p *Pattern) instantiateFrom(g *EGraph, subst Subst, origin string) Id {
	if g == nil {
		return 0
	}
	previous, previousSet := g.insertionOrigin, g.insertionOriginSet
	g.insertionOrigin, g.insertionOriginSet = origin, true
	defer func() { g.insertionOrigin, g.insertionOriginSet = previous, previousSet }()
	return p.Instantiate(g, subst)
}
