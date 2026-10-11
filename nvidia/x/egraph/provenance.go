package egraph

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
	if g.origins == nil {
		g.origins = make(map[string]string)
	}
	key := after.Key()
	if _, ok := g.origins[key]; !ok {
		g.origins[key] = g.nodeOrigin(before)
	}
}

func (p *Pattern) instantiateFrom(g *EGraph, subst Subst, origin string) Id {
	if g == nil {
		return 0
	}
	previous := g.insertionOrigin
	g.insertionOrigin = origin
	defer func() { g.insertionOrigin = previous }()
	return p.Instantiate(g, subst)
}
