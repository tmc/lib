package egraph

// patternIsExistingClass predicts a no-op for enumeration ordering only. It
// mirrors Instantiate's memo lookups but never creates nodes or compresses the
// graph's union-find. Missing or excessively large patterns remain candidates.
func patternIsExistingClass(g *EGraph, p *Pattern, subst Subst, target Id) bool {
	if g == nil {
		return false
	}
	work := 100000
	find := func(id Id) (Id, bool) {
		for {
			if work == 0 || uint64(id) >= uint64(len(g.unionFind)) {
				return 0, false
			}
			work--
			next := g.unionFind[id]
			if next == id {
				return id, true
			}
			id = next
		}
	}
	target, ok := find(target)
	if !ok {
		return false
	}
	var lookup func(*Pattern, int) (Id, bool)
	lookup = func(p *Pattern, depth int) (Id, bool) {
		if p == nil || depth > 256 || work == 0 {
			return 0, false
		}
		work--
		var node ENode
		if p.IsVar {
			if id, ok := subst[p.Op]; ok {
				return find(id)
			}
			// Preserve Instantiate's legacy unbound-variable leaf behavior.
			node = NewLeaf(p.Op)
		} else {
			if len(p.Children) > work {
				return 0, false
			}
			children := make([]Id, len(p.Children))
			for i, child := range p.Children {
				id, ok := lookup(child, depth+1)
				if !ok {
					return 0, false
				}
				children[i] = id
			}
			node = NewNode(p.Op, children...)
		}
		id, ok := g.memo[node.Key()]
		if !ok {
			return 0, false
		}
		return find(id)
	}
	id, ok := lookup(p, 1)
	return ok && id == target
}
