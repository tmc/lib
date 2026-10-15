package egraph

import (
	"reflect"

	"github.com/tmc/lib/egraphcore"
)

// patternIsExistingClass predicts a no-op for enumeration ordering only. It
// mirrors Instantiate's memo lookups but never creates nodes or compresses the
// graph's union-find. Missing or excessively large patterns remain candidates.
func patternIsExistingClass(g *EGraph, p *Pattern, subst Subst, target Id) bool {
	return patternIsExistingClassWithBudget(g, p, subst, target, nil)
}

func patternIsExistingClassWithBudget(g *EGraph, p *Pattern, subst Subst, target Id, budget *egraphcore.WorkBudget) bool {
	if g == nil {
		return false
	}
	work := 100000
	store := g.storage()
	find := func(id Id) (Id, bool) {
		root, visits, ok := store.FindWithoutCompression(id, work)
		work -= visits
		return root, ok
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
			node = ENode{Op: p.Op, Data: p.Op}
		} else {
			if len(p.Children) > work {
				return 0, false
			}
			children, err := graphSlice[Id](budget, len(p.Children), len(p.Children))
			if err != nil {
				return 0, false
			}
			for i, child := range p.Children {
				id, ok := lookup(child, depth+1)
				if !ok {
					return 0, false
				}
				children[i] = id
			}
			node = NewNode(p.Op, children...)
		}
		var foundID Id
		found := false
		key, err := lookupNodeKey(node, budget)
		if err != nil {
			return 0, false
		}
		store.VisitMemoCandidates(key, func(candidate Id) bool {
			if work == 0 {
				return false
			}
			work--
			id, ok := find(candidate)
			if !ok {
				return true
			}
			class := store.Class(id)
			if class == nil {
				return true
			}
			for _, existing := range class.Nodes {
				if work == 0 {
					return false
				}
				work--
				if existing.Subsumed {
					continue
				}
				if existing.Op != node.Op || len(existing.Children) != len(node.Children) || !reflect.DeepEqual(existing.Data, node.Data) {
					continue
				}
				valid := true
				for i, child := range existing.Children {
					canonical, ok := find(child)
					if !ok || canonical != node.Children[i] {
						valid = false
						break
					}
				}
				if valid {
					foundID, found = id, true
					return false
				}
			}
			return true
		})
		return foundID, found
	}
	id, ok := lookup(p, 1)
	return ok && id == target
}
