package egraphcore

import "math"

// CanonicalChildIDs returns the canonical representatives for children in
// their original order, applying path compression to parent as it goes.
func CanonicalChildIDs[ID ~uint32](parent []ID, children []ID) []ID {
	canonical := make([]ID, len(children))
	for i, child := range children {
		canonical[i] = Find(parent, child)
	}
	return canonical
}

// ChildCosts returns the cost for each child ID in order. It returns false if
// any child has no cost or has a non-finite cost.
func ChildCosts[ID comparable](children []ID, costs map[ID]float64) ([]float64, bool) {
	result := make([]float64, len(children))
	for i, child := range children {
		cost, ok := costs[child]
		if !ok || math.IsNaN(cost) || math.IsInf(cost, 0) {
			return nil, false
		}
		result[i] = cost
	}
	return result, true
}
