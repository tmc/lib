package egraphcore

import (
	"fmt"
	"math"
)

// Graph is the checked mutation surface shared by e-graph implementations.
// Node remains a type parameter so each backend can retain its own operator
// and payload representation. Implementations keep backend-specific node
// validation, union policy, and rebuild behavior behind these methods.
type Graph[ID ~uint32, Node any] interface {
	AddChecked(Node) (ID, error)
	Find(ID) ID
	UnionChecked(ID, ID) (ID, bool, error)
	RebuildChecked() error
}

// CostModel assigns a cost to one node using costs already computed for its
// child classes. A model must be deterministic, must not mutate the node or
// its payload, and must return finite, non-negative costs that are monotone
// in every child cost and no smaller than any child cost. Backend-specific
// models supply their own cost policy within that contract. Use measured
// candidate search when ranking by a whole-expression runtime metric.
type CostModel[Node any] interface {
	NodeCost(Node, []float64) float64
}

// ValidateMonotoneCost checks the per-node constraints required by the
// standard best and top-K extractors. It checks the returned value against
// the supplied child costs; callers remain responsible for ensuring their
// cost function is monotone for all possible child-cost inputs.
func ValidateMonotoneCost(cost float64, childCosts []float64) error {
	if math.IsNaN(cost) || math.IsInf(cost, 0) || cost < 0 {
		return fmt.Errorf("cost must be finite and non-negative: %v", cost)
	}
	for _, childCost := range childCosts {
		if cost < childCost {
			return fmt.Errorf("cost %v is less than child cost %v", cost, childCost)
		}
	}
	return nil
}
