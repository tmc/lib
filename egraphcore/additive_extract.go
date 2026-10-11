package egraphcore

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"sort"
)

// AdditiveBestCostNode is the selected node and rounded total cost for an
// e-class returned by ExtractBestAdditive.
type AdditiveBestCostNode[Node any] struct {
	Node Node
	Cost float64
}

// ErrInvalidAdditiveExtraction reports missing callbacks or an invalid work
// limit.
var ErrInvalidAdditiveExtraction = errors.New("egraphcore: invalid additive extraction")

// ErrAdditiveExtractionWorkLimit reports that additive extraction exhausted
// its node and edge work limit.
var ErrAdditiveExtractionWorkLimit = errors.New("egraphcore: additive extraction work limit exceeded")

// ErrAdditiveNegativeCycle reports a reachable cycle whose additive local
// costs make the total unbounded below.
var ErrAdditiveNegativeCycle = errors.New("egraphcore: reachable negative-cost cycle")

// ErrAdditiveUnresolved reports that the requested root has no finite acyclic
// derivation.
var ErrAdditiveUnresolved = errors.New("egraphcore: unresolved additive extraction root")

// ExtractBestAdditive selects a minimum-cost acyclic expression from root.
// It returns ErrAdditiveUnresolved if root has no finite derivation.
func ExtractBestAdditive[ID ~uint32, Node any](
	root ID,
	nodes func(ID) []Node,
	canonical func(ID) ID,
	children func(Node) []ID,
	localCost func(Node) float64,
	maxWork int,
) (map[ID]AdditiveBestCostNode[Node], int, error) {
	selected, work, err := ExtractBestAdditiveFromRoots([]ID{root}, nodes, canonical, children, localCost, nil, maxWork)
	if err != nil {
		return nil, work, err
	}
	root = canonical(root)
	if _, ok := selected[root]; !ok {
		return nil, work, fmt.Errorf("%w: class %v", ErrAdditiveUnresolved, root)
	}
	return selected, work, nil
}

// ExtractBestAdditiveFromRoots selects minimum-cost acyclic expressions from
// one or more reachability seeds and returns selected nodes for all reachable
// classes with finite derivations. The cost of a node is its local cost plus
// the cost of every child use, so a repeated child is counted repeatedly.
// Local float64 costs are interpreted as exact binary rationals while comparing
// candidates. Equal-cost expressions are ranked by unfolded node count, then
// by the order returned from nodes.
//
// Reachability follows eligible node alternatives. Seeds and other classes
// without a finite derivation are omitted. Reachable negative-cost cycles are
// rejected. maxWork bounds seed processing, scanned nodes, and eligible child
// edges across reachability, cost relaxation, cycle detection, size ranking,
// and final selection. The graph callbacks must
// describe an immutable graph for the duration of the call.
func ExtractBestAdditiveFromRoots[ID ~uint32, Node any](
	roots []ID,
	nodes func(ID) []Node,
	canonical func(ID) ID,
	children func(Node) []ID,
	localCost func(Node) float64,
	nodeEligible func(Node) bool,
	maxWork int,
) (map[ID]AdditiveBestCostNode[Node], int, error) {
	if len(roots) == 0 || nodes == nil || canonical == nil || children == nil || localCost == nil || maxWork <= 0 {
		return nil, 0, ErrInvalidAdditiveExtraction
	}
	work := 0
	spend := func(units int) error {
		if units < 0 || units > maxWork-work {
			return ErrAdditiveExtractionWorkLimit
		}
		work += units
		return nil
	}
	if err := spend(len(roots)); err != nil {
		return nil, work, err
	}
	rootSet := make(map[ID]bool, len(roots))
	for _, root := range roots {
		rootSet[canonical(root)] = true
	}
	chargeNodeAndEdges := func(node Node) ([]ID, error) {
		childIDs := children(node)
		if err := spend(1); err != nil {
			return nil, err
		}
		if err := spend(len(childIDs)); err != nil {
			return nil, err
		}
		return childIDs, nil
	}

	// Snapshot the root-reachable graph once, preserving each class's node
	// order for deterministic final ties.
	classNodes := make(map[ID][]Node)
	seen := make(map[ID]bool, len(rootSet))
	stack := make([]ID, 0, len(rootSet))
	for root := range rootSet {
		seen[root] = true
		stack = append(stack, root)
	}
	for len(stack) != 0 {
		id := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, node := range nodes(id) {
			// Charge the scan before eligibility filtering so excluded nodes
			// cannot make graph traversal unbounded under a fixed work limit.
			if err := spend(1); err != nil {
				return nil, work, err
			}
			if nodeEligible != nil && !nodeEligible(node) {
				continue
			}
			childIDs := children(node)
			if err := spend(len(childIDs)); err != nil {
				return nil, work, err
			}
			classNodes[id] = append(classNodes[id], node)
			for _, child := range childIDs {
				child = canonical(child)
				if !seen[child] {
					seen[child] = true
					stack = append(stack, child)
				}
			}
		}
	}
	classIDs := make([]ID, 0, len(classNodes))
	for id := range classNodes {
		classIDs = append(classIDs, id)
	}
	sort.Slice(classIDs, func(i, j int) bool { return classIDs[i] < classIDs[j] })

	localCosts := make(map[ID][]*big.Rat, len(classIDs))
	for _, id := range classIDs {
		class := classNodes[id]
		costs := make([]*big.Rat, len(class))
		for i, node := range class {
			if err := spend(1); err != nil {
				return nil, work, err
			}
			value := localCost(node)
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, work, fmt.Errorf("%w: nonfinite local cost %v", ErrInvalidAdditiveExtraction, value)
			}
			costs[i] = new(big.Rat).SetFloat64(value)
		}
		localCosts[id] = costs
	}

	exactCosts := make(map[ID]*big.Rat, len(classIDs))
	addNodeCost := func(id ID, index int, node Node) (*big.Rat, bool, error) {
		childIDs, err := chargeNodeAndEdges(node)
		if err != nil {
			return nil, false, err
		}
		cost := new(big.Rat).Set(localCosts[id][index])
		for _, child := range childIDs {
			childCost, ok := exactCosts[canonical(child)]
			if !ok {
				return nil, false, nil
			}
			cost.Add(cost, childCost)
		}
		return cost, true, nil
	}
	for _, id := range classIDs {
		for i, node := range classNodes[id] {
			childIDs, err := chargeNodeAndEdges(node)
			if err != nil {
				return nil, work, err
			}
			if len(childIDs) != 0 {
				continue
			}
			cost := new(big.Rat).Set(localCosts[id][i])
			if current, ok := exactCosts[id]; !ok || cost.Cmp(current) < 0 {
				exactCosts[id] = cost
			}
		}
	}

	// Relaxing at most |V| rounds is sufficient for a finite acyclic optimum;
	// a further strict improvement means a reachable negative-cost cycle.
	for pass := 0; pass < len(classIDs); pass++ {
		changed := false
		for _, id := range classIDs {
			for i, node := range classNodes[id] {
				cost, known, err := addNodeCost(id, i, node)
				if err != nil {
					return nil, work, err
				}
				if !known {
					continue
				}
				if current, ok := exactCosts[id]; !ok || cost.Cmp(current) < 0 {
					exactCosts[id] = cost
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}
	for _, id := range classIDs {
		best, resolved := exactCosts[id]
		for i, node := range classNodes[id] {
			cost, known, err := addNodeCost(id, i, node)
			if err != nil {
				return nil, work, err
			}
			if !known {
				continue
			}
			if !resolved {
				continue
			}
			if cost.Cmp(best) < 0 {
				return nil, work, fmt.Errorf("%w at class %v", ErrAdditiveNegativeCycle, id)
			}
		}
	}
	// Compute the smallest unfolded expression size among minimum-cost nodes.
	sizes := make(map[ID]*big.Int, len(classIDs))
	for pass := 0; pass < len(classIDs); pass++ {
		changed := false
		for _, id := range classIDs {
			best, resolved := exactCosts[id]
			if !resolved {
				continue
			}
			for i, node := range classNodes[id] {
				childIDs, err := chargeNodeAndEdges(node)
				if err != nil {
					return nil, work, err
				}
				cost := new(big.Rat).Set(localCosts[id][i])
				size := big.NewInt(1)
				known := true
				for _, child := range childIDs {
					child = canonical(child)
					childCost, costOK := exactCosts[child]
					childSize, sizeOK := sizes[child]
					if !costOK || !sizeOK {
						known = false
						break
					}
					cost.Add(cost, childCost)
					size.Add(size, childSize)
				}
				if !known || cost.Cmp(best) != 0 {
					continue
				}
				if current, ok := sizes[id]; !ok || size.Cmp(current) < 0 {
					sizes[id] = size
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}

	result := make(map[ID]AdditiveBestCostNode[Node], len(exactCosts))
	for _, id := range classIDs {
		best, resolved := exactCosts[id]
		bestSize, hasSize := sizes[id]
		if !resolved || !hasSize {
			continue
		}
		selected := false
		for i, node := range classNodes[id] {
			childIDs, err := chargeNodeAndEdges(node)
			if err != nil {
				return nil, work, err
			}
			cost := new(big.Rat).Set(localCosts[id][i])
			size := big.NewInt(1)
			known := true
			for _, child := range childIDs {
				child = canonical(child)
				childCost, costOK := exactCosts[child]
				childSize, sizeOK := sizes[child]
				if !costOK || !sizeOK {
					known = false
					break
				}
				cost.Add(cost, childCost)
				size.Add(size, childSize)
			}
			if known && cost.Cmp(best) == 0 && size.Cmp(bestSize) == 0 {
				value, _ := best.Float64()
				if math.IsInf(value, 0) || math.IsNaN(value) {
					return nil, work, fmt.Errorf("%w: total cost for class %v is outside float64 range", ErrInvalidAdditiveExtraction, id)
				}
				result[id] = AdditiveBestCostNode[Node]{Node: node, Cost: value}
				selected = true
				break
			}
		}
		if !selected {
			return nil, work, fmt.Errorf("%w: class %v", ErrAdditiveUnresolved, id)
		}
	}
	return result, work, nil
}
