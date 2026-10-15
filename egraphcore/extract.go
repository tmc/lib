package egraphcore

import (
	"errors"
	"slices"
)

// BestCostNode is the selected node and accumulated cost for an e-class.
type BestCostNode[Node any] struct {
	Node Node
	Cost float64
}

// RankedBestCostNode is a selected node, its cost, and caller-defined rank.
// Rank can carry secondary extraction information such as expression size.
type RankedBestCostNode[Node, Rank any] struct {
	Node Node
	Cost float64
	Rank Rank
}

// CostEvaluation computes an eligible node's cost from child costs. It is
// called for every visited node, including nodes with unknown child costs, so
// adapters can preserve work accounting before deciding whether to skip it.
type CostEvaluation[Node any] func(node Node, childCosts []float64, allChildrenKnown bool) (cost float64, eligible bool, err error)

// CostVisit runs once for each node before the helper canonicalizes its
// children or looks up their costs. It can be used for accounting that must
// include nodes whose children are not yet extractable.
type CostVisit[Node any] func(node Node) error

// RankedCostEvaluation evaluates a node from the selected states of its
// children. childStates has one entry per child; entries are zero values when
// allChildrenKnown is false. The adapter decides whether an incomplete child
// set is eligible.
type RankedCostEvaluation[Node, Rank any] func(node Node, childStates []RankedBestCostNode[Node, Rank], allChildrenKnown bool) (cost float64, rank Rank, eligible bool, err error)

// RankedBestCostLookup reads the selected state for one canonical class.
type RankedBestCostLookup[ID ~uint32, Node, Rank any] func(ID) (RankedBestCostNode[Node, Rank], bool)

// RankedCostAcceptance may reject a candidate that is otherwise preferred,
// for example when selecting it would create a cycle. lookup is read-only; an
// error aborts relaxation and is returned to the caller.
type RankedCostAcceptance[ID ~uint32, Node, Rank any] func(class ID, candidate RankedBestCostNode[Node, Rank], lookup RankedBestCostLookup[ID, Node, Rank]) (bool, error)

// ErrInvalidCostRelaxation reports missing callbacks or an invalid pass bound.
var ErrInvalidCostRelaxation = errors.New("egraphcore: invalid cost relaxation")

// ErrCostRelaxationLimit reports that cost relaxation reached its pass or node
// visit limit before reaching a fixed point.
var ErrCostRelaxationLimit = errors.New("egraphcore: cost relaxation limit exceeded")

// RelaxBestCostsWorklist computes selected node states with a deterministic
// dependency worklist. It scans every class once, then requeues parents when a
// selected child state changes. dependencyVisit runs before each node's child
// edges are added to the reverse dependency index; visit runs before each cost
// evaluation. The caller supplies candidate evaluation, preference, rank, and
// acceptance policy; the core owns canonical child lookup and dependency
// scheduling. better must define a stable strict preference to ensure
// termination. maxVisits must be positive and limits node evaluations; on
// exhaustion the partial best-state map is returned with
// ErrCostRelaxationLimit.
func RelaxBestCostsWorklist[ID ~uint32, Node, Rank any](
	classes []ID,
	nodes func(ID) []Node,
	canonicalClass func(ID) ID,
	children func(Node) []ID,
	canonicalChild func(ID) ID,
	dependencyVisit CostVisit[Node],
	visit CostVisit[Node],
	evaluate RankedCostEvaluation[Node, Rank],
	better func(candidate, current RankedBestCostNode[Node, Rank]) bool,
	accept RankedCostAcceptance[ID, Node, Rank],
	maxVisits int,
) (map[ID]RankedBestCostNode[Node, Rank], int, error) {
	return RelaxBestCostsWorklistWithBudget(classes, nodes, canonicalClass, children, canonicalChild, dependencyVisit, visit, evaluate, better, accept, maxVisits, nil)
}

// RelaxBestCostsWorklistWithBudget is RelaxBestCostsWorklist with storage
// reservations through budget. The caller keeps its MemoryScope open through
// the last use of the returned table. Callbacks return borrowed data or reserve
// their own storage. Allocation refusal returns no partial table; other errors
// retain the original partial-table behavior. maxVisits and visit callbacks
// govern work independently of the allocation scope. A nil budget is unlimited.
func RelaxBestCostsWorklistWithBudget[ID ~uint32, Node, Rank any](
	classes []ID,
	nodes func(ID) []Node,
	canonicalClass func(ID) ID,
	children func(Node) []ID,
	canonicalChild func(ID) ID,
	dependencyVisit CostVisit[Node],
	visit CostVisit[Node],
	evaluate RankedCostEvaluation[Node, Rank],
	better func(candidate, current RankedBestCostNode[Node, Rank]) bool,
	accept RankedCostAcceptance[ID, Node, Rank],
	maxVisits int,
	budget *WorkBudget,
) (output map[ID]RankedBestCostNode[Node, Rank], visits int, err error) {
	defer func() {
		if memoryErr := budget.MemoryError(); memoryErr != nil {
			output, err = nil, memoryErr
		}
	}()
	if nodes == nil || canonicalClass == nil || children == nil || canonicalChild == nil || evaluate == nil || better == nil || maxVisits <= 0 {
		return nil, 0, ErrInvalidCostRelaxation
	}
	visits = 0
	classSet, ok := memoryMap[ID, struct{}](budget, len(classes))
	if !ok {
		return nil, visits, budget.MemoryError()
	}
	classIDs, ok := memorySlice[ID](budget, 0, len(classes))
	if !ok {
		return nil, visits, budget.MemoryError()
	}
	for _, id := range classes {
		id = canonicalClass(id)
		if _, ok := classSet[id]; ok {
			continue
		}
		if !memoryPut(budget, classSet, id, struct{}{}) {
			return nil, visits, budget.MemoryError()
		}
		classIDs = append(classIDs, id)
	}
	slices.Sort(classIDs)

	parents, ok := memoryMap[ID, map[ID]struct{}](budget, 0)
	if !ok {
		return nil, visits, budget.MemoryError()
	}
	for _, parent := range classIDs {
		for _, node := range nodes(parent) {
			childIDs := children(node)
			if dependencyVisit != nil {
				if err := dependencyVisit(node); err != nil {
					return nil, 0, err
				}
			}
			for _, child := range childIDs {
				child = canonicalChild(child)
				if parents[child] == nil {
					set, ok := memoryMap[ID, struct{}](budget, 0)
					if !ok {
						return nil, visits, budget.MemoryError()
					}
					if !memoryPut(budget, parents, child, set) {
						return nil, visits, budget.MemoryError()
					}
				}
				if !memoryPut(budget, parents[child], parent, struct{}{}) {
					return nil, visits, budget.MemoryError()
				}
			}
		}
	}

	best, ok := memoryMap[ID, RankedBestCostNode[Node, Rank]](budget, 0)
	if !ok {
		return nil, visits, budget.MemoryError()
	}
	lookup := func(id ID) (RankedBestCostNode[Node, Rank], bool) {
		state, ok := best[canonicalClass(id)]
		return state, ok
	}
	queue, ok := memorySlice[ID](budget, len(classIDs), len(classIDs))
	if !ok {
		return nil, visits, budget.MemoryError()
	}
	copy(queue, classIDs)
	queued, ok := memoryMap[ID, bool](budget, len(classIDs))
	if !ok {
		return nil, visits, budget.MemoryError()
	}
	for _, id := range queue {
		if !memoryPut(budget, queued, id, true) {
			return nil, visits, budget.MemoryError()
		}
	}
	for head := 0; head < len(queue); head++ {
		id := queue[head]
		queued[id] = false
		changed := false
		for _, node := range nodes(id) {
			if visits >= maxVisits {
				return best, visits, ErrCostRelaxationLimit
			}
			visits++
			if visit != nil {
				if err := visit(node); err != nil {
					return best, visits, err
				}
			}
			sourceChildren := children(node)
			childIDs, ok := memorySlice[ID](budget, len(sourceChildren), len(sourceChildren))
			if !ok {
				return nil, visits, budget.MemoryError()
			}
			copy(childIDs, sourceChildren)
			childStates, ok := memorySlice[RankedBestCostNode[Node, Rank]](budget, len(childIDs), len(childIDs))
			if !ok {
				return nil, visits, budget.MemoryError()
			}
			allKnown := true
			for i, child := range childIDs {
				state, ok := lookup(canonicalChild(child))
				if !ok {
					allKnown = false
					continue
				}
				childStates[i] = state
			}
			cost, rank, eligible, err := evaluate(node, childStates, allKnown)
			if err != nil {
				return best, visits, err
			}
			if !eligible {
				continue
			}
			candidate := RankedBestCostNode[Node, Rank]{Node: node, Cost: cost, Rank: rank}
			current, exists := best[id]
			if exists && !better(candidate, current) {
				continue
			}
			if accept != nil {
				accepted, err := accept(id, candidate, lookup)
				if err != nil {
					return best, visits, err
				}
				if !accepted {
					continue
				}
			}
			if !memoryPut(budget, best, id, candidate) {
				return nil, visits, budget.MemoryError()
			}
			changed = true
		}
		if !changed {
			continue
		}
		dependents, ok := memorySlice[ID](budget, 0, len(parents[id]))
		if !ok {
			return nil, visits, budget.MemoryError()
		}
		for parent := range parents[id] {
			dependents = append(dependents, parent)
		}
		slices.Sort(dependents)
		for _, parent := range dependents {
			if !queued[parent] {
				queue, ok = memoryAppend(budget, queue, parent)
				if !ok {
					return nil, visits, budget.MemoryError()
				}
				if !memoryPut(budget, queued, parent, true) {
					return nil, visits, budget.MemoryError()
				}
			}
		}
	}
	return best, visits, nil
}

// RelaxBestCosts computes bottom-up best costs by bounded fixed-point scans.
// The caller supplies class enumeration, node children, canonicalization,
// candidate validity, and tie policy; the core owns child-cost lookup and
// relaxation. A nil better callback selects only strictly lower costs. If
// maxPasses is reached while a pass still changes the result, the partial map
// is returned with ErrCostRelaxationLimit.
func RelaxBestCosts[ID ~uint32, Node any](
	classes []ID,
	nodes func(ID) []Node,
	canonicalClass func(ID) ID,
	children func(Node) []ID,
	canonicalChild func(ID) ID,
	visit CostVisit[Node],
	evaluate CostEvaluation[Node],
	better func(candidate Node, candidateCost float64, current Node, currentCost float64) bool,
	maxPasses int,
) (map[ID]BestCostNode[Node], int, error) {
	if nodes == nil || canonicalClass == nil || children == nil || canonicalChild == nil || evaluate == nil || maxPasses <= 0 {
		return nil, 0, ErrInvalidCostRelaxation
	}
	best := make(map[ID]BestCostNode[Node])
	costs := make(map[ID]float64)
	passes := 0
	for passes < maxPasses {
		passes++
		changed := false
		for _, classID := range classes {
			owner := canonicalClass(classID)
			for _, node := range nodes(classID) {
				if visit != nil {
					if err := visit(node); err != nil {
						return best, passes, err
					}
				}
				childIDs := append([]ID(nil), children(node)...)
				for i, child := range childIDs {
					childIDs[i] = canonicalChild(child)
				}
				childCosts, allChildrenKnown := ChildCosts(childIDs, costs)
				cost, eligible, err := evaluate(node, childCosts, allChildrenKnown)
				if err != nil {
					return best, passes, err
				}
				if !eligible {
					continue
				}
				current, exists := best[owner]
				if exists {
					improves := cost < current.Cost
					if cost == current.Cost && better != nil {
						improves = better(node, cost, current.Node, current.Cost)
					}
					if !improves {
						continue
					}
				}
				best[owner] = BestCostNode[Node]{Node: node, Cost: cost}
				costs[owner] = cost
				changed = true
			}
		}
		if !changed {
			break
		}
		if passes == maxPasses {
			return best, passes, ErrCostRelaxationLimit
		}
	}
	return best, passes, nil
}
