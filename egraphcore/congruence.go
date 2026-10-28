package egraphcore

import "errors"

// OwnedNode associates a canonical node with its current equivalence-class
// owner.
type OwnedNode[ID comparable, Node any] struct {
	Owner ID
	Node  Node
}

// OwnerPair requests a union between two owners of equal nodes.
type OwnerPair[ID comparable] struct {
	Left  ID
	Right ID
}

// CongruencePlan is the result of one global congruence pass. Nodes contains
// one canonical node per owner and structural identity, in first-seen order.
// Merges contains distinct owner pairs whose nodes were structurally equal.
type CongruencePlan[ID comparable, Node any] struct {
	Nodes   []OwnedNode[ID, Node]
	ByOwner map[ID][]Node
	Merges  []OwnerPair[ID]
}

// ErrInvalidCongruencePass reports missing callbacks for a congruence pass.
var ErrInvalidCongruencePass = errors.New("egraphcore: invalid congruence pass")

// PlanCongruencePass deduplicates structurally equal nodes and reports owners
// that must be merged. The input nodes must already have canonical children
// and canonical owner IDs. Equal nodes must have the same key. A key narrows
// equality candidates but never defines equality by itself; equal confirms
// every key match. mergeEquivalent combines
// backend metadata for duplicates, such as a monotone subsumed flag. A nil
// mergeEquivalent keeps the first node. A merge callback must preserve the
// node's key and structural identity.
func PlanCongruencePass[ID comparable, Node any, Key comparable](
	owned []OwnedNode[ID, Node],
	key func(Node) Key,
	equal func(Node, Node) bool,
	mergeEquivalent func(Node, Node) Node,
) (CongruencePlan[ID, Node], error) {
	if key == nil || equal == nil {
		return CongruencePlan[ID, Node]{}, ErrInvalidCongruencePass
	}
	return planCongruencePass(owned, func(n Node) (Key, error) { return key(n), nil }, equal, mergeEquivalent, nil)
}

func planCongruencePass[ID comparable, Node any, Key comparable](owned []OwnedNode[ID, Node], key func(Node) (Key, error), equal func(Node, Node) bool, mergeEquivalent func(Node, Node) Node, budget *WorkBudget) (CongruencePlan[ID, Node], error) {
	nodes, ok := memorySlice[OwnedNode[ID, Node]](budget, 0, len(owned))
	if !ok {
		return CongruencePlan[ID, Node]{}, budget.MemoryError()
	}
	byOwner, ok := memoryMap[ID, []Node](budget, 0)
	if !ok {
		return CongruencePlan[ID, Node]{}, budget.MemoryError()
	}
	merges, ok := memorySlice[OwnerPair[ID]](budget, 0, 0)
	if !ok {
		return CongruencePlan[ID, Node]{}, budget.MemoryError()
	}
	plan := CongruencePlan[ID, Node]{
		Nodes: nodes, ByOwner: byOwner, Merges: merges,
	}
	buckets, ok := memoryMap[Key, []int](budget, 0)
	if !ok {
		return plan, budget.MemoryError()
	}
	mergedOwners, ok := memoryMap[OwnerPair[ID], struct{}](budget, 0)
	if !ok {
		return plan, budget.MemoryError()
	}
	for _, candidate := range owned {
		k, err := key(candidate.Node)
		if err != nil {
			return plan, err
		}
		duplicate := -1
		for _, index := range buckets[k] {
			if equal(plan.Nodes[index].Node, candidate.Node) {
				duplicate = index
				break
			}
		}
		if duplicate < 0 {
			bucket, ok := memoryAppend(budget, buckets[k], len(plan.Nodes))
			if !ok || !memoryPut(budget, buckets, k, bucket) {
				return plan, budget.MemoryError()
			}
			plan.Nodes = append(plan.Nodes, candidate)
			continue
		}
		kept := &plan.Nodes[duplicate]
		if mergeEquivalent != nil {
			kept.Node = mergeEquivalent(kept.Node, candidate.Node)
		}
		if kept.Owner == candidate.Owner {
			continue
		}
		pair := OwnerPair[ID]{Left: kept.Owner, Right: candidate.Owner}
		if _, exists := mergedOwners[pair]; !exists {
			if !memoryPut(budget, mergedOwners, pair, struct{}{}) {
				return plan, budget.MemoryError()
			}
			var ok bool
			plan.Merges, ok = memoryAppend(budget, plan.Merges, pair)
			if !ok {
				return plan, budget.MemoryError()
			}
		}
	}
	for _, node := range plan.Nodes {
		group, ok := memoryAppend(budget, plan.ByOwner[node.Owner], node.Node)
		if !ok || !memoryPut(budget, plan.ByOwner, node.Owner, group) {
			return plan, budget.MemoryError()
		}
	}
	return plan, nil
}

// ApplyCongruencePass plans one set of canonical nodes and applies the owner
// unions required by structural duplicates. The caller owns node collection,
// canonicalization, and any follow-up rebuild work. A pass that merges owners
// must be recollected before its ByOwner groups are published.
func ApplyCongruencePass[ID ~uint32, Node any, Key comparable](
	graph Graph[ID, Node],
	owned []OwnedNode[ID, Node],
	key func(Node) Key,
	equal func(Node, Node) bool,
	mergeEquivalent func(Node, Node) Node,
) (CongruencePlan[ID, Node], int, error) {
	if graphIsNil(graph) {
		return CongruencePlan[ID, Node]{}, 0, ErrInvalidCongruencePass
	}
	if key == nil || equal == nil {
		return CongruencePlan[ID, Node]{}, 0, ErrInvalidCongruencePass
	}
	return applyCongruencePass(graph, owned, func(n Node) (Key, error) { return key(n), nil }, equal, mergeEquivalent, nil)
}

func applyCongruencePass[ID ~uint32, Node any, Key comparable](graph Graph[ID, Node], owned []OwnedNode[ID, Node], key func(Node) (Key, error), equal func(Node, Node) bool, mergeEquivalent func(Node, Node) Node, budget *WorkBudget) (CongruencePlan[ID, Node], int, error) {
	plan, err := planCongruencePass(owned, key, equal, mergeEquivalent, budget)
	if err != nil {
		return CongruencePlan[ID, Node]{}, 0, err
	}
	merged := 0
	for _, pair := range plan.Merges {
		_, changed, err := graph.UnionChecked(pair.Left, pair.Right)
		if err != nil {
			return plan, merged, err
		}
		if changed {
			merged++
		}
	}
	return plan, merged, nil
}
