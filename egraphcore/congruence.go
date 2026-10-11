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
	plan := CongruencePlan[ID, Node]{
		Nodes:   make([]OwnedNode[ID, Node], 0, len(owned)),
		ByOwner: make(map[ID][]Node),
		Merges:  make([]OwnerPair[ID], 0),
	}
	buckets := make(map[Key][]int)
	mergedOwners := make(map[OwnerPair[ID]]struct{})
	for _, candidate := range owned {
		k := key(candidate.Node)
		duplicate := -1
		for _, index := range buckets[k] {
			if equal(plan.Nodes[index].Node, candidate.Node) {
				duplicate = index
				break
			}
		}
		if duplicate < 0 {
			buckets[k] = append(buckets[k], len(plan.Nodes))
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
			mergedOwners[pair] = struct{}{}
			plan.Merges = append(plan.Merges, pair)
		}
	}
	for _, node := range plan.Nodes {
		plan.ByOwner[node.Owner] = append(plan.ByOwner[node.Owner], node.Node)
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
	plan, err := PlanCongruencePass(owned, key, equal, mergeEquivalent)
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
