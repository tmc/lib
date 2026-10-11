package egraphcore

import (
	"fmt"
	"sort"
)

// ChoiceSpace is an immutable, reachable snapshot of alternatives rooted at
// Root. Node payload ownership is defined by the clone callback supplied to
// NewChoiceSpace; callers should clone any mutable fields such as child slices.
type ChoiceSpace[ID ~uint32, Node any] struct {
	root         ID
	classes      []ID
	alternatives map[ID][]Node
	clone        func(Node) Node
}

// NewChoiceSpace prunes alternatives with missing children, orders the
// remaining alternatives, and retains only classes reachable from root.
// clone must return an independent node when Node contains mutable data.
// lessNode defines deterministic alternative order; equal nodes preserve their
// input order. Child IDs and node canonicalization remain backend concerns.
func NewChoiceSpace[ID ~uint32, Node any](
	root ID,
	alternatives map[ID][]Node,
	children func(Node) []ID,
	clone func(Node) Node,
	lessNode func(Node, Node) bool,
) (*ChoiceSpace[ID, Node], error) {
	if children == nil || clone == nil || lessNode == nil {
		return nil, ErrInvalidChoiceGraph
	}
	owned := make(map[ID][]Node, len(alternatives))
	for class, nodes := range alternatives {
		copied := make([]Node, len(nodes))
		for i, node := range nodes {
			copied[i] = clone(node)
		}
		owned[class] = copied
	}
	live, err := PruneEmptyAlternatives(owned, children)
	if err != nil {
		return nil, err
	}
	if len(live[root]) == 0 {
		return nil, fmt.Errorf("%w: root class %d has no live alternatives", ErrInvalidChoiceGraph, root)
	}
	space := &ChoiceSpace[ID, Node]{root: root, alternatives: make(map[ID][]Node), clone: clone}
	queue := []ID{root}
	seen := make(map[ID]bool)
	for len(queue) != 0 {
		class := queue[0]
		queue = queue[1:]
		if seen[class] {
			continue
		}
		seen[class] = true
		nodes := live[class]
		if len(nodes) == 0 {
			continue
		}
		sort.SliceStable(nodes, func(i, j int) bool { return lessNode(nodes[i], nodes[j]) })
		space.alternatives[class] = nodes
		space.classes = append(space.classes, class)
		for _, node := range nodes {
			queue = append(queue, children(node)...)
		}
	}
	sort.Slice(space.classes, func(i, j int) bool { return space.classes[i] < space.classes[j] })
	return space, nil
}

// Root returns the root class captured by the space.
func (s *ChoiceSpace[ID, Node]) Root() ID {
	if s == nil {
		var zero ID
		return zero
	}
	return s.root
}

// Classes returns reachable classes in ascending ID order.
func (s *ChoiceSpace[ID, Node]) Classes() []ID {
	if s == nil {
		return nil
	}
	return append([]ID(nil), s.classes...)
}

// Alternatives returns an independent slice of alternatives for class.
// Nodes themselves follow the ownership contract passed to NewChoiceSpace.
func (s *ChoiceSpace[ID, Node]) Alternatives(class ID) []Node {
	if s == nil {
		return nil
	}
	nodes := s.alternatives[class]
	result := make([]Node, len(nodes))
	for i, node := range nodes {
		result[i] = s.clone(node)
	}
	return result
}

// AlternativeCount reports the number of alternatives for class.
func (s *ChoiceSpace[ID, Node]) AlternativeCount(class ID) int {
	if s == nil {
		return 0
	}
	return len(s.alternatives[class])
}

// NewChoiceSpaceFrom builds a reachable choice space through backend callbacks.
// canonicalClass resolves union-find IDs; canonicalizeNode copies a node and
// canonicalizes backend-specific fields; keep filters backend-specific nodes.
// Each child returned by children must already be canonical.
func NewChoiceSpaceFrom[ID ~uint32, Node any](
	root ID,
	canonicalClass func(ID) (ID, error),
	alternatives func(ID) ([]Node, error),
	canonicalizeNode func(Node) (Node, error),
	children func(Node) []ID,
	keep func(Node) bool,
	clone func(Node) Node,
	lessNode func(Node, Node) bool,
) (*ChoiceSpace[ID, Node], error) {
	if canonicalClass == nil || alternatives == nil || canonicalizeNode == nil || children == nil || clone == nil || lessNode == nil {
		return nil, ErrInvalidChoiceGraph
	}
	canonicalRoot, err := canonicalClass(root)
	if err != nil {
		return nil, err
	}
	all := make(map[ID][]Node)
	queue := []ID{canonicalRoot}
	seen := make(map[ID]bool)
	for len(queue) != 0 {
		class, err := canonicalClass(queue[0])
		queue = queue[1:]
		if err != nil {
			return nil, err
		}
		if seen[class] {
			continue
		}
		seen[class] = true
		nodes, err := alternatives(class)
		if err != nil {
			return nil, err
		}
		for _, node := range nodes {
			if keep != nil && !keep(node) {
				continue
			}
			canonical, err := canonicalizeNode(clone(node))
			if err != nil {
				return nil, err
			}
			all[class] = append(all[class], canonical)
			queue = append(queue, children(canonical)...)
		}
	}
	return NewChoiceSpace(canonicalRoot, all, children, clone, lessNode)
}
