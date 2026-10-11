package egraphcore

import (
	"errors"
	"fmt"
	"sort"
)

// ErrInvalidChoiceGraph reports an incomplete or invalid selected choice.
var ErrInvalidChoiceGraph = errors.New("egraphcore: invalid selected choice graph")

// MaterializeChoice builds the acyclic graph selected by choice in postorder.
// The core validates each reachable choice, detects cycles, and reuses the
// built value when multiple parents refer to the same class. build supplies
// backend-specific value construction. It runs once per reachable class and
// receives child values in node child order. The alternatives and children
// callbacks must return stable results for the duration of the call. Class IDs
// in choice and children must already be canonical.
func MaterializeChoice[ID ~uint32, Node, Value any](
	root ID,
	choice map[ID]int,
	alternatives func(ID) []Node,
	children func(Node) []ID,
	build func(class ID, node Node, children []Value) (Value, error),
) (Value, error) {
	var zero Value
	if alternatives == nil || children == nil || build == nil {
		return zero, ErrInvalidChoiceGraph
	}
	type frame struct {
		class    ID
		node     Node
		children []ID
		next     int
	}
	selectNode := func(class ID) (Node, error) {
		var node Node
		nodes := alternatives(class)
		index, ok := choice[class]
		if !ok {
			return node, fmt.Errorf("%w: missing choice for class %d", ErrInvalidChoiceGraph, class)
		}
		if index < 0 || index >= len(nodes) {
			return node, fmt.Errorf("%w: alternative %d out of range for class %d", ErrInvalidChoiceGraph, index, class)
		}
		return nodes[index], nil
	}
	state := make(map[ID]uint8, len(choice))
	values := make(map[ID]Value, len(choice))
	rootNode, err := selectNode(root)
	if err != nil {
		return zero, err
	}
	state[root] = 1
	stack := []frame{{class: root, node: rootNode, children: append([]ID(nil), children(rootNode)...)}}
	for len(stack) != 0 {
		last := len(stack) - 1
		current := &stack[last]
		if current.next < len(current.children) {
			child := current.children[current.next]
			switch state[child] {
			case 1:
				return zero, fmt.Errorf("%w: cycle at class %d", ErrInvalidChoiceGraph, child)
			case 2:
				current.next++
			default:
				childNode, err := selectNode(child)
				if err != nil {
					return zero, err
				}
				state[child] = 1
				stack = append(stack, frame{class: child, node: childNode, children: append([]ID(nil), children(childNode)...)})
			}
			continue
		}
		childValues := make([]Value, len(current.children))
		for i, child := range current.children {
			childValues[i] = values[child]
		}
		value, err := build(current.class, current.node, childValues)
		if err != nil {
			return zero, fmt.Errorf("egraphcore: build selected class %d: %w", current.class, err)
		}
		values[current.class] = value
		state[current.class] = 2
		stack[last] = frame{}
		stack = stack[:last]
	}
	return values[root], nil
}

type choiceDependent[ID ~uint32] struct {
	class ID
	node  int
}

// PruneEmptyAlternatives removes alternatives whose children have no
// alternatives, propagating newly empty classes through reverse dependencies.
// The input map and slices are left unchanged. Each alternative and distinct
// child edge is processed at most once.
func PruneEmptyAlternatives[ID ~uint32, Node any](alternatives map[ID][]Node, children func(Node) []ID) (map[ID][]Node, error) {
	if children == nil {
		return nil, ErrInvalidChoiceGraph
	}
	remaining := make(map[ID]int, len(alternatives))
	removed := make(map[ID][]bool, len(alternatives))
	dependents := make(map[ID][]choiceDependent[ID])
	empty := make([]ID, 0)
	enqueued := make(map[ID]bool)
	enqueueEmpty := func(class ID) {
		if !enqueued[class] {
			enqueued[class] = true
			empty = append(empty, class)
		}
	}
	for class, nodes := range alternatives {
		remaining[class] = len(nodes)
		removed[class] = make([]bool, len(nodes))
		if len(nodes) == 0 {
			enqueueEmpty(class)
		}
	}
	for class, nodes := range alternatives {
		for nodeIndex, node := range nodes {
			seen := make(map[ID]bool)
			for _, child := range children(node) {
				if seen[child] {
					continue
				}
				seen[child] = true
				dependents[child] = append(dependents[child], choiceDependent[ID]{class: class, node: nodeIndex})
				if _, ok := alternatives[child]; !ok || len(alternatives[child]) == 0 {
					enqueueEmpty(child)
				}
			}
		}
	}
	for head := 0; head < len(empty); head++ {
		for _, dependent := range dependents[empty[head]] {
			if removed[dependent.class][dependent.node] {
				continue
			}
			removed[dependent.class][dependent.node] = true
			remaining[dependent.class]--
			if remaining[dependent.class] == 0 {
				enqueueEmpty(dependent.class)
			}
		}
	}
	pruned := make(map[ID][]Node, len(alternatives))
	for class, nodes := range alternatives {
		kept := make([]Node, 0, remaining[class])
		for index, node := range nodes {
			if !removed[class][index] {
				kept = append(kept, node)
			}
		}
		pruned[class] = kept
	}
	return pruned, nil
}

// SelectedChoiceGraph builds the reachable directed graph induced by one
// alternative selected for each visited class. Repeated child classes in one
// node contribute one edge. The callbacks define backend node storage.
func SelectedChoiceGraph[ID ~uint32, Node any](
	root ID,
	choice map[ID]int,
	alternatives func(ID) []Node,
	children func(Node) []ID,
) (map[ID][]ID, map[ID]bool, error) {
	if alternatives == nil || children == nil {
		return nil, nil, ErrInvalidChoiceGraph
	}
	adj := make(map[ID][]ID)
	reachable := make(map[ID]bool)
	stack := []ID{root}
	for len(stack) != 0 {
		class := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if reachable[class] {
			continue
		}
		reachable[class] = true
		nodes := alternatives(class)
		index, ok := choice[class]
		if !ok {
			return nil, nil, fmt.Errorf("%w: missing choice for class %d", ErrInvalidChoiceGraph, class)
		}
		if index < 0 || index >= len(nodes) {
			return nil, nil, fmt.Errorf("%w: alternative %d out of range for class %d", ErrInvalidChoiceGraph, index, class)
		}
		seen := make(map[ID]bool)
		for _, child := range children(nodes[index]) {
			if !seen[child] {
				adj[class] = append(adj[class], child)
				seen[child] = true
			}
			stack = append(stack, child)
		}
	}
	return adj, reachable, nil
}

// RepairChoiceCycles changes selected alternatives until the reachable choice
// graph is acyclic. allowed may restrict replacement alternatives; a nil
// callback permits all alternatives. pick must return an index in [0,n).
// maxRounds bounds repair attempts and must be positive.
func RepairChoiceCycles[ID ~uint32, Node any](
	root ID,
	choice map[ID]int,
	alternatives func(ID) []Node,
	children func(Node) []ID,
	allowed func(ID, int) bool,
	pick func(n int) int,
	maxRounds int,
) (map[ID]int, error) {
	if alternatives == nil || children == nil || pick == nil || maxRounds <= 0 {
		return nil, ErrInvalidChoiceGraph
	}
	repaired := make(map[ID]int, len(choice))
	for class, index := range choice {
		repaired[class] = index
	}
	for round := 0; round < maxRounds; round++ {
		adj, reachable, err := SelectedChoiceGraph(root, repaired, alternatives, children)
		if err != nil {
			return nil, err
		}
		components := CyclicComponents(adj, reachable)
		if len(components) == 0 {
			return repaired, nil
		}
		changed := false
		for _, component := range components {
			inside := make(map[ID]bool, len(component))
			for _, class := range component {
				inside[class] = true
			}
			for _, class := range component {
				nodes := alternatives(class)
				selected := repaired[class]
				selectedScore := CountInternalChildren(children(nodes[selected]), inside)
				bestReduction := 0
				var best []int
				for index, node := range nodes {
					if index == selected || (allowed != nil && !allowed(class, index)) {
						continue
					}
					reduction := selectedScore - CountInternalChildren(children(node), inside)
					if reduction <= 0 {
						continue
					}
					if reduction > bestReduction {
						bestReduction = reduction
						best = best[:0]
					}
					if reduction == bestReduction {
						best = append(best, index)
					}
				}
				if len(best) != 0 {
					selected := pick(len(best))
					if selected < 0 || selected >= len(best) {
						return nil, fmt.Errorf("%w: picker returned %d for %d alternatives", ErrInvalidChoiceGraph, selected, len(best))
					}
					repaired[class] = best[selected]
					changed = true
				}
			}
		}
		if !changed {
			return nil, fmt.Errorf("%w: cyclic choice has no allowed cycle-reducing alternative", ErrInvalidChoiceGraph)
		}
	}
	return nil, fmt.Errorf("%w: cycle repair exceeded %d rounds", ErrInvalidChoiceGraph, maxRounds)
}

// CyclicComponents returns the strongly connected components that contain a
// cycle in the reachable subgraph. Results are sorted by class ID and use an
// iterative Kosaraju traversal to handle deep graphs without recursion.
func CyclicComponents[ID ~uint32](adj map[ID][]ID, reachable map[ID]bool) [][]ID {
	reverse := make(map[ID][]ID, len(reachable))
	for class := range reachable {
		for _, child := range adj[class] {
			if reachable[child] {
				reverse[child] = append(reverse[child], class)
			}
		}
	}
	visited := make(map[ID]bool, len(reachable))
	finish := make([]ID, 0, len(reachable))
	for start := range reachable {
		if visited[start] {
			continue
		}
		stack := []struct {
			id   ID
			exit bool
		}{{id: start}}
		for len(stack) != 0 {
			item := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if item.exit {
				finish = append(finish, item.id)
				continue
			}
			if visited[item.id] {
				continue
			}
			visited[item.id] = true
			stack = append(stack, struct {
				id   ID
				exit bool
			}{id: item.id, exit: true})
			for _, child := range adj[item.id] {
				if reachable[child] && !visited[child] {
					stack = append(stack, struct {
						id   ID
						exit bool
					}{id: child})
				}
			}
		}
	}
	clear(visited)
	components := make([][]ID, 0)
	for i := len(finish) - 1; i >= 0; i-- {
		start := finish[i]
		if visited[start] {
			continue
		}
		component := make([]ID, 0, 1)
		stack := []ID{start}
		visited[start] = true
		for len(stack) != 0 {
			id := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			component = append(component, id)
			for _, parent := range reverse[id] {
				if !visited[parent] {
					visited[parent] = true
					stack = append(stack, parent)
				}
			}
		}
		if len(component) > 1 || hasChoiceEdge(adj[start], start) {
			sort.Slice(component, func(i, j int) bool { return component[i] < component[j] })
			components = append(components, component)
		}
	}
	sort.Slice(components, func(i, j int) bool { return components[i][0] < components[j][0] })
	return components
}

// CountInternalChildren counts distinct children of node that belong to
// component.
func CountInternalChildren[ID ~uint32](children []ID, component map[ID]bool) int {
	seen := make(map[ID]bool)
	for _, child := range children {
		if component[child] {
			seen[child] = true
		}
	}
	return len(seen)
}

func hasChoiceEdge[ID comparable](ids []ID, want ID) bool {
	for _, id := range ids {
		if id == want {
			return true
		}
	}
	return false
}
