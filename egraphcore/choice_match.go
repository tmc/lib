package egraphcore

// SelectedChoiceMatches reports whether the selected choice graph rooted at
// root has the same structure as valueRoot. equal compares one selected node
// with its corresponding value; valueChildren returns the value's children in
// the same order as children returns the selected node's child classes.
//
// The selected-choice callbacks must return stable results for the duration
// of the call. Class IDs in choice and children must already be canonical.
// Invalid choices, missing callbacks, and mismatching values return false.
// Repeated class/value pairs are compared once, so shared or cyclic structures
// do not require expansion into trees.
func SelectedChoiceMatches[ID ~uint32, Node any, Value comparable](
	root ID,
	choice map[ID]int,
	alternatives func(ID) []Node,
	children func(Node) []ID,
	valueRoot Value,
	valueChildren func(Value) []Value,
	equal func(ID, Node, Value) bool,
) bool {
	if alternatives == nil || children == nil || valueChildren == nil || equal == nil {
		return false
	}
	type pair struct {
		class ID
		value Value
	}
	stack := []pair{{class: root, value: valueRoot}}
	seen := make(map[pair]bool)
	for len(stack) != 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[current] {
			continue
		}
		seen[current] = true

		nodes := alternatives(current.class)
		index, ok := choice[current.class]
		if !ok || index < 0 || index >= len(nodes) {
			return false
		}
		node := nodes[index]
		valueKids := valueChildren(current.value)
		nodeKids := children(node)
		if len(valueKids) != len(nodeKids) || !equal(current.class, node, current.value) {
			return false
		}
		for i, child := range nodeKids {
			stack = append(stack, pair{class: child, value: valueKids[i]})
		}
	}
	return true
}
