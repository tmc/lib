package egraphcore

// CycleVisit runs before a selected node and its child edges are added to the
// traversal stack. work is one unit for the class and one for each child edge.
type CycleVisit[ID comparable] func(id ID, work int) error

// WouldCreateCycle reports whether selecting owner with childIDs would create
// a cycle in the currently selected graph. canonicalize maps class IDs to
// their current representatives; selected returns the chosen node for a
// class; children returns a node's child class IDs. Missing selected nodes
// terminate a path. If a required callback is nil, the function conservatively
// reports a cycle.
func WouldCreateCycle[ID comparable, Node any](
	owner ID,
	childIDs []ID,
	canonicalize func(ID) ID,
	selected func(ID) (Node, bool),
	children func(Node) []ID,
) bool {
	cycle, _ := WouldCreateCycleWithVisit(owner, childIDs, canonicalize, selected, children, nil)
	return cycle
}

// WouldCreateCycleWithVisit reports whether selecting owner with childIDs
// creates a cycle and lets callers bound each selected-node/child-edge scan.
// If visit returns an error, traversal stops and returns that error.
func WouldCreateCycleWithVisit[ID comparable, Node any](
	owner ID,
	childIDs []ID,
	canonicalize func(ID) ID,
	selected func(ID) (Node, bool),
	children func(Node) []ID,
	visit CycleVisit[ID],
) (bool, error) {
	if canonicalize == nil || selected == nil || children == nil {
		return true, nil
	}
	owner = canonicalize(owner)
	seen := make(map[ID]struct{})
	stack := append([]ID(nil), childIDs...)
	for len(stack) != 0 {
		last := len(stack) - 1
		id := canonicalize(stack[last])
		stack = stack[:last]
		if id == owner {
			return true, nil
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		node, ok := selected(id)
		work := 1
		var childIDs []ID
		if ok {
			childIDs = children(node)
			work += len(childIDs)
		}
		if visit != nil {
			if err := visit(id, work); err != nil {
				return false, err
			}
		}
		stack = append(stack, childIDs...)
	}
	return false, nil
}
