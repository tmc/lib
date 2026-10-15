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
	return WouldCreateCycleWithBudget(owner, childIDs, canonicalize, selected, children, visit, nil)
}

// WouldCreateCycleWithBudget is WouldCreateCycleWithVisit with allocation
// reservations through budget. Traversal storage remains owned by the
// budget's scope until it closes. Callbacks return borrowed data or reserve
// their own allocations. A nil budget preserves unlimited allocation.
func WouldCreateCycleWithBudget[ID comparable, Node any](
	owner ID,
	childIDs []ID,
	canonicalize func(ID) ID,
	selected func(ID) (Node, bool),
	children func(Node) []ID,
	visit CycleVisit[ID],
	budget *WorkBudget,
) (cycle bool, err error) {
	defer func() {
		if memoryErr := budget.MemoryError(); memoryErr != nil {
			cycle, err = false, memoryErr
		}
	}()
	if canonicalize == nil || selected == nil || children == nil {
		return true, nil
	}
	owner = canonicalize(owner)
	seen, ok := memoryMap[ID, struct{}](budget, 0)
	if !ok {
		return false, budget.MemoryError()
	}
	stack, ok := memorySlice[ID](budget, len(childIDs), len(childIDs))
	if !ok {
		return false, budget.MemoryError()
	}
	copy(stack, childIDs)
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
		if !memoryPut(budget, seen, id, struct{}{}) {
			return false, budget.MemoryError()
		}
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
		stack, ok = memoryAppend(budget, stack, childIDs...)
		if !ok {
			return false, budget.MemoryError()
		}
	}
	return false, nil
}
