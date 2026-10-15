package egraphcore

import "errors"

// ErrMatchWorkLimit reports that structural pattern matching exhausted its
// work budget before it could determine all matches.
var ErrMatchWorkLimit = errors.New("egraphcore: pattern match work limit exceeded")

// ErrInvalidPattern reports a nil pattern or incomplete matcher callback.
var ErrInvalidPattern = errors.New("egraphcore: invalid pattern matcher input")

// Pattern is a structural term pattern. Variables bind canonical class IDs;
// non-variable leaves match zero-child nodes with the same operator.
type Pattern struct {
	Op       string
	Variable bool
	Children []*Pattern
}

// Match is one structural match rooted at an equivalence class.
type Match[ID comparable] struct {
	ClassID ID
	Subst   map[string]ID
}

// MatcherCallbacks expose graph and pattern operations to the shared
// structural matcher. Pattern is backend-defined, so adapters can retain sort,
// payload, and other pattern metadata. ClassEligible may reject a class for a
// pattern. Candidates may use backend indexes to preserve efficient node
// selection and order. NodeEligible decides whether a candidate satisfies
// node-specific pattern constraints; the matcher separately checks arity. A
// nil ClassEligible accepts every class. All other callbacks are required.
type MatcherCallbacks[ID comparable, Node, P any] struct {
	Find            func(ID) ID
	ValidPattern    func(P) bool
	IsVariable      func(P) (name string, variable bool)
	PatternChildren func(P) []P
	ClassEligible   func(ID, P) bool
	Candidates      func(ID, P) []Node
	NodeChildren    func(Node) []ID
	NodeEligible    func(Node, P) bool
}

// MatchEClass finds every match of pattern rooted at classID. The callbacks
// expose backend-specific graph storage while keeping matching shared. Current
// bindings are copied and canonicalized before matching and are not modified.
// work is shared across recursive calls and may be reused across root classes.
func MatchEClass[ID comparable, Node any](
	pattern *Pattern,
	classID ID,
	current map[string]ID,
	budget *WorkBudget,
	find func(ID) ID,
	nodes func(ID) []Node,
	op func(Node) string,
	children func(Node) []ID,
) ([]map[string]ID, error) {
	if pattern == nil || budget == nil || find == nil || nodes == nil || op == nil || children == nil {
		return nil, ErrInvalidPattern
	}
	return MatchEClassWithMatcher(pattern, classID, current, budget, MatcherCallbacks[ID, Node, *Pattern]{
		Find:         find,
		ValidPattern: func(pattern *Pattern) bool { return validPatternWithBudget(pattern, budget) },
		IsVariable: func(pattern *Pattern) (string, bool) {
			return pattern.Op, pattern.Variable
		},
		PatternChildren: func(pattern *Pattern) []*Pattern { return pattern.Children },
		Candidates:      func(id ID, _ *Pattern) []Node { return nodes(id) },
		NodeEligible: func(node Node, candidate *Pattern) bool {
			return op(node) == candidate.Op
		},
		NodeChildren: children,
	})
}

// MatchEClassWithMatcher finds every match rooted at classID using a
// backend-defined pattern type. Current bindings are copied and canonicalized
// before matching and are not modified. Matching uses an explicit work stack,
// so deeply nested patterns do not consume the Go call stack.
func MatchEClassWithMatcher[ID comparable, Node, P any](
	pattern P,
	classID ID,
	current map[string]ID,
	budget *WorkBudget,
	callbacks MatcherCallbacks[ID, Node, P],
) ([]map[string]ID, error) {
	if budget == nil || callbacks.Find == nil || callbacks.ValidPattern == nil || callbacks.IsVariable == nil || callbacks.PatternChildren == nil || callbacks.Candidates == nil || callbacks.NodeEligible == nil || callbacks.NodeChildren == nil {
		return nil, ErrInvalidPattern
	}
	if budget.Exhausted() {
		return nil, ErrMatchWorkLimit
	}
	if !callbacks.ValidPattern(pattern) {
		return nil, withMemoryError(ErrInvalidPattern, budget)
	}
	results, _, err := matchEClassWithCallbacks(pattern, classID, current, budget, 0, callbacks)
	return results, err
}

// Search applies pattern to roots in order. A positive maxMatches limits the
// returned matches; reaching that count sets limited. Work is shared across
// every root, and an exhausted budget returns ErrMatchWorkLimit without a
// partial result.
func Search[ID comparable, Node any](
	pattern *Pattern,
	roots []ID,
	budget *WorkBudget,
	maxMatches int,
	find func(ID) ID,
	nodes func(ID) []Node,
	op func(Node) string,
	children func(Node) []ID,
) ([]Match[ID], bool, error) {
	if pattern == nil || budget == nil || find == nil || nodes == nil || op == nil || children == nil || maxMatches < 0 {
		return nil, false, ErrInvalidPattern
	}
	callbacks := MatcherCallbacks[ID, Node, *Pattern]{
		Find:         find,
		ValidPattern: func(pattern *Pattern) bool { return validPatternWithBudget(pattern, budget) },
		IsVariable: func(pattern *Pattern) (string, bool) {
			return pattern.Op, pattern.Variable
		},
		PatternChildren: func(pattern *Pattern) []*Pattern { return pattern.Children },
		Candidates:      func(id ID, _ *Pattern) []Node { return nodes(id) },
		NodeEligible: func(node Node, candidate *Pattern) bool {
			return op(node) == candidate.Op
		},
		NodeChildren: children,
	}
	return SearchWithMatcher(pattern, roots, budget, maxMatches, callbacks)
}

// SearchWithMatcher applies pattern to roots in order using backend supplied
// class eligibility, candidate selection, and node predicates. A positive
// maxMatches limits returned matches; reaching that count sets limited. Work
// is shared across all roots, and exhaustion returns no partial result.
func SearchWithMatcher[ID comparable, Node, P any](
	pattern P,
	roots []ID,
	budget *WorkBudget,
	maxMatches int,
	callbacks MatcherCallbacks[ID, Node, P],
) ([]Match[ID], bool, error) {
	if budget == nil || callbacks.Find == nil || callbacks.ValidPattern == nil || callbacks.IsVariable == nil || callbacks.PatternChildren == nil || callbacks.Candidates == nil || callbacks.NodeEligible == nil || callbacks.NodeChildren == nil || maxMatches < 0 {
		return nil, false, ErrInvalidPattern
	}
	if !callbacks.ValidPattern(pattern) {
		return nil, false, withMemoryError(ErrInvalidPattern, budget)
	}
	if budget.Exhausted() {
		return nil, false, ErrMatchWorkLimit
	}
	var matches []Match[ID]
	for _, root := range roots {
		remaining := 0
		if maxMatches > 0 {
			remaining = maxMatches - len(matches)
			if remaining <= 0 {
				return matches, true, nil
			}
		}
		substitutions, limited, err := matchEClassWithCallbacks(pattern, root, nil, budget, remaining, callbacks)
		if err != nil {
			return nil, false, err
		}
		for _, subst := range substitutions {
			var ok bool
			matches, ok = memoryAppend(budget, matches, Match[ID]{ClassID: callbacks.Find(root), Subst: subst})
			if !ok {
				return nil, false, matchMemoryError(budget)
			}
			if maxMatches > 0 && len(matches) >= maxMatches {
				return matches, true, nil
			}
		}
		if limited {
			return matches, true, nil
		}
	}
	return matches, false, nil
}

func matchEClassWithCallbacks[ID comparable, Node, P any](
	pattern P,
	classID ID,
	current map[string]ID,
	budget *WorkBudget,
	maxMatches int,
	callbacks MatcherCallbacks[ID, Node, P],
) ([]map[string]ID, bool, error) {
	type task struct {
		pattern P
		classID ID
		next    *task
	}
	type state struct {
		pending *task
		subst   map[string]ID
	}
	var initial map[string]ID
	if current != nil {
		var ok bool
		initial, ok = memoryMap[string, ID](budget, len(current))
		if !ok {
			return nil, false, matchMemoryError(budget)
		}
	}
	for name, id := range current {
		if !budget.TrySpend(1) {
			return nil, false, ErrMatchWorkLimit
		}
		if name == "" {
			return nil, false, ErrInvalidPattern
		}
		if !memoryPut(budget, initial, name, callbacks.Find(id)) {
			return nil, false, matchMemoryError(budget)
		}
	}
	first, ok := memoryObject(budget, task{pattern: pattern, classID: classID})
	if !ok {
		return nil, false, matchMemoryError(budget)
	}
	stack, ok := memorySlice[state](budget, 1, 1)
	if !ok {
		return nil, false, matchMemoryError(budget)
	}
	stack[0] = state{pending: first, subst: initial}
	var results []map[string]ID
	limited := false
	for len(stack) > 0 {
		last := len(stack) - 1
		work := stack[last]
		stack = stack[:last]
		if work.pending == nil {
			subst, ok := cloneSubstWithBudget(work.subst, budget)
			if !ok {
				return nil, false, matchMemoryError(budget)
			}
			results, ok = memoryAppend(budget, results, subst)
			if !ok {
				return nil, false, matchMemoryError(budget)
			}
			if maxMatches > 0 && len(results) >= maxMatches {
				limited = true
				break
			}
			continue
		}
		if !budget.TrySpend(1) {
			return nil, false, ErrMatchWorkLimit
		}
		item := work.pending
		remaining := item.next
		canonicalID := callbacks.Find(item.classID)
		if callbacks.ClassEligible != nil && !callbacks.ClassEligible(canonicalID, item.pattern) {
			continue
		}
		name, variable := callbacks.IsVariable(item.pattern)
		if variable {
			if name == "" {
				return nil, false, ErrInvalidPattern
			}
			if boundID, ok := work.subst[name]; ok {
				if callbacks.Find(boundID) == canonicalID {
					stack, ok = memoryAppend(budget, stack, state{pending: remaining, subst: work.subst})
					if !ok {
						return nil, false, matchMemoryError(budget)
					}
				}
				continue
			}
			subst, ok := cloneSubstWithBudget(work.subst, budget)
			if !ok || !memoryPut(budget, subst, name, canonicalID) {
				return nil, false, matchMemoryError(budget)
			}
			stack, ok = memoryAppend(budget, stack, state{pending: remaining, subst: subst})
			if !ok {
				return nil, false, matchMemoryError(budget)
			}
			continue
		}
		children := callbacks.PatternChildren(item.pattern)
		candidates := callbacks.Candidates(canonicalID, item.pattern)
		// Push in reverse so the LIFO state stack visits candidates in the
		// adapter-provided order.
		for i := len(candidates) - 1; i >= 0; i-- {
			if !budget.TrySpend(1) {
				return nil, false, ErrMatchWorkLimit
			}
			node := candidates[i]
			if !callbacks.NodeEligible(node, item.pattern) {
				continue
			}
			childIDs := callbacks.NodeChildren(node)
			if len(childIDs) != len(children) {
				continue
			}
			pending := remaining
			for child := len(children) - 1; child >= 0; child-- {
				var ok bool
				pending, ok = memoryObject(budget, task{pattern: children[child], classID: childIDs[child], next: pending})
				if !ok {
					return nil, false, matchMemoryError(budget)
				}
			}
			var ok bool
			stack, ok = memoryAppend(budget, stack, state{pending: pending, subst: work.subst})
			if !ok {
				return nil, false, matchMemoryError(budget)
			}
		}
	}
	return results, limited, nil
}

func validPattern(pattern *Pattern) bool {
	return validPatternWithBudget(pattern, nil)
}

func validPatternWithBudget(pattern *Pattern, budget *WorkBudget) bool {
	type entry struct {
		pattern *Pattern
		exit    bool
	}
	stack, ok := memorySlice[entry](budget, 1, 1)
	if !ok {
		return false
	}
	stack[0] = entry{pattern: pattern}
	state, ok := memoryMap[*Pattern, uint8](budget, 0)
	if !ok {
		return false
	}
	for len(stack) > 0 {
		last := len(stack) - 1
		current := stack[last]
		stack = stack[:last]
		if current.pattern == nil {
			return false
		}
		if current.exit {
			if !memoryPut(budget, state, current.pattern, uint8(2)) {
				return false
			}
			continue
		}
		switch state[current.pattern] {
		case 1:
			return false
		case 2:
			continue
		}
		if !memoryPut(budget, state, current.pattern, uint8(1)) {
			return false
		}
		if current.pattern.Variable && len(current.pattern.Children) != 0 {
			return false
		}
		stack, ok = memoryAppend(budget, stack, entry{pattern: current.pattern, exit: true})
		if !ok {
			return false
		}
		for i := len(current.pattern.Children) - 1; i >= 0; i-- {
			child := current.pattern.Children[i]
			if child == nil {
				return false
			}
			stack, ok = memoryAppend(budget, stack, entry{pattern: child})
			if !ok {
				return false
			}
		}
	}
	return true
}

func cloneSubst[ID comparable](subst map[string]ID) map[string]ID {
	copy := make(map[string]ID, len(subst)+1)
	for name, id := range subst {
		copy[name] = id
	}
	return copy
}

func cloneSubstWithBudget[ID comparable](subst map[string]ID, budget *WorkBudget) (map[string]ID, bool) {
	copy, ok := memoryMap[string, ID](budget, len(subst)+1)
	if !ok {
		return nil, false
	}
	for name, id := range subst {
		if !memoryPut(budget, copy, name, id) {
			return nil, false
		}
	}
	return copy, true
}

func matchMemoryError(budget *WorkBudget) error {
	return withMemoryError(ErrMatchWorkLimit, budget)
}

func withMemoryError(base error, budget *WorkBudget) error {
	if budget.MemoryError() == nil {
		return base
	}
	return errors.Join(base, budget.MemoryError())
}
