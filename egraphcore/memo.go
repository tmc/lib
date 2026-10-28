package egraphcore

import "errors"

// ErrInvalidMemoLookup reports missing callbacks for a memo candidate lookup.
var ErrInvalidMemoLookup = errors.New("egraphcore: invalid memo lookup")

// FindMemoCandidate scans one hash-cons bucket and confirms node equality
// within each distinct canonical owner. The key has already selected the
// bucket; canonicalize and equal define the backend's node identity.
func FindMemoCandidate[ID ~uint32, Node any](
	candidates []ID,
	want Node,
	find func(ID) ID,
	classNodes func(ID) []Node,
	canonicalize func(Node) Node,
	equal func(Node, Node) bool,
) (ID, bool, error) {
	return FindMemoCandidateIter(func(visit func(ID) bool) {
		for _, candidate := range candidates {
			if !visit(candidate) {
				return
			}
		}
	}, want, find, classNodes, canonicalize, equal)
}

// FindMemoCandidateIter performs the same lookup while consuming candidates
// through visitCandidates, which can avoid exposing or copying a memo bucket.
func FindMemoCandidateIter[ID ~uint32, Node any](
	visitCandidates func(func(ID) bool),
	want Node,
	find func(ID) ID,
	classNodes func(ID) []Node,
	canonicalize func(Node) Node,
	equal func(Node, Node) bool,
) (ID, bool, error) {
	if canonicalize == nil {
		var zero ID
		return zero, false, ErrInvalidMemoLookup
	}
	return FindMemoCandidateIterWithBudget(visitCandidates, want, find, classNodes,
		func(node Node) (Node, error) { return canonicalize(node), nil }, equal, nil)
}

// FindMemoCandidateIterWithBudget admits lookup scratch before allocation.
// Allocating callbacks must admit their own storage using the same scope.
func FindMemoCandidateIterWithBudget[ID ~uint32, Node any](
	visitCandidates func(func(ID) bool), want Node, find func(ID) ID,
	classNodes func(ID) []Node, canonicalize func(Node) (Node, error),
	equal func(Node, Node) bool, budget *WorkBudget,
) (ID, bool, error) {
	if visitCandidates == nil || find == nil || classNodes == nil || canonicalize == nil || equal == nil {
		var zero ID
		return zero, false, ErrInvalidMemoLookup
	}
	var small [8]ID
	smallCount := 0
	var seen map[ID]struct{}
	var foundID ID
	found := false
	var lookupErr error
	visitCandidates(func(candidate ID) bool {
		root := find(candidate)
		if seen != nil {
			if _, ok := seen[root]; ok {
				return true
			}
		} else {
			duplicate := false
			for i := 0; i < smallCount; i++ {
				if small[i] == root {
					duplicate = true
					break
				}
			}
			if duplicate {
				return true
			}
		}
		if seen != nil {
			if !memoryPut(budget, seen, root, struct{}{}) {
				lookupErr = budget.MemoryError()
				return false
			}
		} else if smallCount < len(small) {
			small[smallCount] = root
			smallCount++
		} else {
			var ok bool
			seen, ok = memoryMap[ID, struct{}](budget, 16)
			if !ok {
				lookupErr = budget.MemoryError()
				return false
			}
			for i := 0; i < smallCount; i++ {
				if !memoryPut(budget, seen, small[i], struct{}{}) {
					lookupErr = budget.MemoryError()
					return false
				}
			}
			if !memoryPut(budget, seen, root, struct{}{}) {
				lookupErr = budget.MemoryError()
				return false
			}
		}
		for _, node := range classNodes(root) {
			canonical, err := canonicalize(node)
			if err != nil {
				lookupErr = err
				return false
			}
			if equal(canonical, want) {
				foundID = root
				found = true
				return false
			}
		}
		return true
	})
	return foundID, found, lookupErr
}
