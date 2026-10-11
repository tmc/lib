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
	if visitCandidates == nil || find == nil || classNodes == nil || canonicalize == nil || equal == nil {
		var zero ID
		return zero, false, ErrInvalidMemoLookup
	}
	var small [8]ID
	smallCount := 0
	var seen map[ID]struct{}
	var foundID ID
	found := false
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
			seen[root] = struct{}{}
		} else if smallCount < len(small) {
			small[smallCount] = root
			smallCount++
		} else {
			seen = make(map[ID]struct{}, 16)
			for i := 0; i < smallCount; i++ {
				seen[small[i]] = struct{}{}
			}
			seen[root] = struct{}{}
		}
		for _, node := range classNodes(root) {
			if equal(canonicalize(node), want) {
				foundID = root
				found = true
				return false
			}
		}
		return true
	})
	return foundID, found, nil
}
