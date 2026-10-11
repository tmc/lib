package egraphcore

// LinkRoots joins two distinct roots in parent by making newRoot the parent
// of the other root. newRoot must be one of the supplied roots. Callers that
// use Store can select roots with LinkRootsBySize to share deterministic
// union-by-size behavior.
//
// LinkRoots returns false when the roots are equal, either root is out of
// range, or newRoot is not one of the roots.
func LinkRoots[ID ~uint32](parent []ID, root1, root2, newRoot ID) (ID, bool) {
	if root1 == root2 {
		return root1, false
	}
	if uint64(root1) >= uint64(len(parent)) || uint64(root2) >= uint64(len(parent)) {
		return root1, false
	}
	if newRoot != root1 && newRoot != root2 {
		return root1, false
	}
	other := root2
	if newRoot == root2 {
		other = root1
	}
	parent[other] = newRoot
	return newRoot, true
}
