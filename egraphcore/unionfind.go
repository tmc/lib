package egraphcore

// Find returns the root of id in parent and compresses the traversed path.
//
// The caller must ensure parent is not concurrently modified. If id is out of
// range, Find returns id unchanged. This lets graph wrappers retain their own
// invalid-ID contracts while sharing the path-compression operation.
func Find[ID ~uint32](parent []ID, id ID) ID {
	root := id
	for uint64(root) < uint64(len(parent)) && parent[root] != root {
		root = parent[root]
	}

	for current := id; uint64(current) < uint64(len(parent)) && current != root; {
		next := parent[current]
		parent[current] = root
		current = next
	}
	return root
}
