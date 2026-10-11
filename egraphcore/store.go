package egraphcore

import "math"

// Store owns the common, representation-independent state of an e-graph.
// Class values retain each backend's node representation and metadata.
// Callers must serialize mutations to a Store.
type Store[ID ~uint32, Class any] struct {
	parent  []ID
	size    []uint64
	classes map[ID]Class
	memo    map[string][]ID
}

// NewStore creates an initialized graph store.
func NewStore[ID ~uint32, Class any]() *Store[ID, Class] {
	return &Store[ID, Class]{
		classes: make(map[ID]Class),
		memo:    make(map[string][]ID),
	}
}

// Len reports the number of class IDs allocated by the store.
func (s *Store[ID, Class]) Len() int {
	if s == nil {
		return 0
	}
	return len(s.parent)
}

// Find returns the canonical class ID and compresses the traversed path.
func (s *Store[ID, Class]) Find(id ID) ID {
	if s == nil {
		return id
	}
	return Find(s.parent, id)
}

// RootPairStatus reports the result of resolving two class IDs in a Store.
type RootPairStatus uint8

const (
	RootPairInvalidID RootPairStatus = iota + 1
	RootPairSame
	RootPairMissingClass
	RootPairDistinct
)

// ResolveRoots validates two IDs and returns their canonical roots. Same-root
// pairs are reported before class-record checks, matching checked union's
// no-op behavior. Distinct roots require class records for both IDs.
func (s *Store[ID, Class]) ResolveRoots(id1, id2 ID) (ID, ID, RootPairStatus) {
	if s == nil || uint64(id1) >= uint64(len(s.parent)) || uint64(id2) >= uint64(len(s.parent)) {
		return id1, id2, RootPairInvalidID
	}
	root1, root2 := s.Find(id1), s.Find(id2)
	if root1 == root2 {
		return root1, root2, RootPairSame
	}
	if _, ok := s.LookupClass(root1); !ok {
		return root1, root2, RootPairMissingClass
	}
	if _, ok := s.LookupClass(root2); !ok {
		return root1, root2, RootPairMissingClass
	}
	return root1, root2, RootPairDistinct
}

// FindWithoutCompression inspects at most maxVisits parent entries and leaves
// the store unchanged. visits includes the root entry. It returns false for
// invalid IDs, exhausted budgets, or malformed parent cycles.
func (s *Store[ID, Class]) FindWithoutCompression(id ID, maxVisits int) (root ID, visits int, ok bool) {
	if s == nil || maxVisits <= 0 {
		return id, 0, false
	}
	for {
		if visits >= maxVisits || uint64(id) >= uint64(len(s.parent)) {
			return id, visits, false
		}
		visits++
		next := s.parent[id]
		if next == id {
			return id, visits, true
		}
		id = next
	}
}

// CanonicalChildren returns the canonical representatives for children in
// their original order.
func (s *Store[ID, Class]) CanonicalChildren(children []ID) []ID {
	canonical := make([]ID, len(children))
	for i, child := range children {
		canonical[i] = s.Find(child)
	}
	return canonical
}

// addID allocates a new canonical class ID. It returns false when the uint32
// ID space is exhausted or the store is nil.
func (s *Store[ID, Class]) addID() (ID, bool) {
	var zero ID
	if s == nil || uint64(len(s.parent)) > uint64(math.MaxUint32) {
		return zero, false
	}
	id := ID(len(s.parent))
	s.parent = append(s.parent, id)
	s.size = append(s.size, 1)
	return id, true
}

// Insert allocates an ID, constructs its class record, and adds the ID to the
// selected memo bucket. The caller supplies the backend-specific key and class
// construction policy.
func (s *Store[ID, Class]) Insert(key string, newClass func(ID) Class) (ID, bool) {
	var zero ID
	if s == nil || newClass == nil {
		return zero, false
	}
	id, ok := s.addID()
	if !ok {
		return zero, false
	}
	if s.classes == nil {
		s.classes = make(map[ID]Class)
	}
	if s.memo == nil {
		s.memo = make(map[string][]ID)
	}
	s.classes[id] = newClass(id)
	s.memo[key] = append(s.memo[key], id)
	return id, true
}

// LinkRoots makes winner the parent of the other root.
func (s *Store[ID, Class]) LinkRoots(root1, root2, winner ID) (ID, bool) {
	if s == nil || s.Find(root1) != root1 || s.Find(root2) != root2 {
		return root1, false
	}
	root, linked := LinkRoots(s.parent, root1, root2, winner)
	if linked && uint64(root1) < uint64(len(s.size)) && uint64(root2) < uint64(len(s.size)) {
		s.size[winner] = s.size[root1] + s.size[root2]
		loser := root2
		if winner == root2 {
			loser = root1
		}
		s.size[loser] = 0
	}
	return root, linked
}

// LinkRootsBySize links the smaller root below the larger root. Equal-size
// roots are resolved by ID, keeping the result deterministic across backends.
func (s *Store[ID, Class]) LinkRootsBySize(root1, root2 ID) (ID, bool) {
	if s == nil || uint64(root1) >= uint64(len(s.size)) || uint64(root2) >= uint64(len(s.size)) {
		return root1, false
	}
	winner, loser := root1, root2
	if s.size[root1] < s.size[root2] || (s.size[root1] == s.size[root2] && root1 > root2) {
		winner, loser = root2, root1
	}
	return s.LinkRoots(winner, loser, winner)
}

// MergeRoots links two existing class roots, combines their class records,
// and removes the losing record. The winner-selection and record-merge
// policies remain with the backend.
func (s *Store[ID, Class]) MergeRoots(root1, root2, winner ID, merge func(Class, Class) Class) (ID, bool) {
	var zero ID
	if s == nil || root1 == root2 || (winner != root1 && winner != root2) ||
		s.Find(root1) != root1 || s.Find(root2) != root2 {
		return zero, false
	}
	class1, ok1 := s.classes[root1]
	class2, ok2 := s.classes[root2]
	if !ok1 || !ok2 {
		return zero, false
	}
	if _, linked := s.LinkRoots(root1, root2, winner); !linked {
		return zero, false
	}
	other := root2
	if winner == root2 {
		other = root1
	}
	winnerClass, loserClass := class1, class2
	if winner == root2 {
		winnerClass, loserClass = class2, class1
	}
	if merge != nil {
		winnerClass = merge(winnerClass, loserClass)
	}
	s.classes[winner] = winnerClass
	delete(s.classes, other)
	return winner, true
}

// MergeRootsBySize merges two classes using the shared deterministic
// union-by-size policy. The merge callback combines backend-owned class data.
func (s *Store[ID, Class]) MergeRootsBySize(root1, root2 ID, merge func(Class, Class) Class) (ID, bool) {
	if s == nil || uint64(root1) >= uint64(len(s.size)) || uint64(root2) >= uint64(len(s.size)) {
		return root1, false
	}
	winner, loser := root1, root2
	if s.size[root1] < s.size[root2] || (s.size[root1] == s.size[root2] && root1 > root2) {
		winner, loser = root2, root1
	}
	return s.MergeRoots(winner, loser, winner, merge)
}

// Initialized reports whether the class and memo tables have been created.
func (s *Store[ID, Class]) Initialized() bool {
	return s != nil && s.classes != nil && s.memo != nil
}

// Class returns the class record stored for id, or its zero value when absent.
func (s *Store[ID, Class]) Class(id ID) Class {
	var zero Class
	class, ok := s.LookupClass(id)
	if !ok {
		return zero
	}
	return class
}

// LookupClass returns the class record stored for id and whether it exists.
func (s *Store[ID, Class]) LookupClass(id ID) (Class, bool) {
	if s == nil {
		var zero Class
		return zero, false
	}
	class, ok := s.classes[id]
	return class, ok
}

// ClassCount reports the number of stored class records.
func (s *Store[ID, Class]) ClassCount() int {
	if s == nil {
		return 0
	}
	return len(s.classes)
}

// RangeClasses calls visit for each class until it returns false.
func (s *Store[ID, Class]) RangeClasses(visit func(ID, Class) bool) {
	if s == nil || visit == nil {
		return
	}
	for id, class := range s.classes {
		if !visit(id, class) {
			return
		}
	}
}

// VisitMemoCandidates visits the candidate bucket for key in insertion order.
// Returning false from visit stops the scan.
func (s *Store[ID, Class]) VisitMemoCandidates(key string, visit func(ID) bool) {
	if s == nil || visit == nil {
		return
	}
	for _, id := range s.memo[key] {
		if !visit(id) {
			return
		}
	}
}

// AddMemoCandidate appends id to the memo bucket selected by key.
func (s *Store[ID, Class]) AddMemoCandidate(key string, id ID) {
	if s != nil {
		if s.memo == nil {
			s.memo = make(map[string][]ID)
		}
		s.memo[key] = append(s.memo[key], id)
	}
}

// AddMemoCandidateIf appends id unless an existing candidate satisfies same.
// The callback lets each backend define canonical candidate identity.
func (s *Store[ID, Class]) AddMemoCandidateIf(key string, id ID, same func(ID, ID) bool) bool {
	if s == nil {
		return false
	}
	if same != nil {
		for _, candidate := range s.memo[key] {
			if same(candidate, id) {
				return false
			}
		}
	}
	s.AddMemoCandidate(key, id)
	return true
}

// ResetMemo discards all hash-cons buckets while retaining class storage.
func (s *Store[ID, Class]) ResetMemo() {
	if s != nil {
		s.memo = make(map[string][]ID)
	}
}

// Clone copies union-find and memo state and applies cloneClass to each class
// record. A nil cloneClass copies class values as-is.
func (s *Store[ID, Class]) Clone(cloneClass func(Class) Class) *Store[ID, Class] {
	if s == nil {
		return nil
	}
	clone := NewStore[ID, Class]()
	clone.parent = append(clone.parent, s.parent...)
	clone.size = append(clone.size, s.size...)
	for id, class := range s.classes {
		if cloneClass != nil {
			class = cloneClass(class)
		}
		clone.classes[id] = class
	}
	for key, ids := range s.memo {
		clone.memo[key] = append([]ID(nil), ids...)
	}
	return clone
}
