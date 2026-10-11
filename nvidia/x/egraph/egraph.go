package egraph

import (
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/tmc/lib/egraphcore"
)

var _ egraphcore.Graph[Id, ENode] = (*EGraph)(nil)

// Id identifies an equivalence class in an E-Graph.
type Id uint32

// ENode is an operator node within an equivalence class with child e-class IDs.
type ENode struct {
	Op       string
	Children []Id
	Data     any
	// Subsumed nodes remain stored and count toward graph size, but are hidden
	// from rule matching and extraction.
	Subsumed bool
}

// NewNode creates an operator ENode with child e-class IDs.
func NewNode(op string, children ...Id) ENode {
	return ENode{
		Op:       op,
		Children: children,
	}
}

// NewLeaf creates a leaf literal or symbol ENode with zero children.
func NewLeaf(val any) ENode {
	return ENode{
		Op:   fmt.Sprint(val),
		Data: val,
	}
}

// Key returns the canonical hash-consing key for an ENode given canonicalized child IDs.
func (n ENode) Key() string {
	dataKey := ""
	if n.Data != nil {
		dataKey = fmt.Sprintf("{%T:%s}", n.Data, dataValueKey(reflect.ValueOf(n.Data), make(map[visit]bool)))
	}
	if len(n.Children) == 0 {
		return n.Op + dataKey
	}
	var sb strings.Builder
	sb.WriteString(n.Op)
	sb.WriteString(dataKey)
	sb.WriteByte('(')
	for i, c := range n.Children {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(fmt.Sprint(c))
	}
	sb.WriteByte(')')
	return sb.String()
}

// dataValueKey is bounded for reference graphs so a legacy cyclic payload
// cannot make hash-consing recurse forever.  Non-reference values retain the
// usual Go representation used by the key.
func dataValueKey(v reflect.Value, seen map[visit]bool) string {
	if !v.IsValid() {
		return "<nil>"
	}
	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			return "<nil>"
		}
		return dataValueKey(v.Elem(), seen)
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if v.IsNil() {
			return "nil"
		}
		key := visitFor(v)
		if seen[key] {
			return "<cycle>"
		}
		seen[key] = true
		defer delete(seen, key)
		if v.Kind() == reflect.Map {
			parts := make([]string, 0, v.Len())
			for _, k := range v.MapKeys() {
				parts = append(parts, dataValueKey(k, seen)+":"+dataValueKey(v.MapIndex(k), seen))
			}
			sort.Strings(parts)
			return strings.Join(parts, ",")
		}
		if v.Kind() == reflect.Slice {
			parts := make([]string, v.Len())
			for i := range parts {
				parts[i] = dataValueKey(v.Index(i), seen)
			}
			return strings.Join(parts, ",")
		}
		return dataValueKey(v.Elem(), seen)
	case reflect.Struct:
		parts := make([]string, v.NumField())
		for i := range parts {
			if v.Field(i).CanInterface() {
				parts[i] = dataValueKey(v.Field(i), seen)
			} else {
				parts[i] = "<unexported>"
			}
		}
		return strings.Join(parts, ",")
	default:
		if v.CanInterface() {
			return fmt.Sprintf("%#v", v.Interface())
		}
		return "<opaque>"
	}
}

func (n ENode) String() string {
	return n.Key()
}

// Equal returns true if two ENodes have identical operator, data, and children.
func (n ENode) Equal(other ENode) bool {
	if n.Op != other.Op || len(n.Children) != len(other.Children) || !reflect.DeepEqual(n.Data, other.Data) {
		return false
	}
	for i := range n.Children {
		if n.Children[i] != other.Children[i] {
			return false
		}
	}
	return true
}

type parentRef struct {
	node    ENode
	classID Id
}

// EClass represents an equivalence class of equivalent ENodes.
type EClass struct {
	ID      Id
	Nodes   []ENode
	parents []parentRef
}

// EGraph represents an equivalence graph with congruence closure and hash consing.
type EGraph struct {
	store              *egraphcore.Store[Id, *EClass]
	worklist           []Id
	facts              *egraphcore.FactStore[Id]
	revision           uint64
	origins            map[string]string
	insertionOrigin    string
	insertionOriginSet bool
}

// New creates an empty E-Graph.
func New() *EGraph {
	g := &EGraph{
		store: egraphcore.NewStore[Id, *EClass](),
		facts: egraphcore.NewFactStore[Id](),
	}
	_, _ = g.facts.Register(typedMetadataSchema())
	return g
}

// Facts returns the graph-owned store of named typed relations.
// Relations are cloned with the graph and recanonicalized after unions.
func (g *EGraph) Facts() *egraphcore.FactStore[Id] {
	if g == nil {
		return nil
	}
	if g.facts == nil {
		g.facts = egraphcore.NewFactStore[Id]()
	}
	return g.facts
}

// Find finds the canonical representative Id for an equivalence class with path compression.
func (g *EGraph) Find(id Id) Id {
	if g == nil {
		return id
	}
	return g.storage().Find(id)
}

// storage returns the initialized common graph store. It also makes the zero
// value usable by mutation methods that need to add their first class.
func (g *EGraph) storage() *egraphcore.Store[Id, *EClass] {
	if g.store == nil {
		g.store = egraphcore.NewStore[Id, *EClass]()
	}
	return g.store
}

// Canonicalize returns an ENode where all child IDs are replaced by their canonical representatives.
func (g *EGraph) Canonicalize(n ENode) ENode {
	canon := ENode{
		Op:       n.Op,
		Data:     n.Data,
		Subsumed: n.Subsumed,
		Children: make([]Id, len(n.Children)),
	}
	for i, c := range n.Children {
		canon.Children[i] = g.Find(c)
	}
	return canon
}

// Add inserts an ENode into the E-Graph, returning its canonical e-class Id.
// The graph copies payload values it can clone. Use AddChecked to reject
// payloads containing references that cannot be copied safely. It rebuilds
// pending unions before hash-consing; use AddChecked to observe rebuild errors.
func (g *EGraph) Add(node ENode) Id {
	origin := "input"
	if g != nil && g.insertionOriginSet {
		origin = g.insertionOrigin
	}
	return g.addFrom(node, origin)
}

func (g *EGraph) addFrom(node ENode, origin string) Id {
	if g == nil {
		return 0
	}
	if len(g.worklist) != 0 {
		if err := g.RebuildChecked(); err != nil {
			return 0
		}
	}
	node.Data = cloneData(node.Data)
	canon := g.Canonicalize(node)
	key := canon.Key()
	if id, ok, err := egraphcore.FindMemoCandidateIter(
		func(visit func(Id) bool) { g.storage().VisitMemoCandidates(key, visit) },
		canon,
		g.Find,
		func(id Id) []ENode {
			if class := g.storage().Class(id); class != nil {
				return class.Nodes
			}
			return nil
		},
		g.Canonicalize,
		func(a, b ENode) bool { return a.Equal(b) },
	); err == nil && ok {
		return id
	}

	newID, ok := g.storage().Insert(key, func(id Id) *EClass {
		return &EClass{
			ID:    id,
			Nodes: []ENode{canon},
		}
	})
	if !ok {
		return 0
	}

	// Register as parent on children e-classes
	for _, childID := range canon.Children {
		childClass := g.storage().Class(g.Find(childID))
		if childClass != nil {
			childClass.parents = append(childClass.parents, parentRef{
				node:    canon,
				classID: newID,
			})
		}
	}
	if g.origins == nil {
		g.origins = make(map[string]string)
	}
	g.origins[key] = origin
	g.revision++

	return newID
}

// AddChecked validates the operator, child IDs, and payload before insertion.
// Function, channel, and unsafe-pointer payloads are unsupported because they
// cannot be copied safely. It rebuilds pending unions before hash-consing. The
// zero-value graph is initialized by a valid insertion. Legacy Add remains
// available when callers do not need to inspect validation errors.
func (g *EGraph) AddChecked(node ENode) (Id, error) {
	if g == nil {
		return 0, fmt.Errorf("egraph: nil graph")
	}
	if node.Op == "" {
		return 0, fmt.Errorf("egraph: empty operator")
	}
	var classCount int
	if g.store != nil {
		classCount = g.store.Len()
	}
	for _, child := range node.Children {
		if uint64(child) >= uint64(classCount) {
			return 0, fmt.Errorf("egraph: invalid child ID %d", child)
		}
		class := g.store.Class(g.Find(child))
		if class == nil || len(class.Nodes) == 0 {
			return 0, fmt.Errorf("egraph: child class %d has no nodes", child)
		}
	}
	if payloadHasCycle(node.Data, make(map[visit]bool)) {
		return 0, fmt.Errorf("egraph: cyclic mutable payload is unsupported")
	}
	if payloadHasUnexportedMutable(reflect.ValueOf(node.Data), make(map[visit]bool)) {
		return 0, fmt.Errorf("egraph: unexported mutable payload is unsupported")
	}
	if payloadHasUncloneableReference(reflect.ValueOf(node.Data), make(map[visit]bool)) {
		return 0, fmt.Errorf("egraph: function, channel, or unsafe pointer payload is unsupported")
	}
	if len(g.worklist) != 0 {
		if err := g.RebuildChecked(); err != nil {
			return 0, fmt.Errorf("egraph: rebuild before add: %w", err)
		}
	}
	id := g.Add(node)
	if g.store == nil || g.store.Class(id) == nil {
		return 0, fmt.Errorf("egraph: failed to add node")
	}
	return id, nil
}

// Union merges two equivalence classes, returning the canonical merged Id and whether a merge occurred.
func (g *EGraph) Union(id1, id2 Id) (Id, bool) {
	if g == nil {
		return 0, false
	}
	if uint64(id1) >= uint64(g.storage().Len()) || uint64(id2) >= uint64(g.storage().Len()) {
		return id1, false
	}
	root1 := g.Find(id1)
	root2 := g.Find(id2)
	if root1 == root2 {
		return root1, false
	}
	winner, merged := g.unionRoots(root1, root2)
	if merged {
		if facts := g.Facts(); facts != nil {
			_ = facts.Recanonicalize(g.Find)
		}
	}
	return winner, merged
}

// unionRoots merges two roots and updates the pending rebuild worklist.
// Relation recanonicalization is handled by the public union operation so the
// checked and unchecked paths can share class mutation without hiding errors.
func (g *EGraph) unionRoots(root1, root2 Id) (Id, bool) {
	// Merge smaller classes into larger classes with a deterministic ID tie-break.
	c1 := g.storage().Class(root1)
	c2 := g.storage().Class(root2)

	winner := root1
	if c1 != nil && c2 != nil {
		var merged bool
		winner, merged = g.storage().MergeRootsBySize(root1, root2, func(winner, loser *EClass) *EClass {
			winner.Nodes = append(winner.Nodes, loser.Nodes...)
			winner.parents = append(winner.parents, loser.parents...)
			return winner
		})
		if !merged {
			return root1, false
		}
	} else {
		var linked bool
		winner, linked = g.storage().LinkRootsBySize(root1, root2)
		if !linked {
			// Preserve the legacy unchecked Union behavior for partially populated
			// stores: link valid roots even when class records are missing.
			return root1, false
		}
	}
	g.worklist = append(g.worklist, winner)
	g.revision++
	return winner, true
}

// UnionChecked merges two valid classes and reports invalid IDs or missing
// class state. Fact recanonicalization errors are returned after the union;
// callers should rebuild before relying on graph invariants.
func (g *EGraph) UnionChecked(id1, id2 Id) (Id, bool, error) {
	if g == nil {
		return 0, false, fmt.Errorf("egraph: nil graph")
	}
	root1, root2, status := g.storage().ResolveRoots(id1, id2)
	if status == egraphcore.RootPairInvalidID {
		return 0, false, fmt.Errorf("egraph: invalid class ID")
	}
	if status == egraphcore.RootPairSame {
		return root1, false, nil
	}
	if status == egraphcore.RootPairMissingClass || g.storage().Class(root1) == nil || g.storage().Class(root2) == nil {
		return 0, false, fmt.Errorf("egraph: missing class during union (%d, %d)", root1, root2)
	}
	root, merged := g.unionRoots(root1, root2)
	if !merged {
		return 0, false, fmt.Errorf("egraph: failed to union classes (%d, %d)", root1, root2)
	}
	if facts := g.Facts(); facts != nil {
		if err := facts.Recanonicalize(g.Find); err != nil {
			return root, true, err
		}
	}
	return root, true, nil
}

// Rebuild restores congruence invariants across the E-Graph and clears the pending worklist.
func (g *EGraph) Rebuild() int {
	totalRepaired, _ := g.rebuildChecked()
	return totalRepaired
}

func (g *EGraph) rebuildChecked() (int, error) {
	if g == nil {
		return 0, fmt.Errorf("egraph: nil graph")
	}
	pending := append([]Id(nil), g.worklist...)
	restorePending := func() {
		if len(pending) != 0 {
			g.worklist = append(pending, g.worklist...)
		}
	}
	type scope struct {
		classID Id
		parents bool
	}
	var currentPass = -1
	var todo []Id
	maxPasses := g.storage().Len()
	if maxPasses == 0 {
		maxPasses = 1
	}
	report, err := egraphcore.DriveCongruence[Id, ENode, string, scope](g, maxPasses,
		func(pass, index int) (scope, []egraphcore.OwnedNode[Id, ENode], bool, error) {
			if currentPass != pass {
				currentPass = pass
				todo = g.worklist
				g.worklist = nil
			}
			if index >= 2*len(todo) {
				return scope{}, nil, false, nil
			}
			rawID := todo[index/2]
			classID := g.Find(rawID)
			eclass := g.storage().Class(classID)
			current := scope{classID: classID, parents: index%2 != 0}
			if eclass == nil {
				return current, nil, true, nil
			}
			owned := make([]egraphcore.OwnedNode[Id, ENode], 0)
			if current.parents {
				owned = make([]egraphcore.OwnedNode[Id, ENode], 0, len(eclass.parents))
				for _, parent := range eclass.parents {
					canon := g.Canonicalize(parent.node)
					g.preserveOrigin(parent.node, canon)
					owned = append(owned, egraphcore.OwnedNode[Id, ENode]{
						Owner: g.Find(parent.classID),
						Node:  canon,
					})
				}
			} else {
				owned = make([]egraphcore.OwnedNode[Id, ENode], 0, len(eclass.Nodes))
				for _, node := range eclass.Nodes {
					canon := g.Canonicalize(node)
					g.preserveOrigin(node, canon)
					owned = append(owned, egraphcore.OwnedNode[Id, ENode]{
						Owner: classID,
						Node:  canon,
					})
				}
			}
			return current, owned, true, nil
		},
		func(n ENode) string { return n.Key() },
		func(a, b ENode) bool { return a.Equal(b) },
		func(a, b ENode) ENode {
			a.Subsumed = a.Subsumed || b.Subsumed
			return a
		},
		func(current scope, byOwner map[Id][]ENode) (bool, error) {
			classID := g.Find(current.classID)
			eclass := g.storage().Class(classID)
			if eclass == nil {
				return false, nil
			}
			if !current.parents {
				eclass.Nodes = byOwner[classID]
				return false, nil
			}
			owners := make([]Id, 0, len(byOwner))
			for owner := range byOwner {
				owners = append(owners, owner)
			}
			sort.Slice(owners, func(i, j int) bool { return owners[i] < owners[j] })
			var parentCount int
			for _, nodes := range byOwner {
				parentCount += len(nodes)
			}
			newParents := make([]parentRef, 0, parentCount)
			for _, owner := range owners {
				for _, node := range byOwner[owner] {
					key := node.Key()
					g.storage().AddMemoCandidateIf(key, owner, func(candidate, proposed Id) bool {
						return g.Find(candidate) == g.Find(proposed)
					})
					newParents = append(newParents, parentRef{node: node, classID: owner})
				}
			}
			eclass.parents = newParents
			return false, nil
		})
	if err != nil {
		restorePending()
		return report.Merges, err
	}
	if facts := g.Facts(); facts != nil {
		if err := facts.Recanonicalize(g.Find); err != nil {
			restorePending()
			return report.Merges, err
		}
	}

	return report.Merges, nil
}

// RebuildChecked rebuilds the graph through the common graph interface and
// reports congruence-planning or fact-canonicalization errors.
func (g *EGraph) RebuildChecked() error {
	_, err := g.rebuildChecked()
	return err
}

func (g *EGraph) class(id Id) *EClass {
	return g.storage().Class(g.Find(id))
}

func (g *EGraph) classes() []*EClass {
	store := g.storage()
	res := make([]*EClass, 0, store.ClassCount())
	store.RangeClasses(func(_ Id, c *EClass) bool {
		res = append(res, c)
		return true
	})
	sort.Slice(res, func(i, j int) bool { return res[i].ID < res[j].ID })
	return res
}

func cloneEClassView(class *EClass) *EClass {
	if class == nil {
		return nil
	}
	copy := &EClass{ID: class.ID, Nodes: make([]ENode, len(class.Nodes))}
	for i, node := range class.Nodes {
		copy.Nodes[i] = ENode{
			Op:       node.Op,
			Children: append([]Id(nil), node.Children...),
			Data:     cloneData(node.Data),
			Subsumed: node.Subsumed,
		}
	}
	return copy
}

// Class returns a snapshot of the EClass corresponding to id, or nil if the
// graph is nil or the class is not found. Mutating the returned class or nodes
// does not change the graph.
// Payloads are detached when their values can be cloned; use AddChecked to
// reject unsupported payload references.
func (g *EGraph) Class(id Id) *EClass {
	if g == nil {
		return nil
	}
	return cloneEClassView(g.class(id))
}

// Classes returns snapshots of all active canonical EClasses sorted by Id.
// It returns nil for a nil graph.
// Mutating a returned class or nodes does not change the graph. Payloads are
// detached when their values can be cloned; use AddChecked to reject
// unsupported payload references.
func (g *EGraph) Classes() []*EClass {
	if g == nil {
		return nil
	}
	classes := g.classes()
	res := make([]*EClass, len(classes))
	for i, class := range classes {
		res[i] = cloneEClassView(class)
	}
	return res
}

// Size returns total count of canonical e-classes and enodes. A nil graph has
// size zero.
func (g *EGraph) Size() (numClasses int, numNodes int) {
	if g == nil {
		return 0, 0
	}
	store := g.storage()
	numClasses = store.ClassCount()
	store.RangeClasses(func(_ Id, c *EClass) bool {
		numNodes += len(c.Nodes)
		return true
	})
	return numClasses, numNodes
}

// Clone creates an independent deep copy of the EGraph. It returns nil for a
// nil receiver.
func (g *EGraph) Clone() *EGraph {
	if g == nil {
		return nil
	}
	clone := &EGraph{
		worklist:           make([]Id, len(g.worklist)),
		facts:              g.Facts().Clone(),
		revision:           g.revision,
		insertionOrigin:    g.insertionOrigin,
		insertionOriginSet: g.insertionOriginSet,
		origins:            make(map[string]string, len(g.origins)),
	}
	for key, origin := range g.origins {
		clone.origins[key] = origin
	}
	if g.store == nil {
		clone.store = egraphcore.NewStore[Id, *EClass]()
	} else {
		clone.store = g.store.Clone(func(c *EClass) *EClass {
			if c == nil {
				return nil
			}
			nodesCopy := make([]ENode, len(c.Nodes))
			for i, n := range c.Nodes {
				childrenCopy := append([]Id(nil), n.Children...)
				nodesCopy[i] = ENode{Op: n.Op, Children: childrenCopy, Data: cloneData(n.Data), Subsumed: n.Subsumed}
			}
			parentsCopy := make([]parentRef, len(c.parents))
			for i, p := range c.parents {
				chCopy := append([]Id(nil), p.node.Children...)
				parentsCopy[i] = parentRef{node: ENode{Op: p.node.Op, Children: chCopy, Data: cloneData(p.node.Data), Subsumed: p.node.Subsumed}, classID: p.classID}
			}
			return &EClass{ID: c.ID, Nodes: nodesCopy, parents: parentsCopy}
		})
	}
	copy(clone.worklist, g.worklist)

	return clone
}

func cloneData(value any) (out any) {
	defer func() {
		if recover() != nil {
			out = value
		}
	}()
	v := cloneReflect(reflect.ValueOf(value), make(map[cloneVisit]reflect.Value))
	if !v.IsValid() {
		return nil
	}
	return v.Interface()
}

type visit struct {
	typ      reflect.Type
	ptr      uintptr
	length   int
	capacity int
}

func visitFor(v reflect.Value) visit {
	key := visit{typ: v.Type(), ptr: v.Pointer()}
	if v.Kind() == reflect.Slice {
		key.length = v.Len()
		key.capacity = v.Cap()
	}
	return key
}

type cloneVisit struct {
	typ      reflect.Type
	ptr      uintptr
	length   int
	capacity int
}

func cloneVisitFor(v reflect.Value) cloneVisit {
	key := cloneVisit{typ: v.Type(), ptr: v.Pointer()}
	if v.Kind() == reflect.Slice {
		key.length = v.Len()
		key.capacity = v.Cap()
	}
	return key
}

func payloadHasCycle(v any, seen map[visit]bool) bool { return payloadCycle(reflect.ValueOf(v), seen) }
func payloadCycle(v reflect.Value, seen map[visit]bool) bool {
	if !v.IsValid() {
		return false
	}
	if v.Kind() == reflect.Interface {
		return !v.IsNil() && payloadCycle(v.Elem(), seen)
	}
	var key visit
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if v.IsNil() {
			return false
		}
		key = visitFor(v)
		if seen[key] {
			return true
		}
		seen[key] = true
		defer delete(seen, key)
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		return payloadCycle(v.Elem(), seen)
	case reflect.Map:
		for _, k := range v.MapKeys() {
			if payloadCycle(k, seen) || payloadCycle(v.MapIndex(k), seen) {
				return true
			}
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			if payloadCycle(v.Index(i), seen) {
				return true
			}
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if v.Field(i).CanInterface() && payloadCycle(v.Field(i), seen) {
				return true
			}
		}
	}
	return false
}

func payloadHasUnexportedMutable(v reflect.Value, seen map[visit]bool) bool {
	if !v.IsValid() {
		return false
	}
	if !v.CanInterface() {
		return payloadHasMutableValue(v)
	}
	if v.Kind() == reflect.Interface {
		return !v.IsNil() && payloadHasUnexportedMutable(v.Elem(), seen)
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if v.IsNil() {
			return false
		}
		key := visitFor(v)
		if seen[key] {
			return false
		}
		seen[key] = true
		defer delete(seen, key)
	}
	switch v.Kind() {
	case reflect.Pointer:
		return payloadHasUnexportedMutable(v.Elem(), seen)
	case reflect.Map:
		for _, k := range v.MapKeys() {
			if payloadHasUnexportedMutable(k, seen) || payloadHasUnexportedMutable(v.MapIndex(k), seen) {
				return true
			}
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			if payloadHasUnexportedMutable(v.Index(i), seen) {
				return true
			}
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			field := v.Field(i)
			if !field.CanInterface() {
				if payloadHasMutableValue(field) {
					return true
				}
				continue
			}
			if payloadHasUnexportedMutable(field, seen) {
				return true
			}
		}
	}
	return false
}

func payloadHasMutableValue(v reflect.Value) bool {
	if !v.IsValid() {
		return false
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface:
		return !v.IsNil()
	case reflect.Array:
		for i := 0; i < v.Len(); i++ {
			if payloadHasMutableValue(v.Index(i)) {
				return true
			}
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if payloadHasMutableValue(v.Field(i)) {
				return true
			}
		}
	}
	return false
}

func payloadHasUncloneableReference(v reflect.Value, seen map[visit]bool) bool {
	if !v.IsValid() {
		return false
	}
	switch v.Kind() {
	case reflect.Func, reflect.Chan, reflect.UnsafePointer:
		return !v.IsNil()
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if v.IsNil() {
			return false
		}
		key := visitFor(v)
		if seen[key] {
			return false
		}
		seen[key] = true
		defer delete(seen, key)
	}
	switch v.Kind() {
	case reflect.Interface, reflect.Pointer:
		return payloadHasUncloneableReference(v.Elem(), seen)
	case reflect.Map:
		for _, key := range v.MapKeys() {
			if payloadHasUncloneableReference(key, seen) || payloadHasUncloneableReference(v.MapIndex(key), seen) {
				return true
			}
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			if payloadHasUncloneableReference(v.Index(i), seen) {
				return true
			}
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if payloadHasUncloneableReference(v.Field(i), seen) {
				return true
			}
		}
	}
	return false
}

func cloneReflect(v reflect.Value, memo map[cloneVisit]reflect.Value) reflect.Value {
	if !v.IsValid() {
		return v
	}
	switch v.Kind() {
	case reflect.Interface:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		out := reflect.New(v.Type()).Elem()
		out.Set(cloneReflect(v.Elem(), memo))
		return out
	case reflect.Pointer:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		key := cloneVisitFor(v)
		if prior, ok := memo[key]; ok {
			return prior
		}
		out := reflect.New(v.Type().Elem())
		memo[key] = out
		out.Elem().Set(cloneReflect(v.Elem(), memo))
		return out
	case reflect.Map:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		key := cloneVisitFor(v)
		if prior, ok := memo[key]; ok {
			return prior
		}
		out := reflect.MakeMapWithSize(v.Type(), v.Len())
		memo[key] = out
		for _, k := range v.MapKeys() {
			out.SetMapIndex(cloneReflect(k, memo), cloneReflect(v.MapIndex(k), memo))
		}
		return out
	case reflect.Slice:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		key := cloneVisitFor(v)
		if prior, ok := memo[key]; ok {
			return prior
		}
		out := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		memo[key] = out
		for i := 0; i < v.Len(); i++ {
			out.Index(i).Set(cloneReflect(v.Index(i), memo))
		}
		return out
	case reflect.Array:
		out := reflect.New(v.Type()).Elem()
		for i := 0; i < v.Len(); i++ {
			out.Index(i).Set(cloneReflect(v.Index(i), memo))
		}
		return out
	case reflect.Struct:
		out := reflect.New(v.Type()).Elem()
		if v.CanInterface() {
			out.Set(v)
		}
		for i := 0; i < v.NumField(); i++ {
			if out.Field(i).CanSet() && v.Field(i).CanInterface() {
				out.Field(i).Set(cloneReflect(v.Field(i), memo))
			}
		}
		return out
	default:
		return v
	}
}
