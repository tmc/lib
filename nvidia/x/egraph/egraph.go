package egraph

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
)

// Id identifies an equivalence class in an E-Graph.
type Id uint32

// ENode is an operator node within an equivalence class with child e-class IDs.
type ENode struct {
	Op       string
	Children []Id
	Data     any
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
		key := visit{v.Type(), v.Pointer()}
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
	unionFind []Id
	classes   map[Id]*EClass
	memo      map[string]Id
	worklist  []Id
}

// New creates an empty E-Graph.
func New() *EGraph {
	return &EGraph{
		classes: make(map[Id]*EClass),
		memo:    make(map[string]Id),
	}
}

// Find finds the canonical representative Id for an equivalence class with path compression.
func (g *EGraph) Find(id Id) Id {
	root := id
	for root < Id(len(g.unionFind)) && g.unionFind[root] != root {
		root = g.unionFind[root]
	}
	// Path compression
	curr := id
	for curr < Id(len(g.unionFind)) && g.unionFind[curr] != root {
		next := g.unionFind[curr]
		g.unionFind[curr] = root
		curr = next
	}
	return root
}

// Canonicalize returns an ENode where all child IDs are replaced by their canonical representatives.
func (g *EGraph) Canonicalize(n ENode) ENode {
	canon := ENode{
		Op:       n.Op,
		Data:     n.Data,
		Children: make([]Id, len(n.Children)),
	}
	for i, c := range n.Children {
		canon.Children[i] = g.Find(c)
	}
	return canon
}

// Add inserts an ENode into the E-Graph, returning its canonical e-class Id.
func (g *EGraph) Add(node ENode) Id {
	canon := g.Canonicalize(node)
	key := canon.Key()
	if id, ok := g.memo[key]; ok {
		return g.Find(id)
	}

	newID := Id(len(g.unionFind))
	g.unionFind = append(g.unionFind, newID)

	eclass := &EClass{
		ID:    newID,
		Nodes: []ENode{canon},
	}
	g.classes[newID] = eclass
	g.memo[key] = newID

	// Register as parent on children e-classes
	for _, childID := range canon.Children {
		childClass := g.classes[g.Find(childID)]
		if childClass != nil {
			childClass.parents = append(childClass.parents, parentRef{
				node:    canon,
				classID: newID,
			})
		}
	}

	return newID
}

// AddChecked rejects mutable payloads containing reference cycles before
// insertion. Legacy Add remains available for immutable/scalar payloads.
func (g *EGraph) AddChecked(node ENode) (Id, error) {
	if payloadHasCycle(node.Data, make(map[visit]bool)) {
		return 0, fmt.Errorf("egraph: cyclic mutable payload is unsupported")
	}
	if payloadHasUnexportedMutable(reflect.ValueOf(node.Data), make(map[visit]bool)) {
		return 0, fmt.Errorf("egraph: unexported mutable payload is unsupported")
	}
	return g.Add(node), nil
}

// Union merges two equivalence classes, returning the canonical merged Id and whether a merge occurred.
func (g *EGraph) Union(id1, id2 Id) (Id, bool) {
	root1 := g.Find(id1)
	root2 := g.Find(id2)
	if root1 == root2 {
		return root1, false
	}

	// Make root1 the new canonical root (or by size heuristic)
	g.unionFind[root2] = root1
	c1 := g.classes[root1]
	c2 := g.classes[root2]

	if c1 != nil && c2 != nil {
		c1.Nodes = append(c1.Nodes, c2.Nodes...)
		c1.parents = append(c1.parents, c2.parents...)
		delete(g.classes, root2)
	}

	g.worklist = append(g.worklist, root1)
	return root1, true
}

// Rebuild restores congruence invariants across the E-Graph and clears the pending worklist.
func (g *EGraph) Rebuild() int {
	totalRepaired := 0

	for len(g.worklist) > 0 {
		todo := g.worklist
		g.worklist = nil

		for _, rawID := range todo {
			classID := g.Find(rawID)
			eclass := g.classes[classID]
			if eclass == nil {
				continue
			}

			// Canonicalize and deduplicate nodes in eclass
			var uniqueNodes []ENode
			seenNodes := make(map[string]bool)
			for _, n := range eclass.Nodes {
				canon := g.Canonicalize(n)
				key := canon.Key()
				if !seenNodes[key] {
					seenNodes[key] = true
					uniqueNodes = append(uniqueNodes, canon)
				}
			}
			eclass.Nodes = uniqueNodes

			// Repair parents (congruence closure)
			var newParents []parentRef
			memoParents := make(map[string]Id)

			for _, p := range eclass.parents {
				canonP := g.Canonicalize(p.node)
				key := canonP.Key()
				pClassID := g.Find(p.classID)

				if otherClassID, exists := memoParents[key]; exists {
					if otherClassID != pClassID {
						g.Union(pClassID, otherClassID)
						totalRepaired++
					}
				} else {
					memoParents[key] = pClassID
					newParents = append(newParents, parentRef{
						node:    canonP,
						classID: pClassID,
					})
				}
				g.memo[key] = pClassID
			}
			eclass.parents = newParents
		}
	}

	return totalRepaired
}

// Class returns the EClass corresponding to id, or nil if not found.
func (g *EGraph) Class(id Id) *EClass {
	return g.classes[g.Find(id)]
}

// Classes returns all active canonical EClasses sorted by Id.
func (g *EGraph) Classes() []*EClass {
	res := make([]*EClass, 0, len(g.classes))
	for _, c := range g.classes {
		res = append(res, c)
	}
	sort.Slice(res, func(i, j int) bool { return res[i].ID < res[j].ID })
	return res
}

// Size returns total count of canonical e-classes and enodes.
func (g *EGraph) Size() (numClasses int, numNodes int) {
	numClasses = len(g.classes)
	for _, c := range g.classes {
		numNodes += len(c.Nodes)
	}
	return numClasses, numNodes
}

// Clone creates an independent deep copy of the EGraph.
func (g *EGraph) Clone() *EGraph {
	clone := &EGraph{
		unionFind: make([]Id, len(g.unionFind)),
		classes:   make(map[Id]*EClass, len(g.classes)),
		memo:      make(map[string]Id, len(g.memo)),
		worklist:  make([]Id, len(g.worklist)),
	}

	copy(clone.unionFind, g.unionFind)
	copy(clone.worklist, g.worklist)

	for k, v := range g.memo {
		clone.memo[k] = v
	}

	for id, c := range g.classes {
		nodesCopy := make([]ENode, len(c.Nodes))
		for i, n := range c.Nodes {
			childrenCopy := make([]Id, len(n.Children))
			copy(childrenCopy, n.Children)
			nodesCopy[i] = ENode{
				Op:       n.Op,
				Children: childrenCopy,
				Data:     cloneData(n.Data),
			}
		}
		parentsCopy := make([]parentRef, len(c.parents))
		for i, p := range c.parents {
			chCopy := make([]Id, len(p.node.Children))
			copy(chCopy, p.node.Children)
			parentsCopy[i] = parentRef{
				node: ENode{
					Op:       p.node.Op,
					Children: chCopy,
					Data:     cloneData(p.node.Data),
				},
				classID: p.classID,
			}
		}
		clone.classes[id] = &EClass{
			ID:      c.ID,
			Nodes:   nodesCopy,
			parents: parentsCopy,
		}
	}

	return clone
}

func cloneData(value any) (out any) {
	defer func() {
		if recover() != nil {
			out = value
		}
	}()
	v := cloneReflect(reflect.ValueOf(value), make(map[visit]reflect.Value))
	if !v.IsValid() {
		return nil
	}
	return v.Interface()
}

type visit struct {
	typ reflect.Type
	ptr uintptr
}

func payloadHasCycle(v any, seen map[visit]bool) bool { return payloadCycle(reflect.ValueOf(v), seen) }
func payloadCycle(v reflect.Value, seen map[visit]bool) bool {
	if !v.IsValid() {
		return false
	}
	if v.Kind() == reflect.Interface {
		return v.IsNil() || payloadCycle(v.Elem(), seen)
	}
	var key visit
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if v.IsNil() {
			return false
		}
		key = visit{v.Type(), v.Pointer()}
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
	if v.Kind() == reflect.Interface {
		return !v.IsNil() && payloadHasUnexportedMutable(v.Elem(), seen)
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if v.IsNil() {
			return false
		}
		key := visit{v.Type(), v.Pointer()}
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
				switch field.Kind() {
				case reflect.Map, reflect.Slice, reflect.Pointer, reflect.Interface:
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

func cloneReflect(v reflect.Value, memo map[visit]reflect.Value) reflect.Value {
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
		key := visit{v.Type(), v.Pointer()}
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
		key := visit{v.Type(), v.Pointer()}
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
		key := visit{v.Type(), v.Pointer()}
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
		for i := 0; i < v.NumField(); i++ {
			if out.Field(i).CanSet() {
				out.Field(i).Set(cloneReflect(v.Field(i), memo))
			}
		}
		return out
	default:
		return v
	}
}
