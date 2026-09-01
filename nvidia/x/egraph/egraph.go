package egraph

import (
	"fmt"
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
	if len(n.Children) == 0 {
		return n.Op
	}
	var sb strings.Builder
	sb.WriteString(n.Op)
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

func (n ENode) String() string {
	return n.Key()
}

// Equal returns true if two ENodes have identical operator, data, and children.
func (n ENode) Equal(other ENode) bool {
	if n.Op != other.Op || len(n.Children) != len(other.Children) || n.Data != other.Data {
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
