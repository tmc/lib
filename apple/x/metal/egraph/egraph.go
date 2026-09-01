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
	for {
		parent := g.findParent(root)
		if parent == root {
			break
		}
		root = parent
	}
	curr := id
	for curr != root {
		next := g.findParent(curr)
		g.unionFind[curr] = root
		curr = next
	}
	return root
}

func (g *EGraph) findParent(id Id) Id {
	idx := int(id)
	if idx >= len(g.unionFind) {
		return id
	}
	return g.unionFind[idx]
}

// Add adds an ENode to the E-Graph and returns its canonical class ID.
func (g *EGraph) Add(n ENode) Id {
	canonNode := g.canonicalize(n)
	key := canonNode.Key()

	if classID, ok := g.memo[key]; ok {
		return g.Find(classID)
	}

	newID := Id(len(g.unionFind))
	g.unionFind = append(g.unionFind, newID)

	class := &EClass{
		ID:    newID,
		Nodes: []ENode{canonNode},
	}
	g.classes[newID] = class
	g.memo[key] = newID

	for _, childID := range canonNode.Children {
		childClassID := g.Find(childID)
		if childClass, ok := g.classes[childClassID]; ok {
			childClass.parents = append(childClass.parents, parentRef{
				node:    canonNode,
				classID: newID,
			})
		}
	}

	return newID
}

// Union merges two equivalence classes and returns the new canonical representative Id.
func (g *EGraph) Union(id1, id2 Id) (Id, bool) {
	root1 := g.Find(id1)
	root2 := g.Find(id2)
	if root1 == root2 {
		return root1, false
	}

	g.unionFind[root2] = root1

	class1 := g.classes[root1]
	class2 := g.classes[root2]

	class1.Nodes = append(class1.Nodes, class2.Nodes...)
	class1.parents = append(class1.parents, class2.parents...)

	delete(g.classes, root2)
	g.worklist = append(g.worklist, root1)

	return root1, true
}

func (g *EGraph) canonicalize(n ENode) ENode {
	if len(n.Children) == 0 {
		return n
	}
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

// Rebuild restores the congruence invariants and hash-consing memo table.
func (g *EGraph) Rebuild() {
	for len(g.worklist) > 0 {
		todo := g.worklist
		g.worklist = nil

		for _, classID := range todo {
			canonical := g.Find(classID)
			class, ok := g.classes[canonical]
			if !ok {
				continue
			}

			seen := make(map[string]ENode)
			for _, n := range class.Nodes {
				cn := g.canonicalize(n)
				seen[cn.Key()] = cn
			}
			class.Nodes = class.Nodes[:0]
			for _, n := range seen {
				class.Nodes = append(class.Nodes, n)
			}

			for _, p := range class.parents {
				canonP := g.canonicalize(p.node)
				g.memo[canonP.Key()] = g.Find(p.classID)
			}
		}
	}
}

// Classes returns all current canonical equivalence classes.
func (g *EGraph) Classes() []*EClass {
	res := make([]*EClass, 0, len(g.classes))
	for _, c := range g.classes {
		res = append(res, c)
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i].ID < res[j].ID
	})
	return res
}
