package sassequiv

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// A node is one value in a value graph: an operation, a leaf (a constant,
// a special register, a parameter word) or a phi at a block entry. The
// graph of both functions under comparison shares one node table.
type node struct {
	label string
	args  []int
	comm  int       // the first comm args commute
	alias int       // for a phi with one incoming value, that value, or -1
	pc    [2]uint64 // an instruction that produced it, per side
	has   [2]bool
}

type graph struct {
	nodes  []node
	intern map[string]int
}

func newGraph() *graph { return &graph{intern: map[string]int{}} }

// add returns the node for label applied to args, creating it once.
func (g *graph) add(side int, pc uint64, label string, comm int, args ...int) int {
	var k strings.Builder
	k.WriteString(label)
	k.WriteByte(0)
	k.WriteString(strconv.Itoa(comm))
	for _, a := range args {
		k.WriteByte(',')
		k.WriteString(strconv.Itoa(a))
	}
	if id, ok := g.intern[k.String()]; ok {
		g.mark(id, side, pc)
		return id
	}
	id := len(g.nodes)
	g.nodes = append(g.nodes, node{label: label, args: slices.Clone(args), comm: comm, alias: -1})
	g.mark(id, side, pc)
	g.intern[k.String()] = id
	return id
}

// fresh returns a new node that is never shared, for a phi whose
// arguments are set later.
func (g *graph) fresh(side int, pc uint64, label string) int {
	id := len(g.nodes)
	g.nodes = append(g.nodes, node{label: label, alias: -1})
	g.mark(id, side, pc)
	return id
}

func (g *graph) mark(id, side int, pc uint64) {
	if n := &g.nodes[id]; !n.has[side] {
		n.pc[side], n.has[side] = pc, true
	}
}

// classes partitions the nodes by bisimulation: two nodes share a class
// when their labels match and their arguments, in order (or as a
// multiset, for commuting ones), share classes, in the greatest such
// partition. On acyclic graphs that is equality of expression trees; at
// loop-header phis it identifies loops that compute the same recurrence.
func (g *graph) classes() []int {
	g.removeTrivialPhis()
	class := make([]int, len(g.nodes))
	ids := map[string]int{}
	for i, n := range g.nodes {
		k := fmt.Sprintf("%s\x00%d\x00%d", n.label, len(n.args), n.comm)
		c, ok := ids[k]
		if !ok {
			c = len(ids)
			ids[k] = c
		}
		class[i] = c
	}
	count := len(ids)
	sig := make([]int, 0, 8)
	for {
		next := make([]int, len(g.nodes))
		ids := map[string]int{}
		var k strings.Builder
		for i, n := range g.nodes {
			sig = sig[:0]
			for _, a := range n.args {
				sig = append(sig, class[a])
			}
			slices.Sort(sig[:n.comm])
			k.Reset()
			k.WriteString(strconv.Itoa(class[i]))
			for _, s := range sig {
				k.WriteByte(',')
				k.WriteString(strconv.Itoa(s))
			}
			c, ok := ids[k.String()]
			if !ok {
				c = len(ids)
				ids[k.String()] = c
			}
			next[i] = c
		}
		class = next
		if len(ids) == count {
			for i, n := range g.nodes {
				if n.alias >= 0 {
					class[i] = class[g.resolve(i)]
				}
			}
			return class
		}
		count = len(ids)
	}
}

// expr renders node id to a bounded depth, for reports.
func (g *graph) expr(id, depth int) string {
	n := g.nodes[id]
	if len(n.args) == 0 {
		return n.label
	}
	if depth == 0 {
		return n.label + "(...)"
	}
	var b strings.Builder
	b.WriteString(n.label)
	b.WriteByte('(')
	for i, a := range n.args {
		if i > 0 {
			b.WriteString(", ")
		}
		if strings.HasPrefix(g.nodes[a].label, "phi:") || strings.HasPrefix(g.nodes[a].label, "mem") {
			b.WriteString(g.nodes[a].label)
			continue
		}
		b.WriteString(g.expr(a, depth-1))
	}
	b.WriteByte(')')
	return b.String()
}

func (g *graph) resolve(id int) int {
	for g.nodes[id].alias >= 0 {
		id = g.nodes[id].alias
	}
	return id
}

// removeTrivialPhis replaces each phi whose incoming values, other than
// itself, are all one value with that value, until none is left: a value
// that is the same on every path into a block, such as a thread index
// kept in a register around a loop instead of read again inside it.
func (g *graph) removeTrivialPhis() {
	for changed := true; changed; {
		changed = false
		for i := range g.nodes {
			n := &g.nodes[i]
			if n.alias >= 0 || !strings.HasPrefix(n.label, "phi:") {
				continue
			}
			only := -1
			trivial := true
			for _, a := range n.args {
				a = g.resolve(a)
				if a == i || a == only {
					continue
				}
				if only >= 0 {
					trivial = false
					break
				}
				only = a
			}
			if trivial && only >= 0 {
				n.alias = only
				changed = true
			}
		}
	}
	for i := range g.nodes {
		for j, a := range g.nodes[i].args {
			g.nodes[i].args[j] = g.resolve(a)
		}
	}
}
