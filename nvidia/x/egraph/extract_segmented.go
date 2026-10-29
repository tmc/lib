package egraph

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/egraphcore"
)

// ExtractSegmented is [ExtractDAG] for graphs that repeat a structure, such
// as a stack of decoder layers. It splits the graph below root into
// segments at its boundaries: shared classes, read by two classes or twice
// by one node, that every choice of nodes above them reads and through
// which every path from root to the classes below them passes, as the
// residual stream between two layers is. It extracts
// each segment with the boundaries below it priced as leaves of cost zero,
// so a value computed once is paid for once, and extracts segments that
// are equal up to their leaves' costs only once, reusing the choice. The
// cost returned is the sum of the segments' costs.
//
// Whole-graph extraction prices a class read n times n times, so the cost
// of an L-layer stack grows as n to the power L: within a few dozen layers
// float64 costs can no longer resolve a single operator's cost, and at a
// thousand they overflow. Its work grows faster than the graph, too.
// Segmented extraction's work is linear in the number of segments.
//
// A rewrite that fuses across a boundary adds a node that reads below it,
// so the boundary is no longer one and the segments on either side are
// extracted as one. The cost model must not depend on the Ids of a node's
// children. Limits apply as in [ExtractDAG], to the whole extraction.
func ExtractSegmented(ctx context.Context, g *EGraph, root Id, model CostModel, limits ExtractionLimits) (expr *Expr, cost float64, stats ExtractionStats, err error) {
	if err := checkBounded(ctx, g, limits); err != nil {
		return nil, 0, stats, err
	}
	b := &extractionBudget{ctx: ctx, limits: limits}
	var segs, distinct int
	defer func() {
		stats = ExtractionStats{Work: b.work, Nodes: b.nodes, WorkExhausted: b.workExhausted, Segments: segs, Distinct: distinct}
	}()
	if err := b.take(0); err != nil {
		return nil, 0, stats, err
	}
	if model == nil {
		model = AstSizeCost{}
	}
	ext := &Extractor{g: g, costModel: model, costs: make(map[Id]float64), bestNodes: make(map[Id]ENode), revision: g.revision}
	sg, err := newSegmentGraph(g, g.Find(root), b)
	if err != nil {
		return nil, 0, stats, err
	}
	// Leaves have no children, so one pass prices them all.
	if err := ext.relaxInto(sg.leaves, nil, b); err != nil {
		return nil, 0, stats, err
	}
	type choice struct {
		node []int // index of each member's best node, or -1
		cost []float64
	}
	reuse := make(map[string]choice)
	// Segment roots in reverse RPO: each segment after those it reads.
	for i := len(sg.order) - 1; i >= 0; i-- {
		if !sg.cut[i] {
			continue
		}
		s := sg.segment(i, ext.costs)
		segs++
		if c, ok := reuse[s.key]; ok && s.key != "" {
			for j, id := range s.members {
				if c.node[j] >= 0 {
					ext.bestNodes[id] = s.nodes[j][c.node[j]]
					ext.costs[id] = c.cost[j]
				}
			}
			cost += ext.costs[s.members[0]]
			continue
		}
		distinct++
		fixed := make(map[Id]float64, len(s.inputs)+len(s.leaves))
		ids := slices.Clone(s.members)
		for _, id := range s.inputs {
			fixed[id] = 0
			ids = append(ids, id)
		}
		for _, id := range s.leaves {
			fixed[id] = ext.costs[id]
			ids = append(ids, id)
		}
		if err := ext.relaxInto(ids, fixed, b); err != nil {
			return nil, 0, stats, err
		}
		c := choice{node: make([]int, len(s.members)), cost: make([]float64, len(s.members))}
		for j, id := range s.members {
			c.node[j] = -1
			best, ok := ext.bestNodes[id]
			if !ok {
				continue
			}
			c.node[j] = slices.Index(s.keys[j], localKey(best))
			c.cost[j] = ext.costs[id]
		}
		if s.key != "" {
			reuse[s.key] = c
		}
		root := s.members[0]
		rc, ok := ext.costs[root]
		if !ok || math.IsNaN(rc) || math.IsInf(rc, 0) {
			return nil, 0, stats, fmt.Errorf("egraph: no finite expression for e-class %d", root)
		}
		cost += rc
	}
	expr, err = ext.buildExprShared(sg.order[0], make(map[Id]bool), b, 1, make(map[Id]*Expr))
	if err != nil {
		return nil, 0, stats, err
	}
	return expr, cost, stats, nil
}

// relaxInto runs [Extractor.relax] over ids and records the cost and best
// node of each class not in fixed.
func (ext *Extractor) relaxInto(ids []Id, fixed map[Id]float64, b *extractionBudget) error {
	if len(ids) == 0 {
		return nil
	}
	best, err := ext.relax(ids, fixed, b, max(1, b.limits.MaxWork-b.work))
	if errors.Is(err, egraphcore.ErrCostRelaxationLimit) {
		b.workExhausted = true
		return ErrExtractionLimit
	}
	if err != nil {
		return err
	}
	for id, r := range best {
		if _, ok := fixed[id]; ok {
			continue
		}
		ext.costs[id] = r.Cost
		ext.bestNodes[id] = r.Node
	}
	return nil
}

// A segmentGraph is the part of a graph reachable from a root: its leaf
// classes, whose nodes have no children, and its other classes in reverse
// postorder with their dominator tree and boundaries.
type segmentGraph struct {
	g      *EGraph
	leaves []Id
	leaf   map[Id]bool
	order  []Id       // non-leaf classes in reverse postorder; order[0] is the root
	index  map[Id]int // position in order
	nodes  [][]ENode  // eligible nodes of each class
	kids   [][]int    // distinct non-leaf children, not the class itself
	idom   []int      // immediate dominator; idom[0] == 0
	cut    []bool     // the class roots a segment
	seg    []int      // the segment root of each class
}

func newSegmentGraph(g *EGraph, root Id, b *extractionBudget) (*segmentGraph, error) {
	sg := &segmentGraph{g: g, leaf: make(map[Id]bool), index: make(map[Id]int)}
	eligible := func(id Id) []ENode {
		var out []ENode
		if c := g.class(id); c != nil {
			for _, n := range c.Nodes {
				if !n.Subsumed {
					out = append(out, n)
				}
			}
		}
		return out
	}
	isLeaf := func(nodes []ENode) bool {
		for _, n := range nodes {
			if len(n.Children) != 0 {
				return false
			}
		}
		return len(nodes) != 0
	}

	// Depth-first postorder over non-leaf classes.
	type frame struct {
		id    Id
		nodes []ENode
		kids  []Id
		next  int
	}
	visited := make(map[Id]bool)
	var post []Id
	var postNodes [][]ENode
	push := func(stack []frame, id Id) ([]frame, error) {
		visited[id] = true
		nodes := eligible(id)
		var kids []Id
		seen := make(map[Id]bool)
		for _, n := range nodes {
			if err := b.take(1 + len(n.Children)); err != nil {
				return nil, err
			}
			for _, c := range n.Children {
				c = g.Find(c)
				if !seen[c] {
					seen[c] = true
					kids = append(kids, c)
				}
			}
		}
		return append(stack, frame{id: id, nodes: nodes, kids: kids}), nil
	}
	stack, err := push(nil, root)
	if err != nil {
		return nil, err
	}
	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		if f.next < len(f.kids) {
			c := f.kids[f.next]
			f.next++
			if visited[c] {
				continue
			}
			if nodes := eligible(c); isLeaf(nodes) {
				visited[c] = true
				sg.leaf[c] = true
				sg.leaves = append(sg.leaves, c)
				if err := b.take(len(nodes)); err != nil {
					return nil, err
				}
				continue
			}
			if stack, err = push(stack, c); err != nil {
				return nil, err
			}
			continue
		}
		post = append(post, f.id)
		postNodes = append(postNodes, f.nodes)
		stack = stack[:len(stack)-1]
	}
	n := len(post)
	sg.order = make([]Id, n)
	sg.nodes = make([][]ENode, n)
	for i, id := range post {
		sg.order[n-1-i] = id
		sg.nodes[n-1-i] = postNodes[i]
	}
	for i, id := range sg.order {
		sg.index[id] = i
	}
	sg.kids = make([][]int, n)
	// A class is shared if two classes, or one node twice, read it.
	preds := make([][]int, n)
	shared := make([]bool, n)
	for i, nodes := range sg.nodes {
		seen := make(map[int]bool)
		for _, nd := range nodes {
			uses := make(map[int]bool)
			for _, c := range nd.Children {
				j, ok := sg.index[g.Find(c)]
				if !ok || j == i {
					continue
				}
				if uses[j] {
					shared[j] = true
				}
				uses[j] = true
				if seen[j] {
					continue
				}
				seen[j] = true
				sg.kids[i] = append(sg.kids[i], j)
				preds[j] = append(preds[j], i)
			}
		}
	}

	// Dominators (Cooper, Harvey and Kennedy, "A Simple, Fast Dominance
	// Algorithm"), with classes numbered in reverse postorder.
	sg.idom = make([]int, n)
	for i := range sg.idom {
		sg.idom[i] = -1
	}
	sg.idom[0] = 0
	for changed := true; changed; {
		changed = false
		for i := 1; i < n; i++ {
			d := -1
			for _, p := range preds[i] {
				if sg.idom[p] < 0 {
					continue
				}
				if d < 0 {
					d = p
				} else {
					d = sg.intersect(d, p)
				}
			}
			if d != sg.idom[i] {
				sg.idom[i] = d
				changed = true
			}
			if err := b.take(1 + len(preds[i])); err != nil {
				return nil, err
			}
		}
	}

	// A class c is a boundary if it dominates every class below it. An
	// edge u→w denies that to each class on the dominator tree path from
	// u up to, not including, the nearest common dominator of u and w:
	// those dominate u but not w.
	notCut := make([]bool, n)
	for u, kids := range sg.kids {
		for _, w := range kids {
			stop := sg.intersect(u, w)
			for x := u; x != stop; x = sg.idom[x] {
				notCut[x] = true
				if err := b.take(1); err != nil {
					return nil, err
				}
			}
		}
	}
	// A boundary priced at zero must be paid for whatever the classes
	// above it choose: every node of a required class reads it.
	required := make([]bool, n)
	required[0] = true
	for work := []int{0}; len(work) > 0; {
		i := work[len(work)-1]
		work = work[:len(work)-1]
		count := make(map[int]int)
		for _, nd := range sg.nodes[i] {
			seen := make(map[int]bool)
			for _, c := range nd.Children {
				if j, ok := sg.index[g.Find(c)]; ok && !seen[j] {
					seen[j] = true
					count[j]++
				}
			}
		}
		for j, k := range count {
			if k == len(sg.nodes[i]) && !required[j] {
				required[j] = true
				work = append(work, j)
			}
		}
	}
	sg.cut = make([]bool, n)
	sg.seg = make([]int, n)
	for i := range n {
		sg.cut[i] = i == 0 || required[i] && !notCut[i] && (shared[i] || len(preds[i]) >= 2)
		if sg.cut[i] {
			sg.seg[i] = i
		} else {
			sg.seg[i] = sg.seg[sg.idom[i]]
		}
	}
	return sg, nil
}

// intersect returns the nearest common dominator of a and b.
func (sg *segmentGraph) intersect(a, b int) int {
	for a != b {
		for a > b {
			a = sg.idom[a]
		}
		for b > a {
			b = sg.idom[b]
		}
	}
	return a
}

// A segment is the classes a boundary dominates that no lower boundary
// does, and the boundaries and leaves they read.
type segment struct {
	members []Id       // breadth-first from the segment root, members[0]
	nodes   [][]ENode  // each member's eligible nodes, in key order
	keys    [][]string // localKey of each node
	inputs  []Id       // boundaries below
	leaves  []Id
	// key describes the segment up to the Ids of its classes and the
	// identity of its leaves, which appear only as their costs. Segments
	// with equal keys have the same best choices. It is "" if a class has
	// two nodes with the same local key, which the key cannot order.
	key string
}

// segment returns the segment rooted at order[r]. leafCost holds each leaf
// class's cost.
func (sg *segmentGraph) segment(r int, leafCost map[Id]float64) segment {
	var s segment
	num := map[Id]int{sg.order[r]: 0}
	in := make(map[Id]int)
	lf := make(map[Id]bool)
	s.members = []Id{sg.order[r]}
	var key strings.Builder
	ordered := true
	for q := 0; q < len(s.members); q++ {
		id := s.members[q]
		nodes := slices.Clone(sg.nodes[sg.index[id]])
		keys := make([]string, len(nodes))
		for i, n := range nodes {
			keys[i] = localKey(n)
		}
		perm := make([]int, len(nodes))
		for i := range perm {
			perm[i] = i
		}
		slices.SortStableFunc(perm, func(a, b int) int { return strings.Compare(keys[a], keys[b]) })
		sorted := make([]ENode, len(nodes))
		sortedKeys := make([]string, len(nodes))
		for i, p := range perm {
			sorted[i], sortedKeys[i] = nodes[p], keys[p]
			if i > 0 && sortedKeys[i] == sortedKeys[i-1] {
				ordered = false
			}
		}
		s.nodes = append(s.nodes, sorted)
		s.keys = append(s.keys, sortedKeys)
		for i, n := range sorted {
			if i > 0 {
				key.WriteByte(';')
			}
			key.WriteString(sortedKeys[i])
			key.WriteByte('(')
			for _, c := range n.Children {
				c = sg.g.Find(c)
				switch j, ok := sg.index[c]; {
				case !ok:
					if !lf[c] {
						lf[c] = true
						s.leaves = append(s.leaves, c)
					}
					key.WriteString(" l")
					key.WriteString(strconv.FormatFloat(leafCost[c], 'g', -1, 64))
				case sg.seg[j] == r:
					k, ok := num[c]
					if !ok {
						k = len(s.members)
						num[c] = k
						s.members = append(s.members, c)
					}
					key.WriteString(" m")
					key.WriteString(strconv.Itoa(k))
				default:
					k, ok := in[c]
					if !ok {
						k = len(s.inputs)
						in[c] = k
						s.inputs = append(s.inputs, c)
					}
					key.WriteString(" i")
					key.WriteString(strconv.Itoa(k))
				}
			}
			key.WriteByte(')')
		}
		key.WriteByte('\n')
	}
	if ordered {
		s.key = key.String()
	}
	return s
}

// localKey is n's operator and payload, without its children.
func localKey(n ENode) string {
	return ENode{Op: n.Op, Data: n.Data}.Key()
}
