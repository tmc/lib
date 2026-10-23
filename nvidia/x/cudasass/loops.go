package cudasass

import "slices"

// A Loop is a natural loop of a CFG: a header block and the blocks that
// reach a back edge into it without passing through it. A back edge is an
// edge whose target dominates its source, so code that ptxas places after
// the function's exit and branches back from is not a loop unless it is
// part of a cycle through its target.
type Loop struct {
	Header int   // index into CFG.Blocks
	Blocks []int // the loop's blocks, header included, sorted
}

// Dominators returns the immediate dominator of each block, computed from
// block 0, the function's entry, over every edge kind. The entry's
// immediate dominator is itself, and a block unreachable from the entry
// has -1.
func (g *CFG) Dominators() []int {
	n := len(g.Blocks)
	preds := make([][]int, n)
	succs := make([][]int, n)
	for _, e := range g.Edges {
		preds[e.To] = append(preds[e.To], e.From)
		succs[e.From] = append(succs[e.From], e.To)
	}
	// Reverse postorder from the entry.
	order := make([]int, n) // block -> postorder number, or -1
	for i := range order {
		order[i] = -1
	}
	var post []int
	seen := make([]bool, n)
	var visit func(int)
	visit = func(b int) {
		seen[b] = true
		for _, s := range succs[b] {
			if !seen[s] {
				visit(s)
			}
		}
		order[b] = len(post)
		post = append(post, b)
	}
	idom := make([]int, n)
	for i := range idom {
		idom[i] = -1
	}
	if n == 0 {
		return idom
	}
	visit(0)
	idom[0] = 0
	intersect := func(a, b int) int {
		for a != b {
			for order[a] < order[b] {
				a = idom[a]
			}
			for order[b] < order[a] {
				b = idom[b]
			}
		}
		return a
	}
	for changed := true; changed; {
		changed = false
		for i := len(post) - 1; i >= 0; i-- {
			b := post[i]
			if b == 0 {
				continue
			}
			d := -1
			for _, p := range preds[b] {
				if idom[p] < 0 {
					continue
				}
				if d < 0 {
					d = p
				} else {
					d = intersect(p, d)
				}
			}
			if d != idom[b] {
				idom[b], changed = d, true
			}
		}
	}
	return idom
}

// Loops returns g's natural loops, one per header, ordered by header.
// Back edges into the same header form one loop. Blocks unreachable from
// the entry, such as the self-branch that pads the end of every kernel,
// are in no loop.
func (g *CFG) Loops() []Loop {
	idom := g.Dominators()
	dominates := func(a, b int) bool {
		for idom[b] >= 0 {
			if a == b {
				return true
			}
			if idom[b] == b {
				return false
			}
			b = idom[b]
		}
		return false
	}
	preds := make([][]int, len(g.Blocks))
	for _, e := range g.Edges {
		preds[e.To] = append(preds[e.To], e.From)
	}
	body := map[int]map[int]bool{}
	for _, e := range g.Edges {
		if !dominates(e.To, e.From) {
			continue
		}
		h := e.To
		if body[h] == nil {
			body[h] = map[int]bool{h: true}
		}
		stack := []int{e.From}
		for len(stack) > 0 {
			b := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if body[h][b] {
				continue
			}
			body[h][b] = true
			for _, p := range preds[b] {
				if idom[p] >= 0 {
					stack = append(stack, p)
				}
			}
		}
	}
	var loops []Loop
	for h, set := range body {
		l := Loop{Header: h}
		for b := range set {
			l.Blocks = append(l.Blocks, b)
		}
		slices.Sort(l.Blocks)
		loops = append(loops, l)
	}
	slices.SortFunc(loops, func(a, b Loop) int { return a.Header - b.Header })
	return loops
}

// Reducible reports whether every cycle reachable from the entry is a
// natural loop. In an irreducible graph some cycle has two entries and
// no header, and [CFG.Loops] does not report it.
func (g *CFG) Reducible() bool {
	idom := g.Dominators()
	succs := make([][]int, len(g.Blocks))
	for _, e := range g.Edges {
		succs[e.From] = append(succs[e.From], e.To)
	}
	const (
		unseen = iota
		active
		done
	)
	state := make([]int, len(g.Blocks))
	ok := true
	var visit func(int)
	visit = func(b int) {
		state[b] = active
		for _, s := range succs[b] {
			switch state[s] {
			case unseen:
				visit(s)
			case active: // a retreating edge: s must dominate b
				d := b
				for d != s && idom[d] != d {
					d = idom[d]
				}
				if d != s {
					ok = false
				}
			}
		}
		state[b] = done
	}
	if len(g.Blocks) > 0 {
		visit(0)
	}
	return ok
}
