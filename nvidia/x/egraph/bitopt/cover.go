package bitopt

import "math/bits"

// A coverer searches programs cell by cell: each output bit (a cell)
// takes one move, reading that move's source bit there, and no input bit
// is read twice. A program's cost is its non-identity moves plus its
// (output, move) pairs, the parts. The search branches on the cell or the
// input bit with the fewest choices and prunes by a lower bound on the
// cost still to pay.
type coverer struct {
	s     *searcher
	cells []target // every (output, bit) in the masks
	exact bool     // every input bit must be read
	// readers[x]: the (move, bit) pairs that read input bit x.
	readers [][]target
	union   uint32

	limit     int
	minParts  []int     // [output] -> parts it must have
	cell      [][32]int // [output][bit] -> move or -1
	used      set
	pair      [][]int16 // [output][move] -> cells
	pairs     [][]int   // [output] -> moves with pair > 0
	count     []int     // [move] -> cells
	cost      int
	left      int // cells not yet assigned
	firstFree []int
}

func newCoverer(s *searcher) *coverer {
	c := &coverer{s: s}
	n := len(s.p.Outputs)
	for k, m := range s.p.Outputs {
		c.union |= m
		for _, b := range s.targets[k] {
			c.cells = append(c.cells, target{k, b})
		}
	}
	c.exact = len(c.cells) == 32*s.p.Words
	c.readers = make([][]target, 32*s.p.Words)
	for m := range s.moves {
		for b := range 32 {
			if c.union>>b&1 == 0 {
				continue
			}
			if x := s.src[m][0][b]; x >= 0 {
				c.readers[x] = append(c.readers[x], target{m, b})
			}
		}
	}
	c.cell = make([][32]int, n)
	for k := range c.cell {
		for b := range 32 {
			c.cell[k][b] = -1
		}
	}
	c.pair = make([][]int16, n)
	for k := range c.pair {
		c.pair[k] = make([]int16, len(s.moves))
	}
	c.pairs = make([][]int, n)
	c.count = make([]int, len(s.moves))
	c.minParts = make([]int, n)
	c.left = len(c.cells)
	return c
}

// solve returns a program of cost at most limit, or nil.
func (c *coverer) solve(limit int) *Solution {
	c.limit = limit
	if !c.search() {
		return nil
	}
	sol := &Solution{Problem: c.s.p, Parts: make([][]Part, len(c.cell))}
	idx := map[int]int{}
	for k := range c.cell {
		for _, m := range c.pairs[k] {
			var mask uint32
			for b := range 32 {
				if c.cell[k][b] == m {
					mask |= 1 << b
				}
			}
			i, ok := idx[m]
			if !ok {
				i = len(sol.Moves)
				idx[m] = i
				sol.Moves = append(sol.Moves, c.s.moves[m])
			}
			sol.Parts[k] = append(sol.Parts[k], Part{Move: i, Mask: mask})
		}
	}
	return sol
}

// delta returns the cost of giving cell (k, b) move m.
func (c *coverer) delta(k, m int) int {
	d := 0
	if c.pair[k][m] == 0 {
		d++
		if c.count[m] == 0 && c.s.moves[m].Cost() > 0 {
			d++
		}
	}
	return d
}

func (c *coverer) set(k, b, m int) {
	c.cost += c.delta(k, m)
	if c.pair[k][m] == 0 {
		c.pairs[k] = append(c.pairs[k], m)
	}
	c.pair[k][m]++
	c.count[m]++
	c.cell[k][b] = m
	c.used.add(c.s.src[m][0][b])
	c.left--
}

func (c *coverer) unset(k, b, m int) {
	c.left++
	x := c.s.src[m][0][b]
	c.used[x/64] &^= 1 << (x % 64)
	c.cell[k][b] = -1
	c.count[m]--
	c.pair[k][m]--
	if c.pair[k][m] == 0 {
		c.pairs[k] = c.pairs[k][:len(c.pairs[k])-1]
	}
	c.cost -= c.delta(k, m)
}

// open reports whether output k may take a cell: outputs with equal masks
// are interchangeable, so an empty output may start only if every earlier
// output with its mask has started.
func (c *coverer) open(k int) bool {
	if len(c.pairs[k]) > 0 {
		return true
	}
	for j := k - 1; j >= 0; j-- {
		if c.s.p.Outputs[j] == c.s.p.Outputs[k] {
			return len(c.pairs[j]) > 0
		}
	}
	return true
}

// bound returns a lower bound on the cost of any completion.
func (c *coverer) bound() int {
	// Outputs whose free cells their current moves cannot all fill need
	// another part each.
	parts := 0
	for k := range c.s.p.Outputs {
		need := c.minParts[k] - len(c.pairs[k])
		for _, b := range c.s.targets[k] {
			if c.cell[k][b] >= 0 {
				continue
			}
			ok := false
			for _, mv := range c.pairs[k] {
				if x := c.s.src[mv][0][b]; x >= 0 && !c.used.has(x) {
					ok = true
					break
				}
			}
			if !ok {
				need = max(need, 1)
				break
			}
		}
		parts += max(need, 0)
	}
	// Cells that the moves in use cannot fill need new moves, each
	// filling at most one cell per bit of the union mask.
	room := 0
	for m, n := range c.count {
		if n == 0 && c.s.moves[m].Cost() > 0 {
			continue
		}
		for b := range 32 {
			if c.union>>b&1 != 0 {
				if x := c.s.src[m][0][b]; x >= 0 && !c.used.has(x) {
					room++
				}
			}
		}
	}
	moves := 0
	if d := c.left - room; d > 0 {
		w := bits.OnesCount32(c.union)
		moves = (d + w - 1) / w
	}
	return c.cost + moves + max(parts, moves)
}

type choice struct{ k, b, m, d int }

func (c *coverer) search() bool {
	if c.left == 0 {
		for k, n := range c.minParts {
			if len(c.pairs[k]) < n {
				return false
			}
		}
		return true
	}
	if c.bound() > c.limit {
		return false
	}
	budget := c.limit - c.cost
	// Branch on the cell or input with the fewest choices. Every
	// completion makes one of them, so none means a dead end.
	var best, cur []choice
	dead := func() bool {
		if best == nil || len(cur) < len(best) {
			best = append(best[:0], cur...)
		}
		return len(best) == 0
	}
	// Free cells.
	for _, t := range c.cells {
		if c.cell[t.k][t.b] >= 0 || !c.open(t.k) {
			continue
		}
		cur = cur[:0]
		for m := range c.s.moves {
			x := c.s.src[m][0][t.b]
			if x < 0 || c.used.has(x) {
				continue
			}
			if d := c.delta(t.k, m); d <= budget {
				cur = append(cur, choice{t.k, t.b, m, d})
			}
		}
		if dead() {
			return false
		}
	}
	// Free inputs, when every input must be read.
	if c.exact {
		for x := range c.readers {
			if c.used.has(x) {
				continue
			}
			cur = cur[:0]
			for _, r := range c.readers[x] {
				for k := range c.cell {
					if c.s.p.Outputs[k]>>r.b&1 == 0 || c.cell[k][r.b] >= 0 || !c.open(k) {
						continue
					}
					if d := c.delta(k, r.k); d <= budget {
						cur = append(cur, choice{k, r.b, r.k, d})
					}
				}
			}
			if dead() {
				return false
			}
		}
	}
	// Cheapest choices first.
	for d := 0; d <= 2; d++ {
		for _, o := range best {
			if o.d != d {
				continue
			}
			c.set(o.k, o.b, o.m)
			if c.search() {
				return true
			}
			c.unset(o.k, o.b, o.m)
		}
	}
	return false
}

// solve returns a program of cost at most c, or nil. Each output is
// either whole, one move masked once, or split into two or more parts;
// solve picks the whole outputs' moves and leaves the split outputs to
// the coverer.
func (s *searcher) solve(c int) *Solution {
	cv := s.cover
	cv.limit = c
	n := len(s.p.Outputs)
	choice := make([]int, n)
	var rec func(k int) *Solution
	rec = func(k int) *Solution {
		if cv.bound() > c {
			return nil
		}
		if k == n {
			return cv.solve(c)
		}
		same := k > 0 && s.p.Outputs[k-1] == s.p.Outputs[k]
		// Among equal outputs, whole ones come first, with increasing moves.
		if !(same && cv.minParts[k-1] > 1) {
			lo := 0
			if same {
				lo = choice[k-1] + 1
			}
			for j := lo; j < len(s.single[k]); j++ {
				if s.sset[k][j].intersects(cv.used) {
					continue
				}
				m := s.single[k][j]
				choice[k] = j
				for _, b := range s.targets[k] {
					cv.set(k, b, m)
				}
				sol := rec(k + 1)
				for i := len(s.targets[k]) - 1; i >= 0; i-- {
					cv.unset(k, s.targets[k][i], m)
				}
				if sol != nil {
					return sol
				}
			}
		}
		cv.minParts[k] = 2
		sol := rec(k + 1)
		cv.minParts[k] = 0
		return sol
	}
	return rec(0)
}
