package bitopt

import (
	"errors"
	"fmt"
	"math/bits"
)

// ErrNoProgram reports that no program within the cost bound exists in the
// model.
var ErrNoProgram = errors.New("bitopt: no program within the cost bound")

// Solve returns a cheapest program for p with cost at most maxCost, trying
// costs in increasing order, or an error wrapping [ErrNoProgram] when the
// model has none: a proof that every program costs more than maxCost.
//
// The search is exact but exponential in the number of moves a program
// uses; it is fast for one or two words.
func Solve(p Problem, maxCost int) (*Solution, error) {
	if err := p.check(); err != nil {
		return nil, err
	}
	s := newSearcher(p)
	for c := len(p.Outputs); c <= maxCost; c++ {
		if sol := s.solve(c); sol != nil {
			if err := sol.Check(); err != nil {
				return nil, fmt.Errorf("bitopt: internal error: %w", err)
			}
			return sol, nil
		}
	}
	return nil, fmt.Errorf("%w: cost %d", ErrNoProgram, maxCost)
}

// set is a set of input bits.
type set [2]uint64

func (s set) has(x int) bool        { return s[x/64]>>(x%64)&1 != 0 }
func (s *set) add(x int)            { s[x/64] |= 1 << (x % 64) }
func (s set) and(t set) set         { return set{s[0] & t[0], s[1] & t[1]} }
func (s set) or(t set) set          { return set{s[0] | t[0], s[1] | t[1]} }
func (s set) andNot(t set) set      { return set{s[0] &^ t[0], s[1] &^ t[1]} }
func (s set) empty() bool           { return s[0] == 0 && s[1] == 0 }
func (s set) count() int            { return bits.OnesCount64(s[0]) + bits.OnesCount64(s[1]) }
func (s set) intersects(t set) bool { return !s.and(t).empty() }

type target struct{ k, b int }

type searcher struct {
	p       Problem
	moves   []Move
	src     [][][32]int // [move][output][bit] -> input bit or -1
	targets [][]int     // [output] -> mask bits
	single  [][]int     // [output] -> moves that route all its bits to distinct inputs
	sset    [][]set     // [output][i] -> inputs of single[output][i]
	all     set         // every input bit
}

func newSearcher(p Problem) *searcher {
	s := &searcher{p: p}
	for w := range p.Words {
		s.moves = append(s.moves, Move{Op: Ident, A: w, B: w})
	}
	var cands []Move
	for w := range p.Words {
		for n := 1; n < 32; n++ {
			cands = append(cands, Move{Op: Shl, A: w, B: w, N: n}, Move{Op: Shr, A: w, B: w, N: n})
		}
		for n := 1; n < 32; n++ {
			cands = append(cands, Move{Op: Funnel, A: w, B: w, N: n})
		}
	}
	for a := range p.Words {
		for b := range p.Words {
			if a != b {
				for n := 1; n < 32; n++ {
					cands = append(cands, Move{Op: Funnel, A: a, B: b, N: n})
				}
			}
		}
	}
	for a := range p.Words {
		for b := a; b < p.Words; b++ {
			hi := 4
			if b != a {
				hi = 8
			}
			for sel := 0; sel < 1<<16; sel++ {
				ok := true
				for j := range 4 {
					if sel>>(4*j)&0xf >= hi {
						ok = false
						break
					}
				}
				if ok && !(a == b && sel == 0x3210) {
					cands = append(cands, Move{Op: Perm, A: a, B: b, Sel: uint32(sel)})
				}
			}
		}
	}
	var mask uint32
	for _, m := range p.Outputs {
		mask |= m
		var tb []int
		for b := range 32 {
			if m>>b&1 != 0 {
				tb = append(tb, b)
			}
		}
		s.targets = append(s.targets, tb)
	}
	for x := range 32 * p.Words {
		s.all.add(x)
	}
	// Keep one move per action on the output bits; identities first.
	seen := make(map[string]bool)
	key := func(m Move) string {
		var b []byte
		for i := range 32 {
			if mask>>i&1 != 0 {
				b = append(b, byte(m.src(i)+1))
			}
		}
		return string(b)
	}
	for _, m := range s.moves {
		seen[key(m)] = true
	}
	for _, m := range cands {
		if !p.allows(m.Op) {
			continue
		}
		if k := key(m); !seen[k] {
			seen[k] = true
			s.moves = append(s.moves, m)
		}
	}
	s.src = make([][][32]int, len(s.moves))
	for i, m := range s.moves {
		s.src[i] = make([][32]int, len(p.Outputs))
		for k := range p.Outputs {
			for b := range 32 {
				s.src[i][k][b] = m.src(b)
			}
		}
	}
	s.single = make([][]int, len(p.Outputs))
	s.sset = make([][]set, len(p.Outputs))
	for k, tb := range s.targets {
		for i := range s.moves {
			var in set
			ok := true
			for _, b := range tb {
				x := s.src[i][k][b]
				if x < 0 || in.has(x) {
					ok = false
					break
				}
				in.add(x)
			}
			if ok {
				s.single[k] = append(s.single[k], i)
				s.sset[k] = append(s.sset[k], in)
			}
		}
	}
	return s
}

// solve returns a program of cost exactly c, or nil.
func (s *searcher) solve(c int) *Solution {
	n := len(s.p.Outputs)
	parts := make([]int, n)
	var rec func(k, sum int) *Solution
	rec = func(k, sum int) *Solution {
		if k == n {
			return s.try(parts, c-sum)
		}
		// Outputs with equal masks are interchangeable: keep their part
		// counts nonincreasing.
		hi := c - sum - (n - k - 1)
		if k > 0 && s.p.Outputs[k] == s.p.Outputs[k-1] {
			hi = min(hi, parts[k-1])
		}
		for p := 1; p <= hi; p++ {
			parts[k] = p
			if sol := rec(k+1, sum+p); sol != nil {
				return sol
			}
		}
		return nil
	}
	return rec(0, 0)
}

// try looks for a program whose output k has parts[k] parts and whose
// non-identity moves number budget or fewer.
func (s *searcher) try(parts []int, budget int) *Solution {
	n := len(parts)
	var singles, multis []int
	for k := range n {
		if parts[k] == 1 {
			singles = append(singles, k)
		} else {
			multis = append(multis, k)
		}
	}
	choice := make([]int, n) // single output -> move
	var used set
	chosen := map[int]bool{}
	var rec func(i, cost int) *Solution
	rec = func(i, cost int) *Solution {
		if i == len(singles) {
			return s.multi(parts, multis, choice, singles, used, chosen, budget-cost)
		}
		k := singles[i]
		lo := 0
		if i > 0 && s.p.Outputs[singles[i-1]] == s.p.Outputs[k] {
			lo = choice[singles[i-1]] + 1 // equal outputs take increasing moves
		}
		for j := lo; j < len(s.single[k]); j++ {
			m := s.single[k][j]
			in := s.sset[k][j]
			if in.intersects(used) {
				continue
			}
			add := 0
			if !chosen[m] && s.moves[m].Cost() > 0 {
				add = 1
			}
			if cost+add > budget {
				continue
			}
			choice[k] = j
			old := used
			used = used.or(in)
			was := chosen[m]
			chosen[m] = true
			if sol := rec(i+1, cost+add); sol != nil {
				return sol
			}
			used = old
			chosen[m] = was
		}
		return nil
	}
	return rec(0, 0)
}

// multi completes a program whose single-part outputs are fixed: it picks
// up to budget new moves and each multi-part output's moves, then matches
// those outputs' bits to the unused inputs.
func (s *searcher) multi(parts, multis, choice, singles []int, used set, chosen map[int]bool, budget int) *Solution {
	if len(multis) == 0 {
		return s.build(parts, choice, singles, nil, nil)
	}
	free := s.all.andNot(used)
	// Moves that can feed some multi-part output from the free inputs.
	var base, extra []int
	for i := range s.moves {
		useful := false
		for _, k := range multis {
			for _, b := range s.targets[k] {
				if x := s.src[i][k][b]; x >= 0 && free.has(x) {
					useful = true
					break
				}
			}
			if useful {
				break
			}
		}
		if !useful {
			continue
		}
		if chosen[i] || s.moves[i].Cost() == 0 {
			base = append(base, i)
		} else {
			extra = append(extra, i)
		}
	}
	var pick []int
	var recE func(start, left int) *Solution
	recE = func(start, left int) *Solution {
		avail := append(append([]int(nil), base...), pick...)
		if sol := s.assign(parts, multis, choice, singles, avail, pick, free); sol != nil {
			return sol
		}
		if left == 0 {
			return nil
		}
		for j := start; j < len(extra); j++ {
			pick = append(pick, extra[j])
			if sol := recE(j+1, left-1); sol != nil {
				return sol
			}
			pick = pick[:len(pick)-1]
		}
		return nil
	}
	return recE(0, budget)
}

// assign gives each multi-part output parts[k] moves from avail, using
// every move in mustUse, and matches their bits to free.
func (s *searcher) assign(parts, multis, choice, singles, avail, mustUse []int, free set) *Solution {
	sets := make([][]int, len(multis))
	var rec func(i int) *Solution
	rec = func(i int) *Solution {
		if i == len(multis) {
			for _, m := range mustUse {
				found := false
				for _, st := range sets {
					for _, x := range st {
						if x == m {
							found = true
						}
					}
				}
				if !found {
					return nil
				}
			}
			return s.match(parts, multis, choice, singles, sets, free)
		}
		k := multis[i]
		var cand []int
		for _, m := range avail {
			for _, b := range s.targets[k] {
				if x := s.src[m][k][b]; x >= 0 && free.has(x) {
					cand = append(cand, m)
					break
				}
			}
		}
		var cur []int
		var comb func(start int) *Solution
		comb = func(start int) *Solution {
			if len(cur) == parts[k] {
				// Equal outputs with equal part counts take increasing sets.
				if i > 0 && s.p.Outputs[multis[i-1]] == s.p.Outputs[k] && parts[multis[i-1]] == parts[k] && less(cur, sets[i-1]) {
					return nil
				}
				sets[i] = append([]int(nil), cur...)
				return rec(i + 1)
			}
			for j := start; j < len(cand); j++ {
				cur = append(cur, cand[j])
				if sol := comb(j + 1); sol != nil {
					return sol
				}
				cur = cur[:len(cur)-1]
			}
			return nil
		}
		return comb(0)
	}
	return rec(0)
}

func less(a, b []int) bool {
	for i := range a {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return false
}

// match finds distinct free inputs for every bit of the multi-part
// outputs, output k reading through sets[i]'s moves.
func (s *searcher) match(parts, multis, choice, singles []int, sets [][]int, free set) *Solution {
	type edge struct{ x, m int }
	var ts []target
	var adj [][]edge
	for i, k := range multis {
		for _, b := range s.targets[k] {
			var e []edge
			for _, m := range sets[i] {
				if x := s.src[m][k][b]; x >= 0 && free.has(x) {
					e = append(e, edge{x, m})
				}
			}
			if len(e) == 0 {
				return nil
			}
			ts = append(ts, target{k, b})
			adj = append(adj, e)
		}
	}
	owner := make(map[int]int) // input -> target index
	via := make([]int, len(ts))
	var aug func(t int, seen map[int]bool) bool
	aug = func(t int, seen map[int]bool) bool {
		for _, e := range adj[t] {
			if seen[e.x] {
				continue
			}
			seen[e.x] = true
			if o, ok := owner[e.x]; !ok || aug(o, seen) {
				owner[e.x] = t
				via[t] = e.m
				return true
			}
		}
		return false
	}
	for t := range ts {
		if !aug(t, map[int]bool{}) {
			return nil
		}
	}
	return s.build(parts, choice, singles, ts, via)
}

// build assembles a Solution from the single-part choices and the matched
// multi-part bits.
func (s *searcher) build(parts, choice, singles []int, ts []target, via []int) *Solution {
	idx := map[int]int{}
	sol := &Solution{Problem: s.p, Parts: make([][]Part, len(parts))}
	ref := func(m int) int {
		if i, ok := idx[m]; ok {
			return i
		}
		idx[m] = len(sol.Moves)
		sol.Moves = append(sol.Moves, s.moves[m])
		return idx[m]
	}
	for _, k := range singles {
		sol.Parts[k] = []Part{{Move: ref(s.single[k][choice[k]]), Mask: s.p.Outputs[k]}}
	}
	masks := map[target]uint32{} // (output, move) -> mask
	var order []target
	for t, tg := range ts {
		key := target{tg.k, via[t]}
		if _, ok := masks[key]; !ok {
			order = append(order, key)
		}
		masks[key] |= 1 << tg.b
	}
	for _, key := range order {
		sol.Parts[key.k] = append(sol.Parts[key.k], Part{Move: ref(key.b), Mask: masks[key]})
	}
	return sol
}
