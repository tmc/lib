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
// The search is exact but exponential. One word takes milliseconds; two
// words with every op take seconds to find NVFP4's 16-instruction program
// and minutes to rule out 15.
func Solve(p Problem, maxCost int) (*Solution, error) {
	if err := p.check(); err != nil {
		return nil, err
	}
	s := newSearcher(p)
	s.cover = newCoverer(s)
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
	cover   *coverer
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
