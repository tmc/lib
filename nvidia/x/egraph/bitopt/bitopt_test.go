package bitopt

import (
	"errors"
	"math/rand/v2"
	"testing"
)

// nvfp4 is E2M1 to bf16x2 by the subnormal trick: magnitudes at bits 6-8
// and the sign at 15 of each half, four registers from one word.
var nvfp4 = Problem{Words: 1, Outputs: []uint32{0x81C081C0, 0x81C081C0, 0x81C081C0, 0x81C081C0}}

func TestSolveNVFP4(t *testing.T) {
	sol, err := Solve(nvfp4, 9)
	if err != nil {
		t.Fatal(err)
	}
	if got := sol.Cost(); got != 9 {
		t.Errorf("cost %d, want 9", got)
	}
	t.Logf("\n%s", sol.CUDA("dequant"))
}

// TestNoNVFP4Below9 proves that the model has no 8-instruction program for
// any packing.
func TestNoNVFP4Below9(t *testing.T) {
	_, err := Solve(nvfp4, 8)
	if !errors.Is(err, ErrNoProgram) {
		t.Fatalf("Solve(cost 8) = %v, want ErrNoProgram", err)
	}
}

// TestSolveBytes checks a problem with a known answer: four copies of the
// low byte cost q & 0xff and three shifted copies masked, 7 instructions.
func TestSolveBytes(t *testing.T) {
	p := Problem{Words: 1, Outputs: []uint32{0xff, 0xff, 0xff, 0xff}}
	sol, err := Solve(p, 7)
	if err != nil {
		t.Fatal(err)
	}
	if sol.Cost() != 7 {
		t.Errorf("cost %d, want 7", sol.Cost())
	}
	if _, err := Solve(p, 6); !errors.Is(err, ErrNoProgram) {
		t.Errorf("Solve(cost 6) = %v, want ErrNoProgram", err)
	}
}

// TestSolveMatchesNaive compares Solve's optimal cost with a search that
// has no symmetry breaking and no single-part shortcut, on small problems.
func TestSolveMatchesNaive(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping the naive search in -short mode")
	}
	r := rand.New(rand.NewPCG(1, 2))
	costs := map[int]int{} // cost above the number of outputs -> problems
	defer func() { t.Logf("extra cost histogram %v", costs) }()
	for i := range 40 {
		// Shifts and rotations only, so the naive search stays small.
		p := Problem{Words: 1, Ops: []Op{Shl, Shr, Funnel}}
		n := 2 + r.IntN(3)
		for range n {
			var m uint32
			for range 2 + r.IntN(5) {
				m |= 1 << r.IntN(32)
			}
			p.Outputs = append(p.Outputs, m)
		}
		if i%2 == 0 { // equal outputs, which the symmetry breaking prunes
			for k := 1; k < n; k++ {
				p.Outputs[k] = p.Outputs[0]
			}
		}
		want := naiveCost(p, n+3)
		sol, err := Solve(p, n+3)
		got := -1
		if err == nil {
			got = sol.Cost()
		}
		if got != want {
			t.Errorf("%#x: Solve cost %d, naive %d", p.Outputs, got, want)
		}
		if want >= 0 {
			costs[want-len(p.Outputs)]++
		}
	}
}

// naiveCost returns the least cost up to max by enumerating move sets and
// every output's part sets, or -1.
func naiveCost(p Problem, max int) int {
	s := newSearcher(p)
	n := len(p.Outputs)
	nonid := s.moves[p.Words:]
	for c := n; c <= max; c++ {
		for k := 0; k <= c-n; k++ {
			found := false
			var pick []int
			var rec func(start int)
			rec = func(start int) {
				if found {
					return
				}
				if len(pick) == k {
					avail := append([]int(nil), pick...)
					for w := range p.Words {
						avail = append(avail, w)
					}
					if naiveParts(s, avail, c-k) {
						found = true
					}
					return
				}
				for j := start; j < len(nonid); j++ {
					pick = append(pick, p.Words+j)
					rec(j + 1)
					pick = pick[:len(pick)-1]
				}
			}
			rec(0)
			if found {
				return c
			}
		}
	}
	return -1
}

// naiveParts reports whether the outputs can take part sets from avail
// with total size parts and a matching of all their bits.
func naiveParts(s *searcher, avail []int, parts int) bool {
	n := len(s.p.Outputs)
	sets := make([][]int, n)
	var rec func(k, left int) bool
	rec = func(k, left int) bool {
		if k == n {
			return left == 0 && s.match(make([]int, n), seq(n), nil, nil, sets, s.all) != nil
		}
		var cur []int
		var comb func(start int) bool
		comb = func(start int) bool {
			if len(cur) > 0 {
				sets[k] = append([]int(nil), cur...)
				if rec(k+1, left-len(cur)) {
					return true
				}
			}
			if len(cur) == left-(n-k-1) {
				return false
			}
			for j := start; j < len(avail); j++ {
				cur = append(cur, avail[j])
				if comb(j + 1) {
					return true
				}
				cur = cur[:len(cur)-1]
			}
			return false
		}
		return comb(0)
	}
	return rec(0, parts)
}

func seq(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}
