package bitopt

import (
	"fmt"
	"math/bits"
	"strings"
)

// A Problem asks for Outputs registers, each zero outside its mask, whose
// mask bits read distinct bits of Words input words. Input bits no output
// reads are unused.
type Problem struct {
	Words   int      // 1 to 4
	Outputs []uint32 // the bits each output register holds
	Ops     []Op     // the moves a program may use besides Ident; nil means all
}

func (p Problem) allows(op Op) bool {
	if op == Ident || p.Ops == nil {
		return true
	}
	for _, o := range p.Ops {
		if o == op {
			return true
		}
	}
	return false
}

func (p Problem) check() error {
	if p.Words < 1 || p.Words > 4 {
		return fmt.Errorf("bitopt: %d words, want 1 to 4", p.Words)
	}
	if len(p.Outputs) == 0 {
		return fmt.Errorf("bitopt: no outputs")
	}
	n := 0
	for _, m := range p.Outputs {
		n += bits.OnesCount32(m)
	}
	if n > 32*p.Words {
		return fmt.Errorf("bitopt: outputs hold %d bits, inputs %d", n, 32*p.Words)
	}
	return nil
}

// An Op is a move's instruction.
type Op int

const (
	Ident  Op = iota // A
	Shl              // A << N
	Shr              // A >> N
	Funnel           // __funnelshift_l(A, B, N): the high word of B:A << N; a rotation when A == B
	Perm             // __byte_perm(A, B, Sel): bytes of B:A
)

// A Move is one instruction on input words A and B.
type Move struct {
	Op   Op
	A, B int
	N    int    // shift amount
	Sel  uint32 // byte selector for Perm
}

// Cost returns the instructions the move takes: 0 for Ident, else 1.
func (m Move) Cost() int {
	if m.Op == Ident {
		return 0
	}
	return 1
}

// src returns the input bit (32·word + bit) that bit b of the move's value
// reads, or -1 if it is zero.
func (m Move) src(b int) int {
	switch m.Op {
	case Ident:
		return 32*m.A + b
	case Shl:
		if b < m.N {
			return -1
		}
		return 32*m.A + b - m.N
	case Shr:
		if b+m.N >= 32 {
			return -1
		}
		return 32*m.A + b + m.N
	case Funnel:
		if b >= m.N {
			return 32*m.B + b - m.N
		}
		return 32*m.A + b - m.N + 32
	case Perm:
		s := int(m.Sel>>(4*(b/8))) & 7
		w := m.A
		if s >= 4 {
			w = m.B
		}
		return 32*w + 8*(s%4) + b%8
	}
	panic("bitopt: bad op")
}

// eval returns the move's value for input words q.
func (m Move) eval(q []uint32) uint32 {
	a, b := q[m.A], q[m.B]
	switch m.Op {
	case Ident:
		return a
	case Shl:
		return a << m.N
	case Shr:
		return a >> m.N
	case Funnel:
		return uint32((uint64(b)<<32 | uint64(a)) << m.N >> 32)
	case Perm:
		x := uint64(b)<<32 | uint64(a)
		var r uint32
		for j := 0; j < 4; j++ {
			s := m.Sel >> (4 * j) & 7
			r |= uint32(x>>(8*s)&0xff) << (8 * j)
		}
		return r
	}
	panic("bitopt: bad op")
}

func (m Move) cuda() string {
	switch m.Op {
	case Ident:
		return fmt.Sprintf("q[%d]", m.A)
	case Shl:
		return fmt.Sprintf("q[%d] << %d", m.A, m.N)
	case Shr:
		return fmt.Sprintf("q[%d] >> %d", m.A, m.N)
	case Funnel:
		return fmt.Sprintf("__funnelshift_l(q[%d], q[%d], %d)", m.A, m.B, m.N)
	case Perm:
		return fmt.Sprintf("__byte_perm(q[%d], q[%d], 0x%04x)", m.A, m.B, m.Sel)
	}
	panic("bitopt: bad op")
}

func (m Move) String() string { return m.cuda() }

// A Part is a move masked to some of an output's bits.
type Part struct {
	Move int // index in Solution.Moves
	Mask uint32
}

// A Solution is a program for a Problem: output k is the OR over Parts[k]
// of Moves[part.Move] & part.Mask.
type Solution struct {
	Problem Problem
	Moves   []Move
	Parts   [][]Part
}

// Cost returns the program's instruction count in the model: one per
// non-identity move and one per part.
func (s *Solution) Cost() int {
	n := 0
	for _, m := range s.Moves {
		n += m.Cost()
	}
	for _, p := range s.Parts {
		n += len(p)
	}
	return n
}

// Source returns the input bit, 32·word + bit, that bit b of output k
// reads, or -1 if b is not in the output's mask.
func (s *Solution) Source(k, b int) int {
	for _, p := range s.Parts[k] {
		if p.Mask>>b&1 != 0 {
			return s.Moves[p.Move].src(b)
		}
	}
	return -1
}

// Eval runs the program on input words q.
func (s *Solution) Eval(q []uint32) []uint32 {
	v := make([]uint32, len(s.Moves))
	for i, m := range s.Moves {
		v[i] = m.eval(q)
	}
	out := make([]uint32, len(s.Parts))
	for k, ps := range s.Parts {
		for _, p := range ps {
			out[k] |= v[p.Move] & p.Mask
		}
	}
	return out
}

// Pack returns the input words that make output bit (k, b) equal bit b of
// fields[k], for every bit in the outputs' masks.
func (s *Solution) Pack(fields []uint32) []uint32 {
	q := make([]uint32, s.Problem.Words)
	for k, m := range s.Problem.Outputs {
		for b := 0; b < 32; b++ {
			if m>>b&1 != 0 && fields[k]>>b&1 != 0 {
				x := s.Source(k, b)
				q[x/32] |= 1 << (x % 32)
			}
		}
	}
	return q
}

// Check proves the program correct for every input: each output is zero
// outside its mask, each mask bit reads a distinct input bit, and the
// program computes exactly that routing. Every operation routes bits
// without combining them (the parts of an output have disjoint masks), so
// each output bit is a single input bit or zero, and the unit vectors
// determine the program.
func (s *Solution) Check() error {
	p := s.Problem
	if err := p.check(); err != nil {
		return err
	}
	if len(s.Parts) != len(p.Outputs) {
		return fmt.Errorf("bitopt: %d outputs, problem has %d", len(s.Parts), len(p.Outputs))
	}
	for k, ps := range s.Parts {
		var cover uint32
		for _, pt := range ps {
			if pt.Move < 0 || pt.Move >= len(s.Moves) {
				return fmt.Errorf("bitopt: output %d: bad move %d", k, pt.Move)
			}
			if cover&pt.Mask != 0 {
				return fmt.Errorf("bitopt: output %d: parts overlap", k)
			}
			cover |= pt.Mask
		}
		if cover != p.Outputs[k] {
			return fmt.Errorf("bitopt: output %d: parts cover %#08x, want %#08x", k, cover, p.Outputs[k])
		}
	}
	for _, m := range s.Moves {
		if m.A < 0 || m.A >= p.Words || m.B < 0 || m.B >= p.Words {
			return fmt.Errorf("bitopt: move %v reads a missing word", m)
		}
	}
	// want[k][b]: the input bit routed to output bit (k, b).
	used := make(map[int]bool)
	want := make([][32]int, len(p.Outputs))
	for k, m := range p.Outputs {
		for b := range 32 {
			want[k][b] = -1
			if m>>b&1 == 0 {
				continue
			}
			x := s.Source(k, b)
			if x < 0 {
				return fmt.Errorf("bitopt: output %d bit %d reads a zero", k, b)
			}
			if used[x] {
				return fmt.Errorf("bitopt: input bit %d read twice", x)
			}
			used[x] = true
			want[k][b] = x
		}
	}
	q := make([]uint32, p.Words)
	if out := s.Eval(q); !allZero(out) {
		return fmt.Errorf("bitopt: nonzero output for zero input")
	}
	for x := range 32 * p.Words {
		clear(q)
		q[x/32] = 1 << (x % 32)
		out := s.Eval(q)
		for k := range out {
			for b := range 32 {
				got := out[k]>>b&1 != 0
				if got != (want[k][b] == x) {
					return fmt.Errorf("bitopt: input bit %d: output %d bit %d is %v", x, k, b, got)
				}
			}
		}
	}
	return nil
}

func allZero(v []uint32) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

// CUDA returns the program as
//
//	__device__ __forceinline__ void name(const uint32_t (&q)[Words], uint32_t (&a)[len(Outputs)])
func (s *Solution) CUDA(name string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d instructions (bitopt).\n", name, s.Cost())
	fmt.Fprintf(&b, "__device__ __forceinline__ void %s(const uint32_t (&q)[%d], uint32_t (&a)[%d]) {\n",
		name, s.Problem.Words, len(s.Parts))
	ref := make([]string, len(s.Moves))
	for i, m := range s.Moves {
		if m.Op == Ident {
			ref[i] = m.cuda()
			continue
		}
		ref[i] = fmt.Sprintf("t%d", i)
		fmt.Fprintf(&b, "    const uint32_t t%d = %s;\n", i, m.cuda())
	}
	for k, ps := range s.Parts {
		terms := make([]string, len(ps))
		for j, p := range ps {
			terms[j] = fmt.Sprintf("(%s & 0x%08xu)", ref[p.Move], p.Mask)
		}
		fmt.Fprintf(&b, "    a[%d] = %s;\n", k, strings.Join(terms, " | "))
	}
	b.WriteString("}\n")
	return b.String()
}
