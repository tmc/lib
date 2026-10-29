package discover

import (
	"errors"
	"fmt"
	"math/bits"
	"math/rand/v2"
)

// A SpecFunc is the reference semantics a [Program] must match: it maps
// NumInputs input words to one output word.
type SpecFunc func(in []uint32) uint32

// ErrNotEquivalent reports that a program differs from its specification
// or from the program it should replace.
var ErrNotEquivalent = errors.New("discover: not equivalent")

// ErrUnproven reports that [ProveEquivalent] could not decide equivalence
// because some output bit depends on too many input bits to enumerate.
var ErrUnproven = errors.New("discover: equivalence unproven")

// maxProofSupport bounds the input bits one output bit may depend on in
// [ProveEquivalent]; 2^maxProofSupport assignments are enumerated per bit.
const maxProofSupport = 16

// boundaryWords are the corner values of the 64-point boundary
// fingerprint: zero, sign bits, f32 infinities and subnormal limits, f16x2
// infinities, nibble and byte carry boundaries and alternating patterns.
var boundaryWords = [16]uint32{
	0x00000000, 0x80000000, 0xFFFFFFFF, 0x7FFFFFFF,
	0x7F800000, 0xFF800000, 0x00000001, 0x007FFFFF,
	0x0F0F0F0F, 0xF0F0F0F0, 0x00FF00FF, 0xFF00FF00,
	0x55555555, 0xAAAAAAAA, 0x7C007C00, 0x88888888,
}

// BoundaryVectors returns the 64 Stage A input vectors for n inputs, built
// from corner values (zero, ±0 and ±Inf bit patterns, subnormal limits,
// sign-bit and nibble carry boundaries) rotated by whole bytes.
func BoundaryVectors(n int) [][]uint32 {
	vs := make([][]uint32, 64)
	for v := range vs {
		in := make([]uint32, n)
		for k := range in {
			w := boundaryWords[(v*(2*k+1)+3*k)%16]
			in[k] = bits.RotateLeft32(w, 8*((v/16+k)%4))
		}
		vs[v] = in
	}
	return vs
}

// An Oracle checks programs against a [SpecFunc] in two stages: Stage A
// compares the 64 [BoundaryVectors], Stage B a fixed set of pseudo-random
// vectors. Expected outputs are computed once.
type Oracle struct {
	n        int
	boundary [][]uint32
	random   [][]uint32
	wantA    []uint32
	wantB    []uint32
}

// NewOracle returns an oracle for spec over n inputs with randomVectors
// Stage B vectors drawn from a PCG seeded with seed.
func NewOracle(n int, spec SpecFunc, randomVectors int, seed uint64) (*Oracle, error) {
	if n < 1 || n > MaxInputs {
		return nil, fmt.Errorf("new oracle: %d inputs, want 1 to %d", n, MaxInputs)
	}
	if spec == nil {
		return nil, fmt.Errorf("new oracle: nil spec")
	}
	o := &Oracle{n: n, boundary: BoundaryVectors(n)}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	o.random = make([][]uint32, randomVectors)
	for i := range o.random {
		in := make([]uint32, n)
		for k := range in {
			in[k] = rng.Uint32()
		}
		o.random[i] = in
	}
	for _, in := range o.boundary {
		o.wantA = append(o.wantA, spec(in))
	}
	for _, in := range o.random {
		o.wantB = append(o.wantB, spec(in))
	}
	return o, nil
}

// StageA reports whether p matches the specification on the boundary
// vectors.
func (o *Oracle) StageA(p Program) bool { return match(p, o.boundary, o.wantA) }

// StageB reports whether p matches the specification on the random
// vectors.
func (o *Oracle) StageB(p Program) bool { return match(p, o.random, o.wantB) }

func match(p Program, ins [][]uint32, want []uint32) bool {
	for i, in := range ins {
		if p.Eval(in) != want[i] {
			return false
		}
	}
	return true
}

// Verify runs Stage A and Stage B on cand and then proves it equivalent to
// baseline with [ProveEquivalent]. The stages are tests against the
// specification; the proof is exact but relative to baseline, which must
// itself pass the stages.
func (o *Oracle) Verify(baseline, cand Program) error {
	if err := cand.Validate(); err != nil {
		return err
	}
	if cand.NumInputs != o.n {
		return fmt.Errorf("verify: %d inputs, oracle has %d", cand.NumInputs, o.n)
	}
	if !o.StageA(cand) {
		return fmt.Errorf("verify: stage A boundary vectors: %w", ErrNotEquivalent)
	}
	if !o.StageB(cand) {
		return fmt.Errorf("verify: stage B %d random vectors: %w", len(o.random), ErrNotEquivalent)
	}
	return ProveEquivalent(baseline, cand)
}

// bitset is a set of input bits 32·k + b, k < MaxInputs.
type bitset [MaxInputs * 32 / 64]uint64

func (s *bitset) add(x int) { s[x/64] |= 1 << (x % 64) }
func (s bitset) or(t bitset) bitset {
	for i := range s {
		s[i] |= t[i]
	}
	return s
}

func (s bitset) list() []int {
	var xs []int
	for i, w := range s {
		for w != 0 {
			b := bits.TrailingZeros64(w)
			xs = append(xs, 64*i+b)
			w &= w - 1
		}
	}
	return xs
}

// support returns, for each bit of p's result, a superset of the input
// bits it depends on.
func support(p Program) [32]bitset {
	regs := make([][32]bitset, p.NumInputs+len(p.Insts))
	for k := range p.NumInputs {
		for b := range 32 {
			regs[k][b].add(32*k + b)
		}
	}
	src := func(in Inst, k int) [32]bitset {
		if in.Src[k].IsImm {
			return [32]bitset{}
		}
		return regs[in.Src[k].Reg]
	}
	for i, in := range p.Insts {
		var d [32]bitset
		switch {
		case in.Op.bitwise():
			for k := range in.Op.arity() {
				s := src(in, k)
				for b := range 32 {
					d[b] = d[b].or(s[b])
				}
			}
		case in.Op == OpShl:
			a := src(in, 0)
			for b := int(in.Imm); b < 32; b++ {
				d[b] = a[b-int(in.Imm)]
			}
		case in.Op == OpShr:
			a := src(in, 0)
			for b := 0; b+int(in.Imm) < 32; b++ {
				d[b] = a[b+int(in.Imm)]
			}
		case in.Op == OpShf:
			a, hi := src(in, 0), src(in, 1)
			n := int(in.Imm & 31)
			for b := range 32 {
				if b >= n {
					d[b] = hi[b-n]
				} else {
					d[b] = a[b-n+32]
				}
			}
		case in.Op == OpPrmt:
			a, hi := src(in, 0), src(in, 1)
			for j := range 4 {
				s := int(in.Imm >> (4 * j) & 0xf)
				w, base := a, 8*(s&3)
				if s&4 != 0 {
					w = hi
				}
				for b := range 8 {
					if s&8 != 0 {
						d[8*j+b] = w[base+7]
					} else {
						d[8*j+b] = w[base+b]
					}
				}
			}
		}
		regs[p.NumInputs+i] = d
	}
	return regs[len(regs)-1]
}

// ProveEquivalent proves that a and b compute the same function of their
// inputs, or returns an error wrapping [ErrNotEquivalent] with a
// counterexample, or [ErrUnproven].
//
// Every opcode is carry-free, so each result bit is a Boolean function of
// a statically known set of input bits (its support). For each of the 32
// result bits, ProveEquivalent enumerates every assignment of the union of
// the two supports, with all other input bits zero, and compares that bit.
// Input bits outside the support cannot change it, so the enumeration
// covers all 2^(32·NumInputs) inputs.
func ProveEquivalent(a, b Program) error {
	if a.NumInputs != b.NumInputs {
		return fmt.Errorf("prove equivalent: %d and %d inputs: %w", a.NumInputs, b.NumInputs, ErrNotEquivalent)
	}
	if err := a.Validate(); err != nil {
		return fmt.Errorf("prove equivalent: %w", err)
	}
	if err := b.Validate(); err != nil {
		return fmt.Errorf("prove equivalent: %w", err)
	}
	sa, sb := support(a), support(b)
	in := make([]uint32, a.NumInputs)
	for bit := range 32 {
		xs := sa[bit].or(sb[bit]).list()
		if len(xs) > maxProofSupport {
			return fmt.Errorf("prove equivalent: result bit %d depends on %d input bits: %w", bit, len(xs), ErrUnproven)
		}
		for m := range 1 << len(xs) {
			clear(in)
			for i, x := range xs {
				if m>>i&1 != 0 {
					in[x/32] |= 1 << (x % 32)
				}
			}
			if (a.Eval(in)^b.Eval(in))>>bit&1 != 0 {
				return fmt.Errorf("prove equivalent: result bit %d differs at inputs %#08x: %w", bit, in, ErrNotEquivalent)
			}
		}
	}
	return nil
}

// A MutantReport counts the single-point mutants of a program and how many
// verification rejected. Accepted mutants are proven equivalent to the
// program; they are semantically neutral edits, not verifier escapes.
type MutantReport struct {
	Total    int
	Rejected int
	Accepted int
}

// Mutants returns the single-point mutants of p: each lop3 truth-table bit
// flipped, each prmt selector bit flipped, each shift amount moved by one,
// each immediate operand bit flipped, and each and/or/xor replaced by the
// other two.
func Mutants(p Program) []Program {
	var ms []Program
	add := func(i int, in Inst) {
		m := p.clone()
		m.Insts[i] = in
		if m.Validate() == nil {
			ms = append(ms, m)
		}
	}
	for i, in := range p.Insts {
		switch in.Op {
		case OpLop3:
			for b := range 8 {
				in := in
				in.Imm ^= 1 << b
				add(i, in)
			}
		case OpPrmt:
			for b := range 16 {
				in := in
				in.Imm ^= 1 << b
				add(i, in)
			}
		case OpShl, OpShr, OpShf:
			for _, d := range []int{-1, 1} {
				in := in
				in.Imm = uint32(int(in.Imm) + d)
				add(i, in)
			}
		case OpAnd, OpOr, OpXor:
			for _, op := range []Opcode{OpAnd, OpOr, OpXor} {
				if op != in.Op {
					in := in
					in.Op = op
					add(i, in)
				}
			}
		}
		for k := range in.Op.arity() {
			if !in.Src[k].IsImm {
				continue
			}
			for b := range 32 {
				in := in
				in.Src[k].Imm ^= 1 << b
				add(i, in)
			}
		}
	}
	return ms
}

// CheckMutants verifies every mutant of p against o and baseline.
func (o *Oracle) CheckMutants(baseline, p Program) MutantReport {
	var r MutantReport
	for _, m := range Mutants(p) {
		r.Total++
		if o.Verify(baseline, m) != nil {
			r.Rejected++
		} else {
			r.Accepted++
		}
	}
	return r
}

// A CostModel scores programs by modelled cycles. It is a model: single
// in-order issue per cycle, an instruction issuing once its operands are
// ready, with a fixed result latency per opcode. The defaults are not
// measured on hardware.
type CostModel struct {
	Latency [numOpcodes]int
}

// DefaultCostModel returns a model with a 4-cycle result latency for every
// opcode, the dependent-issue latency commonly reported for integer ALU
// instructions on recent SMs.
func DefaultCostModel() CostModel {
	var m CostModel
	for op := OpNot; op < numOpcodes; op++ {
		m.Latency[op] = 4
	}
	return m
}

// Cycles returns the cycle at which p's result is ready.
func (m CostModel) Cycles(p Program) int {
	ready := make([]int, p.NumInputs+len(p.Insts))
	issue := -1
	for i, in := range p.Insts {
		t := issue + 1
		for k := range in.Op.arity() {
			if s := in.Src[k]; !s.IsImm {
				t = max(t, ready[s.Reg])
			}
		}
		issue = t
		ready[p.NumInputs+i] = t + m.Latency[in.Op]
	}
	return ready[len(ready)-1]
}

// PeakRegs returns the largest number of registers live at any instruction
// of p in program order: inputs from entry to their last read, results from
// their definition to their last read.
func PeakRegs(p Program) int {
	n := p.NumInputs + len(p.Insts)
	last := make([]int, n)
	for r := range last {
		last[r] = -1
	}
	for i, in := range p.Insts {
		for k := range in.Op.arity() {
			if s := in.Src[k]; !s.IsImm {
				last[s.Reg] = i
			}
		}
	}
	last[n-1] = len(p.Insts)
	peak := 0
	for i := range p.Insts {
		live := 0
		for r := range n {
			def := r - p.NumInputs
			if def <= i && last[r] >= i {
				live++
			}
		}
		peak = max(peak, live)
	}
	return peak
}
