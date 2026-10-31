package discover

import (
	"fmt"
	"strings"
)

// An Opcode is one PTX instruction of the carry-free 32-bit alphabet that
// [RunMCTSDiscovery] searches. Every opcode either combines operand bits
// lane by lane (not, and, or, xor, lop3) or routes bits to fixed positions
// (shl, shr, shf, prmt), so each result bit depends on a fixed set of input
// bits. [ProveEquivalent] relies on that.
type Opcode uint8

const (
	OpNot  Opcode = iota + 1 // not.b32 d, a
	OpAnd                    // and.b32 d, a, b
	OpOr                     // or.b32 d, a, b
	OpXor                    // xor.b32 d, a, b
	OpShl                    // shl.b32 d, a, Imm
	OpShr                    // shr.b32 d, a, Imm (logical)
	OpShf                    // shf.l.wrap.b32 d, a, b, Imm: the high word of b:a << Imm
	OpLop3                   // lop3.b32 d, a, b, c, Imm: Imm is the truth table F(0xF0, 0xCC, 0xAA)
	OpPrmt                   // prmt.b32 d, a, b, Imm: Imm is the byte selector (default mode)
	numOpcodes
)

var opcodeNames = [...]string{
	OpNot:  "not.b32",
	OpAnd:  "and.b32",
	OpOr:   "or.b32",
	OpXor:  "xor.b32",
	OpShl:  "shl.b32",
	OpShr:  "shr.b32",
	OpShf:  "shf.l.wrap.b32",
	OpLop3: "lop3.b32",
	OpPrmt: "prmt.b32",
}

func (op Opcode) String() string {
	if op == 0 || op >= numOpcodes {
		return fmt.Sprintf("Opcode(%d)", op)
	}
	return opcodeNames[op]
}

// arity returns the number of data operands op reads.
func (op Opcode) arity() int {
	switch op {
	case OpNot, OpShl, OpShr:
		return 1
	case OpAnd, OpOr, OpXor, OpShf, OpPrmt:
		return 2
	case OpLop3:
		return 3
	}
	return 0
}

// bitwise reports whether bit b of op's result depends only on bit b of its
// operands.
func (op Opcode) bitwise() bool {
	switch op {
	case OpNot, OpAnd, OpOr, OpXor, OpLop3:
		return true
	}
	return false
}

// An Operand is a register or an immediate. Register k < NumInputs is input
// k; register NumInputs+i is the result of instruction i.
type Operand struct {
	Reg   int
	Imm   uint32
	IsImm bool
}

// R returns a register operand.
func R(reg int) Operand { return Operand{Reg: reg} }

// I returns an immediate operand.
func I(v uint32) Operand { return Operand{Imm: v, IsImm: true} }

// An Inst is one instruction. Src holds its data operands; Imm holds the
// shift amount, lop3 truth table or prmt selector.
type Inst struct {
	Op  Opcode
	Src [3]Operand
	Imm uint32
}

// A Program is a straight-line PTX basic block over 32-bit registers. Its
// result is the result of its last instruction.
type Program struct {
	NumInputs int
	Insts     []Inst
}

// MaxInputs is the largest NumInputs a [Program] may have.
const MaxInputs = 6

// Validate reports whether p is well formed: every operand reads an input
// or an earlier result, immediates appear only where PTX accepts them, and
// instruction immediates are in range.
func (p Program) Validate() error {
	if p.NumInputs < 1 || p.NumInputs > MaxInputs {
		return fmt.Errorf("program: %d inputs, want 1 to %d", p.NumInputs, MaxInputs)
	}
	if len(p.Insts) == 0 {
		return fmt.Errorf("program: no instructions")
	}
	for i, in := range p.Insts {
		n := in.Op.arity()
		if n == 0 {
			return fmt.Errorf("program: inst %d: bad opcode %d", i, in.Op)
		}
		for k, s := range in.Src {
			if k >= n {
				if s != (Operand{}) {
					return fmt.Errorf("program: inst %d: unused operand %d set", i, k)
				}
				continue
			}
			if s.IsImm {
				if k == 0 || !in.Op.bitwise() && in.Op != OpPrmt {
					return fmt.Errorf("program: inst %d: immediate operand %d of %v", i, k, in.Op)
				}
				continue
			}
			if s.Reg < 0 || s.Reg >= p.NumInputs+i {
				return fmt.Errorf("program: inst %d: operand %d reads register %d", i, k, s.Reg)
			}
		}
		switch in.Op {
		case OpShl, OpShr, OpShf:
			if in.Imm < 1 || in.Imm > 31 {
				return fmt.Errorf("program: inst %d: shift %d, want 1 to 31", i, in.Imm)
			}
		case OpLop3:
			if in.Imm > 0xff {
				return fmt.Errorf("program: inst %d: lop3 table %#x", i, in.Imm)
			}
		case OpPrmt:
			if in.Imm > 0xffff {
				return fmt.Errorf("program: inst %d: prmt selector %#x", i, in.Imm)
			}
		default:
			if in.Imm != 0 {
				return fmt.Errorf("program: inst %d: %v takes no immediate", i, in.Op)
			}
		}
	}
	return nil
}

// Eval runs p on inputs in, which must have p.NumInputs words.
func (p Program) Eval(in []uint32) uint32 {
	var buf [MaxInputs + 64]uint32
	var regs []uint32
	if n := p.NumInputs + len(p.Insts); n <= len(buf) {
		regs = buf[:n]
	} else {
		regs = make([]uint32, n)
	}
	copy(regs, in[:p.NumInputs])
	for i, inst := range p.Insts {
		regs[p.NumInputs+i] = inst.eval(regs)
	}
	return regs[len(regs)-1]
}

func (in Inst) val(regs []uint32, k int) uint32 {
	s := in.Src[k]
	if s.IsImm {
		return s.Imm
	}
	return regs[s.Reg]
}

func (in Inst) eval(regs []uint32) uint32 {
	switch in.Op {
	case OpNot:
		return ^in.val(regs, 0)
	case OpAnd:
		return in.val(regs, 0) & in.val(regs, 1)
	case OpOr:
		return in.val(regs, 0) | in.val(regs, 1)
	case OpXor:
		return in.val(regs, 0) ^ in.val(regs, 1)
	case OpShl:
		return in.val(regs, 0) << in.Imm
	case OpShr:
		return in.val(regs, 0) >> in.Imm
	case OpShf:
		return shf(in.val(regs, 0), in.val(regs, 1), in.Imm)
	case OpLop3:
		return lop3(in.val(regs, 0), in.val(regs, 1), in.val(regs, 2), in.Imm)
	case OpPrmt:
		return prmt(in.val(regs, 0), in.val(regs, 1), in.Imm)
	}
	panic("discover: bad opcode")
}

func shf(a, b, n uint32) uint32 {
	return uint32((uint64(b)<<32 | uint64(a)) << (n & 31) >> 32)
}

// lop3 evaluates the PTX lop3 truth table lut: bit i of lut is the result
// for a = i>>2&1, b = i>>1&1, c = i&1.
func lop3(a, b, c, lut uint32) uint32 {
	var r uint32
	for i := range 8 {
		if lut>>i&1 == 0 {
			continue
		}
		t := ^uint32(0)
		if i&4 != 0 {
			t &= a
		} else {
			t &^= a
		}
		if i&2 != 0 {
			t &= b
		} else {
			t &^= b
		}
		if i&1 != 0 {
			t &= c
		} else {
			t &^= c
		}
		r |= t
	}
	return r
}

// prmt evaluates the default-mode PTX byte permute: nibble j of sel picks
// byte sel&7 of b:a for result byte j, replicating its sign bit when sel&8.
func prmt(a, b, sel uint32) uint32 {
	x := uint64(b)<<32 | uint64(a)
	var r uint32
	for j := range 4 {
		s := sel >> (4 * j) & 0xf
		v := uint32(x >> (8 * (s & 7)) & 0xff)
		if s&8 != 0 {
			if v&0x80 != 0 {
				v = 0xff
			} else {
				v = 0
			}
		}
		r |= v << (8 * j)
	}
	return r
}

// Len returns the number of instructions in p.
func (p Program) Len() int { return len(p.Insts) }

func (p Program) clone() Program {
	return Program{NumInputs: p.NumInputs, Insts: append([]Inst(nil), p.Insts...)}
}

func (p Program) reg(s Operand) string {
	if s.IsImm {
		return fmt.Sprintf("0x%08X", s.Imm)
	}
	if s.Reg < p.NumInputs {
		return fmt.Sprintf("%%in%d", s.Reg)
	}
	return fmt.Sprintf("%%t%d", s.Reg-p.NumInputs)
}

func (p Program) line(i int, reg func(Operand) string, dst string) string {
	in := p.Insts[i]
	ops := []string{dst}
	for k := range in.Op.arity() {
		ops = append(ops, reg(in.Src[k]))
	}
	switch in.Op {
	case OpShl, OpShr, OpShf:
		ops = append(ops, fmt.Sprint(in.Imm))
	case OpLop3:
		ops = append(ops, fmt.Sprintf("0x%02x", in.Imm))
	case OpPrmt:
		ops = append(ops, fmt.Sprintf("0x%04x", in.Imm))
	}
	return in.Op.String() + " " + strings.Join(ops, ", ") + ";"
}

// PTX returns p's instructions, one per line, reading inputs %in0... and
// writing temporaries %t0...; the result is in the last temporary.
func (p Program) PTX() string {
	var b strings.Builder
	for i := range p.Insts {
		b.WriteString(p.line(i, p.reg, fmt.Sprintf("%%t%d", i)))
		b.WriteByte('\n')
	}
	return b.String()
}

func (p Program) String() string {
	return strings.ReplaceAll(strings.TrimSuffix(p.PTX(), "\n"), "\n", " ")
}

// CUDA returns p as a CUDA device function taking NumInputs uint32_t
// arguments, with p's instructions in an inline PTX block.
func (p Program) CUDA(name string) string {
	var b strings.Builder
	var params, cons []string
	for k := range p.NumInputs {
		params = append(params, fmt.Sprintf("uint32_t in%d", k))
		cons = append(cons, fmt.Sprintf("\"r\"(in%d)", k))
	}
	fmt.Fprintf(&b, "__device__ __forceinline__ uint32_t %s(%s) {\n", name, strings.Join(params, ", "))
	b.WriteString("    uint32_t out;\n")
	fmt.Fprintf(&b, "    asm volatile(\"{\\n\\t.reg .b32 t<%d>;\\n\\t\"\n", len(p.Insts))
	reg := func(s Operand) string {
		if s.IsImm {
			return fmt.Sprintf("0x%08X", s.Imm)
		}
		if s.Reg < p.NumInputs {
			return fmt.Sprintf("%%%d", s.Reg+1)
		}
		return fmt.Sprintf("t%d", s.Reg-p.NumInputs)
	}
	for i := range p.Insts {
		fmt.Fprintf(&b, "        \"%s\\n\\t\"\n", p.line(i, reg, fmt.Sprintf("t%d", i)))
	}
	fmt.Fprintf(&b, "        \"mov.b32 %%0, t%d;\\n\\t}\"\n", len(p.Insts)-1)
	fmt.Fprintf(&b, "        : \"=r\"(out) : %s);\n", strings.Join(cons, ", "))
	b.WriteString("    return out;\n}\n")
	return b.String()
}

// normalize removes instructions whose results the last instruction does
// not use and renumbers registers.
func (p Program) normalize() Program {
	n := len(p.Insts)
	live := make([]bool, n)
	live[n-1] = true
	for i := n - 1; i >= 0; i-- {
		if !live[i] {
			continue
		}
		in := p.Insts[i]
		for k := range in.Op.arity() {
			if s := in.Src[k]; !s.IsImm && s.Reg >= p.NumInputs {
				live[s.Reg-p.NumInputs] = true
			}
		}
	}
	remap := make([]int, n)
	out := Program{NumInputs: p.NumInputs}
	for i, in := range p.Insts {
		if !live[i] {
			continue
		}
		for k := range in.Op.arity() {
			if s := in.Src[k]; !s.IsImm && s.Reg >= p.NumInputs {
				in.Src[k].Reg = remap[s.Reg-p.NumInputs]
			}
		}
		remap[i] = p.NumInputs + len(out.Insts)
		out.Insts = append(out.Insts, in)
	}
	return out
}

// users returns, for each instruction, the instructions that read its
// result.
func (p Program) users() [][]int {
	u := make([][]int, len(p.Insts))
	for i, in := range p.Insts {
		for k := range in.Op.arity() {
			if s := in.Src[k]; !s.IsImm && s.Reg >= p.NumInputs {
				j := s.Reg - p.NumInputs
				if len(u[j]) == 0 || u[j][len(u[j])-1] != i {
					u[j] = append(u[j], i)
				}
			}
		}
	}
	return u
}
