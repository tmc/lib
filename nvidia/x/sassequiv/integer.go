package sassequiv

import (
	"slices"
	"strconv"
	"strings"
)

// A linear form is const + sum(coef * atom) modulo 2^32: the value of an
// integer expression that ptxas may compute with IADD3, IMAD, LEA,
// LEA.HI, VIADD or a left shift, in any grouping.
type linear struct {
	k     uint32
	terms map[int]uint32 // atom node -> coefficient
}

func (x *blockExec) lin(v int) linear {
	l := linear{terms: map[int]uint32{}}
	n := x.fx.g.nodes[v]
	switch {
	case strings.HasPrefix(n.label, "imm:"):
		u, _ := strconv.ParseUint(strings.TrimPrefix(n.label, "imm:0x"), 16, 64)
		l.k = uint32(u)
	case n.label == "lin":
		for _, t := range n.args[:n.comm] {
			term := x.fx.g.nodes[t]
			l.terms[term.args[1]] += x.immVal(term.args[0])
		}
		l.k = x.immVal(n.args[len(n.args)-1])
	case strings.HasPrefix(n.label, "mod:-") && n.label == "mod:-":
		l = x.lin(n.args[0]).scale(^uint32(0))
	case n.label == "mod:~":
		l = x.lin(n.args[0]).scale(^uint32(0))
		l.k--
	default:
		l.terms[v] = 1
	}
	return l
}

func (x *blockExec) immVal(v int) uint32 {
	u, _ := strconv.ParseUint(strings.TrimPrefix(x.fx.g.nodes[v].label, "imm:0x"), 16, 64)
	return uint32(u)
}

func (l linear) scale(c uint32) linear {
	m := linear{k: l.k * c, terms: map[int]uint32{}}
	for a, k := range l.terms {
		m.terms[a] = k * c
	}
	return m
}

func (l linear) plus(m linear) linear {
	r := linear{k: l.k + m.k, terms: map[int]uint32{}}
	for a, k := range l.terms {
		r.terms[a] += k
	}
	for a, k := range m.terms {
		r.terms[a] += k
	}
	return r
}

// constant reports whether l has no atoms, and its value.
func (l linear) constant() (uint32, bool) {
	for _, c := range l.terms {
		if c != 0 {
			return 0, false
		}
	}
	return l.k, true
}

// node returns l as a value: an immediate, a lone atom, or a lin node of
// commuting terms coef*atom and then the constant.
func (x *blockExec) node(l linear) int {
	var atoms []int
	for a, c := range l.terms {
		if c != 0 {
			atoms = append(atoms, a)
		}
	}
	slices.Sort(atoms)
	if len(atoms) == 0 {
		return x.imm(uint64(l.k))
	}
	if len(atoms) == 1 && l.k == 0 && l.terms[atoms[0]] == 1 {
		return atoms[0]
	}
	var args []int
	for _, a := range atoms {
		args = append(args, x.add("term", 0, x.imm(uint64(l.terms[a])), a))
	}
	return x.add("lin", len(args), append(args, x.imm(uint64(l.k)))...)
}

// shift returns the shift amount immediate v holds, if it is one below 32.
func (x *blockExec) shift(v int) (uint32, bool) {
	if !strings.HasPrefix(x.fx.g.nodes[v].label, "imm:") {
		return 0, false
	}
	s := x.immVal(v)
	return s, s < 32
}

// integer computes an integer instruction's value in canonical form, or
// reports false to leave it as written. label is the canonical opcode,
// args its sources. Instructions that also write a carry are left alone,
// since the carry depends on how the sum was grouped.
func (x *blockExec) integer(label string, args []int) (int, bool) {
	switch label {
	case "IADD3":
		if len(args) != 3 {
			return 0, false
		}
		return x.node(x.lin(args[0]).plus(x.lin(args[1])).plus(x.lin(args[2]))), true
	case "VIADD":
		if len(args) != 2 {
			return 0, false
		}
		return x.node(x.lin(args[0]).plus(x.lin(args[1]))), true
	case "IMAD", "IMAD.SHL.U32", "IMAD.IADD", "IMAD.U32":
		if len(args) != 3 {
			return 0, false
		}
		a, b, c := x.lin(args[0]), x.lin(args[1]), x.lin(args[2])
		if k, ok := a.constant(); ok {
			return x.node(b.scale(k).plus(c)), true
		}
		if k, ok := b.constant(); ok {
			return x.node(a.scale(k).plus(c)), true
		}
		return x.node(x.lin(x.add("mul.lo", 2, args[0], args[1])).plus(c)), true
	case "LEA":
		if len(args) != 3 {
			return 0, false
		}
		s, ok := x.shift(args[2])
		if !ok {
			return 0, false
		}
		return x.node(x.lin(args[0]).scale(1 << s).plus(x.lin(args[1]))), true
	case "LEA.HI":
		// LEA.HI d, a, b, c, s: b + ({c, a} << s) >> 32.
		if len(args) != 4 {
			return 0, false
		}
		s, ok := x.shift(args[3])
		if !ok || s == 0 {
			return 0, false
		}
		hi := x.lin(x.add("shr.u32", 0, args[0], x.imm(uint64(32-s))))
		return x.node(hi.plus(x.lin(args[1])).plus(x.lin(args[2]).scale(1 << s))), true
	case "LEA.HI.SX32":
		if len(args) != 3 {
			return 0, false
		}
		s, ok := x.shift(args[2])
		if !ok || s == 0 {
			return 0, false
		}
		hi := x.lin(x.add("shr.s32", 0, args[0], x.imm(uint64(32-s))))
		return x.node(hi.plus(x.lin(args[1]))), true
	case "SHF.L.U32", "SHF.L.W.U32":
		if len(args) != 3 || x.fx.g.nodes[args[2]].label != "imm:0x0" {
			return 0, false
		}
		s, ok := x.shift(args[1])
		if !ok {
			return 0, false
		}
		return x.node(x.lin(args[0]).scale(1 << s)), true
	case "SHF.R.U32.HI", "SHF.R.S32.HI":
		if len(args) != 3 || x.fx.g.nodes[args[0]].label != "imm:0x0" {
			return 0, false
		}
		s, ok := x.shift(args[1])
		if !ok {
			return 0, false
		}
		if s == 0 {
			return args[2], true
		}
		kind := "shr.u32"
		if label == "SHF.R.S32.HI" {
			kind = "shr.s32"
		}
		return x.add(kind, 0, args[2], x.imm(uint64(s))), true
	}
	return 0, false
}
