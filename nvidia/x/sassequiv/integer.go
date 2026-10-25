package sassequiv

import (
	"fmt"
	"math/bits"
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
	// A coefficient with t trailing zeros discards the atom's top t
	// bits; simplify each atom to the bits that remain.
	demanded := linear{k: l.k, terms: map[int]uint32{}}
	changed := false
	for _, a := range atoms {
		c := l.terms[a]
		d := a
		if tz := bits.TrailingZeros32(c); tz > 0 {
			d = x.demand(a, 32-tz)
		}
		changed = changed || d != a
		demanded.terms[d] += c
	}
	if changed {
		return x.node(demanded)
	}
	var args []int
	for _, a := range atoms {
		args = append(args, x.add("term", 0, x.imm(uint64(l.terms[a])), a))
	}
	return x.add("lin", len(args), append(args, x.imm(uint64(l.k)))...)
}

// demand returns a value whose low n bits equal v's, simplified by
// discarding what the other bits depend on: PRMT bytes and AND masks
// above bit n.
func (x *blockExec) demand(v int, n int) int {
	g := x.fx.g
	nd := g.nodes[v]
	low := uint32(1)<<n - 1
	if n >= 32 {
		low = ^uint32(0)
	}
	switch {
	case nd.label == "PRMT" && len(nd.args) == 3 && isImm(g, nd.args[1]):
		sel := x.immVal(nd.args[1])
		bytes := (n + 7) / 8
		var out uint32
		usesA, usesB, identity := false, false, true
		for k := range bytes {
			nib := sel >> (4 * k) & 0xf
			out |= nib << (4 * k)
			usesA = usesA || nib&7 < 4
			usesB = usesB || nib&7 >= 4
			identity = identity && nib == uint32(k)
		}
		if identity {
			return x.demand(nd.args[0], n)
		}
		if in := g.nodes[nd.args[0]]; !usesB && in.label == "PRMT" && len(in.args) == 3 && isImm(g, in.args[1]) {
			// Bytes of a PRMT's bytes: compose the selectors.
			inner := x.immVal(in.args[1])
			var sel uint32
			ok := true
			for k := range bytes {
				nib := inner >> (4 * (out >> (4 * k) & 7)) & 0xf
				ok = ok && out>>(4*k)&8 == 0 && nib < 8
				sel |= nib << (4 * k)
			}
			if ok {
				return x.demand(x.add("PRMT", 0, in.args[0], x.imm(uint64(sel)), in.args[2]), n)
			}
		}
		a, b := nd.args[0], nd.args[2]
		if !usesA && usesB {
			// Only B's bytes: select them from A instead.
			var sel uint32
			for k := range bytes {
				sel |= (out>>(4*k)&0xf - 4) << (4 * k)
			}
			return x.demand(x.add("PRMT", 0, b, x.imm(uint64(sel)), x.imm(0)), n)
		}
		if !usesA {
			a = x.imm(0)
		}
		if !usesB {
			b = x.imm(0)
		}
		if bytes == 4 {
			if out == sel && a == nd.args[0] && b == nd.args[2] {
				return v
			}
			return x.add("PRMT", 0, a, x.imm(uint64(out)), b)
		}
		return x.add(fmt.Sprintf("PRMT.low%d", bytes*8), 0, a, x.imm(uint64(out)), b)
	case nd.label == "LOP3.LUT" && len(nd.args) >= 4 && isImm(g, nd.args[1]) && isImm(g, nd.args[3]):
		m, lut := x.immVal(nd.args[1]), x.immVal(nd.args[3])
		switch {
		case lut == 0xc0 && m&low == low: // a & m
			return x.demand(nd.args[0], n)
		case lut == 0xfc && m&low == 0: // a | m
			return x.demand(nd.args[0], n)
		}
	}
	return v
}

func isImm(g *graph, v int) bool { return strings.HasPrefix(g.nodes[v].label, "imm:") }

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
	case "IADD3.X":
		// IADD3.X d, a, b, c, cin, cin2: a + b + c + cin + cin2.
		if len(args) != 5 || x.fx.g.nodes[args[4]].label != "false" {
			return 0, false
		}
		return x.addx(args[:3], args[3]), true
	case "IMAD.X":
		// IMAD.X d, a, b, c, cin: a*b + c + cin.
		if len(args) != 4 {
			return 0, false
		}
		a, b := x.lin(args[0]), x.lin(args[1])
		var p []int
		switch ka, aok := a.constant(); {
		case aok && ka == 0:
		case aok && ka == 1:
			p = []int{args[1]}
		default:
			switch kb, bok := b.constant(); {
			case bok && kb == 0:
			case bok && kb == 1:
				p = []int{args[0]}
			default:
				return 0, false
			}
		}
		return x.addx(append(p, args[2]), args[3]), true
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

// addx returns the high word of a multi-word add: the sum of addends,
// which commute, and the carry cin. Zero addends are dropped.
func (x *blockExec) addx(addends []int, cin int) int {
	var args []int
	for _, a := range addends {
		if x.fx.g.nodes[a].label != "imm:0x0" {
			args = append(args, a)
		}
	}
	slices.Sort(args)
	return x.add("addx", len(args), append(args, cin)...)
}
