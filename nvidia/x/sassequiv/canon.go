package sassequiv

import (
	"fmt"
	"strings"
)

// part is register k of a multi-register value: an element of a vector
// built from separate words, or a projection of one result.
func (x *blockExec) part(v, k int) int {
	if n := x.fx.g.nodes[v]; n.label == "vec" && k < len(n.args) {
		return n.args[k]
	}
	return x.add(fmt.Sprintf("part:%d", k), 0, v)
}

// vec returns a value held in consecutive registers. The registers of one
// multi-register result are that result, and a 64-bit address formed
// with LEA and LEA.HI.X is the IMAD.WIDE it computes: (ext(a) << s) + base.
func (x *blockExec) vec(parts []int) int {
	g := x.fx.g
	if len(parts) > 1 {
		whole := -1
		for k, p := range parts {
			n := g.nodes[p]
			if n.label != fmt.Sprintf("part:%d", k) || whole >= 0 && n.args[0] != whole {
				whole = -1
				break
			}
			whole = n.args[0]
		}
		if whole >= 0 {
			return whole
		}
	}
	if len(parts) == 2 {
		if v, ok := x.lea64(parts[0], parts[1]); ok {
			return v
		}
	}
	return x.add("vec", 0, parts...)
}

// lea64 recognizes lo = LEA a, blo, s (writing carry c) and hi =
// LEA.HI.X a, bhi, 0, s, c (or LEA.HI.X.SX32 a, bhi, s, c).
func (x *blockExec) lea64(lo, hi int) (int, bool) {
	g := x.fx.g
	l, nh := g.nodes[lo], g.nodes[hi]
	if l.label != "LEA" || len(l.args) != 3 {
		return 0, false
	}
	a, blo, s := l.args[0], l.args[1], l.args[2]
	carry := x.add("part:1000", 0, lo)
	var label string
	switch {
	case nh.label == "LEA.HI.X" && len(nh.args) == 5 && nh.args[0] == a && g.nodes[nh.args[2]].label == "imm:0x0" && nh.args[3] == s && nh.args[4] == carry:
		label = "mad64.u32"
	case nh.label == "LEA.HI.X.SX32" && len(nh.args) == 4 && nh.args[0] == a && nh.args[2] == s && nh.args[3] == carry:
		label = "mad64.s32"
	default:
		return 0, false
	}
	n, ok := x.shift(s)
	if !ok {
		return 0, false
	}
	return x.add(label, 2, a, x.imm(1<<n), x.add("vec", 0, blo, nh.args[1])), true
}

// uniform lists the uniform-datapath opcodes that compute what their
// vector namesakes do; ptxas moves work between the two freely.
var uniform = map[string]bool{
	"UIADD3": true, "UIMAD": true, "ULOP3": true, "USHF": true, "ULEA": true,
	"UISETP": true, "USEL": true, "UPRMT": true, "UFLO": true, "UPOPC": true,
	"UBREV": true, "USGXT": true, "UBMSK": true, "UIABS": true, "UIMNMX": true,
}

// canonical returns the label an opcode's result carries.
func canonical(op, base string) string {
	if uniform[base] {
		op = op[1:]
	}
	switch op {
	case "IMAD.WIDE.U32":
		return "mad64.u32"
	case "IMAD.WIDE":
		return "mad64.s32"
	}
	return op
}

// plop3 folds PLOP3.LUT d0, d1, a, b, c, lut, lut1 when constant inputs
// reduce d0 to one input or its negation, as ptxas uses it to copy a
// uniform predicate into a vector one.
func (x *blockExec) plop3(ops []string, guard int) bool {
	if len(ops) != 7 {
		return false
	}
	d0, d1 := parseOperand(ops[0]), parseOperand(ops[1])
	if d1.core != "PT" && d1.core != "UPT" {
		return false
	}
	lut, ok := immediate(ops[5])
	if !ok || !strings.HasPrefix(ops[5], "0x") {
		return false
	}
	in := make([]int, 3)
	known := make([]int, 3) // 0 variable, 1 true, 2 false
	var free []int
	for i := range 3 {
		in[i] = x.val(ops[2+i], 1)
		switch x.fx.g.nodes[in[i]].label {
		case "true":
			known[i] = 1
		case "false":
			known[i] = 2
		default:
			free = append(free, i)
		}
	}
	if len(free) > 1 {
		return false
	}
	eval := func(v bool) bool {
		idx := 0
		for i := range 3 {
			bit := known[i] == 1
			if known[i] == 0 {
				bit = v
			}
			if bit {
				idx |= 4 >> i
			}
		}
		return lut>>idx&1 == 1
	}
	f, t := eval(false), eval(true)
	var r int
	switch {
	case f && t:
		r = x.leaf("true")
	case !f && !t:
		r = x.leaf("false")
	case t:
		r = in[free[0]]
	default:
		r = x.not(in[free[0]])
	}
	x.write(d0.core, r, guard)
	return true
}

var mirror = map[string]string{
	"LT": "GT", "LE": "GE", "LTU": "GTU", "LEU": "GEU",
}

// comparison puts a SETP comparison in one form: a < b becomes b > a, so
// ptxas's choice of operand order does not matter, and EQ and NE commute
// and ignore signedness.
func (x *blockExec) comparison(label string, args []int, comm int) (string, int) {
	t := strings.Split(label, ".")
	switch t[0] {
	case "ISETP", "FSETP", "DSETP", "HSETP2":
	default:
		return label, comm
	}
	if len(t) < 2 || len(args) < 2 {
		return label, comm
	}
	switch c := t[1]; {
	case mirror[c] != "":
		t[1] = mirror[c]
		args[0], args[1] = args[1], args[0]
	case c == "EQ" || c == "NE" || c == "EQU" || c == "NEU" || c == "NUM" || c == "NAN":
		comm = 2
		if t[0] == "ISETP" && len(t) > 2 && t[2] == "U32" {
			t = append(t[:2], t[3:]...)
		}
	}
	return strings.Join(t, "."), comm
}
