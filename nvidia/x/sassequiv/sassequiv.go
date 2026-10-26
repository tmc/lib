package sassequiv

import (
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// A Verdict is the outcome of comparing two functions.
type Verdict int

const (
	Unknown   Verdict = iota // the model cannot decide
	Equal                    // same values stored, same branches taken
	Different                // some stored value, address, order or condition differs
)

func (v Verdict) String() string {
	switch v {
	case Equal:
		return "equal"
	case Different:
		return "different"
	}
	return "unknown"
}

// A Result is the comparison of one function.
type Result struct {
	Function string
	Verdict  Verdict
	// Reason says why the verdict is Unknown, or where the functions
	// first differ: the two expressions and their instructions.
	Reason string
	// Before and After are the addresses of the instructions where a
	// difference shows, when Verdict is Different.
	Before, After uint64
}

func (r Result) String() string {
	if r.Reason == "" {
		return fmt.Sprintf("%s: %s", r.Function, r.Verdict)
	}
	return fmt.Sprintf("%s: %s: %s", r.Function, r.Verdict, r.Reason)
}

// Compare compares two compilations of one function.
func Compare(before, after *cudasass.Function) Result {
	r := Result{Function: before.Name}
	g := newGraph()
	a, b := build(g, 0, before), build(g, 1, after)
	switch {
	case a.why != "":
		r.Reason = "before: " + a.why
		return r
	case b.why != "":
		r.Reason = "after: " + b.why
		return r
	case len(a.blocks) != len(b.blocks) || !slices.Equal(a.edges, b.edges):
		r.Reason = fmt.Sprintf("control flow differs: %d blocks, %d edges before; %d blocks, %d edges after",
			len(a.blocks), len(a.edges), len(b.blocks), len(b.edges))
		return r
	}
	keys := map[string]bool{}
	for k := range a.roots {
		keys[k] = true
	}
	for k := range b.roots {
		keys[k] = true
	}
	sorted := make([]string, 0, len(keys))
	for k := range keys {
		sorted = append(sorted, k)
	}
	sort.Strings(sorted)
	class := g.classes()
	for _, k := range sorted {
		x, okx := a.roots[k]
		y, oky := b.roots[k]
		if !okx || !oky {
			r.Verdict = Different
			r.Reason = fmt.Sprintf("%s is only %s", k, map[bool]string{true: "before", false: "after"}[okx])
			return r
		}
		if class[x] != class[y] {
			d, ok := explain(g, class, x, y)
			if !ok {
				r.Reason = k + ": differs, but no single operation differs (values move between loop iterations or blocks)"
				return r
			}
			x, y := d.x(), d.y()
			r.Before, r.After = g.nodes[x].pc[0], g.nodes[y].pc[1]
			r.Reason = fmt.Sprintf("%s: before %#x %s; after %#x %s", k, r.Before, g.expr(x, 2), r.After, g.expr(y, 2))
			var why string
			r.Verdict, why = classify(g, class, d)
			if why != "" {
				r.Reason = why + ": " + r.Reason
			}
			if r.Verdict == Unknown {
				r.Before, r.After = 0, 0
			}
			return r
		}
	}
	r.Verdict = Equal
	return r
}

// A step is a pair of nodes compared, argument i of the step before.
type step struct {
	x, y, i int
}

// A diff is the path from two inequivalent roots down to the first pair
// of nodes whose labels differ.
type diff []step

func (d diff) x() int { return d[len(d)-1].x }
func (d diff) y() int { return d[len(d)-1].y }

// explain descends from two inequivalent nodes to a pair whose labels
// differ, trying each order of commuting operands, and reports whether it
// found one.
func explain(g *graph, class []int, x, y int) (diff, bool) {
	seen := map[[2]int]bool{}
	var walk func(d diff) (diff, bool)
	walk = func(d diff) (diff, bool) {
		x, y := d.x(), d.y()
		if class[x] == class[y] || seen[[2]int{x, y}] {
			return nil, false
		}
		seen[[2]int{x, y}] = true
		nx, ny := g.nodes[x], g.nodes[y]
		if nx.label != ny.label || len(nx.args) != len(ny.args) || nx.comm != ny.comm {
			return d, true
		}
		ay := slices.Clone(ny.args)
		for _, perm := range permutations(nx.comm) {
			for i, j := range perm {
				ay[i] = ny.args[j]
			}
			for i, a := range nx.args {
				if r, ok := walk(append(d[:len(d):len(d)], step{a, ay[i], i})); ok {
					return r, true
				}
			}
		}
		return nil, false
	}
	return walk(diff{{x, y, 0}})
}

// classify decides whether a localized difference is a difference in the
// computation or possibly another form of the same one.
func classify(g *graph, class []int, d diff) (Verdict, string) {
	x, y := d.x(), d.y()
	nx, ny := g.nodes[x], g.nodes[y]
	parent := -1
	if len(d) > 1 {
		parent = d[len(d)-2].x
	}
	switch {
	case reordered(g, class, d):
		return Unknown, "memory operations reordered"
	case g.mergesUndefined(x) || g.mergesUndefined(y):
		// ptxas leaves a register unwritten on paths that never use
		// it, so an undefined value need not make the results differ.
		return Unknown, "reads a register no path writes"
	case isPhi(g, x) || isPhi(g, y):
		// An operation on merged values against a merge of operations:
		// code moved across a block boundary, which the model does not
		// follow.
		return Unknown, "values merge at different blocks"
	case isFloat(g, x) && isFloat(g, y):
		return Different, ""
	case parent >= 0 && isFloat(g, parent) && !isInteger(g, x) && !isInteger(g, y):
		// A floating-point operation reads a different constant,
		// input, or operand modifier.
		return Different, ""
	case len(nx.args) == 0 && len(ny.args) == 0 && injective(g, class, d):
		return Different, "constant differs"
	case len(nx.args) > 0 && len(ny.args) > 0 && onSide(g, class, 1, x) && onSide(g, class, 0, y):
		// Both values exist on both sides, used in swapped places: two
		// loads' addresses exchanged, say.
		return Different, "values exchanged"
	}
	// Integer arithmetic that differs in form: ptxas computes one index
	// many ways (shifts, masks, LEA, IMAD, induction variables), and the
	// model does not decide whether two forms agree.
	return Unknown, "integer expressions differ in form"
}

// injective reports whether the constants that differ at the end of d
// make the integer value above them differ: each integer operation on
// the path differs in that argument alone and adds it, so no other
// argument can compensate and no mask or shift can discard it.
func injective(g *graph, class []int, d diff) bool {
	if len(d) < 2 {
		return false
	}
	for k := len(d) - 2; k >= 0; k-- {
		px, i := d[k].x, d[k+1].i
		n := g.nodes[px]
		switch {
		case n.label == "lin" && i == len(n.args)-1:
			// The constant term.
		case n.label == "addr":
			// A term of an address.
		case (n.label == "mad64.u32" || n.label == "mad64.s32") && i == 2:
			// The 64-bit addend.
		case (n.label == "IADD3" || n.label == "IADD3.X") && i < 3:
			// An addend of a 32-bit sum, with or without carries.
		case n.label == "vec" || n.label == "desc":
			// A tuple differs where any element does.
			continue
		case isPhi(g, px):
		case isFloat(g, px):
			// A floating-point operation on a different value.
			return true
		case isMemory(g, px):
			// An access to a different address, or a store of a
			// different value.
			return siblingsEqual(g, class, d, k)
		default:
			return false
		}
		if !siblingsEqual(g, class, d, k) {
			return false
		}
	}
	return false
}

// reordered reports whether d reaches its difference in a way that two
// memory operations exchanging places would explain: through a load's
// memory state, which can differ without changing what it reads, or
// through a memory operation whose address and memory state both differ.
// Memory operations on different addresses commute, and the model does
// not decide whether two addresses differ.
func reordered(g *graph, class []int, d diff) bool {
	for k := 0; k+1 < len(d); k++ {
		x, y, i := d[k].x, d[k].y, d[k+1].i
		m := memArg(g, x)
		if m < 0 {
			continue
		}
		addr := 0
		if m == 1 {
			addr = 2 // an effect: guard, memory state, address, ...
		}
		ax, ay := g.nodes[x].args, g.nodes[y].args
		memDiffers := class[ax[m]] != class[ay[m]]
		addrDiffers := addr < len(ax) && addr < len(ay) && class[ax[addr]] != class[ay[addr]]
		switch {
		case i == m && m != 1:
			return true
		case memDiffers && addrDiffers:
			return true
		}
	}
	return false
}

// memArg returns the index of a memory operation's memory state
// argument, or -1.
func memArg(g *graph, id int) int {
	n := g.nodes[id]
	base, _, _ := strings.Cut(n.label, ".")
	switch {
	case effects[base]:
		return 1
	case loads[base]:
		return len(n.args) - 1
	}
	return -1
}

// siblingsEqual reports whether the nodes of step d[k] agree in every
// argument but the one d[k+1] descends into. An argument pair that is a
// step of d before k counts as equal: a loop-carried value equals itself
// until the iteration in which the difference first appears.
func siblingsEqual(g *graph, class []int, d diff, k int) bool {
	nx, ny := g.nodes[d[k].x], g.nodes[d[k].y]
	i := d[k+1].i
	if len(nx.args) != len(ny.args) {
		return false
	}
	eq := func(a, b int) bool {
		if class[a] == class[b] {
			return true
		}
		for _, s := range d[:k] {
			if s.x == a && s.y == b {
				return true
			}
		}
		return false
	}
	// Match commuting arguments in any order.
	var ay []int
	used := make([]bool, nx.comm)
	for j := range nx.comm {
		ay = append(ay, ny.args[j])
	}
	if i < nx.comm {
		used[slices.Index(ay, d[k+1].y)] = true
	}
Comm:
	for j := range nx.comm {
		if j == i {
			continue
		}
		for m, b := range ay {
			if !used[m] && eq(nx.args[j], b) {
				used[m] = true
				continue Comm
			}
		}
		return false
	}
	for j := nx.comm; j < len(nx.args); j++ {
		if j != i && !eq(nx.args[j], ny.args[j]) {
			return false
		}
	}
	return true
}

// isMemory reports whether a node is a memory access.
func isMemory(g *graph, id int) bool {
	base, _, _ := strings.Cut(g.nodes[id].label, ".")
	switch base {
	case "LD", "LDG", "LDS", "LDL", "LDC", "ST", "STG", "STS", "STL", "ATOM", "ATOMG", "ATOMS", "RED", "REDG":
		return true
	}
	return false
}

// isInteger reports whether a node is integer or bitwise arithmetic.
func isInteger(g *graph, id int) bool {
	l := g.nodes[id].label
	base, _, _ := strings.Cut(l, ".")
	switch base {
	case "lin", "mul", "shr", "LOP3", "SHF", "IMAD", "LEA", "IADD3", "PRMT", "SEL",
		"IABS", "IMNMX", "VIMNMX", "VIADDMNMX", "BMSK", "SGXT", "FLO", "POPC", "BREV",
		"mad64", "vec", "addr", "desc":
		return true
	}
	return false
}

// permutations returns the orders of n commuting operands to try.
func permutations(n int) [][]int {
	switch n {
	case 2:
		return [][]int{{0, 1}, {1, 0}}
	case 3:
		return [][]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}
	}
	return [][]int{nil}
}

// CompareAll compares the functions of two disassemblies by name. A
// function present on one side only is Different.
func CompareAll(before, after *cudasass.Disassembly) []Result {
	fa := map[string]*cudasass.Function{}
	for _, f := range after.Functions {
		fa[f.Name] = f
	}
	var out []Result
	seen := map[string]bool{}
	for _, f := range before.Functions {
		seen[f.Name] = true
		g, ok := fa[f.Name]
		if !ok {
			out = append(out, Result{Function: f.Name, Verdict: Different, Reason: "missing after"})
			continue
		}
		out = append(out, Compare(f, g))
	}
	for _, f := range after.Functions {
		if !seen[f.Name] {
			out = append(out, Result{Function: f.Name, Verdict: Different, Reason: "missing before"})
		}
	}
	return out
}

func isPhi(g *graph, id int) bool { return strings.HasPrefix(g.nodes[id].label, "phi:") }

// isFloat reports whether a node is a floating-point or matrix operation.
func isFloat(g *graph, id int) bool {
	base, _, _ := strings.Cut(g.nodes[id].label, ".")
	switch base {
	case "FADD", "FMUL", "FFMA", "FMNMX", "FSETP", "FSEL", "FCHK", "FRND", "FSWZADD",
		"MUFU", "HADD2", "HMUL2", "HFMA2", "HMNMX2", "HSETP2", "DADD", "DMUL", "DFMA",
		"DSETP", "F2F", "F2FP", "F2I", "I2F", "I2FP", "HMMA", "HGMMA", "QGMMA":
		return true
	}
	return false
}

// onSide reports whether side has a node equivalent to id.
func onSide(g *graph, class []int, side, id int) bool {
	for i, n := range g.nodes {
		if n.has[side] && class[i] == class[id] {
			return true
		}
	}
	return false
}
