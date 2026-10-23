package sassequiv

import "strings"

// assume returns v simplified for an instruction that runs only when
// guard holds: a value selected by guard, or by its negation, is the
// selected operand. ptxas allocates registers that a predicated
// instruction writes without regard to what they hold when the
// predicate is false, so the value read under the same predicate is
// otherwise a different merge at every register limit.
func (x *blockExec) assume(v, guard int) int {
	if x.fx.assumed == nil {
		x.fx.assumed = map[[2]int]int{}
	}
	return x.assumeDepth(v, guard, 8)
}

func (x *blockExec) assumeDepth(v, guard, depth int) int {
	key := [2]int{v, guard}
	if r, ok := x.fx.assumed[key]; ok {
		return r
	}
	g := x.fx.g
	n := g.nodes[v]
	r := v
	switch {
	case depth == 0 || len(n.args) == 0 || isPhi(g, v) || memArg(g, v) >= 0 || strings.HasPrefix(n.label, "part:"):
		// Leaves, merges and memory operations stay as they are: a
		// load under the guard read memory under it already.
	case n.label == "sel" && implies(g, guard, n.args[0]):
		r = x.assumeDepth(n.args[1], guard, depth-1)
	case n.label == "sel" && implies(g, guard, x.not(n.args[0])):
		r = x.assumeDepth(n.args[2], guard, depth-1)
	case n.label == "lin":
		l := linear{k: x.immVal(n.args[len(n.args)-1]), terms: map[int]uint32{}}
		changed := false
		for _, t := range n.args[:n.comm] {
			term := g.nodes[t]
			a := x.assumeDepth(term.args[1], guard, depth-1)
			changed = changed || a != term.args[1]
			l = l.plus(x.lin(a).scale(x.immVal(term.args[0])))
		}
		if changed {
			r = x.node(l)
		}
	default:
		args := make([]int, len(n.args))
		changed := false
		for i, a := range n.args {
			args[i] = x.assumeDepth(a, guard, depth-1)
			changed = changed || args[i] != a
		}
		if changed {
			r = x.add(n.label, n.comm, args...)
		}
	}
	x.fx.assumed[key] = r
	return r
}

// implies reports whether guard holding means c holds.
func implies(g *graph, guard, c int) bool {
	return guard == c
}
