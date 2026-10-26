package sassequiv

import (
	"slices"
	"strings"
)

// plainStores are the stores that order only by address: ptxas emits
// those to disjoint addresses in any order.
var plainStores = map[string]bool{"ST": true, "STG": true, "STS": true, "STL": true}

// store returns the memory state after the store label(guard, mem,
// args...). Of two plain stores to provably disjoint addresses off one
// base, the one at the lower offset is the earlier in the result, so a
// pair stored in either order is one state.
func (x *blockExec) store(label string, guard, mem int, args []int) int {
	g := x.fx.g
	m := g.nodes[mem]
	if x.plain(label) && m.label == label && len(args) == 2 && len(m.args) == 4 {
		if d, ok := x.distance(m.args[2], args[0]); ok && d < 0 && disjoint(d, bytesOf(label), bytesOf(label)) {
			inner := x.store(label, guard, m.args[1], args)
			return x.add(m.label, 0, append([]int{m.args[0], inner}, m.args[2:]...)...)
		}
	}
	return x.add(label, 0, append([]int{guard, mem}, args...)...)
}

// plain reports whether label is a plain store with no ordering of its
// own: not .STRONG, .MMIO, or a release.
func (x *blockExec) plain(label string) bool {
	f := strings.Split(label, ".")
	return plainStores[f[0]] && !slices.ContainsFunc(f, func(s string) bool {
		return s == "STRONG" || s == "MMIO" || s == "RELEASE" || s == "SYS" || s == "GPU" || s == "VOLATILE"
	})
}

// distance returns b - a for addresses a and b that differ by a constant:
// the same descriptor, the same 64-bit bases and the same 32-bit terms.
// A guarded store reads its address where its guard holds, so an assumed
// operand is the operand.
func (x *blockExec) distance(a, b int) (int64, bool) {
	g := x.fx.g
	na, nb := g.nodes[a], g.nodes[b]
	if na.label == "desc" && nb.label == "desc" && len(na.args) == 2 && len(nb.args) == 2 && g.strip(na.args[0]) == g.strip(nb.args[0]) {
		return x.distance(na.args[1], nb.args[1])
	}
	if na.label != "addr" || nb.label != "addr" || len(na.args) != len(nb.args) || na.comm != nb.comm {
		return 0, false
	}
	n := len(na.args)
	for i := range n - 1 {
		if g.strip(na.args[i]) != g.strip(nb.args[i]) {
			return 0, false
		}
	}
	la, lb := x.lin(g.strip(na.args[n-1])), x.lin(g.strip(nb.args[n-1]))
	for t, c := range la.terms {
		if lb.terms[t] != c {
			return 0, false
		}
	}
	for t, c := range lb.terms {
		if la.terms[t] != c {
			return 0, false
		}
	}
	return int64(int32(lb.k)) - int64(int32(la.k)), true
}

// disjoint reports whether [0, wa) and [d, d+wb) do not overlap, for a
// distance small enough that no address wraps.
func disjoint(d int64, wa, wb int) bool {
	if d <= -1<<24 || d >= 1<<24 {
		return false
	}
	return d >= int64(wa) || -d >= int64(wb)
}

// bytesOf returns how many bytes a load or store moves.
func bytesOf(label string) int {
	f := strings.Split(label, ".")
	switch {
	case slices.Contains(f, "U8"), slices.Contains(f, "S8"):
		return 1
	case slices.Contains(f, "U16"), slices.Contains(f, "S16"):
		return 2
	}
	return 4 * widthOf(label)
}
