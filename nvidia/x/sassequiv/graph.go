package sassequiv

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// A node is one value in a value graph: an operation, a leaf (a constant,
// a special register, a parameter word) or a phi at a block entry. The
// graph of both functions under comparison shares one node table.
type node struct {
	label string
	args  []int
	comm  int       // the first comm args commute
	alias int       // for a phi with one incoming value, that value, or -1
	pc    [2]uint64 // an instruction that produced it, per side
	has   [2]bool
}

type graph struct {
	nodes  []node
	intern map[string]int
	// edges holds, for each phi, the condition under which control
	// arrives with each argument, or -1 for an unconditional edge.
	edges     map[int][]int
	invariant map[int]bool
	// low holds, for a multi-register value whose register 0 is
	// computed apart from it, that register.
	low map[int]int
}

func newGraph() *graph {
	return &graph{intern: map[string]int{}, edges: map[int][]int{}, invariant: map[int]bool{}, low: map[int]int{}}
}

// once reports whether a value is computed at most once per run of the
// function: it depends on no value that depends on itself, as a value
// carried around a loop does, so it is the same wherever it is read.
func (g *graph) once(id int) bool {
	if r, ok := g.invariant[id]; ok {
		return r
	}
	g.invariant[id] = false // cut cycles
	r := true
	for _, a := range g.nodes[id].args {
		if !r {
			break
		}
		r = g.once(a)
	}
	g.invariant[id] = r
	return r
}

// negation returns not(c) if the graph holds it, or -1.
func (g *graph) negation(c int) int {
	if n := g.nodes[c]; n.label == "not" {
		return g.resolve(n.args[0])
	}
	var k strings.Builder
	k.WriteString("not\x000,")
	k.WriteString(strconv.Itoa(c))
	if id, ok := g.intern[k.String()]; ok {
		return g.resolve(id)
	}
	return -1
}

// contradicts reports whether conditions a and b cannot both hold.
func (g *graph) contradicts(a, b int) bool {
	na, nb := g.nodes[a], g.nodes[b]
	return na.label == "not" && g.resolve(na.args[0]) == b || nb.label == "not" && g.resolve(nb.args[0]) == a
}

// gate resolves assume(v, guard), the value v holds where guard holds:
// a phi whose other incoming edges contradict guard is the value along
// the remaining edge. It reports whether it aliased the node.
func (g *graph) gate(i int) bool {
	n := &g.nodes[i]
	v, guard := g.resolve(n.args[0]), g.resolve(n.args[1])
	side := 0
	if !n.has[0] {
		side = 1
	}
	if s := g.nodes[v]; s.label == "vec" {
		// Each register of a tuple holds where the tuple does.
		args := make([]int, len(s.args))
		for k, a := range s.args {
			args[k] = g.add(side, n.pc[side], "assume", 0, a, guard)
		}
		return g.setAlias(i, g.add(side, n.pc[side], "vec", 0, args...))
	}
	if s := g.nodes[v]; s.label == "sel" {
		c := g.resolve(s.args[0])
		switch {
		case implies(g, guard, c):
			return g.setAlias(i, g.add(side, n.pc[side], "assume", 0, s.args[1], guard))
		case g.contradicts(guard, c) || g.negation(c) >= 0 && implies(g, guard, g.negation(c)):
			return g.setAlias(i, g.add(side, n.pc[side], "assume", 0, s.args[2], guard))
		}
		return false
	}
	if !strings.HasPrefix(g.nodes[v].label, "phi:") {
		return g.setAlias(i, v)
	}
	if !g.once(guard) {
		return false
	}
	conds := g.edges[v]
	only := -1
	for j, a := range g.nodes[v].args {
		a = g.strip(a)
		if j < len(conds) && conds[j] >= 0 && g.contradicts(conds[j], guard) || a == v || a == only {
			continue
		}
		if only >= 0 {
			return false
		}
		only = a
	}
	if only < 0 {
		return false
	}
	return g.setAlias(i, g.add(side, n.pc[side], "assume", 0, only, guard))
}

// setAlias aliases node i to t unless t already resolves to i, and
// reports whether it did.
func (g *graph) setAlias(i, t int) bool {
	if g.resolve(t) == i {
		return false
	}
	g.nodes[i].alias = t
	return true
}

// add returns the node for label applied to args, creating it once.
func (g *graph) add(side int, pc uint64, label string, comm int, args ...int) int {
	var k strings.Builder
	k.WriteString(label)
	k.WriteByte(0)
	k.WriteString(strconv.Itoa(comm))
	for _, a := range args {
		k.WriteByte(',')
		k.WriteString(strconv.Itoa(a))
	}
	if id, ok := g.intern[k.String()]; ok {
		g.mark(id, side, pc)
		return id
	}
	id := len(g.nodes)
	g.nodes = append(g.nodes, node{label: label, args: slices.Clone(args), comm: comm, alias: -1})
	g.mark(id, side, pc)
	g.intern[k.String()] = id
	return id
}

// fresh returns a new node that is never shared, for a phi whose
// arguments are set later.
func (g *graph) fresh(side int, pc uint64, label string) int {
	id := len(g.nodes)
	g.nodes = append(g.nodes, node{label: label, alias: -1})
	g.mark(id, side, pc)
	return id
}

func (g *graph) mark(id, side int, pc uint64) {
	if n := &g.nodes[id]; !n.has[side] {
		n.pc[side], n.has[side] = pc, true
	}
}

// classes partitions the nodes by bisimulation: two nodes share a class
// when their labels match and their arguments, in order (or as a
// multiset, for commuting ones), share classes, in the greatest such
// partition. On acyclic graphs that is equality of expression trees; at
// loop-header phis it identifies loops that compute the same recurrence.
func (g *graph) classes() []int {
	for {
		class := g.refine()
		if !g.joinParts(class) {
			return class
		}
	}
}

// joinParts aliases each vec(part:0(a), part:1(b), ...) whose parts all
// come from one class to a, and reports whether it aliased any. Register
// 0 may instead be the value low records for a. Building
// the graph joins parts of one node; this catches parts of nodes that
// become equal only once trivial phis are removed.
func (g *graph) joinParts(class []int) bool {
	changed := false
	for i := range g.nodes {
		n := &g.nodes[i]
		if n.alias >= 0 || n.label != "vec" || len(n.args) < 2 {
			continue
		}
		last := g.nodes[g.resolve(n.args[len(n.args)-1])]
		if last.label != fmt.Sprintf("part:%d", len(n.args)-1) {
			continue
		}
		whole := last.args[0]
		for k, p := range n.args {
			p = g.resolve(p)
			pn := g.nodes[p]
			low, ok := g.low[whole]
			switch {
			case pn.label == fmt.Sprintf("part:%d", k) && class[pn.args[0]] == class[whole]:
			case k == 0 && ok && class[low] == class[p]:
			default:
				whole = -1
			}
			if whole < 0 {
				break
			}
		}
		if whole >= 0 {
			changed = g.setAlias(i, whole) || changed
		}
	}
	return changed
}

// refine returns the greatest bisimulation's classes after removing
// trivial phis.
func (g *graph) refine() []int {
	g.removeTrivialPhis()
	class := make([]int, len(g.nodes))
	ids := map[string]int{}
	for i, n := range g.nodes {
		k := fmt.Sprintf("%s\x00%d\x00%d", n.label, len(n.args), n.comm)
		c, ok := ids[k]
		if !ok {
			c = len(ids)
			ids[k] = c
		}
		class[i] = c
	}
	count := len(ids)
	sig := make([]int, 0, 8)
	for {
		next := make([]int, len(g.nodes))
		ids := map[string]int{}
		var k strings.Builder
		for i, n := range g.nodes {
			sig = sig[:0]
			for _, a := range n.args {
				sig = append(sig, class[a])
			}
			slices.Sort(sig[:n.comm])
			k.Reset()
			k.WriteString(strconv.Itoa(class[i]))
			for _, s := range sig {
				k.WriteByte(',')
				k.WriteString(strconv.Itoa(s))
			}
			c, ok := ids[k.String()]
			if !ok {
				c = len(ids)
				ids[k.String()] = c
			}
			next[i] = c
		}
		class = next
		if len(ids) == count {
			for i, n := range g.nodes {
				if n.alias >= 0 {
					class[i] = class[g.resolve(i)]
				}
			}
			return class
		}
		count = len(ids)
	}
}

// expr renders node id to a bounded depth, for reports.
func (g *graph) expr(id, depth int) string {
	n := g.nodes[id]
	if len(n.args) == 0 {
		return n.label
	}
	if depth == 0 {
		return n.label + "(...)"
	}
	var b strings.Builder
	b.WriteString(n.label)
	b.WriteByte('(')
	for i, a := range n.args {
		if i > 0 {
			b.WriteString(", ")
		}
		if strings.HasPrefix(g.nodes[a].label, "phi:") || strings.HasPrefix(g.nodes[a].label, "mem") {
			b.WriteString(g.nodes[a].label)
			continue
		}
		b.WriteString(g.expr(a, depth-1))
	}
	b.WriteByte(')')
	return b.String()
}

func (g *graph) resolve(id int) int {
	for g.nodes[id].alias >= 0 {
		id = g.nodes[id].alias
	}
	return id
}

// removeTrivialPhis replaces each phi whose incoming values, other than
// itself, are all one value with that value, until none is left: a value
// that is the same on every path into a block, such as a thread index
// kept in a register around a loop instead of read again inside it.
func (g *graph) removeTrivialPhis() {
	for again := true; again; {
		g.simplify()
		// An assume no edge resolves is the identity.
		again = false
		for i := range g.nodes {
			if n := g.nodes[i]; n.label == "assume" && n.alias < 0 {
				again = g.setAlias(i, n.args[0]) || again
			}
		}
	}
	for i := range g.nodes {
		for j, a := range g.nodes[i].args {
			g.nodes[i].args[j] = g.resolve(a)
		}
	}
}

// simplify aliases trivial phis, selections of one value and resolvable
// assumes until none remain.
func (g *graph) simplify() {
	for changed := true; changed; {
		changed = g.rehash()
		for i := range g.nodes {
			n := &g.nodes[i]
			if n.alias >= 0 {
				continue
			}
			switch {
			case strings.HasPrefix(n.label, "phi:"):
				// A phi of one value and itself is that value.
				only := -1
				trivial := true
				for _, a := range n.args {
					a = g.strip(a)
					if a == i || a == only {
						continue
					}
					if only >= 0 {
						trivial = false
						break
					}
					only = a
				}
				if trivial && only >= 0 {
					changed = g.setAlias(i, only) || changed
				}
			case n.label == "assume":
				changed = g.gate(i) || changed
			case n.label == "lin":
				changed = g.flatten(i) || changed
			case n.label == "sel" && len(n.args) == 3:
				if g.resolve(n.args[1]) == g.resolve(n.args[2]) {
					changed = g.setAlias(i, g.resolve(n.args[1])) || changed
				}
			}
		}
	}
}

// rehash aliases each node to an earlier one with its label and resolved
// arguments, as add would have had the arguments been known, and reports
// whether it aliased any. Phis are never shared.
func (g *graph) rehash() bool {
	changed := false
	seen := map[string]int{}
	var k strings.Builder
	for i := range g.nodes {
		n := &g.nodes[i]
		if n.alias >= 0 || strings.HasPrefix(n.label, "phi:") {
			continue
		}
		k.Reset()
		k.WriteString(n.label)
		k.WriteByte(0)
		k.WriteString(strconv.Itoa(n.comm))
		for _, a := range n.args {
			k.WriteByte(',')
			k.WriteString(strconv.Itoa(g.resolve(a)))
		}
		if j, ok := seen[k.String()]; ok && g.setAlias(i, j) {
			changed = true
			continue
		}
		seen[k.String()] = i
	}
	return changed
}

// flatten re-normalizes lin node i when a term's atom has become a lin
// or an immediate, as resolving a phi or assume can make it, and reports
// whether it aliased the node.
func (g *graph) flatten(i int) bool {
	n := g.nodes[i]
	need := false
	for _, t := range n.args[:n.comm] {
		a := g.resolve(g.nodes[g.resolve(t)].args[1])
		need = need || g.nodes[a].label == "lin" || isImm(g, a)
	}
	if !need {
		return false
	}
	terms := map[int]uint32{}
	k := uint32(0)
	var add func(lin int, c uint32)
	add = func(lin int, c uint32) {
		ln := g.nodes[lin]
		k += c * g.immVal(ln.args[len(ln.args)-1])
		for _, t := range ln.args[:ln.comm] {
			tn := g.nodes[g.resolve(t)]
			a, tc := g.resolve(tn.args[1]), c*g.immVal(tn.args[0])
			switch {
			case isImm(g, a):
				k += tc * g.immVal(a)
			case g.nodes[a].label == "lin":
				add(a, tc)
			default:
				terms[a] += tc
			}
		}
	}
	add(i, 1)
	side := 0
	if !n.has[0] {
		side = 1
	}
	pc := n.pc[side]
	imm := func(v uint32) int { return g.add(side, pc, fmt.Sprintf("imm:%#x", v), 0) }
	var atoms []int
	for a, c := range terms {
		if c != 0 {
			atoms = append(atoms, a)
		}
	}
	slices.Sort(atoms)
	switch {
	case len(atoms) == 0:
		return g.setAlias(i, imm(k))
	case len(atoms) == 1 && k == 0 && terms[atoms[0]] == 1:
		return g.setAlias(i, atoms[0])
	}
	var args []int
	for _, a := range atoms {
		args = append(args, g.add(side, pc, "term", 0, imm(terms[a]), a))
	}
	return g.setAlias(i, g.add(side, pc, "lin", len(args), append(args, imm(k))...))
}

func (g *graph) immVal(v int) uint32 {
	u, _ := strconv.ParseUint(strings.TrimPrefix(g.nodes[g.resolve(v)].label, "imm:0x"), 16, 64)
	return uint32(u)
}

// strip returns the value a stands for, seeing through assumes: assume(v,
// guard) is v wherever its value is used.
func (g *graph) strip(a int) int {
	a = g.resolve(a)
	for g.nodes[a].label == "assume" {
		a = g.resolve(g.nodes[a].args[0])
	}
	return a
}

// undefined reports whether a node is a register or stack slot read
// before the function writes it.
func (g *graph) undefined(id int) bool {
	return strings.HasPrefix(g.nodes[id].label, "undef")
}

// mergesUndefined reports whether a node is undefined or selects or
// merges an undefined value with others.
func (g *graph) mergesUndefined(id int) bool {
	n := g.nodes[id]
	if n.label != "sel" && !strings.HasPrefix(n.label, "phi:") {
		return g.undefined(id)
	}
	for _, a := range n.args {
		if g.undefined(g.strip(a)) {
			return true
		}
	}
	return false
}

// sharedUndefined returns the registers whose undefined value is
// reachable from the roots of both sides. An undefined value is one
// symbol for both, so equal roots may rest on it. Every condition a phi
// merges under is a branch or exit condition, itself a root.
func (g *graph) sharedUndefined(a, b map[string]int) []string {
	reach := func(roots map[string]int) map[int]bool {
		seen := map[int]bool{}
		var walk func(int)
		walk = func(v int) {
			if seen[v] {
				return
			}
			seen[v] = true
			for _, w := range g.nodes[v].args {
				walk(w)
			}
		}
		for _, r := range roots {
			walk(r)
		}
		return seen
	}
	ra, rb := reach(a), reach(b)
	var regs []string
	for v := range ra {
		if rb[v] && g.undefined(v) {
			regs = append(regs, strings.TrimPrefix(g.nodes[v].label, "undef:"))
		}
	}
	slices.SortFunc(regs, func(x, y string) int { return cmpRegister(x, y) })
	return regs
}

// cmpRegister orders register names by file, then number: R2 before R10.
func cmpRegister(x, y string) int {
	split := func(s string) (string, int) {
		i := strings.IndexFunc(s, func(r rune) bool { return r >= '0' && r <= '9' })
		if i < 0 {
			return s, -1
		}
		n, _ := strconv.Atoi(s[i:])
		return s[:i], n
	}
	fx, nx := split(x)
	fy, ny := split(y)
	if c := strings.Compare(fx, fy); c != 0 {
		return c
	}
	return nx - ny
}
