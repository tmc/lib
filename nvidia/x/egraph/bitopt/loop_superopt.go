package bitopt

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
)

// A DAGOp is the operation of one [DAGNode]. Every value is a 32-bit word;
// floating-point operations read and write float32 bit patterns.
type DAGOp uint8

const (
	DAGConst     DAGOp = iota // Imm; an immediate operand, not an instruction
	DAGAnd                    // and.b32 A, B
	DAGOr                     // or.b32 A, B
	DAGXor                    // xor.b32 A, B
	DAGAdd                    // add.u32 A, B
	DAGShl                    // shl.b32 A, Imm
	DAGShr                    // shr.u32 A, Imm
	DAGPrmt                   // prmt.b32 A, B, Imm
	DAGFMul                   // mul.rn.f32 A, B
	DAGFAdd                   // add.rn.f32 A, B
	DAGFMax                   // max.f32 A, B
	DAGCvtE2M1x2              // cvt.rn.satfinite.e2m1x2.f32 A (upper nibble), B (lower nibble), zero-extended
	numDAGOps
)

var dagOpNames = [...]string{"const", "and", "or", "xor", "add", "shl", "shr", "prmt", "fmul", "fadd", "fmax", "cvt.e2m1x2"}

func (op DAGOp) String() string {
	if int(op) < len(dagOpNames) {
		return dagOpNames[op]
	}
	return fmt.Sprintf("DAGOp(%d)", op)
}

// arity returns the number of value operands op reads.
func (op DAGOp) arity() int {
	switch op {
	case DAGConst:
		return 0
	case DAGShl, DAGShr:
		return 1
	}
	return 2
}

func (op DAGOp) commutative() bool {
	switch op {
	case DAGAnd, DAGOr, DAGXor, DAGAdd, DAGFMul, DAGFAdd, DAGFMax:
		return true
	}
	return false
}

// A DAGNode is one operation. A and B index values: 0 through NumInputs-1
// are the inputs, and NumInputs+i is node i.
type DAGNode struct {
	Op   DAGOp
	A, B int
	Imm  uint32
}

// An EpilogueDAG is a straight-line, multi-output computation over 32-bit
// words, such as the epilogue of a GEMM tile between the accumulator
// readback and the store.
type EpilogueDAG struct {
	NumInputs int
	Nodes     []DAGNode
	Outputs   []int // value indices

	// LoadGroups lists inputs that are loaded together, such as the eight
	// registers of one tcgen05.ld .x8. A grouped input is live from its
	// group's load; an ungrouped input is live from entry.
	LoadGroups [][]int
}

// Validate reports whether every operand refers to an earlier value.
func (d EpilogueDAG) Validate() error {
	if d.NumInputs < 0 {
		return errors.New("bitopt: negative input count")
	}
	for i, n := range d.Nodes {
		if n.Op >= numDAGOps {
			return fmt.Errorf("bitopt: node %d: unknown op %d", i, n.Op)
		}
		for k, a := range []int{n.A, n.B}[:n.Op.arity()] {
			if a < 0 || a >= d.NumInputs+i {
				return fmt.Errorf("bitopt: node %d operand %d refers to value %d, not an earlier one", i, k, a)
			}
		}
		if (n.Op == DAGShl || n.Op == DAGShr) && n.Imm > 31 {
			return fmt.Errorf("bitopt: node %d: shift %d", i, n.Imm)
		}
	}
	for _, o := range d.Outputs {
		if o < 0 || o >= d.NumInputs+len(d.Nodes) {
			return fmt.Errorf("bitopt: output %d out of range", o)
		}
	}
	seen := map[int]bool{}
	for _, g := range d.LoadGroups {
		for _, in := range g {
			if in < 0 || in >= d.NumInputs || seen[in] {
				return fmt.Errorf("bitopt: load group input %d out of range or repeated", in)
			}
			seen[in] = true
		}
	}
	return nil
}

// Insts returns the number of instructions the DAG takes: its nodes other
// than constants.
func (d EpilogueDAG) Insts() int {
	n := 0
	for _, x := range d.Nodes {
		if x.Op != DAGConst {
			n++
		}
	}
	return n
}

// canonicalNaN is the NaN PTX's f32 arithmetic returns.
const canonicalNaN = 0x7fffffff

func f32(x float32) uint32 {
	if x != x {
		return canonicalNaN
	}
	return math.Float32bits(x)
}

// e2m1 returns the 4-bit E2M1 code nearest x, ties to even, saturating to
// ±6. NaN, which E2M1 cannot represent, is modelled as +6; the hardware's
// choice is not established here.
func e2m1(x float32) uint32 {
	if x != x {
		return 0x7
	}
	sign := math.Float32bits(x) >> 31 << 3
	a := math.Abs(float64(x))
	var code uint32
	switch {
	case a <= 0.25:
		code = 0
	case a < 0.75:
		code = 1
	case a <= 1.25:
		code = 2
	case a < 1.75:
		code = 3
	case a <= 2.5:
		code = 4
	case a < 3.5:
		code = 5
	case a <= 5:
		code = 6
	default:
		code = 7
	}
	return sign | code
}

func eval1(op DAGOp, a, b, imm uint32) uint32 {
	fa, fb := math.Float32frombits(a), math.Float32frombits(b)
	switch op {
	case DAGConst:
		return imm
	case DAGAnd:
		return a & b
	case DAGOr:
		return a | b
	case DAGXor:
		return a ^ b
	case DAGAdd:
		return a + b
	case DAGShl:
		return a << imm
	case DAGShr:
		return a >> imm
	case DAGPrmt:
		x := uint64(b)<<32 | uint64(a)
		var r uint32
		for j := range 4 {
			s := imm >> (4 * j) & 7
			r |= uint32(x>>(8*s)&0xff) << (8 * j)
		}
		return r
	case DAGFMul:
		return f32(fa * fb)
	case DAGFAdd:
		return f32(fa + fb)
	case DAGFMax:
		switch {
		case fa != fa && fb != fb:
			return canonicalNaN
		case fa != fa:
			return f32(fb)
		case fb != fb:
			return f32(fa)
		case fa == 0 && fb == 0:
			return a & b // +0 unless both are -0
		case fa > fb:
			return a
		}
		return b
	case DAGCvtE2M1x2:
		return e2m1(fa)<<4 | e2m1(fb)
	}
	panic("bitopt: bad DAG op")
}

// Eval returns the DAG's outputs for inputs in.
func (d EpilogueDAG) Eval(in []uint32) []uint32 {
	v := d.values(in)
	out := make([]uint32, len(d.Outputs))
	for i, o := range d.Outputs {
		out[i] = v[o]
	}
	return out
}

func (d EpilogueDAG) values(in []uint32) []uint32 {
	v := make([]uint32, d.NumInputs+len(d.Nodes))
	copy(v, in)
	for i, n := range d.Nodes {
		v[d.NumInputs+i] = eval1(n.Op, v[n.A], v[n.B], n.Imm)
	}
	return v
}

// floatBoundary are the inputs every equivalence check includes: signed
// zeros, the E2M1 rounding ties and saturation points, the float32 limits,
// infinities, a NaN and a subnormal.
var floatBoundary = []float32{
	0, float32(math.Copysign(0, -1)), 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7,
	-0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, -6, -7,
	math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32,
	float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
}

// TestVectors returns the input vectors the equivalence checks use: one
// vector per boundary value with every input set to it, one per boundary
// value and input position with the others random, then n random vectors.
// Random inputs are, in turn, float32 values in [-8, 8), float32 values
// scaled by a random power of two, and raw bit patterns.
//
// Agreement on these vectors is evidence, not proof: the DAGs mix integer
// and floating-point operations, for which this package has no symbolic
// equivalence check.
func TestVectors(numInputs, n int, seed uint64) [][]uint32 {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	random := func(k int) uint32 {
		switch k % 3 {
		case 0:
			return math.Float32bits(r.Float32()*16 - 8)
		case 1:
			return math.Float32bits(float32(math.Ldexp(r.Float64()*2-1, r.IntN(40)-20)))
		}
		return r.Uint32()
	}
	var out [][]uint32
	for _, b := range floatBoundary {
		v := make([]uint32, numInputs)
		for i := range v {
			v[i] = math.Float32bits(b)
		}
		out = append(out, v)
	}
	for _, b := range floatBoundary {
		for p := range numInputs {
			v := make([]uint32, numInputs)
			for i := range v {
				v[i] = random(i + p)
			}
			v[p] = math.Float32bits(b)
			out = append(out, v)
		}
	}
	for k := range n {
		v := make([]uint32, numInputs)
		for i := range v {
			v[i] = random(i + k)
		}
		out = append(out, v)
	}
	return out
}

// CheckEquivalent reports the first of vectors on which a and b produce
// different outputs.
func CheckEquivalent(a, b EpilogueDAG, vectors [][]uint32) error {
	if a.NumInputs != b.NumInputs || len(a.Outputs) != len(b.Outputs) {
		return fmt.Errorf("bitopt: DAG shapes differ: %d→%d and %d→%d", a.NumInputs, len(a.Outputs), b.NumInputs, len(b.Outputs))
	}
	for _, v := range vectors {
		ga, gb := a.Eval(v), b.Eval(v)
		if !slices.Equal(ga, gb) {
			return fmt.Errorf("bitopt: DAGs differ on input %#x: %#x vs %#x", v, ga, gb)
		}
	}
	return nil
}

// PeakLive returns the most values live at once when the DAG runs in node
// order with every input live from entry, as a single wide load would make
// them. A value is live from its definition to its last use; outputs stay
// live to the end; constants are immediates and never live. A node's
// destination may reuse a source register that dies at it.
func (d EpilogueDAG) PeakLive() int {
	order := make([]int, 0, len(d.Nodes)+1)
	order = append(order, -1) // load everything
	for i := range d.Nodes {
		order = append(order, i)
	}
	all := make([]int, d.NumInputs)
	for i := range all {
		all[i] = i
	}
	return d.peak(order, [][]int{all})
}

// peak returns the peak live count of a schedule: a list of node indices,
// and -(g+1) for the load of groups[g].
func (d EpilogueDAG) peak(order []int, groups [][]int) int {
	remaining := d.useCounts()
	isOut := map[int]bool{}
	for _, o := range d.Outputs {
		isOut[o] = true
	}
	live := map[int]bool{}
	grouped := map[int]bool{}
	for _, g := range groups {
		for _, in := range g {
			grouped[in] = true
		}
	}
	for i := range d.NumInputs {
		if !grouped[i] && (remaining[i] > 0 || isOut[i]) {
			live[i] = true
		}
	}
	peak := len(live)
	for _, s := range order {
		if s < 0 {
			for _, in := range groups[-s-1] {
				if remaining[in] > 0 || isOut[in] {
					live[in] = true
				}
			}
			peak = max(peak, len(live))
			continue
		}
		n := d.Nodes[s]
		if n.Op == DAGConst {
			continue
		}
		for _, a := range uniqueArgs(n) {
			if remaining[a]--; remaining[a] == 0 && !isOut[a] {
				delete(live, a)
			}
		}
		v := d.NumInputs + s
		if remaining[v] > 0 || isOut[v] {
			live[v] = true
		}
		peak = max(peak, len(live))
	}
	return peak
}

// uniqueArgs returns the distinct non-constant value operands of n. A
// constant operand is an immediate and is filtered by the caller's use
// counts, which do not count it.
func uniqueArgs(n DAGNode) []int {
	args := []int{n.A, n.B}[:n.Op.arity()]
	if len(args) == 2 && args[0] == args[1] {
		args = args[:1]
	}
	return args
}

// useCounts returns how many nodes read each value, counting a node once
// per distinct operand.
func (d EpilogueDAG) useCounts() []int {
	c := make([]int, d.NumInputs+len(d.Nodes))
	for _, n := range d.Nodes {
		if n.Op == DAGConst {
			continue
		}
		for _, a := range uniqueArgs(n) {
			c[a]++
		}
	}
	return c
}

// A SuperoptResult is the output of [SynthesizeEpilogueDAG].
type SuperoptResult struct {
	OriginalInsts    int
	SuperoptInsts    int
	OriginalPeakLive int // target.PeakLive()
	PeakLiveRegs     int // the result's peak under its schedule and load groups
	DAG              EpilogueDAG
	Schedule         []int // node indices of DAG, and -(g+1) for the load of LoadGroups[g]
	PTX              string
}

// SynthesizeEpilogueDAG returns a DAG equivalent to target with fewer
// instructions and a schedule with fewer live registers.
//
// It works in four steps, each checked:
//
//   - Canonicalization: common subexpressions are merged, unused nodes
//     dropped, and maximal chains of one associative, commutative integer
//     operation (and, or, xor, add) whose inner results have no other use
//     are rebuilt as balanced trees with their leaves in the original
//     order. The rebuilt trees expose pairs of neighbouring terms to the
//     next step.
//   - Window search: for every connected group of up to five nodes whose
//     only result used elsewhere is its root and which reads at most three
//     outside values, every program of at most maxNodes nodes (1 or 2) over
//     those values and the DAG's constants is enumerated, smallest first,
//     and the first one that matches the root on every sampled input
//     replaces the window if it is smaller. prmt is tried only as a
//     single node. Repeated to a fixed point.
//   - Each accepted replacement is checked on the whole DAG against the
//     original over 1024 vectors, and the result over
//     [TestVectors](NumInputs, 10000, 1). A replacement that fails is
//     undone.
//   - Scheduling: nodes and load groups are list-scheduled greedily,
//     choosing at each step the ready node that adds the fewest live
//     registers, preferring a node over a load, and a load only when no
//     node is ready.
//
// The search is bounded, not exhaustive over all programs, and the
// equivalence check is testing on boundary and random vectors, not a
// proof. The PTX is emitted, not assembled here; see [EpilogueDAG.PTX].
func SynthesizeEpilogueDAG(target EpilogueDAG, maxNodes int) (SuperoptResult, error) {
	if err := target.Validate(); err != nil {
		return SuperoptResult{}, err
	}
	if maxNodes < 1 || maxNodes > 2 {
		return SuperoptResult{}, fmt.Errorf("bitopt: maxNodes %d, want 1 or 2", maxNodes)
	}
	check := TestVectors(target.NumInputs, 1024, 2)
	d := canonicalize(target)
	if err := CheckEquivalent(target, d, check); err != nil {
		return SuperoptResult{}, fmt.Errorf("bitopt: internal error: canonicalization: %w", err)
	}
	samples := searchSamples(target.NumInputs, 128, 3)
	failed := map[uint64]bool{} // roots, by structure, with no smaller program
	for {
		keys := structureKeys(d)
		applied := false
		for r := range d.Nodes {
			k := keys[d.NumInputs+r]
			if failed[k] {
				continue
			}
			nd, ok := rewriteWindow(d, r, maxNodes, samples)
			if ok {
				// Rebalancing is not repeated: it would regroup chains the
				// window search has already paired.
				nd = dce(cse(nd))
				ok = nd.Insts() < d.Insts() && CheckEquivalent(target, nd, check) == nil
			}
			if !ok {
				failed[k] = true
				continue
			}
			d, applied = nd, true
			break
		}
		if !applied {
			break
		}
	}
	if err := CheckEquivalent(target, d, TestVectors(target.NumInputs, 10000, 1)); err != nil {
		return SuperoptResult{}, fmt.Errorf("bitopt: result fails the final check: %w", err)
	}
	sched := schedule(d)
	res := SuperoptResult{
		OriginalInsts:    target.Insts(),
		SuperoptInsts:    d.Insts(),
		OriginalPeakLive: target.PeakLive(),
		PeakLiveRegs:     d.peak(sched, d.LoadGroups),
		DAG:              d,
		Schedule:         sched,
	}
	ptx, err := d.ptx("epilogue", sched)
	if err != nil {
		return SuperoptResult{}, err
	}
	res.PTX = ptx
	return res, nil
}

// canonicalize merges common subexpressions, rebalances associative
// chains and drops dead nodes, returning a DAG in dependency order.
func canonicalize(d EpilogueDAG) EpilogueDAG {
	d = cse(d)
	d = rebalance(d)
	d = cse(d)
	return dce(d)
}

// cse merges nodes with the same operation and operands.
func cse(d EpilogueDAG) EpilogueDAG {
	repl := make([]int, d.NumInputs+len(d.Nodes))
	for i := range d.NumInputs {
		repl[i] = i
	}
	out := EpilogueDAG{NumInputs: d.NumInputs, LoadGroups: d.LoadGroups}
	seen := map[DAGNode]int{}
	for i, n := range d.Nodes {
		if n.Op.arity() > 0 {
			n.A = repl[n.A]
		}
		if n.Op.arity() > 1 {
			n.B = repl[n.B]
		}
		if n.Op.arity() < 2 {
			n.B = 0
		}
		if n.Op.arity() < 1 {
			n.A = 0
		}
		if n.Op.commutative() && n.A > n.B {
			n.A, n.B = n.B, n.A
		}
		if !usesImm(n.Op) {
			n.Imm = 0
		}
		if v, ok := seen[n]; ok {
			repl[d.NumInputs+i] = v
			continue
		}
		v := d.NumInputs + len(out.Nodes)
		seen[n] = v
		repl[d.NumInputs+i] = v
		out.Nodes = append(out.Nodes, n)
	}
	for _, o := range d.Outputs {
		out.Outputs = append(out.Outputs, repl[o])
	}
	return out
}

func usesImm(op DAGOp) bool {
	switch op {
	case DAGConst, DAGShl, DAGShr, DAGPrmt:
		return true
	}
	return false
}

// dce drops nodes no output depends on.
func dce(d EpilogueDAG) EpilogueDAG {
	need := make([]bool, d.NumInputs+len(d.Nodes))
	for _, o := range d.Outputs {
		need[o] = true
	}
	for i := len(d.Nodes) - 1; i >= 0; i-- {
		if !need[d.NumInputs+i] {
			continue
		}
		n := d.Nodes[i]
		for _, a := range []int{n.A, n.B}[:n.Op.arity()] {
			need[a] = true
		}
	}
	repl := make([]int, len(need))
	for i := range d.NumInputs {
		repl[i] = i
	}
	out := EpilogueDAG{NumInputs: d.NumInputs, LoadGroups: d.LoadGroups}
	for i, n := range d.Nodes {
		if !need[d.NumInputs+i] {
			continue
		}
		n.A, n.B = repl[n.A], repl[n.B]
		repl[d.NumInputs+i] = d.NumInputs + len(out.Nodes)
		out.Nodes = append(out.Nodes, n)
	}
	for _, o := range d.Outputs {
		out.Outputs = append(out.Outputs, repl[o])
	}
	return out
}

// rebalance rebuilds maximal single-use chains of one associative,
// commutative integer operation as balanced trees over the same leaves in
// left-to-right order.
func rebalance(d EpilogueDAG) EpilogueDAG {
	uses := d.useCounts()
	isOut := map[int]bool{}
	for _, o := range d.Outputs {
		isOut[o] = true
	}
	n := d.NumInputs
	// interior[v] is true for a chain node absorbed into its user's tree.
	interior := make([]bool, n+len(d.Nodes))
	assoc := func(op DAGOp) bool { return op == DAGAnd || op == DAGOr || op == DAGXor || op == DAGAdd }
	for _, x := range d.Nodes {
		if !assoc(x.Op) {
			continue
		}
		for _, a := range []int{x.A, x.B} {
			if a >= n && d.Nodes[a-n].Op == x.Op && uses[a] == 1 && !isOut[a] {
				interior[a] = true
			}
		}
	}
	out := EpilogueDAG{NumInputs: n, LoadGroups: d.LoadGroups}
	repl := make([]int, n+len(d.Nodes))
	for i := range n {
		repl[i] = i
	}
	var leaves func(v int, op DAGOp) []int
	leaves = func(v int, op DAGOp) []int {
		if v >= n && interior[v] && d.Nodes[v-n].Op == op {
			x := d.Nodes[v-n]
			return append(leaves(x.A, op), leaves(x.B, op)...)
		}
		return []int{repl[v]}
	}
	emit := func(x DAGNode) int {
		out.Nodes = append(out.Nodes, x)
		return n + len(out.Nodes) - 1
	}
	var build func(op DAGOp, ls []int) int
	build = func(op DAGOp, ls []int) int {
		if len(ls) == 1 {
			return ls[0]
		}
		h := (len(ls) + 1) / 2
		// Round the left half up to a power of two, so neighbouring pairs
		// stay together.
		for p := 1; p < len(ls); p *= 2 {
			if p >= h {
				h = p
				break
			}
		}
		if h >= len(ls) {
			h = len(ls) / 2
		}
		return emit(DAGNode{Op: op, A: build(op, ls[:h]), B: build(op, ls[h:])})
	}
	for i, x := range d.Nodes {
		v := n + i
		if interior[v] {
			continue // built as part of its user's tree
		}
		if assoc(x.Op) {
			repl[v] = build(x.Op, append(leaves(x.A, x.Op), leaves(x.B, x.Op)...))
			continue
		}
		if x.Op.arity() > 0 {
			x.A = repl[x.A]
		}
		if x.Op.arity() > 1 {
			x.B = repl[x.B]
		}
		repl[v] = emit(x)
	}
	for _, o := range d.Outputs {
		out.Outputs = append(out.Outputs, repl[o])
	}
	return out
}

// window is a set of nodes, identified by index, with one root.
type window struct {
	nodes []int // node indices, ascending
	ext   []int // outside values read, ascending
}

// windows returns the windows rooted at node r of up to five nodes and at
// most three outside non-constant values.
func windows(d EpilogueDAG, r int) []window {
	uses := d.useCounts()
	isOut := map[int]bool{}
	for _, o := range d.Outputs {
		isOut[o] = true
	}
	n := d.NumInputs
	absorbable := func(v int) bool {
		return v >= n && d.Nodes[v-n].Op != DAGConst && uses[v] == 1 && !isOut[v]
	}
	var out []window
	seen := map[string]bool{}
	var grow func(set []int)
	grow = func(set []int) {
		key := fmt.Sprint(set)
		if seen[key] {
			return
		}
		seen[key] = true
		in := map[int]bool{}
		for _, s := range set {
			in[n+s] = true
		}
		var frontier []int
		extSet := map[int]bool{}
		for _, s := range set {
			x := d.Nodes[s]
			for _, a := range []int{x.A, x.B}[:x.Op.arity()] {
				if in[a] {
					continue
				}
				if a >= n && d.Nodes[a-n].Op == DAGConst {
					continue
				}
				extSet[a] = true
				if absorbable(a) {
					frontier = append(frontier, a-n)
				}
			}
		}
		if len(set) > 1 && len(extSet) <= 3 {
			w := window{nodes: slices.Sorted(slices.Values(set))}
			for v := range extSet {
				w.ext = append(w.ext, v)
			}
			slices.Sort(w.ext)
			out = append(out, w)
		}
		if len(set) == 5 {
			return
		}
		for _, f := range frontier {
			grow(slices.Sorted(slices.Values(append(slices.Clone(set), f))))
		}
	}
	grow([]int{r})
	return out
}

// rewriteWindow tries to replace a window rooted at r with a smaller
// program, and returns the rewritten DAG.
func rewriteWindow(d EpilogueDAG, r, maxNodes int, samples [][]uint32) (EpilogueDAG, bool) {
	if d.Nodes[r].Op == DAGConst {
		return d, false
	}
	vals := make([][]uint32, len(samples))
	for i, s := range samples {
		vals[i] = d.values(s)
	}
	var consts []uint32
	for _, x := range d.Nodes {
		if x.Op == DAGConst && !slices.Contains(consts, x.Imm) {
			consts = append(consts, x.Imm)
		}
	}
	if !slices.Contains(consts, 0) {
		consts = append(consts, 0)
	}
	root := d.NumInputs + r
	var best []candidate
	var bestExt []int
	bestSaving := 0
	for _, w := range windows(d, r) {
		cost := 0
		for _, s := range w.nodes {
			if d.Nodes[s].Op != DAGConst {
				cost++
			}
		}
		for size := 1; size <= maxNodes && cost-size > bestSaving; size++ {
			if prog, ok := search(w.ext, consts, size, root, vals); ok {
				best, bestExt, bestSaving = prog, w.ext, cost-size
				break
			}
		}
	}
	if best == nil {
		return d, false
	}
	return splice(d, r, bestExt, best, consts), true
}

// structureKeys returns a hash of each value's expression tree together
// with the fan-out of the nodes near its root, which is what decides the
// windows rooted there.
func structureKeys(d EpilogueDAG) []uint64 {
	uses := d.useCounts()
	h := make([]uint64, d.NumInputs+len(d.Nodes))
	mix := func(x, y uint64) uint64 { return (x ^ y) * 0x100000001b3 }
	for i := range d.NumInputs {
		h[i] = mix(0xcbf29ce484222325, uint64(i)+1)
	}
	for i, x := range d.Nodes {
		k := mix(0xcbf29ce484222325, uint64(x.Op)<<32|uint64(x.Imm))
		for _, a := range []int{x.A, x.B}[:x.Op.arity()] {
			k = mix(k, h[a])
			k = mix(k, uint64(uses[a]))
		}
		h[d.NumInputs+i] = k
	}
	return h
}

// searchSamples returns the vectors the window search matches on: every
// input set to one boundary value, then n vectors in which each input is a
// boundary value one time in four and otherwise random as in
// [TestVectors].
func searchSamples(numInputs, n int, seed uint64) [][]uint32 {
	out := TestVectors(numInputs, 0, seed)[:len(floatBoundary)]
	r := rand.New(rand.NewPCG(seed, ^seed))
	for _, v := range TestVectors(numInputs, n, seed)[len(floatBoundary)*(numInputs+1):] {
		for i := range v {
			if r.IntN(4) == 0 {
				v[i] = math.Float32bits(floatBoundary[r.IntN(len(floatBoundary))])
			}
		}
		out = append(out, v)
	}
	return out
}

// A candidate node refers to operands by position: 0..len(ext)-1 are the
// window's outside values, len(ext)..+len(consts)-1 are constants, and the
// next index is the first candidate node's result.
type candidate struct {
	op   DAGOp
	a, b int
	imm  uint32
}

// search enumerates programs of size nodes over ext and consts and returns
// the first whose last node matches value root on every sample.
func search(ext []int, consts []uint32, size, root int, vals [][]uint32) ([]candidate, bool) {
	ops := len(ext) + len(consts)
	operand := func(s, k int, prev []uint32) uint32 {
		switch {
		case k < len(ext):
			return vals[s][ext[k]]
		case k < ops:
			return consts[k-len(ext)]
		}
		return prev[s]
	}
	matches := func(c candidate, prev []uint32) bool {
		for s := range vals {
			if eval1(c.op, operand(s, c.a, prev), operand(s, c.b, prev), c.imm) != vals[s][root] {
				return false
			}
		}
		return true
	}
	first := enumerate(ops, -1, size == 1)
	if size == 1 {
		for _, c := range first {
			if matches(c, nil) {
				return []candidate{c}, true
			}
		}
		return nil, false
	}
	for _, c1 := range first {
		prev := make([]uint32, len(vals))
		for s := range vals {
			prev[s] = eval1(c1.op, operand(s, c1.a, nil), operand(s, c1.b, nil), c1.imm)
		}
		for _, c2 := range enumerate(ops+1, ops, false) {
			if matches(c2, prev) {
				return []candidate{c1, c2}, true
			}
		}
	}
	return nil, false
}

// enumerate returns the single nodes over operand positions 0..n-1. If
// must is not negative, every candidate reads position must. prmt is
// included only when withPrmt is set.
func enumerate(n, must int, withPrmt bool) []candidate {
	var out []candidate
	reads := func(c candidate) bool {
		return must < 0 || c.a == must || c.op.arity() > 1 && c.b == must
	}
	for op := DAGAnd; op < numDAGOps; op++ {
		switch op {
		case DAGShl, DAGShr:
			for a := range n {
				for s := uint32(1); s < 32; s++ {
					if c := (candidate{op: op, a: a, imm: s}); reads(c) {
						out = append(out, c)
					}
				}
			}
		case DAGPrmt:
			if !withPrmt {
				continue
			}
			for a := range n {
				for b := range n {
					if a == b {
						continue
					}
					for sel := uint32(0); sel < 1<<12; sel++ {
						imm := sel&7 | sel>>3&7<<4 | sel>>6&7<<8 | sel>>9&7<<12
						if c := (candidate{op: op, a: a, b: b, imm: imm}); reads(c) {
							out = append(out, c)
						}
					}
				}
			}
		default:
			for a := range n {
				for b := range n {
					if op.commutative() && b < a {
						continue
					}
					if c := (candidate{op: op, a: a, b: b}); reads(c) {
						out = append(out, c)
					}
				}
			}
		}
	}
	return out
}

// splice replaces node r of d with prog, whose operand positions refer to
// ext, then consts, then earlier program nodes. The window's other nodes
// become dead and are dropped by the caller's canonicalization.
func splice(d EpilogueDAG, r int, ext []int, prog []candidate, consts []uint32) EpilogueDAG {
	n := d.NumInputs
	out := EpilogueDAG{NumInputs: n, LoadGroups: d.LoadGroups}
	repl := make([]int, n+len(d.Nodes))
	for i := range n {
		repl[i] = i
	}
	emit := func(x DAGNode) int {
		out.Nodes = append(out.Nodes, x)
		return n + len(out.Nodes) - 1
	}
	remap := func(x DAGNode) DAGNode {
		if x.Op.arity() > 0 {
			x.A = repl[x.A]
		}
		if x.Op.arity() > 1 {
			x.B = repl[x.B]
		}
		return x
	}
	for i, x := range d.Nodes {
		if i != r {
			repl[n+i] = emit(remap(x))
			continue
		}
		var pv []int // values of program operand positions
		for _, e := range ext {
			pv = append(pv, repl[e])
		}
		for _, c := range consts {
			pv = append(pv, emit(DAGNode{Op: DAGConst, Imm: c}))
		}
		for _, c := range prog {
			x := DAGNode{Op: c.op, A: pv[c.a], Imm: c.imm}
			if c.op.arity() > 1 {
				x.B = pv[c.b]
			}
			pv = append(pv, emit(x))
		}
		repl[n+i] = pv[len(pv)-1]
	}
	for _, o := range d.Outputs {
		out.Outputs = append(out.Outputs, repl[o])
	}
	return out
}

// schedule list-schedules d's nodes and load groups; see
// [SynthesizeEpilogueDAG].
func schedule(d EpilogueDAG) []int {
	n := d.NumInputs
	remaining := d.useCounts()
	isOut := map[int]bool{}
	for _, o := range d.Outputs {
		isOut[o] = true
	}
	avail := make([]bool, n+len(d.Nodes))
	for i := range n {
		avail[i] = true
	}
	for _, g := range d.LoadGroups {
		for _, in := range g {
			avail[in] = false
		}
	}
	done := make([]bool, len(d.Nodes))
	left := 0
	for i, x := range d.Nodes {
		if x.Op == DAGConst {
			avail[n+i], done[i] = true, true
			continue
		}
		left++
	}
	loaded := make([]bool, len(d.LoadGroups))
	var order []int
	for left > 0 {
		best, bestDelta := -1, 0
		for i, x := range d.Nodes {
			if done[i] {
				continue
			}
			args := uniqueArgs(x)
			ready := true
			for _, a := range args {
				ready = ready && avail[a]
			}
			if !ready {
				continue
			}
			delta := 1
			for _, a := range args {
				if remaining[a] == 1 && !isOut[a] {
					delta--
				}
			}
			if best < 0 || delta < bestDelta {
				best, bestDelta = i, delta
			}
		}
		if best >= 0 {
			for _, a := range uniqueArgs(d.Nodes[best]) {
				remaining[a]--
			}
			avail[n+best], done[best] = true, true
			order = append(order, best)
			left--
			continue
		}
		// No node is ready: load the group the earliest pending node needs.
		g := -1
		for i := range d.Nodes {
			if done[i] {
				continue
			}
			for _, a := range uniqueArgs(d.Nodes[i]) {
				for k, grp := range d.LoadGroups {
					if !loaded[k] && slices.Contains(grp, a) {
						g = k
					}
				}
			}
			if g >= 0 {
				break
			}
		}
		if g < 0 {
			panic("bitopt: schedule stuck")
		}
		loaded[g] = true
		for _, in := range d.LoadGroups[g] {
			avail[in] = true
		}
		order = append(order, -(g + 1))
	}
	return order
}

// PTX returns the DAG as a PTX module for sm_100a with one entry, name,
// that runs schedule, as returned in [SuperoptResult.Schedule].
//
// The entry takes the output pointer, the TMEM address of the first load
// group and one .u32 parameter per ungrouped input, in input order. Value v
// lives in %v<v>. Each load group, of 1, 2, 4, ..., 128 registers, is a
// tcgen05.ld.sync.aligned.32x32b from the TMEM address plus the group's
// starting column, the groups occupying consecutive columns in order, and
// is followed by tcgen05.wait::ld.sync.aligned; [CoalesceWaits] can then
// sink and merge the waits. Constants are materialized with mov.b32, and
// output i is stored to out[i]. Each thread computes one row; the TMEM
// lane is the caller's.
//
// The text is checked by tests for structure and wait placement only; it
// has been assembled by ptxas only if the env-gated native test was run.
func (d EpilogueDAG) PTX(name string, schedule []int) (string, error) {
	return d.ptx(name, schedule)
}

func (d EpilogueDAG) ptx(name string, sched []int) (string, error) {
	n := d.NumInputs
	var b strings.Builder
	grouped := map[int]bool{}
	for _, g := range d.LoadGroups {
		for _, in := range g {
			grouped[in] = true
		}
	}
	fmt.Fprintf(&b, "// %d instructions, %d load groups (bitopt.SynthesizeEpilogueDAG)\n", d.Insts(), len(d.LoadGroups))
	b.WriteString(".version 8.8\n.target sm_100a\n.address_size 64\n\n")
	params := []string{".param .u64 out", ".param .u32 taddr"}
	for i := range n {
		if !grouped[i] {
			params = append(params, fmt.Sprintf(".param .u32 in%d", i))
		}
	}
	fmt.Fprintf(&b, ".visible .entry %s(%s)\n{\n", name, strings.Join(params, ", "))
	fmt.Fprintf(&b, ".reg .b32 %%v<%d>;\n.reg .b8 %%e;\n.reg .b32 %%taddr;\n.reg .b64 %%out;\n", n+len(d.Nodes))
	b.WriteString("ld.param.u64 %out, [out];\nld.param.u32 %taddr, [taddr];\n")
	for i := range n {
		if !grouped[i] {
			fmt.Fprintf(&b, "ld.param.u32 %%v%d, [in%d];\n", i, i)
		}
	}
	col := make([]int, len(d.LoadGroups))
	c := 0
	for g, grp := range d.LoadGroups {
		col[g] = c
		c += len(grp)
	}
	for i, x := range d.Nodes {
		if x.Op == DAGConst {
			fmt.Fprintf(&b, "mov.b32 %%v%d, %#x;\n", n+i, x.Imm)
		}
	}
	scheduled := make([]bool, len(d.Nodes))
	for _, s := range sched {
		if s < 0 {
			g := -s - 1
			if g >= len(d.LoadGroups) {
				return "", fmt.Errorf("bitopt: schedule loads group %d of %d", g, len(d.LoadGroups))
			}
			grp := d.LoadGroups[g]
			if w := len(grp); w&(w-1) != 0 || w > 128 {
				return "", fmt.Errorf("bitopt: load group of %d registers; tcgen05.ld 32x32b takes a power of two up to 128", w)
			}
			regs := make([]string, len(grp))
			for k, in := range grp {
				regs[k] = fmt.Sprintf("%%v%d", in)
			}
			fmt.Fprintf(&b, "tcgen05.ld.sync.aligned.32x32b.x%d.b32 {%s}, [%%taddr+%d];\n", len(grp), strings.Join(regs, ", "), col[g])
			b.WriteString("tcgen05.wait::ld.sync.aligned;\n")
			continue
		}
		if s >= len(d.Nodes) || scheduled[s] {
			return "", fmt.Errorf("bitopt: schedule entry %d invalid or repeated", s)
		}
		scheduled[s] = true
		x := d.Nodes[s]
		dst, a, bb := n+s, x.A, x.B
		switch x.Op {
		case DAGConst:
		case DAGAnd, DAGOr, DAGXor:
			fmt.Fprintf(&b, "%s.b32 %%v%d, %%v%d, %%v%d;\n", x.Op, dst, a, bb)
		case DAGAdd:
			fmt.Fprintf(&b, "add.u32 %%v%d, %%v%d, %%v%d;\n", dst, a, bb)
		case DAGShl:
			fmt.Fprintf(&b, "shl.b32 %%v%d, %%v%d, %d;\n", dst, a, x.Imm)
		case DAGShr:
			fmt.Fprintf(&b, "shr.u32 %%v%d, %%v%d, %d;\n", dst, a, x.Imm)
		case DAGPrmt:
			fmt.Fprintf(&b, "prmt.b32 %%v%d, %%v%d, %%v%d, %#x;\n", dst, a, bb, x.Imm)
		case DAGFMul:
			fmt.Fprintf(&b, "mul.rn.f32 %%v%d, %%v%d, %%v%d;\n", dst, a, bb)
		case DAGFAdd:
			fmt.Fprintf(&b, "add.rn.f32 %%v%d, %%v%d, %%v%d;\n", dst, a, bb)
		case DAGFMax:
			fmt.Fprintf(&b, "max.f32 %%v%d, %%v%d, %%v%d;\n", dst, a, bb)
		case DAGCvtE2M1x2:
			fmt.Fprintf(&b, "cvt.rn.satfinite.e2m1x2.f32 %%e, %%v%d, %%v%d;\ncvt.u32.u8 %%v%d, %%e;\n", a, bb, dst)
		}
	}
	for i, x := range d.Nodes {
		if x.Op != DAGConst && !scheduled[i] {
			return "", fmt.Errorf("bitopt: schedule omits node %d", i)
		}
	}
	for i, o := range d.Outputs {
		fmt.Fprintf(&b, "st.global.b32 [%%out+%d], %%v%d;\n", 4*i, o)
	}
	b.WriteString("ret;\n}\n")
	return b.String(), nil
}

// ptxLine is one line of PTX text, classified for the wait passes.
type ptxLine struct {
	text   string
	insn   bool     // an instruction, ending in ';'
	opcode string   // first token of an instruction
	regs   []string // every %register the instruction names
	dests  []string // registers a tcgen05.ld loads
	stop   bool     // a label, brace, branch, barrier or other tcgen05 operation
}

func parsePTXLine(s string) ptxLine {
	l := ptxLine{text: s}
	t := strings.TrimSpace(s)
	if i := strings.Index(t, "//"); i >= 0 {
		t = strings.TrimSpace(t[:i])
	}
	switch {
	case t == "":
		return l
	case strings.HasSuffix(t, ":"), t == "{", t == "}":
		l.stop = true
		return l
	case !strings.HasSuffix(t, ";") || strings.HasPrefix(t, "."):
		return l
	}
	l.insn = true
	f := strings.Fields(strings.TrimPrefix(t, "@"))
	if strings.HasPrefix(t, "@") && len(f) > 1 {
		f = f[1:]
	}
	l.opcode = strings.TrimSuffix(f[0], ";")
	for _, tok := range strings.FieldsFunc(t, func(r rune) bool {
		return !(r == '%' || r == '_' || r >= '0' && r <= '9' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z')
	}) {
		if strings.HasPrefix(tok, "%") {
			l.regs = append(l.regs, tok)
		}
	}
	switch {
	case strings.HasPrefix(l.opcode, "tcgen05.ld."):
		if i, j := strings.Index(t, "{"), strings.Index(t, "}"); i >= 0 && j > i {
			for _, r := range strings.Split(t[i+1:j], ",") {
				l.dests = append(l.dests, strings.TrimSpace(r))
			}
		}
	case strings.HasPrefix(l.opcode, "tcgen05.wait::ld"):
	case strings.HasPrefix(l.opcode, "tcgen05."), strings.HasPrefix(l.opcode, "bra"),
		strings.HasPrefix(l.opcode, "ret"), strings.HasPrefix(l.opcode, "exit"),
		strings.HasPrefix(l.opcode, "call"), strings.HasPrefix(l.opcode, "bar"),
		strings.HasPrefix(l.opcode, "membar"), strings.HasPrefix(l.opcode, "fence"),
		strings.HasPrefix(l.opcode, "mbarrier"), strings.HasPrefix(l.opcode, "wgmma"):
		l.stop = true
	}
	return l
}

func (l ptxLine) isWaitLd() bool { return strings.HasPrefix(l.opcode, "tcgen05.wait::ld") }

func (l ptxLine) touches(pending map[string]bool) string {
	for _, r := range l.regs {
		if pending[r] {
			return r
		}
	}
	return ""
}

// VerifyWaitDataflow checks that no instruction in ptx reads or writes a
// register a tcgen05.ld loaded before a tcgen05.wait::ld separates them,
// and that no load is still pending at a label, brace, branch, barrier or
// other tcgen05 operation. It is a straight-line check of the text, with
// every label treated as a possible entry with nothing pending.
func VerifyWaitDataflow(ptx string) error {
	pending := map[string]bool{}
	for i, s := range strings.Split(ptx, "\n") {
		l := parsePTXLine(s)
		switch {
		case l.isWaitLd():
			clear(pending)
			continue
		case l.stop && len(pending) > 0:
			return fmt.Errorf("bitopt: line %d: %q reached with tcgen05.ld results pending", i+1, strings.TrimSpace(s))
		}
		if l.dests == nil {
			if r := l.touches(pending); r != "" {
				return fmt.Errorf("bitopt: line %d: %q uses %s before tcgen05.wait::ld", i+1, strings.TrimSpace(s), r)
			}
		}
		for _, r := range l.dests {
			pending[r] = true
		}
	}
	return nil
}

// CoalesceWaits sinks each tcgen05.wait::ld.sync.aligned in ptx to just
// before the first instruction that touches a pending tcgen05.ld result,
// or the first label, brace, branch, barrier or other tcgen05 operation,
// and drops a wait with nothing pending, so that loads issued back to back
// share one wait. A wait is not moved past a later tcgen05.ld when the
// earlier results are read before the text's next wait, since their
// readers would then wait for the later load too. Since tcgen05.wait::ld waits for every earlier
// tcgen05.ld of the thread, a later wait covers the loads an earlier one
// did. It returns the rewritten text and the number of waits removed. ptx
// must pass [VerifyWaitDataflow]; so does the result.
func CoalesceWaits(ptx string) (string, int, error) {
	if err := VerifyWaitDataflow(ptx); err != nil {
		return "", 0, fmt.Errorf("bitopt: input: %w", err)
	}
	var out []string
	var wait string // the deferred wait's text
	pending := map[string]bool{}
	in, kept := 0, 0
	flush := func() {
		if wait != "" {
			out = append(out, wait)
			kept++
			wait = ""
			clear(pending)
		}
	}
	lines := strings.Split(ptx, "\n")
	parsed := make([]ptxLine, len(lines))
	for i, s := range lines {
		parsed[i] = parsePTXLine(s)
	}
	// usedBeforeWait reports whether a line after i touches pending before
	// the text's next wait.
	usedBeforeWait := func(i int) bool {
		for _, l := range parsed[i+1:] {
			if l.isWaitLd() || l.stop {
				return false
			}
			if l.touches(pending) != "" {
				return true
			}
		}
		return false
	}
	for i, s := range lines {
		l := parsed[i]
		if l.isWaitLd() {
			in++
			if len(pending) > 0 && wait == "" {
				wait = s
			}
			continue
		}
		// A wait moves past a later load only if nothing reads the earlier
		// results before the text waits again; otherwise the earlier
		// results' readers would also wait for the later load.
		if l.stop || l.touches(pending) != "" || l.dests != nil && wait != "" && usedBeforeWait(i) {
			flush()
		}
		for _, r := range l.dests {
			pending[r] = true
		}
		out = append(out, s)
	}
	flush()
	res := strings.Join(out, "\n")
	if err := VerifyWaitDataflow(res); err != nil {
		return "", 0, fmt.Errorf("bitopt: internal error: %w", err)
	}
	return res, in - kept, nil
}
