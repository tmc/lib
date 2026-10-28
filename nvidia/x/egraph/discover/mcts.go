package discover

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

// MCTSConfig configures [RunMCTSDiscovery].
type MCTSConfig struct {
	MaxIterations    int     // selection-expansion-rollout-backpropagation rounds; default 2000
	UCB1ExplorationC float64 // UCB1 exploration constant; default √2
	MaxProgramLen    int     // instruction cap; default 2 × the baseline's length
	RolloutDepth     int     // random actions per rollout; default 6
	RandomActions    int     // random instruction rewrites offered per state; default 8
	RandomVectors    int     // Stage B vectors; default 10000
	CycleWeight      float64 // reward weight per modelled cycle saved
	RegisterWeight   float64 // reward weight per peak live register saved
	InstWeight       float64 // reward weight per instruction saved
	Cost             *CostModel
	Seed             uint64
}

// DiscoveredKernelIdiom is a program found by [RunMCTSDiscovery] that is
// verified equivalent to its baseline and cheaper in the weighted
// (cycles, peak registers, instructions) cost.
type DiscoveredKernelIdiom struct {
	Name             string
	BaselineInsts    int
	DiscoveredInsts  int
	BaselineCycles   int
	DiscoveredCycles int
	BaselinePeakRegs int
	DiscoveredRegs   int
	Baseline         Program
	Program          Program
	PTX              string
	RuleLHS, RuleRHS string // generalized e-graph rewrite rule; see [Generalize]
	Iteration        int    // MCTS iteration that found Program
	Iterations       int    // MCTS iterations run
	Mutants          MutantReport
}

// ErrNoImprovement reports that the search found no verified program
// cheaper than the baseline.
var ErrNoImprovement = errors.New("discover: no improvement found")

// RunMCTSDiscovery searches for a program equivalent to baseline that is
// cheaper in the weighted cost cfg.CycleWeight·Cycles +
// cfg.RegisterWeight·PeakRegs + cfg.InstWeight·Insts, where Cycles comes
// from cfg.Cost.
//
// The search is single-player Monte Carlo tree search with UCB1 selection.
// A state is a complete program, starting at baseline. Actions rewrite it:
// collapsing a cone of bitwise instructions with at most three leaves into
// one lop3 (its truth table read off by evaluating the cone on 0xF0, 0xCC
// and 0xAA), collapsing a cone of routing instructions with at most two
// register leaves into one prmt when every result byte is a source byte,
// forwarding an operand past an instruction, swapping adjacent independent
// instructions, and replacing an instruction with a random one from the
// alphabet. A state's reward is its relative cost saving if it passes
// Stage A and Stage B of an [Oracle] for spec, and zero otherwise; failing
// states are not expanded. A new best program must also pass
// [ProveEquivalent] against baseline.
//
// The returned idiom has its [MutantReport] filled in and its rule
// generalized with [Generalize]. If nothing improves on baseline,
// RunMCTSDiscovery returns the baseline idiom and an error wrapping
// [ErrNoImprovement].
func RunMCTSDiscovery(name string, spec SpecFunc, baseline Program, cfg MCTSConfig) (DiscoveredKernelIdiom, error) {
	if err := baseline.Validate(); err != nil {
		return DiscoveredKernelIdiom{}, fmt.Errorf("mcts discovery %s: baseline: %w", name, err)
	}
	cfg = cfg.withDefaults(baseline)
	o, err := NewOracle(baseline.NumInputs, spec, cfg.RandomVectors, cfg.Seed)
	if err != nil {
		return DiscoveredKernelIdiom{}, fmt.Errorf("mcts discovery %s: %w", name, err)
	}
	if !o.StageA(baseline) || !o.StageB(baseline) {
		return DiscoveredKernelIdiom{}, fmt.Errorf("mcts discovery %s: baseline does not match spec: %w", name, ErrNotEquivalent)
	}
	s := &search{
		cfg:      cfg,
		oracle:   o,
		baseline: baseline,
		rng:      rand.New(rand.NewPCG(cfg.Seed, cfg.Seed+1)),
		valid:    make(map[string]bool),
		consts:   constPool(baseline),
	}
	s.cost0 = s.cost(baseline)
	s.best, s.bestCost = baseline, s.cost0
	root := &node{prog: baseline}
	iter := 0
	for iter = 1; iter <= cfg.MaxIterations; iter++ {
		s.iter = iter
		s.round(root)
	}
	idiom := DiscoveredKernelIdiom{
		Name:             name,
		BaselineInsts:    baseline.Len(),
		BaselineCycles:   cfg.Cost.Cycles(baseline),
		BaselinePeakRegs: PeakRegs(baseline),
		Baseline:         baseline,
		Program:          s.best,
		DiscoveredInsts:  s.best.Len(),
		DiscoveredCycles: cfg.Cost.Cycles(s.best),
		DiscoveredRegs:   PeakRegs(s.best),
		PTX:              s.best.PTX(),
		Iteration:        s.bestIter,
		Iterations:       iter - 1,
	}
	idiom.RuleLHS, idiom.RuleRHS = Generalize(baseline, s.best)
	if s.bestCost >= s.cost0 {
		return idiom, fmt.Errorf("mcts discovery %s: %d iterations: %w", name, iter-1, ErrNoImprovement)
	}
	idiom.Mutants = o.CheckMutants(baseline, s.best)
	return idiom, nil
}

func (c MCTSConfig) withDefaults(baseline Program) MCTSConfig {
	if c.MaxIterations <= 0 {
		c.MaxIterations = 2000
	}
	if c.UCB1ExplorationC <= 0 {
		c.UCB1ExplorationC = math.Sqrt2
	}
	if c.MaxProgramLen <= 0 {
		c.MaxProgramLen = 2 * baseline.Len()
	}
	c.MaxProgramLen = min(c.MaxProgramLen, 64)
	if c.RolloutDepth <= 0 {
		c.RolloutDepth = 6
	}
	if c.RandomActions <= 0 {
		c.RandomActions = 8
	}
	if c.RandomVectors <= 0 {
		c.RandomVectors = 10000
	}
	if c.CycleWeight == 0 && c.RegisterWeight == 0 && c.InstWeight == 0 {
		c.CycleWeight, c.RegisterWeight, c.InstWeight = 1, 1, 1
	}
	if c.Cost == nil {
		m := DefaultCostModel()
		c.Cost = &m
	}
	return c
}

type search struct {
	cfg      MCTSConfig
	oracle   *Oracle
	baseline Program
	rng      *rand.Rand
	valid    map[string]bool
	consts   []uint32
	cost0    float64
	best     Program
	bestCost float64
	bestIter int
	iter     int
}

type node struct {
	prog     Program
	parent   *node
	children []*node
	untried  []Program
	expanded bool
	visits   int
	total    float64
	max      float64
}

func (s *search) cost(p Program) float64 {
	c := s.cfg
	return c.CycleWeight*float64(c.Cost.Cycles(p)) +
		c.RegisterWeight*float64(PeakRegs(p)) +
		c.InstWeight*float64(p.Len())
}

// reward returns p's relative cost saving if p passes Stage A and Stage B,
// and ok = false otherwise. It records a new best after proving it.
func (s *search) reward(p Program) (r float64, ok bool) {
	key := p.String()
	ok, seen := s.valid[key]
	if !seen {
		ok = p.Validate() == nil && p.NumInputs == s.oracle.n && s.oracle.StageA(p) && s.oracle.StageB(p)
		s.valid[key] = ok
	}
	if !ok {
		return 0, false
	}
	c := s.cost(p)
	if c < s.bestCost {
		if ProveEquivalent(s.baseline, p) != nil {
			s.valid[key] = false
			return 0, false
		}
		s.best, s.bestCost, s.bestIter = p, c, s.iter
	}
	return max(0, (s.cost0-c)/s.cost0), true
}

// round runs one selection, expansion, rollout and backpropagation.
func (s *search) round(root *node) {
	n := root
	for {
		if !n.expanded {
			n.untried = s.actions(n.prog)
			n.expanded = true
		}
		if len(n.untried) > 0 || len(n.children) == 0 {
			break
		}
		n = s.selectChild(n)
	}
	if len(n.untried) > 0 {
		p := n.untried[len(n.untried)-1]
		n.untried = n.untried[:len(n.untried)-1]
		r, ok := s.reward(p)
		if ok {
			child := &node{prog: p, parent: n}
			n.children = append(n.children, child)
			n = child
			r = max(r, s.rollout(p))
		}
		s.backprop(n, r)
		return
	}
	// A leaf with no actions.
	r, _ := s.reward(n.prog)
	s.backprop(n, r)
}

func (s *search) selectChild(n *node) *node {
	var best *node
	bestU := math.Inf(-1)
	lnN := math.Log(float64(n.visits))
	for _, c := range n.children {
		u := math.Inf(1)
		if c.visits > 0 {
			u = c.total/float64(c.visits) + s.cfg.UCB1ExplorationC*math.Sqrt(lnN/float64(c.visits))
		}
		if u > bestU {
			best, bestU = c, u
		}
	}
	return best
}

func (s *search) rollout(p Program) float64 {
	best := 0.0
	for range s.cfg.RolloutDepth {
		acts := s.actions(p)
		if len(acts) == 0 {
			break
		}
		q := acts[s.rng.IntN(len(acts))]
		r, ok := s.reward(q)
		if !ok {
			break
		}
		best = max(best, r)
		p = q
	}
	return best
}

func (s *search) backprop(n *node, r float64) {
	for ; n != nil; n = n.parent {
		n.visits++
		n.total += r
		n.max = max(n.max, r)
	}
}

// actions returns the distinct successors of p in a seeded random order.
func (s *search) actions(p Program) []Program {
	var out []Program
	seen := map[string]bool{p.String(): true}
	add := func(q Program) {
		if q.Len() == 0 || q.Len() > s.cfg.MaxProgramLen {
			return
		}
		q = q.normalize()
		if k := q.String(); !seen[k] {
			seen[k] = true
			out = append(out, q)
		}
	}
	for _, q := range collapseLop3(p) {
		add(q)
	}
	for _, q := range collapsePrmt(p) {
		add(q)
	}
	for _, q := range forwardOperands(p) {
		add(q)
	}
	for _, q := range swapAdjacent(p) {
		add(q)
	}
	for range s.cfg.RandomActions {
		add(s.randomRewrite(p))
	}
	s.rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// cones returns the instruction sets rooted at root whose members other
// than root have every user inside the set, restricted to instructions
// accepted by ok, as bit masks over instruction indices.
func cones(p Program, root int, ok func(Inst) bool) []uint64 {
	users := p.users()
	seen := map[uint64]bool{}
	var out []uint64
	var grow func(c uint64)
	grow = func(c uint64) {
		if seen[c] {
			return
		}
		seen[c] = true
		out = append(out, c)
		for i := range root {
			if c>>i&1 != 0 || !ok(p.Insts[i]) || len(users[i]) == 0 {
				continue
			}
			inside := true
			for _, u := range users[i] {
				if c>>u&1 == 0 {
					inside = false
					break
				}
			}
			if inside {
				grow(c | 1<<i)
			}
		}
	}
	if ok(p.Insts[root]) {
		grow(1 << root)
	}
	return out
}

// coneLeaves returns the operands a cone reads from outside itself:
// registers first, then immediates, each once.
func coneLeaves(p Program, c uint64) []Operand {
	var regs, imms []Operand
	for i, in := range p.Insts {
		if c>>i&1 == 0 {
			continue
		}
		for k := range in.Op.arity() {
			s := in.Src[k]
			if !s.IsImm && s.Reg >= p.NumInputs && c>>(s.Reg-p.NumInputs)&1 != 0 {
				continue
			}
			if s.IsImm {
				if !slices.Contains(imms, s) {
					imms = append(imms, s)
				}
			} else if !slices.Contains(regs, s) {
				regs = append(regs, s)
			}
		}
	}
	return append(regs, imms...)
}

// evalCone evaluates cone c, rooted at its highest instruction, with leaf
// operand leaves[k] taking value vals[k].
func evalCone(p Program, c uint64, leaves []Operand, vals []uint32) uint32 {
	regs := make([]uint32, p.NumInputs+len(p.Insts))
	var last int
	for i, in := range p.Insts {
		if c>>i&1 == 0 {
			continue
		}
		// Evaluate in on its operand values, held in registers 0 to 2.
		x := Inst{Op: in.Op, Imm: in.Imm}
		var ops [3]uint32
		for k := range in.Op.arity() {
			x.Src[k] = R(k)
			s := in.Src[k]
			if !s.IsImm && s.Reg >= p.NumInputs && c>>(s.Reg-p.NumInputs)&1 != 0 {
				ops[k] = regs[s.Reg]
			} else {
				ops[k] = vals[slices.Index(leaves, s)]
			}
		}
		regs[p.NumInputs+i] = x.eval(ops[:])
		last = i
	}
	return regs[p.NumInputs+last]
}

// replaceCone returns p with the cone rooted at root replaced by in.
// Members other than root become dead and are removed by normalize.
func replaceCone(p Program, root int, in Inst) Program {
	q := p.clone()
	q.Insts[root] = in
	return q
}

func collapseLop3(p Program) []Program {
	var out []Program
	for root := range p.Insts {
		for _, c := range cones(p, root, func(in Inst) bool { return in.Op.bitwise() }) {
			if c&(c-1) == 0 && p.Insts[root].Op == OpLop3 {
				continue
			}
			leaves := coneLeaves(p, c)
			if len(leaves) > 3 || leaves[0].IsImm {
				continue
			}
			ops := slices.Clone(leaves)
			for len(ops) < 3 {
				ops = append(ops, ops[0])
			}
			tt := [3]uint32{0xF0, 0xCC, 0xAA}
			vals := make([]uint32, len(leaves))
			for k := range leaves {
				vals[k] = tt[k]
			}
			// Padding operands repeat the first leaf; the table does not
			// depend on them.
			lut := evalCone(p, c, leaves, vals) & 0xff
			out = append(out, replaceCone(p, root, Inst{Op: OpLop3, Src: [3]Operand{ops[0], ops[1], ops[2]}, Imm: lut}))
		}
	}
	return out
}

func routing(in Inst) bool {
	switch in.Op {
	case OpAnd, OpOr, OpShl, OpShr, OpShf, OpPrmt:
		return true
	}
	return false
}

var probeWords = [...]uint32{0x03020100, 0x07060504, 0xA1B2C3D4, 0x5E6F7081, 0x12345678, 0x9ABCDEF0, 0xDEADBEEF, 0x0BADF00D}

func collapsePrmt(p Program) []Program {
	var out []Program
	for root := range p.Insts {
		for _, c := range cones(p, root, routing) {
			if c&(c-1) == 0 {
				continue
			}
			leaves := coneLeaves(p, c)
			var regs []Operand
			for _, l := range leaves {
				if !l.IsImm {
					regs = append(regs, l)
				}
			}
			if len(regs) == 0 || len(regs) > 2 {
				continue
			}
			sel, ok := deriveSelector(p, c, leaves, len(regs))
			if !ok {
				continue
			}
			b := regs[0]
			if len(regs) == 2 {
				b = regs[1]
			}
			out = append(out, replaceCone(p, root, Inst{Op: OpPrmt, Src: [3]Operand{regs[0], b}, Imm: sel}))
		}
	}
	return out
}

// deriveSelector finds, for each result byte of the cone, a byte of b:a it
// equals on every probe, where a and b are the cone's register leaves.
func deriveSelector(p Program, c uint64, leaves []Operand, nregs int) (uint32, bool) {
	type probe struct{ a, b, out uint32 }
	var probes []probe
	vals := make([]uint32, len(leaves))
	for i := range probeWords {
		a := probeWords[i]
		b := probeWords[(i+3)%len(probeWords)]
		if nregs == 1 {
			b = a
		}
		r := 0
		for k, l := range leaves {
			switch {
			case l.IsImm:
				vals[k] = l.Imm
			case r == 0:
				vals[k], r = a, 1
			default:
				vals[k] = b
			}
		}
		probes = append(probes, probe{a, b, evalCone(p, c, leaves, vals)})
	}
	var sel uint32
	for j := range 4 {
		found := false
		for s := range uint32(8) {
			if nregs == 1 && s >= 4 {
				break
			}
			all := true
			for _, pr := range probes {
				x := uint64(pr.b)<<32 | uint64(pr.a)
				if uint32(x>>(8*s))&0xff != pr.out>>(8*j)&0xff {
					all = false
					break
				}
			}
			if all {
				sel |= s << (4 * j)
				found = true
				break
			}
		}
		if !found {
			return 0, false
		}
	}
	return sel, true
}

// forwardOperands returns p with every use of an instruction's result
// replaced by one of that instruction's register operands. The last
// instruction, whose result is the program's, is not forwarded.
func forwardOperands(p Program) []Program {
	var out []Program
	last := len(p.Insts) - 1
	for i, in := range p.Insts[:last] {
		for k := range in.Op.arity() {
			s := in.Src[k]
			if s.IsImm {
				continue
			}
			q := p.clone()
			r := p.NumInputs + i
			for j := i + 1; j <= last; j++ {
				for m := range q.Insts[j].Op.arity() {
					if o := q.Insts[j].Src[m]; !o.IsImm && o.Reg == r {
						q.Insts[j].Src[m] = s
					}
				}
			}
			out = append(out, q)
		}
	}
	return out
}

// swapAdjacent returns p with each pair of adjacent independent
// instructions swapped, keeping the last instruction last.
func swapAdjacent(p Program) []Program {
	var out []Program
	for i := 0; i+2 < len(p.Insts); i++ {
		next := p.Insts[i+1]
		dep := false
		for k := range next.Op.arity() {
			if s := next.Src[k]; !s.IsImm && s.Reg == p.NumInputs+i {
				dep = true
			}
		}
		if dep {
			continue
		}
		q := p.clone()
		a, b := q.Insts[i], q.Insts[i+1]
		ra, rb := p.NumInputs+i, p.NumInputs+i+1
		q.Insts[i], q.Insts[i+1] = b, a
		for j := i + 2; j < len(q.Insts); j++ {
			for m := range q.Insts[j].Op.arity() {
				switch o := &q.Insts[j].Src[m]; {
				case o.IsImm:
				case o.Reg == ra:
					o.Reg = rb
				case o.Reg == rb:
					o.Reg = ra
				}
			}
		}
		out = append(out, q)
	}
	return out
}

// constPool returns the immediates p uses.
func constPool(p Program) []uint32 {
	var c []uint32
	for _, in := range p.Insts {
		for k := range in.Op.arity() {
			if s := in.Src[k]; s.IsImm && !slices.Contains(c, s.Imm) {
				c = append(c, s.Imm)
			}
		}
	}
	return c
}

// randomRewrite replaces one instruction of p with a random instruction
// from the alphabet reading earlier registers or pooled immediates.
func (s *search) randomRewrite(p Program) Program {
	q := p.clone()
	i := s.rng.IntN(len(q.Insts))
	nregs := q.NumInputs + i
	reg := func() Operand { return R(s.rng.IntN(nregs)) }
	op := Opcode(1 + s.rng.IntN(int(numOpcodes)-1))
	in := Inst{Op: op}
	for k := range op.arity() {
		in.Src[k] = reg()
		if k > 0 && (op.bitwise() || op == OpPrmt) && len(s.consts) > 0 && s.rng.IntN(4) == 0 {
			in.Src[k] = I(s.consts[s.rng.IntN(len(s.consts))])
		}
	}
	switch op {
	case OpShl, OpShr, OpShf:
		in.Imm = uint32([]int{8, 16, 24, 1 + s.rng.IntN(31)}[s.rng.IntN(4)])
	case OpLop3:
		in.Imm = uint32(s.rng.IntN(256))
	case OpPrmt:
		for j := range 4 {
			in.Imm |= uint32(s.rng.IntN(8)) << (4 * j)
		}
	}
	q.Insts[i] = in
	return q
}
