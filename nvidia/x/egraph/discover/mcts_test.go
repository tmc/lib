package discover

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// muxParityBaseline computes ((a & m) | (b & ~m)) ^ (c ^ d) with inputs
// a, b, m, c, d in six instructions.
func muxParityBaseline() Program {
	return Program{NumInputs: 5, Insts: []Inst{
		{Op: OpAnd, Src: [3]Operand{R(0), R(2)}}, // t0 = a & m
		{Op: OpNot, Src: [3]Operand{R(2)}},       // t1 = ~m
		{Op: OpAnd, Src: [3]Operand{R(1), R(6)}}, // t2 = b & t1
		{Op: OpOr, Src: [3]Operand{R(5), R(7)}},  // t3 = t0 | t2
		{Op: OpXor, Src: [3]Operand{R(3), R(4)}}, // t4 = c ^ d
		{Op: OpXor, Src: [3]Operand{R(8), R(9)}}, // t5 = t3 ^ t4
	}}
}

func muxParitySpec(in []uint32) uint32 {
	a, b, m, c, d := in[0], in[1], in[2], in[3], in[4]
	return (a&m | b&^m) ^ c ^ d
}

// bytePlaneBaseline computes (x & 0x00FF00FF) | ((y & 0x00FF00FF) << 8).
func bytePlaneBaseline() Program {
	return Program{NumInputs: 2, Insts: []Inst{
		{Op: OpAnd, Src: [3]Operand{R(0), I(0x00FF00FF)}},
		{Op: OpAnd, Src: [3]Operand{R(1), I(0x00FF00FF)}},
		{Op: OpShl, Src: [3]Operand{R(3)}, Imm: 8},
		{Op: OpOr, Src: [3]Operand{R(2), R(4)}},
	}}
}

func bytePlaneSpec(in []uint32) uint32 {
	return in[0]&0x00FF00FF | (in[1]&0x00FF00FF)<<8
}

// rotateBaseline computes x rotated left by one byte.
func rotateBaseline() Program {
	return Program{NumInputs: 1, Insts: []Inst{
		{Op: OpShl, Src: [3]Operand{R(0)}, Imm: 8},
		{Op: OpShr, Src: [3]Operand{R(0)}, Imm: 24},
		{Op: OpOr, Src: [3]Operand{R(1), R(2)}},
	}}
}

func rotateSpec(in []uint32) uint32 { return in[0]<<8 | in[0]>>24 }

func TestRunMCTSDiscoveryRediscoversIdioms(t *testing.T) {
	tests := []struct {
		name     string
		spec     SpecFunc
		baseline Program
		maxInsts int
		wantOps  []Opcode
		// neutral is the number of mutants proven equivalent: prmt(x, x)
		// reads the same word whichever selector bit 2 says.
		neutral int
	}{
		{"mux_parity", muxParitySpec, muxParityBaseline(), 2, []Opcode{OpLop3}, 0},
		{"byte_plane", bytePlaneSpec, bytePlaneBaseline(), 1, []Opcode{OpPrmt}, 0},
		{"rotate_byte", rotateSpec, rotateBaseline(), 1, []Opcode{OpPrmt}, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := MCTSConfig{MaxIterations: 2000, Seed: 1}
			got, err := RunMCTSDiscovery(tt.name, tt.spec, tt.baseline, cfg)
			if err != nil {
				t.Fatalf("RunMCTSDiscovery: %v", err)
			}
			t.Logf("%s: iteration %d/%d, %d->%d insts, %d->%d cycles, %d->%d regs, mutants %+v\n%s",
				tt.name, got.Iteration, got.Iterations, got.BaselineInsts, got.DiscoveredInsts,
				got.BaselineCycles, got.DiscoveredCycles, got.BaselinePeakRegs, got.DiscoveredRegs, got.Mutants, got.PTX)
			if got.DiscoveredInsts > tt.maxInsts {
				t.Errorf("DiscoveredInsts = %d, want <= %d:\n%s", got.DiscoveredInsts, tt.maxInsts, got.PTX)
			}
			if 2*got.DiscoveredInsts > got.BaselineInsts {
				t.Errorf("instructions %d -> %d, want >= 50%% reduction", got.BaselineInsts, got.DiscoveredInsts)
			}
			if 2*got.DiscoveredCycles > got.BaselineCycles {
				t.Errorf("cycles %d -> %d, want >= 50%% reduction", got.BaselineCycles, got.DiscoveredCycles)
			}
			for _, in := range got.Program.Insts {
				ok := false
				for _, op := range tt.wantOps {
					ok = ok || in.Op == op
				}
				if !ok && in.Op != OpXor {
					t.Errorf("discovered program uses %v, want %v:\n%s", in.Op, tt.wantOps, got.PTX)
				}
			}
			if got.Iteration < 1 || got.Iteration > cfg.MaxIterations {
				t.Errorf("Iteration = %d, want in [1, %d]", got.Iteration, cfg.MaxIterations)
			}

			// Soundness: re-verify with an independent oracle seed.
			o, err := NewOracle(tt.baseline.NumInputs, tt.spec, 10000, 99)
			if err != nil {
				t.Fatal(err)
			}
			if err := o.Verify(tt.baseline, got.Program); err != nil {
				t.Errorf("independent Verify: %v", err)
			}
			// Every single-point mutant must be rejected unless it is
			// proven equivalent.
			if got.Mutants.Total == 0 || got.Mutants.Accepted != tt.neutral || got.Mutants.Rejected != got.Mutants.Total-tt.neutral {
				t.Errorf("Mutants = %+v, want all but %d rejected", got.Mutants, tt.neutral)
			}
		})
	}
}

func TestRunMCTSDiscoveryDeterministic(t *testing.T) {
	cfg := MCTSConfig{MaxIterations: 300, Seed: 7}
	a, err := RunMCTSDiscovery("mux", muxParitySpec, muxParityBaseline(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, err := RunMCTSDiscovery("mux", muxParitySpec, muxParityBaseline(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if a.PTX != b.PTX || a.Iteration != b.Iteration {
		t.Errorf("runs differ:\n%s(iter %d)\n%s(iter %d)", a.PTX, a.Iteration, b.PTX, b.Iteration)
	}
}

func TestRunMCTSDiscoveryRejectsWrongBaseline(t *testing.T) {
	bad := bytePlaneBaseline()
	bad.Insts[2].Imm = 16 // shift by 16 instead of 8
	_, err := RunMCTSDiscovery("bad", bytePlaneSpec, bad, MCTSConfig{MaxIterations: 10})
	if !errors.Is(err, ErrNotEquivalent) {
		t.Fatalf("RunMCTSDiscovery(wrong baseline) = %v, want ErrNotEquivalent", err)
	}
}

func TestRunMCTSDiscoveryNoImprovement(t *testing.T) {
	// A single xor is already optimal in the alphabet.
	base := Program{NumInputs: 2, Insts: []Inst{{Op: OpXor, Src: [3]Operand{R(0), R(1)}}}}
	got, err := RunMCTSDiscovery("xor", func(in []uint32) uint32 { return in[0] ^ in[1] }, base, MCTSConfig{MaxIterations: 50})
	if !errors.Is(err, ErrNoImprovement) {
		t.Fatalf("err = %v, want ErrNoImprovement", err)
	}
	if got.DiscoveredInsts != 1 {
		t.Errorf("DiscoveredInsts = %d, want baseline 1", got.DiscoveredInsts)
	}
}

func TestProveEquivalent(t *testing.T) {
	rotPrmt := Program{NumInputs: 1, Insts: []Inst{{Op: OpPrmt, Src: [3]Operand{R(0), R(0)}, Imm: 0x2103}}}
	rotShf := Program{NumInputs: 1, Insts: []Inst{{Op: OpShf, Src: [3]Operand{R(0), R(0)}, Imm: 8}}}
	rotShf16 := Program{NumInputs: 1, Insts: []Inst{{Op: OpShf, Src: [3]Operand{R(0), R(0)}, Imm: 16}}}
	mux := Program{NumInputs: 5, Insts: []Inst{
		{Op: OpLop3, Src: [3]Operand{R(0), R(1), R(2)}, Imm: 0xe4}, // m ? a : b
		{Op: OpLop3, Src: [3]Operand{R(5), R(3), R(4)}, Imm: 0x96},
	}}
	muxBad := mux.clone()
	muxBad.Insts[0].Imm ^= 0x01
	planePrmt := Program{NumInputs: 2, Insts: []Inst{{Op: OpPrmt, Src: [3]Operand{R(0), R(1)}, Imm: 0x6240}}}
	// sign-replicate on byte 3 instead of a copy
	planeSign := Program{NumInputs: 2, Insts: []Inst{{Op: OpPrmt, Src: [3]Operand{R(0), R(1)}, Imm: 0xE240}}}
	zero := Program{NumInputs: 1, Insts: []Inst{{Op: OpAnd, Src: [3]Operand{R(0), I(0)}}}}
	// Bit 0 of each is input bit 24 (shr) or 7 (prmt sign replicate); a
	// support set that misplaced it would enumerate only zeros there.
	shrBit := Program{NumInputs: 1, Insts: []Inst{
		{Op: OpShr, Src: [3]Operand{R(0)}, Imm: 24},
		{Op: OpAnd, Src: [3]Operand{R(1), I(1)}},
	}}
	signBit := Program{NumInputs: 1, Insts: []Inst{
		{Op: OpPrmt, Src: [3]Operand{R(0), R(0)}, Imm: 0x0008},
		{Op: OpAnd, Src: [3]Operand{R(1), I(1)}},
	}}
	tests := []struct {
		name string
		a, b Program
		want error
	}{
		{"shr_support", zero, shrBit, ErrNotEquivalent},
		{"prmt_sign_support", zero, signBit, ErrNotEquivalent},
		{"rotate_prmt", rotateBaseline(), rotPrmt, nil},
		{"rotate_shf", rotateBaseline(), rotShf, nil},
		{"rotate_shf16", rotateBaseline(), rotShf16, ErrNotEquivalent},
		{"mux_lop3", muxParityBaseline(), mux, nil},
		{"mux_lop3_bit0", muxParityBaseline(), muxBad, ErrNotEquivalent},
		{"plane_prmt", bytePlaneBaseline(), planePrmt, nil},
		{"plane_sign_replicate", bytePlaneBaseline(), planeSign, ErrNotEquivalent},
		{"inputs_differ", rotPrmt, planePrmt, ErrNotEquivalent},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ProveEquivalent(tt.a, tt.b)
			if tt.want == nil && err != nil || tt.want != nil && !errors.Is(err, tt.want) {
				t.Fatalf("ProveEquivalent = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestProveEquivalentUnproven(t *testing.T) {
	// Bit 0 of a^b^c^d^e^f after folding each input by 8 and 16 depends
	// on 6·4 = 24 input bits, beyond the enumeration bound.
	p := Program{NumInputs: 6}
	for k := range 6 {
		r := p.NumInputs + len(p.Insts)
		p.Insts = append(p.Insts,
			Inst{Op: OpShr, Src: [3]Operand{R(k)}, Imm: 8},
			Inst{Op: OpXor, Src: [3]Operand{R(k), R(r)}},
			Inst{Op: OpShr, Src: [3]Operand{R(r + 1)}, Imm: 16},
			Inst{Op: OpXor, Src: [3]Operand{R(r + 1), R(r + 2)}},
		)
	}
	acc := p.NumInputs + 3
	for k := 1; k < 6; k++ {
		p.Insts = append(p.Insts, Inst{Op: OpXor, Src: [3]Operand{R(acc), R(p.NumInputs + 4*k + 3)}})
		acc = p.NumInputs + len(p.Insts) - 1
	}
	if err := ProveEquivalent(p, p); !errors.Is(err, ErrUnproven) {
		t.Fatalf("ProveEquivalent = %v, want ErrUnproven", err)
	}
}

func TestOracleStages(t *testing.T) {
	o, err := NewOracle(2, bytePlaneSpec, 10000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(o.boundary) != 64 || len(o.random) != 10000 {
		t.Fatalf("oracle has %d boundary and %d random vectors, want 64 and 10000", len(o.boundary), len(o.random))
	}
	good := bytePlaneBaseline()
	if !o.StageA(good) || !o.StageB(good) {
		t.Fatalf("baseline fails its own spec")
	}
	bad := good.clone()
	bad.Insts[0].Src[1] = I(0x00FF00FE) // one mask bit off
	if o.StageA(bad) && o.StageB(bad) {
		t.Fatalf("stages accept a mask with bit 0 cleared")
	}
	if err := o.Verify(good, bad); !errors.Is(err, ErrNotEquivalent) {
		t.Fatalf("Verify(bad) = %v, want ErrNotEquivalent", err)
	}
}

func TestMutantsCoverEveryImmediate(t *testing.T) {
	p := Program{NumInputs: 3, Insts: []Inst{
		{Op: OpLop3, Src: [3]Operand{R(0), R(1), R(2)}, Imm: 0xca},
		{Op: OpPrmt, Src: [3]Operand{R(3), R(0)}, Imm: 0x5410},
	}}
	if got, want := len(Mutants(p)), 8+16; got != want {
		t.Errorf("len(Mutants) = %d, want %d", got, want)
	}
}

func TestCostModel(t *testing.T) {
	m := DefaultCostModel()
	tests := []struct {
		name         string
		p            Program
		cycles, regs int
	}{
		{"mux_baseline", muxParityBaseline(), 18, 6},
		{"byte_plane_baseline", bytePlaneBaseline(), 13, 3},
		{"rotate_baseline", rotateBaseline(), 9, 3},
		{"single_prmt", Program{NumInputs: 2, Insts: []Inst{{Op: OpPrmt, Src: [3]Operand{R(0), R(1)}, Imm: 0x6240}}}, 4, 3},
	}
	for _, tt := range tests {
		if got := m.Cycles(tt.p); got != tt.cycles {
			t.Errorf("%s: Cycles = %d, want %d", tt.name, got, tt.cycles)
		}
		if got := PeakRegs(tt.p); got != tt.regs {
			t.Errorf("%s: PeakRegs = %d, want %d", tt.name, got, tt.regs)
		}
	}
}

func TestProgramPTXAndCUDA(t *testing.T) {
	p := Program{NumInputs: 2, Insts: []Inst{{Op: OpPrmt, Src: [3]Operand{R(0), R(1)}, Imm: 0x6240}}}
	if got, want := p.PTX(), "prmt.b32 %t0, %in0, %in1, 0x6240;\n"; got != want {
		t.Errorf("PTX = %q, want %q", got, want)
	}
	cu := p.CUDA("plane")
	for _, want := range []string{
		"__device__ __forceinline__ uint32_t plane(uint32_t in0, uint32_t in1)",
		`"prmt.b32 t0, %1, %2, 0x6240;\n\t"`,
		`"mov.b32 %0, t0;\n\t}"`,
		`: "=r"(out) : "r"(in0), "r"(in1));`,
	} {
		if !strings.Contains(cu, want) {
			t.Errorf("CUDA missing %q:\n%s", want, cu)
		}
	}
}

func TestProgramValidate(t *testing.T) {
	tests := []struct {
		name string
		p    Program
	}{
		{"no_insts", Program{NumInputs: 1}},
		{"forward_ref", Program{NumInputs: 1, Insts: []Inst{{Op: OpNot, Src: [3]Operand{R(1)}}}}},
		{"imm_first", Program{NumInputs: 1, Insts: []Inst{{Op: OpAnd, Src: [3]Operand{I(1), R(0)}}}}},
		{"shift_32", Program{NumInputs: 1, Insts: []Inst{{Op: OpShl, Src: [3]Operand{R(0)}, Imm: 32}}}},
		{"lut_9bit", Program{NumInputs: 1, Insts: []Inst{{Op: OpLop3, Src: [3]Operand{R(0), R(0), R(0)}, Imm: 0x100}}}},
		{"extra_operand", Program{NumInputs: 1, Insts: []Inst{{Op: OpNot, Src: [3]Operand{R(0), I(1)}}}}},
		{"too_many_inputs", Program{NumInputs: MaxInputs + 1, Insts: []Inst{{Op: OpNot, Src: [3]Operand{R(0)}}}}},
	}
	for _, tt := range tests {
		if err := tt.p.Validate(); err == nil {
			t.Errorf("%s: Validate = nil, want error", tt.name)
		}
	}
}

func TestRulePromotesIntoEGraph(t *testing.T) {
	idiom, err := RunMCTSDiscovery("mux_parity", muxParitySpec, muxParityBaseline(), MCTSConfig{Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	rule, err := Rule(idiom)
	if err != nil {
		t.Fatalf("Rule: %v", err)
	}
	t.Logf("%s => %s", idiom.RuleLHS, idiom.RuleRHS)

	g := egraph.New()
	leaf := func(name string) egraph.Id { return g.Add(egraph.NewNode(name)) }
	a, b, m, c, d := leaf("x0"), leaf("x1"), leaf("x2"), leaf("x3"), leaf("x4")
	and1 := g.Add(egraph.NewNode("and", a, m))
	notm := g.Add(egraph.NewNode("not", m))
	and2 := g.Add(egraph.NewNode("and", b, notm))
	or := g.Add(egraph.NewNode("or", and1, and2))
	par := g.Add(egraph.NewNode("xor", c, d))
	root := g.Add(egraph.NewNode("xor", or, par))
	if n := rule.Apply(g); n == 0 {
		t.Fatalf("rule %s => %s did not apply", idiom.RuleLHS, idiom.RuleRHS)
	}
	g.Rebuild()
	found := false
	for _, n := range g.Class(root).Nodes {
		found = found || strings.HasPrefix(n.Op, "lop3_")
	}
	if !found {
		t.Errorf("root class has no lop3 node after rewriting")
	}

	// A rule whose sides are not equivalent is refused.
	idiom.Program.Insts[0].Imm ^= 0x80
	if _, err := Rule(idiom); !errors.Is(err, ErrNotEquivalent) {
		t.Errorf("Rule(mutated) = %v, want ErrNotEquivalent", err)
	}
}

func TestSelectChildUCB1(t *testing.T) {
	s := &search{cfg: MCTSConfig{UCB1ExplorationC: 1}}
	parent := &node{visits: 20}
	lo := &node{visits: 10, total: 2}  // mean 0.2
	hi := &node{visits: 10, total: 6}  // mean 0.6
	rare := &node{visits: 1, total: 0} // mean 0, large bonus
	parent.children = []*node{lo, hi}
	if got := s.selectChild(parent); got != hi {
		t.Errorf("selectChild picked mean %.1f, want the higher mean 0.6", got.total/float64(got.visits))
	}
	// √(ln 20 / 1) ≈ 1.73 > 0.6 + √(ln 20 / 10) ≈ 1.15
	parent.children = append(parent.children, rare)
	if got := s.selectChild(parent); got != rare {
		t.Errorf("selectChild ignored the exploration bonus")
	}
	unvisited := &node{}
	parent.children = append(parent.children, unvisited)
	if got := s.selectChild(parent); got != unvisited {
		t.Errorf("selectChild skipped an unvisited child")
	}
}
