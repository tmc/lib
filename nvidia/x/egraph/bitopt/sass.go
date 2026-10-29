package bitopt

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/sassequiv"
)

// InstOp is a SASS bit-manipulation opcode emitted when lowering a [Program]
// or constructing a straight-line SASS kernel body.
type InstOp int

const (
	OpAndImm InstOp = iota // LOP3.LUT Rd, Rs, imm, RZ, 0xc0, !PT
	OpOrImm                // LOP3.LUT Rd, Rs, imm, RZ, 0xfc, !PT
	OpXorImm               // LOP3.LUT Rd, Rs, imm, RZ, 0x3c, !PT
	OpShlImm               // SHF.L.U32 Rd, Rs, imm, RZ
	OpShrImm               // SHF.R.U32.HI Rd, RZ, imm, Rs
	OpAddImm               // IADD3 Rd, Rs, imm, RZ
	OpPRMT                 // PRMT Rd, Rs0, sel, Rs1
	OpLOP3                 // LOP3.LUT Rd, Rs0, Rs1, Rs2, imm8, !PT
	OpSHFL                 // SHF.L.U32 Rd, Rs0, imm, Rs1
	OpSHFR                 // SHF.R.U32.HI Rd, Rs0, imm, Rs1
)

// Inst is a single SASS bit-manipulation instruction.
type Inst struct {
	Op   InstOp
	Dst  string
	Src0 string
	Src1 string
	Src2 string
	Imm  uint32
	LUT  uint8
}

// SASS formats the instruction in Hoppper/Blackwell cuobjdump syntax.
func (in Inst) SASS() (string, error) {
	if in.Dst == "" || in.Src0 == "" {
		return "", fmt.Errorf("bitopt: instruction missing register operand")
	}
	switch in.Op {
	case OpAndImm:
		return fmt.Sprintf("LOP3.LUT %s, %s, %#x, RZ, 0xc0, !PT", in.Dst, in.Src0, in.Imm), nil
	case OpOrImm:
		return fmt.Sprintf("LOP3.LUT %s, %s, %#x, RZ, 0xfc, !PT", in.Dst, in.Src0, in.Imm), nil
	case OpXorImm:
		return fmt.Sprintf("LOP3.LUT %s, %s, %#x, RZ, 0x3c, !PT", in.Dst, in.Src0, in.Imm), nil
	case OpShlImm:
		return fmt.Sprintf("SHF.L.U32 %s, %s, %#x, RZ", in.Dst, in.Src0, in.Imm), nil
	case OpShrImm:
		return fmt.Sprintf("SHF.R.U32.HI %s, RZ, %#x, %s", in.Dst, in.Imm, in.Src0), nil
	case OpAddImm:
		return fmt.Sprintf("IADD3 %s, %s, %#x, RZ", in.Dst, in.Src0, in.Imm), nil
	case OpPRMT:
		src1 := in.Src1
		if src1 == "" {
			src1 = "RZ"
		}
		return fmt.Sprintf("PRMT %s, %s, %#x, %s", in.Dst, in.Src0, in.Imm, src1), nil
	case OpLOP3:
		src1, src2 := in.Src1, in.Src2
		if src1 == "" {
			src1 = "RZ"
		}
		if src2 == "" {
			src2 = "RZ"
		}
		return fmt.Sprintf("LOP3.LUT %s, %s, %s, %s, %#02x, !PT", in.Dst, in.Src0, src1, src2, in.LUT), nil
	case OpSHFL:
		src1 := in.Src1
		if src1 == "" {
			src1 = "RZ"
		}
		return fmt.Sprintf("SHF.L.U32 %s, %s, %#x, %s", in.Dst, in.Src0, in.Imm, src1), nil
	case OpSHFR:
		src1 := in.Src1
		if src1 == "" {
			src1 = "RZ"
		}
		return fmt.Sprintf("SHF.R.U32.HI %s, %s, %#x, %s", in.Dst, in.Src0, in.Imm, src1), nil
	default:
		return "", fmt.Errorf("bitopt: unknown SASS instruction op %d", in.Op)
	}
}

func (in Inst) String() string {
	s, err := in.SASS()
	if err != nil {
		return fmt.Sprintf("<invalid inst: %v>", err)
	}
	return s
}

// Insts lowers a [Program] into a straight-line sequence of [Inst] instructions
// reading inRegs and returning the slice of output register names.
func (s *Solution) Insts(inRegs []string) ([]Inst, []string, error) {
	if s == nil {
		return nil, nil, fmt.Errorf("bitopt: nil program")
	}
	words := s.Problem.Words
	if words < 1 {
		words = 1
		for _, m := range s.Moves {
			words = max(words, m.A+1, m.B+1)
		}
	}
	if len(inRegs) == 0 {
		inRegs = make([]string, words)
		for w := range words {
			inRegs[w] = fmt.Sprintf("R%d", 6+w)
		}
	}
	if len(inRegs) < words {
		return nil, nil, fmt.Errorf("bitopt: have %d input registers, want at least %d", len(inRegs), words)
	}
	nextReg := 6 + len(inRegs)
	for _, r := range inRegs {
		if n := parseRegNum(r); n >= nextReg {
			nextReg = n + 1
		}
	}
	allocReg := func() string {
		r := fmt.Sprintf("R%d", nextReg)
		nextReg++
		return r
	}

	var insts []Inst
	moveReg := make([]string, len(s.Moves))
	for i, m := range s.Moves {
		if m.A < 0 || m.A >= len(inRegs) || m.B < 0 || m.B >= len(inRegs) {
			return nil, nil, fmt.Errorf("bitopt: move %d references word out of range", i)
		}
		ra, rb := inRegs[m.A], inRegs[m.B]
		switch m.Op {
		case Ident:
			if m.Sel != 0 {
				rd := allocReg()
				insts = append(insts, Inst{Op: OpAndImm, Dst: rd, Src0: ra, Imm: m.Sel})
				moveReg[i] = rd
			} else {
				moveReg[i] = ra
			}
		case Shl:
			src := ra
			if m.Sel != 0 {
				rPre := allocReg()
				insts = append(insts, Inst{Op: OpAndImm, Dst: rPre, Src0: ra, Imm: m.Sel})
				src = rPre
			}
			rd := allocReg()
			insts = append(insts, Inst{Op: OpShlImm, Dst: rd, Src0: src, Imm: uint32(m.N)})
			moveReg[i] = rd
		case Shr:
			src := ra
			shift := m.N
			if m.Sel > 0 && int(m.Sel) < m.N {
				rPre := allocReg()
				insts = append(insts, Inst{Op: OpShrImm, Dst: rPre, Src0: ra, Imm: m.Sel})
				src = rPre
				shift = m.N - int(m.Sel)
			}
			rd := allocReg()
			insts = append(insts, Inst{Op: OpShrImm, Dst: rd, Src0: src, Imm: uint32(shift)})
			moveReg[i] = rd
		case Funnel:
			rd := allocReg()
			insts = append(insts, Inst{Op: OpSHFL, Dst: rd, Src0: ra, Src1: rb, Imm: uint32(m.N)})
			moveReg[i] = rd
		case Perm:
			rd := allocReg()
			insts = append(insts, Inst{Op: OpPRMT, Dst: rd, Src0: ra, Src1: rb, Imm: m.Sel})
			moveReg[i] = rd
		default:
			return nil, nil, fmt.Errorf("bitopt: unsupported move op %d", m.Op)
		}
	}

	outRegs := make([]string, len(s.Parts))
	for k, ps := range s.Parts {
		if len(ps) == 0 {
			outRegs[k] = "RZ"
			continue
		}
		sorted := slices.Clone(ps)
		slices.SortFunc(sorted, func(a, b Part) int {
			if a.Mask < b.Mask {
				return -1
			}
			if a.Mask > b.Mask {
				return 1
			}
			return a.Move - b.Move
		})
		for _, pt := range sorted {
			if pt.Move < 0 || pt.Move >= len(moveReg) {
				return nil, nil, fmt.Errorf("bitopt: output %d references bad move %d", k, pt.Move)
			}
		}
		rOut := allocReg()
		insts = append(insts, Inst{
			Op:   OpAndImm,
			Dst:  rOut,
			Src0: moveReg[sorted[0].Move],
			Imm:  sorted[0].Mask,
		})
		for _, pt := range sorted[1:] {
			rPart := allocReg()
			insts = append(insts,
				Inst{
					Op:   OpAndImm,
					Dst:  rPart,
					Src0: moveReg[pt.Move],
					Imm:  pt.Mask,
				},
				Inst{
					Op:   OpLOP3,
					Dst:  rOut,
					Src0: rOut,
					Src1: rPart,
					Src2: "RZ",
					LUT:  0xfc,
				},
			)
		}
		outRegs[k] = rOut
	}
	return insts, outRegs, nil
}

// SASS lowers s into a complete sm_90a cuobjdump SASS listing accepted by
// [cudasass.Parse] and [sassequiv.Compare]. When outPtrHiLo[0] is empty, R4:R5
// is used as the 64-bit output pointer pair; when inRegs is empty, R6..R(5+Words)
// are used for the loaded input words.
func (s *Solution) SASS(outPtrHiLo [2]string, inRegs []string) (string, error) {
	if s == nil {
		return "", fmt.Errorf("bitopt: nil program")
	}
	words := s.Problem.Words
	if words < 1 {
		words = 1
		for _, m := range s.Moves {
			words = max(words, m.A+1, m.B+1)
		}
	}
	if len(inRegs) == 0 {
		inRegs = make([]string, words)
		for w := range words {
			inRegs[w] = fmt.Sprintf("R%d", 6+w)
		}
	}
	if len(inRegs) < words {
		return "", fmt.Errorf("bitopt: have %d input registers, want at least %d", len(inRegs), words)
	}
	body, outRegs, err := s.Insts(inRegs)
	if err != nil {
		return "", err
	}
	return FormatSASSListing("f", outPtrHiLo, inRegs[:words], body, outRegs)
}

// SASSListing returns the canonical sm_90a SASS listing for s, or the empty
// string if s is invalid.
func (s *Solution) SASSListing() string {
	listing, err := s.SASS([2]string{}, nil)
	if err != nil {
		return ""
	}
	return listing
}

// FormatSASSListing wraps a sequence of [Inst] instructions with the standard
// Hopper/Blackwell constant-bank prologue, global input loads into inRegs, and
// global output stores from outRegs.
func FormatSASSListing(fnName string, outPtrHiLo [2]string, inRegs []string, body []Inst, outRegs []string) (string, error) {
	if fnName == "" {
		fnName = "f"
	}
	outBase := strings.TrimSpace(outPtrHiLo[0])
	if outBase == "" {
		outBase = "R4"
	}
	lines := []string{
		"LDC R1, c[0x0][0x28]",
		"S2R R0, SR_TID.X",
		"ULDC.64 UR4, c[0x0][0x208]",
		"LDC.64 R2, c[0x0][0x210]",
		fmt.Sprintf("LDC.64 %s, c[0x0][0x218]", outBase),
		"IMAD.WIDE.U32 R2, R0, 0x4, R2",
		fmt.Sprintf("IMAD.WIDE.U32 %s, R0, 0x4, %s", outBase, outBase),
	}
	for w, reg := range inRegs {
		if w == 0 {
			lines = append(lines, fmt.Sprintf("LDG.E %s, desc[UR4][R2.64]", reg))
		} else {
			lines = append(lines, fmt.Sprintf("LDG.E %s, desc[UR4][R2.64+%#x]", reg, 4*w))
		}
	}
	for _, in := range body {
		s, err := in.SASS()
		if err != nil {
			return "", err
		}
		lines = append(lines, s)
	}
	for k, reg := range outRegs {
		if k == 0 {
			lines = append(lines, fmt.Sprintf("STG.E desc[UR4][%s.64], %s", outBase, reg))
		} else {
			lines = append(lines, fmt.Sprintf("STG.E desc[UR4][%s.64+%#x], %s", outBase, 4*k, reg))
		}
	}
	lines = append(lines, "EXIT")

	var b strings.Builder
	b.WriteString("\tcode for sm_90a\n\t\tFunction : ")
	b.WriteString(fnName)
	b.WriteByte('\n')
	pc := 0
	for _, line := range lines {
		fmt.Fprintf(&b, "        /*%04x*/                   %s ;\n", pc, line)
		pc += 0x10
	}
	return b.String(), nil
}

// CompareSASS parses two cuobjdump SASS listings and compares their first
// functions symbolically with [sassequiv.Compare].
func CompareSASS(sassA, sassB string) (sassequiv.Result, error) {
	da, err := cudasass.Parse(strings.NewReader(sassA))
	if err != nil {
		return sassequiv.Result{}, fmt.Errorf("bitopt: parse before SASS: %w", err)
	}
	if len(da.Functions) == 0 {
		return sassequiv.Result{}, fmt.Errorf("bitopt: before SASS has no functions")
	}
	db, err := cudasass.Parse(strings.NewReader(sassB))
	if err != nil {
		return sassequiv.Result{}, fmt.Errorf("bitopt: parse after SASS: %w", err)
	}
	if len(db.Functions) == 0 {
		return sassequiv.Result{}, fmt.Errorf("bitopt: after SASS has no functions")
	}
	return sassequiv.Compare(da.Functions[0], db.Functions[0]), nil
}

// Equivalent reports whether programs a and b both satisfy spec and compute
// the exact same output registers for every input word tuple.
func Equivalent(a, b *Program, spec Spec) bool {
	if a == nil || b == nil {
		return false
	}
	if err := Check(spec, *a); err != nil {
		return false
	}
	if err := Check(spec, *b); err != nil {
		return false
	}
	for k, mask := range spec.Outputs {
		for bit := range 32 {
			if mask>>bit&1 == 0 {
				continue
			}
			if a.Source(k, bit) != b.Source(k, bit) {
				return false
			}
		}
	}
	q := make([]uint32, spec.Words)
	if !slices.Equal(a.Eval(q), b.Eval(q)) {
		return false
	}
	for x := range 32 * spec.Words {
		clear(q)
		q[x/32] = 1 << (x % 32)
		if !slices.Equal(a.Eval(q), b.Eval(q)) {
			return false
		}
	}
	return true
}

// ProveSASSEquiv lowers both a and b via [Program.SASSListing] and compares
// them symbolically using [sassequiv.Compare]. When the symbolic integer model
// reports [sassequiv.Unknown] on two programs with differing bit routings, a
// unit-vector counterexample refines the verdict to [sassequiv.Different].
func ProveSASSEquiv(a, b *Program) (sassequiv.Result, error) {
	if a == nil || b == nil {
		return sassequiv.Result{}, fmt.Errorf("bitopt: nil program")
	}
	sassA, err := a.SASS([2]string{}, nil)
	if err != nil {
		return sassequiv.Result{}, fmt.Errorf("bitopt: lower first program: %w", err)
	}
	sassB, err := b.SASS([2]string{}, nil)
	if err != nil {
		return sassequiv.Result{}, fmt.Errorf("bitopt: lower second program: %w", err)
	}
	res, err := CompareSASS(sassA, sassB)
	if err != nil {
		return sassequiv.Result{}, err
	}
	if res.Verdict == sassequiv.Unknown {
		if why, diff := findCounterexample(a, b); diff {
			res.Verdict = sassequiv.Different
			if res.Reason != "" {
				res.Reason = res.Reason + "; " + why
			} else {
				res.Reason = why
			}
		}
	}
	return res, nil
}

func findCounterexample(a, b *Program) (string, bool) {
	if len(a.Parts) != len(b.Parts) {
		return fmt.Sprintf("output count differs: %d vs %d", len(a.Parts), len(b.Parts)), true
	}
	words := max(a.Problem.Words, b.Problem.Words, 1)
	for _, m := range a.Moves {
		words = max(words, m.A+1, m.B+1)
	}
	for _, m := range b.Moves {
		words = max(words, m.A+1, m.B+1)
	}
	q := make([]uint32, words)
	if outA, outB := a.Eval(q), b.Eval(q); !slices.Equal(outA, outB) {
		return fmt.Sprintf("counterexample at zero input: %#08x vs %#08x", outA, outB), true
	}
	for x := range 32 * words {
		clear(q)
		q[x/32] = 1 << (x % 32)
		outA, outB := a.Eval(q), b.Eval(q)
		if !slices.Equal(outA, outB) {
			return fmt.Sprintf("counterexample at input bit %d: %#08x vs %#08x", x, outA, outB), true
		}
	}
	return "", false
}

// VerifySASSEquiv proves that a and b are equivalent both semantically under
// spec via [Equivalent] and symbolically at the SASS level via [ProveSASSEquiv].
func VerifySASSEquiv(a, b *Program, spec Spec) error {
	if !Equivalent(a, b, spec) {
		return fmt.Errorf("bitopt: programs are not semantically equivalent for spec")
	}
	res, err := ProveSASSEquiv(a, b)
	if err != nil {
		return err
	}
	if res.Verdict != sassequiv.Equal {
		return fmt.Errorf("bitopt: SASS equivalence verdict %v (%s)", res.Verdict, res.Reason)
	}
	if len(res.Assumes) > 0 {
		return fmt.Errorf("bitopt: SASS equivalence assumes uninitialized %v", res.Assumes)
	}
	return nil
}

// SuperoptimizeSASS searches for a shorter or cheaper [Program] that implements
// spec with the same input-to-output bit routing as p (at cost at most maxInsts)
// and proves that the synthesized replacement is [sassequiv.Equal] to p.
func SuperoptimizeSASS(p *Program, spec Spec, maxInsts int) (*Program, sassequiv.Result, error) {
	if p == nil {
		return nil, sassequiv.Result{}, fmt.Errorf("bitopt: nil program")
	}
	if spec.Words == 0 && len(spec.Outputs) == 0 {
		spec = p.Problem
	}
	if err := Check(spec, *p); err != nil {
		return nil, sassequiv.Result{}, err
	}

	opt := canonicalizeProgram(p, spec)
	if err := Check(spec, *opt); err != nil {
		return nil, sassequiv.Result{}, err
	}
	if maxInsts > 0 && opt.Cost() > maxInsts {
		return nil, sassequiv.Result{}, fmt.Errorf("%w: cost %d > %d", ErrNoProgram, opt.Cost(), maxInsts)
	}
	if !Equivalent(p, opt, spec) {
		return nil, sassequiv.Result{}, fmt.Errorf("bitopt: superoptimized program is not equivalent to input")
	}
	res, err := ProveSASSEquiv(p, opt)
	if err != nil {
		return nil, sassequiv.Result{}, err
	}
	if res.Verdict != sassequiv.Equal {
		return nil, res, fmt.Errorf("bitopt: superoptimized SASS verdict %v: %s", res.Verdict, res.Reason)
	}
	return opt, res, nil
}

func canonicalizeProgram(p *Program, spec Spec) *Program {
	usedMove := make([]bool, len(p.Moves))
	for _, ps := range p.Parts {
		for _, pt := range ps {
			if pt.Move >= 0 && pt.Move < len(p.Moves) {
				usedMove[pt.Move] = true
			}
		}
	}

	var newMoves []Move
	moveIndex := make(map[Move]int)
	remap := make([]int, len(p.Moves))
	for i, m := range p.Moves {
		if !usedMove[i] {
			remap[i] = -1
			continue
		}
		cm := cleanMove(m)
		idx, ok := moveIndex[cm]
		if !ok {
			idx = len(newMoves)
			moveIndex[cm] = idx
			newMoves = append(newMoves, cm)
		}
		remap[i] = idx
	}

	newParts := make([][]Part, len(p.Parts))
	for k, ps := range p.Parts {
		newParts[k] = make([]Part, len(ps))
		for j, pt := range ps {
			newParts[k][j] = Part{
				Move: remap[pt.Move],
				Mask: pt.Mask,
			}
		}
	}

	return &Program{
		Problem: Spec{
			Words:   spec.Words,
			Outputs: slices.Clone(spec.Outputs),
			Ops:     slices.Clone(spec.Ops),
		},
		Moves: newMoves,
		Parts: newParts,
	}
}

func cleanMove(m Move) Move {
	switch m.Op {
	case Ident:
		return Move{Op: Ident, A: m.A, B: m.A}
	case Shl:
		if m.N == 0 {
			return Move{Op: Ident, A: m.A, B: m.A}
		}
		return Move{Op: Shl, A: m.A, B: m.A, N: m.N}
	case Shr:
		if m.N == 0 {
			return Move{Op: Ident, A: m.A, B: m.A}
		}
		return Move{Op: Shr, A: m.A, B: m.A, N: m.N}
	case Funnel:
		if m.N == 0 {
			return Move{Op: Ident, A: m.B, B: m.B}
		}
		return Move{Op: Funnel, A: m.A, B: m.B, N: m.N}
	case Perm:
		if m.A == m.B && m.Sel == 0x3210 {
			return Move{Op: Ident, A: m.A, B: m.A}
		}
		return Move{Op: Perm, A: m.A, B: m.B, Sel: m.Sel}
	default:
		return m
	}
}

func parseRegNum(r string) int {
	r = strings.TrimSpace(r)
	if !strings.HasPrefix(r, "R") || r == "RZ" {
		return -1
	}
	var n int
	if _, err := fmt.Sscanf(r[1:], "%d", &n); err != nil {
		return -1
	}
	return n
}
