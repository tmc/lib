package cudasass

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// This file implements step 4 of lane 34: the feasibility gate for re-encoding.
//
// The question the gate answers is not whether an instruction's bytes can be
// captured and replayed -- [Encoding] already does that, and
// TestEncodingRoundTrip already proves nvdisasm reads them back. It is whether
// a *new* encoding can be produced from the disassembled text, which is what a
// mutation would have to do. [ReEncode] therefore rebuilds each encoding from
// the opcode, predicate and operands nvdisasm printed, and the gate compares
// the result with the bytes nvdisasm printed beside them.
//
// The one thing [ReEncode] does not take from the text is the control code,
// because the text does not carry it. That is a finding, not a shortcut: see
// [Verdict] and testdata/roundtrip/README.md.

// Bit positions of the operand fields shared by the covered forms, observed on
// sm_121a with nvdisasm 13.4 by flipping each bit of a known-good encoding and
// disassembling the result. Word bits 0-63 are Lo, 64-127 are Hi.
//
//	Lo 0-11    opcode
//	Lo 12-14   guarding predicate register, 7 meaning none
//	Lo 15      guarding predicate negated
//	Lo 16-23   destination register
//	Lo 24-31   first source register
//	Lo 32-39   second source register, uniform-register numbering in the
//	           uniform forms
//	Hi 8-15    special-register selector, S2R only
const (
	fOpcode   = 0
	fOpcodeW  = 12
	fPred     = 12
	fPredW    = 3
	fPredNeg  = 15
	fDest     = 16
	fSrcA     = 24
	fSrcB     = 32
	fRegW     = 8
	fSpecial  = 8 // within Hi
	fSpecialW = 8
)

// specialRegisters maps the special-register operand names that appear in the
// covered fixtures to their observed selector values. The selector space is
// far larger; only the entries below were observed and only they are covered.
var specialRegisters = map[string]uint64{
	"SR_LANEID":  0x00,
	"SR_CLOCK":   0x01,
	"SR_TID":     0x20,
	"SR_TID.X":   0x21,
	"SR_TID.Y":   0x22,
	"SR_TID.Z":   0x23,
	"SR_CTAID.X": 0x25,
	"SR_CTAID.Y": 0x26,
	"SR_CTAID.Z": 0x27,
}

// A form is one instruction form whose every bit this model accounts for.
type form struct {
	// name is the form as it is reported in a verdict's covered subset.
	name string
	// opcode is the value of the twelve opcode bits.
	opcode uint64
	// match reports whether inst is of this form.
	match func(inst Instruction) bool
	// encode builds the encoding from inst's text and the control code c.
	encode func(f *form, inst Instruction, c ControlCode) (Encoding, error)
}

func put(w uint64, shift, width uint, v uint64) uint64 {
	mask := uint64(1)<<width - 1
	return w&^(mask<<shift) | (v&mask)<<shift
}

// predicateBits returns the guarding-predicate field for inst.
func predicateBits(inst Instruction) (reg, neg uint64, err error) {
	p := strings.TrimSpace(inst.Predicate)
	if p == "" {
		return PTNum, 0, nil // unguarded is encoded as @PT
	}
	p = strings.TrimPrefix(p, "@")
	if strings.HasPrefix(p, "!") {
		neg, p = 1, p[1:]
	}
	r, ok := parseRegister(p)
	if !ok || r.Class != Pred {
		return 0, 0, fmt.Errorf("unsupported predicate %q", inst.Predicate)
	}
	return uint64(r.Num), neg, nil
}

// gprOperand returns the register number an operand token names, requiring it
// to be a plain register of class want.
func gprOperand(tok string, want RegisterClass) (uint64, error) {
	r, ok := parseRegister(tok)
	if !ok {
		return 0, fmt.Errorf("operand %q is not a register", tok)
	}
	if r.Class != want {
		return 0, fmt.Errorf("operand %q is %s, want %s", tok, r.Class, want)
	}
	// A .reuse suffix is carried by the control code, not by the operand.
	return uint64(r.Num), nil
}

// forms is the covered instruction subset. It is deliberately two forms wide:
// step 4 asks for a narrow subset and a verdict about it, not for an
// assembler. Widening it to manufacture a pass is the outcome the gate exists
// to prevent.
var forms = []*form{
	{
		// NOP takes no operands and no guarding predicate in the fixtures;
		// all 21 occurrences are bit-identical, so the form's only variable
		// part is its control code.
		name:   "NOP",
		opcode: 0x918,
		match: func(inst Instruction) bool {
			return BaseOpcode(inst.Opcode) == "NOP" && strings.TrimSpace(inst.Operands) == ""
		},
		encode: func(f *form, inst Instruction, c ControlCode) (Encoding, error) {
			pred, neg, err := predicateBits(inst)
			if err != nil {
				return Encoding{}, err
			}
			lo := put(0, fOpcode, fOpcodeW, f.opcode)
			lo = put(lo, fPred, fPredW, pred)
			lo = put(lo, fPredNeg, 1, neg)
			return EncodeControl(Encoding{Lo: lo}, c, SM121a)
		},
	},
	{
		// EXIT takes no operands but is commonly guarded, which is what makes
		// it worth covering: it exercises the predicate field against real
		// compiler output rather than against a synthesized encoding.
		name:   "EXIT",
		opcode: 0x94d,
		match: func(inst Instruction) bool {
			return BaseOpcode(inst.Opcode) == "EXIT" && strings.TrimSpace(inst.Operands) == ""
		},
		encode: func(f *form, inst Instruction, c ControlCode) (Encoding, error) {
			pred, neg, err := predicateBits(inst)
			if err != nil {
				return Encoding{}, err
			}
			lo := put(0, fOpcode, fOpcodeW, f.opcode)
			lo = put(lo, fPred, fPredW, pred)
			lo = put(lo, fPredNeg, 1, neg)
			// Hi bits 23-25 are set on every EXIT observed in the fixtures and
			// lie outside the control field.
			return EncodeControl(Encoding{Lo: lo, Hi: 0x03800000}, c, SM121a)
		},
	},
	{
		name:   "S2R Rd, SR",
		opcode: 0x919,
		match: func(inst Instruction) bool {
			if BaseOpcode(inst.Opcode) != "S2R" {
				return false
			}
			ops := splitOperands(inst.Operands)
			if len(ops) != 2 {
				return false
			}
			_, ok := specialRegisters[ops[1]]
			return ok
		},
		encode: func(f *form, inst Instruction, c ControlCode) (Encoding, error) {
			ops := splitOperands(inst.Operands)
			dest, err := gprOperand(ops[0], GPR)
			if err != nil {
				return Encoding{}, err
			}
			sel, ok := specialRegisters[ops[1]]
			if !ok {
				return Encoding{}, fmt.Errorf("special register %q is not in the covered subset", ops[1])
			}
			pred, neg, err := predicateBits(inst)
			if err != nil {
				return Encoding{}, err
			}
			lo := uint64(0)
			lo = put(lo, fOpcode, fOpcodeW, f.opcode)
			lo = put(lo, fPred, fPredW, pred)
			lo = put(lo, fPredNeg, 1, neg)
			lo = put(lo, fDest, fRegW, dest)
			hi := put(0, fSpecial, fSpecialW, sel)
			return EncodeControl(Encoding{Lo: lo, Hi: hi}, c, SM121a)
		},
	},
	{
		name:   "IADD Rd, Ra, URb",
		opcode: 0xc35,
		match: func(inst Instruction) bool {
			if BaseOpcode(inst.Opcode) != "IADD" {
				return false
			}
			ops := splitOperands(inst.Operands)
			if len(ops) != 3 {
				return false
			}
			a, ok1 := parseRegister(ops[1])
			b, ok2 := parseRegister(ops[2])
			return ok1 && ok2 && a.Class == GPR && b.Class == UGPR
		},
		encode: func(f *form, inst Instruction, c ControlCode) (Encoding, error) {
			ops := splitOperands(inst.Operands)
			dest, err := gprOperand(ops[0], GPR)
			if err != nil {
				return Encoding{}, err
			}
			srcA, err := gprOperand(ops[1], GPR)
			if err != nil {
				return Encoding{}, err
			}
			srcB, err := gprOperand(ops[2], UGPR)
			if err != nil {
				return Encoding{}, err
			}
			pred, neg, err := predicateBits(inst)
			if err != nil {
				return Encoding{}, err
			}
			lo := uint64(0)
			lo = put(lo, fOpcode, fOpcodeW, f.opcode)
			lo = put(lo, fPred, fPredW, pred)
			lo = put(lo, fPredNeg, 1, neg)
			lo = put(lo, fDest, fRegW, dest)
			lo = put(lo, fSrcA, fRegW, srcA)
			lo = put(lo, fSrcB, fRegW, srcB)
			// Hi carries the uniform-datapath selector bits this form always
			// sets, observed constant across every IADD Rd, Ra, URb in the
			// fixtures, plus the control code.
			hi := uint64(0x0f8e0000)
			return EncodeControl(Encoding{Lo: lo, Hi: hi}, c, SM121a)
		},
	},
}

// Covered returns the name of the form inst belongs to, and false if inst is
// outside the covered subset.
func Covered(inst Instruction) (string, bool) {
	for _, f := range forms {
		if f.match(inst) {
			return f.name, true
		}
	}
	return "", false
}

// ReEncode rebuilds inst's encoding from its disassembled text.
//
// The opcode, guarding predicate and register operands come from the text. The
// control code does not: nvdisasm prints no control field, so it is decoded
// from inst's captured encoding and re-applied. That asymmetry is the central
// result of the gate rather than an implementation convenience -- see
// [Verdict.ControlCodeIsOutOfBand].
//
// ReEncode returns an error for any instruction outside the covered subset. It
// never falls back to copying the captured bytes, which would make the round
// trip vacuous.
func ReEncode(inst Instruction, arch Arch) (Encoding, error) {
	if arch != SM121a {
		return Encoding{}, fmt.Errorf("re-encoding: %w", &ControlDecodeError{Arch: arch})
	}
	c, err := DecodeControl(inst.Encoding, arch)
	if err != nil {
		return Encoding{}, fmt.Errorf("decode control for %s at %#x: %w", inst.Opcode, inst.Address, err)
	}
	for _, f := range forms {
		if f.match(inst) {
			return f.encode(f, inst, c)
		}
	}
	return Encoding{}, fmt.Errorf("%s at %#x is outside the covered subset", inst.Opcode, inst.Address)
}

// A RoundTripResult is one instruction's outcome in the feasibility gate.
type RoundTripResult struct {
	Address    uint64
	Opcode     string
	Form       string   // the covered form, empty if uncovered
	Want, Got  Encoding // the captured encoding and the re-encoded one
	Identical  bool
	Difference Encoding // bits that differ, zero when Identical
	Err        string   // why re-encoding did not run at all
}

// A Verdict is the outcome of the step 4 feasibility gate on one function.
//
// A negative verdict is a complete and successful outcome for this lane. It
// must be recorded, and it blocks steps 5 and 6; see [Verdict.Blocks].
type Verdict struct {
	Arch     Arch
	Function string
	Results  []RoundTripResult

	// ControlCodeIsOutOfBand records that the disassembled text does not
	// determine the encoding, so a re-encoder must carry control codes
	// separately. This is true on sm_121a and is a property of nvdisasm's
	// output, not of this package.
	ControlCodeIsOutOfBand bool
}

// CoveredForms returns the distinct forms the gate re-encoded, sorted.
func (v Verdict) CoveredForms() []string {
	seen := map[string]bool{}
	for _, r := range v.Results {
		if r.Form != "" {
			seen[r.Form] = true
		}
	}
	out := make([]string, 0, len(seen))
	for f := range seen {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

// Counts returns how many instructions were covered, how many of those
// round-tripped byte-identically, and how many were outside the subset.
func (v Verdict) Counts() (covered, identical, uncovered int) {
	for _, r := range v.Results {
		if r.Form == "" {
			uncovered++
			continue
		}
		covered++
		if r.Identical {
			identical++
		}
	}
	return covered, identical, uncovered
}

// Feasible reports whether every covered instruction round-tripped
// byte-identically.
//
// It is false when nothing was covered: a gate that covered no instruction has
// demonstrated nothing, and reporting that as success is the failure mode the
// gate exists to prevent.
func (v Verdict) Feasible() bool {
	covered, identical, _ := v.Counts()
	return covered > 0 && covered == identical
}

// Blocks reports whether steps 5 and 6 of lane 34 must not run. It is the
// negation of [Verdict.Feasible], stated separately because it is the
// scheduling fact the lane prompt turns on.
func (v Verdict) Blocks() bool { return !v.Feasible() }

// Failures returns the covered instructions that did not round-trip.
func (v Verdict) Failures() []RoundTripResult {
	var out []RoundTripResult
	for _, r := range v.Results {
		if r.Form != "" && !r.Identical {
			out = append(out, r)
		}
	}
	return out
}

// String renders the verdict in the form the handoff records.
func (v Verdict) String() string {
	covered, identical, uncovered := v.Counts()
	b := new(strings.Builder)
	verdict := "NEGATIVE"
	if v.Feasible() {
		verdict = "POSITIVE"
	}
	fmt.Fprintf(b, "%s on %s function %q: %d/%d covered instructions byte-identical, %d outside the subset\n",
		verdict, v.Arch, v.Function, identical, covered, uncovered)
	fmt.Fprintf(b, "covered forms: %s\n", strings.Join(v.CoveredForms(), "; "))
	if v.ControlCodeIsOutOfBand {
		b.WriteString("control codes are not recoverable from the disassembled text and must be carried out of band\n")
	}
	for _, r := range v.Failures() {
		fmt.Fprintf(b, "  %#x %s: want %v, got %v, differing bits %v\n", r.Address, r.Opcode, r.Want, r.Got, r.Difference)
	}
	return b.String()
}

// RoundTrip runs the feasibility gate over fn: for every instruction it
// re-encodes from the text and compares with the captured bytes.
func RoundTrip(fn *Function, arch Arch) Verdict {
	v := Verdict{Arch: arch, ControlCodeIsOutOfBand: arch == SM121a}
	if fn == nil {
		return v
	}
	v.Function = fn.Name
	for _, inst := range fn.Instructions {
		r := RoundTripResult{Address: inst.Address, Opcode: inst.Opcode, Want: inst.Encoding}
		name, ok := Covered(inst)
		if !ok {
			v.Results = append(v.Results, r)
			continue
		}
		r.Form = name
		got, err := ReEncode(inst, arch)
		if err != nil {
			r.Err = err.Error()
			v.Results = append(v.Results, r)
			continue
		}
		r.Got = got
		r.Identical = got == inst.Encoding
		if !r.Identical {
			r.Difference = Encoding{Lo: got.Lo ^ inst.Encoding.Lo, Hi: got.Hi ^ inst.Encoding.Hi}
		}
		v.Results = append(v.Results, r)
	}
	return v
}

// ---------------------------------------------------------------------------
// Step 5: the CUBIN patch path.
//
// Step 4 established that a covered instruction's bytes can be rebuilt from
// its text. Step 5 puts rebuilt bytes back into the CUBIN they came from. Two
// things make that narrower than it sounds.
//
// The first is the control code. nvdisasm prints none, so the schedule cannot
// be recomputed from the text; it is carried out of band from the original
// encoding. Carrying it is sound exactly when the mutation leaves the
// dependency graph unchanged, because then the original schedule is still the
// correct schedule. Every mutation is therefore classified as
// scheduling-invariant or rejected, and the rejection happens when the patch
// is built -- not at load, and not by the device. See [CheckMutation].
//
// The second is that a patch is a byte edit and a byte edit lands wherever it
// is told to. A patch written at the wrong offset produces a CUBIN that still
// loads, so no load test can catch it. [CubinImage.Apply] therefore refuses
// unless the bytes already at the target offset are the ones the plan was
// built from.
// ---------------------------------------------------------------------------

// A Mutation proposes replacing one instruction of a function.
//
// Replacement carries text only: opcode, predicate and operands. Its Encoding
// field is ignored, because the replacement's schedule is not the caller's to
// choose -- [CheckMutation] takes the control code from the instruction being
// replaced, which is the whole reason the mutation class is restricted to
// those that leave the schedule correct.
type Mutation struct {
	Index       int         // index into the function's instruction slice
	Replacement Instruction // the instruction to put there
}

// A MutationPlan is a mutation that passed every patch-time gate. It is the
// only thing [CubinImage.Apply] accepts.
type MutationPlan struct {
	Function string
	Index    int
	Address  uint64
	FormFrom string
	FormTo   string
	Before   Encoding
	After    Encoding
	Control  ControlCode // carried unchanged from Before; see [Verdict.ControlCodeIsOutOfBand]
}

// CheckMutation decides whether m may be applied to fn, and returns the plan
// if it may.
//
// liveOut is the caller's claim about which registers the containing basic
// block's successors read. It is required: a mutation is checked against
// liveness, and a liveness set that was guessed is worse than none. nil is
// accepted only for a block that ends in EXIT, where it is the true answer.
//
// The gates, in order, each fail closed:
//
//   - both the original and the replacement are of a covered form;
//   - the original re-encodes byte-identically, so its bits are understood;
//   - the replacement's encoding carries the original's control code exactly;
//   - the basic block's RAW, WAR, WAW, memory, fence and barrier edges are
//     unchanged by the substitution;
//   - no register the replacement newly defines is live at that point, and no
//     register it stops defining is live at that point. A register rename can
//     change an edge, and a rename onto a register that aliases something live
//     changes the program without changing any edge inside the block, which is
//     the case this check and not the edge comparison catches.
func CheckMutation(fn *Function, arch Arch, m Mutation, liveOut LiveSet) (MutationPlan, error) {
	if arch != SM121a {
		return MutationPlan{}, fmt.Errorf("patch path: %w", &ControlDecodeError{Arch: arch})
	}
	if fn == nil {
		return MutationPlan{}, fmt.Errorf("no function")
	}
	if m.Index < 0 || m.Index >= len(fn.Instructions) {
		return MutationPlan{}, fmt.Errorf("instruction index %d outside function %q of %d instructions", m.Index, fn.Name, len(fn.Instructions))
	}
	orig := fn.Instructions[m.Index]
	if !orig.Encoding.Present() {
		return MutationPlan{}, fmt.Errorf("instruction %d at %#x: %w", m.Index, orig.Address, &ControlDecodeError{Arch: arch, MissingEncoding: true})
	}
	fromForm, ok := Covered(orig)
	if !ok {
		return MutationPlan{}, fmt.Errorf("%s at %#x is outside the covered subset", orig.Opcode, orig.Address)
	}
	rep := m.Replacement
	rep.Address = orig.Address
	rep.Encoding = orig.Encoding // the out-of-band carry of the schedule
	toForm, ok := Covered(rep)
	if !ok {
		return MutationPlan{}, fmt.Errorf("replacement %s %s is outside the covered subset", rep.Opcode, rep.Operands)
	}
	again, err := ReEncode(orig, arch)
	if err != nil {
		return MutationPlan{}, fmt.Errorf("re-encoding the original: %w", err)
	}
	if again != orig.Encoding {
		return MutationPlan{}, fmt.Errorf("instruction %d at %#x does not round-trip byte-identically (%v vs %v); its bits are not understood well enough to patch",
			m.Index, orig.Address, again, orig.Encoding)
	}
	control, err := DecodeControl(orig.Encoding, arch)
	if err != nil {
		return MutationPlan{}, fmt.Errorf("decode control at %#x: %w", orig.Address, err)
	}
	after, err := ReEncode(rep, arch)
	if err != nil {
		return MutationPlan{}, fmt.Errorf("re-encoding the replacement: %w", err)
	}
	gotControl, err := DecodeControl(after, arch)
	if err != nil {
		return MutationPlan{}, fmt.Errorf("decode replacement control: %w", err)
	}
	if gotControl != control {
		return MutationPlan{}, fmt.Errorf("the replacement's control code is %v, want the original's %v", gotControl, control)
	}

	block, pos, err := blockContaining(fn, m.Index)
	if err != nil {
		return MutationPlan{}, err
	}
	if liveOut == nil && !endsInExit(block) {
		return MutationPlan{}, fmt.Errorf("block %#x-%#x does not end in EXIT, so liveOut may not be omitted", block.Start, block.End)
	}
	if err := graphUnchanged(block, pos, rep, arch); err != nil {
		return MutationPlan{}, err
	}
	if err := defsStayDead(block, pos, rep, liveOut, arch); err != nil {
		return MutationPlan{}, err
	}
	return MutationPlan{
		Function: fn.Name,
		Index:    m.Index,
		Address:  orig.Address,
		FormFrom: fromForm,
		FormTo:   toForm,
		Before:   orig.Encoding,
		After:    after,
		Control:  control,
	}, nil
}

// blockContaining returns the basic block holding the instruction at index i
// of fn, and i's position within it.
func blockContaining(fn *Function, i int) (BasicBlock, int, error) {
	n := 0
	for _, b := range BasicBlocks(fn) {
		if i < n+len(b.Instructions) {
			return b, i - n, nil
		}
		n += len(b.Instructions)
	}
	return BasicBlock{}, 0, fmt.Errorf("instruction %d is in no basic block of %q", i, fn.Name)
}

func endsInExit(b BasicBlock) bool {
	if len(b.Instructions) == 0 {
		return false
	}
	return BaseOpcode(b.Instructions[len(b.Instructions)-1].Opcode) == "EXIT"
}

// mutatedBlock returns b with the instruction at pos replaced by rep.
func mutatedBlock(b BasicBlock, pos int, rep Instruction) BasicBlock {
	insts := make([]Instruction, len(b.Instructions))
	copy(insts, b.Instructions)
	insts[pos] = rep
	return BasicBlock{Start: b.Start, End: b.End, Instructions: insts}
}

// graphUnchanged reports an error unless the dependency edges of b are
// identical before and after the substitution, register for register. A
// mutation that adds, drops or re-registers an edge perturbs scheduling, and
// the schedule this path carries out of band is the original one.
func graphUnchanged(b BasicBlock, pos int, rep Instruction, arch Arch) error {
	before, err := Dependencies(b, arch)
	if err != nil {
		return fmt.Errorf("dependencies of the original block: %w", err)
	}
	after, err := Dependencies(mutatedBlock(b, pos, rep), arch)
	if err != nil {
		return fmt.Errorf("dependencies of the mutated block: %w", err)
	}
	added, dropped := edgeDiff(before, after)
	if len(added) == 0 && len(dropped) == 0 {
		return nil
	}
	var parts []string
	for _, e := range dropped {
		parts = append(parts, "drops "+edgeString(e))
	}
	for _, e := range added {
		parts = append(parts, "adds "+edgeString(e))
	}
	return fmt.Errorf("mutation at position %d perturbs scheduling: it %s; the control code carried from the original is only correct for a mutation that leaves the graph unchanged",
		pos, strings.Join(parts, ", "))
}

func edgeString(e Edge) string {
	return fmt.Sprintf("%s %d->%d on %s", e.Kind, e.From, e.To, e.Reg)
}

// edgeDiff returns the edges in after that are not in before, and those in
// before that are not in after.
func edgeDiff(before, after []Edge) (added, dropped []Edge) {
	in := func(set []Edge, e Edge) bool {
		for _, s := range set {
			if s == e {
				return true
			}
		}
		return false
	}
	for _, e := range after {
		if !in(before, e) {
			added = append(added, e)
		}
	}
	for _, e := range before {
		if !in(after, e) {
			dropped = append(dropped, e)
		}
	}
	return added, dropped
}

// defsStayDead reports an error if the substitution changes which live
// registers the instruction defines.
//
// The edge comparison in graphUnchanged only sees uses inside the block. A
// rename onto a register that nothing in the block reads but a successor does
// adds no edge and is still wrong, so liveness is checked separately against
// the set live immediately after the instruction.
func defsStayDead(b BasicBlock, pos int, rep Instruction, liveOut LiveSet, arch Arch) error {
	live, err := Liveness(b, liveOut, arch)
	if err != nil {
		return fmt.Errorf("liveness of the block: %w", err)
	}
	origDefs, _, ok := DefsUses(b.Instructions[pos])
	if !ok {
		return fmt.Errorf("instruction %d at %#x is outside the covered subset", pos, b.Instructions[pos].Address)
	}
	repDefs, _, ok := DefsUses(rep)
	if !ok {
		return fmt.Errorf("replacement %s is outside the covered subset", rep.Opcode)
	}
	has := func(rs []Register, r Register) bool {
		for _, x := range rs {
			if x == r {
				return true
			}
		}
		return false
	}
	after := live[pos]
	for _, d := range repDefs {
		if !has(origDefs, d) && after.Contains(d) {
			return fmt.Errorf("mutation at position %d writes %s, which is live after it; the original did not write it", pos, d)
		}
	}
	for _, d := range origDefs {
		if !has(repDefs, d) && after.Contains(d) {
			return fmt.Errorf("mutation at position %d stops writing %s, which is live after it", pos, d)
		}
	}
	return nil
}

// Errors the patch path returns. Each is a refusal, not a warning: the path
// emits a patched image or it emits nothing.
var (
	// ErrTextSectionAbsent means the image has no text section for the kernel.
	ErrTextSectionAbsent = errors.New("no text section for the kernel")
	// ErrSectionNotCode means the named section is not allocated executable
	// code, so its bytes are not instructions.
	ErrSectionNotCode = errors.New("section is not executable code")
	// ErrSectionNotInFile means the section occupies no file bytes, which a
	// NOBITS section does not.
	ErrSectionNotInFile = errors.New("section has no bytes in the file")
	// ErrPatchUnaligned means the target offset is not on a 16-byte
	// instruction boundary.
	ErrPatchUnaligned = errors.New("patch offset is not instruction-aligned")
	// ErrPatchOutOfRange means the target instruction lies outside the section.
	ErrPatchOutOfRange = errors.New("patch lies outside the section")
	// ErrOriginalMismatch means the bytes already at the target offset are not
	// the ones the plan was built from. This is the guard against a patch that
	// lands at the wrong offset, which is a failure no load test can catch
	// because a wrong-offset patch loads.
	ErrOriginalMismatch = errors.New("bytes at the patch offset are not the instruction the plan was built from")
	// ErrRelocationOverlap means a relocation applies to the bytes being
	// patched, so the loader will rewrite part of what was written.
	ErrRelocationOverlap = errors.New("a relocation applies to the patched bytes")
	// ErrMirroredText means the image carries a second copy of the patched
	// text in a form this path cannot update. See [CubinImage.Mirrors].
	ErrMirroredText = errors.New("the image carries a second copy of the patched text")
	// ErrLockNotHeld means a device load was requested without the GPU lock.
	ErrLockNotHeld = errors.New("GPU lock not held")
	// ErrNothingPatched means an image was offered for loading that no patch
	// reached, so loading it would qualify nothing.
	ErrNothingPatched = errors.New("no patch was applied")
)

// A CubinImage is a CUBIN held for patching.
type CubinImage struct {
	data []byte
	file *elf.File
}

// OpenCubin parses b as a CUBIN. The bytes are copied, so the caller's slice
// is never modified.
func OpenCubin(b []byte) (*CubinImage, error) {
	data := make([]byte, len(b))
	copy(data, b)
	f, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("parsing cubin: %w", err)
	}
	if f.Machine != elf.EM_CUDA {
		return nil, fmt.Errorf("machine is %v, not %v", f.Machine, elf.EM_CUDA)
	}
	return &CubinImage{data: data, file: f}, nil
}

// TextSectionName returns the conventional text section name for a kernel.
func TextSectionName(kernel string) string { return ".text." + kernel }

// Mirrors returns the sections that carry a second copy of the kernel's text.
//
// CUDA 13.4's ptxas emits, beside every `.text.<kernel>`, a
// `.nv.capmerc.text.<kernel>` in an undocumented form (0x102 bytes beside
// addk's 0x200 bytes of text in testdata/roundtrip). What it encodes has not
// been measured, and patching the text section alone leaves it unchanged.
// Which bytes execute was measured once, on the GB10 host, by flipping one
// byte at stride 16, reloading, launching and comparing the result: 40 of 53
// flips in `.text.<kernel>` failed the launch and 0 of 25 in the capmerc
// section did, so under that driver and launch configuration the executed
// bytes come from `.text`. That says nothing about what the capmerc section
// is for, and nothing here repeats the measurement, so the presence of a
// mirror is reported and the load gate refuses.
func (im *CubinImage) Mirrors(kernel string) []string {
	want := TextSectionName(kernel)
	var out []string
	for _, s := range im.file.Sections {
		if s.Name != want && strings.HasSuffix(s.Name, want) {
			out = append(out, s.Name)
		}
	}
	sort.Strings(out)
	return out
}

// An AppliedPatch records one instruction replaced in the image.
type AppliedPatch struct {
	Address    uint64 // the instruction's address within the section
	FileOffset uint64 // where the bytes were written in the file
	Before     Encoding
	After      Encoding
}

// A PatchedImage is the output of the patch path.
//
// Bytes is a complete CUBIN of exactly the input's length: the path overwrites
// instructions in place and moves nothing, so every section offset, alignment
// and program header in the original still describes the result.
type PatchedImage struct {
	Bytes            []byte
	Kernel           string
	Section          string
	Patches          []AppliedPatch
	UnpatchedMirrors []string // second copies of the text this path did not update
}

// Apply writes the plans into the kernel's text section and returns the
// patched image.
//
// It refuses, before writing anything, if the section is missing, is not
// executable code, has no file bytes, if any target is unaligned or out of
// range, if a relocation applies to a target, or if the bytes already at a
// target are not the ones the plan was built from. The last of these is the
// check that makes a misplaced patch a failure here rather than a CUBIN that
// loads and is wrong.
func (im *CubinImage) Apply(kernel string, plans []MutationPlan) (PatchedImage, error) {
	name := TextSectionName(kernel)
	sec := im.file.Section(name)
	if sec == nil {
		return PatchedImage{}, fmt.Errorf("%w: %s", ErrTextSectionAbsent, name)
	}
	if sec.Flags&elf.SHF_EXECINSTR == 0 || sec.Flags&elf.SHF_ALLOC == 0 {
		return PatchedImage{}, fmt.Errorf("%w: %s has flags %v", ErrSectionNotCode, name, sec.Flags)
	}
	if sec.Type == elf.SHT_NOBITS {
		return PatchedImage{}, fmt.Errorf("%w: %s", ErrSectionNotInFile, name)
	}
	if sec.Offset%instructionSize != 0 || sec.Size%instructionSize != 0 {
		return PatchedImage{}, fmt.Errorf("%w: %s is at %#x for %#x bytes, not a whole number of instructions",
			ErrPatchUnaligned, name, sec.Offset, sec.Size)
	}
	if sec.Offset+sec.Size > uint64(len(im.data)) {
		return PatchedImage{}, fmt.Errorf("%w: %s ends past the end of the image", ErrPatchOutOfRange, name)
	}
	if len(plans) == 0 {
		return PatchedImage{}, ErrNothingPatched
	}
	relocs, err := im.relocations(name)
	if err != nil {
		return PatchedImage{}, err
	}

	out := make([]byte, len(im.data))
	copy(out, im.data)
	var applied []AppliedPatch
	for _, p := range plans {
		if p.Address < sec.Addr {
			return PatchedImage{}, fmt.Errorf("%w: address %#x is below section address %#x", ErrPatchOutOfRange, p.Address, sec.Addr)
		}
		rel := p.Address - sec.Addr
		if rel%instructionSize != 0 {
			return PatchedImage{}, fmt.Errorf("%w: address %#x is %d bytes into an instruction", ErrPatchUnaligned, p.Address, rel%instructionSize)
		}
		if rel+instructionSize > sec.Size {
			return PatchedImage{}, fmt.Errorf("%w: address %#x is past the end of %s", ErrPatchOutOfRange, p.Address, name)
		}
		for _, r := range relocs {
			if r >= rel && r < rel+instructionSize {
				return PatchedImage{}, fmt.Errorf("%w: relocation at %#x falls inside the instruction at %#x", ErrRelocationOverlap, r, p.Address)
			}
		}
		off := sec.Offset + rel
		want := p.Before.Bytes()
		if got := out[off : off+instructionSize]; !bytes.Equal(got, want[:]) {
			return PatchedImage{}, fmt.Errorf("%w: at file offset %#x (%s+%#x) the image holds % x, the plan was built from % x",
				ErrOriginalMismatch, off, name, rel, got, want)
		}
		got := p.After.Bytes()
		copy(out[off:off+instructionSize], got[:])
		applied = append(applied, AppliedPatch{Address: p.Address, FileOffset: off, Before: p.Before, After: p.After})
	}
	if len(out) != len(im.data) {
		return PatchedImage{}, fmt.Errorf("patching changed the image length from %d to %d", len(im.data), len(out))
	}
	return PatchedImage{
		Bytes:            out,
		Kernel:           kernel,
		Section:          name,
		Patches:          applied,
		UnpatchedMirrors: im.Mirrors(kernel),
	}, nil
}

// instructionSize is the size of one sm_121a instruction in bytes.
const instructionSize = 16

// relocations returns the section-relative offsets every relocation that
// targets the named section applies to.
func (im *CubinImage) relocations(target string) ([]uint64, error) {
	var out []uint64
	for _, s := range im.file.Sections {
		if s.Name != ".rel"+target && s.Name != ".rela"+target {
			continue
		}
		data, err := s.Data()
		if err != nil {
			return nil, fmt.Errorf("reading %s: %w", s.Name, err)
		}
		size := 16 // Elf64_Rel
		if s.Type == elf.SHT_RELA {
			size = 24 // Elf64_Rela
		}
		if len(data)%size != 0 {
			return nil, fmt.Errorf("%s is %d bytes, not a whole number of %d-byte entries", s.Name, len(data), size)
		}
		for i := 0; i+size <= len(data); i += size {
			out = append(out, binary.LittleEndian.Uint64(data[i:i+8]))
		}
	}
	return out, nil
}

// Loadable reports whether the image may be offered to a device.
//
// It fails closed on the mirror: an image whose text has a second, unpatched
// copy is refused, because whether the device executes the copy this path
// wrote is exactly what a load does not answer.
func (p PatchedImage) Loadable() error {
	if len(p.Patches) == 0 {
		return ErrNothingPatched
	}
	if len(p.UnpatchedMirrors) > 0 {
		return fmt.Errorf("%w: %s; patching %s alone leaves it stale and no load distinguishes the two",
			ErrMirroredText, strings.Join(p.UnpatchedMirrors, ", "), p.Section)
	}
	return nil
}

// AuthorizeLoad is the host-side gate every device load of a patched image
// passes. It loads nothing itself.
//
// env is the environment lookup, normally os.Getenv. The GPU lock is required
// because a candidate binary of unknown validity can hang or fault the device,
// and the device is shared. The lock must be one an ancestor holds, as
// [cudalock.Inherited] checks: run the load under gpu-lock itself, not under
// "gpu-lock env GPU_LOCK_HELD=1", which replaces the holder's pid with a
// value that names no holder.
func AuthorizeLoad(p PatchedImage, env func(string) string) error {
	if err := p.Loadable(); err != nil {
		return err
	}
	if _, err := cudalock.Inherited(env); err != nil {
		return fmt.Errorf("%w: %w; run under gpu-lock", ErrLockNotHeld, err)
	}
	return nil
}
