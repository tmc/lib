package cudasass

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// This file implements the SASS dependency and control-code model for one
// named architecture, sm_121a. Every encoding fact below was observed on that
// target with nvdisasm 13.4 and is recorded as observed, not as documented:
// NVIDIA publishes no SASS encoding reference. See testdata/roundtrip/README.md
// for the probe that produced them and for the exact commands.
//
// Nothing here generalizes to another SM version. [DecodeControl] refuses any
// architecture it was not calibrated against rather than returning a plausible
// decode, because a control-code field read at the wrong offset is worse than
// an absent one: it looks like a barrier index.

// Arch names the architecture an encoding fact was observed on.
type Arch string

// SM121a is the only architecture this model is calibrated for.
const SM121a Arch = "sm_121a"

// Control-code field positions within the 128-bit instruction word, observed on
// sm_121a. The instruction is stored as two little-endian 64-bit words, Lo then
// Hi, so word bit 105 is Hi bit 41.
//
// Observed by flipping each of the 128 bits of a known-good encoding and
// disassembling the result with nvdisasm --binary SM121a:
//
//	bit 105-108  stall count   flipping bit 105 alone is rejected on every
//	                           instruction probed; see [MaxStall]
//	bit 109      yield         clearing it suppresses the .reuse suffix that
//	                           nvdisasm prints, so reuse is only meaningful
//	                           while this bit is set
//	bit 110-112  write barrier index, 7 meaning none
//	bit 113-115  read barrier index, 7 meaning none
//	bit 116-121  wait mask, one bit per barrier
//	bit 122-125  reuse flags, one per source slot; a flag for a slot the
//	                           instruction does not have is rejected
//	bit 126-127  no observed effect on any instruction probed
const (
	ctrlShift    = 41 // within Hi: word bit 105
	stallWidth   = 4
	yieldShift   = 45
	wrtBarShift  = 46
	readBarShift = 49
	waitShift    = 52
	reuseShift   = 58
	barWidth     = 3
	waitWidth    = 6
	reuseWidth   = 4
)

// NoBarrier is the write- or read-barrier index meaning the instruction sets
// no barrier. It is a real encoded value, not an absence marker.
const NoBarrier = 7

// StallRange is the range of stall values observed to be accepted for one
// instruction form on sm_121a.
//
// The stall field is four bits wide, but the accepted range is not the field
// width and is not the same for every instruction. Sweeping all sixteen values
// of the field, nvdisasm accepts every one of them on NOP and rejects 0 and
// 12-15 on IADD Rd, Ra, URb -- and ptxas itself emits NOP with stall 0. A
// range measured on one instruction is therefore a fact about that
// instruction, not about the architecture, and is recorded per form.
type StallRange struct{ Min, Max uint8 }

// observedStallRanges records what was swept, by covered form name. A form
// absent from this map has not been swept; see [ObservedStallRange].
var observedStallRanges = map[string]StallRange{
	"NOP":              {0, 15},
	"IADD Rd, Ra, URb": {1, 11},
}

// ObservedStallRange returns the stall range swept for a covered form, and
// false if that form has not been swept. Callers must not substitute the range
// of another form: the two measured so far disagree.
func ObservedStallRange(form string) (StallRange, bool) {
	r, ok := observedStallRanges[form]
	return r, ok
}

// A ControlCode is the scheduling control field of one SASS instruction.
//
// Control codes are part of correctness, not metadata. A reordering that
// preserves data dependencies but drops a barrier index, a wait mask or the
// yield flag is wrong, and the hardware does not check it.
type ControlCode struct {
	Stall        uint8 // cycles to stall before issuing the next instruction
	Yield        bool  // clearing this yields the warp; reuse is invalid when clear
	WriteBarrier uint8 // scoreboard barrier this instruction sets on write, or NoBarrier
	ReadBarrier  uint8 // scoreboard barrier this instruction sets on read, or NoBarrier
	WaitMask     uint8 // barriers this instruction waits on, one bit each
	Reuse        uint8 // operand-reuse cache flags, one bit per source slot
}

// Waits reports whether the instruction waits on barrier i.
func (c ControlCode) Waits(i int) bool {
	return i >= 0 && i < 6 && c.WaitMask&(1<<uint(i)) != 0
}

// Valid reports whether every field fits the width observed for it on
// sm_121a. It deliberately does not police the stall value: the accepted range
// is per-instruction, not per-architecture, so a range check belongs to the
// form being encoded. See [ObservedStallRange].
func (c ControlCode) Valid() error {
	if c.Stall >= 1<<stallWidth {
		return fmt.Errorf("stall %d wider than the %d-bit field", c.Stall, stallWidth)
	}
	if c.WriteBarrier > NoBarrier || c.ReadBarrier > NoBarrier {
		return fmt.Errorf("barrier index out of range")
	}
	if c.WaitMask >= 1<<waitWidth {
		return fmt.Errorf("wait mask %#x wider than %d bits", c.WaitMask, waitWidth)
	}
	if c.Reuse >= 1<<reuseWidth {
		return fmt.Errorf("reuse flags %#x wider than %d bits", c.Reuse, reuseWidth)
	}
	return nil
}

func (c ControlCode) String() string {
	b := new(strings.Builder)
	fmt.Fprintf(b, "stall=%d", c.Stall)
	if c.Yield {
		b.WriteString(" yield")
	}
	if c.WriteBarrier != NoBarrier {
		fmt.Fprintf(b, " wr=%d", c.WriteBarrier)
	}
	if c.ReadBarrier != NoBarrier {
		fmt.Fprintf(b, " rd=%d", c.ReadBarrier)
	}
	if c.WaitMask != 0 {
		fmt.Fprintf(b, " wait=%#02x", c.WaitMask)
	}
	if c.Reuse != 0 {
		fmt.Fprintf(b, " reuse=%#x", c.Reuse)
	}
	return b.String()
}

// A ControlDecodeError reports a refused static control-code decode. It does
// not report a measured hardware capability.
type ControlDecodeError struct {
	Arch Arch // requested architecture

	// MissingEncoding is true when arch is calibrated but no encoding was captured.
	MissingEncoding bool
}

func (e *ControlDecodeError) Error() string {
	if e.MissingEncoding {
		return fmt.Sprintf("no encoding captured for control-code decode on %s", e.Arch)
	}
	return fmt.Sprintf("control-code decode is calibrated for %s only, not %q", SM121a, e.Arch)
}

// DecodeControl returns the control code carried by e on arch.
//
// It returns a [ControlDecodeError] for any architecture other than [SM121a],
// checked before a missing encoding. A present encoding can carry an all-zero
// ControlCode; nil error distinguishes this known value from refusal.
// The field offsets were calibrated on sm_121a only, and an encoding fact
// is not generalized across SM versions:
// decoding sm_90 bits at sm_121a offsets yields a barrier index that is
// structurally well formed and meaningless.
func DecodeControl(e Encoding, arch Arch) (ControlCode, error) {
	if arch != SM121a {
		return ControlCode{}, &ControlDecodeError{Arch: arch}
	}
	if !e.Present() {
		return ControlCode{}, &ControlDecodeError{Arch: arch, MissingEncoding: true}
	}
	return ControlCode{
		Stall:        uint8(e.Hi >> ctrlShift & (1<<stallWidth - 1)),
		Yield:        e.Hi>>yieldShift&1 != 0,
		WriteBarrier: uint8(e.Hi >> wrtBarShift & (1<<barWidth - 1)),
		ReadBarrier:  uint8(e.Hi >> readBarShift & (1<<barWidth - 1)),
		WaitMask:     uint8(e.Hi >> waitShift & (1<<waitWidth - 1)),
		Reuse:        uint8(e.Hi >> reuseShift & (1<<reuseWidth - 1)),
	}, nil
}

// EncodeControl returns e with its control field replaced by c. It reports an
// error if c holds a value not observed to be legal on arch, rather than
// truncating it into the field.
func EncodeControl(e Encoding, c ControlCode, arch Arch) (Encoding, error) {
	if arch != SM121a {
		return Encoding{}, fmt.Errorf("%w: control codes are calibrated for %s only, not %s", cudaerrors.ErrUnsupported, SM121a, arch)
	}
	if err := c.Valid(); err != nil {
		return Encoding{}, err
	}
	var mask uint64
	for _, f := range [][2]int{{ctrlShift, stallWidth}, {yieldShift, 1}, {wrtBarShift, barWidth}, {readBarShift, barWidth}, {waitShift, waitWidth}, {reuseShift, reuseWidth}} {
		mask |= uint64(1<<uint(f[1])-1) << uint(f[0])
	}
	hi := e.Hi &^ mask
	hi |= uint64(c.Stall) << ctrlShift
	if c.Yield {
		hi |= 1 << yieldShift
	}
	hi |= uint64(c.WriteBarrier) << wrtBarShift
	hi |= uint64(c.ReadBarrier) << readBarShift
	hi |= uint64(c.WaitMask) << waitShift
	hi |= uint64(c.Reuse) << reuseShift
	return Encoding{Lo: e.Lo, Hi: hi}, nil
}

// A RegisterClass distinguishes the register files a SASS operand can name.
type RegisterClass int

const (
	GPR   RegisterClass = iota // R0-R254
	Pred                       // P0-P6
	UGPR                       // UR0-UR62
	UPred                      // UP0-UP6
)

func (c RegisterClass) String() string {
	switch c {
	case GPR:
		return "R"
	case Pred:
		return "P"
	case UGPR:
		return "UR"
	case UPred:
		return "UP"
	}
	return "?"
}

// Sentinel register numbers. RZ and URZ read as zero and discard writes; PT
// and UPT read as true. A write to one of them is not a definition.
const (
	RZNum  = 255
	PTNum  = 7
	URZNum = 63
)

// A Register names one register operand.
type Register struct {
	Class RegisterClass
	Num   int
}

// Sink reports whether r discards writes and so is never a definition.
func (r Register) Sink() bool {
	switch r.Class {
	case GPR:
		return r.Num == RZNum
	case UGPR:
		return r.Num == URZNum
	case Pred, UPred:
		return r.Num == PTNum
	}
	return false
}

func (r Register) String() string { return fmt.Sprintf("%s%d", r.Class, r.Num) }

// parseRegister returns the register a bare operand token names.
func parseRegister(tok string) (Register, bool) {
	tok = strings.TrimSpace(tok)
	tok = strings.TrimPrefix(tok, "!")
	// Suffixes nvdisasm attaches to a register operand, e.g. R7.reuse.
	if i := strings.IndexByte(tok, '.'); i >= 0 {
		tok = tok[:i]
	}
	switch {
	case tok == "RZ":
		return Register{GPR, RZNum}, true
	case tok == "URZ":
		return Register{UGPR, URZNum}, true
	case tok == "PT":
		return Register{Pred, PTNum}, true
	case tok == "UPT":
		return Register{UPred, PTNum}, true
	}
	for _, p := range []struct {
		prefix string
		class  RegisterClass
	}{{"UR", UGPR}, {"UP", UPred}, {"R", GPR}, {"P", Pred}} {
		rest, ok := strings.CutPrefix(tok, p.prefix)
		if !ok {
			continue
		}
		n, err := strconv.Atoi(rest)
		if err != nil {
			continue
		}
		return Register{p.class, n}, true
	}
	return Register{}, false
}

// An OpcodeClass records how one opcode's operands map to definitions and
// uses, and which reordering rules apply to it.
//
// The table is deliberately small. A def/use model guessed from an opcode's
// spelling is the failure this lane exists to avoid: an opcode absent from the
// table is reported as unclassified, and every analysis below refuses to
// decide rather than assuming the common case.
type OpcodeClass struct {
	Defs       int  // leading operands that are destinations
	Loads      bool // reads memory
	Stores     bool // writes memory
	Fence      bool // orders memory operations around it
	Branch     bool // transfers control
	Indirect   bool // branch target comes from a register; see [CFG.IsDegenerate]
	FloatAccum bool // accumulates in floating point; see [Legality]
}

// opcodeClasses covers the instruction subset observed in the sm_121a
// fixtures, and the branches a control flow graph must recognize wherever they
// appear. The base mnemonic is the part before the first dot.
var opcodeClasses = map[string]OpcodeClass{
	"MOV":    {Defs: 1},
	"S2R":    {Defs: 1},
	"S2UR":   {Defs: 1},
	"IADD":   {Defs: 1},
	"IADD3":  {Defs: 1},
	"IMAD":   {Defs: 1},
	"LOP3":   {Defs: 2}, // LOP3.LUT Rd, Ra, Rb, Rc, imm, Pd
	"SHF":    {Defs: 1},
	"POPC":   {Defs: 1},
	"ISETP":  {Defs: 2}, // ISETP.GE.AND P0, PT, R0, R1, PT
	"SEL":    {Defs: 1},
	"PRMT":   {Defs: 1},
	"FADD":   {Defs: 1, FloatAccum: true},
	"FMUL":   {Defs: 1, FloatAccum: true},
	"FFMA":   {Defs: 1, FloatAccum: true},
	"LDC":    {Defs: 1, Loads: true},
	"LDCU":   {Defs: 1, Loads: true},
	"LDG":    {Defs: 1, Loads: true},
	"LDS":    {Defs: 1, Loads: true},
	"STG":    {Stores: true},
	"STS":    {Stores: true},
	"RED":    {Stores: true},
	"ATOM":   {Defs: 1, Loads: true, Stores: true},
	"ATOMG":  {Defs: 1, Loads: true, Stores: true},
	"MEMBAR": {Fence: true},
	"ERRBAR": {Fence: true},
	"BAR":    {Fence: true},
	"BSSY":   {Branch: true},
	"BSYNC":  {Branch: true},
	"BRA":    {Branch: true},
	"BRX":    {Branch: true, Indirect: true},
	"BRXU":   {Branch: true, Indirect: true},
	"JMP":    {Branch: true},
	"JMX":    {Branch: true, Indirect: true},
	"JMXU":   {Branch: true, Indirect: true},
	"KILL":   {Branch: true},
	"CALL":   {Branch: true},
	"RET":    {Branch: true},
	"EXIT":   {Branch: true},
	"NOP":    {},
}

// Classify returns the class of an opcode, and false if it is not in the
// covered subset. Callers must not substitute a default for false.
func Classify(opcode string) (OpcodeClass, bool) {
	c, ok := opcodeClasses[BaseOpcode(opcode)]
	return c, ok
}

// splitOperands splits an operand list on commas outside brackets, so that a
// memory operand such as [R2.64+0x10] stays one token.
func splitOperands(s string) []string {
	var out []string
	depth := 0
	start := 0
	for i, r := range s {
		switch r {
		case '[':
			depth++
		case ']':
			depth--
		case ',':
			if depth == 0 {
				out = append(out, strings.TrimSpace(s[start:i]))
				start = i + 1
			}
		}
	}
	if rest := strings.TrimSpace(s[start:]); rest != "" {
		out = append(out, rest)
	}
	return out
}

// registersIn returns every register named anywhere in an operand token,
// including inside a memory address such as [R2.64+0x10].
func registersIn(tok string) []Register {
	var out []Register
	for _, f := range strings.FieldsFunc(tok, func(r rune) bool {
		return !(r >= '0' && r <= '9') && !(r >= 'A' && r <= 'Z') && r != '.'
	}) {
		if reg, ok := parseRegister(f); ok {
			out = append(out, reg)
		}
	}
	return out
}

// DefsUses returns the registers instruction i defines and uses.
//
// It reports false if i's opcode is outside the covered subset. A caller that
// treats an unclassified instruction as defining nothing will reorder across a
// definition it could not see, so the boolean is not advisory.
func DefsUses(i Instruction) (defs, uses []Register, ok bool) {
	class, ok := Classify(i.Opcode)
	if !ok {
		return nil, nil, false
	}
	// A guarding predicate is a use, whether or not it is negated.
	if p := strings.TrimSpace(i.Predicate); p != "" {
		if reg, ok := parseRegister(strings.TrimPrefix(p, "@")); ok && !reg.Sink() {
			uses = append(uses, reg)
		}
	}
	ops := splitOperands(i.Operands)
	for n, tok := range ops {
		regs := registersIn(tok)
		// A destination operand that is a memory reference is an address
		// computation, so its registers are uses, not definitions.
		isMem := strings.HasPrefix(tok, "[")
		if n < class.Defs && !isMem {
			for _, r := range regs {
				if r.Sink() {
					continue
				}
				defs = append(defs, r)
			}
			continue
		}
		for _, r := range regs {
			if r.Sink() {
				continue
			}
			uses = append(uses, r)
		}
	}
	return defs, uses, true
}

// A BasicBlock is a maximal straight-line run of instructions: control enters
// only at the first and leaves only after the last.
type BasicBlock struct {
	Start, End   uint64 // address of the first instruction, and one past the last
	Instructions []Instruction
}

// BasicBlocks reconstructs the basic blocks of fn.
//
// A block ends after any branch, and begins at any address named as a branch
// target. Targets are taken from the operand text nvdisasm prints, which in an
// ELF section is a label reference such as `(.L_x_0), resolved through
// [Function.Labels], and in a raw binary is a 0x-prefixed address. An operand
// that is neither, such as the barrier register B0 of a BSSY or the register
// of an indirect branch, names no target. [Disassembler.ControlFlowGraph] is
// nvdisasm's own answer and is what TestBasicBlockReconstructionAgainstNvdisasm
// compares against.
func BasicBlocks(fn *Function) []BasicBlock {
	if fn == nil || len(fn.Instructions) == 0 {
		return nil
	}
	leaders := map[uint64]bool{fn.Instructions[0].Address: true}
	for n, inst := range fn.Instructions {
		class, ok := Classify(inst.Opcode)
		if !ok || !class.Branch {
			continue
		}
		if n+1 < len(fn.Instructions) {
			leaders[fn.Instructions[n+1].Address] = true
		}
		for _, tok := range splitOperands(inst.Operands) {
			if a, ok := fn.target(tok); ok {
				leaders[a] = true
			}
		}
	}
	var blocks []BasicBlock
	cur := BasicBlock{Start: fn.Instructions[0].Address}
	for n, inst := range fn.Instructions {
		if n > 0 && leaders[inst.Address] {
			cur.End = inst.Address
			blocks = append(blocks, cur)
			cur = BasicBlock{Start: inst.Address}
		}
		cur.Instructions = append(cur.Instructions, inst)
	}
	last := fn.Instructions[len(fn.Instructions)-1]
	cur.End = last.Address + 16
	blocks = append(blocks, cur)
	return blocks
}

// target returns the address a branch operand names: a 0x-prefixed address,
// bare or quoted as nvdisasm prints it (0x180, `0x180`, `(0x180)), or a label
// in fn.Labels (`(.L_x_0)).
func (fn *Function) target(tok string) (uint64, bool) {
	tok = strings.Trim(strings.TrimSpace(tok), "`()")
	if hex, ok := strings.CutPrefix(tok, "0x"); ok {
		a, err := strconv.ParseUint(hex, 16, 64)
		return a, err == nil
	}
	a, ok := fn.Labels[tok]
	return a, ok
}

// A DepKind names the reason one instruction must precede another.
type DepKind int

const (
	RAW     DepKind = iota // read after write: the later reads what the earlier wrote
	WAR                    // write after read: the later overwrites what the earlier read
	WAW                    // write after write: both write the same register
	Memory                 // both touch memory and their order is observable
	Fence                  // one is a fence and the other touches memory
	Barrier                // the later waits on a scoreboard barrier the earlier sets
)

func (k DepKind) String() string {
	return [...]string{"RAW", "WAR", "WAW", "Memory", "Fence", "Barrier"}[k]
}

// An Edge records that instruction From must issue before instruction To.
// From and To are indices into the block the edge was computed over.
type Edge struct {
	From, To int
	Kind     DepKind
	Reg      Register // the register involved, for RAW, WAR and WAW
}

// Dependencies returns the ordering edges within one basic block.
//
// It reports an error naming the first unclassified opcode rather than
// returning a partial graph: an edge set that silently omits the dependencies
// of an instruction nobody modelled is indistinguishable from one that proves
// a swap is safe.
func Dependencies(b BasicBlock, arch Arch) ([]Edge, error) {
	if arch != SM121a {
		return nil, fmt.Errorf("dependency model: %w", &ControlDecodeError{Arch: arch})
	}
	type rw struct{ defs, uses []Register }
	info := make([]rw, len(b.Instructions))
	for n, inst := range b.Instructions {
		d, u, ok := DefsUses(inst)
		if !ok {
			return nil, fmt.Errorf("instruction %d at %#x: opcode %q is outside the covered subset", n, inst.Address, inst.Opcode)
		}
		info[n] = rw{d, u}
	}
	var edges []Edge
	has := func(rs []Register, r Register) bool {
		for _, x := range rs {
			if x == r {
				return true
			}
		}
		return false
	}
	for later := range b.Instructions {
		for earlier := 0; earlier < later; earlier++ {
			for _, r := range info[later].uses {
				if has(info[earlier].defs, r) {
					edges = append(edges, Edge{earlier, later, RAW, r})
				}
			}
			for _, r := range info[later].defs {
				if has(info[earlier].uses, r) {
					edges = append(edges, Edge{earlier, later, WAR, r})
				}
				if has(info[earlier].defs, r) {
					edges = append(edges, Edge{earlier, later, WAW, r})
				}
			}
			ec, _ := Classify(b.Instructions[earlier].Opcode)
			lc, _ := Classify(b.Instructions[later].Opcode)
			switch {
			case ec.Fence && (lc.Loads || lc.Stores), (ec.Loads || ec.Stores) && lc.Fence:
				edges = append(edges, Edge{earlier, later, Fence, Register{}})
			case (ec.Stores && (lc.Loads || lc.Stores)), (ec.Loads && lc.Stores):
				edges = append(edges, Edge{earlier, later, Memory, Register{}})
			}
			// A barrier the earlier instruction sets and the later waits on.
			eCtl, err1 := DecodeControl(b.Instructions[earlier].Encoding, arch)
			lCtl, err2 := DecodeControl(b.Instructions[later].Encoding, arch)
			if err1 == nil && err2 == nil {
				for _, set := range []uint8{eCtl.WriteBarrier, eCtl.ReadBarrier} {
					if set != NoBarrier && lCtl.Waits(int(set)) {
						edges = append(edges, Edge{earlier, later, Barrier, Register{}})
					}
				}
			}
		}
	}
	sort.SliceStable(edges, func(i, j int) bool {
		if edges[i].From != edges[j].From {
			return edges[i].From < edges[j].From
		}
		return edges[i].To < edges[j].To
	})
	return edges, nil
}

// A NumericalPolicy states whether floating-point accumulation order may be
// changed for the kernel under analysis.
//
// Reordering is not semantics-preserving in general. The default zero value
// forbids it, because a kernel that has not declared a policy has not
// permitted the change.
type NumericalPolicy struct {
	// AllowFloatReassociation permits swapping instructions that accumulate
	// in floating point. It must come from the kernel's declared policy, not
	// from an observation that the result happened to match.
	AllowFloatReassociation bool
}

// A Legality is the verdict on one proposed reordering.
//
// Legal is true only if every check ran and passed. A check that could not be
// decided sets Undecided and leaves Legal false: a reordering this model
// cannot reason about is refused, never permitted by default.
type Legality struct {
	Legal     bool
	Undecided bool
	Reasons   []string
}

func (l Legality) String() string {
	switch {
	case l.Legal:
		return "legal"
	case l.Undecided:
		return "undecided: " + strings.Join(l.Reasons, "; ")
	default:
		return "illegal: " + strings.Join(l.Reasons, "; ")
	}
}

// CanSwap reports whether the instructions at positions i and j of b, which
// must be adjacent, may exchange places.
//
// Legality is decided statically, on host, with no device. A true verdict is a
// claim about data dependencies, memory ordering and control codes only; it is
// not a claim that the resulting sequence is faster, and not a claim that it
// loads, which is what the step 4 feasibility gate decides.
func CanSwap(b BasicBlock, i, j int, arch Arch, policy NumericalPolicy) Legality {
	if j != i+1 {
		if i == j+1 {
			i, j = j, i
		} else {
			return Legality{Reasons: []string{fmt.Sprintf("positions %d and %d are not adjacent", i, j)}}
		}
	}
	if i < 0 || j >= len(b.Instructions) {
		return Legality{Reasons: []string{"position out of range"}}
	}
	edges, err := Dependencies(b, arch)
	if err != nil {
		return Legality{Undecided: true, Reasons: []string{err.Error()}}
	}
	var reasons []string
	for _, e := range edges {
		if e.From == i && e.To == j {
			switch e.Kind {
			case RAW, WAR, WAW:
				reasons = append(reasons, fmt.Sprintf("%s dependency on %s", e.Kind, e.Reg))
			default:
				reasons = append(reasons, fmt.Sprintf("%s ordering edge", e.Kind))
			}
		}
	}
	a, c := b.Instructions[i], b.Instructions[j]
	ac, ok1 := Classify(a.Opcode)
	cc, ok2 := Classify(c.Opcode)
	if !ok1 || !ok2 {
		return Legality{Undecided: true, Reasons: []string{"an opcode is outside the covered subset"}}
	}
	if (ac.FloatAccum || cc.FloatAccum) && !policy.AllowFloatReassociation {
		reasons = append(reasons, "floating-point accumulation order is not reassociable under the declared policy")
	}
	// Control codes must survive the move. A barrier the earlier instruction
	// sets and the later waits on is already an edge above; what remains is
	// whether the pair's control codes can be recomputed at all.
	if _, _, err := RecomputeControl(a, c, arch); err != nil {
		reasons = append(reasons, "control codes cannot be recomputed: "+err.Error())
	}
	if len(reasons) > 0 {
		return Legality{Reasons: reasons}
	}
	return Legality{Legal: true}
}

// RecomputeControl returns the control codes the two instructions must carry
// after a and c exchange places, or an error if they cannot be recomputed with
// confidence.
//
// The rule this model can state is narrow and it is stated as such. A barrier
// index that an instruction sets travels with that instruction; a wait mask
// travels with the waiter. What this model cannot do is recompute a stall
// count, because the stall an instruction needs depends on the latency of the
// instruction now preceding it and no latency table for sm_121a is available
// here. A swap of two instructions whose stall counts differ is therefore
// refused rather than approximated.
func RecomputeControl(a, c Instruction, arch Arch) (aAfter, cAfter ControlCode, err error) {
	ca, err1 := DecodeControl(a.Encoding, arch)
	cc, err2 := DecodeControl(c.Encoding, arch)
	if err1 != nil || err2 != nil {
		return ControlCode{}, ControlCode{}, fmt.Errorf("decode controls for both instructions on %s: %w", arch, errors.Join(err1, err2))
	}
	if ca.Stall != cc.Stall {
		return ControlCode{}, ControlCode{}, fmt.Errorf("stall counts differ (%d and %d) and no sm_121a latency table is available to recompute them", ca.Stall, cc.Stall)
	}
	if ca.WaitMask != 0 || cc.WaitMask != 0 {
		return ControlCode{}, ControlCode{}, fmt.Errorf("one instruction waits on a scoreboard barrier; moving it changes when the wait is observed")
	}
	// The codes travel with their instructions.
	return ca, cc, nil
}

// A LiveSet is a set of registers live at one point.
type LiveSet map[Register]bool

// Contains reports whether r is in the set.
func (s LiveSet) Contains(r Register) bool { return s[r] }

// Registers returns the set's members sorted by class then number, so that
// output is stable.
func (s LiveSet) Registers() []Register {
	out := make([]Register, 0, len(s))
	for r := range s {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Class != out[j].Class {
			return out[i].Class < out[j].Class
		}
		return out[i].Num < out[j].Num
	})
	return out
}

// Liveness returns the set of registers live immediately after each
// instruction of b, given the registers live on exit from the block.
//
// This is a within-block backward pass: liveOut is the caller's claim about
// what the block's successors read, and Liveness does not compute it. A caller
// that passes nil is asserting the block's successors read nothing, which is
// true only for a block ending in EXIT.
//
// It reports an error for the same reason [Dependencies] does: an unclassified
// opcode makes every set after it wrong, and a liveness set that is quietly
// wrong is worse than none, because it is what a reordering is checked against.
func Liveness(b BasicBlock, liveOut LiveSet, arch Arch) ([]LiveSet, error) {
	if arch != SM121a {
		return nil, fmt.Errorf("liveness is calibrated for %s only, not %s", SM121a, arch)
	}
	n := len(b.Instructions)
	out := make([]LiveSet, n)
	live := LiveSet{}
	for r, ok := range liveOut {
		if ok {
			live[r] = true
		}
	}
	for i := n - 1; i >= 0; i-- {
		cur := LiveSet{}
		for r := range live {
			cur[r] = true
		}
		out[i] = cur
		defs, uses, ok := DefsUses(b.Instructions[i])
		if !ok {
			return nil, fmt.Errorf("instruction %d at %#x: opcode %q is outside the covered subset",
				i, b.Instructions[i].Address, b.Instructions[i].Opcode)
		}
		next := LiveSet{}
		for r := range live {
			next[r] = true
		}
		for _, d := range defs {
			delete(next, d)
		}
		for _, u := range uses {
			next[u] = true
		}
		live = next
	}
	return out, nil
}

// StallSampleUnavailable is the percentage [StallProfile] carries when no
// sampler measured the instruction. It is not zero: an instruction that was
// never sampled and an instruction sampled at zero percent are different
// facts, and only the second is a measurement. The sentinel follows
// cudaroofline's FloorUnavailable.
const StallSampleUnavailable = -1.0

// A StallProfile is a measured stall percentage for one instruction.
//
// This package measures nothing. It disassembles, models dependencies and
// re-encodes, all on host, and none of that observes a warp stalling. A
// profile therefore reaches this package only from a sampler that ran on the
// device, carrying the sampler's name in Source; a profile with no Source is
// absent, never zero.
type StallProfile struct {
	Reason   string  // the stall reason the sampler named, e.g. "long_scoreboard"
	Percent  float64 // percentage of samples, or StallSampleUnavailable
	Measured bool    // whether a sampler produced Percent
	Source   string  // the sampler that measured it; empty when absent
}

// Valid reports whether the profile is internally consistent. It rejects both
// a measurement with no source and an absence carrying a plausible number,
// because a synthesized percentage is indistinguishable from a measured one
// once it is past this check.
func (p StallProfile) Valid() error {
	if p.Measured {
		if p.Source == "" {
			return fmt.Errorf("stall profile %q claims a measurement with no source", p.Reason)
		}
		if p.Percent < 0 || p.Percent > 100 {
			return fmt.Errorf("stall profile %q: measured percentage %g outside 0-100", p.Reason, p.Percent)
		}
		return nil
	}
	if p.Percent != StallSampleUnavailable {
		return fmt.Errorf("stall profile %q is unmeasured but carries percentage %g; unmeasured is %g",
			p.Reason, p.Percent, StallSampleUnavailable)
	}
	if p.Source != "" {
		return fmt.Errorf("stall profile %q is unmeasured but names source %q", p.Reason, p.Source)
	}
	return nil
}

// InstructionStalls returns the stall profiles this package holds for inst,
// and false when it holds none.
//
// It always returns false. The function exists so that a caller asking for
// stall percentages receives an explicit absence rather than discovering there
// is no accessor and reaching for the control code's [ControlCode.Stall],
// which is a static issue delay in cycles chosen by ptxas, not a measured
// fraction of sampled warps.
func InstructionStalls(inst Instruction) ([]StallProfile, bool) {
	return nil, false
}
