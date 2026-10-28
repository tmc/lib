package cudasass

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// This file rewrites the control codes of a basic block -- stall counts,
// scoreboard indices, wait masks, reuse flags and the yield bit -- without
// touching any other bit of any instruction, and checks the result against a
// register-hazard model before keeping it.
//
// What the model knows and does not know is the point of the file, so it is
// stated here once:
//
//   - Issue time is the running sum of stall counts. A wait on a scoreboard
//     can only delay issue, so ignoring it makes every fixed-latency check
//     conservative.
//   - An instruction that sets a write barrier is variable-latency: its
//     results may be read or overwritten only after some instruction at or
//     before the reader waits on that barrier. An instruction that sets a
//     read barrier may have its sources overwritten only after such a wait.
//   - Any other instruction that defines a register is fixed-latency, and its
//     result latency comes from a [Latencies] table supplied by the caller.
//     NVIDIA publishes no latency table; [ObservedLatencies] is derived from
//     the separations ptxas emitted in this package's sm_121a fixtures, which
//     bound the true latency from above only if ptxas was right.
//   - Functional-unit throughput, register-bank conflicts and the operand
//     reuse cache's capacity are not modelled. They cost cycles, not
//     correctness, on the assumption -- not measured here -- that the
//     hardware throttles issue for them itself.

// Latencies gives the result latency of fixed-latency instructions, by base
// opcode (see [BaseOpcode]).
type Latencies struct {
	// Result is the number of cycles after issue before a non-branch
	// instruction may read the result.
	Result map[string]uint8
	// Branch is the number of cycles after issue before a branch may read
	// the result, typically a predicate. An opcode absent here uses Result.
	Branch map[string]uint8
	// Pair overrides Result for one {producer, reader} pair of base
	// opcodes.
	Pair map[[2]string]uint8
}

func (l Latencies) latency(producer, reader string, branch bool) (uint8, bool) {
	base := BaseOpcode(producer)
	if branch {
		if n, ok := l.Branch[base]; ok {
			return n, true
		}
	}
	if n, ok := l.Pair[[2]string{base, BaseOpcode(reader)}]; ok && reader != "" {
		return n, true
	}
	n, ok := l.Result[base]
	return n, ok
}

// ObservedLatencies returns, for each fixed-latency producer seen with a
// reader in the same basic block of this package's sm_121a fixtures
// (ptxas 13.3 and 13.4; testdata/*.sass and testdata/roundtrip), the
// smallest producer-to-reader separation ptxas emitted.
//
// These are not hardware measurements. A separation ptxas emitted is an
// upper bound on the true latency if ptxas was right, so using it keeps an
// optimized schedule no tighter than one ptxas itself produced for that
// producer. The bound is weaker for a reader class not seen with that
// producer; where one pair was seen tighter than the rest (IMAD feeding
// IMAD), it is a Pair entry and the producer's Result is the larger value
// seen with every other reader.
func ObservedLatencies() Latencies {
	return Latencies{
		Result: map[string]uint8{
			"IMAD": 5, "IADD": 5, "LOP3": 5, "MOV": 5, "LEA": 5,
			"FADD": 5, "FFMA": 5, "I2FP": 5,
			"UIADD3": 4, "ULOP3": 4, "UIMAD": 6,
			// Predicate producers: every reader seen is at 13 or more.
			"ISETP": 13, "FSETP": 13, "UISETP": 9,
		},
		Branch: map[string]uint8{
			"ISETP": 13, "FSETP": 13, "UISETP": 9,
		},
		Pair: map[[2]string]uint8{
			{"IMAD", "IMAD"}: 4,
		},
	}
}

// controlOpcodes extends the dependency table with opcodes that appear in
// the sm_121a fixtures or in tensor-core inner loops, for control-code
// analysis only. Tensor reports whether the opcode issues to a tensor or
// paired-FMA pipe, where a run of independent instructions is a candidate
// for keeping the warp from yielding.
var controlOpcodes = map[string]struct {
	OpcodeClass
	Tensor bool
}{
	"HFMA2":  {OpcodeClass: OpcodeClass{Defs: 1}},
	"I2FP":   {OpcodeClass: OpcodeClass{Defs: 1}},
	"F2FP":   {OpcodeClass: OpcodeClass{Defs: 1}},
	"FSETP":  {OpcodeClass: OpcodeClass{Defs: 2}},
	"LEA":    {OpcodeClass: OpcodeClass{Defs: 1}},
	"UIADD3": {OpcodeClass: OpcodeClass{Defs: 1}},
	"UIMAD":  {OpcodeClass: OpcodeClass{Defs: 1}},
	"ULOP3":  {OpcodeClass: OpcodeClass{Defs: 1}},
	"UISETP": {OpcodeClass: OpcodeClass{Defs: 2}},
	"UMOV":   {OpcodeClass: OpcodeClass{Defs: 1}},
	"MUFU":   {OpcodeClass: OpcodeClass{Defs: 1}},
	"LDL":    {OpcodeClass: OpcodeClass{Defs: 1, Loads: true}},
	"STL":    {OpcodeClass: OpcodeClass{Stores: true}},
	"DEPBAR": {OpcodeClass: OpcodeClass{Fence: true}},
	"UTCBAR": {OpcodeClass: OpcodeClass{Fence: true}},
	"DFMA":   {OpcodeClass: OpcodeClass{Defs: 1, FloatAccum: true}},
	"DADD":   {OpcodeClass: OpcodeClass{Defs: 1, FloatAccum: true}},
	"DMUL":   {OpcodeClass: OpcodeClass{Defs: 1, FloatAccum: true}},
	"FFMA2":  {OpcodeClass: OpcodeClass{Defs: 1, FloatAccum: true}, Tensor: true},
	"FMUL2":  {OpcodeClass: OpcodeClass{Defs: 1, FloatAccum: true}, Tensor: true},
	"FADD2":  {OpcodeClass: OpcodeClass{Defs: 1, FloatAccum: true}, Tensor: true},
	"HMMA":   {OpcodeClass: OpcodeClass{Defs: 1, FloatAccum: true}, Tensor: true},
	"IMMA":   {OpcodeClass: OpcodeClass{Defs: 1}, Tensor: true},
	"QMMA":   {OpcodeClass: OpcodeClass{Defs: 1, FloatAccum: true}, Tensor: true},
	"OMMA":   {OpcodeClass: OpcodeClass{Defs: 1, FloatAccum: true}, Tensor: true},
	// tcgen05.mma in SASS reads descriptors from uniform registers and
	// writes tensor memory, not registers.
	"UTCHMMA": {Tensor: true},
	"UTCQMMA": {Tensor: true},
	"UTCOMMA": {Tensor: true},
	"UTCIMMA": {Tensor: true},
}

// operandWidth returns how many consecutive GPRs every register operand of
// opcode spans, overestimating where the exact fragment shape is not
// modelled. An overestimate adds dependencies and is safe; an underestimate
// would hide one.
func operandWidth(opcode string) int {
	switch base := BaseOpcode(opcode); base {
	case "HMMA", "IMMA", "QMMA", "OMMA":
		return 4
	case "DFMA", "DADD", "DMUL":
		return 2
	}
	return 1
}

// defWidth returns how many consecutive GPRs the destination of opcode
// spans.
func defWidth(opcode string) int {
	w := operandWidth(opcode)
	for _, m := range strings.Split(strings.ToUpper(opcode), ".")[1:] {
		switch m {
		case "64", "WIDE":
			w = max(w, 2)
		case "128":
			w = max(w, 4)
		}
	}
	return w
}

// wideRegisters returns the registers an operand token names, widening a
// GPR to width registers, or to the width its own .64 or .128 suffix states.
func wideRegisters(tok string, width int) []Register {
	var out []Register
	for _, f := range strings.FieldsFunc(tok, func(r rune) bool {
		return !(r >= '0' && r <= '9') && !(r >= 'A' && r <= 'Z') && r != '.'
	}) {
		reg, ok := parseRegister(f)
		if !ok || reg.Sink() {
			continue
		}
		w := 1
		if reg.Class == GPR {
			w = width
			switch {
			case strings.HasSuffix(f, ".64"):
				w = max(w, 2)
			case strings.HasSuffix(f, ".128"):
				w = max(w, 4)
			}
		}
		for k := range w {
			out = append(out, Register{reg.Class, reg.Num + k})
		}
	}
	return out
}

// controlDefsUses returns the registers inst defines and uses for hazard
// analysis, widened to the register pairs and quads 64-bit, 128-bit and
// matrix operands occupy.
func controlDefsUses(inst Instruction) (defs, uses []Register, class OpcodeClass, tensor, ok bool) {
	class, ok = Classify(inst.Opcode)
	if !ok {
		c, found := controlOpcodes[BaseOpcode(inst.Opcode)]
		if !found {
			return nil, nil, OpcodeClass{}, false, false
		}
		class, tensor = c.OpcodeClass, c.Tensor
	}
	if p := strings.TrimSpace(inst.Predicate); p != "" {
		if reg, ok := parseRegister(strings.TrimPrefix(p, "@")); ok && !reg.Sink() {
			uses = append(uses, reg)
		}
	}
	// The operand table gives a count of leading destinations, which does
	// not fit every spelling: LOP3.LUT has a second leading register that
	// is a source in "LOP3.LUT R0, R0, R5, RZ, 0xfc, !PT", and a predicate
	// destination may trail, as in IADD3's carry-out. So a leading operand
	// after the first is both a definition and a use, and so is every
	// predicate operand. Too many dependencies cost cycles; too few would
	// be wrong.
	src := operandWidth(inst.Opcode)
	for n, tok := range splitOperands(inst.Operands) {
		if n < class.Defs && !strings.HasPrefix(tok, "[") {
			regs := wideRegisters(tok, defWidth(inst.Opcode))
			defs = append(defs, regs...)
			if n > 0 {
				uses = append(uses, regs...)
			}
			continue
		}
		for _, r := range wideRegisters(tok, src) {
			uses = append(uses, r)
			if r.Class == Pred || r.Class == UPred {
				defs = append(defs, r)
			}
		}
	}
	return defs, uses, class, tensor, true
}

// A Bundle is one instruction of a basic block, with the facts the
// control-code optimizer needs about it.
type Bundle struct {
	Inst   Instruction // Inst.Encoding is never changed except for its control field
	Ctrl   ControlCode
	Defs   []Register
	Uses   []Register
	Class  OpcodeClass
	Tensor bool // issues to a tensor or paired-FMA pipe; see [ControlOptStats.StickyTensorBursts]
}

// variable reports whether the bundle's results are guarded by a scoreboard.
func (b *Bundle) variable() bool { return b.Ctrl.WriteBarrier != NoBarrier }

// Bundles decodes the control codes of block's instructions on arch and
// returns them as bundles. It reports an error for an opcode outside the
// modelled subset or an instruction with no encoding.
func Bundles(block BasicBlock, arch Arch) ([]Bundle, error) {
	out := make([]Bundle, 0, len(block.Instructions))
	for n, inst := range block.Instructions {
		ctrl, err := DecodeControl(inst.Encoding, arch)
		if err != nil {
			return nil, fmt.Errorf("instruction %d at %#x: %w", n, inst.Address, err)
		}
		defs, uses, class, tensor, ok := controlDefsUses(inst)
		if !ok {
			return nil, fmt.Errorf("instruction %d at %#x: opcode %q is outside the control-code model", n, inst.Address, inst.Opcode)
		}
		out = append(out, Bundle{Inst: inst, Ctrl: ctrl, Defs: defs, Uses: uses, Class: class, Tensor: tensor})
	}
	return out, nil
}

// A HazardError reports a register dependency a control-code schedule does
// not honour.
type HazardError struct {
	Kind     DepKind
	From, To int // indices into the block
	Reg      Register
	Reason   string
}

func (e *HazardError) Error() string {
	return fmt.Sprintf("%v hazard on %v from instruction %d to %d: %s", e.Kind, e.Reg, e.From, e.To, e.Reason)
}

// VerifyHazardSafety checks every register dependency within block against
// its control codes, using lat for fixed-latency results:
//
//   - a fixed-latency result is read only after its latency has elapsed in
//     accumulated stall counts, and is overwritten by a later fixed-latency
//     instruction only if that write completes later;
//   - a result guarded by a write barrier is read or overwritten only after
//     some instruction at or before the reader waits on that barrier;
//   - a source guarded by a read barrier is overwritten only after such a
//     wait.
//
// It returns a [*HazardError] for the first violation, and an error if a
// fixed-latency producer with a reader in the block has no entry in lat.
// Dependencies that cross the block boundary are not checked here; see
// [OptimizeBasicBlockControl].
func VerifyHazardSafety(block []Bundle, lat Latencies) error {
	issue := issueTimes(block)
	for p := range block {
		bp := &block[p]
		for _, r := range bp.Defs {
			if err := checkDef(block, issue, p, r, lat); err != nil {
				return err
			}
		}
		if bp.Ctrl.ReadBarrier != NoBarrier {
			for _, r := range bp.Uses {
				for c := p + 1; c < len(block); c++ {
					if !has(block[c].Defs, r) {
						continue
					}
					if !waitedBetween(block, p, c, bp.Ctrl.ReadBarrier) {
						return &HazardError{WAR, p, c, r, fmt.Sprintf("overwrites a source read under barrier %d with no wait on it", bp.Ctrl.ReadBarrier)}
					}
					break
				}
			}
		}
	}
	return nil
}

// checkDef checks the readers and the next writer of register r, defined
// by instruction p.
func checkDef(block []Bundle, issue []int, p int, r Register, lat Latencies) error {
	bp := &block[p]
	for c := p + 1; c < len(block); c++ {
		bc := &block[c]
		reads, writes := has(bc.Uses, r), has(bc.Defs, r)
		if !reads && !writes {
			continue
		}
		kind := RAW
		if !reads {
			kind = WAW
		}
		switch {
		case bp.variable():
			if !waitedBetween(block, p, c, bp.Ctrl.WriteBarrier) {
				return &HazardError{kind, p, c, r, fmt.Sprintf("reaches a result guarded by barrier %d with no wait on it", bp.Ctrl.WriteBarrier)}
			}
		case reads:
			n, ok := lat.latency(bp.Inst.Opcode, bc.Inst.Opcode, bc.Class.Branch)
			if !ok {
				return fmt.Errorf("instruction %d (%s) has a reader at %d and no latency entry", p, bp.Inst.Opcode, c)
			}
			if d := issue[c] - issue[p]; d < int(n) {
				return &HazardError{RAW, p, c, r, fmt.Sprintf("reads %d cycles after issue, latency is %d", d, n)}
			}
		case !bc.variable():
			np, ok1 := lat.latency(bp.Inst.Opcode, "", false)
			nc, ok2 := lat.latency(bc.Inst.Opcode, "", false)
			if ok1 && ok2 && issue[c]+int(nc) <= issue[p]+int(np) {
				return &HazardError{WAW, p, c, r, "the later write does not complete after the earlier one"}
			}
		}
		if writes {
			return nil // later accesses see the new definition
		}
	}
	return nil
}

// issueTimes returns each instruction's issue cycle relative to the first.
func issueTimes(block []Bundle) []int {
	t := make([]int, len(block)+1)
	for i := range block {
		t[i+1] = t[i] + int(block[i].Ctrl.Stall)
	}
	return t
}

// waitedBetween reports whether an instruction in (p, c] waits on barrier
// b.
func waitedBetween(block []Bundle, p, c int, b uint8) bool {
	for k := p + 1; k <= c; k++ {
		if block[k].Ctrl.Waits(int(b)) {
			return true
		}
	}
	return false
}

func has(rs []Register, r Register) bool {
	for _, x := range rs {
		if x == r {
			return true
		}
	}
	return false
}

// ControlOptStats summarizes what [OptimizeBasicBlockControl] changed.
type ControlOptStats struct {
	InstructionsScanned int
	StallCyclesSaved    int // static stall cycles removed from the block
	ReuseHintsAdded     int // operand reuse flags set
	SplitScoreboards    int // producers moved off a shared write barrier
	StickyTensorBursts  int // runs of tensor instructions whose yield bit was set
}

func (s *ControlOptStats) add(t ControlOptStats) {
	s.InstructionsScanned += t.InstructionsScanned
	s.StallCyclesSaved += t.StallCyclesSaved
	s.ReuseHintsAdded += t.ReuseHintsAdded
	s.SplitScoreboards += t.SplitScoreboards
	s.StickyTensorBursts += t.StickyTensorBursts
}

// maxLatency returns the largest latency in lat, the bound used for
// registers live into the block from an unknown producer.
func (l Latencies) maxLatency() int {
	m := 0
	for _, t := range []map[string]uint8{l.Result, l.Branch} {
		for _, n := range t {
			m = max(m, int(n))
		}
	}
	return m
}

// boundarySafe checks the dependencies that cross the block boundary, which
// [VerifyHazardSafety] cannot see, by requiring that the candidate keeps
// as much distance as the original where the original's distance could have
// been needed:
//
//   - for the first read of a register not defined earlier in the block,
//     the cycles from block entry to the read stay at least min(original,
//     the largest latency in lat), since its producer is unknown;
//   - for a fixed-latency result still live at block exit, the cycles from
//     its issue to the end of the block stay at least min(original, its
//     latency), since its reader is unknown.
func boundarySafe(orig, cand []Bundle, lat Latencies) error {
	io, ic := issueTimes(orig), issueTimes(cand)
	maxLat := lat.maxLatency()
	seen := map[Register]bool{} // defined in the block, or read once already
	for c := range cand {
		for _, r := range cand[c].Uses {
			if seen[r] {
				continue
			}
			if need := min(io[c], maxLat); ic[c] < need {
				return &HazardError{RAW, -1, c, r, fmt.Sprintf("live-in read %d cycles after block entry, was %d", ic[c], io[c])}
			}
		}
		for _, r := range cand[c].Uses {
			seen[r] = true
		}
		for _, r := range cand[c].Defs {
			seen[r] = true
		}
	}
	end := len(cand)
	for p := range cand {
		if cand[p].variable() {
			continue
		}
		for _, r := range cand[p].Defs {
			if redefined(cand, p, r) {
				continue
			}
			n, ok := lat.latency(cand[p].Inst.Opcode, "", true)
			if !ok {
				n = uint8(maxLat)
			}
			if need := min(io[end]-io[p], int(n)); ic[end]-ic[p] < need {
				return &HazardError{RAW, p, end, r, fmt.Sprintf("live-out result %d cycles before block exit, was %d", ic[end]-ic[p], io[end]-io[p])}
			}
		}
	}
	return nil
}

func redefined(block []Bundle, p int, r Register) bool {
	for c := p + 1; c < len(block); c++ {
		if has(block[c].Defs, r) {
			return true
		}
	}
	return false
}

// OptimizeBasicBlockControl returns block with tightened control codes and
// what changed. It never changes an instruction's opcode, operands or
// order, and changes its encoding only in the control field.
//
// The input must itself pass [VerifyHazardSafety] with lat; otherwise the
// block is refused, since the model would be checking a schedule it already
// considers wrong. Each change is made, checked with [VerifyHazardSafety]
// and a boundary check, and kept only if both pass. The passes, in order:
//
//   - Scoreboard splitting: a variable-latency producer that shares a write
//     barrier with another, whose first reader comes after an earlier wait
//     on that barrier, is moved to a barrier the block neither sets nor
//     waits on, and its first reader waits on the new barrier instead. The
//     earlier wait then no longer waits for it. A producer with no reader in
//     the block is not moved, since a successor block may wait on its
//     barrier.
//   - Stall tightening: the stall of an instruction is lowered to the
//     smallest value, not below 1, that still passes both checks, when it
//     and the instruction after it are fixed-latency, not a branch or
//     fence, and of a class lat has a latency for. Other
//     stalls are left alone because they may pace a functional unit, which
//     the model does not know: in testdata/sm121a-loop-13.3.sass ptxas
//     pads a DFMA chain with NOPs stalling 15 cycles each although the
//     scoreboard already orders the chain.
//   - Operand reuse: on FFMA, FADD, FMUL, IMAD, IADD and IADD3 whose yield
//     bit is set, the reuse flag for source slot s is set when the next
//     instruction is one of those, reads the same GPR in slot s and does not
//     wait on a barrier, and this instruction does not write it. Slot 0 is
//     bit 0, as observed on sm_121a; slots 1 and 2 as bits 1 and 2 are
//     assumed.
//   - Sticky tensor issue: in a run of two or more consecutive tensor-pipe
//     instructions (see [Bundle.Tensor]) none of which waits on a barrier,
//     the yield bit is set on every instruction but the last.
//
// Encodings of instructions whose control code changed are rewritten with
// [EncodeControl] when the instruction carries one.
func OptimizeBasicBlockControl(block []Bundle, lat Latencies) ([]Bundle, ControlOptStats, error) {
	stats := ControlOptStats{InstructionsScanned: len(block)}
	if err := VerifyHazardSafety(block, lat); err != nil {
		return nil, stats, fmt.Errorf("input schedule: %w", err)
	}
	orig := block
	cur := make([]Bundle, len(block))
	copy(cur, block)
	try := func(change func(c []Bundle)) bool {
		cand := make([]Bundle, len(cur))
		copy(cand, cur)
		change(cand)
		if VerifyHazardSafety(cand, lat) != nil || boundarySafe(orig, cand, lat) != nil {
			return false
		}
		cur = cand
		return true
	}

	stats.SplitScoreboards = splitScoreboards(cur, try)

	for i := range cur {
		if !tightenable(&cur[i], lat) || i+1 < len(cur) && !tightenable(&cur[i+1], lat) || cur[i].Ctrl.Stall <= 1 {
			continue
		}
		for s := uint8(1); s < cur[i].Ctrl.Stall; s++ {
			if try(func(c []Bundle) { c[i].Ctrl.Stall = s }) {
				break
			}
		}
	}

	for i := 0; i+1 < len(cur); i++ {
		for _, s := range reuseSlots(&cur[i], &cur[i+1]) {
			if try(func(c []Bundle) { c[i].Ctrl.Reuse |= 1 << s }) {
				stats.ReuseHintsAdded++
			}
		}
	}

	for i := 0; i < len(cur); {
		j := i
		for j < len(cur) && cur[j].Tensor && (j == i || cur[j].Ctrl.WaitMask == 0) {
			j++
		}
		if j-i >= 2 {
			changed := false
			if try(func(c []Bundle) {
				for k := i; k < j-1; k++ {
					changed = changed || !c[k].Ctrl.Yield
					c[k].Ctrl.Yield = true
				}
			}) && changed {
				stats.StickyTensorBursts++
			}
		}
		i = max(j, i+1)
	}

	for i := range cur {
		stats.StallCyclesSaved += int(orig[i].Ctrl.Stall) - int(cur[i].Ctrl.Stall)
		if cur[i].Ctrl == orig[i].Ctrl || !cur[i].Inst.Encoding.Present() {
			continue
		}
		enc, err := EncodeControl(cur[i].Inst.Encoding, cur[i].Ctrl, SM121a)
		if err != nil {
			return nil, stats, fmt.Errorf("instruction %d at %#x: %w", i, cur[i].Inst.Address, err)
		}
		if enc.Lo != orig[i].Inst.Encoding.Lo {
			return nil, stats, fmt.Errorf("instruction %d at %#x: re-encoding changed the low word", i, cur[i].Inst.Address)
		}
		cur[i].Inst.Encoding = enc
	}
	return cur, stats, nil
}

// tightenable reports whether b is a fixed-latency instruction of a class
// lat has a latency for, which is what stall tightening requires of an
// instruction and of the one after it.
func tightenable(b *Bundle, lat Latencies) bool {
	if b.variable() || b.Ctrl.ReadBarrier != NoBarrier || b.Class.Branch || b.Class.Fence {
		return false
	}
	_, ok := lat.Result[BaseOpcode(b.Inst.Opcode)]
	return ok
}

// splitScoreboards moves producers off shared write barriers; see
// [OptimizeBasicBlockControl]. It returns the number moved.
func splitScoreboards(cur []Bundle, try func(func([]Bundle)) bool) int {
	used := func() (free []uint8) {
		var busy [6]bool
		for _, b := range cur {
			for _, x := range []uint8{b.Ctrl.WriteBarrier, b.Ctrl.ReadBarrier} {
				if x < 6 {
					busy[x] = true
				}
			}
			for k := range 6 {
				if b.Ctrl.Waits(k) {
					busy[k] = true
				}
			}
		}
		for k := range uint8(6) {
			if !busy[k] {
				free = append(free, k)
			}
		}
		return free
	}
	moved := 0
	for q := range cur {
		if !cur[q].variable() {
			continue
		}
		bar := cur[q].Ctrl.WriteBarrier
		first := firstAccess(cur, q)
		if first < 0 {
			continue
		}
		early := -1
		for k := q + 1; k < first; k++ {
			if cur[k].Ctrl.Waits(int(bar)) {
				early = k
				break
			}
		}
		if early < 0 {
			continue
		}
		free := used()
		if len(free) == 0 {
			break
		}
		f := free[0]
		if try(func(c []Bundle) {
			c[q].Ctrl.WriteBarrier = f
			c[first].Ctrl.WaitMask |= 1 << f
		}) {
			moved++
		}
	}
	return moved
}

// firstAccess returns the index of the first instruction after q that reads
// or writes one of q's definitions, or -1.
func firstAccess(block []Bundle, q int) int {
	for c := q + 1; c < len(block); c++ {
		for _, r := range block[q].Defs {
			if has(block[c].Uses, r) || has(block[c].Defs, r) {
				return c
			}
		}
	}
	return -1
}

// reuseOpcodes are the opcodes whose source operands are plain register
// slots in order, for which a reuse flag is set by [OptimizeBasicBlockControl].
var reuseOpcodes = map[string]bool{"FFMA": true, "FADD": true, "FMUL": true, "IMAD": true, "IADD": true, "IADD3": true}

// reuseSlots returns the source slots a may mark for reuse by b.
func reuseSlots(a, b *Bundle) []uint8 {
	if !a.Ctrl.Yield || b.Ctrl.WaitMask != 0 || !reuseOpcodes[a.Inst.Opcode] || !reuseOpcodes[b.Inst.Opcode] {
		return nil
	}
	sa, sb := sourceSlots(a.Inst), sourceSlots(b.Inst)
	var out []uint8
	for s := 0; s < min(len(sa), len(sb), 3); s++ {
		r := sa[s]
		if r.Num < 0 || r != sb[s] || r.Sink() || has(a.Defs, r) || a.Ctrl.Reuse&(1<<s) != 0 {
			continue
		}
		out = append(out, uint8(s))
	}
	return out
}

// sourceSlots returns the source operands of inst in slot order. A slot
// that is not a plain GPR (an immediate, a constant, a negated or uniform
// operand) has Num -1.
func sourceSlots(inst Instruction) []Register {
	ops := splitOperands(inst.Operands)
	if len(ops) < 2 {
		return nil
	}
	var out []Register
	for _, tok := range ops[1:] {
		tok = strings.TrimSuffix(tok, ".reuse")
		r, ok := parseRegister(tok)
		if !ok || r.Class != GPR || strings.ContainsAny(tok, "-|[.") {
			r = Register{Class: GPR, Num: -1}
		}
		out = append(out, r)
	}
	return out
}

// A ControlPlan is the result of [OptimizeFunctionControl].
type ControlPlan struct {
	Plans   []MutationPlan // one per instruction whose control code changed
	Stats   ControlOptStats
	Skipped []error // one per basic block left unchanged, with the reason
}

// ControlOptions configures [OptimizeFunctionControl].
type ControlOptions struct {
	Latencies Latencies

	// Samples, if not nil, maps instruction addresses to stall samples, as
	// a PC-sampling profiler reports them. Only blocks holding at least
	// MinSamples samples are optimized, hottest first.
	Samples    map[uint64]uint64
	MinSamples uint64
}

// OptimizeFunctionControl runs [OptimizeBasicBlockControl] over the basic
// blocks of fn on arch and returns the changes as [MutationPlan] values for
// [CubinImage.Apply]. A block the model cannot analyse, or whose input
// schedule it rejects, is skipped and reported in Skipped.
//
// In each plan, Before and After differ only in the control field and
// Control is the new control code.
func OptimizeFunctionControl(fn *Function, arch Arch, opts ControlOptions) (ControlPlan, error) {
	if arch != SM121a {
		return ControlPlan{}, fmt.Errorf("control optimizer: %w", &ControlDecodeError{Arch: arch})
	}
	blocks := BasicBlocks(fn)
	heat := make([]uint64, len(blocks))
	order := make([]int, 0, len(blocks))
	for i, b := range blocks {
		for _, inst := range b.Instructions {
			heat[i] += opts.Samples[inst.Address]
		}
		if opts.Samples == nil || heat[i] >= max(opts.MinSamples, 1) {
			order = append(order, i)
		}
	}
	sort.SliceStable(order, func(a, b int) bool { return heat[order[a]] > heat[order[b]] })

	index := map[uint64]int{}
	for i, inst := range fn.Instructions {
		index[inst.Address] = i
	}
	var out ControlPlan
	for _, bi := range order {
		b := blocks[bi]
		bundles, err := Bundles(b, arch)
		if err == nil {
			var opt []Bundle
			var st ControlOptStats
			opt, st, err = OptimizeBasicBlockControl(bundles, opts.Latencies)
			if err == nil {
				out.Stats.add(st)
				for k := range opt {
					if opt[k].Ctrl == bundles[k].Ctrl {
						continue
					}
					out.Plans = append(out.Plans, MutationPlan{
						Function: fn.Name,
						Index:    index[opt[k].Inst.Address],
						Address:  opt[k].Inst.Address,
						FormFrom: opt[k].Inst.Opcode,
						FormTo:   opt[k].Inst.Opcode,
						Before:   bundles[k].Inst.Encoding,
						After:    opt[k].Inst.Encoding,
						Control:  opt[k].Ctrl,
					})
				}
				continue
			}
		}
		out.Skipped = append(out.Skipped, fmt.Errorf("block %#x-%#x: %w", b.Start, b.End, err))
	}
	sort.Slice(out.Plans, func(i, j int) bool { return out.Plans[i].Address < out.Plans[j].Address })
	return out, nil
}

// ErrControlChanged reports that a control-only plan changes more than the
// control field.
var ErrControlChanged = errors.New("plan changes bits outside the control field")

// CheckControlOnly reports whether every plan changes only the control
// field of its instruction on arch.
func CheckControlOnly(plans []MutationPlan, arch Arch) error {
	for _, p := range plans {
		c, err := DecodeControl(p.Before, arch)
		if err != nil {
			return err
		}
		back, err := EncodeControl(p.After, c, arch)
		if err != nil {
			return err
		}
		if back != p.Before {
			return fmt.Errorf("%w: at %#x", ErrControlChanged, p.Address)
		}
	}
	return nil
}
