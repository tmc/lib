package sassequiv

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// A fn is one function's value graph under construction.
type fn struct {
	g      *graph
	side   int
	f      *cudasass.Function
	blocks []cudasass.BasicBlock // reachable from the entry, in address order
	preds  [][]int
	edges  []cudasass.FlowEdge // between reachable blocks, renumbered
	outs   []map[string]int    // values each block defines, at its end
	ins    []map[string]int    // values read at each block's entry
	state  []int               // 0 not run, 1 running, 2 done
	phis   []phiRef
	roots  map[string]int
	why    string // why the function cannot be compared, if it cannot

	assumed map[[2]int]int // memoizes assume
}

type phiRef struct {
	id, block int
	key       string
}

// errUnknown carries the reason a function falls outside the model.
type errUnknown struct{ why string }

func unknownf(format string, args ...any) {
	panic(errUnknown{fmt.Sprintf(format, args...)})
}

// build constructs the value graph of f on side side of g.
func build(g *graph, side int, f *cudasass.Function) (fx *fn) {
	fx = &fn{g: g, side: side, f: f, roots: map[string]int{}}
	defer func() {
		if r := recover(); r != nil {
			e, ok := r.(errUnknown)
			if !ok {
				panic(r)
			}
			fx.why = e.why
		}
	}()
	cfg, err := cudasass.NewCFG(f)
	if err != nil {
		unknownf("control flow: %v", err)
	}
	if cfg.IsDegenerate {
		unknownf("indirect branch at %#x", cfg.Blocks[cfg.Indirect[0]].Instructions[len(cfg.Blocks[cfg.Indirect[0]].Instructions)-1].Address)
	}
	edges := subroutineEdges(cfg)
	// Number the blocks reachable from the entry in depth-first preorder,
	// fall-through edges first, so that two layouts of one control flow
	// graph, such as subroutines placed in another order, number alike.
	succ := make([][]cudasass.FlowEdge, len(cfg.Blocks))
	for _, e := range edges {
		succ[e.From] = append(succ[e.From], e)
	}
	index := make([]int, len(cfg.Blocks))
	for i := range index {
		index[i] = -1
	}
	var order []int
	var visit func(int)
	visit = func(b int) {
		index[b] = len(order)
		order = append(order, b)
		ss := slices.Clone(succ[b])
		slices.SortStableFunc(ss, func(p, q cudasass.FlowEdge) int { return int(p.Kind) - int(q.Kind) })
		for _, e := range ss {
			if index[e.To] < 0 {
				visit(e.To)
			}
		}
	}
	visit(0)
	for _, b := range order {
		fx.blocks = append(fx.blocks, cfg.Blocks[b])
	}
	n := len(fx.blocks)
	fx.preds = make([][]int, n)
	for _, e := range edges {
		if index[e.From] < 0 {
			continue
		}
		e := cudasass.FlowEdge{From: index[e.From], To: index[e.To], Kind: e.Kind}
		fx.edges = append(fx.edges, e)
		if !slices.Contains(fx.preds[e.To], e.From) {
			fx.preds[e.To] = append(fx.preds[e.To], e.From)
		}
	}
	for _, p := range fx.preds {
		slices.Sort(p)
	}
	slices.SortFunc(fx.edges, func(a, b cudasass.FlowEdge) int {
		if a.From != b.From {
			return a.From - b.From
		}
		if a.To != b.To {
			return a.To - b.To
		}
		return int(a.Kind) - int(b.Kind)
	})
	fx.outs = make([]map[string]int, n)
	fx.ins = make([]map[string]int, n)
	fx.state = make([]int, n)
	for b := range n {
		fx.ins[b] = map[string]int{}
	}
	for b := range n {
		fx.run(b)
	}
	// Fill phis; filling can create more.
	for i := 0; i < len(fx.phis); i++ {
		p := fx.phis[i]
		var args []int
		if p.block == 0 {
			args = append(args, fx.entry(p.key))
		}
		for _, q := range fx.preds[p.block] {
			args = append(args, fx.out(q, p.key))
		}
		fx.g.nodes[p.id].args = args
	}
	return fx
}

// entry is the value of key when the function starts.
func (fx *fn) entry(key string) int {
	return fx.g.add(fx.side, fx.f.BaseAddress, "in:"+key, 0)
}

// in is the value of key at the start of block b.
func (fx *fn) in(b int, key string) int {
	if v, ok := fx.ins[b][key]; ok {
		return v
	}
	var v int
	if b != 0 && len(fx.preds[b]) == 1 {
		v = fx.out(fx.preds[b][0], key)
	} else {
		v = fx.g.fresh(fx.side, fx.blocks[b].Start, "phi:"+strconv.Itoa(b))
		fx.phis = append(fx.phis, phiRef{v, b, key})
	}
	fx.ins[b][key] = v
	return v
}

// out is the value of key at the end of block b.
func (fx *fn) out(b int, key string) int {
	fx.run(b)
	if v, ok := fx.outs[b][key]; ok {
		return v
	}
	return fx.in(b, key)
}

func (fx *fn) run(b int) {
	switch fx.state[b] {
	case 2:
		return
	case 1:
		unknownf("block %d reads itself through single-predecessor blocks", b)
	}
	fx.state[b] = 1
	x := &blockExec{fx: fx, b: b, st: map[string]int{}, guard: -1}
	for _, in := range fx.blocks[b].Instructions {
		x.inst(in)
	}
	fx.outs[b] = x.st
	fx.state[b] = 2
}

// A blockExec runs one block's instructions symbolically.
type blockExec struct {
	fx *fn
	b  int
	st map[string]int
	pc uint64

	guard int // the predicate of the instruction being run, or -1
}

func (x *blockExec) add(label string, comm int, args ...int) int {
	return x.fx.g.add(x.fx.side, x.pc, label, comm, args...)
}

func (x *blockExec) leaf(label string) int { return x.add(label, 0) }

func (x *blockExec) imm(v uint64) int { return x.leaf(fmt.Sprintf("imm:%#x", v)) }

func (x *blockExec) read(key string) int {
	switch {
	case zero(key):
		return x.imm(0)
	case key == "PT" || key == "UPT":
		return x.leaf("true")
	}
	if v, ok := x.st[key]; ok {
		return v
	}
	return x.fx.in(x.b, key)
}

func (x *blockExec) write(key string, v, guard int) {
	if zero(key) || key == "PT" || key == "UPT" {
		return
	}
	if guard >= 0 {
		if old := x.read(key); old != v {
			v = x.add("sel", 0, guard, v, old)
		}
	}
	x.st[key] = v
}

func (x *blockExec) not(v int) int {
	n := x.fx.g.nodes[v]
	switch {
	case n.label == "true":
		return x.leaf("false")
	case n.label == "false":
		return x.leaf("true")
	case n.label == "not":
		return n.args[0]
	}
	return x.add("not", 0, v)
}

// Opcodes that do not affect values, memory or control flow as modeled.
var ignored = map[string]bool{
	"NOP": true, "BSSY": true, "BSYNC": true, "WARPSYNC": true, "YIELD": true,
	"DEPBAR": true, "LDGDEPBAR": true, "WARPGROUP": true, "ERRBAR": true,
	"CGAERRBAR": true, "CCTL": true, "ENDCOLLECTIVE": true, "NANOSLEEP": true,
	"PMTRIG": true, "BMOV": true,
}

// Opcodes that change memory or order it, in program order.
var effects = map[string]bool{
	"ST": true, "STG": true, "STS": true, "STL": true, "RED": true, "REDG": true,
	"ATOM": true, "ATOMG": true, "ATOMS": true, "BAR": true, "MEMBAR": true,
	"FENCE": true, "UCGABAR_ARV": true, "UCGABAR_WAIT": true, "SYNCS": true,
	"LDGSTS": true, "UTMALDG": true, "UTMASTG": true, "UBLKCP": true,
	"UTMACMDFLUSH": true, "UTMAPF": true, "ARRIVES": true, "STSM": true,
}

// Opcodes that read memory into registers.
var loads = map[string]bool{
	"LD": true, "LDG": true, "LDS": true, "LDL": true, "LDSM": true, "LDTRAM": true, "HGMMA": true, "QGMMA": true, "IGMMA": true,
	"BGMMA": true,
}

// Opcodes the model cannot represent.
var unmodeled = map[string]bool{
	"BRX": true, "BRXU": true, "JMX": true, "JMXU": true,
	"KILL": true, "BPT": true, "BREAK": true, "WARPSYNC.EXCLUSIVE": true,
}

// Special registers that read differently each time.
var volatileSR = regexp.MustCompile(`^SR_(CLOCK|GLOBALTIMER)`)

// Commuting operands: how many leading sources of an opcode commute.
func commuting(op, base string) int {
	switch base {
	case "FADD", "FMUL", "FFMA", "DADD", "DMUL", "DFMA", "HADD2", "HMUL2",
		"HFMA2", "IMAD", "FMNMX", "IMNMX", "VIMNMX", "UIMAD", "IMUL", "DMNMX":
		return 2
	case "IADD3", "UIADD3":
		if strings.Contains(op, ".X") {
			return 0
		}
		return 3
	}
	return 0
}

func (x *blockExec) inst(in cudasass.Instruction) {
	x.pc = in.Address
	x.guard = -1
	guard := -1
	switch p := strings.TrimSpace(in.Predicate); p {
	case "", "@PT", "@UPT":
	case "@!PT", "@!UPT":
		return
	default:
		o := parseOperand(strings.TrimPrefix(p, "@"))
		guard = x.read(o.core)
		if strings.Contains(o.prefix, "!") {
			guard = x.not(guard)
		}
		x.guard = guard
	}
	op := in.Opcode
	base := cudasass.BaseOpcode(op)
	ops := splitOperands(in.Operands)
	if guard < 0 {
		guard = -1
	}
	switch {
	case ignored[base]:
		return
	case unmodeled[base]:
		unknownf("unmodeled instruction %s at %#x", op, in.Address)
	case base == "BRA" || base == "JMP" || base == "RET" || base == "CALL":
		x.fx.roots[fmt.Sprintf("branch:%d", x.b)] = x.cond(guard)
		return
	case base == "EXIT":
		x.fx.roots[fmt.Sprintf("exit:%d", x.b)] = x.add("exit", 0, x.cond(guard), x.read("MEM"))
		return
	}
	if strings.Contains(base, "MMA") && !loads[base] && base != "HMMA" {
		unknownf("unmodeled instruction %s at %#x", op, in.Address)
	}
	if x.move(op, base, ops, guard) {
		return
	}
	switch base {
	case "STL", "LDL":
		if x.local(op, base, ops, guard) {
			return
		}
	case "P2R":
		x.p2r(ops, guard)
		return
	case "R2P":
		x.r2p(ops, guard)
		return
	case "PLOP3", "UPLOP3":
		if x.plop3(ops, guard) {
			return
		}
	}
	dsts, srcs := destinations(base, ops)
	widths := srcWidths(op, base, srcs)
	var args []int
	for i, s := range srcs {
		args = append(args, x.val(s, widths[i]))
	}
	comm := min(commuting(op, base), len(args))
	if strings.HasPrefix(op, "IMAD.WIDE") || strings.HasPrefix(op, "UIMAD.WIDE") {
		comm = min(2, len(args))
	}
	label := canonical(op, base)
	label, comm = x.comparison(label, args, comm)
	if loads[base] || volatileSR.MatchString(strings.Join(srcs, " ")) {
		args = append(args, x.read("MEM"))
	}
	if effects[base] {
		mem := x.add(label, 0, append([]int{x.cond(guard), x.read("MEM")}, args...)...)
		x.st["MEM"] = mem
		x.writeDsts(op, base, dsts, x.add("result", 0, mem), -1)
		return
	}
	if len(dsts) == 1 && !isPred(dsts[0].core) {
		if v, ok := x.integer(label, args); ok {
			x.writeDsts(op, base, dsts, v, guard)
			return
		}
	}
	if (label == "LOP3.LUT" || label == "PLOP3.LUT") && len(dsts) == 2 && isPred(dsts[0].core) && !isPred(dsts[1].core) {
		// LOP3.LUT P, R, ...: R is the result, P whether it is nonzero.
		v := x.add(label, comm, args...)
		x.write(dsts[1].core, v, guard)
		x.write(dsts[0].core, x.add("ISETP.NE.AND", 2, v, x.imm(0), x.leaf("true")), guard)
		return
	}
	v := x.add(label, comm, args...)
	x.writeDsts(op, base, dsts, v, guard)
}

// cond is guard as a value: true when the instruction is unguarded.
func (x *blockExec) cond(guard int) int {
	if guard < 0 {
		return x.leaf("true")
	}
	return guard
}

func (x *blockExec) writeDsts(op, base string, dsts []operand, v, guard int) {
	for i, d := range dsts {
		w := 1
		if i == 0 && !isPred(d.core) {
			w = dstWidth(op, base)
		}
		if w < 0 {
			unknownf("%s at %#x: unknown destination width", op, x.pc)
		}
		for k := range w {
			part := v
			if w > 1 || i > 0 {
				part = x.part(v, i*1000+k)
			}
			x.write(regN(d.core, k), part, guard)
		}
	}
}

// move handles register and constant moves, which carry their source's
// value: ptxas materializes the same constant with MOV, IMAD.MOV, HFMA2
// or UMOV, and copies values between register files freely.
func (x *blockExec) move(op, base string, ops []string, guard int) bool {
	if len(ops) < 2 {
		return false
	}
	d := parseOperand(ops[0])
	if d.prefix != "" || d.abs || d.suffix != "" || !isReg(d.core) || isPred(d.core) {
		return false
	}
	var src string
	switch {
	case base == "LDC" || base == "ULDC":
		if len(ops) != 2 {
			return false
		}
		w := widthOf(op)
		v := x.val(ops[1], w)
		if w == 1 {
			x.write(d.core, v, guard)
			return true
		}
		for k := range w {
			x.write(regN(d.core, k), x.part(v, k), guard)
		}
		return true
	case (op == "S2R" || op == "S2UR") && len(ops) == 2:
		src = ops[1]
	case op == "MOV" || op == "UMOV" || op == "R2UR" || op == "MOV.SPILL":
		if len(ops) != 2 && !(len(ops) == 3 && ops[2] == "0xf") {
			return false
		}
		src = ops[1]
	case (op == "IMAD.MOV.U32" || op == "IMAD.MOV" || op == "UIMAD.MOV.U32") && len(ops) == 4 && zero(ops[1]) && zero(ops[2]):
		src = ops[3]
	case (op == "HFMA2.MMA" || op == "HFMA2") && len(ops) == 5 && (ops[1] == "-RZ" || ops[1] == "RZ") && ops[2] == "RZ":
		hi, ok1 := half(ops[3])
		lo, ok2 := half(ops[4])
		if !ok1 || !ok2 {
			return false
		}
		x.write(d.core, x.imm(hi<<16|lo), guard)
		return true
	case op == "CS2R" && len(ops) == 2 && ops[1] == "SRZ":
		x.write(d.core, x.imm(0), guard)
		x.write(regN(d.core, 1), x.imm(0), guard)
		return true
	default:
		return false
	}
	s := parseOperand(src)
	if s.prefix != "" || s.abs || s.suffix != "" {
		return false
	}
	x.write(d.core, x.val(src, 1), guard)
	return true
}

// local treats a stack slot at a constant offset from R1, the stack
// pointer, as a register, so spill code and the slot ptxas picks do not
// matter. It reports false for any other local access.
func (x *blockExec) local(op, base string, ops []string, guard int) bool {
	if len(ops) != 2 {
		return false
	}
	addr, reg := ops[0], ops[1]
	if base == "LDL" {
		addr, reg = ops[1], ops[0]
	}
	m := regexp.MustCompile(`^\[R1(?:\+(0x[0-9a-f]+))?\]$`).FindStringSubmatch(addr)
	if m == nil {
		return false
	}
	off := uint64(0)
	if m[1] != "" {
		off, _ = strconv.ParseUint(m[1][2:], 16, 64)
	}
	w := widthOf(op)
	r := parseOperand(reg)
	for k := range w {
		slot := fmt.Sprintf("L%#x", off+uint64(4*k))
		if base == "STL" {
			x.write(slot, x.val(regN(r.core, k), 1), guard)
		} else {
			x.write(regN(r.core, k), x.read(slot), guard)
		}
	}
	return true
}

func (x *blockExec) p2r(ops []string, guard int) {
	if len(ops) != 4 {
		unknownf("P2R at %#x: %d operands", x.pc, len(ops))
	}
	args := []int{x.val(ops[2], 1), x.val(ops[3], 1)}
	if strings.TrimSpace(ops[1]) == "PR" {
		for i := range 7 {
			args = append(args, x.read("P"+strconv.Itoa(i)))
		}
	} else {
		args = append(args, x.val(ops[1], 1))
	}
	x.write(parseOperand(ops[0]).core, x.add("P2R", 0, args...), guard)
}

func (x *blockExec) r2p(ops []string, guard int) {
	if len(ops) != 3 || strings.TrimSpace(ops[0]) != "PR" {
		unknownf("R2P at %#x", x.pc)
	}
	src, mask := x.val(ops[1], 1), x.val(ops[2], 1)
	for i := range 7 {
		p := "P" + strconv.Itoa(i)
		x.write(p, x.add("R2P:"+strconv.Itoa(i), 0, src, mask, x.read(p)), guard)
	}
}

// destinations splits ops into the registers an instruction writes and its
// sources. The first operand is a destination when it is a register; a
// predicate destination may be followed by a second destination (ISETP's
// second predicate, SHFL's register), and a register destination by
// predicates it also writes (IADD3 and LEA carries, VOTE).
func destinations(base string, ops []string) (dsts []operand, srcs []string) {
	if len(ops) == 0 || effects[base] && base != "ATOM" && base != "ATOMG" && base != "ATOMS" && base != "SYNCS" {
		return nil, ops
	}
	first := parseOperand(ops[0])
	if !isReg(first.core) || first.core == "PR" || first.prefix != "" {
		return nil, ops
	}
	dsts = append(dsts, first)
	i := 1
	if isPred(first.core) {
		if base != "FCHK" && len(ops) > 1 {
			if o := parseOperand(ops[1]); isReg(o.core) && o.core != "PR" && o.prefix == "" {
				dsts = append(dsts, o)
				i = 2
			}
		}
	} else {
		for ; i < len(ops); i++ {
			o := parseOperand(ops[i])
			if !isReg(o.core) || !isPred(o.core) || o.core == "PR" || o.prefix != "" {
				break
			}
			dsts = append(dsts, o)
		}
	}
	return dsts, ops[i:]
}

var mmaShape = regexp.MustCompile(`^[HQIB]GMMA\.64x(\d+)x\d+\.(F32|F16|S32)`)

// widthOf returns the register count a .64 or .128 opcode moves.
func widthOf(op string) int {
	switch {
	case strings.Contains(op, ".128"):
		return 4
	case strings.Contains(op, ".64"):
		return 2
	}
	return 1
}

// dstWidth returns how many consecutive registers the first destination
// spans, or -1 when the opcode is not understood well enough to say.
func dstWidth(op, base string) int {
	switch {
	case loads[base] && strings.HasSuffix(base, "GMMA"):
		m := mmaShape.FindStringSubmatch(op)
		if m == nil {
			return -1
		}
		n, _ := strconv.Atoi(m[1])
		if m[2] == "F16" {
			return n / 4
		}
		return n / 2
	case base == "HMMA":
		if strings.HasPrefix(op, "HMMA.16816.F32") || strings.HasPrefix(op, "HMMA.1688.F32") {
			return 4
		}
		if strings.HasPrefix(op, "HMMA.16816.F16") || strings.HasPrefix(op, "HMMA.1688.F16") {
			return 2
		}
		return -1
	case base == "LDSM" || base == "STSM":
		switch {
		case strings.HasSuffix(op, ".4"):
			return 4
		case strings.HasSuffix(op, ".2"):
			return 2
		}
		return 1
	case strings.HasPrefix(op, "IMAD.WIDE") || strings.HasPrefix(op, "UIMAD.WIDE"):
		return 2
	case base == "CS2R":
		if strings.Contains(op, ".32") {
			return 1
		}
		return 2
	case base == "DADD" || base == "DMUL" || base == "DFMA" || base == "DMNMX":
		return 2
	case base == "F2F" || base == "I2F" || base == "F2I" || base == "I2FP" || base == "F2IP":
		if t := types(op); len(t) > 0 && strings.HasSuffix(t[0], "64") {
			return 2
		}
		return 1
	}
	return widthOf(op)
}

// types returns the type tokens of a conversion opcode, destination first.
func types(op string) []string {
	var t []string
	for _, s := range strings.Split(op, ".")[1:] {
		if regexp.MustCompile(`^[FSU](8|16|32|64)$|^BF16$|^E4M3$|^E5M2$`).MatchString(s) {
			t = append(t, s)
		}
	}
	return t
}

// srcWidths returns the register count of each source operand.
func srcWidths(op, base string, srcs []string) []int {
	n := len(srcs)
	w := make([]int, n)
	for i := range w {
		w[i] = 1
	}
	set := func(i, v int) {
		if i < n {
			w[i] = v
		}
	}
	switch {
	case base == "DADD" || base == "DMUL" || base == "DFMA" || base == "DMNMX" || base == "DSETP":
		for i := range w {
			w[i] = 2
		}
	case strings.HasPrefix(op, "IMAD.WIDE") || strings.HasPrefix(op, "UIMAD.WIDE"):
		set(2, 2)
	case base == "F2F" || base == "F2I" || base == "F2IP" || base == "I2F" || base == "I2FP":
		if t := types(op); len(t) > 1 && strings.HasSuffix(t[1], "64") {
			set(0, 2)
		}
	case base == "HMMA":
		c := dstWidth(op, base)
		switch {
		case strings.HasPrefix(op, "HMMA.16816"):
			set(0, 4)
			set(1, 2)
		case strings.HasPrefix(op, "HMMA.1688"):
			set(0, 2)
			set(1, 1)
		}
		set(2, c)
	case strings.HasSuffix(base, "GMMA") && loads[base]:
		// A in registers (4) or with B in a gdesc, then C.
		d := dstWidth(op, base)
		if n > 0 && strings.HasPrefix(srcs[0], "gdesc") {
			set(1, d)
		} else {
			set(0, 4)
			set(2, d)
		}
	case base == "ST" || base == "STG" || base == "STS" || base == "STL" || base == "STSM":
		set(1, max(widthOf(op), dstWidth(op, base)))
	case base == "ATOM" || base == "ATOMG" || base == "ATOMS" || base == "RED" || base == "REDG":
		for i := 1; i < n; i++ {
			set(i, widthOf(op))
		}
	}
	return w
}

var cbankRE = regexp.MustCompile(`^c\[(0x[0-9a-f]+)\]\[(.*)\]$`)

// val returns the value of a source operand spanning width registers.
func (x *blockExec) val(s string, width int) int {
	o := parseOperand(s)
	if o.prefix == "-" && !o.abs && o.suffix == "" {
		if v, ok := immediate("-" + o.core); ok {
			return x.imm(v)
		}
	}
	if o.prefix == "!" && isPred(o.core) {
		return x.not(x.core(o, 1))
	}
	v := x.core(o, width)
	if x.guard >= 0 {
		v = x.assume(v, x.guard)
	}
	if o.prefix != "" || o.abs || (o.suffix != "" && o.suffix != ".64") {
		label := "mod:" + o.prefix
		if o.abs {
			label += "|abs|"
		}
		v = x.add(label+o.suffix, 0, v)
	}
	return v
}

func (x *blockExec) core(o operand, width int) int {
	c := o.core
	if o.suffix == ".64" {
		width = max(width, 2)
	}
	switch {
	case c == "PR":
		unknownf("PR operand at %#x", x.pc)
	case isReg(c):
		if isPred(c) || width <= 1 {
			return x.read(c)
		}
		parts := make([]int, width)
		for k := range parts {
			parts[k] = x.read(regN(c, k))
		}
		return x.vec(parts)
	case sregRE.MatchString(c):
		if volatileSR.MatchString(c) {
			return x.add(c, 0, x.read("MEM"))
		}
		return x.leaf(c)
	case strings.HasPrefix(c, "c["):
		m := cbankRE.FindStringSubmatch(c)
		if m == nil {
			return x.leaf("tok:" + c)
		}
		if off, err := strconv.ParseUint(strings.TrimPrefix(m[2], "0x"), 16, 64); err == nil && strings.HasPrefix(m[2], "0x") {
			parts := make([]int, max(width, 1))
			for k := range parts {
				parts[k] = x.leaf(fmt.Sprintf("c[%s][%#x]", m[1], off+uint64(4*k)))
			}
			if len(parts) == 1 {
				return parts[0]
			}
			return x.add("vec", 0, parts...)
		}
		return x.add("ldc:"+m[1], 0, x.addr(m[2]))
	case strings.HasPrefix(c, "desc["):
		i := strings.Index(c, "][")
		if i < 0 || !strings.HasSuffix(c, "]") {
			return x.leaf("tok:" + c)
		}
		desc := x.val(c[len("desc["):i], 2)
		return x.add("desc", 0, desc, x.addr(c[i+2:len(c)-1]))
	case strings.HasPrefix(c, "gdesc["):
		end := strings.Index(c, "]")
		r := parseOperand(c[len("gdesc["):end])
		label := "gdesc" + c[end+1:]
		return x.add(label, 0, x.core(r, 4), x.read("MEM"))
	case strings.HasPrefix(c, "["):
		return x.addr(strings.TrimSuffix(strings.TrimPrefix(c, "["), "]"))
	}
	if v, ok := immediate(c); ok {
		return x.imm(v)
	}
	return x.leaf("tok:" + c)
}

// addr returns an address expression: for 32-bit operands the linear
// form of their sum and offset, so where ptxas splits a constant between
// a register and the offset does not matter; a 64-bit base keeps its
// offset beside it.
func (x *blockExec) addr(s string) int {
	s = strings.ReplaceAll(s, "+-", "+~")
	sum := linear{terms: map[int]uint32{}}
	var wide []int
	for _, t := range strings.Split(s, "+") {
		t = strings.TrimSpace(t)
		if t == "" {
			continue
		}
		if strings.HasPrefix(t, "~") {
			t = "-" + t[1:]
		}
		if v, ok := immediate(t); ok && strings.Contains(t, "0x") {
			sum.k += uint32(v)
			continue
		}
		o := parseOperand(t)
		if zero(o.core) {
			continue
		}
		if o.suffix == ".64" {
			wide = append(wide, x.val(t, 2))
			continue
		}
		sum = sum.plus(x.lin(x.val(t, 1)))
	}
	if len(wide) == 0 {
		return x.add("addr", 0, x.node(sum))
	}
	return x.add("addr", len(wide), append(wide, x.node(sum))...)
}

// subroutineEdges returns cfg's edges with calls followed: a call block
// flows into its subroutine instead of the instruction after it, and a
// block ending in RET flows to every instruction after a call. Merging
// the return paths of all call sites admits paths no execution takes,
// but both functions under comparison admit the same ones.
func subroutineEdges(cfg *cudasass.CFG) []cudasass.FlowEdge {
	last := func(b int) cudasass.Instruction {
		ins := cfg.Blocks[b].Instructions
		return ins[len(ins)-1]
	}
	calls := map[int]bool{}
	for _, e := range cfg.Edges {
		if e.Kind == cudasass.Call {
			calls[e.From] = true
		}
	}
	for b := range cfg.Blocks {
		if cudasass.BaseOpcode(last(b).Opcode) == "CALL" && !calls[b] {
			unknownf("call at %#x to a function outside this one", last(b).Address)
		}
	}
	var edges []cudasass.FlowEdge
	for _, e := range cfg.Edges {
		if calls[e.From] && e.Kind == cudasass.FallThrough {
			continue
		}
		edges = append(edges, e)
	}
	for b := range cfg.Blocks {
		if cudasass.BaseOpcode(last(b).Opcode) != "RET" {
			continue
		}
		for c := range calls {
			if c+1 < len(cfg.Blocks) {
				edges = append(edges, cudasass.FlowEdge{From: b, To: c + 1, Kind: cudasass.Taken})
			}
		}
	}
	slices.SortFunc(edges, func(a, b cudasass.FlowEdge) int {
		if a.From != b.From {
			return a.From - b.From
		}
		if a.To != b.To {
			return a.To - b.To
		}
		return int(a.Kind) - int(b.Kind)
	})
	return slices.Compact(edges)
}
