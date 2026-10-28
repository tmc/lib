package qualify

import (
	"fmt"
	"math"
	"math/bits"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// Determinism rules reported in [DeterminismViolation.Rule].
const (
	RuleNoFPAtomics        = "NO_FP_ATOMICS"        // floating-point add through atom, red, multimem or bulk reduce
	RuleBatchInvariantTile = "BATCH_INVARIANT_TILE" // reduction chunking that depends on batch size
	RulePhiloxSRSeed       = "PHILOX_SR_SEED"       // stochastic-rounding bits derived from timers or scheduling state
	RuleNondetEntropy      = "NONDET_ENTROPY"       // timer or scheduling state flowing into floating-point data
)

// A DeterminismViolation describes a PTX instruction or schedule choice that
// breaks bit-reproducibility.
type DeterminismViolation struct {
	Rule     string // one of the Rule constants
	Location string // "line N" for PTX, "batch N" for schedules
	Detail   string
}

func (v DeterminismViolation) String() string {
	return v.Rule + " at " + v.Location + ": " + v.Detail
}

// DeterministicConfig controls bit-exact, batch-invariant execution.
type DeterministicConfig struct {
	GlobalSeed   uint64
	PhiloxRounds int // rounds of Philox4x32 in emitted code; 0 means 7
}

func (c DeterministicConfig) rounds() int {
	if c.PhiloxRounds == 0 {
		return 7
	}
	return c.PhiloxRounds
}

// Special registers whose values depend on timing or on where the hardware
// scheduled a thread rather than on which data the thread handles.
var nondetSpecial = []string{"%clock", "%clock64", "%clock_hi", "%globaltimer", "%globaltimer_lo", "%globaltimer_hi", "%smid", "%warpid", "%nwarpid"}

var fpTypes = map[string]bool{
	"f16": true, "f16x2": true, "bf16": true, "bf16x2": true, "f32": true, "f64": true,
	"e4m3x2": true, "e5m2x2": true, "e4m3x4": true, "e5m2x4": true,
}

// AuditPTXDeterminism scans PTX text for instructions that make results
// depend on execution order or timing:
//
//   - floating-point add reductions through atom, red, red.async,
//     multimem.red, multimem.ld_reduce or cp.reduce.async.bulk
//     ([RuleNoFPAtomics]); min and max reductions and integer atomics are
//     order-independent and allowed;
//   - cvt.rs whose random-bits operand is derived from %clock,
//     %globaltimer, %smid or %warpid ([RulePhiloxSRSeed]);
//   - any such register flowing into a floating-point arithmetic result
//     ([RuleNondetEntropy]).
//
// Timer reads that only feed integer compares, branches and integer stores,
// as in a timeout loop, are allowed. The data-flow analysis is a forward
// taint over virtual registers in program order; it does not follow loops
// back or track memory.
func AuditPTXDeterminism(ptx string, cfg DeterministicConfig) ([]DeterminismViolation, error) {
	if cfg.PhiloxRounds < 0 {
		return nil, fmt.Errorf("audit ptx determinism: negative PhiloxRounds %d", cfg.PhiloxRounds)
	}
	var out []DeterminismViolation
	tainted := map[string]bool{}
	isTainted := func(op string) bool {
		for _, r := range ptxRegs(op) {
			if tainted[r] || slices.Contains(nondetSpecial, r) {
				return true
			}
			if dot := strings.IndexByte(r, '.'); dot > 0 && slices.Contains(nondetSpecial, r[:dot]) {
				return true
			}
		}
		return false
	}
	for _, st := range ptxStatements(ptx) {
		mn, ops := splitInstruction(st.text)
		if mn == "" {
			continue
		}
		parts := strings.Split(mn, ".")
		loc := fmt.Sprintf("line %d", st.line)
		if isFPAddReduce(parts) {
			out = append(out, DeterminismViolation{Rule: RuleNoFPAtomics, Location: loc, Detail: mn})
		}
		if len(ops) == 0 {
			continue
		}
		srcTainted := false
		for _, op := range ops[1:] {
			if isTainted(op) {
				srcTainted = true
			}
		}
		switch {
		case parts[0] == "cvt" && slices.Contains(parts, "rs"):
			if isTainted(ops[len(ops)-1]) {
				out = append(out, DeterminismViolation{Rule: RulePhiloxSRSeed, Location: loc, Detail: "random bits of " + mn + " derive from timer or scheduling state"})
			}
		case srcTainted && isFPArith(parts):
			out = append(out, DeterminismViolation{Rule: RuleNondetEntropy, Location: loc, Detail: mn + " reads timer or scheduling state"})
		}
		if srcTainted && !isStore(parts[0]) {
			for _, r := range ptxRegs(ops[0]) {
				tainted[r] = true
			}
		}
	}
	return out, nil
}

func isStore(op string) bool {
	return op == "st" || op == "red" || op == "stmatrix" || op == "bar" || op == "barrier"
}

func isFPAddReduce(parts []string) bool {
	switch {
	case parts[0] == "atom", parts[0] == "red":
	case parts[0] == "multimem" && len(parts) > 1 && (parts[1] == "red" || parts[1] == "ld_reduce"):
	case parts[0] == "cp" && len(parts) > 1 && parts[1] == "reduce":
	default:
		return false
	}
	if !slices.Contains(parts, "add") {
		return false
	}
	for _, p := range parts {
		if fpTypes[p] || p == "noftz" {
			return true
		}
	}
	return false
}

func isFPArith(parts []string) bool {
	switch parts[0] {
	case "add", "sub", "mul", "fma", "mad", "div", "rcp", "sqrt", "rsqrt", "ex2", "lg2", "min", "max", "cvt":
	default:
		return false
	}
	for _, p := range parts[1:] {
		if fpTypes[p] {
			return true
		}
	}
	return false
}

type ptxStatement struct {
	line int
	text string
}

// ptxStatements strips comments and splits PTX into ';'-terminated
// statements, each tagged with the line it starts on. Labels and
// braces are dropped.
func ptxStatements(src string) []ptxStatement {
	var out []ptxStatement
	var cur strings.Builder
	line, start := 1, 1
	for i := 0; i < len(src); i++ {
		c := src[i]
		switch {
		case c == '/' && i+1 < len(src) && src[i+1] == '/':
			for i < len(src) && src[i] != '\n' {
				i++
			}
			i--
			continue
		case c == '/' && i+1 < len(src) && src[i+1] == '*':
			i += 2
			for i+1 < len(src) && !(src[i] == '*' && src[i+1] == '/') {
				if src[i] == '\n' {
					line++
				}
				i++
			}
			i++
			continue
		case c == '\n':
			line++
		}
		if c == ';' {
			s := strings.TrimSpace(cur.String())
			if s != "" {
				out = append(out, ptxStatement{line: start, text: s})
			}
			cur.Reset()
			continue
		}
		if strings.TrimSpace(cur.String()) == "" {
			start = line
		}
		cur.WriteByte(c)
	}
	return out
}

// splitInstruction returns the mnemonic and comma-separated operands of a
// PTX statement, dropping any label, brace and predicate guard.
func splitInstruction(s string) (string, []string) {
	for {
		s = strings.TrimLeft(s, " \t\n{}")
		i := strings.IndexByte(s, ':')
		if i <= 0 || strings.IndexFunc(s[:i], notIdent) >= 0 {
			break
		}
		s = s[i+1:]
	}
	s = strings.TrimLeft(s, " \t\n{}")
	if strings.HasPrefix(s, "@") {
		i := strings.IndexAny(s, " \t")
		if i < 0 {
			return "", nil
		}
		s = strings.TrimSpace(s[i:])
	}
	if s == "" || s[0] == '.' {
		return "", nil
	}
	i := strings.IndexAny(s, " \t")
	if i < 0 {
		return s, nil
	}
	mn, rest := s[:i], s[i:]
	var ops []string
	depth, from := 0, 0
	for j := 0; j < len(rest); j++ {
		switch rest[j] {
		case '{', '[', '(':
			depth++
		case '}', ']', ')':
			depth--
		case ',':
			if depth == 0 {
				ops = append(ops, strings.TrimSpace(rest[from:j]))
				from = j + 1
			}
		}
	}
	ops = append(ops, strings.TrimSpace(rest[from:]))
	return mn, ops
}

func notIdent(r rune) bool {
	return !(r == '_' || r == '$' || r >= '0' && r <= '9' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z')
}

// ptxRegs returns the %-prefixed register names in a PTX operand.
func ptxRegs(op string) []string {
	var out []string
	for i := 0; i < len(op); i++ {
		if op[i] != '%' {
			continue
		}
		j := i + 1
		for j < len(op) && (op[j] == '_' || op[j] == '.' || op[j] >= '0' && op[j] <= '9' || op[j] >= 'a' && op[j] <= 'z' || op[j] >= 'A' && op[j] <= 'Z') {
			j++
		}
		out = append(out, op[i:j])
		i = j - 1
	}
	return out
}

// Philox4x32 constants from Salmon et al., "Parallel random numbers: as easy
// as 1, 2, 3" (SC11).
const (
	philoxM0 = 0xD2511F53
	philoxM1 = 0xCD9E8D57
	philoxW0 = 0x9E3779B9
	philoxW1 = 0xBB67AE85
)

// Philox4x32 computes rounds rounds of the Philox4x32 counter-based
// bijection, as the code from [EmitDeterministicStochasticRoundPTX] and
// [DeterministicCUDA] does. With rounds = 10 it matches the Random123
// reference generator philox4x32_10.
func Philox4x32(counter [4]uint32, key [2]uint32, rounds int) [4]uint32 {
	c := counter
	k := key
	for r := range rounds {
		if r > 0 {
			k[0] += philoxW0
			k[1] += philoxW1
		}
		hi0, lo0 := bits.Mul32(philoxM0, c[0])
		hi1, lo1 := bits.Mul32(philoxM1, c[2])
		c = [4]uint32{hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0}
	}
	return c
}

// StochasticRoundKey returns the Philox key for a layer: the low seed word
// XOR layer, and the high seed word.
func StochasticRoundKey(seed uint64, layer uint32) [2]uint32 {
	return [2]uint32{uint32(seed) ^ layer, uint32(seed >> 32)}
}

// StochasticRoundBits returns the 32 random bits that the deterministic
// stochastic-rounding code feeds cvt.rs for the four columns starting at
// col (a multiple of 4) of token: the first word of
// Philox4x32(ctr = {token, col, 0, 0}, key = StochasticRoundKey(seed, layer)).
// The bits depend only on (seed, layer, token, col), not on batch packing
// or on which thread or CTA converts the values.
func StochasticRoundBits(cfg DeterministicConfig, layer, token, col uint32) uint32 {
	return Philox4x32([4]uint32{token, col, 0, 0}, StochasticRoundKey(cfg.GlobalSeed, layer), cfg.rounds())[0]
}

var e2m1Grid = [8]float64{0, 0.5, 1, 1.5, 2, 3, 4, 6}

// StochasticRoundE2M1x4 models cvt.rs.satfinite.e2m1x4.f32 d, {a, b, e, f},
// rbits on finite inputs and returns the packed 16-bit result: a in bits
// 15:12, b in 11:8, e in 7:4 and f in 3:0.
//
// The model follows the PTX ISA 9.4 text for .rs: the magnitude is truncated
// to the E2M1 grid and rounded away from zero when adding the 8 random bits
// to the top 8 discarded bits carries out. Operand a uses rbits[31:24], b
// rbits[15:8], e rbits[23:16] and f rbits[7:0]; the ISA describes this
// assignment only in a figure, so it is unverified until a run on sm_100a,
// sm_103a or sm_107a compares against the hardware. Results saturate to
// ±6. NaN inputs are not modeled and convert to 0.
func StochasticRoundE2M1x4(v [4]float32, rbits uint32) uint16 {
	r := [4]uint8{uint8(rbits >> 24), uint8(rbits >> 8), uint8(rbits >> 16), uint8(rbits)}
	var d uint16
	for i := range 4 {
		d = d<<4 | uint16(stochasticRoundE2M1(v[i], r[i]))
	}
	return d
}

func stochasticRoundE2M1(v float32, r uint8) uint8 {
	if v != v {
		return 0
	}
	var sign uint8
	if math.Signbit(float64(v)) {
		sign = 8
	}
	a := math.Abs(float64(v))
	if a >= 6 {
		return sign | 7
	}
	// ulp is the spacing of the E2M1 grid at a: 0.5 below 2, then 1, then 2.
	ulp := 0.5
	switch {
	case a >= 4:
		ulp = 2
	case a >= 2:
		ulp = 1
	}
	q := math.Floor(a / ulp)
	frac8 := uint(math.Floor((a/ulp - q) * 256))
	if frac8+uint(r) >= 256 {
		q++
	}
	m := q * ulp
	for i, g := range e2m1Grid {
		if g == m {
			if i == 0 {
				return 0
			}
			return sign | uint8(i)
		}
	}
	return sign | 7
}

// DecodeE2M1 returns the value of a 4-bit E2M1 code.
func DecodeE2M1(code uint8) float32 {
	v := float32(e2m1Grid[code&7])
	if code&8 != 0 {
		return -v
	}
	return v
}

// EmitDeterministicStochasticRoundPTX emits a PTX 8.7 kernel, for sm_100a,
// sm_103a or sm_107a, that converts a row-major f32 tensor [tokens][cols]
// to packed e2m1x4 with cvt.rs.satfinite.e2m1x4.f32, taking each 32-bit
// rbits word from an inline Philox4x32 keyed by (GlobalSeed, layer) with
// counter (token0 + token, col, 0, 0). Thread i of the grid converts the
// four columns 4*(i % (cols/4)) of token i / (cols/4), so the output does not
// depend on the launch configuration as long as it covers tokens*cols/4
// threads.
//
// Kernel parameters, in order: .u64 in, .u64 out (u16 per 4 columns),
// .u32 layer, .u32 token0, .u32 tokens, .u32 cols.
//
// The PTX is checked here only structurally (by [AuditPTXDeterminism] and
// tests); cvt.rs is not available on sm_121a, so assembling and running it
// needs an sm_100a-family device. [DeterministicCUDA] is the CUDA C++ twin
// that runs in cudahostemu and on sm_121a with a software rounding step.
func EmitDeterministicStochasticRoundPTX(cfg DeterministicConfig, funcName string) (string, error) {
	if funcName == "" {
		return "", fmt.Errorf("emit stochastic round ptx: empty function name")
	}
	rounds := cfg.rounds()
	if rounds < 1 || rounds > 16 {
		return "", fmt.Errorf("emit stochastic round ptx: PhiloxRounds %d out of range 1..16", rounds)
	}
	key := StochasticRoundKey(cfg.GlobalSeed, 0)
	var b strings.Builder
	fmt.Fprintf(&b, `.version 8.7
.target sm_100a
.address_size 64

// Philox4x32-%d stochastic rounding keyed by (seed=%#x, layer, token, col).
.visible .entry %s(
	.param .u64 p_in,
	.param .u64 p_out,
	.param .u32 p_layer,
	.param .u32 p_token0,
	.param .u32 p_tokens,
	.param .u32 p_cols
)
{
	.reg .pred %%p;
	.reg .b32 %%r<32>;
	.reg .b64 %%rd<8>;
	.reg .f32 %%f<4>;
	.reg .b16 %%h;

	ld.param.u64 %%rd0, [p_in];
	ld.param.u64 %%rd1, [p_out];
	ld.param.u32 %%r0, [p_layer];
	ld.param.u32 %%r1, [p_token0];
	ld.param.u32 %%r2, [p_tokens];
	ld.param.u32 %%r3, [p_cols];
	mov.u32 %%r4, %%ctaid.x;
	mov.u32 %%r5, %%ntid.x;
	mov.u32 %%r6, %%tid.x;
	mad.lo.u32 %%r7, %%r4, %%r5, %%r6;      // global thread index
	shr.u32 %%r8, %%r3, 2;                // groups per token
	mul.lo.u32 %%r9, %%r8, %%r2;
	setp.ge.u32 %%p, %%r7, %%r9;
	@%%p bra DONE;
	div.u32 %%r10, %%r7, %%r8;             // token
	rem.u32 %%r11, %%r7, %%r8;             // group
	shl.b32 %%r12, %%r11, 2;               // col
	// counter = (token0 + token, col, 0, 0), key = (seed.lo ^ layer, seed.hi)
	add.u32 %%r13, %%r1, %%r10;
	mov.u32 %%r14, %%r12;
	mov.u32 %%r15, 0;
	mov.u32 %%r16, 0;
	xor.b32 %%r17, %%r0, %d;
	mov.u32 %%r18, %d;
`, rounds, cfg.GlobalSeed, funcName, key[0], key[1])
	for r := range rounds {
		if r > 0 {
			fmt.Fprintf(&b, "\tadd.u32 %%r17, %%r17, %d;\n\tadd.u32 %%r18, %%r18, %d;\n", uint32(philoxW0), uint32(philoxW1))
		}
		fmt.Fprintf(&b, `	// round %d
	mul.hi.u32 %%r19, %%r13, %d;
	mul.lo.u32 %%r20, %%r13, %d;
	mul.hi.u32 %%r21, %%r15, %d;
	mul.lo.u32 %%r22, %%r15, %d;
	xor.b32 %%r23, %%r21, %%r14;
	xor.b32 %%r13, %%r23, %%r17;
	mov.u32 %%r14, %%r22;
	xor.b32 %%r23, %%r19, %%r16;
	xor.b32 %%r15, %%r23, %%r18;
	mov.u32 %%r16, %%r20;
`, r, uint32(philoxM0), uint32(philoxM0), uint32(philoxM1), uint32(philoxM1))
	}
	b.WriteString(`	// load four f32, round with rbits = counter word 0, store e2m1x4
	mul.lo.u32 %r24, %r10, %r3;
	add.u32 %r24, %r24, %r12;
	mul.wide.u32 %rd2, %r24, 4;
	add.u64 %rd3, %rd0, %rd2;
	ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd3];
	cvt.rs.satfinite.e2m1x4.f32 %h, {%f0, %f1, %f2, %f3}, %r13;
	mul.wide.u32 %rd4, %r7, 2;
	add.u64 %rd5, %rd1, %rd4;
	st.global.b16 [%rd5], %h;
DONE:
	ret;
}
`)
	return b.String(), nil
}

// A Combine says how partial sums of one output from different K ranges
// are added together.
type Combine uint8

const (
	// CombineFixedTree adds the per-chunk partial sums in a binary tree fixed
	// by the chunk count alone: the same tree whichever CTA or split
	// produced each partial.
	CombineFixedTree Combine = iota
	// CombineAtomic accumulates each split sequentially and then adds the
	// split totals in arrival order, as a Split-K or Stream-K kernel with a
	// floating-point atomic add does.
	CombineAtomic
)

// A ReductionPlan is how a tiled GEMM followed by a row softmax reduces one
// output row for a given batch.
type ReductionPlan struct {
	KChunk       int     // K elements per MMA chunk, summed in order
	SplitK       int     // CTAs sharing one output tile's K range; 0 means 1
	Combine      Combine // how chunk or split partials meet
	SoftmaxChunk int     // columns per online-softmax step; 0 means all
}

// A Schedule picks a reduction plan for a batch size.
type Schedule interface {
	Plan(batch int) ReductionPlan
}

// ScheduleFunc adapts a function to a [Schedule].
type ScheduleFunc func(batch int) ReductionPlan

// Plan returns f(batch).
func (f ScheduleFunc) Plan(batch int) ReductionPlan { return f(batch) }

// FixedSchedule returns a schedule that uses p for every batch size.
func FixedSchedule(p ReductionPlan) Schedule {
	return ScheduleFunc(func(int) ReductionPlan { return p })
}

// AuditSchedule reports schedule choices that make a row's result depend on
// batch size: a K chunk or softmax chunk that changes with the batch
// ([RuleBatchInvariantTile]), and split-K partials combined by atomic add
// ([RuleNoFPAtomics]).
func AuditSchedule(s Schedule, batchSizes []int) []DeterminismViolation {
	var out []DeterminismViolation
	if len(batchSizes) == 0 {
		return nil
	}
	base := s.Plan(batchSizes[0])
	for _, bs := range batchSizes {
		p := s.Plan(bs)
		loc := fmt.Sprintf("batch %d", bs)
		if p.KChunk != base.KChunk || p.SoftmaxChunk != base.SoftmaxChunk {
			out = append(out, DeterminismViolation{Rule: RuleBatchInvariantTile, Location: loc,
				Detail: fmt.Sprintf("KChunk=%d SoftmaxChunk=%d, batch %d uses %d and %d", p.KChunk, p.SoftmaxChunk, batchSizes[0], base.KChunk, base.SoftmaxChunk)})
		}
		if p.Combine == CombineAtomic && p.SplitK > 1 {
			out = append(out, DeterminismViolation{Rule: RuleNoFPAtomics, Location: loc,
				Detail: fmt.Sprintf("SplitK=%d partials combined by atomic add", p.SplitK)})
		}
	}
	return out
}

// FixedTreeSum adds vals in the binary tree used by [CombineFixedTree] and by
// the code from [DeterministicCUDA]: at stride s = 1, 2, 4, ..., element i
// (a multiple of 2s) absorbs element i+s. Each add rounds to float32.
func FixedTreeSum(vals []float32) float32 {
	if len(vals) == 0 {
		return 0
	}
	p := slices.Clone(vals)
	for s := 1; s < len(p); s *= 2 {
		for i := 0; i+s < len(p); i += 2 * s {
			p[i] = p[i] + p[i+s]
		}
	}
	return p[0]
}

// chunkSums returns the float32 partial dot product of x and w over each
// chunk of k elements, accumulated in order with a rounded multiply then a
// rounded add (no fused multiply-add), matching __fmul_rn and __fadd_rn.
func chunkSums(x, w []float32, k int) []float32 {
	n := (len(x) + k - 1) / k
	out := make([]float32, n)
	for c := range n {
		var acc float32
		for i := c * k; i < min(len(x), (c+1)*k); i++ {
			p := float32(x[i] * w[i])
			acc = float32(acc + p)
		}
		out[c] = acc
	}
	return out
}

// planDot returns x·w reduced by plan p. jitter seeds the arrival order of
// atomic split partials.
func planDot(x, w []float32, p ReductionPlan, jitter uint64) float32 {
	parts := chunkSums(x, w, p.KChunk)
	if p.Combine == CombineFixedTree {
		return FixedTreeSum(parts)
	}
	split := max(p.SplitK, 1)
	per := (len(parts) + split - 1) / split
	var totals []float32
	for s := 0; s*per < len(parts); s++ {
		var acc float32
		for _, v := range parts[s*per : min(len(parts), (s+1)*per)] {
			acc = float32(acc + v)
		}
		totals = append(totals, acc)
	}
	// Arrival order: a permutation drawn from a splitmix64 stream.
	for i := len(totals) - 1; i > 0; i-- {
		jitter += 0x9E3779B97F4A7C15
		z := jitter
		z = (z ^ z>>30) * 0xBF58476D1CE4E5B9
		z = (z ^ z>>27) * 0x94D049BB133111EB
		z ^= z >> 31
		j := int(z % uint64(i+1))
		totals[i], totals[j] = totals[j], totals[i]
	}
	var acc float32
	for _, v := range totals {
		acc = float32(acc + v)
	}
	return acc
}

// onlineSoftmax returns softmax(z) computed with running (max, sum) state
// over chunks of n columns.
func onlineSoftmax(z []float32, n int) []float32 {
	if n <= 0 {
		n = len(z)
	}
	m := float32(math.Inf(-1))
	var l float32
	for c := 0; c < len(z); c += n {
		cm := m
		for _, v := range z[c:min(len(z), c+n)] {
			cm = max(cm, v)
		}
		l = float32(l * float32(math.Exp(float64(m-cm))))
		for _, v := range z[c:min(len(z), c+n)] {
			l = float32(l + float32(math.Exp(float64(v-cm))))
		}
		m = cm
	}
	out := make([]float32, len(z))
	for i, v := range z {
		out[i] = float32(float32(math.Exp(float64(v-m))) / l)
	}
	return out
}

// SimulateRow returns the logits and softmax probabilities of row r of a
// batch under schedule s: logits[c] = batch[r]·weights[c], reduced as
// s.Plan(len(batch)) says, then an online softmax over the logits.
func SimulateRow(s Schedule, batch [][]float32, r int, weights [][]float32) (logits, probs []float32, err error) {
	p := s.Plan(len(batch))
	if p.KChunk <= 0 {
		return nil, nil, fmt.Errorf("simulate row: batch %d: KChunk %d", len(batch), p.KChunk)
	}
	logits = make([]float32, len(weights))
	for c, w := range weights {
		if len(w) != len(batch[r]) {
			return nil, nil, fmt.Errorf("simulate row: weight row %d has %d elements, want %d", c, len(w), len(batch[r]))
		}
		logits[c] = planDot(batch[r], w, p, uint64(len(batch))<<32|uint64(r)<<16|uint64(c))
	}
	return logits, onlineSoftmax(logits, p.SoftmaxChunk), nil
}

// BatchInvarianceCheck runs targetRow alone (batch 1) and embedded at a
// varying offset in each batch of batchSizes, the other rows filled with
// fixed pseudo-random data, through the GEMM and softmax of [SimulateRow]
// under s. It returns an error naming the first batch size and column
// whose logit or probability bits differ from batch 1.
func BatchInvarianceCheck(s Schedule, targetRow []float32, weights [][]float32, batchSizes []int) error {
	ref, refP, err := SimulateRow(s, [][]float32{targetRow}, 0, weights)
	if err != nil {
		return fmt.Errorf("batch invariance: %w", err)
	}
	for _, bs := range batchSizes {
		if bs < 1 {
			return fmt.Errorf("batch invariance: batch size %d", bs)
		}
		at := (bs * 19 / 37) % bs
		batch := make([][]float32, bs)
		for i := range batch {
			if i == at {
				batch[i] = targetRow
				continue
			}
			row := make([]float32, len(targetRow))
			for j := range row {
				row[j] = float32(int32(hash32(uint32(bs), uint32(i), uint32(j)))) / (1 << 31)
			}
			batch[i] = row
		}
		got, gotP, err := SimulateRow(s, batch, at, weights)
		if err != nil {
			return fmt.Errorf("batch invariance: %w", err)
		}
		for c := range ref {
			if math.Float32bits(got[c]) != math.Float32bits(ref[c]) {
				return fmt.Errorf("batch invariance: batch %d row %d column %d: logit %#08x, batch 1 gives %#08x",
					bs, at, c, math.Float32bits(got[c]), math.Float32bits(ref[c]))
			}
			if math.Float32bits(gotP[c]) != math.Float32bits(refP[c]) {
				return fmt.Errorf("batch invariance: batch %d row %d column %d: probability %#08x, batch 1 gives %#08x",
					bs, at, c, math.Float32bits(gotP[c]), math.Float32bits(refP[c]))
			}
		}
	}
	return nil
}

func hash32(a, b, c uint32) uint32 {
	h := a*0x9E3779B1 ^ b*0x85EBCA77 ^ c*0xC2B2AE3D
	h ^= h >> 15
	h *= 0x2C1B3C6D
	h ^= h >> 12
	h *= 0x297A2D39
	h ^= h >> 15
	return h
}

// NondeterministicOps lists e-graph operator names whose lowering reduces
// floating-point values in an order fixed only at run time.
var NondeterministicOps = []string{"splitk_atomic", "streamk", "atomic_add", "multimem_ld_reduce", "multimem_red_add"}

// FilterDeterministicCandidates returns the candidates whose expression
// trees use none of [NondeterministicOps]. The e-graph lowerings in this
// module do not currently produce these operators; the filter is the hook
// a lowering that adds them must pass through in deterministic mode.
func FilterDeterministicCandidates(cands []egraph.CandidateAST) []egraph.CandidateAST {
	var out []egraph.CandidateAST
	for _, c := range cands {
		if !usesOp(c.Expr, NondeterministicOps) {
			out = append(out, c)
		}
	}
	return out
}

func usesOp(e *egraph.Expr, ops []string) bool {
	if e == nil {
		return false
	}
	if slices.Contains(ops, e.Op) {
		return true
	}
	for _, c := range e.Children {
		if usesOp(c, ops) {
			return true
		}
	}
	return false
}
