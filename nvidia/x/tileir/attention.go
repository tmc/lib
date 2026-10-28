package tileir

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// AttentionCandidate is the input-bound portion of egraph.OnlineAttentionCandidate.
// Its methods must not be called concurrently with mutation. A candidate identity
// is a caller assertion, not qualification: LowerOnlineAttention independently
// checks the graph, inputs and both outputs. This interface avoids an import cycle
// with egraph's existing Tile IR consumer.
type AttentionCandidate interface {
	Snapshot() *cudair.Snapshot
	Inputs() map[string][]float32
	Identity() string
	Output() []float32
	Reference() []float32
}

// AttentionOptions declares the finite online-row recipe and its input-bound
// comparison policy. Excluded is nil for dense attention, otherwise true marks
// a score which must already be -Inf. It does not introduce a mask operator.
// Absolute and Relative permit abs(got-want) <= Absolute+Relative*abs(want).
// Their zero values require identical FP32 bits, including signed zero.
type AttentionOptions struct {
	KernelName         string
	TileSize           int
	Excluded           []bool
	Absolute, Relative float64
}

// AttentionLowering owns an input-bound typed online-attention kernel. Its zero
// value is invalid. The three pointer arguments are scores[N], values[N,D] and
// output[D], in that order. Launch exactly Grid(1,1,1). Callers must upload the
// exact Inputs bytes, use disjoint buffers and retain them through completion.
// Rebinding other values requires a new lowering and numerical check.
//
// The typed module uses finite unrolling, token-ordered masked pointer accesses,
// FP64 online state and full exp, followed by FP32 output conversion. It covers
// the existing lane-07 bounds N,D in [1,64]. There are no nested regions and no
// C++ source fallback. Compiler/device math need independent qualification;
// host admission cannot establish CUDA exp bits, resources or performance.
type AttentionLowering struct {
	result   *LoweringResult
	inputs   map[string][]float32
	options  AttentionOptions
	identity string
}

// Result returns the existing backend lowering, usable by AssembleSemanticMemory.
// The returned structure is borrowed. Verify detects edits to its module, ABI,
// grid or semantic source map. Source is empty: this route emits typed bytecode.
func (a *AttentionLowering) Result() *LoweringResult {
	if a == nil {
		return nil
	}
	return a.result
}

// Identity returns the bound candidate, input, mask, policy and bytecode digest.
func (a *AttentionLowering) Identity() string {
	if a == nil {
		return ""
	}
	return a.identity
}

// Inputs returns detached input values under their semantic names.
func (a *AttentionLowering) Inputs() map[string][]float32 {
	if a == nil {
		return nil
	}
	out := make(map[string][]float32, len(a.inputs))
	for k, v := range a.inputs {
		out[k] = slices.Clone(v)
	}
	return out
}

// Verify checks the retained module/ABI/source-map binding. It does not inspect
// device buffers, authenticate a candidate or establish compiler acceptance.
func (a *AttentionLowering) Verify() error {
	if a == nil || a.result == nil || a.result.Memory == nil || a.identity == "" {
		return fmt.Errorf("tileir: missing online attention lowering")
	}
	return a.result.Memory.Verify(a.result)
}

// LowerOnlineAttention lowers the exact readonly nine-node attention graph
// admitted by lane 07. It consumes a checked candidate through the narrow
// AttentionCandidate contract, validates that contract again, and emits actual
// pointer-memory Tile IR. Unrelated exp/reduction graphs remain unsupported by
// LowerSemanticRegion. Options must explicitly repeat the mask and numerical
// policy; they cannot silently turn a dense candidate into masked attention.
//
// Excluded scores must already be -Inf, included scores and all values finite.
// A fully excluded row emits canonical FP32 NaNs and must match the unchanged
// graph's NaN bits. Nonempty rows are checked against cudaref and the candidate.
// Snapshot is immutable; all input/output/options slices are copied or consumed
// before return. No compiler, device allocation or launch occurs here.
func LowerOnlineAttention(candidate AttentionCandidate, opts AttentionOptions) (*AttentionLowering, error) {
	if candidate == nil || reflect.ValueOf(candidate).Kind() == reflect.Pointer && reflect.ValueOf(candidate).IsNil() || opts.TileSize < 1 || opts.TileSize > 64 ||
		opts.Absolute < 0 || opts.Relative < 0 || !attentionFinite(opts.Absolute) || !attentionFinite(opts.Relative) {
		return nil, fmt.Errorf("%w: invalid attention options or candidate", ErrUnsupportedTileOp)
	}
	identity := candidate.Identity()
	if raw, err := hex.DecodeString(identity); err != nil || len(raw) != 32 {
		return nil, fmt.Errorf("%w: invalid attention candidate identity", ErrUnsupportedTileOp)
	}
	snap := candidate.Snapshot()
	roles, err := attentionRegion(snap)
	if err != nil {
		return nil, err
	}
	n, d := roles[0].Shape[0], roles[1].Shape[1]
	if opts.TileSize > n || opts.Excluded != nil && len(opts.Excluded) != n {
		return nil, fmt.Errorf("%w: attention tile or mask extent", ErrUnsupportedTileOp)
	}
	opts.Excluded = slices.Clone(opts.Excluded)
	provided := candidate.Inputs()
	if len(provided) != 2 {
		return nil, fmt.Errorf("%w: attention requires two exact inputs", ErrUnsupportedTileOp)
	}
	inputs := make(map[string][]float32, 2)
	for _, node := range roles[:2] {
		data, ok := provided[node.Name]
		if !ok || len(data) != node.NumElements() {
			return nil, fmt.Errorf("%w: attention input extent", ErrUnsupportedTileOp)
		}
		for i, v := range data {
			if node.ID == roles[0].ID && opts.Excluded != nil && opts.Excluded[i] {
				if !math.IsInf(float64(v), -1) {
					return nil, fmt.Errorf("%w: excluded score must already be -Inf", ErrUnsupportedTileOp)
				}
			} else if !attentionFinite(float64(v)) {
				return nil, fmt.Errorf("%w: nonfinite included attention input", ErrUnsupportedTileOp)
			}
		}
		inputs[node.Name] = slices.Clone(data)
	}
	reference, err := cudaref.EvaluateSnapshot(snap, inputs)
	if err != nil {
		return nil, fmt.Errorf("tileir: attention reference: %w", err)
	}
	want := reference[roles[8].Name]
	got := attentionRow(inputs[roles[0].Name], inputs[roles[1].Name], d, opts)
	for _, output := range [][]float32{candidate.Reference(), candidate.Output(), got} {
		if !attentionMatches(output, want, opts) {
			return nil, fmt.Errorf("%w: attention numerical policy", ErrUnsupportedTileOp)
		}
	}
	r, err := attentionModule(roles, opts)
	if err != nil {
		return nil, err
	}
	// JSON cannot represent -Inf, so bind the exact input bit patterns.
	bits := make(map[string][]uint32, len(inputs))
	for key, values := range inputs {
		for _, value := range values {
			bits[key] = append(bits[key], math.Float32bits(value))
		}
	}
	data, err := json.Marshal(struct {
		Candidate string
		Options   AttentionOptions
		Inputs    map[string][]uint32
		Bytecode  string
	}{identity, opts, bits, r.Memory.BytecodeDigest()})
	if err != nil {
		return nil, fmt.Errorf("tileir: bind attention identity: %w", err)
	}
	return &AttentionLowering{result: r, inputs: inputs, options: opts, identity: memoryDigest(data)}, nil
}

func attentionFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func attentionMatches(got, want []float32, opts AttentionOptions) bool {
	if len(got) != len(want) {
		return false
	}
	for i, v := range got {
		if opts.Excluded != nil && !slices.Contains(opts.Excluded, false) {
			if math.Float32bits(v) != 0x7fc00000 || math.Float32bits(want[i]) != 0x7fc00000 {
				return false
			}
			continue
		}
		if !attentionFinite(float64(v)) || !attentionFinite(float64(want[i])) {
			return false
		}
		if opts.Absolute == 0 && opts.Relative == 0 {
			if math.Float32bits(v) != math.Float32bits(want[i]) {
				return false
			}
			continue
		}
		tolerance := opts.Absolute + opts.Relative*math.Abs(float64(want[i]))
		if !attentionFinite(tolerance) || math.Abs(float64(v)-float64(want[i])) > tolerance {
			return false
		}
	}
	return true
}

// attentionRow is an input-bound host check, not the backend implementation.
// The tests execute the emitted operation list independently of this helper.
func attentionRow(scores, values []float32, d int, opts AttentionOptions) []float32 {
	out := make([]float32, d)
	u := make([]float64, d)
	m, l := math.Inf(-1), 0.0
	for start := 0; start < len(scores); start += opts.TileSize {
		end := min(start+opts.TileSize, len(scores))
		tileMax := math.Inf(-1)
		for k := start; k < end; k++ {
			if opts.Excluded == nil || !opts.Excluded[k] {
				tileMax = math.Max(tileMax, float64(scores[k]))
			}
		}
		if math.IsInf(tileMax, -1) {
			continue
		}
		next := math.Max(m, tileMax)
		alpha := 1.0
		if l != 0 {
			alpha = math.Exp(m - next)
		}
		pSum := 0.0
		pv := make([]float64, d)
		for k := start; k < end; k++ {
			if opts.Excluded != nil && opts.Excluded[k] {
				continue
			}
			p := math.Exp(float64(scores[k]) - next)
			pSum += p
			for j := range u {
				pv[j] += p * float64(values[k*d+j])
			}
		}
		l = l*alpha + pSum
		for j := range u {
			u[j] = u[j]*alpha + pv[j]
		}
		m = next
	}
	for j := range out {
		if l == 0 {
			out[j] = math.Float32frombits(0x7fc00000)
		} else {
			out[j] = float32(u[j] / l)
		}
	}
	return out
}

func attentionRegion(s *cudair.Snapshot) ([]*cudair.Node, error) {
	bad := fmt.Errorf("%w: exact readonly FP32 online-attention graph required", ErrUnsupportedTileOp)
	if s == nil || len(s.Nodes()) != 9 || len(s.Inputs()) != 2 || len(s.Outputs()) != 1 {
		return nil, bad
	}
	match := func(n *cudair.Node, kind cudair.NodeKind, op string, arity int) bool {
		return n != nil && n.Kind == kind && n.Op == op && len(n.Inputs) == arity
	}
	out := s.Outputs()[0]
	if !match(out, cudair.KindMatMul, "matmul", 2) {
		return nil, bad
	}
	reshape, values := out.Inputs[0], out.Inputs[1]
	if !match(reshape, cudair.KindReshape, "reshape", 1) || !match(values, cudair.KindInput, "input", 0) || len(values.Shape) != 2 {
		return nil, bad
	}
	p := reshape.Inputs[0]
	if !match(p, cudair.KindBinary, "div", 2) {
		return nil, bad
	}
	e, sum := p.Inputs[0], p.Inputs[1]
	if !match(e, cudair.KindUnary, "exp", 1) || !match(sum, cudair.KindReduce, "sum", 1) ||
		sum.Inputs[0].ID != e.ID || !slices.Equal(sum.Axes, []int{0}) {
		return nil, bad
	}
	shift := e.Inputs[0]
	if !match(shift, cudair.KindBinary, "sub", 2) {
		return nil, bad
	}
	scores, max := shift.Inputs[0], shift.Inputs[1]
	if !match(scores, cudair.KindInput, "input", 0) || len(scores.Shape) != 1 || !match(max, cudair.KindReduce, "max", 1) ||
		max.Inputs[0].ID != scores.ID || !slices.Equal(max.Axes, []int{0}) {
		return nil, bad
	}
	n, d := scores.Shape[0], values.Shape[1]
	if n < 1 || n > 64 || d < 1 || d > 64 || values.Shape[0] != n || scores.Name == values.Name {
		return nil, bad
	}
	roles := []*cudair.Node{scores, values, max, shift, e, sum, p, reshape, out}
	shapes := [][]int{{n}, {n, d}, {1}, {n}, {n}, {1}, {n}, {1, n}, {1, d}}
	seen := make(map[int]bool, 9)
	for i, node := range roles {
		strides, err := cudashape.RowMajorStrides(node.Shape)
		if seen[node.ID] || !slices.Equal(node.Shape, shapes[i]) || node.Type != cudair.Float32 ||
			node.Layout != cudashape.LayoutRowMajor || err != nil || !slices.Equal(node.Strides, strides) {
			return nil, bad
		}
		seen[node.ID] = true
		if i == 7 {
			if node.Effect != cudair.EffectInPlaceAlias || node.StorageAlias == nil || node.StorageAlias.ID != p.ID {
				return nil, bad
			}
		} else if node.Effect != cudair.EffectReadOnly || node.StorageAlias != nil {
			return nil, bad
		}
	}
	for _, node := range s.Nodes() {
		if !seen[node.ID] {
			return nil, bad
		}
	}
	return roles, nil
}

type attentionBuilder struct {
	fn     *Function
	serial int
	token  string
	ref    *MemorySource
}

func (b *attentionBuilder) emit(code string, typ Type, operands []string, attrs string) string {
	b.serial++
	v := fmt.Sprintf("%%a%d", b.serial)
	op := &Operation{OpCode: "cuda_tile." + code, Results: []string{v}, TypeVal: typ, Operands: operands, Attributes: attrs}
	if code == "return" {
		op.Results = nil
		v = ""
	}
	if b.ref != nil {
		b.ref.Operations = append(b.ref.Operations, len(b.fn.Operations))
		b.ref.Values = append(b.ref.Values, op.Results...)
	}
	b.fn.Operations = append(b.fn.Operations, op)
	return v
}

func (b *attentionBuilder) load(ptr, mask, padding string, typ Type) string {
	v := b.emit("load_ptr_tko", NewFunction(nil, []Type{typ, Token}), []string{ptr, mask, padding, b.token}, reductionWeak)
	b.serial++
	b.token = fmt.Sprintf("%%a%d", b.serial)
	b.fn.Operations[len(b.fn.Operations)-1].Results = []string{v, b.token}
	return v
}

func attentionModule(roles []*cudair.Node, opts AttentionOptions) (*LoweringResult, error) {
	name := opts.KernelName
	if name == "" {
		name = "tile_online_attention"
	}
	if len(name) > 128 {
		return nil, fmt.Errorf("tileir: attention identifier too long")
	}
	for i, c := range name {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || i > 0 && c >= '0' && c <= '9') {
			return nil, fmt.Errorf("tileir: invalid attention identifier")
		}
	}
	n, d := roles[0].Shape[0], roles[1].Shape[1]
	w := int64(1)
	for w < int64(d) {
		w *= 2
	}
	v64, v32 := NewTile([]int64{w}, Float64), NewTile([]int64{w}, Float32)
	vi, maskType := NewTile([]int64{w}, Int64), NewTile([]int64{w}, Int1)
	ptr, vptr := NewTile(nil, NewPointer(Float32)), NewTile([]int64{w}, NewPointer(Float32))
	b := &attentionBuilder{fn: &Function{Name: name, Entry: true}}
	refs := make([]MemorySource, 0, len(roles)+3)
	for _, node := range roles {
		refs = append(refs, MemorySource{NodeID: node.ID, Role: node.Op, ABIIndex: -1, Location: node.SourceLoc})
	}
	var params []cudakernel.Param
	names := make(map[string]bool)
	bases := make([]string, 3)
	for i, node := range []*cudair.Node{roles[0], roles[1], roles[8]} {
		if node.Name == "" || names[node.Name] {
			return nil, fmt.Errorf("tileir: attention ABI names must be distinct")
		}
		names[node.Name] = true
		params = append(params, cudakernel.Param{Name: node.Name, Type: "f32", ByteWidth: 8, Alignment: 8, IsPointer: true})
		arg := fmt.Sprintf("%%arg%d", i)
		b.fn.Params = append(b.fn.Params, &Param{Name: arg, TypeVal: ptr})
		refs = append(refs, MemorySource{NodeID: node.ID, Role: []string{"scores", "values", "output"}[i], ABIIndex: i, Location: node.SourceLoc})
		b.ref = &refs[len(refs)-1]
		one := b.emit("reshape", NewTile([]int64{1}, NewPointer(Float32)), []string{arg}, "")
		bases[i] = b.emit("broadcast", vptr, []string{one}, "")
	}
	b.ref = &refs[8]
	b.token = b.emit("make_token", Token, nil, "")
	iconst := func(i int) string { return b.emit("constant", vi, nil, fmt.Sprintf("{value = %d}", i)) }
	zeroIndex := iconst(0)
	lanes := b.emit("iota", vi, nil, "")
	valid := b.emit("cmpi", maskType, []string{lanes, iconst(d)}, reductionBound)
	safe := b.emit("select", vi, []string{valid, lanes, zeroIndex}, "")
	zero32 := b.emit("constant", v32, nil, "{value = 0.0}")
	zero64 := b.emit("constant", v64, nil, "{value = 0.0}")
	one64 := b.emit("constant", v64, nil, "{value = 1.0}")
	l, u, m := zero64, zero64, ""
	floatOp := func(code string, operands ...string) string { return b.emit(code, v64, operands, reductionNearest) }
	exp := func(x string) string { return b.emit("exp", v64, []string{x}, "{rounding_mode = full}") }
	maximum := func(a, c string) string {
		gt := b.emit("cmpf", maskType, []string{a, c}, reductionGreater)
		return b.emit("select", v64, []string{gt, a, c}, "")
	}
	for start := 0; start < n; start += opts.TileSize {
		var scores []string
		var keys []int
		for k := start; k < min(start+opts.TileSize, n); k++ {
			if opts.Excluded != nil && opts.Excluded[k] {
				continue
			}
			b.ref = &refs[9]
			p := b.emit("offset", vptr, []string{bases[0], iconst(k)}, "")
			s := b.load(p, valid, zero32, v32)
			b.ref = &refs[2]
			scores = append(scores, b.emit("ftof", v64, []string{s}, reductionNearest))
			keys = append(keys, k)
		}
		if len(scores) == 0 {
			continue
		}
		next := scores[0]
		for _, s := range scores[1:] {
			next = maximum(next, s)
		}
		alpha := one64
		if m != "" {
			next = maximum(m, next)
			alpha = exp(floatOp("subf", m, next))
		}
		pSum, pv := zero64, zero64
		for i, s := range scores {
			b.ref = &refs[4]
			p := exp(floatOp("subf", s, next))
			b.ref = &refs[5]
			pSum = floatOp("addf", pSum, p)
			b.ref = &refs[10]
			offset := b.emit("addi", vi, []string{iconst(keys[i] * d), lanes}, reductionNoWrap)
			offset = b.emit("select", vi, []string{valid, offset, zeroIndex}, "")
			address := b.emit("offset", vptr, []string{bases[1], offset}, "")
			v := b.load(address, valid, zero32, v32)
			b.ref = &refs[8]
			wide := b.emit("ftof", v64, []string{v}, reductionNearest)
			pv = floatOp("addf", pv, floatOp("mulf", p, wide))
		}
		l = floatOp("addf", floatOp("mulf", l, alpha), pSum)
		u = floatOp("addf", floatOp("mulf", u, alpha), pv)
		m = next
	}
	b.ref = &refs[8]
	var out string
	if m == "" {
		out = b.emit("constant", v32, nil, reductionQuietNaN32)
	} else {
		out = b.emit("ftof", v32, []string{floatOp("divf", u, l)}, reductionNearest)
	}
	b.ref = &refs[11]
	dst := b.emit("offset", vptr, []string{bases[2], safe}, "")
	b.token = b.emit("store_ptr_tko", Token, []string{dst, out, valid, b.token}, reductionWeak)
	b.ref = nil
	b.emit("return", nil, nil, "")
	schema, err := cudakernel.NewSchema(params...)
	if err != nil {
		return nil, err
	}
	r := &LoweringResult{Module: &Module{Name: name + "_mod", Producer: "lib/nvidia/x/tileir", Functions: []*Function{b.fn}}, Schema: schema, Grid: [3]uint32{1, 1, 1}}
	data, err := r.Module.EncodeBytecode()
	if err != nil {
		return nil, fmt.Errorf("tileir: online attention bytecode: %w", err)
	}
	r.Memory = &SemanticMemoryBinding{byteDigest: memoryDigest(data), references: refs}
	r.Memory.contractDigest = memoryContract(r, refs)
	return r, nil
}
