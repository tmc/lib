package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func TestCheckEgraphTarget(t *testing.T) {
	tests := []struct {
		dev, target string
		want        string // error substring; "" means no error
	}{
		{"sm_121a", "sm_121a", ""},
		{"sm_121a", "GB10", ""},
		{"sm_103a", "GB300", ""},
		{"sm_100a", "sm_100a", ""},
		{"sm_121a", "sm_103a", "lowers its gemms to tcgen05"},
		{"sm_121a", "GB300", "lowers its gemms to tcgen05"},
		{"sm_121a", "sm_100a", "lowers its gemms to tcgen05"},
		{"sm_100a", "sm_103a", "is not this device's sm_100a"},
		{"sm_121a", "sm_90a", "is not this device's sm_121a"},
	}
	for _, tt := range tests {
		t.Run(tt.dev+"/"+tt.target, func(t *testing.T) {
			p, err := cudadevicetable.LookupProfile(tt.target)
			if err != nil {
				t.Fatal(err)
			}
			err = checkEgraphTarget(tt.dev, p)
			switch {
			case tt.want == "" && err != nil:
				t.Fatalf("checkEgraphTarget: %v", err)
			case tt.want != "" && (err == nil || !strings.Contains(err.Error(), tt.want)):
				t.Fatalf("checkEgraphTarget = %v, want an error containing %q", err, tt.want)
			}
		})
	}
}

// tcgen05Config is a small Llama whose projections all have the shapes
// UMMAGEMM's decode form takes: N a multiple of 128 and K of 64.
var tcgen05Config = ModelConfig{
	ModelType:         "llama",
	HiddenSize:        128,
	IntermediateSize:  256,
	NumHiddenLayers:   2,
	NumAttentionHeads: 2,
	NumKeyValueHeads:  1,
	VocabSize:         256,
	RmsNormEps:        1e-5,
	RopeTheta:         10000,
	MaxSeqLen:         16,
}

// TestEgraphDecoderTCGen05Emulated binds the e-graph decoder's launches for
// sm_100a, whose projections are tcgen05 UMMAGEMMs, with no GPU: the
// buffers are synthetic device addresses and the tensor maps are recorded
// instead of encoded. It then runs a few greedy decode steps of exactly
// those launches (kernel, grid, block, shared memory and arguments) through
// cudahostemu's descriptor-level tcgen05 emulation and checks each step's
// logits and token against the CPU reference [Forward] on the same BF16
// weights. Embedding and argmax, which llama-infer's own kernels run, are
// host loops here.
//
// A control run binds every tensor map over a weight to the first buffer
// of that shape, as a cache keyed without the address would: it must
// disagree with the reference.
func TestEgraphDecoderTCGen05Emulated(t *testing.T) {
	if testing.Short() {
		t.Skip("emulates a whole decode step on the host")
	}
	if _, err := cudahostemu.Compiler(); err != nil {
		t.Skip("no host C++ compiler; emulated decode not run")
	}
	cfg := tcgen05Config
	w := randomBF16Weights(&cfg, rand.New(rand.NewPCG(5, 8)))
	const steps = 4
	const first = 7

	run := func(t *testing.T, stale bool) (logits [][]float32, tokens []int32, encodes int) {
		e := newEmuDecoder(t, &cfg, w, "sm_100a", stale)
		logits, tokens = e.run(t, first, steps)
		return logits, tokens, e.encodes
	}

	got, toks, encodes := run(t, false)
	kv := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, cfg.NumKeyValueHeads*cfg.HeadDim())
	tok := int32(first)
	var worst float64
	var refs []int32
	for pos := range steps {
		want := Forward(w, &cfg, kv, tok, pos)
		d, scale := maxAbsDiff(got[pos], want)
		worst = max(worst, d/scale)
		if d > 0.02*scale {
			t.Fatalf("step %d: logits differ by %g (max |logit| %g)", pos, d, scale)
		}
		ref := argmax(want)
		refs = append(refs, ref)
		if toks[pos] != ref {
			// Accept a different token only where the reference's top two
			// are closer than the BF16 activations can resolve.
			if margin := want[ref] - want[toks[pos]]; float64(margin) > 0.02*scale {
				t.Fatalf("step %d: token %d, reference %d (margin %g)", pos, toks[pos], ref, margin)
			}
		}
		// Teacher-force the emulated token so both stay on one sequence.
		tok = toks[pos]
	}
	t.Logf("EMULATED sm_100a decode, %d steps: tokens %v, reference %v; worst logit error %.3g of max |logit|; %d tensor maps encoded once at bind", steps, toks, refs, worst, encodes)

	bad, _, _ := run(t, true)
	kv = NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, cfg.NumKeyValueHeads*cfg.HeadDim())
	tok = first
	for pos := range steps {
		want := Forward(w, &cfg, kv, tok, pos)
		if d, scale := maxAbsDiff(bad[pos], want); d > 0.02*scale {
			t.Logf("control: stale tensor maps detected at step %d (logits differ by %.3g of max |logit|)", pos, d/scale)
			return
		}
		tok = toks[pos]
	}
	t.Fatal("control: tensor maps bound to the wrong weights went undetected")
}

// An emuDecoder is an egraphDecoder bound to synthetic device addresses,
// one 4 GiB window per named buffer, rendered as a cudahostemu program.
type emuDecoder struct {
	eg       *egraphDecoder
	cfg      *ModelConfig
	bufs     []cudahostemu.Buffer
	names    map[driver.Function]string
	maps     map[*driver.TensorMap]string // map → C++ declaration
	mapOrder []*driver.TensorMap
	encodes  int
	b        egraphBuffers
}

func newEmuDecoder(t *testing.T, cfg *ModelConfig, w *ModelWeights, target string, stale bool) *emuDecoder {
	t.Helper()
	plans, err := buildEgraphPlans(cfg, false, egraphOptions{Profile: targetProfile(t, target), Attn: true, AttnSplits: 2})
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []*hopper.Plan{plans.attnIn, plans.post, plans.head} {
		for _, k := range p.Kernels {
			if k.GEMM != nil || k.Tile != nil {
				t.Fatalf("%s: gemm not lowered to tcgen05 for %s", p.Name, target)
			}
		}
	}
	e := &emuDecoder{
		cfg:   cfg,
		names: make(map[driver.Function]string),
		maps:  make(map[*driver.TensorMap]string),
	}
	e.eg = &egraphDecoder{
		plans:   plans,
		megas:   make(map[*hopper.Plan]hopper.Megakernel),
		fns:     make(map[string]driver.Function),
		tempPtr: make(map[string]driver.DevicePtr),
		posKind: make(map[driver.Function]string),
		posArgs: make(map[driver.Function][]egraphPosArg),
	}
	_, names, err := e.eg.source()
	if err != nil {
		t.Fatal(err)
	}
	for i, n := range names {
		fn := driver.Function(i + 1)
		e.eg.fns[n] = fn
		e.names[fn] = n
	}
	// The first buffer of each map shape, for the stale control.
	firstOf := make(map[hopper.TensorMap2D]driver.DevicePtr)
	weights := make(map[driver.DevicePtr]bool)
	e.eg.encode = func(tm hopper.TensorMap2D, base driver.DevicePtr) (*driver.TensorMap, error) {
		e.encodes++
		if b, ok := firstOf[tm]; !ok {
			firstOf[tm] = base
		} else if stale && weights[base] {
			base = b
		}
		m := new(driver.TensorMap)
		e.maps[m] = fmt.Sprintf("eg_umma_tmap_2d(%s, %d, %d, %d, %d, %d, %d, %v)",
			e.ptr(base), tm.ElemBytes, tm.Inner, tm.Outer, tm.Stride, tm.BoxInner, tm.BoxOuter, tm.Swizzle128)
		e.mapOrder = append(e.mapOrder, m)
		return m, nil
	}

	h, hd := cfg.HiddenSize, cfg.HeadDim()
	qDim, kvDim := cfg.NumAttentionHeads*hd, cfg.NumKeyValueHeads*hd
	for _, p := range e.eg.planList() {
		for _, tmp := range p.Temps {
			n := tmp.DType.Bits() / 8
			for _, d := range tmp.Shape {
				n *= d
			}
			e.eg.tempPtr[tmp.Name] = e.alloc(tmp.Name, cudahostemu.Poison(n))
		}
	}
	pages := make([]int32, plans.sdpa.MaxPages)
	for i := range pages {
		pages[i] = int32(i)
	}
	cs := make([]float32, cfg.MaxSeqLen*hd)
	for pos := range cfg.MaxSeqLen {
		for i := range hd / 2 {
			a := float64(pos) * float64(ropeInvFreq(i, hd, cfg.RopeTheta, cfg.RoPEScaling))
			cs[pos*hd+i] = float32(math.Cos(a))
			cs[pos*hd+hd/2+i] = float32(math.Sin(a))
		}
	}
	e.alloc("embed", bf16le(w.EmbedTokens))
	e.b = egraphBuffers{
		x:        e.alloc("x", cudahostemu.Poison(4*h)),
		x2:       e.alloc("x2", cudahostemu.Poison(4*h)),
		qkv:      e.alloc("qkv", cudahostemu.Poison(2*(qDim+2*kvDim))),
		attnOut:  e.alloc("attn_out", cudahostemu.Poison(4*qDim)),
		norm:     e.alloc("norm", f32le(w.Norm)),
		lmHead:   e.alloc("lm_head", bf16le(w.LMHead)),
		logits:   e.alloc("logits", cudahostemu.Poison(4*cfg.VocabSize)),
		q:        e.alloc("q", cudahostemu.Poison(4*qDim)),
		qbf:      e.alloc("qbf", cudahostemu.Poison(2*qDim)),
		cs:       e.alloc("cos_sin", f32le(cs)),
		ropePage: e.alloc("rope_page", i32le([]int32{0})),
		seqLen1:  e.alloc("seqlen1", i32le([]int32{1})),
		pages:    e.alloc("pages", i32le(pages)),
		ws:       e.alloc("ws", cudahostemu.Poison(4*plans.sdpa.WorkspaceFloats(1))),
	}
	weights[e.b.lmHead] = true
	for i, l := range w.Layers {
		lb := egraphLayerBuffers{
			attnNorm: e.alloc(fmt.Sprintf("l%d_attn_norm", i), f32le(l.InputNorm)),
			wqkv:     e.alloc(fmt.Sprintf("l%d_wqkv", i), bf16le(concat(l.WQ, l.WK, l.WV))),
			wo:       e.alloc(fmt.Sprintf("l%d_wo", i), bf16le(l.WO)),
			ffnNorm:  e.alloc(fmt.Sprintf("l%d_ffn_norm", i), f32le(l.PostNorm)),
			wgu:      e.alloc(fmt.Sprintf("l%d_wgu", i), bf16le(concat(l.WGate, l.WUp))),
			wdown:    e.alloc(fmt.Sprintf("l%d_wdown", i), bf16le(l.WDown)),
			k:        e.alloc(fmt.Sprintf("l%d_k", i), make([]byte, 2*cfg.MaxSeqLen*kvDim)),
			v:        e.alloc(fmt.Sprintf("l%d_v", i), make([]byte, 2*cfg.MaxSeqLen*kvDim)),
		}
		for _, p := range []driver.DevicePtr{lb.wqkv, lb.wo, lb.wgu, lb.wdown} {
			weights[p] = true
		}
		e.b.layers = append(e.b.layers, lb)
	}
	if err := e.eg.bindDecode(cfg, e.b); err != nil {
		t.Fatal(err)
	}
	if e.encodes != len(e.eg.tmaps) {
		t.Fatalf("%d tensor map encodes for %d distinct maps", e.encodes, len(e.eg.tmaps))
	}
	return e
}

// alloc adds a buffer holding data and returns its synthetic address.
func (e *emuDecoder) alloc(name string, data []byte) driver.DevicePtr {
	e.bufs = append(e.bufs, cudahostemu.Buffer{Name: name, CType: "uint8_t", Data: data})
	return driver.DevicePtr(len(e.bufs)) << 32
}

// ptr returns the C++ expression for the synthetic address p.
func (e *emuDecoder) ptr(p driver.DevicePtr) string {
	if p == 0 {
		return "(void*)nullptr"
	}
	return fmt.Sprintf("(void*)(%s + %d)", e.bufs[p>>32-1].Name, p&(1<<32-1))
}

// launch renders l as a call of eg_call at position pos, a C++ variable.
func (e *emuDecoder) launch(t *testing.T, l egraphLaunch) string {
	t.Helper()
	name, ok := e.names[l.fn]
	if !ok {
		t.Fatalf("launch of unknown function %d", l.fn)
	}
	args := make([]string, len(l.args))
	for i, a := range l.args {
		switch a := a.(type) {
		case int32:
			args[i] = fmt.Sprintf("(int32_t)%d", a)
		case uint32:
			args[i] = fmt.Sprintf("(uint32_t)%d", a)
		case driver.DevicePtr:
			args[i] = e.ptr(a)
		case *driver.TensorMap:
			j := slices.Index(e.mapOrder, a)
			if j < 0 {
				t.Fatalf("%s: unrecorded tensor map", name)
			}
			args[i] = fmt.Sprintf("tm%d", j)
		default:
			t.Fatalf("%s: argument %d has type %T", name, i, a)
		}
	}
	for _, pa := range l.pos {
		if pa.stride == 0 {
			args[pa.idx] = "(int32_t)(pos + 1)"
			continue
		}
		base := l.args[pa.idx].(driver.DevicePtr)
		args[pa.idx] = fmt.Sprintf("(void*)(%s + %d + (size_t)pos * %d)", e.bufs[base>>32-1].Name, base&(1<<32-1), pa.stride)
	}
	return fmt.Sprintf("eg_call(%s, dim3(%d, %d, %d), dim3(%d, %d, %d), %d, %s);", name,
		l.grid[0], l.grid[1], l.grid[2], l.block[0], l.block[1], l.block[2], l.smem, strings.Join(args, ", "))
}

// emuCall launches kernel k with each argument converted to its parameter
// type, as cuLaunchKernel reads each parameter's bytes.
const emuCall = `
#include <type_traits>
template <class P, class A> static P eg_arg(A a) {
    if constexpr (std::is_pointer_v<P>) return reinterpret_cast<P>(a);
    else return static_cast<P>(a);
}
template <class... P, class... A> static void eg_call(void (*k)(P...), dim3 g, dim3 b, size_t s, A... a) {
    static_assert(sizeof...(P) == sizeof...(A), "argument count");
    eg_emu_launch(g, b, s, k, eg_arg<P>(a)...);
}
`

// run decodes steps tokens greedily from first and returns each step's
// logits and token.
func (e *emuDecoder) run(t *testing.T, first int32, steps int) ([][]float32, []int32) {
	t.Helper()
	cfg := e.cfg
	src, _, err := e.eg.source()
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for i, m := range e.mapOrder {
		fmt.Fprintf(&b, "const eg_tmap tm%d = %s;\n", i, e.maps[m])
	}
	fmt.Fprintf(&b, "int32_t tok = %d;\nfor (int pos = 0; pos < %d; ++pos) {\n", first, steps)
	fmt.Fprintf(&b, "for (int i = 0; i < %d; ++i) { uint32_t u = (uint32_t)((uint16_t*)embed)[tok * %d + i] << 16; memcpy((float*)x + i, &u, 4); }\n",
		cfg.HiddenSize, cfg.HiddenSize)
	for i := range cfg.NumHiddenLayers {
		for _, l := range e.eg.layerLaunches(i) {
			b.WriteString(e.launch(t, l) + "\n")
		}
	}
	for _, l := range e.eg.head {
		b.WriteString(e.launch(t, l) + "\n")
	}
	fmt.Fprintf(&b, `const float* lg = (const float*)logits;
int best = 0;
for (int i = 1; i < %d; ++i) if (lg[i] > lg[best]) best = i;
memcpy((float*)emu_logits + pos * %d, lg, %d);
((int32_t*)emu_tokens)[pos] = best;
tok = best;
}`, cfg.VocabSize, cfg.VocabSize, 4*cfg.VocabSize)
	bufs := append(slices.Clone(e.bufs),
		cudahostemu.Buffer{Name: "emu_logits", CType: "uint8_t", Data: cudahostemu.Poison(4 * steps * cfg.VocabSize)},
		cudahostemu.Buffer{Name: "emu_tokens", CType: "uint8_t", Data: make([]byte, 4*steps)})
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src + emuCall,
		Launch:  b.String(),
		Buffers: bufs,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated decode not run")
	}
	if err != nil {
		t.Fatalf("emulated decode: %v", err)
	}
	logits := make([][]float32, steps)
	tokens := make([]int32, steps)
	for s := range steps {
		logits[s] = make([]float32, cfg.VocabSize)
		for i := range logits[s] {
			logits[s][i] = math.Float32frombits(binary.LittleEndian.Uint32(res["emu_logits"][4*(s*cfg.VocabSize+i):]))
		}
		tokens[s] = int32(binary.LittleEndian.Uint32(res["emu_tokens"][4*s:]))
	}
	return logits, tokens
}

// randomBF16Weights returns weights for cfg drawn from a normal
// distribution and rounded to BF16, so the CPU reference reads the values
// the BF16 kernels do.
func randomBF16Weights(cfg *ModelConfig, r *rand.Rand) *ModelWeights {
	h, hd := cfg.HiddenSize, cfg.HeadDim()
	qDim, kvDim, inter := cfg.NumAttentionHeads*hd, cfg.NumKeyValueHeads*hd, cfg.IntermediateSize
	mat := func(out, in int) []float32 {
		v := make([]float32, out*in)
		s := 1 / math.Sqrt(float64(in))
		for i := range v {
			v[i] = hopper.BF16.Round(float32(r.NormFloat64() * s))
		}
		return v
	}
	norm := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(1 + 0.1*r.NormFloat64())
		}
		return v
	}
	w := &ModelWeights{EmbedTokens: mat(cfg.VocabSize, h), Norm: norm(h), LMHead: mat(cfg.VocabSize, h)}
	for range cfg.NumHiddenLayers {
		w.Layers = append(w.Layers, LayerWeights{
			InputNorm: norm(h), PostNorm: norm(h),
			WQ: mat(qDim, h), WK: mat(kvDim, h), WV: mat(kvDim, h), WO: mat(h, qDim),
			WGate: mat(inter, h), WUp: mat(inter, h), WDown: mat(h, inter),
		})
	}
	return w
}

func bf16le(v []float32) []byte {
	b := make([]byte, 2*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint16(b[2*i:], hopper.BF16Bits(f))
	}
	return b
}

// maxAbsDiff returns the largest |got-want| and the largest |want|.
func maxAbsDiff(got, want []float32) (d, scale float64) {
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) {
			return math.Inf(1), 1
		}
		d = max(d, math.Abs(g-w))
		scale = max(scale, math.Abs(w))
	}
	return d, scale
}

func argmax(v []float32) int32 {
	best := 0
	for i, f := range v {
		if f > v[best] {
			best = i
		}
	}
	return int32(best)
}
