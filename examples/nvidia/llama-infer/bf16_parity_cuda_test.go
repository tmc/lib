package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

// The BF16 gates are bit equality, preregistered before any device run.
// LinearGEMVBF16 keeps LinearGEMV's partition and expression order, which is
// a reason to expect equal bits on widened weights, not a proof: contraction
// into FMAs is the compiler's choice. A mismatch is a finding, and no
// tolerance replaces these gates without a new ruling.

// bf16Widen is the host widening of a bfloat16 bit pattern. It is exact.
func bf16Widen(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// bf16Weights returns n random bfloat16 values in [-1, 1]. Bounding them
// keeps every K1 dot product far from float32 overflow, so a non-finite
// output is a defect and never the fixture: weights spanning every exponent
// made rows of 2048 overflow in any summation order. Special values are
// covered by K2 instead.
func bf16Weights(r *rand.Rand, n int) []uint16 {
	w := make([]uint16, n)
	for i := range w {
		w[i] = safetensors.Float32ToBFloat16(float32(r.Float64()*2 - 1))
	}
	return w
}

// bf16Activations returns n float32 values whose low mantissa bits are in
// use, so that a dropped or misplaced weight bit changes the products.
func bf16Activations(r *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(r.Float64()*2 - 1)
	}
	return x
}

func uploadBF16(d *parityDevice, host []uint16) cudamem.Slice[uint16] {
	d.t.Helper()
	s, err := cudamem.Alloc[uint16](d.dev, len(host))
	d.t.Cleanup(func() {
		if err := s.Free(); err != nil {
			d.t.Error(err)
		}
	})
	if err != nil {
		d.t.Fatal(err)
	}
	if err := s.CopyFromHost(host); err != nil {
		d.t.Fatal(err)
	}
	return s
}

func runGEMV(d *parityDevice, fn driver.Function, x cudamem.Slice[float32], w any, in, out int) []float32 {
	d.t.Helper()
	o := d.upload(make([]float32, out))
	if err := cudalaunch.Kernel(fn).Grid(uint32(out), 1, 1).Block(128, 1, 1).
		Launch(x, w, nullDevicePtr, o, int32(in), int32(out)); err != nil {
		d.t.Fatal(err)
	}
	return d.download(o)
}

// bitMismatch reports the first index where got is non-finite or its bits
// differ from want, or -1.
func bitMismatch(got, want []float32) int {
	if len(got) != len(want) {
		return 0
	}
	for i := range want {
		g := float64(got[i])
		if math.IsNaN(g) || math.IsInf(g, 0) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			return i
		}
	}
	return -1
}

// gemvBF16Mismatch runs gm's LinearGEMVBF16 on random bfloat16 weights and
// LinearGEMV from ref on the same weights widened, and reports the first
// index whose bits differ, or -1.
func gemvBF16Mismatch(d *parityDevice, gm, ref *GPUModules, in, out int) (int, []float32, []float32) {
	d.t.Helper()
	r := rand.New(rand.NewPCG(uint64(in), uint64(out)))
	x, w16 := bf16Activations(r, in), bf16Weights(r, in*out)
	w32 := make([]float32, len(w16))
	for i, b := range w16 {
		w32[i] = bf16Widen(b)
	}
	dx := d.upload(x)
	got := runGEMV(d, gm.gemvBF16Fn, dx, uploadBF16(d, w16), in, out)
	want := runGEMV(d, ref.gemvFn, dx, d.upload(w32), in, out)
	return bitMismatch(got, want), got, want
}

var gemvBF16Shapes = func() (s [][2]int) {
	for _, in := range []int{5, 7, 8, 13, 64, 2048} {
		for _, out := range []int{1, 3, 64} {
			s = append(s, [2]int{in, out})
		}
	}
	return s
}()

// TestBF16FixturesFinite checks without a device that every K1 case has a
// finite exact result, so that a non-finite device output can only be a
// kernel defect.
func TestBF16FixturesFinite(t *testing.T) {
	for _, s := range gemvBF16Shapes {
		in, out := s[0], s[1]
		r := rand.New(rand.NewPCG(uint64(in), uint64(out)))
		x, w16 := bf16Activations(r, in), bf16Weights(r, in*out)
		for o := 0; o < out; o++ {
			bound := 0.0
			for i := 0; i < in; i++ {
				bound += math.Abs(float64(x[i]) * float64(bf16Widen(w16[o*in+i])))
			}
			if bound > math.MaxFloat32/2 {
				t.Errorf("in %d out %d row %d: |x|.|w| = %g risks overflow", in, out, o, bound)
			}
		}
	}
}

// TestLinearGEMVBF16BitExact is oracle K1.
func TestLinearGEMVBF16BitExact(t *testing.T) {
	d := newParityDevice(t)
	for _, s := range gemvBF16Shapes {
		in, out := s[0], s[1]
		t.Run(fmt.Sprintf("in %d out %d", in, out), func(t *testing.T) {
			d := &parityDevice{t: t, dev: d.dev, gm: d.gm}
			if i, got, want := gemvBF16Mismatch(d, d.gm, d.gm, in, out); i >= 0 {
				t.Errorf("out[%d]: bf16 %#08x, widened f32 %#08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		})
	}
}

// convertMismatches widens every bfloat16 bit pattern with gm's probe and
// counts results whose bits differ from bf16Widen.
func convertMismatches(d *parityDevice, gm *GPUModules) int {
	d.t.Helper()
	const n = 1 << 16
	in := make([]uint16, n)
	for i := range in {
		in[i] = uint16(i)
	}
	o := d.upload(make([]float32, n))
	if err := cudalaunch.Kernel(gm.convertBF16Fn).Grid(n/256, 1, 1).Block(256, 1, 1).
		Launch(uploadBF16(d, in), o, int32(n)); err != nil {
		d.t.Fatal(err)
	}
	bad := 0
	for i, v := range d.download(o) {
		if math.Float32bits(v) != math.Float32bits(bf16Widen(in[i])) {
			bad++
		}
	}
	return bad
}

// TestWeightConvertProbeBF16 is oracle K2: every bit pattern, including -0,
// subnormals, infinities and NaN payloads.
func TestWeightConvertProbeBF16(t *testing.T) {
	d := newParityDevice(t)
	if n := convertMismatches(d, d.gm); n != 0 {
		t.Errorf("%d of 65536 bit patterns widened wrongly", n)
	}
}

// embedBF16Mismatch looks up token tok in a random bfloat16 table with gm's
// EmbeddingLookupBF16 and in the same table widened with ref's
// EmbeddingLookup, and reports the first index whose bits differ, or -1.
func embedBF16Mismatch(d *parityDevice, gm, ref *GPUModules, vocab, hidden, tok int) int {
	d.t.Helper()
	r := rand.New(rand.NewPCG(uint64(vocab), uint64(hidden)))
	w16 := bf16Weights(r, vocab*hidden)
	w32 := make([]float32, len(w16))
	for i, b := range w16 {
		w32[i] = bf16Widen(b)
	}
	dt, err := cudamem.Alloc[int32](d.dev, 1)
	d.t.Cleanup(func() {
		if err := dt.Free(); err != nil {
			d.t.Error(err)
		}
	})
	if err != nil {
		d.t.Fatal(err)
	}
	if err := dt.CopyFromHost([]int32{int32(tok)}); err != nil {
		d.t.Fatal(err)
	}
	run := func(fn driver.Function, table any) []float32 {
		o := d.upload(make([]float32, hidden))
		if err := cudalaunch.Kernel(fn).Grid(1, 1, 1).Block(256, 1, 1).
			Launch(table, dt, o, int32(hidden)); err != nil {
			d.t.Fatal(err)
		}
		return d.download(o)
	}
	return bitMismatch(run(gm.embedBF16Fn, uploadBF16(d, w16)), run(ref.embedFn, d.upload(w32)))
}

// embedBF16Shapes are (vocab, hidden, token) cases, including the first and
// last rows.
var embedBF16Shapes = [][3]int{{3, 5, 0}, {3, 5, 2}, {4, 8, 1}, {5, 13, 4}, {7, 64, 3}}

// TestEmbeddingLookupBF16BitExact is oracle E1.
func TestEmbeddingLookupBF16BitExact(t *testing.T) {
	d := newParityDevice(t)
	for _, s := range embedBF16Shapes {
		if i := embedBF16Mismatch(d, d.gm, d.gm, s[0], s[1], s[2]); i >= 0 {
			t.Errorf("vocab %d hidden %d token %d: out[%d] differs", s[0], s[1], s[2], i)
		}
	}
}

// Source fragments the BF16 controls replace.
const (
	bf16LanesSource  = "bf16_to_f32(wb.x), bf16_to_f32(wb.y)"
	bf16LanesSwapped = "bf16_to_f32(wb.y), bf16_to_f32(wb.x)"
	bf16WidenSource  = "((unsigned int)b) << 16"
	bf16EmbedRow     = "(size_t)(*token_id) * hidden_dim"
)

// TestBF16MutationsApply checks without a device that every fragment the
// controls replace occurs exactly once, so each mutant changes the kernel.
func TestBF16MutationsApply(t *testing.T) {
	for _, old := range []string{bf16LanesSource, bf16WidenSource, bf16EmbedRow} {
		if n := strings.Count(forwardBF16CUDA, old); n != 1 {
			t.Errorf("%q occurs %d times in forward_bf16.cu, want 1", old, n)
		}
	}
}

// mutateBF16 compiles the forward kernels with old replaced by new in the
// BF16 source. A replacement that does not apply is fatal: a mutant that is
// not compiled cannot be detected.
func mutateBF16(d *parityDevice, old, new string) *GPUModules {
	d.t.Helper()
	if n := strings.Count(forwardBF16CUDA, old); n != 1 {
		d.t.Fatalf("mutation %q matches %d times, want 1", old, n)
	}
	gm, err := compileGPUModules(d.dev, ForwardKernelsCUDA+"\n"+strings.Replace(forwardBF16CUDA, old, new, 1))
	if gm != nil {
		d.t.Cleanup(func() {
			if err := gm.Close(); err != nil {
				d.t.Error(err)
			}
		})
	}
	if err != nil {
		d.t.Fatalf("mutation %q: %v", old, err)
	}
	return gm
}

// TestBF16Controls checks that the K1 and K2 gates see the defects they
// exist to catch. Every entry in mustFail must be detected. inactivePath
// entries change code a shape never runs and must not be detected there:
// they show the gate is specific to the path, and are not failures.
func TestBF16Controls(t *testing.T) {
	d := newParityDevice(t)
	lanes, swapped, widen := bf16LanesSource, bf16LanesSwapped, bf16WidenSource

	mustFail := []struct {
		name     string
		old, new string
		shapes   [][2]int // K1 shapes that must detect it
		convert  bool     // K2 must detect it
		embed    bool     // E1 must detect it on every embedBF16Shapes case that has a next row
	}{
		{"permuted ushort4 lanes", lanes, swapped, [][2]int{{8, 3}, {64, 3}}, false, false},
		{"shift by 15", widen, "((unsigned int)b) << 15", [][2]int{{8, 3}, {13, 3}, {64, 3}}, true, true},
		{"sign bit dropped", widen, "((unsigned int)(b & 0x7fff)) << 16", nil, true, false},
		{"embedding row off by one", bf16EmbedRow, "(size_t)(*token_id + 1) * hidden_dim", nil, false, true},
	}
	for _, m := range mustFail {
		gm := mutateBF16(d, m.old, m.new)
		for _, s := range m.shapes {
			if i, _, _ := gemvBF16Mismatch(d, gm, d.gm, s[0], s[1]); i < 0 {
				t.Errorf("%s: K1 in %d out %d did not detect it", m.name, s[0], s[1])
			}
		}
		if m.convert && convertMismatches(d, gm) == 0 {
			t.Errorf("%s: K2 did not detect it", m.name)
		}
		if m.embed {
			// Only rows with a next row: reading past the table's last row
			// would rest detection on memory outside the allocation.
			n := 0
			for _, s := range embedBF16Shapes {
				if s[2]+1 >= s[0] {
					continue
				}
				n++
				if embedBF16Mismatch(d, gm, d.gm, s[0], s[1], s[2]) < 0 {
					t.Errorf("%s: E1 vocab %d hidden %d token %d did not detect it", m.name, s[0], s[1], s[2])
				}
			}
			if n == 0 {
				t.Errorf("%s: no embedding case has a next row", m.name)
			}
		}
	}

	inactivePath := []struct {
		name     string
		old, new string
		shapes   [][2]int // shapes that never take the mutated path
	}{
		{"permuted ushort4 lanes, scalar rows", lanes, swapped, [][2]int{{5, 3}, {7, 3}, {13, 3}}},
	}
	for _, m := range inactivePath {
		gm := mutateBF16(d, m.old, m.new)
		for _, s := range m.shapes {
			if i, _, _ := gemvBF16Mismatch(d, gm, d.gm, s[0], s[1]); i >= 0 {
				t.Errorf("%s: in %d out %d changed out[%d] on a path it does not run", m.name, s[0], s[1], i)
			}
		}
	}
}

// roundedModel returns a copy of w whose matrices hold bfloat16 values
// widened back to float32, so that f32 and bf16 storage hold the same
// numbers. A tied lm_head stays tied.
func roundedModel(w *ModelWeights) *ModelWeights {
	round := func(v []float32) []float32 {
		r := make([]float32, len(v))
		for i, b := range roundBF16(v) {
			r[i] = bf16Widen(b)
		}
		return r
	}
	c := *w
	c.EmbedTokens = round(w.EmbedTokens)
	c.LMHead = round(w.LMHead)
	if tiedHead(w) {
		c.LMHead = c.EmbedTokens
	}
	c.Layers = slices.Clone(w.Layers)
	for i := range c.Layers {
		l := &c.Layers[i]
		l.WQ, l.WK, l.WV, l.WO = round(l.WQ), round(l.WK), round(l.WV), round(l.WO)
		l.WGate, l.WUp, l.WDown = round(l.WGate), round(l.WUp), round(l.WDown)
	}
	return &c
}

type forwardKit struct {
	d                *parityDevice
	attnFn, sampleFn driver.Function
}

func newForwardKit(t *testing.T) *forwardKit {
	d := newParityDevice(t)
	major, minor, err := d.dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	attnCompiled, err := cudaattn.Compile(major, minor)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	attnMod, err := attnCompiled.Load(d.dev)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	t.Cleanup(func() { attnMod.Close() })
	attnFn, err := attnMod.Function("CausalFlashAttentionDecode")
	if err != nil {
		t.Fatal(err)
	}
	sampleCompiled, err := cudasample.Compile(major, minor)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	sampleMod, err := sampleCompiled.Load(d.dev)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	t.Cleanup(func() { sampleMod.Close() })
	sampleFn, err := sampleMod.Function("ArgmaxLogits")
	if err != nil {
		t.Fatal(err)
	}
	return &forwardKit{d: d, attnFn: attnFn, sampleFn: sampleFn}
}

const bf16ParitySteps = 6

// decode runs bf16ParitySteps greedy steps from token 7 with w's matrices
// stored as dtype, and returns each step's logits and the chosen tokens.
func (k *forwardKit) decode(t *testing.T, w *ModelWeights, cfg ModelConfig, dtype weightDType) ([][]float32, []int32) {
	t.Helper()
	gw, err := uploadModel(k.d.dev, w, &cfg, dtype)
	if gw != nil {
		defer func() {
			if err := gw.Free(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	kv, err := cudakv.New(k.d.dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, cfg.HeadDim(), cfg.MaxSeqLen)
	if kv != nil {
		defer func() {
			if err := kv.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	sp, err := NewGPUScratchpad(k.d.dev, &cfg)
	if sp != nil {
		defer func() {
			if err := sp.Free(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	var logits [][]float32
	var toks []int32
	tok := int32(7)
	for pos := 0; pos < bf16ParitySteps; pos++ {
		next, err := GPUForwardPass(gw, &cfg, kv, sp, k.d.gm, k.attnFn, k.sampleFn, tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		logits = append(logits, k.d.download(sp.Logits))
		toks = append(toks, next)
		tok = next
	}
	return logits, toks
}

// firstDifference reports the first step whose logits differ in any bit or
// whose token differs, or -1.
func firstDifference(a, b [][]float32, at, bt []int32) int {
	for i := range a {
		if bitMismatch(a[i], b[i]) >= 0 || at[i] != bt[i] {
			return i
		}
	}
	return -1
}

func bf16ModelConfigs() []struct {
	name string
	cfg  ModelConfig
	tied bool
} {
	odd := DefaultModelConfig()
	// Query width equals hidden: CPU Forward's output projection assumes it.
	odd.HiddenSize, odd.IntermediateSize, odd.NumAttentionHeads, odd.NumKeyValueHeads, odd.VocabSize = 18, 30, 3, 1, 50
	return []struct {
		name string
		cfg  ModelConfig
		tied bool
	}{
		{"default", DefaultModelConfig(), false},
		{"default tied", DefaultModelConfig(), true},
		{"odd rows", odd, false},
	}
}

// distinctWeights returns weights shaped for cfg whose every matrix, norm
// and layer holds its own random values. GenerateDeterministicWeights gives
// every layer the same values and several matrices equal ones, which would
// hide a swapped or wrong-layer matrix.
func distinctWeights(cfg *ModelConfig) *ModelWeights {
	w := GenerateDeterministicWeights(cfg)
	r := rand.New(rand.NewPCG(1, 2))
	fill := func(v []float32, center, scale float64) {
		for i := range v {
			v[i] = float32(center + (r.Float64()*2-1)*scale)
		}
	}
	fill(w.EmbedTokens, 0, 0.1)
	fill(w.LMHead, 0, 0.02)
	fill(w.Norm, 1, 0.1)
	for i := range w.Layers {
		l := &w.Layers[i]
		fill(l.InputNorm, 1, 0.1)
		fill(l.PostNorm, 1, 0.1)
		for _, m := range [][]float32{l.WQ, l.WK, l.WV, l.WO, l.WGate, l.WUp, l.WDown} {
			fill(m, 0, 0.02)
		}
	}
	return w
}

func bf16Model(cfg ModelConfig, tied bool) *ModelWeights {
	w := distinctWeights(&cfg)
	if tied {
		w.LMHead = w.EmbedTokens
	}
	return roundedModel(w)
}

// TestBF16ModelParity is oracle M1 (bf16 storage is bit-identical to f32
// storage of the same values, at every step) and M2 (bf16 storage agrees
// with the CPU Forward within the preregistered logit tolerance, with the
// same argmax).
func TestBF16ModelParity(t *testing.T) {
	k := newForwardKit(t)
	for _, tt := range bf16ModelConfigs() {
		t.Run(tt.name, func(t *testing.T) {
			w := bf16Model(tt.cfg, tt.tied)
			f32, f32Toks := k.decode(t, w, tt.cfg, weightF32)
			b16, b16Toks := k.decode(t, w, tt.cfg, weightBF16)
			if i := firstDifference(f32, b16, f32Toks, b16Toks); i >= 0 {
				j := bitMismatch(b16[i], f32[i])
				t.Errorf("M1 step %d: bf16 differs from f32 (logit %d, tokens %d/%d)", i, j, b16Toks[i], f32Toks[i])
			}
			cfg := tt.cfg
			hostKV := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, cfg.NumKeyValueHeads*cfg.HeadDim())
			tok := int32(7)
			for pos := range b16 {
				want := Forward(w, &cfg, hostKV, tok, pos)
				if i := parityMismatch(b16[pos], want, logitTolerance); i >= 0 {
					t.Fatalf("M2 step %d logit %d: got %g want %g", pos, i, b16[pos][i], want[i])
				}
				if hostNext := sampleArgmax(want); b16Toks[pos] != hostNext {
					t.Fatalf("M2 step %d: device token %d, host token %d", pos, b16Toks[pos], hostNext)
				}
				tok = b16Toks[pos]
			}
		})
	}
}

// TestBF16ModelControls are must-fail controls for M1. Each changes the
// numbers one matrix holds by a real bfloat16 representation difference,
// and M1 must see it.
func TestBF16ModelControls(t *testing.T) {
	k := newForwardKit(t)
	cfg := DefaultModelConfig()

	// MC1: one layer's q_proj kept as unrounded float32, as a model that
	// stored one matrix in the wrong type would compute.
	src := distinctWeights(&cfg)
	w := roundedModel(src)
	mixed := roundedModel(src)
	mixed.Layers = slices.Clone(mixed.Layers)
	mixed.Layers[0].WQ = src.Layers[0].WQ
	if slices.Equal(mixed.Layers[0].WQ, w.Layers[0].WQ) {
		t.Fatal("MC1 is vacuous: q_proj is already bfloat16-representable")
	}
	b16, b16Toks := k.decode(t, w, cfg, weightBF16)
	f32, f32Toks := k.decode(t, mixed, cfg, weightF32)
	if firstDifference(b16, f32, b16Toks, f32Toks) < 0 {
		t.Error("MC1 one unrounded float32 matrix: not detected")
	}

	// MC2: an untied lm_head one bfloat16 step away from the table it
	// should be tied to.
	tied := bf16Model(cfg, true)
	bumped := *tied
	bumped.LMHead = slices.Clone(tied.EmbedTokens)
	bumped.LMHead[0] = bf16Widen(roundBF16(bumped.LMHead[:1])[0] + 1)
	tb, tbToks := k.decode(t, tied, cfg, weightBF16)
	pb, pbToks := k.decode(t, &bumped, cfg, weightBF16)
	if firstDifference(tb, pb, tbToks, pbToks) < 0 {
		t.Error("MC2 lm_head one step from the tied table: not detected")
	}
}
