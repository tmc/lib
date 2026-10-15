package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestRoundBF16(t *testing.T) {
	for _, tt := range []struct {
		name string
		in   uint32 // float32 bits
		want uint16
	}{
		{"one", 0x3f800000, 0x3f80},
		{"tie to even, down", 0x3f808000, 0x3f80},
		{"tie to even, up", 0x3f818000, 0x3f82},
		{"above tie", 0x3f808001, 0x3f81},
		{"below tie", 0x3f807fff, 0x3f80},
		{"negative zero", 0x80000000, 0x8000},
		{"subnormal tie", 0x00008000, 0x0000},
		{"subnormal", 0x00018000, 0x0002},
		{"infinity", 0x7f800000, 0x7f80},
		{"overflow to infinity", 0x7f7fffff, 0x7f80},
		{"negative overflow", 0xff7fffff, 0xff80},
	} {
		if got := roundBF16([]float32{math.Float32frombits(tt.in)})[0]; got != tt.want {
			t.Errorf("%s: roundBF16(%#08x) = %#04x, want %#04x", tt.name, tt.in, got, tt.want)
		}
	}
	nan := roundBF16([]float32{float32(math.NaN())})[0]
	if nan&0x7f80 != 0x7f80 || nan&0x7f == 0 {
		t.Errorf("NaN rounded to %#04x, not a NaN", nan)
	}
	// Rounding a widened value returns its bits: a BF16 checkpoint loads exactly.
	for b := 0; b < 1<<16; b++ {
		h := uint16(b)
		if h&0x7f80 == 0x7f80 && h&0x7f != 0 {
			continue // NaN payloads are quieted, not preserved
		}
		if got := roundBF16([]float32{bf16Widen(h)})[0]; got != h {
			t.Fatalf("round(widen(%#04x)) = %#04x", h, got)
		}
	}
}

func TestParseWeightDType(t *testing.T) {
	for _, s := range []string{"f32", "bf16"} {
		if d, err := parseWeightDType(s); err != nil || string(d) != s {
			t.Errorf("parseWeightDType(%q) = %q, %v", s, d, err)
		}
	}
	for _, s := range []string{"", "BF16", "fp16", "f16", "float32"} {
		if _, err := parseWeightDType(s); err == nil {
			t.Errorf("parseWeightDType(%q) accepted", s)
		}
	}
}

// fakeBuffer is a deviceBuffer that records frees in a shared log. Its first
// fails frees return err.
type fakeBuffer struct {
	name  string
	fails int
	err   error
	log   *[]string
}

func (b *fakeBuffer) Ptr() driver.DevicePtr { return 1 }

func (b *fakeBuffer) Free() error {
	if b.fails > 0 {
		b.fails--
		return b.err
	}
	*b.log = append(*b.log, b.name)
	return nil
}

var errFakeFree = errors.New("fake free failed")

// fakeModel returns a one-layer model whose matrices are fake buffers. With
// tied set, LMHead borrows EmbedTokens. embedFails sets how many times the
// embedding table's free fails before it succeeds.
func fakeModel(log *[]string, tied bool, embedFails int) *GPUModelWeights {
	w := &GPUModelWeights{DType: weightBF16, Layers: []GPULayerWeights{fakeLayer(log, "")}}
	w.EmbedTokens = deviceWeight{buf: &fakeBuffer{name: "embed", fails: embedFails, err: errFakeFree, log: log}, dtype: weightBF16}
	if tied {
		w.LMHead = w.EmbedTokens.borrow()
	} else {
		w.LMHead = deviceWeight{buf: &fakeBuffer{name: "head", log: log}, dtype: weightBF16}
	}
	return w
}

// fakeLayer returns a layer whose matrices are fake buffers named
// fakeLayerFrees(suffix).
func fakeLayer(log *[]string, suffix string) GPULayerWeights {
	var l GPULayerWeights
	names := fakeLayerFrees(suffix)
	for i, m := range layerMatrices(&l) {
		*m = deviceWeight{buf: &fakeBuffer{name: names[i], log: log}, dtype: weightBF16}
	}
	return l
}

// fakeLayerFrees returns the names of a fake layer's matrices in the order
// free releases them.
func fakeLayerFrees(suffix string) []string {
	var names []string
	for _, s := range []string{"wq", "wk", "wv", "wo", "gate", "up", "down"} {
		names = append(names, s+suffix)
	}
	return names
}

// layerMatrices returns pointers to l's matrices in the order free
// releases them.
func layerMatrices(l *GPULayerWeights) []*deviceWeight {
	return []*deviceWeight{&l.WQ, &l.WK, &l.WV, &l.WO, &l.WGate, &l.WUp, &l.WDown}
}

// layerVectors returns pointers to l's norms and biases in the order free
// frees them.
func layerVectors(l *GPULayerWeights) []*cudamem.Slice[float32] {
	return []*cudamem.Slice[float32]{&l.InputNorm, &l.PostNorm, &l.QBias, &l.KBias, &l.VBias}
}

// ownershipFailures returns how free departs from releasing each allocation
// exactly once: across a tied and an untied model, a repeated free, and a
// first free of the embedding table that fails and is retried.
func ownershipFailures(free func(*GPUModelWeights) error) []string {
	var bad []string
	count := func(log []string, name string) int {
		n := 0
		for _, s := range log {
			if s == name {
				n++
			}
		}
		return n
	}
	check := func(scenario string, log []string, want map[string]int) {
		for name, n := range want {
			if got := count(log, name); got != n {
				bad = append(bad, fmt.Sprintf("%s: %s freed %d times, want %d", scenario, name, got, n))
			}
		}
	}
	layer := func(m map[string]int) map[string]int {
		for _, s := range fakeLayerFrees("") {
			m[s] = 1
		}
		return m
	}

	var log []string
	w := fakeModel(&log, true, 0)
	if err := free(w); err != nil {
		bad = append(bad, "tied: "+err.Error())
	}
	if err := free(w); err != nil {
		bad = append(bad, "tied, repeated: "+err.Error())
	}
	check("tied, freed twice", log, layer(map[string]int{"embed": 1, "head": 0}))
	if w.EmbedTokens.buf != nil || w.LMHead.buf != nil {
		bad = append(bad, "tied: weights not zeroed after free")
	}

	log = nil
	w = fakeModel(&log, false, 0)
	free(w)
	free(w)
	check("untied, freed twice", log, layer(map[string]int{"embed": 1, "head": 1}))

	log = nil
	w = fakeModel(&log, true, 1)
	err := free(w)
	if !errors.Is(err, errFakeFree) {
		bad = append(bad, fmt.Sprintf("failed free: error %v does not wrap the free's error", err))
	}
	if w.EmbedTokens.buf == nil {
		bad = append(bad, "failed free: embedding table dropped, cannot retry")
	}
	check("after a failed free", log, layer(map[string]int{"embed": 0, "head": 0}))
	if err := free(w); err != nil {
		bad = append(bad, "retry: "+err.Error())
	}
	free(w)
	check("after retry and repeat", log, layer(map[string]int{"embed": 1, "head": 0}))
	return bad
}

func TestGPUModelWeightsFree(t *testing.T) {
	for _, f := range ownershipFailures((*GPUModelWeights).Free) {
		t.Error(f)
	}
}

// TestGPUModelWeightsFreeControls are must-fail controls: each mutant
// breaks one ownership rule and must be reported by ownershipFailures.
func TestGPUModelWeightsFreeControls(t *testing.T) {
	for _, m := range []struct {
		name string
		free func(*GPUModelWeights) error
	}{
		{"O1 tied head owned", func(w *GPUModelWeights) error {
			w.LMHead.borrowed = false
			return w.Free()
		}},
		{"O2 embedding table skipped", func(w *GPUModelWeights) error {
			w.EmbedTokens = deviceWeight{}
			return w.Free()
		}},
		{"O3 not zeroed after free", func(w *GPUModelWeights) error {
			var errs []error
			for _, m := range w.matrices() {
				if m.buf != nil && !m.borrowed {
					errs = append(errs, m.buf.Free())
				}
			}
			return errors.Join(errs...)
		}},
		{"O4 zeroed after a failed free", func(w *GPUModelWeights) error {
			var errs []error
			for _, m := range w.matrices() {
				if m.buf != nil && !m.borrowed {
					errs = append(errs, m.buf.Free())
				}
				*m = deviceWeight{}
			}
			return errors.Join(errs...)
		}},
		{"O5 error replaced", func(w *GPUModelWeights) error {
			if err := w.Free(); err != nil {
				return errors.New("free failed")
			}
			return nil
		}},
	} {
		if len(ownershipFailures(m.free)) == 0 {
			t.Errorf("%s: not detected", m.name)
		}
	}
}

func TestCheckDTypes(t *testing.T) {
	var log []string
	w := fakeModel(&log, true, 0)
	if err := w.checkDTypes(); err != nil {
		t.Fatalf("uniform model refused: %v", err)
	}
	w.Layers[0].WK.dtype = weightF32
	if err := w.checkDTypes(); err == nil {
		t.Error("mixed dtypes accepted")
	}
	w = fakeModel(&log, true, 0)
	w.Layers[0].WDown = deviceWeight{}
	if err := w.checkDTypes(); err == nil {
		t.Error("missing matrix accepted")
	}
	// The tied head borrows the table, so it carries the table's dtype.
	if got := slices.Index(w.matrices(), &w.LMHead); got != 1 {
		t.Errorf("LMHead at matrices()[%d], want 1", got)
	}
}

// TestRoundedModel checks without a device that the M1 fixtures hold the
// differences the controls rely on.
func TestRoundedModel(t *testing.T) {
	cfg := DefaultModelConfig()
	src := GenerateDeterministicWeights(&cfg)
	r := roundedModel(src)
	changed := 0
	for i, v := range src.Layers[0].WQ {
		if r.Layers[0].WQ[i] != v {
			changed++
		}
	}
	if changed == 0 {
		t.Error("rounding left q_proj unchanged: MC1 would be vacuous")
	}
	if !slices.Equal(roundedModel(r).Layers[0].WQ, r.Layers[0].WQ) {
		t.Error("rounding is not idempotent")
	}
	if tiedHead(r) {
		t.Error("untied source became tied")
	}
	src.LMHead = src.EmbedTokens
	if !tiedHead(roundedModel(src)) {
		t.Error("tied source became untied")
	}
}

// TestCheckDTypesStorage checks that a label disagreeing with the buffer's
// element type is refused: a float32 buffer labelled bf16 would launch the
// bfloat16 kernel on float32 bytes.
func TestCheckDTypesStorage(t *testing.T) {
	var log []string
	w := fakeModel(&log, false, 0)
	w.Layers[0].WV = deviceWeight{buf: cudamem.Slice[float32]{}, dtype: weightBF16}
	if err := w.checkDTypes(); err == nil {
		t.Error("float32 buffer labelled bf16 accepted")
	}
	w.Layers[0].WV = deviceWeight{buf: cudamem.Slice[uint16]{}, dtype: weightBF16}
	if err := w.checkDTypes(); err != nil {
		t.Errorf("bfloat16 buffer labelled bf16 refused: %v", err)
	}
}

// TestForwardPassChecksDTypes checks without a device that the forward
// pass refuses a mixed model before it touches any device state: every
// device argument here is nil.
func TestForwardPassChecksDTypes(t *testing.T) {
	var log []string
	w := fakeModel(&log, true, 0)
	w.Layers[0].WK.dtype = weightF32
	cfg := DefaultModelConfig()
	cfg.NumHiddenLayers = 1
	if _, err := enqueueGPUForwardPass(w, &cfg, nil, nil, nil, 0, 0, 7, 0, nil); err == nil || !strings.Contains(err.Error(), "model is bf16") {
		t.Fatalf("mixed model: err = %v, want a dtype refusal", err)
	}
}

// TestDistinctWeights checks that the M1 fixture gives every matrix its own
// values, so that a swapped or wrong-layer matrix changes the result.
func TestDistinctWeights(t *testing.T) {
	cfg := DefaultModelConfig()
	w := distinctWeights(&cfg)
	var all [][]float32
	for i := range w.Layers {
		l := &w.Layers[i]
		all = append(all, l.WQ, l.WK, l.WV, l.WO, l.WGate, l.WUp, l.WDown, l.InputNorm, l.PostNorm)
	}
	all = append(all, w.EmbedTokens, w.LMHead, w.Norm)
	for i := range all {
		for j := range i {
			n := min(len(all[i]), len(all[j]))
			if slices.Equal(all[i][:n], all[j][:n]) {
				t.Errorf("weights %d and %d share their first %d values", i, j, n)
			}
		}
	}
}

func TestCheckToken(t *testing.T) {
	for _, tt := range []struct {
		id    int32
		vocab int
		ok    bool
	}{
		{-1, 5, false},
		{0, 5, true},
		{4, 5, true}, // positive control: the last token is accepted
		{5, 5, false},
		{0, 1, true},
		{1, 1, false},
		{0, 0, false},
	} {
		if err := checkToken(tt.id, tt.vocab); (err == nil) != tt.ok {
			t.Errorf("checkToken(%d, %d) = %v, want ok %v", tt.id, tt.vocab, err, tt.ok)
		}
	}
}

// TestForwardPassChecksToken checks without a device that the forward pass
// refuses an out-of-range token before it touches any device state: every
// device argument here is nil.
func TestForwardPassChecksToken(t *testing.T) {
	var log []string
	w := fakeModel(&log, true, 0)
	cfg := DefaultModelConfig()
	cfg.NumHiddenLayers = 1
	for _, tok := range []int32{-1, int32(cfg.VocabSize)} {
		if _, err := enqueueGPUForwardPass(w, &cfg, nil, nil, nil, 0, 0, tok, 0, nil); err == nil || !strings.Contains(err.Error(), "outside vocabulary") {
			t.Errorf("token %d: err = %v, want a range refusal", tok, err)
		}
	}
}

func TestCheckPos(t *testing.T) {
	for _, tt := range []struct {
		pos, maxSeq int
		ok          bool
	}{
		{-1, 4, false},
		{0, 4, true},
		{3, 4, true}, // positive control: the last position is accepted
		{4, 4, false},
		{0, 0, false},
	} {
		if err := checkPos(tt.pos, tt.maxSeq); (err == nil) != tt.ok {
			t.Errorf("checkPos(%d, %d) = %v, want ok %v", tt.pos, tt.maxSeq, err, tt.ok)
		}
	}
}

// TestForwardPassChecksPos checks without a device that the forward pass
// refuses a position outside the KV cache before it touches device state.
func TestForwardPassChecksPos(t *testing.T) {
	var log []string
	w := fakeModel(&log, true, 0)
	cfg := DefaultModelConfig()
	cfg.NumHiddenLayers = 1
	for _, pos := range []int{-1, cfg.MaxSeqLen} {
		if _, err := enqueueGPUForwardPass(w, &cfg, nil, nil, nil, 0, 0, 0, pos, nil); err == nil || !strings.Contains(err.Error(), "outside the kv cache") {
			t.Errorf("pos %d: err = %v, want a position refusal", pos, err)
		}
	}
}

func TestEffectivePrompt(t *testing.T) {
	for _, tt := range []struct {
		name   string
		tokens []int32
		vocab  int
		want   []int32 // nil means an error
	}{
		{"prompt", []int32{0, 4}, 5, []int32{0, 4}},
		{"empty uses token 1", nil, 2, []int32{1}},
		{"default outside a one-token vocabulary", nil, 1, nil},
		{"token past the vocabulary", []int32{0, 300}, 256, nil},
		{"negative token", []int32{-2}, 256, nil},
	} {
		got, err := effectivePrompt(tt.tokens, tt.vocab)
		if tt.want == nil {
			if err == nil {
				t.Errorf("%s: accepted %v", tt.name, got)
			}
			continue
		}
		if err != nil || !slices.Equal(got, tt.want) {
			t.Errorf("%s: got %v, %v; want %v", tt.name, got, err, tt.want)
		}
	}
}

func TestScratchpadFree(t *testing.T) {
	var sp GPUScratchpad
	if err := sp.Free(); err != nil {
		t.Fatalf("zero scratchpad: %v", err)
	}
	if err := sp.Free(); err != nil {
		t.Fatalf("repeated Free: %v", err)
	}
}

// fakeSlots frees device slices as freeOwned would, without a device: a
// slot in fails fails its first free with that error and stays owned; a
// successful free leaves the slot zero, so freeing it again does nothing.
type fakeSlots[T any] struct {
	fails    map[*cudamem.Slice[T]]error
	attempts map[*cudamem.Slice[T]]int
	frees    map[*cudamem.Slice[T]]int
}

func newFakeSlots[T any](fails map[*cudamem.Slice[T]]error) *fakeSlots[T] {
	return &fakeSlots[T]{fails: fails, attempts: make(map[*cudamem.Slice[T]]int), frees: make(map[*cudamem.Slice[T]]int)}
}

func (f *fakeSlots[T]) free(s *cudamem.Slice[T]) error {
	f.attempts[s]++
	if err, ok := f.fails[s]; ok {
		delete(f.fails, s)
		return err
	}
	if f.frees[s] == 0 {
		f.frees[s]++
	}
	return nil
}

// check reports each slot not attempted exactly attempts times, or not
// freed once unless it is in failed.
func (f *fakeSlots[T]) check(round string, slots []*cudamem.Slice[T], attempts int, failed ...*cudamem.Slice[T]) []string {
	var bad []string
	for i, s := range slots {
		if f.attempts[s] != attempts {
			bad = append(bad, fmt.Sprintf("%s: slot %d attempted %d times, want %d", round, i, f.attempts[s], attempts))
		}
		want := 1
		if slices.Contains(failed, s) {
			want = 0
		}
		if f.frees[s] != want {
			bad = append(bad, fmt.Sprintf("%s: slot %d freed %d times, want %d", round, i, f.frees[s], want))
		}
	}
	return bad
}

// wantJoined reports each of want that err does not wrap.
func wantJoined(round string, err error, want ...error) []string {
	var bad []string
	for _, e := range want {
		if !errors.Is(err, e) {
			bad = append(bad, fmt.Sprintf("%s: error %v does not wrap %v", round, err, e))
		}
	}
	return bad
}

var errFakeVec = errors.New("fake vector free failed")

// scratchReleaseFailures returns how release departs from: with three
// buffers failing, every buffer attempted once and the three errors
// joined with their identity; a retry that frees the three and reports
// nothing; and a repeat that frees nothing.
func scratchReleaseFailures(release func(*GPUScratchpad, func(*cudamem.Slice[float32]) error, func(*cudamem.Slice[int32]) error) error) []string {
	var sp GPUScratchpad
	f32s := []*cudamem.Slice[float32]{&sp.X, &sp.NormX, &sp.Q, &sp.K, &sp.V, &sp.AttnOut, &sp.Gate, &sp.Up,
		&sp.FFNAct, &sp.FFNOut, &sp.Logits}
	i32s := []*cudamem.Slice[int32]{&sp.TokenIn, &sp.TokenOut}
	f32 := newFakeSlots(map[*cudamem.Slice[float32]]error{&sp.Q: cudamem.ErrInUse, &sp.Logits: errFakeFree})
	i32 := newFakeSlots(map[*cudamem.Slice[int32]]error{&sp.TokenOut: errFakeVec})

	err := release(&sp, f32.free, i32.free)
	bad := wantJoined("failed free", err, cudamem.ErrInUse, errFakeFree, errFakeVec)
	bad = append(bad, f32.check("failed free", f32s, 1, &sp.Q, &sp.Logits)...)
	bad = append(bad, i32.check("failed free", i32s, 1, &sp.TokenOut)...)
	for i, round := range []string{"retry", "repeat"} {
		if err := release(&sp, f32.free, i32.free); err != nil {
			bad = append(bad, round+": "+err.Error())
		}
		bad = append(bad, f32.check(round, f32s, i+2)...)
		bad = append(bad, i32.check(round, i32s, i+2)...)
	}
	return bad
}

// TestScratchpadRelease checks release with failing fakes. That the real
// Free passes freeOwned, which keeps a failed buffer and zeroes a freed
// one, is checked by TestFreeDelegates and TestFreeOwned.
func TestScratchpadRelease(t *testing.T) {
	for _, f := range scratchReleaseFailures((*GPUScratchpad).release) {
		t.Error(f)
	}
	for _, m := range []struct {
		name    string
		release func(*GPUScratchpad, func(*cudamem.Slice[float32]) error, func(*cudamem.Slice[int32]) error) error
	}{
		{"S1 stops at the first failure", func(sp *GPUScratchpad, f32 func(*cudamem.Slice[float32]) error, i32 func(*cudamem.Slice[int32]) error) error {
			for _, s := range []*cudamem.Slice[float32]{&sp.X, &sp.NormX, &sp.Q, &sp.K, &sp.V, &sp.AttnOut, &sp.Gate, &sp.Up, &sp.FFNAct, &sp.FFNOut, &sp.Logits} {
				if err := f32(s); err != nil {
					return err
				}
			}
			return errors.Join(i32(&sp.TokenIn), i32(&sp.TokenOut))
		}},
		{"S2 error replaced", func(sp *GPUScratchpad, f32 func(*cudamem.Slice[float32]) error, i32 func(*cudamem.Slice[int32]) error) error {
			if err := sp.release(f32, i32); err != nil {
				return errors.New("free scratchpad failed")
			}
			return nil
		}},
		{"S3 error dropped", func(sp *GPUScratchpad, f32 func(*cudamem.Slice[float32]) error, i32 func(*cudamem.Slice[int32]) error) error {
			sp.release(f32, i32)
			return nil
		}},
		{"S4 free functions bypassed", func(sp *GPUScratchpad, f32 func(*cudamem.Slice[float32]) error, i32 func(*cudamem.Slice[int32]) error) error {
			return sp.Free()
		}},
	} {
		if len(scratchReleaseFailures(m.release)) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}

type weightsFreeFunc func(*GPUModelWeights, func(*cudamem.Slice[float32]) error) error

// weightsFreeFailures returns how free departs from, for a two-layer
// model: with the embedding table and two vectors failing, every matrix
// and vector attempted once, the failed embedding table kept, the other
// matrices zeroed, and the three errors joined with their identity; a
// retry that frees the three and reports nothing; and a repeat that frees
// nothing.
func weightsFreeFailures(free weightsFreeFunc) []string {
	var log []string
	w := fakeModel(&log, true, 1)
	w.Layers = append(w.Layers, fakeLayer(&log, "1"))
	vecs := []*cudamem.Slice[float32]{&w.Norm}
	var mats []*deviceWeight
	for i := range w.Layers {
		vecs = append(vecs, layerVectors(&w.Layers[i])...)
		mats = append(mats, layerMatrices(&w.Layers[i])...)
	}
	l := &w.Layers[1]
	vec := newFakeSlots(map[*cudamem.Slice[float32]]error{&w.Norm: cudamem.ErrInUse, &l.VBias: errFakeVec})
	frees := func(round string, want []string) []string {
		if !slices.Equal(log, want) {
			return []string{fmt.Sprintf("%s: matrix frees %q, want %q", round, log, want)}
		}
		return nil
	}

	err := free(w, vec.free)
	bad := wantJoined("failed free", err, errFakeFree, cudamem.ErrInUse, errFakeVec)
	want := append(fakeLayerFrees(""), fakeLayerFrees("1")...)
	bad = append(bad, frees("failed free", want)...)
	if w.EmbedTokens.buf == nil {
		bad = append(bad, "failed free: embedding table dropped, cannot retry")
	}
	for _, m := range mats {
		if m.buf != nil {
			bad = append(bad, "failed free: freed layer matrix not zeroed")
			break
		}
	}
	bad = append(bad, vec.check("failed free", vecs, 1, &w.Norm, &l.VBias)...)
	want = append(want, "embed")
	for i, round := range []string{"retry", "repeat"} {
		if err := free(w, vec.free); err != nil {
			bad = append(bad, round+": "+err.Error())
		}
		bad = append(bad, frees(round, want)...)
		bad = append(bad, vec.check(round, vecs, i+2)...)
	}
	if w.EmbedTokens.buf != nil || w.LMHead.buf != nil {
		bad = append(bad, "retry: weights not zeroed after free")
	}
	return bad
}

// TestGPUModelWeightsFreeVectors checks free with failing fakes. That the
// real Free passes freeOwned is checked by TestFreeDelegates.
func TestGPUModelWeightsFreeVectors(t *testing.T) {
	for _, f := range weightsFreeFailures((*GPUModelWeights).free) {
		t.Error(f)
	}
	for _, m := range []struct {
		name string
		free weightsFreeFunc
	}{
		{"W1 error dropped", func(w *GPUModelWeights, vec func(*cudamem.Slice[float32]) error) error {
			w.free(vec)
			return nil
		}},
		{"W2 error replaced", func(w *GPUModelWeights, vec func(*cudamem.Slice[float32]) error) error {
			if err := w.free(vec); err != nil {
				return errors.New("free weights failed")
			}
			return nil
		}},
		{"W3 stops at the first vector failure", func(w *GPUModelWeights, vec func(*cudamem.Slice[float32]) error) error {
			var errs []error
			for _, m := range w.matrices() {
				errs = append(errs, m.release())
			}
			for _, v := range w.vectors() {
				if err := vec(v); err != nil {
					return errors.Join(append(errs, err)...)
				}
			}
			return errors.Join(errs...)
		}},
		{"W4 vectors skipped", func(w *GPUModelWeights, vec func(*cudamem.Slice[float32]) error) error {
			var errs []error
			for _, m := range w.matrices() {
				errs = append(errs, m.release())
			}
			return errors.Join(errs...)
		}},
		{"W5 vector free bypassed", func(w *GPUModelWeights, vec func(*cudamem.Slice[float32]) error) error {
			return w.Free()
		}},
		{"W6 first layer for every layer", func(w *GPUModelWeights, vec func(*cudamem.Slice[float32]) error) error {
			errs := []error{w.EmbedTokens.release(), w.LMHead.release()}
			for range w.Layers {
				for _, m := range layerMatrices(&w.Layers[0]) {
					errs = append(errs, m.release())
				}
			}
			errs = append(errs, vec(&w.Norm))
			for range w.Layers {
				for _, v := range layerVectors(&w.Layers[0]) {
					errs = append(errs, vec(v))
				}
			}
			return errors.Join(errs...)
		}},
	} {
		if len(weightsFreeFailures(m.free)) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}

// freeDelegations lists each public Free whose behaviour is tested through
// a private method with fake free functions. Free must do nothing but
// return that method's result for the real functions.
var freeDelegations = []struct {
	recv, method string
	args         []string
}{
	{"GPUScratchpad", "release", []string{"freeOwned[cudamem.Slice[float32]]", "freeOwned[cudamem.Slice[int32]]"}},
	{"GPUModelWeights", "free", []string{"freeOwned[cudamem.Slice[float32]]"}},
}

// findMethod returns the declaration of the method (*recv).name in f, or nil.
func findMethod(f *ast.File, recv, name string) *ast.FuncDecl {
	for _, decl := range f.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Name.Name != name || fd.Recv == nil || len(fd.Recv.List) != 1 {
			continue
		}
		if star, ok := fd.Recv.List[0].Type.(*ast.StarExpr); ok && types.ExprString(star.X) == recv {
			return fd
		}
	}
	return nil
}

// freeDelegationFailures reports each Free in freeDelegations that src does
// not declare with the body "return r.method(args...)", r being its
// receiver, and each private method that assigns to anything but errs.
// Zeroing a buffer is the job of the free function the method is passed,
// which keeps a buffer whose free fails; the fakes cannot see a zero
// cudamem.Slice zeroed again.
func freeDelegationFailures(src string) ([]string, error) {
	f, err := parser.ParseFile(token.NewFileSet(), "gpu_model.go", src, 0)
	if err != nil {
		return nil, err
	}
	var bad []string
	for _, d := range freeDelegations {
		fn := findMethod(f, d.recv, "Free")
		if fn == nil || len(fn.Recv.List[0].Names) != 1 {
			bad = append(bad, fmt.Sprintf("(*%s).Free: not declared with a named receiver", d.recv))
			continue
		}
		want := fmt.Sprintf("%s.%s(%s)", fn.Recv.List[0].Names[0].Name, d.method, strings.Join(d.args, ", "))
		got := ""
		if body := fn.Body; body != nil && len(body.List) == 1 {
			if ret, ok := body.List[0].(*ast.ReturnStmt); ok && len(ret.Results) == 1 {
				got = types.ExprString(ret.Results[0])
			}
		}
		if got != want {
			bad = append(bad, fmt.Sprintf("(*%s).Free: body is not \"return %s\"", d.recv, want))
		}

		priv := findMethod(f, d.recv, d.method)
		if priv == nil || priv.Body == nil {
			bad = append(bad, fmt.Sprintf("(*%s).%s: not declared", d.recv, d.method))
			continue
		}
		ast.Inspect(priv.Body, func(n ast.Node) bool {
			var lhs []ast.Expr
			switch n := n.(type) {
			case *ast.AssignStmt:
				lhs = n.Lhs
			case *ast.IncDecStmt:
				lhs = []ast.Expr{n.X}
			}
			for _, x := range lhs {
				if id, ok := x.(*ast.Ident); !ok || id.Name != "errs" {
					bad = append(bad, fmt.Sprintf("(*%s).%s: assigns to %s", d.recv, d.method, types.ExprString(x)))
				}
			}
			return true
		})
	}
	return bad, nil
}

// TestFreeDelegates checks that each public Free is only a call of the
// private method tested with fakes, passing the real free functions and
// returning its error, and that the private method leaves the buffers to
// those functions. Must-fail controls edit the real source.
func TestFreeDelegates(t *testing.T) {
	b, err := os.ReadFile("gpu_model.go")
	if err != nil {
		t.Fatal(err)
	}
	src := string(b)
	bad, err := freeDelegationFailures(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range bad {
		t.Error(f)
	}
	const (
		sp  = "return sp.release(freeOwned[cudamem.Slice[float32]], freeOwned[cudamem.Slice[int32]])"
		w   = "return w.free(freeOwned[cudamem.Slice[float32]])"
		vec = "errs = append(errs, vec(v))"
	)
	for _, m := range []struct{ name, old, new string }{
		{"D1 scratchpad error dropped", sp, "_ = sp.release(freeOwned[cudamem.Slice[float32]], freeOwned[cudamem.Slice[int32]])\n\treturn nil"},
		{"D2 weights error dropped", w, "w.free(freeOwned[cudamem.Slice[float32]])\n\treturn nil"},
		{"D3 weights error wrapped", w, `return fmt.Errorf("free weights: %w", w.free(freeOwned[cudamem.Slice[float32]]))`},
		{"D4 vectors not zeroed", w, "return w.free(func(s *cudamem.Slice[float32]) error { return s.Free() })"},
		{"D5 scratchpad functions swapped", sp, "return sp.release(freeOwned[cudamem.Slice[float32]], func(s *cudamem.Slice[int32]) error { return s.Free() })"},
		{"D6 Free renamed", "func (w *GPUModelWeights) Free()", "func (w *GPUModelWeights) FreeAll()"},
		{"D7 failed vector zeroed", vec, vec + "\n\t\t*v = cudamem.Slice[float32]{}"},
		{"D8 private method renamed", "func (w *GPUModelWeights) free(", "func (w *GPUModelWeights) freeAll("},
	} {
		if !strings.Contains(src, m.old) {
			t.Fatalf("must-fail control %s: %q not in gpu_model.go", m.name, m.old)
		}
		bad, err := freeDelegationFailures(strings.Replace(src, m.old, m.new, 1))
		if err != nil {
			t.Fatalf("must-fail control %s: %v", m.name, err)
		}
		if len(bad) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}

// fakeOwned is an owner whose first *fails frees fail. Its zero value frees
// as a no-op, as a zero cudamem.Slice does.
type fakeOwned struct {
	name  string
	fails *int
	log   *[]string
}

func (f fakeOwned) Free() error {
	if f.log == nil {
		return nil
	}
	if *f.fails > 0 {
		*f.fails--
		return errFakeFree
	}
	*f.log = append(*f.log, f.name)
	return nil
}

// freeOwnedFailures returns how free departs from: a failed free keeps the
// owner and returns the free's error; a retry frees it once and zeroes it;
// a repeat after that does nothing.
func freeOwnedFailures(free func(*fakeOwned) error) []string {
	var bad []string
	var log []string
	fails := 1
	s := fakeOwned{name: "buf", fails: &fails, log: &log}
	if err := free(&s); err != errFakeFree {
		bad = append(bad, fmt.Sprintf("failed free returned %v, want the free's error", err))
	}
	if s.name != "buf" {
		bad = append(bad, "failed free dropped the owner")
	}
	if err := free(&s); err != nil {
		bad = append(bad, "retry: "+err.Error())
	}
	if s.name != "" {
		bad = append(bad, "successful free left the owner set")
	}
	if err := free(&s); err != nil {
		bad = append(bad, "repeat: "+err.Error())
	}
	if !slices.Equal(log, []string{"buf"}) {
		bad = append(bad, fmt.Sprintf("frees %q, want [buf]", log))
	}
	return bad
}

func TestFreeOwned(t *testing.T) {
	for _, f := range freeOwnedFailures(freeOwned[fakeOwned]) {
		t.Error(f)
	}
	for _, m := range []struct {
		name string
		free func(*fakeOwned) error
	}{
		{"R1 zeroed before the free is checked", func(s *fakeOwned) error {
			err := s.Free()
			*s = fakeOwned{}
			return err
		}},
		{"R2 never zeroed", func(s *fakeOwned) error { return s.Free() }},
		{"R3 error dropped", func(s *fakeOwned) error {
			if s.Free() == nil {
				*s = fakeOwned{}
			}
			return nil
		}},
		{"R4 error wrapped", func(s *fakeOwned) error {
			if err := s.Free(); err != nil {
				return fmt.Errorf("free: %w", err)
			}
			*s = fakeOwned{}
			return nil
		}},
	} {
		if len(freeOwnedFailures(m.free)) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}

// fakeAlloc is an allocation for uploadAs: ptr 0 is a zero slice. It
// records copies, with their element type and values, in a shared log.
type fakeAlloc[T any] struct {
	ptr     driver.DevicePtr
	copyErr error
	log     *[]string
}

func (a fakeAlloc[T]) Ptr() driver.DevicePtr { return a.ptr }
func (a fakeAlloc[T]) Free() error           { return nil }

func (a fakeAlloc[T]) CopyFromHost(src []T) error {
	*a.log = append(*a.log, fmt.Sprintf("copy %T %v", src, src))
	return a.copyErr
}

// fakeAllocator returns an allocator that logs its name and the length
// asked for, then returns a and err.
func fakeAllocator[T any](name string, a fakeAlloc[T], err error) func(int) (fakeAlloc[T], error) {
	return func(n int) (fakeAlloc[T], error) {
		*a.log = append(*a.log, fmt.Sprintf("%s %d", name, n))
		return a, err
	}
}

var (
	errFakeAlloc = errors.New("fake alloc failed")
	errFakeCopy  = errors.New("fake copy failed")
)

type uploadFunc func(dtype weightDType, host []float32, alloc func(int) (fakeAlloc[float32], error)) (deviceWeight, error)

// uploadFailures returns how upload departs from: allocating len(host),
// keeping any nonzero allocation with the allocation or copy error, a zero
// weight for a zero allocation, no copy after a failed allocation, and
// errors kept as is.
func uploadFailures(upload uploadFunc) []string {
	var bad []string
	host := []float32{1, 2}
	allocated := []string{"alloc 2"}
	copied := []string{"alloc 2", "copy []float32 [1 2]"}
	for _, tt := range []struct {
		name     string
		ptr      driver.DevicePtr
		allocErr error
		copyErr  error
		owned    bool
		log      []string
	}{
		{"success", 7, nil, nil, true, copied},
		{"copy fails", 7, nil, errFakeCopy, true, copied},
		{"alloc fails, nonzero slice", 7, errFakeAlloc, nil, true, allocated},
		{"alloc fails, zero slice", 0, errFakeAlloc, nil, false, allocated},
	} {
		var log []string
		a := fakeAlloc[float32]{ptr: tt.ptr, copyErr: tt.copyErr, log: &log}
		w, err := upload(weightBF16, host, fakeAllocator("alloc", a, tt.allocErr))
		want := tt.allocErr
		if want == nil {
			want = tt.copyErr
		}
		if err != want {
			bad = append(bad, fmt.Sprintf("%s: error %v, want %v", tt.name, err, want))
		}
		if owned := w.buf != nil; owned != tt.owned {
			bad = append(bad, fmt.Sprintf("%s: owns %v, want %v", tt.name, owned, tt.owned))
		}
		if tt.owned && (w.Ptr() != tt.ptr || w.dtype != weightBF16) {
			bad = append(bad, fmt.Sprintf("%s: weight %v, want the allocation as bf16", tt.name, w))
		}
		if !slices.Equal(log, tt.log) {
			bad = append(bad, fmt.Sprintf("%s: log %q, want %q", tt.name, log, tt.log))
		}
	}
	return bad
}

func TestUploadAs(t *testing.T) {
	for _, f := range uploadFailures(uploadAs[float32, fakeAlloc[float32]]) {
		t.Error(f)
	}
	for _, m := range []struct {
		name   string
		upload uploadFunc
	}{
		{"A1 allocation dropped with its error", func(dtype weightDType, host []float32, alloc func(int) (fakeAlloc[float32], error)) (deviceWeight, error) {
			s, err := alloc(len(host))
			if err != nil {
				return deviceWeight{}, err
			}
			return deviceWeight{buf: s, dtype: dtype}, s.CopyFromHost(host)
		}},
		{"A2 zero allocation kept", func(dtype weightDType, host []float32, alloc func(int) (fakeAlloc[float32], error)) (deviceWeight, error) {
			s, err := alloc(len(host))
			if err != nil {
				return deviceWeight{buf: s, dtype: dtype}, err
			}
			return deviceWeight{buf: s, dtype: dtype}, s.CopyFromHost(host)
		}},
		{"A3 allocation error wrapped", func(dtype weightDType, host []float32, alloc func(int) (fakeAlloc[float32], error)) (deviceWeight, error) {
			w, err := uploadAs(dtype, host, alloc)
			if err != nil {
				err = fmt.Errorf("upload: %w", err)
			}
			return w, err
		}},
		{"A4 wrong length allocated", func(dtype weightDType, host []float32, alloc func(int) (fakeAlloc[float32], error)) (deviceWeight, error) {
			return uploadAs(dtype, host, func(n int) (fakeAlloc[float32], error) { return alloc(n + 1) })
		}},
	} {
		if len(uploadFailures(m.upload)) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}

type uploadWeightFunc func(host []float32, dtype weightDType, alloc32 func(int) (fakeAlloc[float32], error), alloc16 func(int) (fakeAlloc[uint16], error)) (deviceWeight, error)

// uploadWeightFailures returns how upload departs from uploadWeight: for
// each dtype, calling that dtype's allocator, and only it, with len(host),
// copying f32 values as they are and bf16 values rounded by roundBF16,
// labelling the weight with the dtype, keeping a nonzero allocation with its
// allocation or copy error, and refusing an unknown dtype without
// allocating.
func uploadWeightFailures(upload uploadWeightFunc) []string {
	var bad []string
	var host []float32
	for _, b := range []uint32{0x3f800000, 0x3f818000, 0x3f808000, 0x3f808001, 0x3f807fff, 0x7fffffff} {
		host = append(host, math.Float32frombits(b))
	}
	// Truncation would give 0x3f80 0x3f81 0x3f80 0x3f80 0x3f80 0x7fff;
	// rounding half up, 0x3f80 0x3f82 0x3f81 0x3f81 0x3f80 0x8000.
	bf16 := []uint16{0x3f80, 0x3f82, 0x3f80, 0x3f81, 0x3f80, 0x7fff}
	for _, tt := range []struct {
		name     string
		dtype    weightDType
		ptr      driver.DevicePtr
		allocErr error
		copyErr  error
		owned    bool
		log      []string
	}{
		{"f32", weightF32, 7, nil, nil, true, []string{"alloc32 6", fmt.Sprintf("copy []float32 %v", host)}},
		{"f32 copy fails", weightF32, 7, nil, errFakeCopy, true, []string{"alloc32 6", fmt.Sprintf("copy []float32 %v", host)}},
		{"f32 alloc fails, nonzero slice", weightF32, 7, errFakeAlloc, nil, true, []string{"alloc32 6"}},
		{"f32 alloc fails, zero slice", weightF32, 0, errFakeAlloc, nil, false, []string{"alloc32 6"}},
		{"bf16", weightBF16, 7, nil, nil, true, []string{"alloc16 6", fmt.Sprintf("copy []uint16 %v", bf16)}},
		{"bf16 copy fails", weightBF16, 7, nil, errFakeCopy, true, []string{"alloc16 6", fmt.Sprintf("copy []uint16 %v", bf16)}},
		{"bf16 alloc fails, nonzero slice", weightBF16, 7, errFakeAlloc, nil, true, []string{"alloc16 6"}},
		{"bf16 alloc fails, zero slice", weightBF16, 0, errFakeAlloc, nil, false, []string{"alloc16 6"}},
		{"unknown dtype", "f16", 7, nil, nil, false, nil},
	} {
		var log []string
		a32 := fakeAlloc[float32]{ptr: tt.ptr, copyErr: tt.copyErr, log: &log}
		a16 := fakeAlloc[uint16]{ptr: tt.ptr, copyErr: tt.copyErr, log: &log}
		w, err := upload(host, tt.dtype, fakeAllocator("alloc32", a32, tt.allocErr), fakeAllocator("alloc16", a16, tt.allocErr))
		want := tt.allocErr
		if want == nil {
			want = tt.copyErr
		}
		switch {
		case tt.dtype != weightF32 && tt.dtype != weightBF16:
			if err == nil {
				bad = append(bad, fmt.Sprintf("%s: no error", tt.name))
			}
		case err != want:
			bad = append(bad, fmt.Sprintf("%s: error %v, want %v", tt.name, err, want))
		}
		if owned := w.buf != nil; owned != tt.owned {
			bad = append(bad, fmt.Sprintf("%s: owns %v, want %v", tt.name, owned, tt.owned))
		}
		if tt.owned && (w.Ptr() != tt.ptr || w.dtype != tt.dtype || w.borrowed) {
			bad = append(bad, fmt.Sprintf("%s: weight %+v, want the allocation as %s", tt.name, w, tt.dtype))
		}
		if !slices.Equal(log, tt.log) {
			bad = append(bad, fmt.Sprintf("%s: log %q, want %q", tt.name, log, tt.log))
		}
	}
	return bad
}

func TestUploadWeightWith(t *testing.T) {
	for _, f := range uploadWeightFailures(uploadWeightWith[fakeAlloc[float32], fakeAlloc[uint16]]) {
		t.Error(f)
	}
	for _, m := range []struct {
		name   string
		upload uploadWeightFunc
	}{
		{"W1 bf16 allocation dropped with its error", func(host []float32, dtype weightDType, alloc32 func(int) (fakeAlloc[float32], error), alloc16 func(int) (fakeAlloc[uint16], error)) (deviceWeight, error) {
			w, err := uploadWeightWith(host, dtype, alloc32, alloc16)
			if err != nil && dtype == weightBF16 {
				return deviceWeight{}, err
			}
			return w, err
		}},
		{"W2 bf16 truncated, not rounded", func(host []float32, dtype weightDType, alloc32 func(int) (fakeAlloc[float32], error), alloc16 func(int) (fakeAlloc[uint16], error)) (deviceWeight, error) {
			if dtype != weightBF16 {
				return uploadWeightWith(host, dtype, alloc32, alloc16)
			}
			b := make([]uint16, len(host))
			for i, f := range host {
				b[i] = uint16(math.Float32bits(f) >> 16)
			}
			return uploadAs(dtype, b, alloc16)
		}},
		{"W3 f32 labelled bf16", func(host []float32, dtype weightDType, alloc32 func(int) (fakeAlloc[float32], error), alloc16 func(int) (fakeAlloc[uint16], error)) (deviceWeight, error) {
			if dtype == weightF32 {
				return uploadAs(weightBF16, host, alloc32)
			}
			return uploadWeightWith(host, dtype, alloc32, alloc16)
		}},
		{"W4 bf16 rounded half up", func(host []float32, dtype weightDType, alloc32 func(int) (fakeAlloc[float32], error), alloc16 func(int) (fakeAlloc[uint16], error)) (deviceWeight, error) {
			if dtype != weightBF16 {
				return uploadWeightWith(host, dtype, alloc32, alloc16)
			}
			b := make([]uint16, len(host))
			for i, f := range host {
				b[i] = uint16((math.Float32bits(f) + 0x8000) >> 16)
			}
			return uploadAs(dtype, b, alloc16)
		}},
	} {
		if len(uploadWeightFailures(m.upload)) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}
