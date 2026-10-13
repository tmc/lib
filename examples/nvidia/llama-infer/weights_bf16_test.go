package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
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
	buf := func(name string) deviceWeight {
		return deviceWeight{buf: &fakeBuffer{name: name, log: log}, dtype: weightBF16}
	}
	w := &GPUModelWeights{DType: weightBF16, Layers: make([]GPULayerWeights, 1)}
	w.EmbedTokens = deviceWeight{buf: &fakeBuffer{name: "embed", fails: embedFails, err: errFakeFree, log: log}, dtype: weightBF16}
	if tied {
		w.LMHead = w.EmbedTokens.borrow()
	} else {
		w.LMHead = buf("head")
	}
	l := &w.Layers[0]
	l.WQ, l.WK, l.WV, l.WO = buf("wq"), buf("wk"), buf("wv"), buf("wo")
	l.WGate, l.WUp, l.WDown = buf("gate"), buf("up"), buf("down")
	return w
}

var fakeLayerFrees = []string{"wq", "wk", "wv", "wo", "gate", "up", "down"}

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
		for _, s := range fakeLayerFrees {
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

// scratchReleaseFailures returns how release, given free functions that fail
// for two buffers, departs from attempting every buffer once and reporting
// both failures with their identity.
func scratchReleaseFailures(release func(*GPUScratchpad, func(*cudamem.Slice[float32]) error, func(*cudamem.Slice[int32]) error) error) []string {
	var sp GPUScratchpad
	calls := make(map[any]int)
	f32 := func(s *cudamem.Slice[float32]) error {
		calls[s]++
		switch s {
		case &sp.Q:
			return cudamem.ErrInUse
		case &sp.Logits:
			return errFakeFree
		}
		return nil
	}
	i32 := func(s *cudamem.Slice[int32]) error {
		calls[s]++
		return nil
	}
	err := release(&sp, f32, i32)
	var bad []string
	for _, e := range []error{cudamem.ErrInUse, errFakeFree} {
		if !errors.Is(err, e) {
			bad = append(bad, fmt.Sprintf("error %v does not wrap %v", err, e))
		}
	}
	for _, p := range []any{&sp.X, &sp.NormX, &sp.Q, &sp.K, &sp.V, &sp.AttnOut, &sp.Gate, &sp.Up,
		&sp.FFNAct, &sp.FFNOut, &sp.Logits, &sp.TokenIn, &sp.TokenOut} {
		if calls[p] != 1 {
			bad = append(bad, fmt.Sprintf("buffer %p freed %d times, want 1", p, calls[p]))
		}
	}
	return bad
}

// TestScratchpadRelease checks every buffer is released once and failures
// keep their identity. Keeping a failed buffer is freeOwned's job, checked
// by TestFreeOwned.
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
	} {
		if len(scratchReleaseFailures(m.release)) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}

// ownerConstructors can return an owner together with an error.
var ownerConstructors = map[string]bool{"uploadModel": true, "UploadModelToGPU": true, "NewGPUScratchpad": true}

// ownerCustody checks every call in src to an owner constructor: the owner
// must be named, and the next statement must be "if owner != nil" with a
// body that calls owner.Free, so that an owner returned with an error is
// released before the error is examined.
func ownerCustody(name, src string) (calls int, bad []string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return 0, nil, err
	}
	ast.Inspect(f, func(n ast.Node) bool {
		block, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			assign, ok := stmt.(*ast.AssignStmt)
			if !ok || len(assign.Rhs) != 1 {
				continue
			}
			call, ok := assign.Rhs[0].(*ast.CallExpr)
			if !ok {
				continue
			}
			fn, ok := call.Fun.(*ast.Ident)
			if !ok || !ownerConstructors[fn.Name] {
				continue
			}
			calls++
			pos := fset.Position(assign.Pos())
			owner, ok := assign.Lhs[0].(*ast.Ident)
			if !ok || owner.Name == "_" {
				bad = append(bad, fmt.Sprintf("%v: %s owner discarded", pos, fn.Name))
				continue
			}
			if i+1 >= len(block.List) || !freesIfNonNil(block.List[i+1], owner.Name) {
				bad = append(bad, fmt.Sprintf("%v: %s owner %s not freed before the error is examined", pos, fn.Name, owner.Name))
			}
		}
		return true
	})
	return calls, bad, nil
}

// freesIfNonNil reports whether stmt is "if owner != nil { ... }" and its
// body calls owner.Free.
func freesIfNonNil(stmt ast.Stmt, owner string) bool {
	is, ok := stmt.(*ast.IfStmt)
	if !ok || is.Init != nil || is.Else != nil {
		return false
	}
	cond, ok := is.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.NEQ {
		return false
	}
	x, xok := cond.X.(*ast.Ident)
	y, yok := cond.Y.(*ast.Ident)
	if !xok || !yok || x.Name != owner || y.Name != "nil" {
		return false
	}
	frees := false
	ast.Inspect(is.Body, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok && sel.Sel.Name == "Free" {
			if id, ok := sel.X.(*ast.Ident); ok && id.Name == owner {
				frees = true
			}
		}
		return !frees
	})
	return frees
}

// TestCallersRegisterReturnedOwners checks that the callers this package
// owns release an owner a constructor returns with its error. A constructor
// that keeps unfreed buffers for retry helps only if the caller keeps them.
func TestCallersRegisterReturnedOwners(t *testing.T) {
	for _, tt := range []struct {
		file  string
		calls int
	}{
		{"main.go", 2},
		{"bf16_parity_cuda_test.go", 2},
		{"fused_parity_cuda_test.go", 2},
	} {
		src, err := os.ReadFile(tt.file)
		if err != nil {
			t.Fatal(err)
		}
		calls, bad, err := ownerCustody(tt.file, string(src))
		if err != nil {
			t.Fatal(err)
		}
		if calls != tt.calls {
			t.Errorf("%s: %d constructor calls, want %d", tt.file, calls, tt.calls)
		}
		for _, b := range bad {
			t.Error(b)
		}
	}
}

// TestOwnerCustodyControls checks ownerCustody on small sources: the
// accepted form, and must-fail forms that each mishandle the owner.
func TestOwnerCustodyControls(t *testing.T) {
	const ok = `
		sp, e := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			defer func() { _ = sp.Free() }()
		}
		if e != nil {
			return e
		}`
	for _, tt := range []struct {
		name string
		body string
		bad  bool
	}{
		{"positive control: owner freed first, any error name", ok, false},
		{"error examined first", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if err != nil {
			return err
		}
		defer sp.Free()`, true},
		{"guard without a free", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			_ = sp
		}`, true},
		{"frees another owner", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			defer gw.Free()
		}`, true},
		{"owner discarded", `
		_, err = UploadModelToGPU(dev, w, cfg)`, true},
		{"assignment, not declaration", `
		gw, err = uploadModel(dev, w, cfg, dtype)
		return err`, true},
	} {
		src := "package p\nfunc f() error {\n" + tt.body + "\nreturn nil\n}\n"
		calls, bad, err := ownerCustody("control.go", src)
		if err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if calls != 1 {
			t.Errorf("%s: %d calls found, want 1", tt.name, calls)
		}
		if got := len(bad) > 0; got != tt.bad {
			t.Errorf("%s: flagged %v, want %v (%q)", tt.name, got, tt.bad, bad)
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
// records copies in a shared log.
type fakeAlloc struct {
	ptr     driver.DevicePtr
	copyErr error
	log     *[]string
}

func (a fakeAlloc) Ptr() driver.DevicePtr { return a.ptr }
func (a fakeAlloc) Free() error           { return nil }

func (a fakeAlloc) CopyFromHost(src []float32) error {
	*a.log = append(*a.log, fmt.Sprintf("copy %d", len(src)))
	return a.copyErr
}

var errFakeAlloc = errors.New("fake alloc failed")

type uploadFunc func(dtype weightDType, host []float32, alloc func(int) (fakeAlloc, error)) (deviceWeight, error)

// uploadFailures returns how upload departs from: keeping any nonzero
// allocation with the allocation or copy error, a zero weight for a zero
// allocation, no copy after a failed allocation, and errors kept as is.
func uploadFailures(upload uploadFunc) []string {
	var bad []string
	for _, tt := range []struct {
		name     string
		ptr      driver.DevicePtr
		allocErr error
		copyErr  error
		owned    bool
		copies   int
	}{
		{"success", 7, nil, nil, true, 1},
		{"copy fails", 7, nil, errFakeFree, true, 1},
		{"alloc fails, nonzero slice", 7, errFakeAlloc, nil, true, 0},
		{"alloc fails, zero slice", 0, errFakeAlloc, nil, false, 0},
	} {
		var log []string
		a := fakeAlloc{ptr: tt.ptr, copyErr: tt.copyErr, log: &log}
		w, err := upload(weightBF16, []float32{1, 2}, func(n int) (fakeAlloc, error) { return a, tt.allocErr })
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
		if len(log) != tt.copies {
			bad = append(bad, fmt.Sprintf("%s: %d copies, want %d", tt.name, len(log), tt.copies))
		}
	}
	return bad
}

func TestUploadAs(t *testing.T) {
	for _, f := range uploadFailures(uploadAs[float32, fakeAlloc]) {
		t.Error(f)
	}
	for _, m := range []struct {
		name   string
		upload uploadFunc
	}{
		{"A1 allocation dropped with its error", func(dtype weightDType, host []float32, alloc func(int) (fakeAlloc, error)) (deviceWeight, error) {
			s, err := alloc(len(host))
			if err != nil {
				return deviceWeight{}, err
			}
			return deviceWeight{buf: s, dtype: dtype}, s.CopyFromHost(host)
		}},
		{"A2 zero allocation kept", func(dtype weightDType, host []float32, alloc func(int) (fakeAlloc, error)) (deviceWeight, error) {
			s, err := alloc(len(host))
			if err != nil {
				return deviceWeight{buf: s, dtype: dtype}, err
			}
			return deviceWeight{buf: s, dtype: dtype}, s.CopyFromHost(host)
		}},
		{"A3 allocation error wrapped", func(dtype weightDType, host []float32, alloc func(int) (fakeAlloc, error)) (deviceWeight, error) {
			w, err := uploadAs(dtype, host, alloc)
			if err != nil {
				err = fmt.Errorf("upload: %w", err)
			}
			return w, err
		}},
	} {
		if len(uploadFailures(m.upload)) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}
