package main

import (
	"errors"
	"fmt"
	"hash/fnv"
	"slices"
	"testing"

	"github.com/tmc/lib/lm/speculative"
)

const specVocab = 7

// fakeKV is a host model of the KV cache and a forward pass. Feeding tok at
// pos writes it to the cache and predicts from every entry in [0, pos], as
// attention reads, so a stale entry in that prefix changes the prediction.
type fakeKV struct {
	kv    []int32
	calls int
}

var errOutsideCache = errors.New("outside the fake kv cache")

func (f *fakeKV) forward(tok int32, pos int) (int32, error) {
	f.calls++
	if pos < 0 || pos >= len(f.kv) {
		return 0, errOutsideCache
	}
	f.kv[pos] = tok
	h := fnv.New32a()
	for _, t := range f.kv[:pos+1] {
		h.Write([]byte{byte(t)})
	}
	return int32(h.Sum32() % specVocab), nil
}

// greedy decodes without drafts from pending token next at pos, as main's
// plain path does, until steps tokens are emitted or the cache is full. It
// returns the emitted tokens and the final pending token.
func greedy(maxSeq int, next int32, pos, steps int) ([]int32, int32) {
	f := &fakeKV{kv: make([]int32, maxSeq)}
	var out []int32
	for len(out) < steps && pos < maxSeq {
		out = append(out, next)
		next, _ = f.forward(next, pos)
		pos++
	}
	return out, next
}

type stepFunc func(forward func(int32, int) (int32, error), next int32, drafts []int32, pos, maxSeq, budget int) ([]int32, int32, int, error)

// specDecode decodes as main's loop does, drafting with draft (given the
// number of tokens emitted so far) and stepping with step.
func specDecode(step stepFunc, draft func(emitted int) []int32, maxSeq int, next int32, pos, steps int) ([]int32, error) {
	f := &fakeKV{kv: make([]int32, maxSeq)}
	var out []int32
	for len(out) < steps && pos < maxSeq {
		drafts := draft(len(out))
		if len(drafts) == 0 {
			out = append(out, next)
			var err error
			if next, err = f.forward(next, pos); err != nil {
				return out, err
			}
			pos++
			continue
		}
		emit, n, p, err := step(f.forward, next, drafts, pos, maxSeq, steps-len(out))
		if err != nil {
			return out, err
		}
		out = append(out, emit...)
		next, pos = n, p
	}
	return out, nil
}

// specFailures returns how step departs from plain greedy decoding over
// drafters that are right, wrong, wrong first or last, and empty, at
// positions near the end of the cache and with small output budgets.
func specFailures(step stepFunc) []string {
	return specFailuresAt(step, []int{0, specMaxSeq - 4, specMaxSeq - 3, specMaxSeq - 2, specMaxSeq - 1}, []int{1, 2, 3, 20})
}

const specMaxSeq = 12

// specFailuresAt is specFailures for the given starting positions and
// output budgets.
func specFailuresAt(step stepFunc, p0s, budgets []int) []string {
	const maxSeq, draftLen = specMaxSeq, 3
	var bad []string
	for _, p0 := range p0s {
		for _, steps := range budgets {
			want, _ := greedy(maxSeq, 3, p0, steps)
			// ref holds every token greedy decoding would emit, then the
			// pending token, so a right drafter can read ahead of want.
			ref, last := greedy(maxSeq, 3, p0, maxSeq)
			ref = append(ref, last)
			right := func(e int) []int32 {
				if e+1 >= len(ref) {
					return nil
				}
				return slices.Clone(ref[e+1 : min(e+1+draftLen, len(ref))])
			}
			corrupt := func(at func(n int) int) func(int) []int32 {
				return func(e int) []int32 {
					d := right(e)
					if len(d) > 0 {
						i := at(len(d))
						d[i] = (d[i] + 1) % specVocab
					}
					return d
				}
			}
			for _, dr := range []struct {
				name  string
				draft func(int) []int32
			}{
				{"right", right},
				{"wrong first", corrupt(func(int) int { return 0 })},
				{"wrong last", corrupt(func(n int) int { return n - 1 })},
				{"all wrong", func(e int) []int32 {
					d := right(e)
					for i := range d {
						d[i] = (d[i] + 1) % specVocab
					}
					return d
				}},
				{"empty", func(int) []int32 { return nil }},
			} {
				got, err := specDecode(step, dr.draft, maxSeq, 3, p0, steps)
				if err != nil || !slices.Equal(got, want) {
					bad = append(bad, fmt.Sprintf("pos %d steps %d drafter %s: got %v, %v; want %v", p0, steps, dr.name, got, err, want))
				}
			}
		}
	}
	return bad
}

func TestSpeculativeStepMatchesGreedy(t *testing.T) {
	for _, f := range specFailures(speculativeStep) {
		t.Error(f)
	}
}

// TestSpeculativeStepControls checks that specFailures detects each defect
// the previous speculative loop had, as variants of speculativeStep.
func TestSpeculativeStepControls(t *testing.T) {
	variant := func(capSeq, capBudget, emitNext bool, newPos func(pos, k, accepted int) int, newNext func(targets []int32, k, accepted int) int32) stepFunc {
		return func(forward func(int32, int) (int32, error), next int32, drafts []int32, pos, maxSeq, budget int) ([]int32, int32, int, error) {
			k := len(drafts)
			if capSeq {
				k = min(k, maxSeq-pos-1)
			}
			if capBudget {
				k = min(k, budget-1)
			}
			var targets []int32
			for i, tok := range append([]int32{next}, drafts[:k]...) {
				p, err := forward(tok, pos+i)
				if err != nil {
					return nil, next, pos, err
				}
				targets = append(targets, p)
			}
			a := 0
			for a < k && drafts[a] == targets[a] {
				a++
			}
			var emit []int32
			if emitNext {
				emit = append(emit, next)
			}
			emit = append(emit, drafts[:a]...)
			return emit, newNext(targets, k, a), newPos(pos, k, a), nil
		}
	}
	rollback := func(pos, k, a int) int { return pos + 1 + a }
	pending := func(t []int32, k, a int) int32 { return t[a] }
	// The positive control: the variant harness, with every fix, passes.
	if bad := specFailures(variant(true, true, true, rollback, pending)); len(bad) > 0 {
		t.Errorf("positive control failed: %s", bad[0])
	}
	for _, m := range []struct {
		name string
		step stepFunc
	}{
		{"M1 no rollback", variant(true, true, true, func(pos, k, a int) int { return pos + 1 + k }, pending)},
		{"M2 pending token not emitted", variant(true, true, false, rollback, pending)},
		{"M3 no kv capacity cap", variant(false, true, true, rollback, pending)},
		{"M4 no budget cap", variant(true, false, true, rollback, pending)},
		{"M5 pending token after the drafts", variant(true, true, true, rollback, func(t []int32, k, a int) int32 { return t[k] })},
	} {
		if len(specFailures(m.step)) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
	// M1 must also fail where it cannot run out of cache: from position 0,
	// three output tokens feed at most 3+2*3 positions of 12. There only the
	// fake's dependence on stale entries in the prefix can detect it.
	if len(specFailuresAt(variant(true, true, true, func(pos, k, a int) int { return pos + 1 + k }, pending), []int{0}, []int{3})) == 0 {
		t.Error("must-fail control M1 without cache exhaustion: not detected")
	}
}

func TestSpeculativeStepRefuses(t *testing.T) {
	f := &fakeKV{kv: make([]int32, 4)}
	for _, tt := range []struct {
		name             string
		forward          func(int32, int) (int32, error)
		pos, max, budget int
	}{
		{"nil forward", nil, 0, 4, 1},
		{"no cache", f.forward, 0, 0, 1},
		{"negative position", f.forward, -1, 4, 1},
		{"exhausted cache", f.forward, 4, 4, 1},
		{"negative budget", f.forward, 0, 4, -1},
	} {
		emit, next, pos, err := speculativeStep(tt.forward, 5, []int32{1, 2}, tt.pos, tt.max, tt.budget)
		if err == nil || emit != nil || next != 5 || pos != tt.pos {
			t.Errorf("%s: got %v, %d, %d, %v; want an error and no state change", tt.name, emit, next, pos, err)
		}
	}
	emit, next, pos, err := speculativeStep(f.forward, 5, []int32{1, 2}, 1, 4, 0)
	if err != nil || emit != nil || next != 5 || pos != 1 {
		t.Errorf("zero budget: got %v, %d, %d, %v; want no change", emit, next, pos, err)
	}
	if f.calls != 0 {
		t.Errorf("forward called %d times, want 0", f.calls)
	}
	// A forward error is returned with its identity.
	_, _, _, err = speculativeStep(func(int32, int) (int32, error) { return 0, errFakeFree }, 5, nil, 0, 4, 1)
	if !errors.Is(err, errFakeFree) {
		t.Errorf("forward error: got %v", err)
	}
}

// counted wraps forward to count its calls in *n.
func counted(forward func(int32, int) (int32, error), n *int) func(int32, int) (int32, error) {
	return func(tok int32, pos int) (int32, error) {
		*n++
		return forward(tok, pos)
	}
}

// cycleModel predicts the token after tok in the cycle 0, 1, 2, 3, 4,
// except that at positions breakEvery-1 mod breakEvery (if breakEvery > 0)
// it skips two tokens ahead. It ignores the cache, so its predictions depend
// only on tok and pos.
func cycleModel(maxSeq, breakEvery int) func(int32, int) (int32, error) {
	return func(tok int32, pos int) (int32, error) {
		if pos < 0 || pos >= maxSeq {
			return 0, errOutsideCache
		}
		if breakEvery > 0 && pos%breakEvery == breakEvery-1 {
			return (tok + 3) % 5, nil
		}
		return (tok + 1) % 5, nil
	}
}

// plainGreedy prefills prompt and then decodes greedily without
// decodeState until steps tokens are emitted or the cache is full. It
// returns the emitted tokens and the final pending token and position.
func plainGreedy(forward func(int32, int) (int32, error), prompt []int32, maxSeq, steps int) ([]int32, int32, int, error) {
	var next int32
	var err error
	for pos, tok := range prompt {
		if next, err = forward(tok, pos); err != nil {
			return nil, 0, 0, err
		}
	}
	var out []int32
	pos := len(prompt)
	for ; len(out) < steps && pos < maxSeq; pos++ {
		out = append(out, next)
		if next, err = forward(next, pos); err != nil {
			return out, next, pos, err
		}
	}
	return out, next, pos, nil
}

// decodeRun is the result of mainDecode.
type decodeRun struct {
	st         *decodeState
	next       int32    // final pending token
	pos        int      // final position
	forwards   int      // forward and plain calls after the prefill
	speculated int      // tokens emitted by speculative steps
	shown      []int32  // tokens passed to show
	badPlain   []string // plain calls whose step or token disagree with st
}

// mainDecode prefills prompt as runGPUInference does and then runs decode
// with forward as both the speculative forward and the plain step, drafting
// up to draftN tokens per step with an n-gram drafter if draftN > 0.
func mainDecode(forward func(int32, int) (int32, error), prompt []int32, maxSeq, steps, draftN int) (*decodeRun, error) {
	var next int32
	var err error
	for pos, tok := range prompt {
		if next, err = forward(tok, pos); err != nil {
			return nil, err
		}
	}
	var drafter *speculative.NgramDrafter
	if draftN > 0 {
		drafter = speculative.NewNgramDrafter(3)
	}
	r := &decodeRun{st: newDecodeState(prompt, steps, drafter)}
	plainCalls := 0
	plain := func(step int, tok int32, pos int) (int32, error) {
		plainCalls++
		// decode emits tok before calling plain, and step counts the
		// tokens emitted before it, as runGPUInference's labels need.
		if g := r.st.generated; step != len(g)-1 || tok != g[len(g)-1] {
			r.badPlain = append(r.badPlain, fmt.Sprintf("plain(step %d, tok %d, pos %d) after emitting %v", step, tok, pos, g))
		}
		return forward(tok, pos)
	}
	show := func(tok int32) error {
		r.shown = append(r.shown, tok)
		return nil
	}
	r.next, r.pos, err = decode(r.st, counted(forward, &r.forwards), plain, show, next, len(prompt), maxSeq, steps, draftN)
	r.forwards += plainCalls
	r.speculated = len(r.st.generated) - plainCalls
	return r, err
}

// draftsDiffer reports the first n-gram over the token values below
// specVocab from which a and b draft differently, as a witness that their
// histories differ.
func draftsDiffer(a, b *speculative.NgramDrafter, maxDraft int) (ngram, da, db []int32, differ bool) {
	for i := range specVocab * specVocab * specVocab {
		g := []int32{int32(i / (specVocab * specVocab)), int32(i / specVocab % specVocab), int32(i % specVocab)}
		da, db = a.Draft(g, maxDraft), b.Draft(g, maxDraft)
		if !slices.Equal(da, db) {
			return g, da, db, true
		}
	}
	return nil, nil, nil, false
}

// TestDecodeStateNgram runs decode, the decode loop of runGPUInference,
// with the n-gram drafter main uses and checks that the output is plain
// greedy decoding ending at the same pending token and position, that every
// emitted token was shown in order, that plain was called with the step and
// token main labels it with, that the drafter drafts as one fed exactly the
// emitted context does, and, for a model the drafter can predict, that it
// drafted at every step from the context and then the pending token, so
// every token was speculated and no draft was rejected.
func TestDecodeStateNgram(t *testing.T) {
	cycle := []int32{0, 1, 2, 3, 4, 0, 1}
	kvHash := func(m int) func(int32, int) (int32, error) { return (&fakeKV{kv: make([]int32, m)}).forward }
	for _, tt := range []struct {
		name          string
		model         func(maxSeq int) func(int32, int) (int32, error)
		prompt        []int32
		maxSeq, steps int
		draftN        int
		exact         bool // every token must be emitted by a speculative step with no draft rejected
	}{
		{"plain", func(m int) func(int32, int) (int32, error) { return cycleModel(m, 7) }, cycle, 32, 16, 0, false},
		{"plain kv hash", kvHash, []int32{1, 2, 3}, 24, 16, 0, false},
		{"plain cache end", kvHash, []int32{1, 2, 3}, 12, 20, 0, false},
		{"cycle", func(m int) func(int32, int) (int32, error) { return cycleModel(m, 0) }, cycle, 32, 16, 3, true},
		{"cycle long drafts", func(m int) func(int32, int) (int32, error) { return cycleModel(m, 0) }, cycle, 32, 20, 8, true},
		// From position 7 the 4 drafts would reach position 11, past the
		// cache of 11, so the speculative step must clip them to 3.
		{"cycle cache end", func(m int) func(int32, int) (int32, error) { return cycleModel(m, 0) }, cycle, 11, 20, 4, true},
		{"cycle budget 2", func(m int) func(int32, int) (int32, error) { return cycleModel(m, 0) }, cycle, 32, 2, 4, true},
		{"cycle with breaks", func(m int) func(int32, int) (int32, error) { return cycleModel(m, 7) }, cycle, 40, 30, 3, false},
		{"kv hash", kvHash, []int32{1, 2, 3, 1, 2, 3, 1, 2}, 24, 16, 3, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			want, wantNext, wantPos, err := plainGreedy(tt.model(tt.maxSeq), tt.prompt, tt.maxSeq, tt.steps)
			if err != nil {
				t.Fatal(err)
			}
			r, err := mainDecode(tt.model(tt.maxSeq), tt.prompt, tt.maxSeq, tt.steps, tt.draftN)
			if err != nil {
				t.Fatal(err)
			}
			st, forwards, speculated := r.st, r.forwards, r.speculated
			if !slices.Equal(st.generated, want) {
				t.Errorf("generated %v, want %v", st.generated, want)
			}
			if r.next != wantNext || r.pos != wantPos {
				t.Errorf("decode ended with pending token %d at position %d, want %d at %d", r.next, r.pos, wantNext, wantPos)
			}
			if !slices.Equal(r.shown, want) {
				t.Errorf("shown %v, want %v", r.shown, want)
			}
			for _, b := range r.badPlain {
				t.Error(b)
			}
			wantContext := slices.Concat(tt.prompt, want)
			if !slices.Equal(st.context, wantContext) {
				t.Errorf("context %v, want %v", st.context, wantContext)
			}
			if st.drafter != nil {
				ref := speculative.NewNgramDrafter(3)
				ref.Feed(wantContext)
				if g, got, w, differ := draftsDiffer(st.drafter, ref, tt.maxSeq); differ {
					t.Errorf("drafter history is not the emitted context: from %v it drafts %v, want %v", g, got, w)
				}
			}
			if tt.exact {
				// A drafter missing part of the context drafts nothing at
				// some step, which then emits one token without speculating.
				if speculated != len(want) {
					t.Errorf("%d of %d tokens speculated: the drafter had nothing to draft", speculated, len(want))
				}
				if forwards != len(want) {
					t.Errorf("%d forward calls for %d tokens: drafts were rejected", forwards, len(want))
				}
			}
			t.Logf("%d tokens, %d speculated, %d forward calls", len(want), speculated, forwards)
		})
	}
}

// TestDecodeErrors checks that decode returns an error from plain unchanged
// and wraps one from a speculative step with its position, that both leave
// the pending token and position as they were before the failed step, and
// that a plain step shows its pending token before plain feeds it, so the
// token of a failed plain step has been shown.
func TestDecodeErrors(t *testing.T) {
	errFail := errors.New("forward failed")
	model := cycleModel(32, 0)
	failAt9 := func(tok int32, pos int) (int32, error) {
		if pos == 9 {
			return 0, errFail
		}
		return model(tok, pos)
	}
	// After the prompt the pending token is 2 at position 7.
	prompt := []int32{0, 1, 2, 3, 4, 0, 1}
	for _, tt := range []struct {
		name          string
		draftN        int
		wantErr       string
		wantGenerated []int32
		wantShown     []int32
		wantNext      int32
		wantPos       int
	}{
		// Plain steps feed 2 and 3, then fail feeding 4 at position 9.
		{"plain", 0, "forward failed", []int32{2, 3, 4}, []int32{2, 3, 4}, 4, 9},
		// The first speculative step feeds 2, 3, 4, 0 from position 7.
		{"speculative", 3, "speculative step at position 7: position 9: forward failed", nil, nil, 2, 7},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var drafter *speculative.NgramDrafter
			if tt.draftN > 0 {
				drafter = speculative.NewNgramDrafter(3)
			}
			st := newDecodeState(prompt, 16, drafter)
			plain := func(step int, tok int32, pos int) (int32, error) { return failAt9(tok, pos) }
			var shown []int32
			show := func(tok int32) error {
				shown = append(shown, tok)
				return nil
			}
			next, pos, err := decode(st, failAt9, plain, show, 2, len(prompt), 32, 16, tt.draftN)
			if !errors.Is(err, errFail) || err.Error() != tt.wantErr {
				t.Errorf("err = %v, want %q wrapping %v", err, tt.wantErr, errFail)
			}
			if next != tt.wantNext || pos != tt.wantPos {
				t.Errorf("returned pending token %d at position %d, want %d at %d", next, pos, tt.wantNext, tt.wantPos)
			}
			if !slices.Equal(st.generated, tt.wantGenerated) {
				t.Errorf("generated %v, want %v", st.generated, tt.wantGenerated)
			}
			if !slices.Equal(shown, tt.wantShown) {
				t.Errorf("shown %v, want %v", shown, tt.wantShown)
			}
		})
	}
}

// TestDecodeShowError checks that decode returns an error from show
// unchanged, shows nothing and runs no forward after it, and leaves the
// pending token and position as they were before the failed step.
func TestDecodeShowError(t *testing.T) {
	errShow := errors.New("show failed")
	// After the prompt the pending token is 2 at position 7.
	prompt := []int32{0, 1, 2, 3, 4, 0, 1}
	for _, tt := range []struct {
		name     string
		draftN   int
		wantNext int32
		wantPos  int
	}{
		// The first plain step shows 2 and feeds it; the second shows 3
		// and fails before feeding it.
		{"plain", 0, 3, 8},
		// The first speculative step emits 2, 3, 4, 0; showing 3 fails.
		{"speculative", 3, 2, 7},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var drafter *speculative.NgramDrafter
			if tt.draftN > 0 {
				drafter = speculative.NewNgramDrafter(3)
			}
			st := newDecodeState(prompt, 16, drafter)
			forwards, atFailure := 0, -1
			forward := counted(cycleModel(32, 0), &forwards)
			plain := func(step int, tok int32, pos int) (int32, error) { return forward(tok, pos) }
			shown := 0
			show := func(int32) error {
				if shown++; shown == 2 {
					atFailure = forwards
					return errShow
				}
				return nil
			}
			next, pos, err := decode(st, forward, plain, show, 2, len(prompt), 32, 16, tt.draftN)
			if !errors.Is(err, errShow) || err.Error() != errShow.Error() {
				t.Errorf("err = %v, want %v unchanged", err, errShow)
			}
			if shown != 2 {
				t.Errorf("show called %d times, want 2", shown)
			}
			if forwards != atFailure {
				t.Errorf("%d forwards after show failed, want 0", forwards-atFailure)
			}
			if next != tt.wantNext || pos != tt.wantPos {
				t.Errorf("returned pending token %d at position %d, want %d at %d", next, pos, tt.wantNext, tt.wantPos)
			}
		})
	}
}
