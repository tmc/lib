package main

import (
	"errors"
	"fmt"
	"hash/fnv"
	"slices"
	"testing"
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
