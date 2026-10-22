package dsv4

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

type sampledTestBackend struct {
	state                                                             *runState
	rows                                                              [][]float32
	evalFail, verifyFail, commitFail, restoreFail, rowFail, closeFail bool
	verifies, closes, observations                                    int
	observeFail                                                       bool
	factoryErr, ownerFail, factoryNoRows                              bool
	factoryCalls, ownerCloses                                         int
}

func (b *sampledTestBackend) NewState(int) (State, error) { b.state = &runState{}; return b.state, nil }
func (b *sampledTestBackend) Close() error                { return nil }
func (b *sampledTestBackend) NewSpeculativeVerifier(s State) (SpeculativeVerifier, error) {
	b.factoryCalls++
	if s != b.state {
		return nil, errors.New("foreign state")
	}
	v := &sampledTestVerifier{b: b}
	if b.factoryNoRows {
		return &sampledTopOnlyVerifier{SpeculativeVerifier: v, v: v}, nil
	}
	if b.factoryErr {
		return v, errors.New("factory fault")
	}
	return v, nil
}
func (b *sampledTestBackend) Eval(s State, tok, pos int) error {
	if b.evalFail && pos > 1 {
		return errors.New("eval fault")
	}
	r := s.(*runState)
	if pos != len(r.history) {
		return errors.New("wrong position")
	}
	r.history = append(r.history, tok)
	return nil
}
func (b *sampledTestBackend) Logits(s State) []float32 {
	return b.rows[min(len(s.(*runState).history), len(b.rows)-1)]
}

type sampledTestSnapshot struct {
	history []int
	b       *sampledTestBackend
}

func (s *sampledTestSnapshot) Close() error {
	s.b.closes++
	if s.b.closeFail {
		return errors.New("close fault")
	}
	return nil
}

type sampledTestVerifier struct {
	b    *sampledTestBackend
	base int
	rows [][]float32
}

type sampledTopOnlyVerifier struct {
	SpeculativeVerifier
	v *sampledTestVerifier
}

func (v *sampledTopOnlyVerifier) Close() error { return v.v.Close() }

func TestSampledFactoryCapabilityFailure(t *testing.T) {
	base, b, d := sampledFixture(t, ExactSampling, sampler.Params{}, 7, [][]float32{{0, 9, -9}}, []int{1, 1})
	b.factoryNoRows = true
	s, err := NewSampledSessionForBackend(base.session, d, sampler.Params{}, sampler.RNG{State: 7}, ExactSampling, SampledPolicy{}, 2)
	if s != nil || !errors.Is(err, ErrUnimplemented) || b.ownerCloses != 1 || base.session.broken {
		t.Fatalf("capability failure %v, %v, closes=%d", s, err, b.ownerCloses)
	}
	n := b.factoryCalls
	_, err = NewSampledSessionForBackend(base.session, d, sampler.Params{}, sampler.RNG{}, ExactSampling, SampledPolicy{}, 0)
	if err == nil || b.factoryCalls != n {
		t.Fatal("invalid policy allocated verifier")
	}
}

func (v *sampledTestVerifier) BoundTo(s State) bool { return s == v.b.state }

func (v *sampledTestVerifier) Close() error {
	v.b.ownerCloses++
	if v.b.ownerFail {
		return errors.New("owner close fault")
	}
	return nil
}

func TestSampledFactoryOwnership(t *testing.T) {
	for _, factoryFailure := range []bool{false, true} {
		t.Run(fmt.Sprint(factoryFailure), func(t *testing.T) {
			base, b, d := sampledFixture(t, ExactSampling, sampler.Params{}, 7, [][]float32{{0, 9, -9}}, []int{1, 1})
			b.factoryErr, b.ownerFail = factoryFailure, true
			s, err := NewSampledSessionForBackend(base.session, d, sampler.Params{}, sampler.RNG{State: 7}, ExactSampling, SampledPolicy{}, 2)
			if s == nil || (err != nil) != factoryFailure || s.owner == nil {
				t.Fatalf("factory result %v, %v", s, err)
			}
			if err := s.Close(); err == nil || s.owner == nil {
				t.Fatal("failed cleanup lost owner")
			}
			b.ownerFail = false
			if err := s.Close(); err != nil || s.owner != nil || base.session.closed {
				t.Fatalf("cleanup result %v", err)
			}
			n := b.ownerCloses
			if err := s.Close(); err != nil || b.ownerCloses != n {
				t.Fatal("cleanup not idempotent")
			}
		})
	}
}

func (v *sampledTestVerifier) CurrentLogits() []float32 { return v.b.Logits(v.b.state) }
func (v *sampledTestVerifier) Snapshot() (Snapshot, error) {
	return &sampledTestSnapshot{append([]int(nil), v.b.state.history...), v.b}, nil
}
func (v *sampledTestVerifier) Restore(s Snapshot) error {
	if v.b.restoreFail {
		return errors.New("restore fault")
	}
	v.b.state.history = append([]int(nil), s.(*sampledTestSnapshot).history...)
	return nil
}
func (v *sampledTestVerifier) VerifyBatch(_ int, d []int, pos int) ([]int, error) {
	v.b.verifies++
	v.base = pos
	v.rows = nil
	tops := make([]int, len(d))
	for i, tok := range d {
		tops[i] = argmax(v.CurrentLogits())
		v.b.state.history = append(v.b.state.history, tok)
		v.rows = append(v.rows, append([]float32(nil), v.CurrentLogits()...))
	}
	if v.b.verifyFail {
		return nil, errors.New("verify fault")
	}
	return tops, nil
}
func (v *sampledTestVerifier) RowLogits(i int) ([]float32, error) {
	if v.b.rowFail {
		return nil, errors.New("row fault")
	}
	return v.rows[i], nil
}
func (v *sampledTestVerifier) CommitPrefix(n int) error {
	if v.b.commitFail {
		return errors.New("commit fault")
	}
	v.b.state.history = v.b.state.history[:v.base+n]
	return nil
}

type sampledTestDraft struct {
	preparedDrafter
	b *sampledTestBackend
}

func (d *sampledTestDraft) Prepare(seed, pos int) error { d.seed, d.position = seed, pos; return nil }
func (d *sampledTestDraft) Observe(pos, n int) error {
	d.b.observations += n
	if d.b.observeFail && pos > 1 {
		return errors.New("observe fault")
	}
	return nil
}

func sampledFixture(t *testing.T, mode SampledMode, params sampler.Params, seed uint64, rows [][]float32, drafts []int) (*SampledSession, *sampledTestBackend, *sampledTestDraft) {
	t.Helper()
	b := &sampledTestBackend{rows: rows}
	x, e := NewSession(b, 64, NewStopSet(2))
	if e != nil {
		t.Fatal(e)
	}
	if e = x.Sync([]int{0}); e != nil {
		t.Fatal(e)
	}
	d := &sampledTestDraft{preparedDrafter: preparedDrafter{proposal: drafts}, b: b}
	s, e := NewSampledSession(x, &sampledTestVerifier{b: b}, d, params, sampler.RNG{State: seed}, mode, SampledPolicy{}, 2)
	if e != nil {
		t.Fatal(e)
	}
	return s, b, d
}

func TestSampledSessionReference(t *testing.T) {
	// Independent conditional CDF, using the already C-qualified sampler
	// distribution builder. This does not call ExactVerify for its expected values.
	logits := []float32{float32(math.Log(.62)), float32(math.Log(.27)), float32(math.Log(.11))}
	params := sampler.Params{Temperature: 1, TopP: 1}
	p32, e := sampler.New(3).Probabilities(logits, params)
	if e != nil {
		t.Fatal(e)
	}
	p := []float64{float64(p32[0]), float64(p32[1]), float64(p32[2])}
	total := p[0] + p[1] + p[2]
	for seed := uint64(1); seed <= 1000; seed++ {
		s, b, _ := sampledFixture(t, ExactSampling, params, seed, [][]float32{logits, logits, logits, logits, logits}, []int{0, 1})
		rng := sampler.RNG{State: seed}
		first := sampler.New(3).Sample(logits, params, &rng)
		want := []int{first}
		committed := []int{0}
		accepted := 0
		if first != 2 {
			committed = append(committed, first)
			for _, draft := range []int{0, 1} {
				ok := float64(rng.Float32()) < p[draft]/total
				token := draft
				if !ok {
					u := float64(rng.Float32())
					cdf := 0.0
					for id, mass := range p {
						if id == draft {
							continue
						}
						cdf += mass / (total - p[draft])
						if u < cdf {
							token = id
							break
						}
					}
				}
				want = append(want, token)
				if token != 2 {
					committed = append(committed, token)
				}
				if ok {
					accepted++
				}
				if !ok || token == 2 {
					break
				}
			}
		}
		got, e := s.Next(12)
		if e != nil || !reflect.DeepEqual(got.Tokens, want) || !reflect.DeepEqual(b.state.history, committed) || got.Accepted != accepted || s.Random() != rng {
			t.Fatalf("seed %d got=%+v error=%v history=%v RNG=%x want=%v/%v/%x", seed, got, e, b.state.history, s.Random().State, want, committed, rng.State)
		}
	}
}

func TestSampledSessionFilteredSupport(t *testing.T) {
	for _, mode := range []SampledMode{ExactSampling, OpportunisticSampling} {
		rows := [][]float32{{0, 4, -4}, {0, 4, -4}, {0, 4, -4}, {4, 0, -4}, {0, 4, -4}}
		for seed := uint64(1); seed < 30; seed++ {
			s, b, _ := sampledFixture(t, mode, sampler.Params{Temperature: 1, TopK: 1, TopP: .5, MinP: .5}, seed, rows, []int{1, 1})
			r, e := s.Next(12)
			if e != nil {
				t.Fatal(e)
			}
			want := []int{1, 1}
			if mode == ExactSampling {
				want = append(want, 0)
			}
			if !reflect.DeepEqual(r.Tokens, want) || !reflect.DeepEqual(b.state.history, append([]int{0}, want...)) {
				t.Fatal("filtered support or row alignment", r, b.state.history)
			}
		}
	}
}

func TestSampledSessionFailureFrontiers(t *testing.T) {
	rows := [][]float32{{0, 9, -9}, {0, 9, -9}, {0, 9, -9}, {0, 9, -9}, {0, 9, -9}}
	for _, tc := range []struct {
		name   string
		set    func(*sampledTestBackend, *sampledTestDraft)
		poison bool
	}{
		{"verify", func(b *sampledTestBackend, d *sampledTestDraft) { b.verifyFail = true }, false},
		{"row", func(b *sampledTestBackend, d *sampledTestDraft) { b.rowFail = true }, false},
		{"commit", func(b *sampledTestBackend, d *sampledTestDraft) { b.commitFail = true }, false},
		{"support", func(b *sampledTestBackend, d *sampledTestDraft) { d.commitErr = errors.New("support fault") }, false},
		{"restore", func(b *sampledTestBackend, d *sampledTestDraft) { b.verifyFail = true; b.restoreFail = true }, true},
		{"discard", func(b *sampledTestBackend, d *sampledTestDraft) {
			b.verifyFail = true
			d.discardErr = errors.New("discard fault")
		}, true},
		{"close", func(b *sampledTestBackend, d *sampledTestDraft) { b.closeFail = true }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, b, d := sampledFixture(t, ExactSampling, sampler.Params{Temperature: 0}, 77, rows, []int{1, 1})
			tc.set(b, d)
			r, e := s.Next(12)
			want := []int{1}
			if tc.name == "close" {
				want = []int{1, 1, 1}
			}
			if e == nil || errors.Is(e, ErrDesynchronized) != tc.poison || !reflect.DeepEqual(r.Tokens, want) || s.Random().State != 77 || b.closes != 1 {
				t.Fatal(r, e, s.Random(), b.closes)
			}
			if !tc.poison && (!reflect.DeepEqual(b.state.history, []int{0, 1}) || !reflect.DeepEqual(s.session.Tokens(), []int{0, 1}) || len(d.lookahead) > 0) {
				t.Fatal("rollback leaked suffix")
			}
			if tc.poison {
				if _, e = s.Next(12); !errors.Is(e, ErrDesynchronized) {
					t.Fatal("poison reused", e)
				}
			}
		})
	}
}

func TestSampledSessionStopsAndBudget(t *testing.T) {
	for _, drafts := range [][]int{{1, 2}, {2, 1}, {1, 1}} {
		rows := [][]float32{{0, 9, -9}, {0, 9, -9}, {0, 9, -9}, {0, -9, 9}, {0, -9, 9}}
		s, b, _ := sampledFixture(t, ExactSampling, sampler.Params{}, 7, rows, drafts)
		r, e := s.Next(12)
		if e != nil {
			t.Fatal(e)
		}
		if len(r.Tokens) > 3 || len(b.state.history) > 3 {
			t.Fatal("stop committed", r, b.state.history)
		}
	}
	s, b, _ := sampledFixture(t, ExactSampling, sampler.Params{}, 7, [][]float32{{0, 9, -9}, {0, 9, -9}, {0, 9, -9}}, []int{1, 1})
	r, e := s.Next(1)
	if e != nil || len(r.Tokens) != 1 || b.verifies != 0 || b.observations != 1 {
		t.Fatal("budget", r, e)
	}
}

func TestSampledPolicy(t *testing.T) {
	var p SampledPolicy
	for i := 0; i < 4; i++ {
		if !p.Allow(20) {
			t.Fatal("unexpected initial skip")
		}
		p.Note(1, false, nil)
	}
	if p.Skip != 2 || p.Allow(20) || p.Allow(20) || !p.Allow(20) {
		t.Fatal("window backoff", p)
	}
	if p.Allow(9) {
		t.Fatal("tail drafting")
	}
	low := float32(.5)
	p.Note(0, true, &low)
	if p.Skip != 4 {
		t.Fatal("short lifetime", p)
	}
	var cold SampledPolicy
	cold.Note(0, true, &low)
	if cold.Skip != 7 {
		t.Fatal("cold backoff", cold)
	}
	strict := SampledPolicy{Strict: true}
	if strict.Allow(100) {
		t.Fatal("strict drafting")
	}
}

func TestSampledStrictAndBackoffObservation(t *testing.T) {
	for _, policy := range []SampledPolicy{{Strict: true}, {Skip: 2}, {}} {
		s, b, d := sampledFixture(t, ExactSampling, sampler.Params{}, 7, [][]float32{{0, 9, -9}, {0, 9, -9}, {0, 9, -9}}, []int{1, 1})
		s.policy = policy
		budget := 12
		if !policy.Strict && policy.Skip == 0 {
			budget = 9
		}
		r, e := s.Next(budget)
		if e != nil || b.verifies != 0 || d.drafts != 0 || b.observations != 1 || !reflect.DeepEqual(r.Tokens, []int{1}) {
			t.Fatal("skipped support history", r, e, b.observations)
		}
	}
}

func TestSampledSessionRejectsForeignRequest(t *testing.T) {
	s, _, d := sampledFixture(t, ExactSampling, sampler.Params{}, 7, [][]float32{{0, 9, -9}, {0, 9, -9}}, []int{1, 1})
	foreign := &sampledTestVerifier{b: &sampledTestBackend{state: &runState{}}}
	if _, e := NewSampledSession(s.session, foreign, d, sampler.Params{}, sampler.RNG{}, ExactSampling, SampledPolicy{}, 2); e == nil {
		t.Fatal("foreign request admitted")
	}
}

func TestSampledCleanupRetry(t *testing.T) {
	for _, verifyFault := range []bool{false, true} {
		s, b, _ := sampledFixture(t, ExactSampling, sampler.Params{}, 7, [][]float32{{0, 9, -9}, {0, 9, -9}, {0, 9, -9}}, []int{1, 1})
		b.verifyFail, b.closeFail = verifyFault, true
		r, e := s.Next(12)
		if !errors.Is(e, ErrDesynchronized) || s.pending == nil || b.closes != 1 {
			t.Fatal("failed snapshot owner lost", r, e, s.pending, b.closes)
		}
		if e = s.Close(); e == nil || s.pending == nil || b.closes != 2 {
			t.Fatal("failed retry lost owner", e, b.closes)
		}
		b.closeFail = false
		if e = s.Close(); e != nil || s.pending != nil || b.closes != 3 {
			t.Fatal("cleanup retry", e, b.closes)
		}
		if e = s.Close(); e != nil || b.closes != 3 {
			t.Fatal("duplicate close", e, b.closes)
		}
		if _, e = s.Next(12); !errors.Is(e, ErrClosed) {
			t.Fatal("closed sampler reused", e)
		}
	}
}

func TestSampledObservationFailureReportsCommit(t *testing.T) {
	s, b, _ := sampledFixture(t, ExactSampling, sampler.Params{}, 7, [][]float32{{0, 9, -9}, {0, 9, -9}, {0, 9, -9}}, []int{1, 1})
	b.observeFail = true
	r, e := s.Next(12)
	if !errors.Is(e, ErrDesynchronized) || r.Committed != 3 || r.Accepted != 2 || !reflect.DeepEqual(r.Tokens, []int{1, 1, 1}) || !reflect.DeepEqual(s.session.Tokens(), b.state.history) || len(b.state.history) != 4 || b.closes != 1 {
		t.Fatal("irreversible committed suffix hidden", r, e, b.state.history)
	}
}

func TestSampledBackendBinding(t *testing.T) {
	s, _, d := sampledFixture(t, ExactSampling, sampler.Params{}, 7, [][]float32{{0, 9, -9}, {0, 9, -9}}, []int{1, 1})
	bound, e := NewSampledSessionForBackend(s.session, d, sampler.Params{}, sampler.RNG{}, ExactSampling, SampledPolicy{}, 2)
	if e != nil || bound.session.state != s.session.state {
		t.Fatal("native binding created another request", e)
	}
	if _, e = bound.Next(1); e != nil {
		t.Fatal(e)
	}
}

func TestSampledSuffixFailureRestoresRandom(t *testing.T) {
	logits := []float32{float32(math.Log(.62)), float32(math.Log(.27)), float32(math.Log(.11))}
	params := sampler.Params{Temperature: 1, TopP: 1}
	var seed uint64
	var boundary sampler.RNG
	for seed = 1; seed < 100; seed++ {
		boundary = sampler.RNG{State: seed}
		first := sampler.New(3).Sample(logits, params, &boundary)
		trial := boundary
		if first != 2 && trial.Float32() < .62 {
			break
		}
	}
	s, b, d := sampledFixture(t, ExactSampling, params, seed, [][]float32{logits, logits, logits, logits}, []int{0, 1})
	b.verifyFail = true
	r, e := s.Next(12)
	if e == nil || errors.Is(e, ErrDesynchronized) || s.Random() != boundary || len(r.Tokens) != 1 || len(b.state.history) != 2 || len(d.lookahead) != 0 {
		t.Fatal("suffix RNG rollback", r, e, s.Random(), boundary)
	}
}

func TestSampledCorrectionFailureRollback(t *testing.T) {
	rows := [][]float32{{0, 4, -4}, {0, 4, -4}, {0, 4, -4}, {4, 0, -4}, {0, 4, -4}}
	s, b, d := sampledFixture(t, ExactSampling, sampler.Params{Temperature: 1, TopK: 1}, 7, rows, []int{1, 1})
	b.evalFail = true
	r, e := s.Next(12)
	if e == nil || errors.Is(e, ErrDesynchronized) || r.Committed != 1 || !reflect.DeepEqual(s.session.Tokens(), []int{0, 1}) || !reflect.DeepEqual(b.state.history, []int{0, 1}) || len(d.lookahead) != 0 {
		t.Fatal("correction rollback", r, e, b.state.history)
	}
}
