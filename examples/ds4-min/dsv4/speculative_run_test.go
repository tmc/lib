package dsv4

import (
	"errors"
	"testing"
)

type runState struct {
	contractState
	history []int
	closed  bool
}

func (s *runState) Close() error { s.closed = true; return nil }
func (s *runState) Snapshot() (Snapshot, error) {
	return &speculativeSnapshot{append([]int(nil), s.history...)}, nil
}
func (s *runState) Restore(x Snapshot) error {
	s.history = append([]int(nil), x.(*speculativeSnapshot).values...)
	return nil
}

type runBackend struct {
	states    []*runState
	reject    bool
	fail      bool
	badCommit bool
}

func (b *runBackend) NewState(int) (State, error) {
	s := new(runState)
	b.states = append(b.states, s)
	return s, nil
}
func (b *runBackend) Close() error { return nil }
func (b *runBackend) Eval(s State, token, pos int) error {
	r := s.(*runState)
	if pos != len(r.history) {
		return errors.New("timeline mismatch")
	}
	r.history = append(r.history, token)
	return nil
}
func (b *runBackend) Logits(State) []float32 { return []float32{0, 2} }
func (b *runBackend) NewSpeculativeVerifier(s State) (SpeculativeVerifier, error) {
	return &runVerifier{b: b, s: s.(*runState)}, nil
}

type runVerifier struct {
	b    *runBackend
	s    *runState
	base int
}

func (v *runVerifier) CurrentLogits() []float32    { return v.b.Logits(v.s) }
func (v *runVerifier) Snapshot() (Snapshot, error) { return v.s.Snapshot() }
func (v *runVerifier) Restore(s Snapshot) error    { return v.s.Restore(s) }
func (v *runVerifier) VerifyBatch(_ int, d []int, pos int) ([]int, error) {
	v.base = len(v.s.history)
	tops := make([]int, len(d))
	for i, token := range d {
		if err := v.b.Eval(v.s, token, pos+i); err != nil {
			return nil, err
		}
		tops[i] = 1
	}
	if v.b.fail {
		return nil, errors.New("planted verify failure")
	}
	if v.b.reject {
		tops[0] = 0
	}
	return tops, nil
}
func (v *runVerifier) CommitPrefix(n int) error {
	if v.b.badCommit {
		n++
	}
	if v.base+n > len(v.s.history) {
		return errors.New("bad prefix")
	}
	v.s.history = v.s.history[:v.base+n]
	return nil
}

type runDrafter struct{}

func (runDrafter) Draft(_, _, n int) ([]int, error) {
	d := make([]int, n)
	for i := range d {
		d[i] = 1
	}
	return d, nil
}

func TestRunSpeculativeCaseTransitions(t *testing.T) {
	for _, reject := range []bool{false, true} {
		b := &runBackend{reject: reject}
		r, err := RunSpeculativeCase(b, runDrafter{}, "host", []int{1, 1}, 16, 6, 2)
		if err != nil {
			t.Fatal(err)
		}
		if len(r.Trace.Strict) != 6 || len(r.Trace.Speculative) != 6 || len(b.states[1].history) != 8 {
			t.Fatalf("incomplete timeline: %+v", r)
		}
		if reject && r.Trace.AcceptedDrafts != 0 || !reject && r.Trace.AcceptedDrafts != 6 {
			t.Fatalf("acceptance count: %d", r.Trace.AcceptedDrafts)
		}
		for _, s := range b.states {
			if !s.closed {
				t.Fatal("state not closed")
			}
		}
	}
}
func TestRunSpeculativeCaseFailureCleanup(t *testing.T) {
	b := &runBackend{fail: true}
	if _, err := RunSpeculativeCase(b, runDrafter{}, "host", []int{1}, 16, 6, 2); err == nil {
		t.Fatal("accepted verify failure")
	}
	if len(b.states[1].history) != 1 {
		t.Fatal("failed verification did not restore")
	}
	for _, s := range b.states {
		if !s.closed {
			t.Fatal("leaked state")
		}
	}
}
func TestRunSpeculativeCaseDetectsCommitMutation(t *testing.T) {
	b := &runBackend{badCommit: true}
	if _, err := RunSpeculativeCase(b, runDrafter{}, "host", []int{1}, 16, 6, 2); err == nil {
		t.Fatal("off-by-one commit passed")
	}
}
