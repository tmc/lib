package dsv4

import (
	"errors"
	"io"
	"testing"
)

type cleanupRunBackend struct {
	runBackend
	releaseErr     error
	partialErr     error
	factoryErr     error
	verifierErr    error
	verifierCloses int
	releases       int
	lateClose      bool
}

func (b *cleanupRunBackend) NewSpeculativeVerifier(s State) (SpeculativeVerifier, error) {
	return &cleanupRunVerifier{runVerifier: runVerifier{b: &b.runBackend, s: s.(*runState)}, owner: b}, b.factoryErr
}

type cleanupRunVerifier struct {
	runVerifier
	owner *cleanupRunBackend
}

func (v *cleanupRunVerifier) Close() error {
	v.owner.verifierCloses++
	if v.s.closed {
		v.owner.lateClose = true
	}
	return v.owner.verifierErr
}

func (v *cleanupRunVerifier) Snapshot() (Snapshot, error) {
	x, _ := v.s.Snapshot()
	return &cleanupRunSnapshot{Snapshot: x, owner: v.owner, state: v.s}, v.owner.partialErr
}

type cleanupRunSnapshot struct {
	Snapshot
	owner *cleanupRunBackend
	state *runState
}

func (s *cleanupRunSnapshot) Close() error {
	s.owner.releases++
	if s.state.closed {
		s.owner.lateClose = true
	}
	if s.owner.releaseErr != nil {
		return s.owner.releaseErr
	}
	return s.Snapshot.Close()
}

func (v *cleanupRunVerifier) Restore(s Snapshot) error {
	return v.runVerifier.Restore(s.(*cleanupRunSnapshot).Snapshot)
}

func TestRunSpeculativeCaseRetainsCleanupPrerequisites(t *testing.T) {
	for _, partial := range []bool{false, true} {
		t.Run(map[bool]string{false: "cycle", true: "partial snapshot"}[partial], func(t *testing.T) {
			fault := errors.New("snapshot release failure")
			b := &cleanupRunBackend{releaseErr: fault}
			if partial {
				b.partialErr = errors.New("partial snapshot failure")
			}
			_, err := RunSpeculativeCase(b, runDrafter{}, "host", []int{1}, 16, 2, 1)
			if !errors.Is(err, fault) || !errors.Is(err, ErrDesynchronized) || partial && !errors.Is(err, b.partialErr) {
				t.Fatalf("lost error chain: %v", err)
			}
			var owner io.Closer
			if !errors.As(err, &owner) {
				t.Fatal("missing cleanup owner")
			}
			for _, s := range b.states {
				if s.closed {
					t.Fatal("state closed before retained frontier")
				}
			}
			if b.verifierCloses != 0 {
				t.Fatal("verifier closed before retained frontier")
			}
			if err := owner.Close(); !errors.Is(err, fault) {
				t.Fatalf("retry lost failure: %v", err)
			} else if !errors.As(err, &owner) {
				t.Fatal("retry error lost run cleanup owner")
			}
			b.releaseErr = nil
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			releases := b.releases
			if err := owner.Close(); err != nil || b.releases != releases || b.lateClose {
				t.Fatalf("non-idempotent or late cleanup: %v", err)
			}
			for _, s := range b.states {
				if !s.closed {
					t.Fatal("state retained after successful cleanup")
				}
			}
			if b.verifierCloses != 1 {
				t.Fatalf("verifier closes = %d", b.verifierCloses)
			}
		})
	}
}

func TestSpeculatorFactoryOwner(t *testing.T) {
	for _, partial := range []bool{false, true} {
		t.Run(map[bool]string{false: "normal", true: "partial constructor"}[partial], func(t *testing.T) {
			fault, primary := errors.New("verifier release failure"), errors.New("factory failure")
			b := &cleanupRunBackend{verifierErr: fault}
			state, _ := b.NewState(16)
			if partial {
				b.factoryErr = primary
			}
			s, err := NewSpeculatorForBackend(b, state, runDrafter{})
			if !partial {
				if err != nil || s == nil {
					t.Fatalf("constructor: %v", err)
				}
				err = s.Close()
			} else if s != nil || !errors.Is(err, primary) {
				t.Fatalf("partial constructor: %v", err)
			}
			var owner io.Closer
			if !errors.Is(err, fault) || !errors.As(err, &owner) || b.verifierCloses != 1 {
				t.Fatalf("missing factory cleanup owner: %v", err)
			}
			if err := owner.Close(); !errors.Is(err, fault) || partial && !errors.Is(err, primary) {
				t.Fatalf("retry error chain: %v", err)
			}
			b.verifierErr = nil
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if err := owner.Close(); err != nil || b.verifierCloses != 3 || b.lateClose {
				t.Fatalf("factory cleanup not idempotent: %v", err)
			}
		})
	}
}

func TestRunSpeculativeCaseFactoryFailureRetainsStates(t *testing.T) {
	for _, factoryErr := range []error{nil, errors.New("factory failure")} {
		t.Run(map[bool]string{false: "normal run", true: "partial factory"}[factoryErr != nil], func(t *testing.T) {
			fault := errors.New("verifier release failure")
			b := &cleanupRunBackend{factoryErr: factoryErr, verifierErr: fault}
			_, err := RunSpeculativeCase(b, runDrafter{}, "host", []int{1}, 16, 2, 1)
			var owner io.Closer
			if !errors.Is(err, fault) || !errors.As(err, &owner) {
				t.Fatalf("missing constructor cleanup owner: %v", err)
			}
			for _, s := range b.states {
				if s.closed {
					t.Fatal("state closed before partial verifier")
				}
			}
			b.verifierErr = nil
			if err := owner.Close(); err != nil || b.lateClose {
				t.Fatalf("constructor cleanup: %v", err)
			}
			for _, s := range b.states {
				if !s.closed {
					t.Fatal("state leaked after verifier release")
				}
			}
		})
	}
}

func TestSpeculatorDirectVerifierRemainsCallerOwned(t *testing.T) {
	b := new(cleanupRunBackend)
	state, _ := b.NewState(16)
	v, _ := b.NewSpeculativeVerifier(state)
	s, err := NewSpeculator(v, runDrafter{})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil || b.verifierCloses != 0 {
		t.Fatal("direct constructor stole verifier ownership")
	}
}
