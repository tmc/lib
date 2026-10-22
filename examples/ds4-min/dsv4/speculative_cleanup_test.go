package dsv4

import (
	"errors"
	"io"
	"testing"
)

func TestSpeculatorCleanupOwner(t *testing.T) {
	for _, tt := range []struct {
		name                                string
		prepared, partial, executionFailure bool
	}{
		{"ordinary", false, false, false},
		{"prepared", true, false, false},
		{"partial snapshot", true, true, false},
		{"execution error", true, false, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s, v, _ := preparedFixture(t)
			primary, cleanup := errors.New("primary fault"), errors.New("release fault")
			v.closeErr = cleanup
			if tt.partial {
				v.snapshotErr = primary
			}
			if tt.executionFailure {
				v.verifyErr = primary
			}
			var err error
			if tt.prepared {
				_, err = s.VerifyPrepared(8, 1, 3)
			} else {
				_, err = s.Verify(8, 1, 3)
			}
			if !errors.Is(err, cleanup) || !errors.Is(err, ErrDesynchronized) || v.closes != 1 || s.pending == nil {
				t.Fatalf("lost failed cleanup: %v, closes=%d", err, v.closes)
			}
			if (tt.partial || tt.executionFailure) && !errors.Is(err, primary) {
				t.Fatalf("lost primary fault: %v", err)
			}
			var owner io.Closer
			if !errors.As(err, &owner) {
				t.Fatal("error does not carry cleanup owner")
			}
			n := v.snapshots
			if _, err := s.Verify(8, 1, 3); !errors.Is(err, ErrDesynchronized) || v.snapshots != n {
				t.Fatal("failure resumed ordinary verification")
			}
			if _, err := s.VerifyPrepared(8, 1, 3); !errors.Is(err, ErrDesynchronized) || v.snapshots != n {
				t.Fatal("failure resumed prepared verification")
			}
			if err := owner.Close(); !errors.Is(err, cleanup) || s.pending == nil || v.closes != 2 {
				t.Fatalf("retry lost owner: %v", err)
			}
			v.closeErr = nil
			if err := owner.Close(); err != nil || s.pending != nil || v.closes != 3 {
				t.Fatalf("cleanup retry: %v", err)
			}
			if err := s.Close(); err != nil || v.closes != 3 {
				t.Fatal("close not idempotent")
			}
			if _, err := s.Verify(8, 1, 3); !errors.Is(err, ErrClosed) || v.snapshots != n {
				t.Fatal("cleanup reopened ordinary verification")
			}
			if _, err := s.VerifyPrepared(8, 1, 3); !errors.Is(err, ErrClosed) || v.snapshots != n {
				t.Fatal("cleanup reopened prepared verification")
			}
		})
	}
}

func TestSpeculatorCloseBeforeUse(t *testing.T) {
	s, v, _ := preparedFixture(t)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Verify(8, 1, 3); !errors.Is(err, ErrClosed) || v.snapshots != 0 {
		t.Fatal("closed speculator performed work")
	}
	var zero *Speculator
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
}
