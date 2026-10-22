//go:build darwin && arm64

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type metalOuterCloseVerifier struct{ snapshot dsv4.Snapshot }

func (v *metalOuterCloseVerifier) CurrentLogits() []float32 { return []float32{0, 1} }
func (v *metalOuterCloseVerifier) VerifyBatch(int, []int, int) ([]int, error) {
	return []int{1}, nil
}
func (v *metalOuterCloseVerifier) CommitPrefix(int) error { return nil }
func (v *metalOuterCloseVerifier) Snapshot() (dsv4.Snapshot, error) {
	return v.snapshot, nil
}
func (v *metalOuterCloseVerifier) Restore(dsv4.Snapshot) error { return nil }

type metalOuterCloseDrafter struct{}

func (metalOuterCloseDrafter) Draft(int, int, int) ([]int, error) { return []int{1}, nil }

func TestMetalDSparkOuterOwnerClose(t *testing.T) {
	for _, fail := range []bool{false, true} {
		outer := &metalLifecycleSnapshot{fail: true}
		s, err := dsv4.NewSpeculator(&metalOuterCloseVerifier{outer}, metalOuterCloseDrafter{})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Verify(0, 0, 1); !errors.Is(err, dsv4.ErrDesynchronized) || outer.calls != 1 {
			t.Fatalf("retained outer frontier setup: %v, calls=%d", err, outer.calls)
		}
		outer.fail = fail
		raw := &metalLifecycleSnapshot{}
		v := &metalSpeculativeVerifier{frontier: raw}
		forward := &metalDSparkNative{}
		d := &metalDSparkDrafter{speculator: s, verifier: v, forward: forward}
		err = d.Close()
		if outer.calls != 2 {
			t.Fatalf("outer owner bypassed retained snapshot: calls=%d", outer.calls)
		}
		if fail {
			if !errors.Is(err, dsv4.ErrDesynchronized) || d.speculator != s || d.verifier != v || d.forward != forward || v.closed || raw.calls != 0 || forward.closed {
				t.Fatalf("failed outer close released downstream owner: %v", err)
			}
			outer.fail = false
			if err := d.Close(); err != nil || outer.calls != 3 {
				t.Fatalf("outer retry: %v, calls=%d", err, outer.calls)
			}
		} else if err != nil {
			t.Fatal(err)
		}
		if d.speculator != nil || d.verifier != nil || d.forward != nil || !v.closed || raw.calls != 1 || !forward.closed {
			t.Fatal("successful cleanup did not close downstream owners")
		}
		if _, err := s.Verify(0, 0, 1); !errors.Is(err, dsv4.ErrClosed) {
			t.Fatalf("closed outer verifier resumed: %v", err)
		}
		calls := outer.calls
		if err := d.Close(); err != nil || outer.calls != calls || raw.calls != 1 {
			t.Fatalf("idempotent owner close: %v", err)
		}
	}
}

func TestMetalDSparkConstructorRetainsOuterOwner(t *testing.T) {
	a := &metalStateAdapter{}
	target := &metalRequestState{owner: a, runtime: &metalRuntime{dsparkFeatures: &metalDSparkFeatures{}}}
	w := &metalDSparkWeights{}
	f := &metalDSparkNative{}
	v := &metalSpeculativeVerifier{}
	d, s, err := newMetalDSparkSpeculatorWithFactory(a, target, w, .5, metalDSparkConstructorFactory{
		forward:  func(*metalRequestState, *metalDSparkWeights) (*metalDSparkNative, error) { return f, nil },
		verifier: func(*metalRequestState) (dsv4.SpeculativeVerifier, error) { return v, nil },
	})
	if err != nil || d == nil || s == nil || d.speculator != s {
		t.Fatalf("constructor did not retain outer owner: %v", err)
	}
	frontier := &metalLifecycleSnapshot{fail: true}
	v.frontier = frontier
	if err := d.Close(); !errors.Is(err, dsv4.ErrDesynchronized) || d.speculator != nil || d.verifier != v || d.forward != f || f.closed {
		t.Fatalf("raw verifier failure released support: %v", err)
	}
	frontier.fail = false
	if err := d.Close(); err != nil || d.verifier != nil || d.forward != nil || !f.closed {
		t.Fatalf("raw verifier retry: %v", err)
	}
	if _, err := s.Verify(0, 0, 1); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("constructor-owned speculator resumed: %v", err)
	}
}
