package cuptipcsampling

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestSamplingOwnershipBeforeNative(t *testing.T) {
	a, _ := testSamplingSession(t)
	b, _ := testSamplingSession(t)
	if err := a.begin(); err != nil {
		t.Fatal(err)
	}
	b.control.api.version = func() (uint32, error) { t.Fatal("losing session called native"); return 0, nil }
	if err := b.begin(); err == nil {
		t.Fatal("two owners admitted")
	}
	if _, err := b.finish(true); err != nil {
		t.Fatal(err)
	}
	samplingOwner.Lock()
	owner := samplingOwner.session
	samplingOwner.Unlock()
	if owner != a {
		t.Fatal("loser released winner ownership")
	}
	if _, err := a.finish(true); err != nil {
		t.Fatal(err)
	}
}

func TestSamplingOwnershipRetainedAfterDisableFailure(t *testing.T) {
	a, _ := testSamplingSession(t)
	b, _ := testSamplingSession(t)
	if err := a.begin(); err != nil {
		t.Fatal(err)
	}
	a.control.api.disable = func(*cupti.PCSamplingDisableParams) error { return errors.New("disable failure") }
	if _, err := a.finish(true); err == nil {
		t.Fatal("disable succeeded unexpectedly")
	}
	if err := b.begin(); err == nil {
		t.Fatal("ownership lost on failed disable")
	}
	a.control.api.disable = func(*cupti.PCSamplingDisableParams) error { return nil }
	_, _ = a.finish(true)
	if err := b.begin(); err != nil {
		t.Fatal(err)
	}
	if _, err := b.finish(true); err != nil {
		t.Fatal(err)
	}
}

func TestSamplingConcurrentOwnership(t *testing.T) {
	a, _ := testSamplingSession(t)
	b, _ := testSamplingSession(t)
	type result struct {
		s   *samplingSession
		err error
	}
	ready := make(chan struct{})
	results := make(chan result, 2)
	for _, s := range []*samplingSession{a, b} {
		go func() { <-ready; results <- result{s, s.begin()} }()
	}
	close(ready)
	x, y := <-results, <-results
	if (x.err == nil) == (y.err == nil) {
		t.Fatalf("expected one owner: %v, %v", x.err, y.err)
	}
	for _, r := range []result{x, y} {
		if _, err := r.s.finish(true); err != nil {
			t.Fatal(err)
		}
	}
}
