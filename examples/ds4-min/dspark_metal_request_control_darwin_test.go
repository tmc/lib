//go:build darwin && arm64

package main

import (
	"errors"
	"testing"
)

type requestRetryCloser struct {
	err   error
	calls int
}

func (c *requestRetryCloser) Close() error {
	c.calls++
	return c.err
}

func TestMetalDSparkRequestOwnerRetry(t *testing.T) {
	failure := errors.New("retained constructor owner")
	pending := &requestRetryCloser{err: failure}
	f := &requestHeadOwner{pending: pending, target: &metalRuntime{borrowed: true}}
	if err := f.Close(); !errors.Is(err, failure) {
		t.Fatalf("first close: %v", err)
	}
	if f.closed || f.pending != pending || f.target == nil || pending.calls != 1 {
		t.Fatal("failed cleanup released owner or downstream request")
	}
	pending.err = nil
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if !f.closed || f.pending != nil || f.target != nil || pending.calls != 2 {
		t.Fatal("retry failed to finish cleanup")
	}
	if err := f.Close(); err != nil || pending.calls != 2 {
		t.Fatal("close not idempotent")
	}
}

func TestMetalDSparkRequestReadoutControls(t *testing.T) {
	want := [8]float64{2, 0, 0, 0, .5, .5, 0, 0}
	if changed, _ := requestHeadMismatch(want, want); changed != 0 {
		t.Fatal("baseline comparator")
	}
	for field := range want {
		got := want
		got[field]++
		changed, fields := requestHeadMismatch(got, want)
		if changed != 1 || !fields[field] {
			t.Fatalf("field %d omitted from denominator", field)
		}
	}
	seen := make(map[string]bool)
	for _, arm := range requestHeadArms() {
		if seen[arm.Name] || arm.Name == "" {
			t.Fatal("duplicate arm")
		}
		seen[arm.Name] = true
	}
	if len(seen) != 8 {
		t.Fatal("eight records required")
	}
}
