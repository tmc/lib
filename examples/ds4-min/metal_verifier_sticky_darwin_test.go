//go:build darwin && arm64

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestMetalLifecycleStickyReleaseFailure(t *testing.T) {
	v := metalLifecycleVerifier()
	frontier := &metalLifecycleSnapshot{fail: true}
	v.frontier = frontier
	if err := v.releaseFrontier(); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Fatalf("release failure not desynchronized: %v", err)
	}
	if _, err := v.Snapshot(); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Errorf("failed verifier admitted snapshot allocation: %v", err)
	}
	frontier.fail = false
	if err := v.releaseFrontier(); err != nil || v.frontier != nil || frontier.calls != 2 {
		t.Fatalf("cleanup retry failed: %v", err)
	}
	if _, err := v.VerifyBatch(1, []int{2}, 3); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Errorf("cleanup reopened execution: %v", err)
	}
	if err := v.CommitPrefix(0); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Errorf("cleanup reopened commit: %v", err)
	}
	if _, err := v.Snapshot(); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Errorf("cleanup reopened snapshots: %v", err)
	}
	if v.CurrentLogits() != nil || (&metalSampledVerifier{v}).BoundTo(v.state) {
		t.Error("cleanup reopened logits or sampled binding")
	}
	if err := v.Close(); err != nil || frontier.calls != 2 {
		t.Errorf("terminal cleanup not idempotent: %v", err)
	}
}
