//go:build darwin && arm64

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type metalLifecycleSnapshot struct {
	fail  bool
	calls int
}

func (s *metalLifecycleSnapshot) Close() error {
	s.calls++
	if s.fail {
		return errors.New("release fault")
	}
	return nil
}

func metalLifecycleVerifier() *metalSpeculativeVerifier {
	a := &metalStateAdapter{}
	s := &metalRequestState{owner: a, runtime: &metalRuntime{speculativeEpoch: 7, pos: 3}}
	return &metalSpeculativeVerifier{adapter: a, state: s}
}

func TestMetalLifecycleCleanupRetry(t *testing.T) {
	v := metalLifecycleVerifier()
	frontier := &metalLifecycleSnapshot{fail: true}
	v.frontier = frontier
	c, ok := any(v).(interface{ Close() error })
	if !ok {
		t.Fatal("verifier has no retryable cleanup owner")
	}
	if err := c.Close(); err == nil || v.frontier != frontier {
		t.Fatal("failed release lost owner")
	}
	if _, err := v.VerifyBatch(1, []int{2}, 3); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("closed verifier resumed: %v", err)
	}
	frontier.fail = false
	if err := c.Close(); err != nil || v.frontier != nil || frontier.calls != 2 {
		t.Fatalf("retry failed: %v", err)
	}
}

func TestMetalLifecycleForeignCommit(t *testing.T) {
	v := metalLifecycleVerifier()
	frontier := &metalLifecycleSnapshot{}
	v.frontier, v.drafts, v.pos = frontier, []int{2}, 2
	v.rowEpoch = 6 // Another state transition replaced the verified frontier.
	if err := v.CommitPrefix(1); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("foreign transition admitted: %v", err)
	}
	if v.frontier != frontier || frontier.calls != 0 || v.state.runtime.speculativeEpoch != 7 {
		t.Fatal("foreign commit mutated or released state")
	}
}

func TestMetalLifecycleRollbackDesynchronized(t *testing.T) {
	v := metalLifecycleVerifier()
	v.frontier = &metalLifecycleSnapshot{fail: true}
	fault := errors.New("execution fault")
	err := v.abort(fault)
	if !errors.Is(err, fault) || !errors.Is(err, dsv4.ErrDesynchronized) || v.frontier == nil {
		t.Fatalf("rollback failure lost poison or owner: %v", err)
	}
}

func TestMetalLifecycleClosedEmptyBatch(t *testing.T) {
	a := &metalStateAdapter{closed: true}
	s := &metalRequestState{owner: a, runtime: &metalRuntime{}}
	if err := a.EvalBatch(s, nil, 0); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("closed empty batch: %v", err)
	}
}

func TestMetalLifecycleForeignSnapshot(t *testing.T) {
	v := metalLifecycleVerifier()
	foreign := &metalSnapshot{owner: v.adapter, request: &metalRequestState{owner: v.adapter}}
	if err := v.state.Restore(foreign); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("snapshot without request identity admitted: %v", err)
	}
}

func TestMetalLifecycleReplacementRelease(t *testing.T) {
	v := metalLifecycleVerifier()
	frontier := &metalLifecycleSnapshot{fail: true}
	v.frontier = frontier
	if _, err := v.VerifyBatch(1, []int{2}, 3); !errors.Is(err, dsv4.ErrDesynchronized) || v.frontier != frontier {
		t.Fatalf("replacement lost release owner: %v", err)
	}
	if v.CurrentLogits() != nil || (&metalSampledVerifier{v}).BoundTo(v.state) {
		t.Fatal("failed cleanup exposed usable verifier")
	}
	if _, err := v.VerifyBatch(1, []int{2}, 3); !errors.Is(err, dsv4.ErrDesynchronized) || frontier.calls != 1 {
		t.Fatal("failed cleanup resumed native verification")
	}
	frontier.fail = false
	if err := v.Close(); err != nil || frontier.calls != 2 {
		t.Fatalf("replacement retry: %v", err)
	}
}
