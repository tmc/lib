package cudaplan

import (
	"context"
	"errors"
	"testing"
)

func TestCloseDrainsFailedInvocation(t *testing.T) {
	p := &ExecutablePlan{inFlight: true}
	attempts, releases := 0, 0
	p.failedInvocation = newHostCompletion(func(context.Context) error {
		// A drain must not hold the plan lock.
		if !p.mu.TryLock() {
			t.Fatal("drain holds plan lock")
		}
		p.mu.Unlock()
		attempts++
		if attempts == 1 {
			return errors.New("stream unavailable")
		}
		return nil
	}, nil, func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		releases++
		p.inFlight = false
	})
	if err := p.Close(); err == nil || p.closed || !p.inFlight || releases != 0 {
		t.Fatalf("failed drain lost ownership: err=%v closed=%v inFlight=%v releases=%d", err, p.closed, p.inFlight, releases)
	}
	if err := p.Close(); err != nil || !p.closed || p.inFlight || releases != 1 {
		t.Fatalf("successful retry: err=%v closed=%v inFlight=%v releases=%d", err, p.closed, p.inFlight, releases)
	}
	if err := p.Close(); err != nil || releases != 1 {
		t.Fatalf("repeated close: err=%v releases=%d", err, releases)
	}
}
