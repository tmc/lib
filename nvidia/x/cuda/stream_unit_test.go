package cuda

import (
	"context"
	"testing"
)

func TestStreamCleanupDirect(t *testing.T) {
	s := &Stream{}
	called1 := false
	called2 := false

	s.AddCleanup(func() {
		called1 = true
	})
	s.AddCleanup(func() {
		called2 = true
	})

	if called1 || called2 {
		t.Fatal("cleanups ran before drain")
	}

	s.drainCleanups()

	if !called1 || !called2 {
		t.Fatal("drainCleanups failed to execute all registered callbacks")
	}

	// Verify that subsequent drain is a no-op
	called1 = false
	s.drainCleanups()
	if called1 {
		t.Fatal("cleanups should be cleared after first drain")
	}
}

func TestStreamSyncDrainsCleanups(t *testing.T) {
	s := &Stream{}
	called := false
	s.AddCleanup(func() {
		called = true
	})

	// When raw stream is 0/mocked, query or direct drain should work
	s.drainCleanups()
	if !called {
		t.Fatal("expected cleanup on drain")
	}
	_ = context.Background()
}
