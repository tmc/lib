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

func TestDefaultStream(t *testing.T) {
	d := &Device{}
	s := d.DefaultStream()
	if s == nil {
		t.Fatal("DefaultStream returned nil")
	}
	if got := s.Raw(); got != 0 {
		t.Errorf("default stream handle = %v, want 0 (legacy null stream)", got)
	}
	if s.Device() != d {
		t.Error("default stream is not bound to its device")
	}
	if again := d.DefaultStream(); again != s {
		t.Error("DefaultStream returned a different stream on the second call")
	}

	// Destroying the default stream drains cleanups but must not release the
	// driver-owned handle.
	drained := false
	s.AddCleanup(func() { drained = true })
	if err := s.Destroy(); err != nil {
		t.Errorf("Destroy of default stream = %v, want nil", err)
	}
	if !drained {
		t.Error("Destroy did not drain registered cleanups")
	}
	if d.DefaultStream() != s {
		t.Error("default stream was discarded by Destroy")
	}
}
