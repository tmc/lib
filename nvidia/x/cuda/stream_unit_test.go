package cuda

import (
	"context"
	"errors"
	"testing"
	"time"
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

func TestStreamSynchronizeCanceledRetainsCleanups(t *testing.T) {
	for _, tt := range []struct {
		name     string
		deadline bool
		want     error
	}{
		{"canceled", false, context.Canceled},
		{"deadline", true, context.DeadlineExceeded},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			if tt.deadline {
				cancel()
				ctx, cancel = context.WithDeadline(context.Background(), time.Unix(0, 0))
			} else {
				cancel()
			}
			defer cancel()
			s := &Stream{}
			calls := 0
			s.AddCleanup(func() { calls++ })
			if err := s.Synchronize(ctx); !errors.Is(err, tt.want) {
				t.Fatalf("Synchronize = %v, want %v", err, tt.want)
			}
			if calls != 0 || len(s.cleanups) != 1 {
				t.Fatalf("cleanup calls = %d, retained = %d", calls, len(s.cleanups))
			}
			if !s.lifeMu.TryLock() {
				t.Fatal("Synchronize retained lifecycle lock")
			}
			s.lifeMu.Unlock()
			s.drainCleanups()
			s.drainCleanups()
			if calls != 1 {
				t.Fatalf("cleanup calls after drain = %d, want 1", calls)
			}
		})
	}
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

	// The zero Device used by this unit test has no CUDA context, so exercise
	// the cleanup state directly rather than pretending a physical stream
	// synchronization succeeded.
	drained := false
	s.AddCleanup(func() { drained = true })
	s.drainCleanups()
	if !drained {
		t.Error("Close did not drain registered cleanups")
	}
	if d.DefaultStream() != s {
		t.Error("default stream was discarded by Close")
	}
}
