package cudaplan

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestNewStreamCompletionRejectsMissingStream(t *testing.T) {
	for _, tt := range []struct {
		name   string
		stream *cuda.Stream
	}{
		{"nil", nil},
		{"unbound", new(cuda.Stream)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			c, err := NewStreamCompletion(tt.stream, func() { called = true })
			if c != nil || err == nil || called {
				t.Fatalf("completion=%v error=%v released=%v", c, err, called)
			}
		})
	}
}

func TestNewStreamCompletionRetainsOnNativeDrainError(t *testing.T) {
	// An uninitialized device has no context. Its default stream can be adopted
	// without native calls, but the real Synchronize must refuse to drain it.
	stream := new(cuda.Device).DefaultStream()
	released := false
	c, err := NewStreamCompletion(stream, func() { released = true })
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err := c.Close(); err == nil || c.Done() || released {
			t.Fatalf("error=%v done=%v released=%v", err, c.Done(), released)
		}
	}
}

func TestNewStreamCompletionOwnsCallbackSnapshot(t *testing.T) {
	var released atomic.Int32
	callbacks := []func(){func() { released.Add(1) }, nil}
	c, err := NewStreamCompletion(new(cuda.Device).DefaultStream(), callbacks...)
	if err != nil {
		t.Fatal(err)
	}
	callbacks[0] = func() { released.Add(100) }
	if c.Done() || len(c.ExecutedSteps()) != 0 || released.Load() != 0 {
		t.Fatal("adoption claimed completion or ran callbacks")
	}
	// Only the drain seam is controlled here; the public constructor's callback
	// ownership and the existing completion state remain the actual code path.
	cause := errors.New("drain unavailable")
	c.state.wait = func(ctx context.Context) error {
		if ctx != nil && ctx.Err() != nil {
			return ctx.Err()
		}
		return cause
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := c.Wait(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := c.Wait(context.Background()); !errors.Is(err, cause) {
		t.Fatal(err)
	}
	if c.Done() || released.Load() != 0 {
		t.Fatal("failed wait released borrowed resources")
	}
	c.state.wait = func(context.Context) error { return nil }
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := c.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if !c.Done() || released.Load() != 1 {
		t.Fatalf("done=%v releases=%d", c.Done(), released.Load())
	}
}
