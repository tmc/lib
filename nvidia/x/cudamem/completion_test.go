package cudamem

import (
	"context"
	"errors"
	"sync"
	"testing"
)

func TestCompletionCancellationRetainsResources(t *testing.T) {
	waits, releases, destroys := 0, 0, 0
	s := &transferState{
		wait: func(ctx context.Context) error {
			waits++
			if ctx != nil {
				return ctx.Err()
			}
			return nil
		},
		release: func() { releases++ },
		destroy: func() error { destroys++; return nil },
	}
	c := &Completion{state: s}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := c.Wait(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("Wait = %v", err)
	}
	if releases != 0 || destroys != 0 {
		t.Fatal("released resources on cancellation")
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if waits != 2 || releases != 1 || destroys != 1 {
		t.Fatalf("waits=%d releases=%d destroys=%d", waits, releases, destroys)
	}
}

func TestCompletionRetry(t *testing.T) {
	failure := errors.New("injected")
	for _, stage := range []string{"wait", "destroy"} {
		t.Run(stage, func(t *testing.T) {
			fail := true
			releases := 0
			s := &transferState{
				wait: func(context.Context) error {
					if fail && stage == "wait" {
						return failure
					}
					return nil
				},
				release: func() { releases++ },
				destroy: func() error {
					if fail && stage == "destroy" {
						return failure
					}
					return nil
				},
			}
			c := &Completion{state: s}
			if err := c.Close(); !errors.Is(err, failure) {
				t.Fatalf("Close = %v", err)
			}
			if stage == "wait" && releases != 0 {
				t.Fatal("released in-flight resources")
			}
			if s.destroy == nil {
				t.Fatal("lost cleanup on failure")
			}
			fail = false
			if err := c.Close(); err != nil {
				t.Fatal(err)
			}
			if releases != 1 || s.destroy != nil {
				t.Fatal("retry did not clean up once")
			}
		})
	}
}

func TestCompletionConcurrentAliases(t *testing.T) {
	releases, destroys := 0, 0
	s := &transferState{
		wait:    func(context.Context) error { return nil },
		release: func() { releases++ },
		destroy: func() error { destroys++; return nil },
	}
	c := Completion{state: s}
	alias := c
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(s.complete) // Simulate stream cleanup racing an event wait.
		wg.Go(func() {
			if err := alias.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if releases != 1 || destroys != 1 {
		t.Fatalf("releases=%d destroys=%d", releases, destroys)
	}
}

func TestHostTransferLease(t *testing.T) {
	a := &hostAllocation{ptr: 1}
	h := HostAllocation[byte]{ptr: 1, alloc: a}
	release, err := h.retain()
	if err != nil {
		t.Fatal(err)
	}
	alias := h
	if err := alias.Free(); !errors.Is(err, ErrInUse) {
		t.Fatalf("Free = %v", err)
	}
	release()
	release()
	if a.users != 0 {
		t.Fatalf("users = %d", a.users)
	}
	a.ptr = 0 // Do not pass the synthetic address to CUDA.
	if _, err := h.retain(); err == nil {
		t.Fatal("retained freed host memory")
	}
}
