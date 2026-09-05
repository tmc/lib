package cudamem

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
)

// Completion owns an asynchronous transfer's completion event and allocation
// leases. Call Wait or Close to release its event. Stream synchronization also
// releases allocation leases, but does not destroy the event. Copies share the
// same state. The zero value represents a completed transfer.
type Completion struct{ state *transferState }

type transferState struct {
	mu      sync.Mutex
	wait    func(context.Context) error
	destroy func() error
	release func()
	once    sync.Once
	done    atomic.Bool
}

func (s *transferState) complete() {
	s.once.Do(func() { s.release(); s.done.Store(true) })
}

// Wait waits for the transfer and releases its resources. Cancellation stops
// only the wait: it does not cancel CUDA work or release in-flight allocations.
// Native wait errors likewise retain allocations. A cleanup error can be retried
// by calling Wait or Close. A nil context waits without cancellation.
func (c *Completion) Wait(ctx context.Context) error {
	if c == nil || c.state == nil {
		return nil
	}
	s := c.state
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.done.Load() {
		if err := s.wait(ctx); err != nil {
			return fmt.Errorf("cudamem: wait for transfer: %w", err)
		}
		s.complete()
	}
	if s.destroy != nil {
		if err := s.destroy(); err != nil {
			return fmt.Errorf("cudamem: destroy transfer event: %w", err)
		}
		s.destroy = nil
	}
	return nil
}

// Close waits without cancellation and releases the completion event. It is
// safe to call repeatedly or concurrently with Wait.
func (c *Completion) Close() error { return c.Wait(nil) }
