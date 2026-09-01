package cuda

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/driver"
)

// Stream represents an asynchronous sequence of GPU commands.
type Stream struct {
	raw      driver.Stream
	device   *Device
	mu       sync.Mutex
	cleanups []func()
}

// Raw returns the underlying driver.Stream handle.
func (s *Stream) Raw() driver.Stream {
	return s.raw
}

// Device returns the device on which this stream was created.
func (s *Stream) Device() *Device {
	return s.device
}

// AddCleanup registers a host callback to be invoked once the stream is synchronized or destroyed.
// This is used to tie resource lifetimes (such as host memory pins) to stream completion.
func (s *Stream) AddCleanup(fn func()) {
	if fn == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanups = append(s.cleanups, fn)
}

func (s *Stream) drainCleanups() {
	s.mu.Lock()
	cleanups := s.cleanups
	s.cleanups = nil
	s.mu.Unlock()

	for _, fn := range cleanups {
		fn()
	}
}

// Query reports whether all previously enqueued operations on the stream have completed.
// It returns true if the stream is idle, or false if work is still pending.
func (s *Stream) Query() (bool, error) {
	err := s.raw.Query()
	if err == nil {
		return true, nil
	}
	if errors.Is(err, driver.ErrNotReady) {
		return false, nil
	}
	return false, fmt.Errorf("cuda: query stream: %w", err)
}

// Synchronize blocks until all operations enqueued on the stream have finished execution.
// If ctx is cancelled or times out before the operations complete, Synchronize returns
// the context error immediately. Note that GPU hardware execution continues until the
// active kernel finishes.
func (s *Stream) Synchronize(ctx context.Context) error {
	if ctx == nil || ctx.Done() == nil {
		if err := s.raw.Synchronize(); err != nil {
			return fmt.Errorf("cuda: stream synchronize: %w", err)
		}
		s.drainCleanups()
		return nil
	}

	// Check if already cancelled
	if err := ctx.Err(); err != nil {
		return err
	}

	// For context-aware synchronization, poll stream status with backoff
	backoff := 50 * time.Microsecond
	maxBackoff := 5 * time.Millisecond

	for {
		idle, err := s.Query()
		if err != nil {
			return err
		}
		if idle {
			s.drainCleanups()
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
		}
	}
}

// WaitEvent makes all future work submitted to the stream wait until the specified event has been recorded.
func (s *Stream) WaitEvent(e *Event) error {
	if e == nil {
		return errors.New("cuda: nil event passed to WaitEvent")
	}
	if err := s.raw.WaitEvent(e.raw, 0); err != nil {
		return fmt.Errorf("cuda: stream wait event: %w", err)
	}
	return nil
}

// Destroy destroys the stream. Any unfinished work will continue to execute on the GPU,
// but the stream handle becomes invalid for new command submissions.
func (s *Stream) Destroy() error {
	defer s.drainCleanups()
	if err := s.raw.Destroy(); err != nil {
		return fmt.Errorf("cuda: stream destroy: %w", err)
	}
	return nil
}
