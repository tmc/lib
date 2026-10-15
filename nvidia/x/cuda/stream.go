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
	raw       driver.Stream
	device    *Device
	isDefault bool
	mu        sync.Mutex
	cleanups  []func()
	lifeMu    sync.RWMutex
	destroyed bool
}

// Raw returns the underlying driver.Stream handle. The handle is invalid after
// Close. Callers using it directly must serialize that use with Close;
// wrapper methods do this automatically.
func (s *Stream) Raw() driver.Stream {
	return s.raw
}

// WithHandle calls fn with the live driver handle while excluding Close.
// It protects only the host-side call: resources referenced by asynchronous
// work must remain live until the stream completes. The callback must not call
// methods on s. A nil or zero-value Stream is invalid; the default stream
// returned by [Device.DefaultStream] may have handle zero.
func (s *Stream) WithHandle(fn func(driver.Stream) error) error {
	if s == nil {
		return errors.New("cuda: nil stream")
	}
	if fn == nil {
		return errors.New("cuda: nil stream handle function")
	}
	s.lifeMu.RLock()
	defer s.lifeMu.RUnlock()
	if s.destroyed {
		return errors.New("cuda: stream is destroyed")
	}
	if s.raw == 0 && !s.isDefault {
		return errors.New("cuda: invalid stream handle")
	}
	return fn(s.raw)
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
	s.lifeMu.RLock()
	if s.destroyed {
		s.lifeMu.RUnlock()
		fn()
		return
	}
	s.mu.Lock()
	s.cleanups = append(s.cleanups, fn)
	s.mu.Unlock()
	s.lifeMu.RUnlock()
}

func (s *Stream) drainCleanups() {
	for _, fn := range s.takeCleanups() {
		fn()
	}
}

func (s *Stream) takeCleanups() []func() {
	s.mu.Lock()
	cleanups := s.cleanups
	s.cleanups = nil
	s.mu.Unlock()
	return cleanups
}

// Query reports whether all previously enqueued operations on the stream have completed.
// It returns true if the stream is idle, or false if work is still pending.
func (s *Stream) Query() (bool, error) {
	s.lifeMu.RLock()
	defer s.lifeMu.RUnlock()
	if s.destroyed {
		return false, errors.New("cuda: stream is destroyed")
	}
	return s.query()
}

func (s *Stream) query() (bool, error) {
	var queryErr error
	err := s.device.WithContext(func() error {
		queryErr = s.raw.Query()
		return nil
	})
	if err != nil {
		return false, err
	}
	err = queryErr
	if err == nil {
		return true, nil
	}
	if errors.Is(err, driver.ErrNotReady) {
		return false, nil
	}
	return false, fmt.Errorf("cuda: query stream: %w", err)
}

// Synchronize waits for operations enqueued on the stream to finish, then runs
// registered cleanup callbacks. A nil context waits without cancellation.
//
// Cancellation stops polling and returns the context error; it does not cancel
// GPU work or run cleanup callbacks. Retained resources must remain live until a
// later successful Synchronize or Close. Cancellation cannot interrupt waiting
// for another stream operation's lifecycle lock, a driver call, or cleanup
// callbacks. If completion is observed before cancellation, Synchronize succeeds.
func (s *Stream) Synchronize(ctx context.Context) error {
	s.lifeMu.Lock()
	if s.destroyed {
		s.lifeMu.Unlock()
		return errors.New("cuda: stream is destroyed")
	}
	if ctx == nil || ctx.Done() == nil {
		if err := s.device.WithContext(s.raw.Synchronize); err != nil {
			s.lifeMu.Unlock()
			return fmt.Errorf("cuda: stream synchronize: %w", err)
		}
		cleanups := s.takeCleanups()
		s.lifeMu.Unlock()
		for _, fn := range cleanups {
			fn()
		}
		return nil
	}

	// Check if already cancelled
	if err := ctx.Err(); err != nil {
		s.lifeMu.Unlock()
		return err
	}

	// For context-aware synchronization, poll stream status with backoff
	backoff := 50 * time.Microsecond
	maxBackoff := 5 * time.Millisecond

	for {
		idle, err := s.query()
		if err != nil {
			s.lifeMu.Unlock()
			return err
		}
		if idle {
			cleanups := s.takeCleanups()
			s.lifeMu.Unlock()
			for _, fn := range cleanups {
				fn()
			}
			return nil
		}

		select {
		case <-ctx.Done():
			s.lifeMu.Unlock()
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
	s.lifeMu.RLock()
	defer s.lifeMu.RUnlock()
	e.mu.RLock()
	defer e.mu.RUnlock()
	if s.destroyed {
		return errors.New("cuda: stream is destroyed")
	}
	if e.destroyed {
		return errors.New("cuda: event is destroyed")
	}
	if err := s.device.WithContext(func() error { return s.raw.WaitEvent(e.raw, 0) }); err != nil {
		return fmt.Errorf("cuda: stream wait event: %w", err)
	}
	return nil
}

// Close waits for all in-flight work on the stream to complete before draining
// registered cleanups and destroying the stream handle. Closing the stream
// returned by [Device.DefaultStream] drains its cleanups but keeps the handle,
// which belongs to the driver rather than to the caller. Repeated calls are safe.
// If destruction succeeds but context restoration fails, Close returns the
// error after marking the stream destroyed and draining its cleanups.
func (s *Stream) Close() error {
	return s.closeWith(s.device.WithContext,
		func() error { return s.raw.Synchronize() },
		func() error { return s.raw.Close() })
}

// closeWith keeps operation results separate from context restoration errors.
// Callbacks are per-call so lifecycle tests need no driver or global hooks.
func (s *Stream) closeWith(run func(func() error) error, synchronize, destroy func() error) error {
	s.lifeMu.Lock()
	if s.destroyed {
		s.lifeMu.Unlock()
		return nil
	}
	// Do not release host pinners unless completion was established.
	if err := run(synchronize); err != nil {
		s.lifeMu.Unlock()
		return fmt.Errorf("cuda: stream synchronize before destroy: %w", err)
	}
	if s.isDefault {
		cleanups := s.takeCleanups()
		s.lifeMu.Unlock()
		for _, fn := range cleanups {
			fn()
		}
		return nil
	}
	err := run(func() error {
		if err := destroy(); err != nil {
			return err
		}
		s.destroyed = true
		return nil
	})
	if err != nil && !s.destroyed {
		s.lifeMu.Unlock()
		return fmt.Errorf("cuda: stream destroy: %w", err)
	}
	cleanups := s.takeCleanups()
	s.lifeMu.Unlock()
	for _, fn := range cleanups {
		fn()
	}
	if err != nil {
		return fmt.Errorf("cuda: stream destroy: %w", err)
	}
	return nil
}
