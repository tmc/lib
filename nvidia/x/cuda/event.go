package cuda

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/driver"
)

// Event represents a marker in a CUDA stream used for synchronization and timing.
type Event struct {
	raw       driver.Event
	device    *Device
	mu        sync.RWMutex
	destroyed bool
}

// Raw returns the underlying driver.Event handle. The handle is invalid after
// Close. Callers using it directly must serialize that use with Close;
// wrapper methods do this automatically.
func (e *Event) Raw() driver.Event {
	return e.raw
}

// Device returns the device on which this event was created.
func (e *Event) Device() *Device {
	return e.device
}

// Record captures the current execution progress of the given stream into the event.
func (e *Event) Record(s *Stream) error {
	if s == nil {
		return errors.New("cuda: nil stream passed to Record")
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
	if err := e.device.WithContext(func() error { return e.raw.Record(s.raw) }); err != nil {
		return fmt.Errorf("cuda: record event: %w", err)
	}
	return nil
}

// Query reports whether all captured work preceding this event has completed.
// It returns true if the event has been completed, or false if work is still in progress.
func (e *Event) Query() (bool, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.destroyed {
		return false, errors.New("cuda: event is destroyed")
	}
	return e.query()
}

func (e *Event) query() (bool, error) {
	var queryErr error
	err := e.device.WithContext(func() error {
		queryErr = e.raw.Query()
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
	return false, fmt.Errorf("cuda: query event: %w", err)
}

// Synchronize blocks the host until all captured work in this event has completed.
func (e *Event) Synchronize(ctx context.Context) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.destroyed {
		return errors.New("cuda: event is destroyed")
	}
	if ctx == nil || ctx.Done() == nil {
		if err := e.device.WithContext(e.raw.Synchronize); err != nil {
			return fmt.Errorf("cuda: event synchronize: %w", err)
		}
		return nil
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	backoff := 50 * time.Microsecond
	maxBackoff := 5 * time.Millisecond

	for {
		done, err := e.query()
		if err != nil {
			return err
		}
		if done {
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

// eventPairMu serializes acquisition of two event read locks. It is never held
// while calling into a device context or the driver.
var eventPairMu sync.Mutex

// ElapsedSince computes the elapsed duration between the start event and this ending event.
// Both events must have completed before ElapsedSince is called.
func (e *Event) ElapsedSince(start *Event) (time.Duration, error) {
	if start == nil {
		return 0, errors.New("cuda: nil start event passed to ElapsedSince")
	}
	// Serialize pair acquisition so a pending writer cannot leave opposite
	// pairs holding one read lock each. No handle is read before both locks.
	eventPairMu.Lock()
	e.mu.RLock()
	if start != e {
		start.mu.RLock()
	}
	eventPairMu.Unlock()
	defer e.mu.RUnlock()
	if start != e {
		defer start.mu.RUnlock()
	}
	if e.destroyed || start.destroyed {
		return 0, errors.New("cuda: event is destroyed")
	}
	var ms float32
	err := e.device.WithContext(func() error {
		var err error
		ms, err = start.raw.ElapsedTime(e.raw)
		return err
	})
	if err != nil {
		return 0, fmt.Errorf("cuda: event elapsed time: %w", err)
	}
	// driver returns float32 in milliseconds
	return time.Duration(float64(ms) * float64(time.Millisecond)), nil
}

// Done returns a new channel that closes when Query reports completion or an
// error, including an error caused by Close or Device.Close. A closed channel
// does not establish successful GPU completion. Use Synchronize when the result
// or cancellation matters.
//
// Each call starts a polling goroutine that runs until Query reports completion
// or an error, even if the caller stops receiving. Record the event before calling
// Done, and do not record it again until the channel closes. Done does not capture
// a particular recording, and a closed channel stays closed if the event is reused.
func (e *Event) Done() <-chan struct{} {
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		backoff := 50 * time.Microsecond
		maxBackoff := 5 * time.Millisecond
		for {
			done, err := e.Query()
			if err != nil || done {
				return
			}
			time.Sleep(backoff)
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
		}
	}()
	return ch
}

// Close destroys the event. Repeated calls are safe. If destruction succeeds
// but context restoration fails, Close returns the error and the event remains
// destroyed.
func (e *Event) Close() error {
	return e.closeWith(e.device.WithContext, func() error { return e.raw.Close() })
}

// closeWith records successful destruction before context restoration can fail.
func (e *Event) closeWith(run func(func() error) error, destroy func() error) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.destroyed {
		return nil
	}
	if err := run(func() error {
		if err := destroy(); err != nil {
			return err
		}
		e.destroyed = true
		return nil
	}); err != nil {
		return fmt.Errorf("cuda: destroy event: %w", err)
	}
	return nil
}
