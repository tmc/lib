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
	mu        sync.Mutex
	destroyed bool
}

// Raw returns the underlying driver.Event handle.
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
	if err := e.device.WithContext(func() error { return e.raw.Record(s.raw) }); err != nil {
		return fmt.Errorf("cuda: record event: %w", err)
	}
	return nil
}

// Query reports whether all captured work preceding this event has completed.
// It returns true if the event has been completed, or false if work is still in progress.
func (e *Event) Query() (bool, error) {
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
		done, err := e.Query()
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

// ElapsedSince computes the elapsed duration between the start event and this ending event.
// Both events must have completed before ElapsedSince is called.
func (e *Event) ElapsedSince(start *Event) (time.Duration, error) {
	if start == nil {
		return 0, errors.New("cuda: nil start event passed to ElapsedSince")
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

// Done returns a channel that is closed once the event completes execution on the GPU.
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

// Destroy destroys the event. Repeated calls are safe.
func (e *Event) Destroy() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.destroyed {
		return nil
	}
	if err := e.device.WithContext(e.raw.Destroy); err != nil {
		return fmt.Errorf("cuda: destroy event: %w", err)
	}
	e.destroyed = true
	return nil
}
