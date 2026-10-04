package cudalayer

import (
	"errors"
	"io"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

// mlpCleanup retains every unresolved resource. Completion must be established
// before any device free or module unload. A failed device remains quarantined:
// Close can be retried, but cannot manufacture successful completion.
type mlpCleanup struct {
	mu          sync.Mutex
	synchronize func() error
	inFlight    bool
	frees       []func() error
	closers     []io.Closer
}

// Commit native ownership changes inside the context callback. The wrapper
// can report restoration failure after a successful free, so its return value
// alone cannot determine whether retrying MemFree is safe.
func mlpFreeBuffer(pointer driver.DevicePtr, run func(func() error) error, free func(driver.DevicePtr) error) func() error {
	pending := pointer
	return func() error {
		if pending == 0 {
			return nil
		}
		return run(func() error {
			if err := free(pending); err != nil {
				return err
			}
			pending = 0
			return nil
		})
	}
}

func (c *mlpCleanup) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.inFlight {
		if err := c.synchronize(); err != nil {
			return err
		}
		c.inFlight = false
	}
	var errs []error
	for i, f := range c.frees {
		if f == nil {
			continue
		}
		if err := f(); err != nil {
			errs = append(errs, err)
		} else {
			c.frees[i] = nil
		}
	}
	for i, closer := range c.closers {
		if closer == nil {
			continue
		}
		if err := closer.Close(); err != nil {
			errs = append(errs, err)
		} else {
			c.closers[i] = nil
		}
	}
	return errors.Join(errs...)
}

type mlpCleanupError struct {
	cause error
	owner *mlpCleanup
}

func (e *mlpCleanupError) Error() string { return e.cause.Error() }
func (e *mlpCleanupError) Unwrap() error { return e.cause }
func (e *mlpCleanupError) Close() error  { return e.owner.Close() }

// Preserve cleanup ownership returned by the compiler/loader without trying
// compilation or loading again. Callers discover the enclosing owner with As.
func mlpAdoptCleanup(err error) error {
	var closer io.Closer
	if errors.As(err, &closer) {
		return &mlpCleanupError{cause: err, owner: &mlpCleanup{closers: []io.Closer{closer}}}
	}
	return err
}
