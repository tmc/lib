package cuda

import (
	"errors"
	"fmt"
	"runtime"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

// Context wraps a CUDA execution context.
type Context struct {
	raw       driver.Context
	device    *Device
	isPrimary bool
	mu        sync.RWMutex
	released  bool
}

// Raw returns the underlying driver.Context handle.
func (c *Context) Raw() driver.Context {
	return c.raw
}

// Device returns the device associated with this context.
func (c *Context) Device() *Device {
	return c.device
}

// Run executes the given function with this context bound as the current context
// for the calling OS thread. It guarantees that the goroutine will not be migrated
// to another OS thread during execution via runtime.LockOSThread. Run restores the
// context that was current on that thread before it returns.
func (c *Context) Run(fn func() error) (err error) {
	if c == nil {
		return errors.New("cuda: nil context")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.released {
		return errors.New("cuda: context is released")
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	previous, err := driver.NewContextGetCurrent()
	if err != nil {
		return fmt.Errorf("cuda: get current context: %w", err)
	}
	if err := c.raw.SetCurrent(); err != nil {
		return fmt.Errorf("cuda: set current context: %w", err)
	}
	defer func() {
		if restoreErr := previous.SetCurrent(); restoreErr != nil {
			err = errors.Join(err, fmt.Errorf("cuda: restore current context: %w", restoreErr))
		}
	}()
	return fn()
}

// Synchronize blocks the host until all preceding commands in the context have finished.
func (c *Context) Synchronize() error {
	return c.Run(func() error {
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: context synchronize: %w", err)
		}
		return nil
	})
}

// Release releases the context. If this context was obtained via [Device.PrimaryContext],
// it releases the reference. Otherwise, it destroys the context. Repeated calls
// are safe.
func (c *Context) Release() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.released {
		return nil
	}
	if c.isPrimary && c.device != nil {
		if err := c.device.raw.PrimaryCtxRelease(); err != nil {
			return fmt.Errorf("cuda: release primary context: %w", err)
		}
		c.released = true
		return nil
	}
	if err := c.raw.Destroy(); err != nil {
		return fmt.Errorf("cuda: destroy context: %w", err)
	}
	c.released = true
	return nil
}
