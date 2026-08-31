package cuda

import (
	"fmt"
	"runtime"

	"github.com/tmc/lib/nvidia/driver"
)

// Context wraps a CUDA execution context.
type Context struct {
	raw       driver.Context
	device    *Device
	isPrimary bool
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
// to another OS thread during execution via runtime.LockOSThread.
func (c *Context) Run(fn func() error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := c.raw.SetCurrent(); err != nil {
		return fmt.Errorf("cuda: set current context: %w", err)
	}
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
// it releases the reference. Otherwise, it destroys the context.
func (c *Context) Release() error {
	if c.isPrimary && c.device != nil {
		if err := c.device.raw.PrimaryCtxRelease(); err != nil {
			return fmt.Errorf("cuda: release primary context: %w", err)
		}
		return nil
	}
	if err := c.raw.Destroy(); err != nil {
		return fmt.Errorf("cuda: destroy context: %w", err)
	}
	return nil
}
