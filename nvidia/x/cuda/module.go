package cuda

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// Module owns a driver module and a retained primary context. Its zero value
// is closed. A Module must not be copied after first use.
type Module struct {
	mu  sync.Mutex
	raw driver.Module
	ctx *Context
}

// OwnModule takes ownership of raw on success. raw must have been loaded in
// dev's primary context and must not subsequently be unloaded by the caller.
// On failure, ownership remains with the caller. The retained context lets the
// module outlive dev; Close releases that reference after unloading the module.
func OwnModule(dev *Device, raw driver.Module) (*Module, error) {
	if dev == nil || raw == 0 {
		return nil, fmt.Errorf("cuda: nil device or module")
	}
	dev.mu.RLock()
	defer dev.mu.RUnlock()
	if dev.ctx == nil {
		return nil, fmt.Errorf("cuda: closed device")
	}
	ctx, err := dev.raw.PrimaryCtxRetain()
	if err != nil {
		return nil, fmt.Errorf("cuda: retain module context: %w", err)
	}
	return &Module{raw: raw, ctx: &Context{raw: ctx, device: dev, isPrimary: true}}, nil
}

// Close waits for work in the module's context, unloads the module, and releases
// its context reference. It is safe to repeat and concurrent with Function and
// Kernel.Launch. Failed cleanup retains the remaining resources for retry.
func (m *Module) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	var err error
	if m.raw != 0 {
		err = m.ctx.Run(func() error {
			if err := driver.CtxSynchronize(); err != nil {
				return err
			}
			if err := m.raw.Unload(); err != nil {
				return err
			}
			m.raw = 0
			return nil
		})
		if m.raw != 0 {
			return fmt.Errorf("cuda: close module: %w", err)
		}
	}
	if m.ctx != nil {
		releaseErr := m.ctx.Release()
		if releaseErr == nil {
			m.ctx = nil
		}
		err = errors.Join(err, releaseErr)
	}
	if err != nil {
		return fmt.Errorf("cuda: close module: %w", err)
	}
	return nil
}

// Kernel is a function whose module lifetime is checked on every launch.
// Its zero value cannot be launched.
type Kernel struct {
	module *Module
	raw    driver.Function
}

// Function resolves a kernel and retains its module owner. Names must be
// nonempty and contain no NUL byte. Closing the module invalidates its kernels.
func (m *Module) Function(name string) (*Kernel, error) {
	if m == nil {
		return nil, fmt.Errorf("cuda: closed module")
	}
	if name == "" || strings.ContainsRune(name, 0) {
		return nil, fmt.Errorf("cuda: invalid kernel name")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.raw == 0 {
		return nil, fmt.Errorf("cuda: closed module")
	}
	var fn driver.Function
	err := m.ctx.Run(func() error {
		var err error
		fn, err = m.raw.GetFunction(name)
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("cuda: find kernel %q: %w", name, err)
	}
	return &Kernel{module: m, raw: fn}, nil
}

// Launch submits a kernel with grid and block dimensions in x, y, z order.
// stream and device arguments must belong to the module's context. Argument
// values are copied during the call; device allocations must remain live until
// execution finishes. Closing the module waits for that execution.
func (k *Kernel) Launch(grid, block [3]uint32, sharedBytes uint32, stream driver.Stream, args []unsafe.Pointer) error {
	if k == nil || k.module == nil {
		return fmt.Errorf("cuda: closed module")
	}
	m := k.module
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.raw == 0 {
		return fmt.Errorf("cuda: closed module")
	}
	for i := range grid {
		if grid[i] == 0 || block[i] == 0 {
			return fmt.Errorf("cuda: zero launch dimension")
		}
	}
	if err := m.ctx.Run(func() error {
		return k.raw.LaunchKernel(grid[0], grid[1], grid[2], block[0], block[1], block[2], sharedBytes, stream, args)
	}); err != nil {
		return fmt.Errorf("cuda: launch kernel: %w", err)
	}
	return nil
}
