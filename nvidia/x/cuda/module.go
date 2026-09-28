package cuda

import (
	"errors"
	"fmt"
	"math"
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
	return m.close(m.ctx.Run, driver.CtxSynchronize, m.raw.Close, m.ctx.Release)
}

// close releases resources in order. The caller holds m.mu.
func (m *Module) close(run func(func() error) error, synchronize, unload, release func() error) error {
	var err error
	if m.raw != 0 {
		err = run(func() error {
			if err := synchronize(); err != nil {
				return err
			}
			if err := unload(); err != nil {
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
		releaseErr := release()
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

// Device returns the device whose context owns the kernel.
func (k *Kernel) Device() *Device {
	if k == nil || k.module == nil {
		return nil
	}
	m := k.module
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ctx == nil {
		return nil
	}
	return m.ctx.device
}

// PotentialBlockSize suggests a block size with reasonable occupancy for a
// constant dynamic shared-memory allocation. It holds the module open while
// querying the driver.
func (k *Kernel) PotentialBlockSize(dynamicSMemBytes uint, blockSizeLimit int) (minGridSize, blockSize int, err error) {
	if blockSizeLimit < 0 || blockSizeLimit > math.MaxInt32 {
		return 0, 0, fmt.Errorf("cuda: invalid block size limit %d", blockSizeLimit)
	}
	err = k.run(func() error {
		grid, block, err := k.raw.MaxPotentialBlockSize(0, dynamicSMemBytes, int32(blockSizeLimit))
		minGridSize, blockSize = int(grid), int(block)
		return err
	})
	if err != nil {
		return 0, 0, fmt.Errorf("cuda: potential block size: %w", err)
	}
	return minGridSize, blockSize, nil
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
		fn, err = m.raw.Function(name)
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("cuda: find kernel %q: %w", name, err)
	}
	return &Kernel{module: m, raw: fn}, nil
}

// WithGlobal calls fn with a device global's address and byte size, in the
// module's context and while excluding Close. The address must not escape the
// call. The callback must not call methods on m.
func (m *Module) WithGlobal(name string, fn func(driver.DevicePtr, uint) error) error {
	if m == nil {
		return fmt.Errorf("cuda: closed module")
	}
	if name == "" || strings.ContainsRune(name, 0) {
		return fmt.Errorf("cuda: invalid global name")
	}
	if fn == nil {
		return fmt.Errorf("cuda: nil global function")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.raw == 0 {
		return fmt.Errorf("cuda: closed module")
	}
	return m.ctx.Run(func() error {
		ptr, size, err := m.raw.Global(name)
		if err != nil {
			return fmt.Errorf("cuda: find global %q: %w", name, err)
		}
		return fn(ptr, size)
	})
}

// Attribute queries an integer attribute of the kernel, such as its static
// shared-memory size or register count.
func (k *Kernel) Attribute(attrib driver.FunctionAttribute) (int, error) {
	var val int32
	err := k.run(func() error {
		var err error
		val, err = k.raw.Attribute(attrib)
		return err
	})
	if err != nil {
		return 0, fmt.Errorf("cuda: kernel get attribute %v: %w", attrib, err)
	}
	return int(val), nil
}

// SetAttribute sets an integer attribute of the kernel. The driver accepts
// only a few attributes here; raising
// [driver.FuncAttributeMaxDynamicSharedSizeBytes] above the default is the
// common use, and the device caps it at
// [driver.DevAttrMaxSharedMemoryPerBlockOptin] less the static allocation.
func (k *Kernel) SetAttribute(attrib driver.FunctionAttribute, value int) error {
	if value < 0 || value > math.MaxInt32 {
		return fmt.Errorf("cuda: kernel set attribute %v: value %d out of range", attrib, value)
	}
	if err := k.run(func() error { return k.raw.SetAttribute(attrib, int32(value)) }); err != nil {
		return fmt.Errorf("cuda: kernel set attribute %v: %w", attrib, err)
	}
	return nil
}

// run executes fn in the module's context while holding the module open.
func (k *Kernel) run(fn func() error) error {
	if k == nil || k.module == nil {
		return fmt.Errorf("cuda: closed module")
	}
	m := k.module
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.raw == 0 {
		return fmt.Errorf("cuda: closed module")
	}
	return m.ctx.Run(fn)
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
		return k.raw.LaunchKernel(driver.Dim3{X: grid[0], Y: grid[1], Z: grid[2]}, driver.Dim3{X: block[0], Y: block[1], Z: block[2]}, sharedBytes, stream, args)
	}); err != nil {
		return fmt.Errorf("cuda: launch kernel: %w", err)
	}
	return nil
}

// Raw returns the underlying driver.Function handle.
func (k *Kernel) Raw() driver.Function {
	if k == nil {
		return 0
	}
	return k.raw
}

// LaunchEx submits a kernel with extended launch configuration (such as launch attributes).
func (k *Kernel) LaunchEx(config *driver.LaunchConfig, args []unsafe.Pointer) error {
	if k == nil || k.module == nil {
		return fmt.Errorf("cuda: closed module")
	}
	m := k.module
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.raw == 0 {
		return fmt.Errorf("cuda: closed module")
	}
	if config == nil {
		return fmt.Errorf("cuda: nil launch config")
	}
	if config.GridDim.X == 0 || config.GridDim.Y == 0 || config.GridDim.Z == 0 ||
		config.BlockDim.X == 0 || config.BlockDim.Y == 0 || config.BlockDim.Z == 0 {
		return fmt.Errorf("cuda: zero launch dimension")
	}
	if err := m.ctx.Run(func() error {
		return k.raw.LaunchKernelEx(config, args)
	}); err != nil {
		return fmt.Errorf("cuda: launch kernel ex: %w", err)
	}
	return nil
}
