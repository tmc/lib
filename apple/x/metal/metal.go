package metal

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
)

// ErrClosed is returned by operations on a closed Device.
var ErrClosed = errors.New("metal: device is closed")

// Device represents an Apple Metal GPU device.
type Device struct {
	mu     sync.RWMutex
	closed bool
	raw    metal.MTLDevice
	queue  metal.MTLCommandQueue
}

// NewDevice returns the system default Metal GPU device.
func NewDevice() (*Device, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev := metal.MTLCreateSystemDefaultDevice()
	if dev.GetID() == 0 {
		return nil, fmt.Errorf("metal: no default Metal device found")
	}
	queue := dev.NewCommandQueue()
	if queue.GetID() == 0 {
		objectivec.ObjectFromID(dev.GetID()).Release()
		return nil, fmt.Errorf("metal: failed creating command queue")
	}
	return &Device{
		raw:   dev,
		queue: queue,
	}, nil
}

// Name returns the device name.
func (d *Device) Name() string {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return ""
	}
	return d.raw.Name()
}

// Raw returns the underlying Metal device. The result is borrowed and remains
// valid only until Close. Raw must not run concurrently with Close.
func (d *Device) Raw() metal.MTLDevice {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.raw
}

// Queue returns the device's borrowed default command queue. It returns nil
// after Close. Before Close it is non-nil because [NewDevice] rejects a device
// that cannot create a queue. The result must not be used concurrently with
// Close.
func (d *Device) Queue() metal.MTLCommandQueue {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.queue
}

// Close releases the command queue and Metal device. It is idempotent. The
// caller must complete all command buffers and release device-derived objects
// before calling Close. Close waits for Device methods already in progress;
// borrowed values returned by Raw and Queue must not be used concurrently with
// Close.
func (d *Device) Close() error {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	d.closed = true
	if d.queue != nil && d.queue.GetID() != 0 {
		objectivec.ObjectFromID(d.queue.GetID()).Release()
	}
	if d.raw != nil && d.raw.GetID() != 0 {
		objectivec.ObjectFromID(d.raw.GetID()).Release()
	}
	d.queue = nil
	d.raw = nil
	return nil
}

// CompileLibrary compiles MSL source into a library owned by the caller. The
// caller must release the library after creating its functions and pipelines.
func (d *Device) CompileLibrary(source string) (metal.MTLLibrary, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return nil, ErrClosed
	}
	lib, err := d.raw.NewLibraryWithSourceOptionsError(source, metal.MTLCompileOptions{})
	if err != nil {
		return nil, fmt.Errorf("metal: compile MSL library: %w", err)
	}
	return lib, nil
}

// NewComputePipeline creates a pipeline owned by the caller. The temporary
// function used to create the pipeline is released before the method returns.
func (d *Device) NewComputePipeline(lib metal.MTLLibrary, name string) (metal.MTLComputePipelineState, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return nil, ErrClosed
	}
	fn := lib.NewFunctionWithName(name)
	if fn.GetID() == 0 {
		return nil, fmt.Errorf("metal: kernel function %q not found in library", name)
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	pso, err := d.raw.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, fmt.Errorf("metal: create compute pipeline for %q: %w", name, err)
	}
	return pso, nil
}

// AllocBuffer allocates a shared Metal buffer owned by the caller.
func (d *Device) AllocBuffer(bytes int) (metal.MTLBuffer, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return nil, ErrClosed
	}
	if bytes <= 0 {
		bytes = 4
	}
	buf := d.raw.NewBufferWithLengthOptions(uint(bytes), metal.MTLResourceStorageModeShared)
	if buf.GetID() == 0 {
		return nil, fmt.Errorf("metal: alloc buffer with %d bytes failed", bytes)
	}
	return buf, nil
}

// AllocBufferBytes copies host data into a shared buffer owned by the caller.
func (d *Device) AllocBufferBytes(data []byte) (metal.MTLBuffer, error) {
	if len(data) == 0 {
		return d.AllocBuffer(4)
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return nil, ErrClosed
	}
	buf := d.raw.NewBufferWithBytesLengthOptions(unsafe.Pointer(&data[0]), uint(len(data)), metal.MTLResourceStorageModeShared)
	if buf.GetID() == 0 {
		return nil, fmt.Errorf("metal: alloc buffer with %d bytes failed", len(data))
	}
	return buf, nil
}
