package metal

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// Device represents an Apple Metal GPU device.
type Device struct {
	raw   metal.MTLDevice
	queue metal.MTLCommandQueue
}

// NewDevice returns the system default Metal GPU device.
func NewDevice() (*Device, error) {
	runtime.LockOSThread()
	dev := metal.MTLCreateSystemDefaultDevice()
	if dev.GetID() == 0 {
		return nil, fmt.Errorf("metal: no default Metal device found")
	}
	queue := dev.NewCommandQueue()
	if queue.GetID() == 0 {
		return nil, fmt.Errorf("metal: failed creating command queue")
	}
	return &Device{
		raw:   dev,
		queue: queue,
	}, nil
}

// Name returns the device name.
func (d *Device) Name() string {
	return d.raw.Name()
}

// Raw returns the underlying metal.MTLDevice.
func (d *Device) Raw() metal.MTLDevice {
	return d.raw
}

// Queue returns the device's default command queue. It is never nil: a Device
// that could not make one is never returned by [NewDevice], so callers need no
// nil check before encoding.
func (d *Device) Queue() metal.MTLCommandQueue {
	return d.queue
}

// Close releases resources associated with the device.
//
// It currently releases nothing: the default command queue outlives any
// command buffer built from it, and this package has no way to know when the
// last one has completed.
func (d *Device) Close() error {
	return nil
}

// CompileLibrary compiles a Metal Shading Language (MSL) source string into an MTLLibrary.
func (d *Device) CompileLibrary(source string) (metal.MTLLibrary, error) {
	lib, err := d.raw.NewLibraryWithSourceOptionsError(source, metal.MTLCompileOptions{})
	if err != nil {
		return nil, fmt.Errorf("metal: compile MSL library: %w", err)
	}
	return lib, nil
}

// NewComputePipeline compiles a named kernel function from an MTLLibrary into a compute pipeline state.
func (d *Device) NewComputePipeline(lib metal.MTLLibrary, name string) (metal.MTLComputePipelineState, error) {
	fn := lib.NewFunctionWithName(name)
	if fn.GetID() == 0 {
		return nil, fmt.Errorf("metal: kernel function %q not found in library", name)
	}
	pso, err := d.raw.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, fmt.Errorf("metal: create compute pipeline for %q: %w", name, err)
	}
	return pso, nil
}

// AllocBuffer allocates a shared Metal buffer of the given byte length.
func (d *Device) AllocBuffer(bytes int) (metal.MTLBuffer, error) {
	if bytes <= 0 {
		bytes = 4
	}
	buf := d.raw.NewBufferWithLengthOptions(uint(bytes), metal.MTLResourceStorageModeShared)
	if buf.GetID() == 0 {
		return nil, fmt.Errorf("metal: alloc buffer with %d bytes failed", bytes)
	}
	return buf, nil
}

// AllocBufferBytes copies host data into a newly allocated shared Metal buffer.
func (d *Device) AllocBufferBytes(data []byte) (metal.MTLBuffer, error) {
	if len(data) == 0 {
		return d.AllocBuffer(4)
	}
	buf := d.raw.NewBufferWithBytesLengthOptions(unsafe.Pointer(&data[0]), uint(len(data)), metal.MTLResourceStorageModeShared)
	if buf.GetID() == 0 {
		return nil, fmt.Errorf("metal: alloc buffer with %d bytes failed", len(data))
	}
	return buf, nil
}
