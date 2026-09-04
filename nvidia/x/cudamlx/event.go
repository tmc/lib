package cudamlx

import (
	"reflect"
	"runtime"
	"sync/atomic"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// Event wraps a CUDA event. It can synchronize between GPU streams, or
// let the CPU wait on a GPU stream, but cannot wait on CPU work. Port of
// CudaEvent in event.cu (without the per-thread event pool).
type Event struct {
	dev *Device
	ev  driver.Event
}

// NewEvent creates an event on the device with the given driver flags.
func NewEvent(dev *Device, flags driver.EventFlags) (*Event, error) {
	var ev driver.Event
	err := dev.WithCurrent(func() error {
		var err error
		ev, err = driver.NewEvent(uint32(flags))
		return err
	})
	if err != nil {
		return nil, err
	}
	return &Event{dev: dev, ev: ev}, nil
}

// Record records the event on stream.
func (e *Event) Record(stream driver.Stream) error {
	return e.dev.WithCurrent(func() error { return e.ev.Record(stream) })
}

// Wait blocks the calling goroutine until the event completes.
func (e *Event) Wait() error {
	return e.dev.WithCurrent(func() error { return e.ev.Synchronize() })
}

// WaitStream makes stream wait for the event.
func (e *Event) WaitStream(stream driver.Stream) error {
	return e.dev.WithCurrent(func() error { return stream.WaitEvent(e.ev, 0) })
}

// Completed reports whether recorded work has finished. It is true when
// Record has not been called.
func (e *Event) Completed() bool {
	return e.ev.Query() == nil
}

// Destroy releases the event.
func (e *Event) Destroy() error {
	return e.ev.Destroy()
}

// SharedEvent synchronizes between CPU and GPU through a 32-bit value in
// unified memory. It is much slower than Event, which should be
// preferred when possible. Port of AtomicEvent in event.cu, with the
// hand-written signal/wait kernels replaced by cuStreamWriteValue32 and
// cuStreamWaitValue32.
type SharedEvent struct {
	dev *Device
	buf *Buffer
	val *uint32 // host view of buf
}

// NewSharedEvent creates a shared event with value zero.
func NewSharedEvent(dev *Device) (*SharedEvent, error) {
	a, err := GetAllocator()
	if err != nil {
		return nil, err
	}
	buf, err := a.Malloc(4)
	if err != nil {
		return nil, err
	}
	e := &SharedEvent{dev: dev, buf: buf, val: hostUint32(buf.Data)}
	atomic.StoreUint32(e.val, 0)
	return e, nil
}

// hostUint32 returns a host view of the unified allocation at addr. The
// memory is outside the Go heap, so the slice header is built by hand, as
// in x/cudamem.
func hostUint32(addr uintptr) *uint32 {
	var s []uint32
	h := (*reflect.SliceHeader)(unsafe.Pointer(&s))
	h.Data = addr
	h.Len, h.Cap = 1, 1
	return &s[0]
}

// Value returns the current event value.
func (e *SharedEvent) Value() uint32 {
	return atomic.LoadUint32(e.val)
}

// IsSignaled reports whether the event value has reached value.
func (e *SharedEvent) IsSignaled(value uint32) bool {
	return e.Value() >= value
}

// Signal sets the event value from the CPU.
func (e *SharedEvent) Signal(value uint32) {
	atomic.StoreUint32(e.val, value)
}

// SignalStream sets the event value from the GPU when stream reaches
// this point.
func (e *SharedEvent) SignalStream(stream driver.Stream, value uint32) error {
	return e.dev.WithCurrent(func() error {
		return stream.WriteValue32(driver.DevicePtr(e.buf.Data), value, 0)
	})
}

// Wait blocks the calling goroutine until the event value reaches value.
func (e *SharedEvent) Wait(value uint32) {
	for atomic.LoadUint32(e.val) < value {
		runtime.Gosched()
	}
}

// WaitStream makes stream wait until the event value reaches value.
func (e *SharedEvent) WaitStream(stream driver.Stream, value uint32) error {
	return e.dev.WithCurrent(func() error {
		return stream.WaitValue32(driver.DevicePtr(e.buf.Data), value, uint32(driver.StreamWaitValueGeq))
	})
}

// Free returns the event's buffer to the allocator.
func (e *SharedEvent) Free() {
	if a, err := GetAllocator(); err == nil {
		a.Free(e.buf)
	}
}
