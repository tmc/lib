package xla

import (
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/openxla"
)

// sharedAllocation is rooted independently of Buffer until PJRT releases its view.
// Destroy may return before the callback; it must not unpin this storage.
type sharedAllocation struct {
	data []byte
	pin  runtime.Pinner
}

var sharedAllocations sync.Map // aligned data pointer -> *sharedAllocation
var sharedDelete = purego.NewCallback(func(data, _ unsafe.Pointer) {
	releaseShared(data)
})

func releaseShared(data unsafe.Pointer) {
	if value, ok := sharedAllocations.LoadAndDelete(data); ok {
		value.(*sharedAllocation).pin.Unpin()
	}
}

func allocateShared(size int) (*sharedAllocation, unsafe.Pointer, error) {
	const alignment = 64 // Conservative alignment qualified with the retained CPU plugin.
	if size > int(^uint(0)>>1)-alignment {
		return nil, nil, fmt.Errorf("xla: shared allocation size overflows int")
	}
	storage := make([]byte, size+alignment)
	base := unsafe.Pointer(unsafe.SliceData(storage))
	offset := (alignment - uintptr(base)%alignment) % alignment
	data := unsafe.Add(base, offset)
	a := &sharedAllocation{data: unsafe.Slice((*byte)(data), size)}
	a.pin.Pin(base)
	sharedAllocations.Store(data, a)
	return a, data, nil
}

// NewSharedBuffer allocates zeroed, dense row-major CPU storage. The returned
// slice may be mutated between executions, but never during execution or Close.
// It is valid only while the buffer is alive. Storage remains pinned until PJRT
// releases its view, which can occur after Close. dev must belong to e.
func (e *Engine) NewSharedBuffer(typ ElementType, dims []int64, dev *openxla.Device) (*Buffer, []byte, error) {
	size, err := bufferSize(typ, dims)
	if err != nil {
		return nil, nil, err
	}
	if e == nil || e.client == nil {
		return nil, nil, fmt.Errorf("xla: engine is closed")
	}
	devices, err := e.Devices()
	if err != nil {
		return nil, nil, err
	}
	if dev == nil || !slices.Contains(devices, dev) {
		return nil, nil, fmt.Errorf("xla: device does not belong to engine")
	}
	// Test real storage on this device before handing PJRT a host address.
	probe, err := e.NewBufferFromHost([]byte{0}, U8, []int64{1}, dev)
	if err != nil {
		return nil, nil, err
	}
	cpu, err := probe.wrapper.IsOnCPU()
	if err == nil && !cpu {
		err = fmt.Errorf("xla: shared storage requires a CPU device")
	}
	if err == nil {
		err = probe.wrapper.WaitReady()
	}
	if err = errors.Join(err, probe.Close()); err != nil {
		return nil, nil, err
	}

	a, data, err := allocateShared(size)
	if err != nil {
		return nil, nil, err
	}
	minor := make([]int64, len(dims))
	for i := range minor {
		minor[i] = int64(len(dims) - 1 - i)
	}
	layout := &openxla.Buffer_MemoryLayout{Struct_size: openxla.PJRT_Buffer_MemoryLayout_STRUCT_SIZE, Type: openxla.Buffer_MemoryLayout_Type_Tiled}
	layout.Union_pad = openxla.Buffer_MemoryLayout_Tiled{Struct_size: openxla.PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE, Minor_to_major: unsafe.SliceData(minor), Minor_to_major_size: uintptr(len(minor))}
	var pin runtime.Pinner
	pin.Pin(layout)
	if len(minor) != 0 {
		pin.Pin(&minor[0])
	}
	wrapper, err := e.client.CreateViewOfDeviceBuffer(data, typ, dims, layout, dev, sharedDelete, nil)
	pin.Unpin()
	if err != nil {
		releaseShared(data)
		return nil, nil, err
	}
	b := &Buffer{wrapper: wrapper, dims: slices.Clone(dims), elemTyp: typ, shared: a}
	cpu, err = wrapper.IsOnCPU()
	if err == nil && !cpu {
		err = fmt.Errorf("xla: shared view is not on CPU")
	}
	if err == nil {
		err = wrapper.WaitReady()
	}
	if err == nil {
		var gotType ElementType
		var gotDims []int64
		gotType, gotDims, err = wrapper.Shape()
		if err == nil && (gotType != typ || !slices.Equal(gotDims, dims)) {
			err = fmt.Errorf("xla: shared view shape mismatch")
		}
	}
	if err != nil {
		return nil, nil, errors.Join(err, b.Close())
	}
	return b, a.data, nil
}

// IsShared reports whether this buffer uses storage allocated by NewSharedBuffer.
func (b *Buffer) IsShared() bool { return b != nil && b.wrapper != nil && b.shared != nil }

// SharedData returns the owned CPU storage after its producer completes. Native
// executable outputs are not owned shared allocations and are rejected. The
// caller must synchronize access with execution and Close.
func (b *Buffer) SharedData() ([]byte, error) {
	if b == nil || b.wrapper == nil {
		return nil, fmt.Errorf("xla: buffer is closed")
	}
	if b.shared == nil {
		return nil, fmt.Errorf("xla: buffer has no owned shared storage")
	}
	if err := b.wrapper.WaitReady(); err != nil {
		return nil, err
	}
	return b.shared.data, nil
}
