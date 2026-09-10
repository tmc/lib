package xla

import (
	"fmt"
	"io"
	"slices"
	"sync"
	"unsafe"

	"github.com/tmc/lib/openxla"
)

// ElementType specifies the data type of a buffer's elements.
// This is a direct alias to openxla.Buffer_Type.
type ElementType = openxla.Buffer_Type

const (
	F32  = openxla.Buffer_Type_F32
	S32  = openxla.Buffer_Type_S32
	U8   = openxla.Buffer_Type_U8
	S8   = openxla.Buffer_Type_S8
	F16  = openxla.Buffer_Type_F16
	BF16 = openxla.Buffer_Type_BF16
	F64  = openxla.Buffer_Type_F64
)

// Buffer represents a typed device memory buffer.
type Buffer struct {
	viewMu  sync.Mutex // Guards scoped host access and Close.
	viewing bool
	engine  *Engine
	wrapper *openxla.BufferWrapper
	dims    []int64
	elemTyp ElementType
	shared  *sharedAllocation
}

// NewBufferFromHost copies data to a device buffer with the given element type
// and shape. data must contain exactly the bytes described by dims; a nil shape
// describes a scalar. Packed element types are not supported. dev must be a
// non-nil device belonging to this engine.
func (e *Engine) NewBufferFromHost(data []byte, elemTyp ElementType, dims []int64, dev *openxla.Device) (*Buffer, error) {
	size, err := bufferSize(elemTyp, dims)
	if err != nil {
		return nil, err
	}
	if len(data) != size {
		return nil, fmt.Errorf("xla: host buffer has %d bytes, want %d", len(data), size)
	}
	if e == nil || e.client == nil {
		return nil, fmt.Errorf("xla: engine is closed")
	}

	ptr := unsafe.Pointer(unsafe.SliceData(data))
	wrapper, err := e.client.BufferFromHost(ptr, elemTyp, dims, dev)
	if err != nil {
		return nil, fmt.Errorf("xla: failed to create buffer from host: %w", err)
	}

	return &Buffer{
		engine:  e,
		wrapper: wrapper,
		dims:    slices.Clone(dims),
		elemTyp: elemTyp,
	}, nil
}

// NewBufferFromFloats creates an F32 device buffer directly from a []float32.
func (e *Engine) NewBufferFromFloats(data []float32, dims []int64, dev *openxla.Device) (*Buffer, error) {
	size, err := bufferSize(F32, dims)
	if err != nil {
		return nil, err
	}
	if len(data)*4 != size {
		return nil, fmt.Errorf("xla: host buffer has %d elements (%d bytes), want %d bytes", len(data), len(data)*4, size)
	}
	if e == nil || e.client == nil {
		return nil, fmt.Errorf("xla: engine is closed")
	}
	ptr := unsafe.Pointer(unsafe.SliceData(data))
	wrapper, err := e.client.BufferFromHost(ptr, F32, dims, dev)
	if err != nil {
		return nil, fmt.Errorf("xla: failed to create buffer from host: %w", err)
	}
	return &Buffer{
		engine:  e,
		wrapper: wrapper,
		dims:    slices.Clone(dims),
		elemTyp: F32,
	}, nil
}

// NewBufferFromInts creates an S64 device buffer directly from a []int64.
func (e *Engine) NewBufferFromInts(data []int64, dims []int64, dev *openxla.Device) (*Buffer, error) {
	size, err := bufferSize(openxla.Buffer_Type_S64, dims)
	if err != nil {
		return nil, err
	}
	if len(data)*8 != size {
		return nil, fmt.Errorf("xla: host buffer has %d elements (%d bytes), want %d bytes", len(data), len(data)*8, size)
	}
	if e == nil || e.client == nil {
		return nil, fmt.Errorf("xla: engine is closed")
	}
	ptr := unsafe.Pointer(unsafe.SliceData(data))
	wrapper, err := e.client.BufferFromHost(ptr, openxla.Buffer_Type_S64, dims, dev)
	if err != nil {
		return nil, fmt.Errorf("xla: failed to create buffer from host: %w", err)
	}
	return &Buffer{
		engine:  e,
		wrapper: wrapper,
		dims:    slices.Clone(dims),
		elemTyp: openxla.Buffer_Type_S64,
	}, nil
}

// Device returns the device on which the buffer resides.
func (b *Buffer) Device() (*openxla.Device, error) {
	if b == nil || b.wrapper == nil {
		return nil, fmt.Errorf("xla: buffer is closed")
	}
	return b.wrapper.Device()
}

// Shape returns the element type and a copy of the buffer dimensions.
// A closed buffer returns an error.
func (b *Buffer) Shape() (ElementType, []int64, error) {
	if b == nil || b.wrapper == nil {
		return 0, nil, fmt.Errorf("xla: buffer is closed")
	}
	return b.elemTyp, slices.Clone(b.dims), nil
}

// ToHost copies the device buffer synchronously into the provided host slice.
func (b *Buffer) ToHost(dst []byte) error {
	if b == nil || b.wrapper == nil {
		return fmt.Errorf("xla: buffer is closed")
	}
	size, err := bufferSize(b.elemTyp, b.dims)
	if err != nil {
		return err
	}
	if len(dst) < size {
		return fmt.Errorf("xla: host buffer has %d bytes, need %d", len(dst), size)
	}
	if size == 0 {
		return nil
	}
	ptr := unsafe.Pointer(&dst[0])
	if err := b.wrapper.ToHost(ptr, uintptr(len(dst))); err != nil {
		return fmt.Errorf("xla: failed to transfer to host: %w", err)
	}
	return nil
}

// ReadFloats copies an F32 device buffer directly into the provided float32 slice.
func (b *Buffer) ReadFloats(dst []float32) error {
	if b == nil || b.wrapper == nil {
		return fmt.Errorf("xla: buffer is closed")
	}
	if b.elemTyp != F32 {
		return fmt.Errorf("xla: buffer element type %v is not F32", b.elemTyp)
	}
	size, err := bufferSize(b.elemTyp, b.dims)
	if err != nil {
		return err
	}
	wantFloats := size / 4
	if len(dst) < wantFloats {
		return fmt.Errorf("xla: host float slice has %d elements, need %d", len(dst), wantFloats)
	}
	if size == 0 {
		return nil
	}
	ptr := unsafe.Pointer(&dst[0])
	if err := b.wrapper.ToHost(ptr, uintptr(size)); err != nil {
		return fmt.Errorf("xla: failed to transfer to host: %w", err)
	}
	return nil
}

// WriteTo implements io.WriterTo, copying the device memory to the provided Writer.
func (b *Buffer) WriteTo(w io.Writer) (int64, error) {
	if b == nil || b.wrapper == nil {
		return 0, fmt.Errorf("xla: buffer is closed")
	}
	size, err := bufferSize(b.elemTyp, b.dims)
	if err != nil {
		return 0, err
	}

	buf := make([]byte, size)
	if err := b.ToHost(buf); err != nil {
		return 0, err
	}

	n, err := w.Write(buf)
	if n < len(buf) && err == nil {
		err = io.ErrShortWrite
	}
	return int64(n), err
}

// bufferSize returns the number of bytes in a dense, unpacked host buffer.
func bufferSize(elemTyp ElementType, dims []int64) (int, error) {
	var width int
	switch elemTyp {
	case openxla.Buffer_Type_PRED, S8, U8,
		openxla.Buffer_Type_F8E5M2, openxla.Buffer_Type_F8E4M3FN,
		openxla.Buffer_Type_F8E4M3B11FNUZ, openxla.Buffer_Type_F8E5M2FNUZ,
		openxla.Buffer_Type_F8E4M3FNUZ, openxla.Buffer_Type_F8E4M3,
		openxla.Buffer_Type_F8E3M4, openxla.Buffer_Type_F8E8M0FNU:
		width = 1
	case F16, openxla.Buffer_Type_BF16, openxla.Buffer_Type_S16, openxla.Buffer_Type_U16:
		width = 2
	case F32, S32, openxla.Buffer_Type_U32:
		width = 4
	case F64, openxla.Buffer_Type_S64, openxla.Buffer_Type_U64, openxla.Buffer_Type_C64:
		width = 8
	case openxla.Buffer_Type_C128:
		width = 16
	default:
		return 0, fmt.Errorf("xla: unsupported host element type %d", elemTyp)
	}
	empty := false
	for _, d := range dims {
		if d < 0 {
			return 0, fmt.Errorf("xla: negative dimension %d", d)
		}
		empty = empty || d == 0
	}
	if empty {
		return 0, nil
	}
	size := width
	for _, d := range dims {
		if d > int64(int(^uint(0)>>1)/size) {
			return 0, fmt.Errorf("xla: buffer size overflows int")
		}
		size *= int(d)
	}
	return size, nil
}

// Close destroys the device buffer, releasing its memory.
func (b *Buffer) Close() error {
	if b == nil {
		return nil
	}
	b.viewMu.Lock()
	defer b.viewMu.Unlock()
	if b.viewing {
		return fmt.Errorf("xla: buffer has an active host view")
	}
	if b.wrapper == nil {
		return nil
	}
	resource := b.wrapper
	b.wrapper = nil
	b.shared = nil
	b.engine = nil
	if err := resource.Destroy(); err != nil {
		return fmt.Errorf("xla: close buffer: %w", err)
	}
	return nil
}
