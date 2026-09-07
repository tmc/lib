package xla

import (
	"fmt"
	"io"
	"slices"
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
	wrapper *openxla.BufferWrapper
	dims    []int64
	elemTyp ElementType
}

// NewBufferFromHost copies data to a device buffer with the given element type
// and shape. data must contain exactly the bytes described by dims; a nil shape
// describes a scalar. Packed element types are not supported.
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
		wrapper: wrapper,
		dims:    slices.Clone(dims),
		elemTyp: elemTyp,
	}, nil
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
	if b == nil || b.wrapper == nil {
		return nil
	}
	resource := b.wrapper
	b.wrapper = nil
	if err := resource.Destroy(); err != nil {
		return fmt.Errorf("xla: close buffer: %w", err)
	}
	return nil
}
