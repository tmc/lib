package xla

import (
	"fmt"
	"io"
	"unsafe"

	"github.com/tmc/lib/openxla"
)

// ElementType specifies the data type of a buffer's elements.
// This is a direct alias to openxla.Buffer_Type.
type ElementType = openxla.Buffer_Type

const (
	F32 = openxla.Buffer_Type_F32
	S32 = openxla.Buffer_Type_S32
	U8  = openxla.Buffer_Type_U8
	S8  = openxla.Buffer_Type_S8
	F16 = openxla.Buffer_Type_F16
	F64 = openxla.Buffer_Type_F64
)

// Buffer represents a typed device memory buffer.
type Buffer struct {
	wrapper *openxla.BufferWrapper
	dims    []int64
	elemTyp ElementType
}

// NewBufferFromHost creates a device buffer by transferring data from host memory.
func (e *Engine) NewBufferFromHost(data []byte, elemTyp ElementType, dims []int64, dev *openxla.Device) (*Buffer, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("xla: cannot create buffer from empty slice")
	}

	ptr := unsafe.Pointer(&data[0])
	wrapper, err := e.client.BufferFromHost(ptr, elemTyp, dims, dev)
	if err != nil {
		return nil, fmt.Errorf("xla: failed to create buffer from host: %w", err)
	}

	return &Buffer{
		wrapper: wrapper,
		dims:    dims,
		elemTyp: elemTyp,
	}, nil
}

// ToHost copies the device buffer synchronously into the provided host slice.
func (b *Buffer) ToHost(dst []byte) error {
	if len(dst) == 0 {
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
	size := int64(1)
	for _, d := range b.dims {
		size *= d
	}

	switch b.elemTyp {
	case F64:
		size *= 8
	case F32, S32, openxla.Buffer_Type_U32:
		size *= 4
	case F16, openxla.Buffer_Type_S16, openxla.Buffer_Type_U16:
		size *= 2
	case S8, U8:
		size *= 1
	default:
		return 0, fmt.Errorf("xla: unsupported element type for WriteTo")
	}

	buf := make([]byte, size)
	if err := b.ToHost(buf); err != nil {
		return 0, err
	}

	n, err := w.Write(buf)
	return int64(n), err
}

// ReadFrom implements io.ReaderFrom, allocating a new device buffer and populating it.
// Note: Since this needs an Engine, it's typically better to use NewBufferFromHost or Engine methods,
// but if this Buffer was created uninitialized, it could read into it.
// For now, we'll return an error indicating it's not supported to overwrite an existing buffer.
func (b *Buffer) ReadFrom(r io.Reader) (int64, error) {
	return 0, fmt.Errorf("xla: ReadFrom not implemented for immutable device buffers")
}

// Close destroys the device buffer, releasing its memory.
func (b *Buffer) Close() error {
	if b.wrapper != nil {
		if err := b.wrapper.Destroy(); err != nil {
			return fmt.Errorf("xla: failed to close buffer: %w", err)
		}
		b.wrapper = nil
	}
	return nil
}
