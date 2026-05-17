package nbd

import (
	"errors"
	"io"
	"sync"
)

var (
	ErrReadOnly        = errors.New("nbd: read only")
	ErrInvalidArgument = errors.New("nbd: invalid argument")
	ErrUnsupported     = errors.New("nbd: unsupported")
	ErrClosed          = errors.New("nbd: closed")
)

// Device is the storage exported by an NBD server.
type Device interface {
	io.ReaderAt
	io.WriterAt
	Size() uint64
}

// Syncer is implemented by devices that can flush pending writes.
type Syncer interface {
	Sync() error
}

// Trimmer is implemented by devices that can discard a byte range.
type Trimmer interface {
	Trim(off, length uint64) error
}

// MemoryDevice is an in-memory Device for tests and small tools.
type MemoryDevice struct {
	mu       sync.Mutex
	data     []byte
	readOnly bool
}

// NewMemoryDevice returns a writable in-memory device of size bytes.
func NewMemoryDevice(size uint64) *MemoryDevice {
	return &MemoryDevice{data: make([]byte, size)}
}

// NewReadOnlyMemoryDevice returns a read-only in-memory device backed by data.
func NewReadOnlyMemoryDevice(data []byte) *MemoryDevice {
	cp := append([]byte(nil), data...)
	return &MemoryDevice{data: cp, readOnly: true}
}

func (m *MemoryDevice) Size() uint64 {
	if m == nil {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return uint64(len(m.data))
}

func (m *MemoryDevice) ReadAt(p []byte, off int64) (int, error) {
	if m == nil {
		return 0, ErrClosed
	}
	if off < 0 {
		return 0, ErrInvalidArgument
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if uint64(off) >= uint64(len(m.data)) {
		return 0, io.EOF
	}
	n := copy(p, m.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (m *MemoryDevice) WriteAt(p []byte, off int64) (int, error) {
	if m == nil {
		return 0, ErrClosed
	}
	if m.readOnly {
		return 0, ErrReadOnly
	}
	if off < 0 {
		return 0, ErrInvalidArgument
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if uint64(off) > uint64(len(m.data)) {
		return 0, io.ErrShortWrite
	}
	n := copy(m.data[off:], p)
	if n < len(p) {
		return n, io.ErrShortWrite
	}
	return n, nil
}

func (m *MemoryDevice) Sync() error { return nil }

func (m *MemoryDevice) Trim(off, length uint64) error {
	if m == nil {
		return ErrClosed
	}
	if m.readOnly {
		return ErrReadOnly
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if off > uint64(len(m.data)) || length > uint64(len(m.data))-off {
		return ErrInvalidArgument
	}
	clear(m.data[off : off+length])
	return nil
}
