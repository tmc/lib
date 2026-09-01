package mempool

import (
	"fmt"
	"os"
	"sync"
	"syscall"
	"unsafe"

	"github.com/tmc/apple/metal"
	applemetal "github.com/tmc/lib/apple/x/metal"
)

// Slice represents a typed, contiguous slice of memory residing in Apple Unified Memory.
type Slice[T any] struct {
	buf      metal.MTLBuffer
	len      int
	elemSize uintptr
}

// Alloc allocates a new typed Slice[T] of length elements in shared unified memory.
func Alloc[T any](dev *applemetal.Device, length int) (Slice[T], error) {
	if length < 0 {
		return Slice[T]{}, fmt.Errorf("mempool: negative allocation length %d", length)
	}
	var t T
	elemSize := unsafe.Sizeof(t)
	if elemSize == 0 {
		elemSize = 1
	}
	totalBytes := int(uintptr(length) * elemSize)
	if totalBytes == 0 {
		totalBytes = int(elemSize)
	}

	buf, err := dev.AllocBuffer(totalBytes)
	if err != nil {
		return Slice[T]{}, fmt.Errorf("mempool: alloc %d elements: %w", length, err)
	}

	return Slice[T]{
		buf:      buf,
		len:      length,
		elemSize: elemSize,
	}, nil
}

// AllocFromSlice copies a Go slice into a newly allocated Metal buffer.
func AllocFromSlice[T any](dev *applemetal.Device, data []T) (Slice[T], error) {
	if len(data) == 0 {
		return Alloc[T](dev, 0)
	}
	var t T
	elemSize := unsafe.Sizeof(t)
	if elemSize == 0 {
		elemSize = 1
	}
	totalBytes := int(uintptr(len(data)) * elemSize)

	buf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&data[0]), uint(totalBytes), metal.MTLResourceStorageModeShared)
	if buf.GetID() == 0 {
		return Slice[T]{}, fmt.Errorf("mempool: failed allocating buffer for %d elements", len(data))
	}

	return Slice[T]{
		buf:      buf,
		len:      len(data),
		elemSize: elemSize,
	}, nil
}

// AllocMMapZeroCopy maps an open file into virtual memory and creates a zero-copy shared MTLBuffer directly aliasing it.
func AllocMMapZeroCopy[T any](dev *applemetal.Device, f *os.File, offset int64, length int) (Slice[T], error) {
	var t T
	elemSize := unsafe.Sizeof(t)
	if elemSize == 0 {
		elemSize = 1
	}
	totalBytes := int(uintptr(length) * elemSize)
	if totalBytes <= 0 {
		return Slice[T]{}, fmt.Errorf("mempool: invalid zero-copy length %d", length)
	}

	data, err := syscall.Mmap(int(f.Fd()), offset, totalBytes, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return Slice[T]{}, fmt.Errorf("mempool: mmap file: %w", err)
	}

	buf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&data[0]), uint(totalBytes), metal.MTLResourceStorageModeShared)
	if buf.GetID() == 0 {
		_ = syscall.Munmap(data)
		return Slice[T]{}, fmt.Errorf("mempool: failed creating zero-copy Metal buffer from mmap")
	}

	return Slice[T]{
		buf:      buf,
		len:      length,
		elemSize: elemSize,
	}, nil
}

// Buffer returns the underlying metal.MTLBuffer.
func (s Slice[T]) Buffer() metal.MTLBuffer {
	return s.buf
}

// Len returns the number of elements in the slice.
func (s Slice[T]) Len() int {
	return s.len
}

// SizeBytes returns the size of the slice in bytes.
func (s Slice[T]) SizeBytes() int {
	return int(uintptr(s.len) * s.elemSize)
}

// HostSlice returns a direct unsafe Go slice pointing to the shared unified memory buffer.
func (s Slice[T]) HostSlice() []T {
	if s.len == 0 || s.buf.GetID() == 0 {
		return nil
	}
	ptr := (*T)(s.buf.Contents())
	return unsafe.Slice(ptr, s.len)
}

// Free releases the buffer.
func (s Slice[T]) Free() error {
	return nil
}

// Pool manages reusable scratchpad buffers in Apple Unified Memory.
type Pool struct {
	dev     *applemetal.Device
	mu      sync.Mutex
	buffers map[int][]metal.MTLBuffer
}

// NewPool creates a new unified memory pool.
func NewPool(dev *applemetal.Device) *Pool {
	return &Pool{
		dev:     dev,
		buffers: make(map[int][]metal.MTLBuffer),
	}
}

// Get acquires a buffer of at least sizeBytes from the pool, or allocates a new one.
func (p *Pool) Get(sizeBytes int) (metal.MTLBuffer, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	bucket := roundUpPowerOf2(sizeBytes)
	list := p.buffers[bucket]
	if len(list) > 0 {
		buf := list[len(list)-1]
		p.buffers[bucket] = list[:len(list)-1]
		return buf, nil
	}

	return p.dev.AllocBuffer(bucket)
}

// Put returns a buffer back to the pool.
func (p *Pool) Put(buf metal.MTLBuffer) {
	if buf.GetID() == 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	bucket := int(buf.Length())
	p.buffers[bucket] = append(p.buffers[bucket], buf)
}

func roundUpPowerOf2(n int) int {
	if n <= 256 {
		return 256
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n++
	return n
}
