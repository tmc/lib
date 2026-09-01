package metalmem

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
	cap      int
	elemSize uintptr
}

// Alloc allocates a new typed Slice[T] of length elements in shared unified memory.
func Alloc[T any](dev *applemetal.Device, length int) (Slice[T], error) {
	if length < 0 {
		return Slice[T]{}, fmt.Errorf("metalmem: negative allocation length %d", length)
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
		return Slice[T]{}, fmt.Errorf("metalmem: alloc %d elements: %w", length, err)
	}

	return Slice[T]{
		buf:      buf,
		len:      length,
		cap:      length,
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
		return Slice[T]{}, fmt.Errorf("metalmem: failed allocating buffer for %d elements", len(data))
	}

	return Slice[T]{
		buf:      buf,
		len:      len(data),
		cap:      len(data),
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
		return Slice[T]{}, fmt.Errorf("metalmem: invalid zero-copy length %d", length)
	}

	data, err := syscall.Mmap(int(f.Fd()), offset, totalBytes, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return Slice[T]{}, fmt.Errorf("metalmem: mmap file: %w", err)
	}

	buf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&data[0]), uint(totalBytes), metal.MTLResourceStorageModeShared)
	if buf.GetID() == 0 {
		_ = syscall.Munmap(data)
		return Slice[T]{}, fmt.Errorf("metalmem: failed creating zero-copy Metal buffer from mmap")
	}

	return Slice[T]{
		buf:      buf,
		len:      length,
		cap:      length,
		elemSize: elemSize,
	}, nil
}

// Buffer returns the underlying MTLBuffer handle.
func (s Slice[T]) Buffer() metal.MTLBuffer {
	return s.buf
}

// Len returns the number of elements in the slice.
func (s Slice[T]) Len() int {
	return s.len
}

// Cap returns the capacity of the slice.
func (s Slice[T]) Cap() int {
	return s.cap
}

// ElemSize returns the byte size of each element.
func (s Slice[T]) ElemSize() uintptr {
	return s.elemSize
}

// TotalBytes returns the total bytes occupied by the elements.
func (s Slice[T]) TotalBytes() uint {
	return uint(uintptr(s.len) * s.elemSize)
}

// Free releases the buffer reference.
func (s Slice[T]) Free() {
	// Metal buffers in purego are ARC managed or explicitly released if applicable.
}

// HostSlice returns a direct CPU-accessible Go slice aliasing the unified buffer.
func (s Slice[T]) HostSlice() []T {
	if s.len == 0 || s.buf == nil || s.buf.GetID() == 0 {
		return nil
	}
	ptr := s.buf.Contents()
	if ptr == nil {
		return nil
	}
	return unsafe.Slice((*T)(ptr), s.len)
}

// Slice returns a sub-slice of the GPU buffer.
func (s Slice[T]) Slice(start, end int) (Slice[T], error) {
	if start < 0 || end < start || end > s.cap {
		return Slice[T]{}, fmt.Errorf("metalmem: slice bounds out of range [%d:%d] with cap %d", start, end, s.cap)
	}
	return Slice[T]{
		buf:      s.buf,
		len:      end - start,
		cap:      s.cap - start,
		elemSize: s.elemSize,
	}, nil
}

// CopyFromHost synchronously copies data from a CPU slice into the unified Metal buffer.
func (s Slice[T]) CopyFromHost(src []T) error {
	if len(src) == 0 {
		return nil
	}
	if len(src) > s.len {
		return fmt.Errorf("metalmem: source length %d exceeds destination length %d", len(src), s.len)
	}
	dstPtr := s.buf.Contents()
	if dstPtr == nil {
		return fmt.Errorf("metalmem: buffer contents pointer is nil")
	}
	srcPtr := unsafe.Pointer(&src[0])
	bytesToCopy := uintptr(len(src)) * s.elemSize

	// Direct memory copy in unified memory address space
	copy(unsafe.Slice((*byte)(dstPtr), bytesToCopy), unsafe.Slice((*byte)(srcPtr), bytesToCopy))
	return nil
}

// CopyToHost synchronously copies data from the unified Metal buffer into a CPU slice.
func (s Slice[T]) CopyToHost(dst []T) error {
	if len(dst) == 0 {
		return nil
	}
	if len(dst) < s.len {
		return fmt.Errorf("metalmem: destination length %d smaller than source length %d", len(dst), s.len)
	}
	srcPtr := s.buf.Contents()
	if srcPtr == nil {
		return fmt.Errorf("metalmem: buffer contents pointer is nil")
	}
	dstPtr := unsafe.Pointer(&dst[0])
	bytesToCopy := uintptr(s.len) * s.elemSize

	copy(unsafe.Slice((*byte)(dstPtr), bytesToCopy), unsafe.Slice((*byte)(srcPtr), bytesToCopy))
	return nil
}

// Pool provides a recycled slab cache for transient activations.
type Pool struct {
	mu    sync.Mutex
	dev   *applemetal.Device
	slabs map[int][]metal.MTLBuffer
}

// NewPool creates an allocation recycling pool.
func NewPool(dev *applemetal.Device) *Pool {
	return &Pool{
		dev:   dev,
		slabs: make(map[int][]metal.MTLBuffer),
	}
}

// Get retrieves or allocates a buffer with at least minBytes.
func (p *Pool) Get(minBytes int) (metal.MTLBuffer, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	bucket := 1024
	for bucket < minBytes {
		bucket *= 2
	}

	if list := p.slabs[bucket]; len(list) > 0 {
		buf := list[len(list)-1]
		p.slabs[bucket] = list[:len(list)-1]
		return buf, nil
	}

	return p.dev.AllocBuffer(bucket)
}

// Put returns a buffer back to the recycling pool.
func (p *Pool) Put(buf metal.MTLBuffer, capacity int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	bucket := 1024
	for bucket < capacity {
		bucket *= 2
	}
	p.slabs[bucket] = append(p.slabs[bucket], buf)
}
