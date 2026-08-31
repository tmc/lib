package cudamem

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// AllocAsync allocates a new typed Slice[T] of length elements ordered within stream.
// It allocates from the default memory pool associated with stream's device.
func AllocAsync[T any](stream *cuda.Stream, length int) (Slice[T], error) {
	if length < 0 {
		return Slice[T]{}, errors.New("cudamem: negative allocation length")
	}
	if stream == nil {
		return Slice[T]{}, errors.New("cudamem: nil stream passed to AllocAsync")
	}
	var t T
	elemSize := unsafe.Sizeof(t)
	if elemSize == 0 {
		elemSize = 1
	}
	totalBytes := uint(uintptr(length) * elemSize)

	dptr, err := stream.Raw().MemAllocAsync(totalBytes)
	if err != nil {
		return Slice[T]{}, fmt.Errorf("cudamem: async alloc %d elements (%d bytes): %w", length, totalBytes, err)
	}

	return Slice[T]{
		ptr:      dptr,
		len:      length,
		cap:      length,
		elemSize: elemSize,
	}, nil
}

// FreeAsync asynchronously deallocates the device memory within the specified stream.
func (s Slice[T]) FreeAsync(stream *cuda.Stream) error {
	if s.ptr == 0 {
		return nil
	}
	if stream == nil {
		return errors.New("cudamem: nil stream passed to FreeAsync")
	}
	if err := s.ptr.MemFreeAsync(stream.Raw()); err != nil {
		return fmt.Errorf("cudamem: async free device memory: %w", err)
	}
	return nil
}

// Pool represents a stream-ordered memory pool.
type Pool struct {
	raw    driver.MemoryPool
	device *cuda.Device
}

// DefaultPool returns the default memory pool for dev.
func DefaultPool(dev *cuda.Device) (*Pool, error) {
	if dev == nil {
		return nil, errors.New("cudamem: nil device passed to DefaultPool")
	}
	raw, err := dev.Raw().GetDefaultMemPool()
	if err != nil {
		return nil, fmt.Errorf("cudamem: get default mem pool: %w", err)
	}
	return &Pool{
		raw:    raw,
		device: dev,
	}, nil
}

// TrimTo releases unallocated memory held by the pool back to the OS until
// at most minBytesToKeep remain reserved.
func (p *Pool) TrimTo(minBytesToKeep uint64) error {
	if err := p.raw.TrimTo(uint(minBytesToKeep)); err != nil {
		return fmt.Errorf("cudamem: trim pool to %d bytes: %w", minBytesToKeep, err)
	}
	return nil
}
