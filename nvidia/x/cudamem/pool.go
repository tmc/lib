package cudamem

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// Bucketing quanta. These are policy choices made by this package to limit
// memory pool fragmentation; they are not device properties, and they are
// deliberately not queried from the driver.
//
// [driver.Stream.MemAllocAsyncBytes], the allocator [AllocAsync] uses, documents no
// alignment or granularity requirement on its bytesize argument, and no
// [driver.MempoolAttribute] reports the pool's internal sub-allocation quantum.
// The driver's granularity query, [driver.MemGetAllocationGranularity], governs
// [driver.MemCreate] VMM allocations instead — a different allocator, whose size
// genuinely must be a multiple of the queried value; package nvlshm queries it
// for that path. The pool's own backing allocations are documented under
// [driver.MemoryPool.TrimTo] as happening "at different granularity from the
// user allocations", so rounding a request here does not align the pool's
// OS-level allocations either.
//
// Changing these values changes fragmentation and waste, never correctness.
const (
	smallQuantum = 256             // tier 1: rounding quantum
	smallMax     = 1024            // tier 1: largest request
	pow2Max      = 1024 * 1024     // tier 2: largest request
	largeQuantum = 2 * 1024 * 1024 // tier 3: rounding quantum
)

// BucketSize returns the quantized allocation size used for a request of bytes,
// which limits memory pool fragmentation under variable sequence lengths:
//   - <= 1 KiB: rounded up to a multiple of 256 bytes
//   - 1 KiB - 1 MiB: rounded up to the next power of two
//   - > 1 MiB: rounded up to a multiple of 2 MiB
//
// The result is never smaller than bytes. A request so large that rounding it up
// would overflow uint is returned unrounded, so that a caller allocates, and
// fails on, the size it asked for rather than silently receiving a smaller one.
func BucketSize(bytes uint) uint {
	if bytes == 0 {
		return smallQuantum
	}
	if bytes <= smallMax {
		return (bytes + smallQuantum - 1) &^ (smallQuantum - 1)
	}
	if bytes <= pow2Max {
		return 1 << uint(bits.Len64(uint64(bytes-1)))
	}
	if bytes > math.MaxUint-(largeQuantum-1) {
		return bytes
	}
	return (bytes + largeQuantum - 1) &^ (largeQuantum - 1)
}

// AllocAsync allocates a new typed Slice[T] of length elements ordered within stream.
// It allocates from the default memory pool associated with stream's device, using
// tiered size bucketing to prevent pool fragmentation.
// A nonzero error-result slice remains owned by the caller. Queue FreeAsync on
// the same stream, or synchronize that stream before using synchronous Free.
// Preserve the stream and device until queued allocation/free work completes.
func AllocAsync[T any](stream *cuda.Stream, length int) (Slice[T], error) {
	if length < 0 {
		return Slice[T]{}, errors.New("cudamem: negative allocation length")
	}
	if stream == nil {
		return Slice[T]{}, errors.New("cudamem: nil stream passed to AllocAsync")
	}
	dev := stream.Device()
	return allocateAsyncSlice[T](dev, length, func(fn func(driver.Stream) error) error {
		return stream.WithHandle(func(handle driver.Stream) error {
			return dev.WithContext(func() error { return fn(handle) })
		})
	}, driver.Stream.MemAllocAsyncBytes)
}

func allocateAsyncSlice[T any](dev *cuda.Device, length int, run func(func(driver.Stream) error) error, allocate func(driver.Stream, uint) (driver.DevicePtr, error)) (Slice[T], error) {
	if length < 0 {
		return Slice[T]{}, errors.New("cudamem: negative allocation length")
	}
	var t T
	elemSize := unsafe.Sizeof(t)
	if elemSize == 0 {
		elemSize = 1
	}
	if uintptr(length) > uintptr(^uint(0))/elemSize {
		return Slice[T]{}, fmt.Errorf("cudamem: async allocation size overflows for %d elements", length)
	}
	totalBytes := uint(uintptr(length) * elemSize)
	allocBytes := BucketSize(totalBytes)
	capacity := uintptr(allocBytes) / elemSize
	if capacity > uintptr(math.MaxInt) {
		return Slice[T]{}, errors.New("cudamem: async allocation capacity overflows int")
	}

	var dptr driver.DevicePtr
	err := run(func(handle driver.Stream) error {
		var err error
		dptr, err = allocate(handle, allocBytes)
		return err
	})
	var slice Slice[T]
	if dptr != 0 {
		slice = Slice[T]{ptr: dptr, len: length, cap: int(capacity), elemSize: elemSize,
			alloc: &allocation{ptr: dptr, bytes: uint64(allocBytes), device: dev}}
	}
	if err != nil {
		return slice, fmt.Errorf("cudamem: async alloc %d elements (%d bytes, bucket %d): %w", length, totalBytes, allocBytes, err)
	}
	return slice, nil
}

// FreeAsync asynchronously deallocates the device memory within the specified stream.
// Repeated calls through copies or subslices of s are safe. Tracked bytes remain live
// until the stream is synchronized.
func (s Slice[T]) FreeAsync(stream *cuda.Stream) error {
	if s.ptr == 0 {
		return nil
	}
	if stream == nil {
		return errors.New("cudamem: nil stream passed to FreeAsync")
	}
	if s.alloc == nil {
		return errors.New("cudamem: async allocation has no ownership record")
	}
	dev := stream.Device()
	return s.alloc.freeAsync(func(fn func(driver.Stream) error) error {
		return stream.WithHandle(func(handle driver.Stream) error {
			return dev.WithContext(func() error {
				return fn(handle)
			})
		})
	}, driver.DevicePtr.FreeAsync, stream.AddCleanup)
}

func (a *allocation) freeAsync(run func(func(driver.Stream) error) error, free func(driver.DevicePtr, driver.Stream) error, addCleanup func(func())) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.ptr == 0 {
		return nil
	}
	queued := false
	err := run(func(stream driver.Stream) error {
		if err := free(a.ptr, stream); err != nil {
			return err
		}
		a.ptr = 0
		queued = true
		return nil
	})
	if queued {
		// Register even when context restoration failed after enqueue. Wait
		// until outside WithHandle; AddCleanup acquires the stream lifetime lock.
		tracker, bytes := a.tracker, a.bytes
		addCleanup(func() { tracker.freed(bytes) })
	}
	if err != nil {
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
	raw, err := dev.Raw().DefaultMemPool()
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
