package cudamem

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// Slice represents a typed, contiguous slice of memory on the GPU. Copies and
// subslices share ownership: freeing any one releases the whole allocation once,
// after which none of them may be used.
type Slice[T any] struct {
	ptr      driver.DevicePtr
	len      int
	cap      int
	elemSize uintptr
	alloc    *allocation
}

type allocation struct {
	mu      sync.Mutex
	users   int
	ptr     driver.DevicePtr
	bytes   uint64
	tracker *Tracker
	device  *cuda.Device
}

// Alloc allocates a new typed Slice[T] of length elements in GPU device memory.
func Alloc[T any](dev *cuda.Device, length int) (Slice[T], error) {
	return AllocTracked[T](dev, nil, length)
}

// AllocTracked allocates device memory and accounts for its lifetime in tracker.
func AllocTracked[T any](dev *cuda.Device, tracker *Tracker, length int) (Slice[T], error) {
	if length < 0 {
		return Slice[T]{}, errors.New("cudamem: negative allocation length")
	}
	var t T
	elemSize := unsafe.Sizeof(t)
	if elemSize == 0 {
		elemSize = 1
	}
	if uintptr(length) > uintptr(^uint(0))/elemSize {
		return Slice[T]{}, fmt.Errorf("cudamem: allocation size overflows for %d elements", length)
	}
	totalBytes := uint(uintptr(length) * elemSize)

	var dptr driver.DevicePtr
	err := dev.WithContext(func() error {
		p, err := driver.MemAlloc(totalBytes)
		if err != nil {
			return err
		}
		dptr = p
		return nil
	})
	if err != nil {
		return Slice[T]{}, fmt.Errorf("cudamem: alloc %d elements (%d bytes): %w", length, totalBytes, err)
	}

	tracker.allocated(uint64(totalBytes))
	return Slice[T]{
		ptr:      dptr,
		len:      length,
		cap:      length,
		elemSize: elemSize,
		alloc:    &allocation{ptr: dptr, bytes: uint64(totalBytes), tracker: tracker, device: dev},
	}, nil
}

// AllocManaged allocates a new typed Slice[T] of length elements in Unified/Managed memory.
func AllocManaged[T any](dev *cuda.Device, length int) (Slice[T], error) {
	if length < 0 {
		return Slice[T]{}, errors.New("cudamem: negative allocation length")
	}
	var t T
	elemSize := unsafe.Sizeof(t)
	if elemSize == 0 {
		elemSize = 1
	}
	totalBytes := uint(uintptr(length) * elemSize)

	var dptr driver.DevicePtr
	err := dev.WithContext(func() error {
		p, err := driver.MemAllocManaged(totalBytes, uint32(driver.MemAttachGlobal))
		if err != nil {
			return err
		}
		dptr = p
		return nil
	})
	if err != nil {
		return Slice[T]{}, fmt.Errorf("cudamem: alloc managed %d elements (%d bytes): %w", length, totalBytes, err)
	}

	return Slice[T]{
		ptr:      dptr,
		len:      length,
		cap:      length,
		elemSize: elemSize,
		alloc:    &allocation{ptr: dptr, bytes: uint64(totalBytes), device: dev},
	}, nil
}

// Ptr returns the raw device pointer address.
func (s Slice[T]) Ptr() driver.DevicePtr {
	if s.alloc != nil {
		s.alloc.mu.Lock()
		defer s.alloc.mu.Unlock()
		if s.alloc.ptr == 0 {
			return 0
		}
	}
	return s.ptr
}

// Len returns the number of elements in the device slice.
func (s Slice[T]) Len() int {
	return s.len
}

// Cap returns the capacity of the device slice.
func (s Slice[T]) Cap() int {
	return s.cap
}

// SizeBytes returns the total size of the active slice in bytes.
func (s Slice[T]) SizeBytes() uintptr {
	return uintptr(s.len) * s.elemSize
}

// Slice creates a subslice of s with bounds [low:high].
func (s Slice[T]) Slice(low, high int) Slice[T] {
	if low < 0 || high < low || high > s.cap {
		panic(fmt.Sprintf("cudamem: bounds out of range [%d:%d] with cap %d", low, high, s.cap))
	}
	newPtr := s.ptr + driver.DevicePtr(uintptr(low)*s.elemSize)
	return Slice[T]{
		ptr:      newPtr,
		len:      high - low,
		cap:      s.cap - low,
		elemSize: s.elemSize,
		alloc:    s.alloc,
	}
}

// Free deallocates the device memory. Repeated calls through copies or subslices
// of s are safe. Free returns ErrInUse while an owned asynchronous transfer is
// pending; synchronize or destroy its stream before retrying.
func (s Slice[T]) Free() error {
	if s.ptr == 0 {
		return nil
	}
	if s.alloc != nil {
		s.alloc.mu.Lock()
		defer s.alloc.mu.Unlock()
		if s.alloc.ptr == 0 {
			return nil
		}
		if s.alloc.users != 0 {
			return ErrInUse
		}
		if err := s.alloc.device.WithContext(s.alloc.ptr.MemFree); err != nil {
			return fmt.Errorf("cudamem: free device memory: %w", err)
		}
		s.alloc.ptr = 0
		s.alloc.tracker.freed(s.alloc.bytes)
		return nil
	}
	if err := s.ptr.MemFree(); err != nil {
		return fmt.Errorf("cudamem: free device memory: %w", err)
	}
	return nil
}

// CopyFromHost copies elements from a host Go slice into device memory.
// It uses runtime.Pinner to ensure the host slice is not relocated by GC during transfer.
func (s Slice[T]) CopyFromHost(src []T) error {
	if len(src) == 0 {
		return nil
	}
	if len(src) > s.len {
		return fmt.Errorf("cudamem: source length %d exceeds destination capacity %d", len(src), s.len)
	}

	var pinner runtime.Pinner
	pinner.Pin(&src[0])
	defer pinner.Unpin()

	hostPtr := unsafe.Pointer(&src[0])
	totalBytes := uint(uintptr(len(src)) * s.elemSize)

	if err := s.withPointer(func(ptr driver.DevicePtr) error { return ptr.MemcpyHtoD(hostPtr, totalBytes) }); err != nil {
		return fmt.Errorf("cudamem: copy host to device: %w", err)
	}
	return nil
}

// CopyToHost copies elements from device memory into a destination host Go slice.
// It uses runtime.Pinner to ensure the destination slice is not relocated by GC during transfer.
func (s Slice[T]) CopyToHost(dst []T) error {
	if len(dst) == 0 {
		return nil
	}
	if len(dst) < s.len {
		return fmt.Errorf("cudamem: destination length %d is smaller than source length %d", len(dst), s.len)
	}

	var pinner runtime.Pinner
	pinner.Pin(&dst[0])
	defer pinner.Unpin()

	hostPtr := unsafe.Pointer(&dst[0])
	totalBytes := uint(uintptr(s.len) * s.elemSize)

	if err := s.withPointer(func(ptr driver.DevicePtr) error { return ptr.MemcpyDtoH(hostPtr, totalBytes) }); err != nil {
		return fmt.Errorf("cudamem: copy device to host: %w", err)
	}
	return nil
}

// CopyFromHostAsync asynchronously enqueues a host-to-device copy onto the specified stream.
// The Go backing array and device allocation are retained until successful
// stream synchronization or destruction. runtime.Pinner provides Go address
// stability, not CUDA page-locking; pageable transfers may block or be staged.
// The caller must not modify src until the transfer completes.
// Capturing the copy in a graph does not retain allocations for later replays;
// the caller must own all referenced resources for the graph's lifetime.
func (s Slice[T]) CopyFromHostAsync(src []T, stream *cuda.Stream) error {
	if len(src) == 0 {
		return nil
	}
	if len(src) > s.len {
		return fmt.Errorf("cudamem: source length %d exceeds destination capacity %d", len(src), s.len)
	}
	if stream == nil {
		return errors.New("cudamem: nil stream passed to CopyFromHostAsync")
	}
	release, err := s.retain(stream.Device())
	if err != nil {
		return err
	}

	var unpinOnce sync.Once
	pinner := new(runtime.Pinner)
	pinner.Pin(&src[0])
	unpin := func() {
		unpinOnce.Do(func() { pinner.Unpin(); release() })
	}
	hostPtr := unsafe.Pointer(&src[0])
	totalBytes := uint(uintptr(len(src)) * s.elemSize)
	dev := stream.Device()
	attempted := false

	if err := s.withPointer(func(ptr driver.DevicePtr) error {
		return stream.WithHandle(func(handle driver.Stream) error {
			return dev.WithContext(func() error {
				attempted = true
				return ptr.MemcpyHtoDAsync(hostPtr, totalBytes, handle)
			})
		})
	}); err != nil {
		failedTransferCleanup(attempted, stream.AddCleanup, unpin)
		return fmt.Errorf("cudamem: async copy host to device: %w", err)
	}
	stream.AddCleanup(unpin)
	return nil
}

// CopyToHostAsync asynchronously enqueues a device-to-host copy onto the specified stream.
// The Go backing array and device allocation are retained until successful
// stream synchronization or destruction. runtime.Pinner does not CUDA-page-lock
// the backing array. The caller must not access dst until the transfer completes.
func (s Slice[T]) CopyToHostAsync(dst []T, stream *cuda.Stream) error {
	if len(dst) == 0 {
		return nil
	}
	if len(dst) < s.len {
		return fmt.Errorf("cudamem: destination length %d is smaller than source length %d", len(dst), s.len)
	}
	if stream == nil {
		return errors.New("cudamem: nil stream passed to CopyToHostAsync")
	}
	release, err := s.retain(stream.Device())
	if err != nil {
		return err
	}

	var unpinOnce sync.Once
	pinner := new(runtime.Pinner)
	pinner.Pin(&dst[0])
	unpin := func() {
		unpinOnce.Do(func() { pinner.Unpin(); release() })
	}
	hostPtr := unsafe.Pointer(&dst[0])
	totalBytes := uint(uintptr(s.len) * s.elemSize)
	dev := stream.Device()
	attempted := false

	if err := s.withPointer(func(ptr driver.DevicePtr) error {
		return stream.WithHandle(func(handle driver.Stream) error {
			return dev.WithContext(func() error {
				attempted = true
				return ptr.MemcpyDtoHAsync(hostPtr, totalBytes, handle)
			})
		})
	}); err != nil {
		failedTransferCleanup(attempted, stream.AddCleanup, unpin)
		return fmt.Errorf("cudamem: async copy device to host: %w", err)
	}
	stream.AddCleanup(unpin)
	return nil
}

// CopyDeviceToDevice copies elements from another device slice on the same device.
func (s Slice[T]) CopyDeviceToDevice(src Slice[T]) error {
	if src.len == 0 {
		return nil
	}
	if src.len > s.len {
		return fmt.Errorf("cudamem: source length %d exceeds destination capacity %d", src.len, s.len)
	}
	totalBytes := uint(uintptr(src.len) * s.elemSize)
	if err := s.withSource(src, func(dst, src driver.DevicePtr) error {
		return dst.MemcpyDtoD(src, totalBytes)
	}); err != nil {
		return fmt.Errorf("cudamem: copy device to device: %w", err)
	}
	return nil
}

// CopyDeviceToDeviceAsync asynchronously enqueues a device-to-device copy onto the specified stream.
// Both allocations are retained until successful stream synchronization or
// destruction. Free returns ErrInUse until then.
func (s Slice[T]) CopyDeviceToDeviceAsync(src Slice[T], stream *cuda.Stream) error {
	if src.len == 0 {
		return nil
	}
	if src.len > s.len {
		return fmt.Errorf("cudamem: source length %d exceeds destination capacity %d", src.len, s.len)
	}
	if stream == nil {
		return errors.New("cudamem: nil stream passed to CopyDeviceToDeviceAsync")
	}
	releaseDst, err := s.retain(stream.Device())
	if err != nil {
		return err
	}
	releaseSrc, err := src.retain(stream.Device())
	if err != nil {
		releaseDst()
		return err
	}
	release := func() { releaseSrc(); releaseDst() }
	totalBytes := uint(uintptr(src.len) * s.elemSize)
	dev := stream.Device()
	attempted := false
	if err := s.withSource(src, func(dst, src driver.DevicePtr) error {
		return stream.WithHandle(func(handle driver.Stream) error {
			return dev.WithContext(func() error {
				attempted = true
				return dst.MemcpyDtoDAsync(src, totalBytes, handle)
			})
		})
	}); err != nil {
		failedTransferCleanup(attempted, stream.AddCleanup, release)
		return fmt.Errorf("cudamem: async copy device to device: %w", err)
	}
	stream.AddCleanup(release)
	return nil
}

// HostAllocation represents page-locked (pinned) host memory accessible to both the CPU and GPU.
// Copies share ownership and may release the allocation at most once.
type HostAllocation[T any] struct {
	ptr   uintptr
	len   int
	slice []T
	alloc *hostAllocation
}

type hostAllocation struct {
	mu    sync.Mutex
	ptr   uintptr
	users int
}

// AllocHost allocates page-locked host memory accessible by CUDA devices. The
// caller must have made the intended CUDA context current, for example with
// [cuda.Device.WithContext].
func AllocHost[T any](length int) (HostAllocation[T], error) {
	if length < 0 {
		return HostAllocation[T]{}, errors.New("cudamem: negative allocation length")
	}
	var t T
	elemSize := unsafe.Sizeof(t)
	if elemSize == 0 {
		elemSize = 1
	}
	totalBytes := uint(uintptr(length) * elemSize)

	var rawPtr uintptr
	if err := driver.MemAllocHost(&rawPtr, totalBytes); err != nil {
		return HostAllocation[T]{}, fmt.Errorf("cudamem: alloc host %d elements (%d bytes): %w", length, totalBytes, err)
	}

	var slice []T
	header := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
	header.Data = rawPtr
	header.Len = length
	header.Cap = length

	return HostAllocation[T]{
		ptr:   rawPtr,
		len:   length,
		slice: slice,
		alloc: &hostAllocation{ptr: rawPtr},
	}, nil
}

// Ptr returns the raw host memory pointer address.
func (ha HostAllocation[T]) Ptr() uintptr {
	if ha.alloc != nil {
		ha.alloc.mu.Lock()
		defer ha.alloc.mu.Unlock()
		return ha.alloc.ptr
	}
	return ha.ptr
}

// Len returns the number of elements in the host allocation.
func (ha HostAllocation[T]) Len() int {
	if ha.alloc != nil {
		ha.alloc.mu.Lock()
		defer ha.alloc.mu.Unlock()
		if ha.alloc.ptr == 0 {
			return 0
		}
	}
	return ha.len
}

// Slice returns the accessible Go slice view of the pinned memory. The view is
// valid only until Free is called through any copy of ha. Callers must
// serialize all access to the returned slice with Free.
func (ha HostAllocation[T]) Slice() []T {
	if ha.alloc != nil {
		ha.alloc.mu.Lock()
		defer ha.alloc.mu.Unlock()
		if ha.alloc.ptr == 0 {
			return nil
		}
	}
	return ha.slice
}

// Free deallocates the page-locked host memory. Repeated calls through copies of
// ha are safe. Free returns ErrInUse while a pinned transfer retains ha.
func (ha HostAllocation[T]) Free() error {
	if ha.ptr == 0 {
		return nil
	}
	if ha.alloc == nil {
		return errors.New("cudamem: host allocation has no ownership record")
	}
	ha.alloc.mu.Lock()
	defer ha.alloc.mu.Unlock()
	if ha.alloc.ptr == 0 {
		return nil
	}
	if ha.alloc.users != 0 {
		return ErrInUse
	}
	if err := driver.MemFreeHost(ha.alloc.ptr); err != nil {
		return fmt.Errorf("cudamem: free host memory: %w", err)
	}
	ha.alloc.ptr = 0
	return nil
}

func (s Slice[T]) withPointer(fn func(driver.DevicePtr) error) error {
	if s.alloc == nil {
		return fn(s.ptr)
	}
	s.alloc.mu.Lock()
	defer s.alloc.mu.Unlock()
	if s.alloc.ptr == 0 {
		return errors.New("cudamem: allocation is freed")
	}
	return s.alloc.device.WithContext(func() error { return fn(s.ptr) })
}

func (s Slice[T]) withSource(src Slice[T], fn func(dst, src driver.DevicePtr) error) error {
	if s.alloc == nil || src.alloc == nil {
		return errors.New("cudamem: allocation has no ownership record")
	}
	if s.alloc == src.alloc {
		s.alloc.mu.Lock()
		defer s.alloc.mu.Unlock()
		if s.alloc.ptr == 0 {
			return errors.New("cudamem: allocation is freed")
		}
		return s.alloc.device.WithContext(func() error { return fn(s.ptr, src.ptr) })
	}
	first, second := s.alloc, src.alloc
	if uintptr(unsafe.Pointer(first)) > uintptr(unsafe.Pointer(second)) {
		first, second = second, first
	}
	first.mu.Lock()
	defer first.mu.Unlock()
	second.mu.Lock()
	defer second.mu.Unlock()
	if s.alloc.ptr == 0 {
		return errors.New("cudamem: destination allocation is freed")
	}
	if src.alloc.ptr == 0 {
		return errors.New("cudamem: source allocation is freed")
	}
	return s.alloc.device.WithContext(func() error { return fn(s.ptr, src.ptr) })
}
