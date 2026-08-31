package cudamem

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// Slice represents a typed, contiguous slice of memory on the GPU.
type Slice[T any] struct {
	ptr      driver.DevicePtr
	len      int
	cap      int
	elemSize uintptr
}

// Alloc allocates a new typed Slice[T] of length elements in GPU device memory.
func Alloc[T any](dev *cuda.Device, length int) (Slice[T], error) {
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

	return Slice[T]{
		ptr:      dptr,
		len:      length,
		cap:      length,
		elemSize: elemSize,
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
	}, nil
}

// Ptr returns the raw device pointer address.
func (s Slice[T]) Ptr() driver.DevicePtr {
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
	}
}

// Free deallocates the device memory.
func (s Slice[T]) Free() error {
	if s.ptr == 0 {
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

	if err := s.ptr.MemcpyHtoD(hostPtr, totalBytes); err != nil {
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

	if err := s.ptr.MemcpyDtoH(hostPtr, totalBytes); err != nil {
		return fmt.Errorf("cudamem: copy device to host: %w", err)
	}
	return nil
}

// CopyFromHostAsync asynchronously enqueues a host-to-device copy onto the specified stream.
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

	var pinner runtime.Pinner
	pinner.Pin(&src[0])
	defer pinner.Unpin()

	hostPtr := unsafe.Pointer(&src[0])
	totalBytes := uint(uintptr(len(src)) * s.elemSize)

	if err := s.ptr.MemcpyHtoDAsync(hostPtr, totalBytes, stream.Raw()); err != nil {
		return fmt.Errorf("cudamem: async copy host to device: %w", err)
	}
	return nil
}

// CopyToHostAsync asynchronously enqueues a device-to-host copy onto the specified stream.
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

	var pinner runtime.Pinner
	pinner.Pin(&dst[0])
	defer pinner.Unpin()

	hostPtr := unsafe.Pointer(&dst[0])
	totalBytes := uint(uintptr(s.len) * s.elemSize)

	if err := s.ptr.MemcpyDtoHAsync(hostPtr, totalBytes, stream.Raw()); err != nil {
		return fmt.Errorf("cudamem: async copy device to host: %w", err)
	}
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
	if err := s.ptr.MemcpyDtoD(src.ptr, totalBytes); err != nil {
		return fmt.Errorf("cudamem: copy device to device: %w", err)
	}
	return nil
}

// CopyDeviceToDeviceAsync asynchronously enqueues a device-to-device copy onto the specified stream.
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
	totalBytes := uint(uintptr(src.len) * s.elemSize)
	if err := s.ptr.MemcpyDtoDAsync(src.ptr, totalBytes, stream.Raw()); err != nil {
		return fmt.Errorf("cudamem: async copy device to device: %w", err)
	}
	return nil
}

// HostAllocation represents page-locked (pinned) host memory accessible to both the CPU and GPU.
type HostAllocation[T any] struct {
	ptr   uintptr
	len   int
	slice []T
}

// AllocHost allocates page-locked host memory accessible by CUDA devices.
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
	}, nil
}

// Ptr returns the raw host memory pointer address.
func (ha HostAllocation[T]) Ptr() uintptr {
	return ha.ptr
}

// Len returns the number of elements in the host allocation.
func (ha HostAllocation[T]) Len() int {
	return ha.len
}

// Slice returns the accessible Go slice view of the pinned memory.
func (ha HostAllocation[T]) Slice() []T {
	return ha.slice
}

// Free deallocates the page-locked host memory.
func (ha HostAllocation[T]) Free() error {
	if ha.ptr == 0 {
		return nil
	}
	if err := driver.MemFreeHost(ha.ptr); err != nil {
		return fmt.Errorf("cudamem: free host memory: %w", err)
	}
	return nil
}
