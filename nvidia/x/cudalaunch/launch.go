package cudalaunch

import (
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)


// Dim3 specifies 3-dimensional execution configurations for grids and blocks.
type Dim3 struct {
	X uint32
	Y uint32
	Z uint32
}

// Dims creates a Dim3 configuration. Unspecified Y or Z dimensions default to 1.
func Dims(x uint32, yz ...uint32) Dim3 {
	d := Dim3{X: x, Y: 1, Z: 1}
	if len(yz) > 0 {
		d.Y = yz[0]
	}
	if len(yz) > 1 {
		d.Z = yz[1]
	}
	return d
}

// DevicePtrProvider represents any type that exposes a raw GPU device pointer.
type DevicePtrProvider interface {
	Ptr() driver.DevicePtr
}

// Launcher builds and executes a CUDA kernel dispatch.
type Launcher struct {
	fn     driver.Function
	grid   Dim3
	block  Dim3
	shmem  uint32
	stream *cuda.Stream
}

// Kernel creates a new Launcher for the specified driver function.
func Kernel(fn driver.Function) *Launcher {
	return &Launcher{
		fn:    fn,
		grid:  Dim3{X: 1, Y: 1, Z: 1},
		block: Dim3{X: 1, Y: 1, Z: 1},
	}
}

// Grid configures the grid dimensions (in blocks).
func (l *Launcher) Grid(x uint32, yz ...uint32) *Launcher {
	l.grid = Dims(x, yz...)
	return l
}

// Block configures the block dimensions (in threads).
func (l *Launcher) Block(x uint32, yz ...uint32) *Launcher {
	l.block = Dims(x, yz...)
	return l
}

// SharedMem configures the dynamic shared memory size in bytes per block.
func (l *Launcher) SharedMem(bytes uint32) *Launcher {
	l.shmem = bytes
	return l
}

// On assigns the CUDA stream for asynchronous execution.
func (l *Launcher) On(s *cuda.Stream) *Launcher {
	l.stream = s
	return l
}

// Launch packs the provided Go arguments and dispatches the kernel.
func (l *Launcher) Launch(args ...any) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var streamHandle driver.Stream
	if l.stream != nil {
		streamHandle = l.stream.Raw()
	}

	kernelParams, cleanup, err := packArgs(args)
	if err != nil {
		return fmt.Errorf("cudalaunch: pack args: %w", err)
	}
	defer cleanup()

	err = l.fn.LaunchKernel(
		l.grid.X, l.grid.Y, l.grid.Z,
		l.block.X, l.block.Y, l.block.Z,
		l.shmem,
		streamHandle,
		kernelParams,
	)
	if err != nil {
		return fmt.Errorf("cudalaunch: launch kernel: %w", err)
	}
	return nil
}


// packArgs marshals Go values into an array of pointers as expected by cuLaunchKernel.
func packArgs(args []any) ([]unsafe.Pointer, func(), error) {
	if len(args) == 0 {
		return nil, func() {}, nil
	}

	params := make([]unsafe.Pointer, len(args))
	heapCopies := make([]any, len(args))
	var pinner runtime.Pinner

	for i, arg := range args {
		if arg == nil {
			return nil, nil, fmt.Errorf("arg %d is nil", i)
		}

		// Check if argument satisfies DevicePtrProvider (e.g. cudamem.Slice)
		if provider, ok := arg.(DevicePtrProvider); ok {
			ptrVal := provider.Ptr()
			if uintptr(ptrVal)%8 != 0 {
				return nil, nil, fmt.Errorf("arg %d: device pointer 0x%x is not 8-byte aligned", i, ptrVal)
			}
			heapCopies[i] = ptrVal
			params[i] = unsafe.Pointer(&ptrVal)
			continue
		}

		switch v := arg.(type) {
		case driver.DevicePtr:
			if uintptr(v)%8 != 0 {
				return nil, nil, fmt.Errorf("arg %d: device pointer 0x%x is not 8-byte aligned", i, v)
			}
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)

		case int8:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case int16:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case int32:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case int64:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case int:
			val := int64(v)
			heapCopies[i] = val
			params[i] = unsafe.Pointer(&val)
		case uint8:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case uint16:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case uint32:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case uint64:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case uint:
			val := uint64(v)
			heapCopies[i] = val
			params[i] = unsafe.Pointer(&val)
		case uintptr:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case float32:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case float64:
			heapCopies[i] = v
			params[i] = unsafe.Pointer(&v)
		case bool:
			var byteVal uint8
			if v {
				byteVal = 1
			}
			heapCopies[i] = byteVal
			params[i] = unsafe.Pointer(&byteVal)
		default:
			rv := reflect.ValueOf(arg)
			if rv.Kind() == reflect.Pointer {
				if rv.IsNil() {
					return nil, nil, fmt.Errorf("arg %d is a nil pointer", i)
				}
				p := rv.UnsafePointer()
				params[i] = p
				continue
			}
			if rv.Kind() == reflect.Struct {
				// Allocate new heap copy of struct
				copyPtr := reflect.New(rv.Type())
				copyPtr.Elem().Set(rv)
				ptr := copyPtr.UnsafePointer()
				params[i] = ptr
				heapCopies[i] = copyPtr.Interface()
				continue
			}
			return nil, nil, fmt.Errorf("arg %d has unsupported type %T", i, arg)
		}
	}

	cleanup := func() {
		pinner.Unpin()
		_ = heapCopies // retain references
	}

	return params, cleanup, nil
}

// OccupancyMaxPotentialBlockSize calculates the maximum potential block size and minimum
// number of blocks per SM for a kernel function.
func OccupancyMaxPotentialBlockSize(fn driver.Function, dynamicSMemBytes uint, blockSizeLimit int) (minGridSize, blockSize int, err error) {
	var minGrid int32
	var block int32
	err = fn.MaxPotentialBlockSize(&minGrid, &block, 0, dynamicSMemBytes, int32(blockSizeLimit))
	if err != nil {
		return 0, 0, fmt.Errorf("cudalaunch: occupancy calculation: %w", err)
	}
	return int(minGrid), int(block), nil
}

