package cudalaunch

import (
	"fmt"
	"reflect"
	"runtime"
	"sync/atomic"
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

// DeviceSlice represents a device buffer that also reports its length, such as
// [github.com/tmc/lib/nvidia/x/cudamem.Slice]. An empty buffer packs as a null
// device pointer, so an optional kernel argument can be passed as a zero Slice.
type DeviceSlice interface {
	DevicePtrProvider
	Len() int
}

// Launcher builds and executes a CUDA kernel dispatch.
type Launcher struct {
	fn      driver.Function
	owned   *cuda.Kernel
	grid    Dim3
	block   Dim3
	shmem   uint32
	stream  *cuda.Stream
	counter *Counter
	attrs   []driver.LaunchAttribute
}

// Counter counts successful CUDA kernel submissions.
type Counter struct{ n atomic.Uint64 }

// Count adds successful launches to counter.
func (l *Launcher) Count(counter *Counter) *Launcher {
	l.counter = counter
	return l
}

// Value returns the number of successful launches.
func (c *Counter) Value() uint64 {
	if c == nil {
		return 0
	}
	return c.n.Load()
}

// Kernel creates a new Launcher for the specified driver function.
func Kernel(fn driver.Function) *Launcher {
	return &Launcher{
		fn:    fn,
		grid:  Dim3{X: 1, Y: 1, Z: 1},
		block: Dim3{X: 1, Y: 1, Z: 1},
	}
}

// OwnedKernel creates a launcher that checks the kernel's module lifetime.
// The stream and device arguments must belong to the module's context.
func OwnedKernel(fn *cuda.Kernel) *Launcher {
	l := Kernel(0)
	l.owned = fn
	return l
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

// WithAttrs appends launch attributes to the kernel configuration.
func (l *Launcher) WithAttrs(attrs ...driver.LaunchAttribute) *Launcher {
	l.attrs = append(l.attrs, attrs...)
	return l
}

// Attrs returns a copy of configured launch attributes.
func (l *Launcher) Attrs() []driver.LaunchAttribute {
	if len(l.attrs) == 0 {
		return nil
	}
	out := make([]driver.LaunchAttribute, len(l.attrs))
	copy(out, l.attrs)
	return out
}

// WithProgrammaticStreamSerialization enables or disables programmatic stream
// serialization (Programmatic Dependent Launch) for this launch.
// When enabled, subsequent work in the stream is released when all CTAs
// in this grid execute griddepcontrol.launch_dependents.
func (l *Launcher) WithProgrammaticStreamSerialization(enable bool) *Launcher {
	return l.WithAttrs(ProgrammaticStreamSerializationAttr(enable))
}

// WithCooperative enables or disables cooperative grid launch.
func (l *Launcher) WithCooperative(enable bool) *Launcher {
	return l.WithAttrs(CooperativeAttr(enable))
}

// ProgrammaticStreamSerializationAttr returns a LaunchAttribute enabling or disabling
// programmatic stream serialization (PDL) for cuLaunchKernelEx.
func ProgrammaticStreamSerializationAttr(enable bool) driver.LaunchAttribute {
	var val uint64
	if enable {
		val = 1
	}
	return driver.LaunchAttribute{
		Id:    driver.LaunchAttributeProgrammaticStreamSerialization,
		Value: [8]uint64{val},
	}
}

// CooperativeAttr returns a LaunchAttribute enabling or disabling cooperative grid launch.
func CooperativeAttr(enable bool) driver.LaunchAttribute {
	var val uint64
	if enable {
		val = 1
	}
	return driver.LaunchAttribute{
		Id:    driver.LaunchAttributeCooperative,
		Value: [8]uint64{val},
	}
}

// Launch packs the provided Go arguments and dispatches the kernel.
func (l *Launcher) Launch(args ...any) error {
	if l.owned == nil && l.fn == 0 {
		return fmt.Errorf("cudalaunch: nil kernel")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	kernelParams, cleanup, err := packArgs(args)
	if err != nil {
		return fmt.Errorf("cudalaunch: pack args: %w", err)
	}
	defer cleanup()

	var pinner runtime.Pinner
	if len(l.attrs) > 0 {
		pinner.Pin(&l.attrs[0])
		defer pinner.Unpin()
	}

	launch := func(stream driver.Stream) error {
		if len(l.attrs) > 0 {
			cfg := driver.LaunchConfig{
				GridDimX:       l.grid.X,
				GridDimY:       l.grid.Y,
				GridDimZ:       l.grid.Z,
				BlockDimX:      l.block.X,
				BlockDimY:      l.block.Y,
				BlockDimZ:      l.block.Z,
				SharedMemBytes: l.shmem,
				HStream:        stream,
				Attrs:          uintptr(unsafe.Pointer(&l.attrs[0])),
				NumAttrs:       uint32(len(l.attrs)),
			}
			if l.owned != nil {
				return l.owned.LaunchEx(&cfg, kernelParams)
			}
			return l.fn.LaunchKernelEx(&cfg, kernelParams, nil)
		}

		if l.owned != nil {
			return l.owned.Launch([3]uint32{l.grid.X, l.grid.Y, l.grid.Z}, [3]uint32{l.block.X, l.block.Y, l.block.Z}, l.shmem, stream, kernelParams)
		}
		return l.fn.LaunchKernel(
			l.grid.X, l.grid.Y, l.grid.Z,
			l.block.X, l.block.Y, l.block.Z,
			l.shmem,
			stream,
			kernelParams,
		)
	}
	if l.stream == nil {
		err = launch(0)
	} else {
		dev := l.stream.Device()
		err = l.stream.WithHandle(func(stream driver.Stream) error {
			if l.owned != nil {
				return launch(stream)
			}
			return dev.WithContext(func() error { return launch(stream) })
		})
	}
	if err != nil {
		return fmt.Errorf("cudalaunch: launch kernel: %w", err)
	}
	if l.counter != nil {
		l.counter.n.Add(1)
	}
	return nil
}

// PackArgs marshals Go values into an array of pointers as expected by cuLaunchKernel.
// Struct arguments are guaranteed to have 16-byte alignment per NVCC ABI requirements.
func PackArgs(args ...any) ([]unsafe.Pointer, func(), error) {
	return packArgs(args)
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

		// Check if argument satisfies DevicePtrProvider (e.g. cudamem.Slice).
		// An empty DeviceSlice packs as a null pointer so that optional kernel
		// arguments can be passed as a zero slice.
		if provider, ok := arg.(DevicePtrProvider); ok {
			ptrVal := provider.Ptr()
			if s, ok := arg.(DeviceSlice); ok && s.Len() == 0 {
				ptrVal = 0
			}
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
				sz := rv.Type().Size()
				// Ensure 16-byte alignment for NVCC ABI compatibility (e.g. float4, uint4)
				buf := make([]byte, sz+15)
				offset := (16 - (uintptr(unsafe.Pointer(&buf[0])) % 16)) % 16
				alignedPtr := unsafe.Pointer(uintptr(unsafe.Pointer(&buf[0])) + offset)

				// Copy struct data into 16-byte aligned buffer
				copyPtr := reflect.New(rv.Type())
				copyPtr.Elem().Set(rv)
				srcSlice := unsafe.Slice((*byte)(copyPtr.UnsafePointer()), sz)
				dstSlice := unsafe.Slice((*byte)(alignedPtr), sz)
				copy(dstSlice, srcSlice)

				params[i] = alignedPtr
				heapCopies[i] = buf
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
	minGrid, block, err := fn.MaxPotentialBlockSize(0, dynamicSMemBytes, int32(blockSizeLimit))
	if err != nil {
		return 0, 0, fmt.Errorf("cudalaunch: occupancy calculation: %w", err)
	}
	return int(minGrid), int(block), nil
}
