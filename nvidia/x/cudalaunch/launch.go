package cudalaunch

import (
	"encoding/binary"
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
	sig     *Signature
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

// OptimalBlockSize asks the CUDA driver for a block size with reasonable
// occupancy and the minimum grid size needed to occupy the device.
// It is a resource estimate, not a throughput measurement.
func OptimalBlockSize(dev *cuda.Device, fn *cuda.Kernel, dynamicSMem int) (blockSize, minGridSize int, err error) {
	var owner *cuda.Device
	if fn != nil {
		owner = fn.Device()
	}
	if dev == nil || owner == nil || owner.Raw() != dev.Raw() {
		return 0, 0, fmt.Errorf("cudalaunch: kernel does not belong to device")
	}
	if dynamicSMem < 0 || uint64(dynamicSMem) > uint64(^uint32(0)) {
		return 0, 0, fmt.Errorf("cudalaunch: invalid dynamic shared memory size %d", dynamicSMem)
	}
	grid, block, err := fn.PotentialBlockSize(uint(dynamicSMem), 0)
	if err != nil {
		return 0, 0, fmt.Errorf("cudalaunch: choose block size: %w", err)
	}
	return block, grid, nil
}

// AutoBlock sets a one-dimensional block size selected by the driver.
// The caller still chooses the grid size based on its work count.
func (l *Launcher) AutoBlock(dev *cuda.Device, dynamicSMem int) error {
	if l == nil || l.owned == nil {
		return fmt.Errorf("cudalaunch: AutoBlock requires an owned kernel")
	}
	block, _, err := OptimalBlockSize(dev, l.owned, dynamicSMem)
	if err != nil {
		return err
	}
	l.block = Dims(uint32(block))
	l.shmem = uint32(dynamicSMem)
	return nil
}

// CheckSharedMemLimit verifies whether reqShmem dynamic shared memory bytes can be allocated
// on dev given staticSmem bytes of static shared memory. It returns the maximum dynamic shared
// memory allowed by the device opt-in limit.
func CheckSharedMemLimit(dev *cuda.Device, staticSmem, reqShmem int) (int, error) {
	if dev == nil {
		return 0, fmt.Errorf("cudalaunch: nil device")
	}
	if staticSmem < 0 || reqShmem < 0 {
		return 0, fmt.Errorf("cudalaunch: negative shared memory size")
	}
	devMaxOptin, err := dev.Attribute(driver.DevAttrMaxSharedMemoryPerBlockOptin)
	if err != nil {
		return 0, fmt.Errorf("cudalaunch: query device max shared memory opt-in: %w", err)
	}
	maxDynamic := devMaxOptin - staticSmem
	if maxDynamic < 0 {
		maxDynamic = 0
	}
	if reqShmem > maxDynamic {
		return maxDynamic, fmt.Errorf("cudalaunch: requested dynamic shared memory %d bytes exceeds device maximum %d bytes (device limit %d, static allocation %d bytes)",
			reqShmem, maxDynamic, devMaxOptin, staticSmem)
	}
	return maxDynamic, nil
}

// PrepareSharedMem validates dynamic shared memory against device opt-in limits
// and raises FuncAttributeMaxDynamicSharedSizeBytes on the kernel if needed.
// It preserves larger previously established opt-in limits and accounts for static
// shared memory allocation. For an owned kernel, the device is inferred from the kernel;
// for a raw driver function, an explicit device must be passed (or inferred from the stream).
func (l *Launcher) PrepareSharedMem(targetDev ...*cuda.Device) error {
	if l == nil {
		return fmt.Errorf("cudalaunch: nil launcher")
	}
	if l.shmem == 0 {
		return nil
	}

	var dev *cuda.Device
	if len(targetDev) > 0 && targetDev[0] != nil {
		dev = targetDev[0]
	}

	if l.owned != nil {
		ownedDev := l.owned.Device()
		if ownedDev == nil {
			return fmt.Errorf("cudalaunch: kernel does not belong to a device")
		}
		if dev != nil && dev.Raw() != ownedDev.Raw() {
			return fmt.Errorf("cudalaunch: device mismatch: kernel belongs to device %d, requested %d", ownedDev.Ordinal(), dev.Ordinal())
		}
		dev = ownedDev

		staticSmem, err := l.owned.Attribute(driver.FuncAttributeSharedSizeBytes)
		if err != nil {
			return fmt.Errorf("cudalaunch: query kernel static shared memory: %w", err)
		}

		if _, err := CheckSharedMemLimit(dev, staticSmem, int(l.shmem)); err != nil {
			return err
		}

		currentDynamic, err := l.owned.Attribute(driver.FuncAttributeMaxDynamicSharedSizeBytes)
		if err != nil {
			return fmt.Errorf("cudalaunch: query kernel dynamic shared memory limit: %w", err)
		}

		if int(l.shmem) <= currentDynamic {
			return nil
		}

		if err := l.owned.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, int(l.shmem)); err != nil {
			return fmt.Errorf("cudalaunch: set max dynamic shared memory: %w", err)
		}
		return nil
	}

	// Raw function handle path:
	if l.fn == 0 {
		return fmt.Errorf("cudalaunch: nil kernel function")
	}
	if dev == nil && l.stream != nil {
		dev = l.stream.Device()
	}
	if dev == nil {
		return fmt.Errorf("cudalaunch: raw function handle requires a device or stream for shared memory preparation")
	}

	static32, err := l.fn.GetAttribute(driver.FuncAttributeSharedSizeBytes)
	if err != nil {
		return fmt.Errorf("cudalaunch: query function static shared memory: %w", err)
	}
	staticSmem := int(static32)

	if _, err := CheckSharedMemLimit(dev, staticSmem, int(l.shmem)); err != nil {
		return err
	}

	current32, err := l.fn.GetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes)
	if err != nil {
		return fmt.Errorf("cudalaunch: query function dynamic shared memory limit: %w", err)
	}
	currentDynamic := int(current32)

	if int(l.shmem) <= currentDynamic {
		return nil
	}

	err = dev.WithContext(func() error {
		return l.fn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, int32(l.shmem))
	})
	if err != nil {
		return fmt.Errorf("cudalaunch: set max dynamic shared memory: %w", err)
	}
	return nil
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

// ClusterDimAttr returns a LaunchAttribute setting the cluster dimensions
// for cuLaunchKernelEx or kernel graph nodes.
func ClusterDimAttr(dim Dim3) driver.LaunchAttribute {
	var attr driver.LaunchAttribute
	attr.Id = driver.LaunchAttributeClusterDimension
	// CUlaunchAttributeValue.clusterDim is struct { unsigned int x, y, z; }.
	// LittleEndian packing into Value[0] (x, y) and Value[1] (z).
	slice := unsafe.Slice((*byte)(unsafe.Pointer(&attr.Value[0])), 12)
	binary.LittleEndian.PutUint32(slice[0:4], dim.X)
	binary.LittleEndian.PutUint32(slice[4:8], dim.Y)
	binary.LittleEndian.PutUint32(slice[8:12], dim.Z)
	return attr
}

// WithClusterDim configures cluster dimensions for this launch.
func (l *Launcher) WithClusterDim(dim Dim3) *Launcher {
	return l.WithAttrs(ClusterDimAttr(dim))
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

// WithDeviceUpdatable enables or disables marking the kernel node as device-updatable
// when launched into a capturing stream.
func (l *Launcher) WithDeviceUpdatable(enable bool) *Launcher {
	return l.WithAttrs(DeviceUpdatableAttr(enable))
}

// DeviceUpdatableAttr returns a LaunchAttribute enabling or disabling device-updatable
// status for a captured kernel node launch.
func DeviceUpdatableAttr(enable bool) driver.LaunchAttribute {
	var val uint64
	if enable {
		val = 1
	}
	return driver.LaunchAttribute{
		Id:    driver.LaunchAttributeDeviceUpdatableKernelNode,
		Value: [8]uint64{val},
	}
}

// DeviceNode returns the device-updatable kernel node handle recorded by the driver
// for a captured launch, if this launcher had DeviceUpdatable configured and the handle is non-zero.
func (l *Launcher) DeviceNode() (uintptr, bool) {
	for _, attr := range l.attrs {
		if attr.Id == driver.LaunchAttributeDeviceUpdatableKernelNode {
			h := uintptr(attr.Value[1])
			if h != 0 {
				return h, true
			}
			return 0, false
		}
	}
	return 0, false
}

// WithSignature sets the expected kernel parameter signature for validation during Launch.
func (l *Launcher) WithSignature(sig Signature) *Launcher {
	sig.Params = append([]Param(nil), sig.Params...)
	l.sig = &sig
	return l
}

// Launch packs the provided Go arguments and dispatches the kernel.
func (l *Launcher) Launch(args ...any) error {
	if l.owned == nil && l.fn == 0 {
		return fmt.Errorf("cudalaunch: nil kernel")
	}
	if l.shmem > 0 {
		if l.owned != nil || l.stream != nil {
			if err := l.PrepareSharedMem(); err != nil {
				return err
			}
		}
	}
	if l.sig != nil {
		if err := Validate(*l.sig, args...); err != nil {
			return err
		}
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	kernelParams, err := packArgs(args)
	if err != nil {
		return fmt.Errorf("cudalaunch: pack args: %w", err)
	}

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
func PackArgs(args ...any) ([]unsafe.Pointer, error) {
	return packArgs(args)
}

// packArgs marshals Go values into an array of pointers as expected by cuLaunchKernel.
// The returned params slice must remain live through the driver call. Its
// pointers retain packed scalar and struct storage; caller-provided pointers
// retain their usual caller-owned lifetime.
func packArgs(args []any) ([]unsafe.Pointer, error) {
	if len(args) == 0 {
		return nil, nil
	}

	params := make([]unsafe.Pointer, len(args))

	for i, arg := range args {
		if arg == nil {
			return nil, fmt.Errorf("arg %d is nil", i)
		}

		// Check if argument satisfies DevicePtrProvider (e.g. cudamem.Slice).
		// An empty DeviceSlice packs as a null pointer so that optional kernel
		// arguments can be passed as a zero slice.
		//
		// The device address itself is not checked for alignment, and must not
		// be: cuLaunchKernel constrains the host side of the call, requiring
		// each kernelParams entry to point at a region holding the parameter,
		// and Go already aligns these slots. The pointee type is not knowable
		// here, so any fixed rule is both too strict (a const float* needs 4)
		// and too loose (a float4* needs 16). A 4-byte-aligned base pointer
		// launches correctly on the device; an earlier %8 check refused it.
		if provider, ok := arg.(DevicePtrProvider); ok {
			ptrVal := provider.Ptr()
			if s, ok := arg.(DeviceSlice); ok && s.Len() == 0 {
				ptrVal = 0
			}
			params[i] = unsafe.Pointer(&ptrVal)
			continue
		}

		switch v := arg.(type) {
		case driver.DevicePtr:
			params[i] = unsafe.Pointer(&v)

		case int8:
			params[i] = unsafe.Pointer(&v)
		case int16:
			params[i] = unsafe.Pointer(&v)
		case int32:
			params[i] = unsafe.Pointer(&v)
		case int64:
			params[i] = unsafe.Pointer(&v)
		case int:
			val := int64(v)
			params[i] = unsafe.Pointer(&val)
		case uint8:
			params[i] = unsafe.Pointer(&v)
		case uint16:
			params[i] = unsafe.Pointer(&v)
		case uint32:
			params[i] = unsafe.Pointer(&v)
		case uint64:
			params[i] = unsafe.Pointer(&v)
		case uint:
			val := uint64(v)
			params[i] = unsafe.Pointer(&val)
		case uintptr:
			params[i] = unsafe.Pointer(&v)
		case float32:
			params[i] = unsafe.Pointer(&v)
		case float64:
			params[i] = unsafe.Pointer(&v)
		case bool:
			var byteVal uint8
			if v {
				byteVal = 1
			}
			params[i] = unsafe.Pointer(&byteVal)
		default:
			rv := reflect.ValueOf(arg)
			if rv.Kind() == reflect.Pointer {
				if rv.IsNil() {
					return nil, fmt.Errorf("arg %d is a nil pointer", i)
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
				continue
			}
			return nil, fmt.Errorf("arg %d has unsupported type %T", i, arg)
		}
	}

	return params, nil
}

// OccupancyMaxPotentialBlockSize suggests a block size and the minimum grid
// size needed to reach the driver's occupancy estimate.
func OccupancyMaxPotentialBlockSize(fn driver.Function, dynamicSMemBytes uint, blockSizeLimit int) (minGridSize, blockSize int, err error) {
	minGrid, block, err := fn.MaxPotentialBlockSize(0, dynamicSMemBytes, int32(blockSizeLimit))
	if err != nil {
		return 0, 0, fmt.Errorf("cudalaunch: occupancy calculation: %w", err)
	}
	return int(minGrid), int(block), nil
}
