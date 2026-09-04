package metalmlx

import (
	"fmt"
	"os"
	"strconv"
	"sync"
	"syscall"
	"unsafe"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
)

// MathMode selects how aggressively the shader compiler may reassociate
// floating point math.
type MathMode int

const (
	MathModeSafe MathMode = iota
	MathModeRelaxed
	MathModeFast
)

// CompileOptions are the options a library is compiled with.
type CompileOptions struct {
	MathMode MathMode
}

// FunctionConstant is one entry of a specialized function's constant values.
// Value must point at a value of the Metal type given by Type.
type FunctionConstant struct {
	Value unsafe.Pointer
	Type  metal.MTLDataType
	Index uint
}

// Device is a Metal device with its library and pipeline caches, its
// residency sets, and its allocator. Port of metal::Device in
// mlx/backend/metal/device.cpp.
type Device struct {
	mu        sync.RWMutex
	closed    bool
	encoders  int
	raw       metal.MTLDevice
	residency *ResidencySets
	alloc     *Allocator

	arch            string
	archGen         int
	maxOpsPerBuffer int
	maxMBPerBuffer  int

	libMu   sync.RWMutex
	libs    map[string]metal.MTLLibrary
	kernMu  sync.RWMutex
	kernels map[objc.ID]map[string]metal.MTLComputePipelineState
}

var (
	deviceOnce sync.Once
	theDevice  *Device
	deviceErr  error
)

// GetDevice returns the process-wide device, creating it on first use.
func GetDevice() (*Device, error) {
	deviceOnce.Do(func() { theDevice, deviceErr = NewDevice() })
	return theDevice, deviceErr
}

// NewDevice opens the system default Metal device.
func NewDevice() (*Device, error) {
	raw := metal.MTLCreateSystemDefaultDevice()
	if raw.GetID() == 0 {
		return nil, fmt.Errorf("metalmlx: no Metal device available")
	}
	d := &Device{
		raw:     raw,
		libs:    make(map[string]metal.MTLLibrary),
		kernels: make(map[objc.ID]map[string]metal.MTLComputePipelineState),
	}
	residency, err := newResidencySets(raw)
	if err != nil {
		return nil, err
	}
	d.residency = residency

	d.arch = os.Getenv("MLX_METAL_GPU_ARCH")
	if d.arch == "" {
		d.arch = raw.Architecture().Name()
	}
	if len(d.arch) >= 3 {
		tens := int(d.arch[len(d.arch)-3] - '0')
		ones := int(d.arch[len(d.arch)-2] - '0')
		if tens < 0 || tens > 9 {
			tens = 0
		}
		if ones < 0 || ones > 9 {
			ones = 0
		}
		d.archGen = tens*10 + ones
	}
	// The commit thresholds follow the GPU class, which is the last letter of
	// the architecture name.
	d.maxOpsPerBuffer, d.maxMBPerBuffer = commandBufferLimits(d.arch)
	d.maxOpsPerBuffer = envInt("MLX_MAX_OPS_PER_BUFFER", d.maxOpsPerBuffer)
	d.maxMBPerBuffer = envInt("MLX_MAX_MB_PER_BUFFER", d.maxMBPerBuffer)

	d.alloc = newAllocator(d)
	return d, nil
}

func commandBufferLimits(arch string) (ops, mb int) {
	class := byte(0)
	if len(arch) != 0 {
		class = arch[len(arch)-1]
	}
	switch class {
	case 'p': // phone
		return 20, 40
	case 's': // max
		return 50, 50
	case 'd': // ultra
		return 50, 50
	default: // base, pro, and anything unknown
		return 40, 40
	}
}

// Close releases resources owned by a device made by [NewDevice]. Close is
// idempotent. It returns an error while an encoder or allocation is live and
// must not run concurrently with another Device method.
//
// The process-wide device returned by [GetDevice] is process-owned and must not
// be closed by callers.
func (d *Device) Close() error {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	if d.encoders != 0 {
		return fmt.Errorf("metalmlx: close device with %d live command encoders", d.encoders)
	}
	if d.alloc != nil && d.alloc.ActiveMemory() != 0 {
		return fmt.Errorf("metalmlx: close device with %d bytes of live allocations", d.alloc.ActiveMemory())
	}
	d.closed = true
	if d.alloc != nil {
		d.alloc.close()
	}
	if d.residency != nil {
		d.residency.close()
	}
	d.kernMu.Lock()
	for _, kernels := range d.kernels {
		for _, kernel := range kernels {
			release(kernel)
		}
	}
	clear(d.kernels)
	d.kernMu.Unlock()
	d.libMu.Lock()
	for _, lib := range d.libs {
		release(lib)
	}
	clear(d.libs)
	d.libMu.Unlock()
	release(d.raw)
	d.raw = nil
	return nil
}

// Raw returns the underlying Metal device. The object is borrowed and remains
// valid only until Close.
func (d *Device) Raw() metal.MTLDevice { return d.raw }

// Name returns the device name.
func (d *Device) Name() string { return d.raw.Name() }

// Architecture returns the GPU architecture name, such as applegpu_g16s.
func (d *Device) Architecture() string { return d.arch }

// ArchitectureGen returns the architecture generation, such as 16.
func (d *Device) ArchitectureGen() int { return d.archGen }

// MaxOpsMBPerBuffer returns the dispatch count and megabyte thresholds above
// which the command encoder commits.
func (d *Device) MaxOpsMBPerBuffer() (ops, mb int) {
	return d.maxOpsPerBuffer, d.maxMBPerBuffer
}

// Allocator returns the device's caching allocator. The allocator is owned by
// the Device and remains valid only until Close.
func (d *Device) Allocator() *Allocator { return d.alloc }

// ResidencySets returns the device's residency sets. The sets are owned by the
// Device and remain valid only until Close.
func (d *Device) ResidencySets() *ResidencySets { return d.residency }

// LibraryFromPath loads a .metallib and caches it under name. A later call
// with the same name returns the cached library and ignores path. The returned
// library is borrowed from the Device and remains valid only until Close.
func (d *Device) LibraryFromPath(name, path string) (metal.MTLLibrary, error) {
	if lib, ok := d.cachedLibrary(name); ok {
		return lib, nil
	}
	d.libMu.Lock()
	defer d.libMu.Unlock()
	if lib, ok := d.libs[name]; ok {
		return lib, nil
	}
	lib, err := d.raw.NewLibraryWithURLError(foundation.NewURLFileURLWithPath(path))
	if err != nil {
		return nil, fmt.Errorf("metalmlx: load metallib %s: %w", path, err)
	}
	d.libs[name] = lib
	return lib, nil
}

// LibraryFromSource compiles Metal source and caches the library under name.
// The builder is only called on a miss, so callers can assemble the source
// lazily as MLX does. The returned library is borrowed from the Device and
// remains valid only until Close.
func (d *Device) LibraryFromSource(name string, opts CompileOptions, builder func() (string, error)) (metal.MTLLibrary, error) {
	if lib, ok := d.cachedLibrary(name); ok {
		return lib, nil
	}
	d.libMu.Lock()
	defer d.libMu.Unlock()
	if lib, ok := d.libs[name]; ok {
		return lib, nil
	}
	source, err := builder()
	if err != nil {
		return nil, err
	}
	lib, err := d.buildLibrary(source, opts)
	if err != nil {
		return nil, err
	}
	d.libs[name] = lib
	return lib, nil
}

// ClearLibrary drops a cached library.
func (d *Device) ClearLibrary(name string) {
	d.libMu.Lock()
	defer d.libMu.Unlock()
	delete(d.libs, name)
}

func (d *Device) cachedLibrary(name string) (metal.MTLLibrary, bool) {
	d.libMu.RLock()
	defer d.libMu.RUnlock()
	lib, ok := d.libs[name]
	return lib, ok
}

func (d *Device) buildLibrary(source string, opts CompileOptions) (metal.MTLLibrary, error) {
	options := metal.NewMTLCompileOptions()
	options.SetMathMode(metal.MTLMathMode(opts.MathMode))
	options.SetLanguageVersion(languageVersion())
	lib, err := d.raw.NewLibraryWithSourceOptionsError(source, options)
	if err != nil {
		return nil, fmt.Errorf("metalmlx: build metal library from source: %w", err)
	}
	return lib, nil
}

// Kernel returns the compute pipeline state for a function in lib, compiling
// and caching it on first use. The returned pipeline is borrowed from the
// Device and remains valid only until Close.
func (d *Device) Kernel(lib metal.MTLLibrary, name string) (metal.MTLComputePipelineState, error) {
	return d.KernelSpecialized(lib, name, name, nil)
}

// KernelSpecialized returns the pipeline state for baseName specialized with
// function constants. The pipeline is cached under hashName, which must
// identify the constant values; with no constants pass hashName == baseName.
func (d *Device) KernelSpecialized(lib metal.MTLLibrary, baseName, hashName string, consts []FunctionConstant) (metal.MTLComputePipelineState, error) {
	if !valid(lib) {
		return nil, fmt.Errorf("metalmlx: nil library for kernel %q", baseName)
	}
	if hashName == "" {
		hashName = baseName
	}
	libID := lib.GetID()
	d.kernMu.RLock()
	kernel, ok := d.kernels[libID][hashName]
	d.kernMu.RUnlock()
	if ok {
		return kernel, nil
	}

	d.kernMu.Lock()
	defer d.kernMu.Unlock()
	if kernel, ok := d.kernels[libID][hashName]; ok {
		return kernel, nil
	}
	fn, err := d.function(lib, baseName, hashName, consts)
	if err != nil {
		return nil, err
	}
	kernel, err = d.raw.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, fmt.Errorf("metalmlx: build pipeline for %s: %w", baseName, err)
	}
	if d.kernels[libID] == nil {
		d.kernels[libID] = make(map[string]metal.MTLComputePipelineState)
	}
	d.kernels[libID][hashName] = kernel
	return kernel, nil
}

func (d *Device) function(lib metal.MTLLibrary, name, specialized string, consts []FunctionConstant) (metal.MTLFunction, error) {
	if len(consts) == 0 && specialized == name {
		fn := lib.NewFunctionWithName(name)
		if !valid(fn) {
			return nil, fmt.Errorf("metalmlx: function %q not found in library", name)
		}
		return fn, nil
	}
	values := metal.NewMTLFunctionConstantValues()
	for _, c := range consts {
		values.SetConstantValueTypeAtIndex(c.Value, c.Type, c.Index)
	}
	desc := metal.NewMTLFunctionDescriptor()
	desc.SetName(name)
	desc.SetSpecializedName(specialized)
	desc.SetConstantValues(values)
	fn, err := lib.NewFunctionWithDescriptorError(desc)
	if err != nil {
		return nil, fmt.Errorf("metalmlx: load function %s: %w", name, err)
	}
	return fn, nil
}

// languageVersion returns the newest shading language the bindings name.
func languageVersion() metal.MTLLanguageVersion {
	return metal.MTLLanguageVersion4_0
}

func envInt(name string, def int) int {
	v, err := strconv.Atoi(os.Getenv(name))
	if err != nil {
		return def
	}
	return v
}

func vmPageSize() int { return os.Getpagesize() }

// sysctlUint64 reads a 64-bit sysctl. syscall.Sysctl drops one trailing NUL,
// which on a little-endian host is a high-order zero byte, so the value is
// padded back out here.
func sysctlUint64(name string) (uint64, error) {
	s, err := syscall.Sysctl(name)
	if err != nil {
		return 0, err
	}
	var v uint64
	for i := 0; i < len(s) && i < 8; i++ {
		v |= uint64(s[i]) << (8 * i)
	}
	return v, nil
}

// hostMemorySize returns the size of physical memory, which on Apple silicon
// is also the GPU memory.
func hostMemorySize() uint64 {
	n, err := sysctlUint64("hw.memsize")
	if err != nil {
		return 0
	}
	return n
}

// resourceLimit returns the maximum number of GPU resources, falling back to
// the value MLX uses when the sysctl is missing.
func resourceLimit() uint64 {
	n, err := sysctlUint64("iogpu.rsrc_limit")
	if err != nil || n == 0 {
		return 499000
	}
	return n
}

// valid reports whether an object reference is a real object. A protocol
// value that wraps a nil pointer, such as the heap of a device buffer, is a
// non-nil Go interface holding a zero id, so a nil check is not enough.
func valid(o objectivec.IObject) bool {
	return o != nil && o.GetID() != 0
}

// release drops one reference to an Objective-C object. The protocol
// interfaces only carry their own methods, so this goes through the object.
func release(o objectivec.IObject) {
	objectivec.ObjectFromID(o.GetID()).Release()
}

// retain adds a reference, so an autoreleased object outlives its pool.
func retain(o objectivec.IObject) {
	objectivec.ObjectFromID(o.GetID()).Retain()
}
