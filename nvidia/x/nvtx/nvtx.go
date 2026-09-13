package nvtx

import (
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// RangeID represents a process-unique identifier for a non-nested NVTX range.
type RangeID uint64

// StringHandle represents an immutable registered string token in an NVTX domain.
type StringHandle uintptr

const (
	nvtxVersion = 3

	// Export Table IDs
	etidCallbacks   = 1
	etidVersionInfo = 3

	// Callback Modules
	cbModuleCore   = 1
	cbModuleCUDA   = 2
	cbModuleOpenCL = 3
	cbModuleCUDART = 4
	cbModuleCore2  = 5
	cbModuleSync   = 6

	// Callback table sizes are the NVTX release-v3 NVTX_CBID_*_SIZE
	// constants. Each backing table has one additional null guard.
	cbCoreSize   = 16
	cbCUDASize   = 9
	cbOpenCLSize = 15
	cbCUDARTSize = 7
	cbCore2Size  = 16
	cbSyncSize   = 7

	// Message Types
	messageTypeAscii = 1
)

type eventAttributes struct {
	version     uint16
	size        uint16
	category    uint32
	colorType   int32
	color       uint32
	payloadType int32
	reserved0   int32
	payload     uint64
	messageType int32
	_           int32 // padding
	messagePtr  uintptr
}

// Export table structs per NVTX v3 C specification
type exportTableCallbacks struct {
	structSize             uintptr
	getModuleFunctionTable uintptr
}

type exportTableVersionInfo struct {
	structSize              uintptr
	version                 uint32
	reserved0               uint32
	setInjectionNvtxVersion uintptr
}

var (
	initOnce sync.Once
	enabled  bool

	// Global slot storage for function pointers populated by the injection tool
	fnMarkEx        uintptr
	fnMarkA         uintptr
	fnMarkW         uintptr
	fnRangeStartEx  uintptr
	fnRangeStartA   uintptr
	fnRangeStartW   uintptr
	fnRangeEnd      uintptr
	fnRangePushEx   uintptr
	fnRangePushA    uintptr
	fnRangePushW    uintptr
	fnRangePop      uintptr
	fnNameCategoryA uintptr
	fnNameCategoryW uintptr
	fnNameOsThreadA uintptr
	fnNameOsThreadW uintptr

	fnDomainMarkEx          uintptr
	fnDomainRangeStartEx    uintptr
	fnDomainRangeEnd        uintptr
	fnDomainRangePushEx     uintptr
	fnDomainRangePop        uintptr
	fnDomainResourceCreate  uintptr
	fnDomainResourceDestroy uintptr
	fnDomainNameCategoryA   uintptr
	fnDomainNameCategoryW   uintptr
	fnDomainRegisterStringA uintptr
	fnDomainRegisterStringW uintptr
	fnDomainCreateA         uintptr
	fnDomainCreateW         uintptr
	fnDomainDestroy         uintptr
	fnInitialize            uintptr

	// Native CUDA resource naming slots populated by the NVTX injection
	// provider. The slot order is NVTX_CBID_CUDA_NameCu* in nvtxTypes.h.
	fnNameCuDeviceA  uintptr
	fnNameCuDeviceW  uintptr
	fnNameCuContextA uintptr
	fnNameCuContextW uintptr
	fnNameCuStreamA  uintptr
	fnNameCuStreamW  uintptr
	fnNameCuEventA   uintptr
	fnNameCuEventW   uintptr

	// Slot pointer tables passed to the injection library
	tableCORE   [cbCoreSize + 1]uintptr
	tableCUDA   [cbCUDASize + 1]uintptr
	slotsCUDA   [cbCUDASize - 1]uintptr
	tableOpenCL [cbOpenCLSize + 1]uintptr
	slotsOpenCL [cbOpenCLSize - 1]uintptr
	tableCUDART [cbCUDARTSize + 1]uintptr
	slotsCUDART [cbCUDARTSize - 1]uintptr
	tableCORE2  [cbCore2Size + 1]uintptr
	tableSync   [cbSyncSize + 1]uintptr
	slotsSync   [cbSyncSize - 1]uintptr

	etblCallbacks   exportTableCallbacks
	etblVersionInfo exportTableVersionInfo
)

func initTables() {
	// Table CORE: indices correspond to NvtxCallbackIdCore
	tableCORE[0] = 0
	tableCORE[1] = uintptr(unsafe.Pointer(&fnMarkEx))
	tableCORE[2] = uintptr(unsafe.Pointer(&fnMarkA))
	tableCORE[3] = uintptr(unsafe.Pointer(&fnMarkW))
	tableCORE[4] = uintptr(unsafe.Pointer(&fnRangeStartEx))
	tableCORE[5] = uintptr(unsafe.Pointer(&fnRangeStartA))
	tableCORE[6] = uintptr(unsafe.Pointer(&fnRangeStartW))
	tableCORE[7] = uintptr(unsafe.Pointer(&fnRangeEnd))
	tableCORE[8] = uintptr(unsafe.Pointer(&fnRangePushEx))
	tableCORE[9] = uintptr(unsafe.Pointer(&fnRangePushA))
	tableCORE[10] = uintptr(unsafe.Pointer(&fnRangePushW))
	tableCORE[11] = uintptr(unsafe.Pointer(&fnRangePop))
	tableCORE[12] = uintptr(unsafe.Pointer(&fnNameCategoryA))
	tableCORE[13] = uintptr(unsafe.Pointer(&fnNameCategoryW))
	tableCORE[14] = uintptr(unsafe.Pointer(&fnNameOsThreadA))
	tableCORE[15] = uintptr(unsafe.Pointer(&fnNameOsThreadW))
	tableCORE[16] = 0 // null terminator

	// Table CORE2: indices correspond to NvtxCallbackIdCore2
	tableCORE2[0] = 0
	tableCORE2[1] = uintptr(unsafe.Pointer(&fnDomainMarkEx))
	tableCORE2[2] = uintptr(unsafe.Pointer(&fnDomainRangeStartEx))
	tableCORE2[3] = uintptr(unsafe.Pointer(&fnDomainRangeEnd))
	tableCORE2[4] = uintptr(unsafe.Pointer(&fnDomainRangePushEx))
	tableCORE2[5] = uintptr(unsafe.Pointer(&fnDomainRangePop))
	tableCORE2[6] = uintptr(unsafe.Pointer(&fnDomainResourceCreate))
	tableCORE2[7] = uintptr(unsafe.Pointer(&fnDomainResourceDestroy))
	tableCORE2[8] = uintptr(unsafe.Pointer(&fnDomainNameCategoryA))
	tableCORE2[9] = uintptr(unsafe.Pointer(&fnDomainNameCategoryW))
	tableCORE2[10] = uintptr(unsafe.Pointer(&fnDomainRegisterStringA))
	tableCORE2[11] = uintptr(unsafe.Pointer(&fnDomainRegisterStringW))
	tableCORE2[12] = uintptr(unsafe.Pointer(&fnDomainCreateA))
	tableCORE2[13] = uintptr(unsafe.Pointer(&fnDomainCreateW))
	tableCORE2[14] = uintptr(unsafe.Pointer(&fnDomainDestroy))
	tableCORE2[15] = uintptr(unsafe.Pointer(&fnInitialize))
	tableCORE2[16] = 0 // null terminator

	slotsCUDA = [cbCUDASize - 1]uintptr{
		uintptr(unsafe.Pointer(&fnNameCuDeviceA)),
		uintptr(unsafe.Pointer(&fnNameCuDeviceW)),
		uintptr(unsafe.Pointer(&fnNameCuContextA)),
		uintptr(unsafe.Pointer(&fnNameCuContextW)),
		uintptr(unsafe.Pointer(&fnNameCuStreamA)),
		uintptr(unsafe.Pointer(&fnNameCuStreamW)),
		uintptr(unsafe.Pointer(&fnNameCuEventA)),
		uintptr(unsafe.Pointer(&fnNameCuEventW)),
	}
	initCallbackTable(tableCUDA[:], slotsCUDA[:])
	initCallbackTable(tableOpenCL[:], slotsOpenCL[:])
	initCallbackTable(tableCUDART[:], slotsCUDART[:])
	initCallbackTable(tableSync[:], slotsSync[:])

	etblCallbacks = exportTableCallbacks{
		structSize:             unsafe.Sizeof(exportTableCallbacks{}),
		getModuleFunctionTable: purego.NewCallback(getModuleFunctionTable),
	}

	etblVersionInfo = exportTableVersionInfo{
		structSize:              unsafe.Sizeof(exportTableVersionInfo{}),
		version:                 nvtxVersion,
		reserved0:               0,
		setInjectionNvtxVersion: purego.NewCallback(setInjectionNvtxVersion),
	}
}

func initCallbackTable(table, slots []uintptr) {
	for i := range slots {
		table[i+1] = uintptr(unsafe.Pointer(&slots[i]))
	}
}

func getModuleFunctionTable(callbackModule int32, outTable *uintptr, outSize *uint32) int32 {
	switch callbackModule {
	case cbModuleCore:
		setCallbackTable(outTable, outSize, tableCORE[:])
		return 1
	case cbModuleCUDA:
		setCallbackTable(outTable, outSize, tableCUDA[:])
		return 1
	case cbModuleOpenCL:
		setCallbackTable(outTable, outSize, tableOpenCL[:])
		return 1
	case cbModuleCUDART:
		setCallbackTable(outTable, outSize, tableCUDART[:])
		return 1
	case cbModuleCore2:
		setCallbackTable(outTable, outSize, tableCORE2[:])
		return 1
	case cbModuleSync:
		setCallbackTable(outTable, outSize, tableSync[:])
		return 1
	default:
		return 0
	}
}

func setCallbackTable(outTable *uintptr, outSize *uint32, table []uintptr) {
	if outTable != nil {
		*outTable = uintptr(unsafe.Pointer(&table[0]))
	}
	if outSize != nil {
		*outSize = uint32(len(table) - 1)
	}
}

func setInjectionNvtxVersion(version uint32) {
	_ = version
}

func getExportTable(exportTableId uint32) uintptr {
	switch exportTableId {
	case etidCallbacks:
		return uintptr(unsafe.Pointer(&etblCallbacks))
	case etidVersionInfo:
		return uintptr(unsafe.Pointer(&etblVersionInfo))
	default:
		return 0
	}
}

func initInjection() {
	injectionPath := os.Getenv("NVTX_INJECTION64_PATH")
	if injectionPath == "" {
		injectionPath = os.Getenv("NVTX_INJECTION32_PATH")
	}
	if injectionPath == "" {
		return
	}

	handle, err := openDynamicLibrary(injectionPath)
	if err != nil {
		return
	}

	initFn, err := lookupLibrarySymbol(handle, "InitializeInjectionNvtx2")
	if err != nil || initFn == 0 {
		initFn, err = lookupLibrarySymbol(handle, "InitializeInjectionNvtx")
		if err != nil || initFn == 0 {
			return
		}
	}

	initTables()
	cbGetExportTable := purego.NewCallback(getExportTable)

	r, _, _ := purego.SyscallN(initFn, cbGetExportTable)
	if r != 0 {
		enabled = true
	}
}

func ensureInit() {
	initOnce.Do(initInjection)
}

// Enabled reports whether an active NVTX injection provider (e.g. Nsight Systems) is attached.
func Enabled() bool {
	ensureInit()
	return enabled
}

func cString(s string) *byte {
	if s == "" {
		return nil
	}
	b := append([]byte(s), 0)
	return &b[0]
}

// Mark records an instantaneous event marker on the global timeline.
func Mark(msg string) {
	ensureInit()
	if !enabled || fnMarkA == 0 {
		return
	}
	purego.SyscallN(fnMarkA, uintptr(unsafe.Pointer(cString(msg))))
}

// RangePush pushes a nested timeline range on the current thread.
// Returns the 0-based depth level of the range being started.
func RangePush(msg string) int {
	ensureInit()
	if !enabled || fnRangePushA == 0 {
		return 0
	}
	r, _, _ := purego.SyscallN(fnRangePushA, uintptr(unsafe.Pointer(cString(msg))))
	return int(int32(r))
}

// RangePop pops the most recent nested timeline range from the current thread.
// Returns the remaining stack depth.
func RangePop() int {
	ensureInit()
	if !enabled || fnRangePop == 0 {
		return 0
	}
	r, _, _ := purego.SyscallN(fnRangePop)
	return int(int32(r))
}

// ScopedRange starts a nested range and pins the calling goroutine to its
// current operating-system thread until the returned function is called. Call
// the returned function from the same goroutine, normally with defer:
//
//	defer nvtx.ScopedRange("MatMul")()
func ScopedRange(msg string) func() {
	ensureInit()
	if !enabled || fnRangePushA == 0 || fnRangePop == 0 {
		return func() {}
	}
	runtime.LockOSThread()
	r, _, _ := purego.SyscallN(fnRangePushA, uintptr(unsafe.Pointer(cString(msg))))
	if int32(r) < 0 {
		runtime.UnlockOSThread()
		return func() {}
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			purego.SyscallN(fnRangePop)
			runtime.UnlockOSThread()
		})
	}
}

// RangeStart starts a non-nested range spanning across goroutines or asynchronous streams.
func RangeStart(msg string) RangeID {
	ensureInit()
	if !enabled || fnRangeStartA == 0 {
		return 0
	}
	r, _, _ := purego.SyscallN(fnRangeStartA, uintptr(unsafe.Pointer(cString(msg))))
	return RangeID(r)
}

// RangeEnd ends a non-nested range previously started by RangeStart.
func RangeEnd(id RangeID) {
	ensureInit()
	if !enabled || fnRangeEnd == 0 || id == 0 {
		return
	}
	purego.SyscallN(fnRangeEnd, uintptr(id))
}

// Domain represents an isolated NVTX instrumentation namespace. Its methods
// may be called concurrently. Destroy waits for active scoped ranges and
// native calls, then makes subsequent calls no-ops.
type Domain struct {
	mu   sync.RWMutex
	raw  uintptr
	name string
}

// CreateDomain creates a named custom NVTX domain.
func CreateDomain(name string) *Domain {
	ensureInit()
	d := &Domain{name: name}
	if enabled && fnDomainCreateA != 0 {
		r, _, _ := purego.SyscallN(fnDomainCreateA, uintptr(unsafe.Pointer(cString(name))))
		d.raw = r
	}
	return d
}

// Destroy destroys the custom domain and invalidates its resources.
func (d *Domain) Destroy() {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.raw != 0 {
		if fnDomainDestroy != 0 {
			purego.SyscallN(fnDomainDestroy, d.raw)
		}
		d.raw = 0
	}
}

// Mark records an instantaneous event marker within this domain.
func (d *Domain) Mark(msg string) {
	if d == nil || !enabled || fnDomainMarkEx == 0 {
		return
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.raw == 0 {
		return
	}
	attrs := eventAttributes{
		version:     nvtxVersion,
		size:        uint16(unsafe.Sizeof(eventAttributes{})),
		messageType: messageTypeAscii,
		messagePtr:  uintptr(unsafe.Pointer(cString(msg))),
	}
	purego.SyscallN(fnDomainMarkEx, d.raw, uintptr(unsafe.Pointer(&attrs)))
}

// RangePush pushes a nested timeline range on the current thread within this domain.
func (d *Domain) RangePush(msg string) int {
	if d == nil || !enabled || fnDomainRangePushEx == 0 {
		return 0
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.raw == 0 {
		return 0
	}
	attrs := eventAttributes{
		version:     nvtxVersion,
		size:        uint16(unsafe.Sizeof(eventAttributes{})),
		messageType: messageTypeAscii,
		messagePtr:  uintptr(unsafe.Pointer(cString(msg))),
	}
	r, _, _ := purego.SyscallN(fnDomainRangePushEx, d.raw, uintptr(unsafe.Pointer(&attrs)))
	return int(int32(r))
}

// RangePop pops the most recent nested timeline range within this domain.
func (d *Domain) RangePop() int {
	if d == nil || !enabled || fnDomainRangePop == 0 {
		return 0
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.raw == 0 {
		return 0
	}
	r, _, _ := purego.SyscallN(fnDomainRangePop, d.raw)
	return int(int32(r))
}

// ScopedRange starts a nested range in d and pins the calling goroutine to its
// current operating-system thread until the returned function is called. It
// also keeps d alive for the range lifetime. Call the returned function from
// the same goroutine, normally with defer.
func (d *Domain) ScopedRange(msg string) func() {
	if d == nil || !enabled || fnDomainRangePushEx == 0 || fnDomainRangePop == 0 {
		return func() {}
	}
	d.mu.RLock()
	if d.raw == 0 {
		d.mu.RUnlock()
		return func() {}
	}
	raw := d.raw
	runtime.LockOSThread()
	r, _, _ := purego.SyscallN(fnDomainRangePushEx, raw, uintptr(unsafe.Pointer(&eventAttributes{
		version:     nvtxVersion,
		size:        uint16(unsafe.Sizeof(eventAttributes{})),
		messageType: messageTypeAscii,
		messagePtr:  uintptr(unsafe.Pointer(cString(msg))),
	})))
	if int32(r) < 0 {
		runtime.UnlockOSThread()
		d.mu.RUnlock()
		return func() {}
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			purego.SyscallN(fnDomainRangePop, raw)
			runtime.UnlockOSThread()
			d.mu.RUnlock()
		})
	}
}

// RangeStart starts a non-nested range within this domain.
func (d *Domain) RangeStart(msg string) RangeID {
	if d == nil || !enabled || fnDomainRangeStartEx == 0 {
		return 0
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.raw == 0 {
		return 0
	}
	attrs := eventAttributes{
		version:     nvtxVersion,
		size:        uint16(unsafe.Sizeof(eventAttributes{})),
		messageType: messageTypeAscii,
		messagePtr:  uintptr(unsafe.Pointer(cString(msg))),
	}
	r, _, _ := purego.SyscallN(fnDomainRangeStartEx, d.raw, uintptr(unsafe.Pointer(&attrs)))
	return RangeID(r)
}

// RangeEnd ends a non-nested range within this domain.
func (d *Domain) RangeEnd(id RangeID) {
	if d == nil || !enabled || fnDomainRangeEnd == 0 || id == 0 {
		return
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.raw == 0 {
		return
	}
	purego.SyscallN(fnDomainRangeEnd, d.raw, uintptr(id))
}

// RegisterString registers an immutable string in the domain for zero-overhead repeated referencing.
func (d *Domain) RegisterString(s string) StringHandle {
	if d == nil || !enabled || fnDomainRegisterStringA == 0 {
		return 0
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.raw == 0 {
		return 0
	}
	r, _, _ := purego.SyscallN(fnDomainRegisterStringA, d.raw, uintptr(unsafe.Pointer(cString(s))))
	return StringHandle(r)
}

// NameOSThread assigns a human-readable name to the specified operating system thread in Nsight timelines.
func NameOSThread(threadID uint32, name string) {
	ensureInit()
	if !enabled || fnNameOsThreadA == 0 {
		return
	}
	purego.SyscallN(fnNameOsThreadA, uintptr(threadID), uintptr(unsafe.Pointer(cString(name))))
}

// NameCategory assigns a name to a global integer category ID.
func NameCategory(category uint32, name string) {
	ensureInit()
	if !enabled || fnNameCategoryA == 0 {
		return
	}
	purego.SyscallN(fnNameCategoryA, uintptr(category), uintptr(unsafe.Pointer(cString(name))))
}

// NameCUDADevice assigns a name to a CUDA driver device.
//
// The call is a no-op when no NVTX injection provider is attached.
func NameCUDADevice(device int32, name string) {
	ensureInit()
	if !enabled || fnNameCuDeviceA == 0 {
		return
	}
	purego.SyscallN(fnNameCuDeviceA, uintptr(device), uintptr(unsafe.Pointer(cString(name))))
}

// NameCUDAContext assigns a name to a CUDA driver context.
//
// The context is the opaque CUcontext value represented by uintptr. The call
// is a no-op when no NVTX injection provider is attached.
func NameCUDAContext(context uintptr, name string) {
	ensureInit()
	if !enabled || fnNameCuContextA == 0 {
		return
	}
	purego.SyscallN(fnNameCuContextA, context, uintptr(unsafe.Pointer(cString(name))))
}

// NameCUDAStream assigns a name to a CUDA driver stream.
//
// The stream is the opaque CUstream value represented by uintptr. The call is
// a no-op when no NVTX injection provider is attached.
func NameCUDAStream(stream uintptr, name string) {
	ensureInit()
	if !enabled || fnNameCuStreamA == 0 {
		return
	}
	purego.SyscallN(fnNameCuStreamA, stream, uintptr(unsafe.Pointer(cString(name))))
}

// NameCUDAEvent assigns a name to a CUDA driver event.
//
// The event is the opaque CUevent value represented by uintptr. The call is
// a no-op when no NVTX injection provider is attached.
func NameCUDAEvent(event uintptr, name string) {
	ensureInit()
	if !enabled || fnNameCuEventA == 0 {
		return
	}
	purego.SyscallN(fnNameCuEventA, event, uintptr(unsafe.Pointer(cString(name))))
}

// NameCategory assigns a name to an integer category ID in d.
func (d *Domain) NameCategory(category uint32, name string) {
	if d == nil || !enabled || fnDomainNameCategoryA == 0 {
		return
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.raw == 0 {
		return
	}
	purego.SyscallN(fnDomainNameCategoryA, d.raw, uintptr(category), uintptr(unsafe.Pointer(cString(name))))
}

// Raw returns the underlying native NVTX domain handle.
func (d *Domain) Raw() uintptr {
	if d == nil {
		return 0
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.raw
}
