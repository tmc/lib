package nvtx

import (
	"os"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// RangeID represents a process-unique identifier for a non-nested NVTX range.
type RangeID uint64

// StringHandle represents an immutable registered string token in an NVTX domain.
type StringHandle uintptr

var (
	initOnce sync.Once
	enabled  bool

	fnMarkA               func(*byte)
	fnRangePushA          func(*byte) int32
	fnRangePop            func() int32
	fnRangeStartA         func(*byte) uint64
	fnRangeEnd            func(uint64)
	fnDomainCreateA       func(*byte) uintptr
	fnDomainDestroy       func(uintptr)
	fnDomainRangePushA    func(uintptr, *byte) int32
	fnDomainRangePop      func(uintptr) int32
	fnDomainRangeStartA   func(uintptr, *byte) uint64
	fnDomainRangeEnd      func(uintptr, uint64)
	fnDomainMarkA         func(uintptr, *byte)
	fnDomainRegisterStringA func(uintptr, *byte) uintptr
	fnNameOsThreadA       func(uint32, *byte)
	fnNameCategoryA       func(uintptr, uint32, *byte)
)

func initInjection() {
	injectionPath := os.Getenv("NVTX_INJECTION64_PATH")
	if injectionPath == "" {
		injectionPath = os.Getenv("NVTX_INJECTION32_PATH")
	}
	if injectionPath == "" {
		return
	}

	handle, err := purego.Dlopen(injectionPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}

	bind := func(name string, fptr any) {
		sym, err := purego.Dlsym(handle, name)
		if err == nil && sym != 0 {
			purego.RegisterFunc(fptr, sym)
		}
	}

	bind("nvtxMarkA", &fnMarkA)
	bind("nvtxRangePushA", &fnRangePushA)
	bind("nvtxRangePop", &fnRangePop)
	bind("nvtxRangeStartA", &fnRangeStartA)
	bind("nvtxRangeEnd", &fnRangeEnd)
	bind("nvtxDomainCreateA", &fnDomainCreateA)
	bind("nvtxDomainDestroy", &fnDomainDestroy)
	bind("nvtxDomainRangePushA", &fnDomainRangePushA)
	bind("nvtxDomainRangePop", &fnDomainRangePop)
	bind("nvtxDomainRangeStartA", &fnDomainRangeStartA)
	bind("nvtxDomainRangeEnd", &fnDomainRangeEnd)
	bind("nvtxDomainMarkA", &fnDomainMarkA)
	bind("nvtxDomainRegisterStringA", &fnDomainRegisterStringA)
	bind("nvtxNameOsThreadA", &fnNameOsThreadA)
	bind("nvtxNameCategoryA", &fnNameCategoryA)

	enabled = true
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
	if !enabled || fnMarkA == nil {
		return
	}
	fnMarkA(cString(msg))
}

// RangePush pushes a nested timeline range on the current thread.
// Returns the 0-based depth level of the range being started.
func RangePush(msg string) int {
	ensureInit()
	if !enabled || fnRangePushA == nil {
		return 0
	}
	return int(fnRangePushA(cString(msg)))
}

// RangePop pops the most recent nested timeline range from the current thread.
// Returns the remaining stack depth.
func RangePop() int {
	ensureInit()
	if !enabled || fnRangePop == nil {
		return 0
	}
	return int(fnRangePop())
}

// ScopedRange is a convenience helper for defer blocks:
//
//	defer nvtx.ScopedRange("MatMul")()
func ScopedRange(msg string) func() {
	RangePush(msg)
	return func() {
		RangePop()
	}
}

// RangeStart starts a non-nested range spanning across goroutines or asynchronous streams.
func RangeStart(msg string) RangeID {
	ensureInit()
	if !enabled || fnRangeStartA == nil {
		return 0
	}
	return RangeID(fnRangeStartA(cString(msg)))
}

// RangeEnd ends a non-nested range previously started by RangeStart.
func RangeEnd(id RangeID) {
	ensureInit()
	if !enabled || fnRangeEnd == nil || id == 0 {
		return
	}
	fnRangeEnd(uint64(id))
}

// Domain represents an isolated NVTX instrumentation namespace.
type Domain struct {
	raw  uintptr
	name string
}

// CreateDomain creates a named custom NVTX domain.
func CreateDomain(name string) *Domain {
	ensureInit()
	d := &Domain{name: name}
	if enabled && fnDomainCreateA != nil {
		d.raw = fnDomainCreateA(cString(name))
	}
	return d
}

// Destroy destroys the custom domain and invalidates its resources.
func (d *Domain) Destroy() {
	if d.raw != 0 && fnDomainDestroy != nil {
		fnDomainDestroy(d.raw)
		d.raw = 0
	}
}

// Mark records an instantaneous event marker within this domain.
func (d *Domain) Mark(msg string) {
	if !enabled || d.raw == 0 || fnDomainMarkA == nil {
		return
	}
	fnDomainMarkA(d.raw, cString(msg))
}

// RangePush pushes a nested timeline range on the current thread within this domain.
func (d *Domain) RangePush(msg string) int {
	if !enabled || d.raw == 0 || fnDomainRangePushA == nil {
		return 0
	}
	return int(fnDomainRangePushA(d.raw, cString(msg)))
}

// RangePop pops the most recent nested timeline range within this domain.
func (d *Domain) RangePop() int {
	if !enabled || d.raw == 0 || fnDomainRangePop == nil {
		return 0
	}
	return int(fnDomainRangePop(d.raw))
}

// ScopedRange is a convenience helper for defer blocks within this domain.
func (d *Domain) ScopedRange(msg string) func() {
	d.RangePush(msg)
	return func() {
		d.RangePop()
	}
}

// RangeStart starts a non-nested range within this domain.
func (d *Domain) RangeStart(msg string) RangeID {
	if !enabled || d.raw == 0 || fnDomainRangeStartA == nil {
		return 0
	}
	return RangeID(fnDomainRangeStartA(d.raw, cString(msg)))
}

// RangeEnd ends a non-nested range within this domain.
func (d *Domain) RangeEnd(id RangeID) {
	if !enabled || d.raw == 0 || fnDomainRangeEnd == nil || id == 0 {
		return
	}
	fnDomainRangeEnd(d.raw, uint64(id))
}

// RegisterString registers an immutable string in the domain for zero-overhead repeated referencing.
func (d *Domain) RegisterString(s string) StringHandle {
	if !enabled || d.raw == 0 || fnDomainRegisterStringA == nil {
		return 0
	}
	return StringHandle(fnDomainRegisterStringA(d.raw, cString(s)))
}

// NameOSThread assigns a human-readable name to the specified operating system thread in Nsight timelines.
func NameOSThread(threadID uint32, name string) {
	ensureInit()
	if !enabled || fnNameOsThreadA == nil {
		return
	}
	fnNameOsThreadA(threadID, cString(name))
}

// NameCategory assigns a name to an integer category ID within a domain.
func NameCategory(d *Domain, category uint32, name string) {
	ensureInit()
	if !enabled || fnNameCategoryA == nil {
		return
	}
	var domainRaw uintptr
	if d != nil {
		domainRaw = d.raw
	}
	fnNameCategoryA(domainRaw, category, cString(name))
}

// Raw returns the underlying native NVTX domain handle.
func (d *Domain) Raw() uintptr {
	return d.raw
}

// Silence unused package warnings
var _ = unsafe.Pointer(nil)
