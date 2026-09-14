//go:build cupti_physical

// Package cupti_test drives CUPTI against a real driver. CUPTI is almost
// entirely callbacks and out-parameters over version-tagged structs, so the
// tests here are built around facts the host already knows: the name of the
// driver function it is about to call, and the exact number of bytes it is
// about to copy. Reading those back out of a CUPTI record proves the struct
// offsets and the callback signature, which a status code cannot.
//
// Set CUPTI_PHYSICAL_PERTURB to run a negative control.
package cupti_test

import (
	"os"
	"runtime"
	"sync"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
)

func perturb(name string) bool { return os.Getenv("CUPTI_PHYSICAL_PERTURB") == name }

// goString reads a NUL-terminated C string from an address CUPTI handed back.
// goString reads a NUL-terminated C string from pointer p. The uintptr is a C
// pointer field, never the address of Go memory. Interpret its pointer
// representation once and index with [unsafe.Add]: holding a pointer value in a
// uintptr across statements is what makes the conversion unsound. An
// unterminated string is truncated rather than walked off the end of its
// mapping.
func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	base := *(*unsafe.Pointer)(unsafe.Pointer(&p))
	var b []byte
	for i := 0; i < 1<<20; i++ {
		c := *(*byte)(unsafe.Add(base, i))
		if c == 0 {
			break
		}
		b = append(b, c)
	}
	return string(b)
}

// withContext retains the primary context on device 0 for the duration of t.
func withContext(t *testing.T) {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA driver unavailable: %v", err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("retain primary context: %v", err)
	}
	t.Cleanup(func() {
		if err := dev.PrimaryCtxRelease(); err != nil {
			t.Errorf("release context: %v", err)
		}
	})
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("set current context: %v", err)
	}
}

// TestVersion checks the one entry point that needs no state.
func TestVersion(t *testing.T) {
	v, err := cupti.GetVersion()
	if err != nil {
		t.Skipf("CUPTI unavailable: %v", err)
	}
	t.Logf("CUPTI version %d", v)
	if perturb("version") {
		v = 0
	}
	// CUPTI reports its version two ways. Through CUDA 12 it was a small
	// sequential API version; CUDA 13 switched to the toolkit encoding
	// major*10000 + minor*100 + patch, so 130001 is CUPTI from CUDA 13.0.
	// A binding that truncated or sign-extended the out parameter would
	// land outside both ranges rather than merely at an unexpected value.
	switch {
	case v == 0:
		t.Fatalf("CUPTI version is zero")
	case v < 1000:
		t.Logf("legacy CUPTI API version %d", v)
	case v/10000 >= 10 && v/10000 < 100:
		t.Logf("CUPTI from CUDA %d.%d", v/10000, v/100%100)
	default:
		t.Fatalf("implausible CUPTI version %d", v)
	}
	// GetTimestamp returns a monotonic nanosecond clock; two reads must be
	// ordered and close together. A truncated 64-bit return would show up as a
	// wildly varying or zero value.
	first, err := cupti.GetTimestamp()
	if err != nil {
		t.Fatalf("GetTimestamp: %v", err)
	}
	second, err := cupti.GetTimestamp()
	if err != nil {
		t.Fatalf("GetTimestamp: %v", err)
	}
	if first == 0 || second < first {
		t.Fatalf("GetTimestamp returned %d then %d", first, second)
	}
	if second-first > 1e9 {
		t.Fatalf("two back-to-back GetTimestamp calls differ by %d ns", second-first)
	}
	t.Logf("GetTimestamp %d then %d, %d ns apart", first, second, second-first)
}

// callbackObservation records what the subscriber callback saw.
type callbackObservation struct {
	mu     sync.Mutex
	names  map[string]int
	domain cupti.CallbackDomain
	sites  map[cupti.ApiCallbackSite]int
	total  int
}

// TestCallbackSubscription subscribes to the driver API domain, makes one
// driver call whose name the test already knows, and checks that the callback
// reported that name. FunctionName sits at offset 8 of CUpti_CallbackData,
// after an enum, so recovering the right string is evidence that the struct
// and the callback's three-argument signature both match the ABI.
func TestCallbackSubscription(t *testing.T) {
	withContext(t)
	if _, err := cupti.GetVersion(); err != nil {
		t.Skipf("CUPTI unavailable: %v", err)
	}

	obs := &callbackObservation{
		names: make(map[string]int),
		sites: make(map[cupti.ApiCallbackSite]int),
	}
	// The callback runs on the calling thread inside the driver call. It must
	// not call into the testing package, so it only records.
	cb := purego.NewCallback(func(userdata uintptr, domain cupti.CallbackDomain, cbid uint32, cbdata uintptr) uintptr {
		if cbdata == 0 {
			return 0
		}
		// cbdata addresses CUPTI's own callback record, not Go memory, so
		// the conversion go vet flags here cannot observe a moved object.
		data := (*cupti.CallbackData)(unsafe.Pointer(cbdata))
		obs.mu.Lock()
		obs.total++
		obs.domain = domain
		obs.sites[data.CallbackSite]++
		obs.names[goString(data.FunctionName)]++
		obs.mu.Unlock()
		return 0
	})

	sub, err := cupti.Subscribe(cb, 0)
	if err != nil {
		t.Skipf("cuptiSubscribe: %v (CUPTI may be restricted on this host)", err)
	}
	defer func() {
		if err := sub.Unsubscribe(); err != nil {
			t.Errorf("unsubscribe: %v", err)
		}
	}()
	if err := sub.EnableDomain(1, cupti.CbDomainDriverApi); err != nil {
		t.Skipf("EnableDomain: %v", err)
	}

	// cuMemAlloc is the call under observation. The driver binding resolves it
	// as the versioned symbol cuMemAlloc_v2, which is the name CUPTI reports.
	ptr, err := driver.MemAlloc(4096)
	if err != nil {
		t.Fatalf("MemAlloc: %v", err)
	}
	if err := ptr.MemFree(); err != nil {
		t.Fatalf("MemFree: %v", err)
	}
	if err := sub.EnableDomain(0, cupti.CbDomainDriverApi); err != nil {
		t.Fatalf("disable domain: %v", err)
	}

	obs.mu.Lock()
	defer obs.mu.Unlock()
	if obs.total == 0 {
		t.Fatal("the driver API callback never fired")
	}
	t.Logf("%d driver API callbacks, domain %v, sites %v", obs.total, obs.domain, obs.sites)
	if obs.domain != cupti.CbDomainDriverApi {
		t.Fatalf("callback domain = %v, want %v", obs.domain, cupti.CbDomainDriverApi)
	}
	// Every API callback fires once on entry and once on exit.
	if obs.sites[cupti.ApiEnter] == 0 || obs.sites[cupti.ApiExit] == 0 {
		t.Fatalf("expected both enter and exit callbacks, got %v", obs.sites)
	}
	want := "cuMemAlloc_v2"
	if perturb("callback-name") {
		want = "cuMemAllocPitch_v2"
	}
	if obs.names[want] == 0 {
		got := make([]string, 0, len(obs.names))
		for n := range obs.names {
			got = append(got, n)
		}
		t.Fatalf("no callback reported FunctionName %q; saw %q", want, got)
	}
	t.Logf("callback reported FunctionName %q %d times, as expected for the call the test made",
		want, obs.names[want])
	if n := obs.names["cuMemFree_v2"]; n == 0 {
		t.Fatalf("no callback reported cuMemFree_v2; saw %v", obs.names)
	}
}

// activityCollector owns the buffers CUPTI fills with activity records.
type activityCollector struct {
	mu      sync.Mutex
	memcpy  []cupti.ActivityMemcpy
	kinds   map[cupti.ActivityKind]int
	buffers int
	pinned  uintptr // host buffer, allocated outside the Go heap
}

const activityBufferSize = 64 << 10

// TestActivityMemcpyRecord runs a memcpy of a size only the test knows and
// then reads the byte count back out of the activity record CUPTI produced.
// Bytes sits at offset 8 of CUpti_ActivityMemcpy, after an enum and four
// single-byte fields; if the Go struct disagreed with the C one by even a
// field, the recovered count would not be the size that was copied.
func TestActivityMemcpyRecord(t *testing.T) {
	withContext(t)
	if _, err := cupti.GetVersion(); err != nil {
		t.Skipf("CUPTI unavailable: %v", err)
	}

	// CUPTI writes into this buffer from its own threads, so it must not be
	// Go heap memory. cuMemAllocHost gives page-aligned host memory.
	var host uintptr
	if err := driver.MemAllocHost(&host, activityBufferSize); err != nil {
		t.Skipf("MemAllocHost: %v", err)
	}
	defer func() {
		if err := driver.MemFreeHost(host); err != nil {
			t.Errorf("free host buffer: %v", err)
		}
	}()

	col := &activityCollector{kinds: make(map[cupti.ActivityKind]int), pinned: host}
	var inUse sync.Mutex

	requested := purego.NewCallback(func(buffer **uint8, size *uint, maxRecords *uint) uintptr {
		inUse.Lock()
		// host is a driver allocation held for the life of the test, so the
		// conversion go vet flags here addresses pinned native memory.
		*buffer = (*uint8)(unsafe.Pointer(host))
		*size = activityBufferSize
		*maxRecords = 0
		return 0
	})
	completed := purego.NewCallback(func(ctx uintptr, streamID uint32, buffer *uint8, size uint, validSize uint) uintptr {
		defer inUse.Unlock()
		col.mu.Lock()
		col.buffers++
		col.mu.Unlock()
		if validSize == 0 {
			return 0
		}
		var record *cupti.Activity
		for {
			// ActivityGetNextRecord advances record through the buffer and
			// returns an error once the buffer is exhausted.
			if err := cupti.ActivityGetNextRecord(buffer, validSize, &record); err != nil {
				return 0
			}
			if record == nil {
				return 0
			}
			col.mu.Lock()
			col.kinds[record.Kind]++
			if record.Kind == cupti.ActivityKindMemcpy {
				col.memcpy = append(col.memcpy, *(*cupti.ActivityMemcpy)(unsafe.Pointer(record)))
			}
			col.mu.Unlock()
		}
	})

	if err := cupti.ActivityRegisterCallbacks(requested, completed); err != nil {
		t.Skipf("ActivityRegisterCallbacks: %v", err)
	}
	if err := cupti.ActivityEnable(cupti.ActivityKindMemcpy); err != nil {
		t.Skipf("ActivityEnable(Memcpy): %v (CUPTI activity tracing may be restricted)", err)
	}
	defer cupti.ActivityDisable(cupti.ActivityKindMemcpy)

	// A size that is not a round number, so it cannot be confused with a
	// default, a page size, or a field read from the wrong offset.
	const copyBytes = 3_145_733
	src := make([]byte, copyBytes)
	for i := range src {
		src[i] = byte(i)
	}
	dev, err := driver.MemAlloc(copyBytes)
	if err != nil {
		t.Fatalf("MemAlloc: %v", err)
	}
	defer func() {
		if err := dev.MemFree(); err != nil {
			t.Errorf("free device buffer: %v", err)
		}
	}()
	if err := dev.MemcpyHtoD(unsafe.Pointer(&src[0]), copyBytes); err != nil {
		t.Fatalf("MemcpyHtoD: %v", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	if err := cupti.ActivityFlushAll(1); err != nil {
		t.Fatalf("ActivityFlushAll: %v", err)
	}

	col.mu.Lock()
	defer col.mu.Unlock()
	t.Logf("%d activity buffers delivered, record kinds %v", col.buffers, col.kinds)
	if len(col.memcpy) == 0 {
		t.Fatalf("no memcpy activity records were produced (buffers: %d, kinds: %v)", col.buffers, col.kinds)
	}
	want := uint64(copyBytes)
	if perturb("memcpy-bytes") {
		want++
	}
	var found *cupti.ActivityMemcpy
	for i := range col.memcpy {
		if col.memcpy[i].Bytes == want {
			found = &col.memcpy[i]
			break
		}
	}
	if found == nil {
		sizes := make([]uint64, len(col.memcpy))
		for i := range col.memcpy {
			sizes[i] = col.memcpy[i].Bytes
		}
		t.Fatalf("no memcpy record reports Bytes = %d; saw %v", want, sizes)
	}
	t.Logf("memcpy record: kind=%v copyKind=%d bytes=%d device=%d start=%d end=%d correlation=%d",
		found.Kind, found.CopyKind, found.Bytes, found.DeviceId, found.Start, found.End, found.CorrelationId)

	// Timestamps and ids around the byte count must also be sane; garbage here
	// would mean the fields after Bytes are misaligned even though Bytes is not.
	if found.Kind != cupti.ActivityKindMemcpy {
		t.Fatalf("record Kind = %v, want %v", found.Kind, cupti.ActivityKindMemcpy)
	}
	if found.End < found.Start {
		t.Fatalf("record ends (%d) before it starts (%d)", found.End, found.Start)
	}
	if found.End-found.Start > 10e9 {
		t.Fatalf("memcpy took %d ns, which is not plausible", found.End-found.Start)
	}
	if found.DeviceId != 0 {
		t.Fatalf("record DeviceId = %d, want 0 (only device 0 was used)", found.DeviceId)
	}
	if found.CorrelationId == 0 {
		t.Fatalf("record CorrelationId = 0, want the id of the driver call")
	}
	t.Logf("copied %d bytes and CUPTI reported exactly that, with a %d ns duration on device %d",
		copyBytes, found.End-found.Start, found.DeviceId)
}
