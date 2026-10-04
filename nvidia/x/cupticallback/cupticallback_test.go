package cupticallback

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
)

func TestVersionQualification(t *testing.T) {
	tests := []struct {
		name    string
		version uint32
		want    bool
	}{
		{name: "CUDA 12.6 API version", version: 24, want: true},
		{name: "CUDA 12.8 API version", version: 26, want: true},
		{name: "CUDA 12.9.41 L4 API version", version: 27, want: true},
		{name: "CUDA 12.9 GB10 API version", version: 28, want: true},
		{name: "CUDA 13.0 API version", version: 130001, want: true},
		{name: "CUDA 13.1 API version", version: 130101, want: true},
		{name: "CUDA 13.2 header version", version: 130200, want: true},
		{name: "CUDA 13.2 runtime version", version: 130202, want: true},
		{name: "CUDA 13.3 API version", version: 130301, want: true},
		{name: "CUDA 13.4 API version", version: 130400, want: true},
		{name: "unmeasured 25 uses an earlier layout", version: 25, want: true},
		{name: "unmeasured 29 uses an earlier layout", version: 29, want: true},
		{name: "zero version", version: 0, want: false},
		{name: "CUDA 12.2 API version", version: 20, want: false},
		{name: "CUDA 13.4 L4 runtime version", version: 130401, want: true},
		{name: "unmeasured 130100 uses an earlier layout", version: 130100, want: true},
		{name: "unmeasured 130201 uses an earlier layout", version: 130201, want: true},
		{name: "unmeasured 130500 uses an earlier layout", version: 130500, want: true},
		{name: "unmeasured CUDA 14", version: 140000, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := supportedVersion(tt.version); got != tt.want {
				t.Errorf("supportedVersion(%d) = %v, want %v", tt.version, got, tt.want)
			}
		})
	}
}

func TestDecodeEvent(t *testing.T) {
	fnNameBytes := []byte("cuMemAlloc_v2\x00")
	symNameBytes := []byte("my_kernel\x00")
	corrData := uint64(12345678)
	params := uint64(0x1111)
	returnValue := uint64(0x2222)

	rawApi := cupti.CallbackData{
		CallbackSite:        cupti.ApiEnter,
		FunctionName:        uintptr(unsafe.Pointer(&fnNameBytes[0])),
		SymbolName:          uintptr(unsafe.Pointer(&symNameBytes[0])),
		FunctionParams:      uintptr(unsafe.Pointer(&params)),
		FunctionReturnValue: uintptr(unsafe.Pointer(&returnValue)),
		Context:             driver.Context(0x1000),
		ContextUid:          42,
		CorrelationData:     uintptr(unsafe.Pointer(&corrData)),
		CorrelationId:       999,
	}

	evt := decodeEvent(1, cupti.CbDomainDriverApi, 10, unsafe.Pointer(&rawApi), false)
	runtime.KeepAlive(fnNameBytes)
	runtime.KeepAlive(symNameBytes)
	runtime.KeepAlive(params)
	runtime.KeepAlive(returnValue)
	runtime.KeepAlive(corrData)
	if evt.Domain != cupti.CbDomainDriverApi {
		t.Errorf("Domain = %v, want %v", evt.Domain, cupti.CbDomainDriverApi)
	}
	if evt.ID != 10 {
		t.Errorf("ID = %d, want 10", evt.ID)
	}
	if evt.Name != "cuMemAlloc_v2" {
		t.Errorf("Name = %q, want cuMemAlloc_v2", evt.Name)
	}
	if evt.Api == nil {
		t.Fatal("evt.Api is nil")
	}
	if evt.Api.FunctionName != "cuMemAlloc_v2" {
		t.Errorf("FunctionName = %q, want cuMemAlloc_v2", evt.Api.FunctionName)
	}
	if evt.Api.SymbolName != "my_kernel" {
		t.Errorf("SymbolName = %q, want my_kernel", evt.Api.SymbolName)
	}
	if evt.Api.FunctionParams != unsafe.Pointer(&params) {
		t.Errorf("FunctionParams = %p, want %p", evt.Api.FunctionParams, &params)
	}
	if evt.Api.FunctionReturnValue != unsafe.Pointer(&returnValue) {
		t.Errorf("FunctionReturnValue = %p, want %p", evt.Api.FunctionReturnValue, &returnValue)
	}
	if evt.Api.ContextUid != 42 {
		t.Errorf("ContextUid = %d, want 42", evt.Api.ContextUid)
	}
	if evt.Api.CorrelationId != 999 {
		t.Errorf("CorrelationId = %d, want 999", evt.Api.CorrelationId)
	}
	if evt.Api.CorrelationData != 12345678 {
		t.Errorf("CorrelationData = %d, want 12345678", evt.Api.CorrelationData)
	}
	if evt.Reentrant {
		t.Error("Reentrant = true, want false")
	}

	// Test ResourceData decoding
	rawRes := cupti.ResourceData{
		Context:            driver.Context(0x2000),
		ResourceHandle:     [1]uint64{0x8888},
		ResourceDescriptor: 0x9999,
	}
	evtRes := decodeEvent(0, cupti.CbDomainResource, 20, unsafe.Pointer(&rawRes), true)
	if evtRes.Res == nil {
		t.Fatal("evtRes.Res is nil")
	}
	if evtRes.Res.ResourceHandle != 0x8888 {
		t.Errorf("ResourceHandle = 0x%x, want 0x8888", evtRes.Res.ResourceHandle)
	}
	if !evtRes.Reentrant {
		t.Error("Reentrant = false, want true")
	}

	// Test SynchronizeData decoding
	rawSync := cupti.SynchronizeData{
		Context: driver.Context(0x3000),
		Stream:  driver.Stream(0x4000),
	}
	evtSync := decodeEvent(0, cupti.CbDomainSynchronize, 30, unsafe.Pointer(&rawSync), false)
	if evtSync.Sync == nil {
		t.Fatal("evtSync.Sync is nil")
	}
	if evtSync.Sync.Stream != 0x4000 {
		t.Errorf("Stream = 0x%x, want 0x4000", evtSync.Sync.Stream)
	}

	// Test NvtxData decoding
	nvtxBytes := []byte("nvtxRangePushA\x00")
	rawNvtx := cupti.NvtxData{
		FunctionName: uintptr(unsafe.Pointer(&nvtxBytes[0])),
	}
	evtNvtx := decodeEvent(0, cupti.CbDomainNvtx, 40, unsafe.Pointer(&rawNvtx), false)
	runtime.KeepAlive(nvtxBytes)
	if evtNvtx.Nvtx == nil {
		t.Fatal("evtNvtx.Nvtx is nil")
	}
	if evtNvtx.Nvtx.FunctionName != "nvtxRangePushA" {
		t.Errorf("Nvtx FunctionName = %q, want nvtxRangePushA", evtNvtx.Nvtx.FunctionName)
	}

	// Test StateData decoding
	rawState := cupti.StateData{
		Notification: [2]uint64{11, 22},
	}
	evtState := decodeEvent(0, cupti.CbDomainState, 50, unsafe.Pointer(&rawState), false)
	if evtState.State == nil {
		t.Fatal("evtState.State is nil")
	}
	if evtState.State.Notification != [2]uint64{11, 22} {
		t.Errorf("Notification = %v, want [11, 22]", evtState.State.Notification)
	}
}

func TestDispatcher(t *testing.T) {
	var calledDriver, calledRuntime, calledResource, calledSync, calledNvtx, calledState, calledDefault bool

	d := &Dispatcher{
		OnDriverApi: func(cbid uint32, data *ApiData) {
			calledDriver = true
		},
		OnRuntimeApi: func(cbid uint32, data *ApiData) {
			calledRuntime = true
		},
		OnResource: func(cbid uint32, data *ResourceData) {
			calledResource = true
		},
		OnSynchronize: func(cbid uint32, data *SynchronizeData) {
			calledSync = true
		},
		OnNvtx: func(cbid uint32, data *NvtxData) {
			calledNvtx = true
		},
		OnState: func(cbid uint32, data *StateData) {
			calledState = true
		},
		OnDefault: func(evt *Event) {
			calledDefault = true
		},
	}

	d.HandleCallback(&Event{Domain: cupti.CbDomainDriverApi})
	if !calledDriver {
		t.Error("expected OnDriverApi to be called")
	}

	d.HandleCallback(&Event{Domain: cupti.CbDomainRuntimeApi})
	if !calledRuntime {
		t.Error("expected OnRuntimeApi to be called")
	}

	d.HandleCallback(&Event{Domain: cupti.CbDomainResource})
	if !calledResource {
		t.Error("expected OnResource to be called")
	}

	d.HandleCallback(&Event{Domain: cupti.CbDomainSynchronize})
	if !calledSync {
		t.Error("expected OnSynchronize to be called")
	}

	d.HandleCallback(&Event{Domain: cupti.CbDomainNvtx})
	if !calledNvtx {
		t.Error("expected OnNvtx to be called")
	}

	d.HandleCallback(&Event{Domain: cupti.CbDomainState})
	if !calledState {
		t.Error("expected OnState to be called")
	}

	d.HandleCallback(&Event{Domain: cupti.CbDomainInvalid})
	if !calledDefault {
		t.Error("expected OnDefault to be called")
	}
}

func TestReentrancyTracker(t *testing.T) {
	tr := &reentrancyTracker{active: make(map[uint64]int)}

	d1, tid1 := tr.enter()
	if d1 != 1 {
		t.Errorf("initial depth = %d, want 1", d1)
	}

	d2, _ := tr.enter()
	if d2 != 2 {
		t.Errorf("nested depth = %d, want 2", d2)
	}

	tr.exit(tid1)
	tr.exit(tid1)

	if len(tr.active) != 0 {
		t.Errorf("active map length = %d after all exits, want 0", len(tr.active))
	}
}

// TestGeneratedVersionIsQualified pins the qualification table to the version
// the Go types were generated from. cupti.ApiVersion is that version, and
// the structs this package passes to CUPTI are its shapes.
//
// The table holds version literals, so nothing otherwise connects it to
// [cupti.ApiVersion] -- a regeneration moves the constant and leaves every
// literal, and neither the compiler nor any other test notices. Whatever else
// the table admits, it must admit the version whose struct shapes these types
// actually are.
func TestGeneratedVersionIsQualified(t *testing.T) {
	if !supportedVersion(cupti.ApiVersion) {
		t.Errorf("supportedVersion(cupti.ApiVersion = %d) = false, want true", cupti.ApiVersion)
	}
}
