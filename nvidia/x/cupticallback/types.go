package cupticallback

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// ApiData represents copied, safe data from a driver or runtime API callback.
type ApiData struct {
	CallbackSite    cupti.ApiCallbackSite
	FunctionName    string
	SymbolName      string
	Context         cupti.Context
	ContextUid      uint32
	CorrelationId   uint32
	CorrelationData uint64
}

// ResourceData represents copied, safe data from a resource callback.
type ResourceData struct {
	Context            cupti.Context
	ResourceHandle     uint64
	ResourceDescriptor uintptr
}

// SynchronizeData represents copied, safe data from a synchronization callback.
type SynchronizeData struct {
	Context cupti.Context
	Stream  cupti.Stream
}

// NvtxData represents copied, safe data from an NVTX callback.
type NvtxData struct {
	FunctionName string
}

// StateData represents copied, safe data from a state callback.
type StateData struct {
	Notification [2]uint64
}

// Event represents a decoded, safe-to-retain CUPTI callback event.
type Event struct {
	Userdata  uintptr
	Domain    cupti.CallbackDomain
	ID        uint32
	Name      string
	Reentrant bool

	Api   *ApiData
	Res   *ResourceData
	Sync  *SynchronizeData
	Nvtx  *NvtxData
	State *StateData

	// Raw points to the native callback payload. Valid only during callback execution.
	Raw unsafe.Pointer
}

// Handler handles CUPTI callback events.
type Handler interface {
	HandleCallback(evt *Event)
}

// HandlerFunc is an adapter allowing ordinary functions to serve as Handlers.
type HandlerFunc func(evt *Event)

// HandleCallback calls f(evt).
func (f HandlerFunc) HandleCallback(evt *Event) {
	f(evt)
}

// RawCallbackFunc receives raw CUPTI callback invocations before any decoding.
// Pointers inside cbdata belong to the library and must not be dereferenced after
// the callback returns.
type RawCallbackFunc func(userdata uintptr, domain cupti.CallbackDomain, cbid uint32, cbdata unsafe.Pointer)

// Options configures callback subscription behavior.
type Options struct {
	// AllowReentrant allows nested callbacks on the same thread when user callback code
	// invokes CUDA driver or runtime functions. Default is false (suppresses reentrancy).
	AllowReentrant bool

	// RawCallback, if non-nil, receives raw callback invocations directly.
	RawCallback RawCallbackFunc

	// Userdata is an opaque value passed back to callback invocations.
	// It must not be a Go pointer.
	Userdata uintptr
}

// Dispatcher routes callback events to per-domain handler functions.
type Dispatcher struct {
	OnDriverApi   func(cbid uint32, data *ApiData)
	OnRuntimeApi  func(cbid uint32, data *ApiData)
	OnResource    func(cbid uint32, data *ResourceData)
	OnSynchronize func(cbid uint32, data *SynchronizeData)
	OnNvtx        func(cbid uint32, data *NvtxData)
	OnState       func(cbid uint32, data *StateData)
	OnDefault     func(evt *Event)
}

// HandleCallback routes evt to the matching domain handler.
func (d *Dispatcher) HandleCallback(evt *Event) {
	switch evt.Domain {
	case cupti.CbDomainDriverApi:
		if d.OnDriverApi != nil {
			d.OnDriverApi(evt.ID, evt.Api)
			return
		}
	case cupti.CbDomainRuntimeApi:
		if d.OnRuntimeApi != nil {
			d.OnRuntimeApi(evt.ID, evt.Api)
			return
		}
	case cupti.CbDomainResource:
		if d.OnResource != nil {
			d.OnResource(evt.ID, evt.Res)
			return
		}
	case cupti.CbDomainSynchronize:
		if d.OnSynchronize != nil {
			d.OnSynchronize(evt.ID, evt.Sync)
			return
		}
	case cupti.CbDomainNvtx:
		if d.OnNvtx != nil {
			d.OnNvtx(evt.ID, evt.Nvtx)
			return
		}
	case cupti.CbDomainState:
		if d.OnState != nil {
			d.OnState(evt.ID, evt.State)
			return
		}
	}
	if d.OnDefault != nil {
		d.OnDefault(evt)
	}
}
