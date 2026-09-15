package cupticallback

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/cupti"
)

var (
	trampolineOnce sync.Once
	trampoline     uintptr
	trampolineErr  error

	// subscribeMu serializes Subscribe so two goroutines cannot race to
	// establish a subscription.
	subscribeMu sync.Mutex

	handlerSlot atomic.Pointer[handlerState]
	tracker     = &reentrancyTracker{active: make(map[uint64]int)}
)

type handlerState struct {
	opts    Options
	handler Handler
}

func getTrampoline() (uintptr, error) {
	trampolineOnce.Do(func() {
		defer func() {
			if r := recover(); r != nil {
				trampolineErr = fmt.Errorf("cupticallback: create callback trampoline: %v", r)
			}
		}()
		trampoline = purego.NewCallback(dispatch)
	})
	if trampolineErr != nil {
		return 0, trampolineErr
	}
	return trampoline, nil
}

func dispatch(userdata uintptr, domain cupti.CallbackDomain, cbid uint32, cbdata unsafe.Pointer) {
	h := handlerSlot.Load()
	if h == nil {
		return
	}

	depth, tid := tracker.enter()
	defer tracker.exit(tid)

	reentrant := depth > 1
	if reentrant && !h.opts.AllowReentrant {
		return
	}

	if h.opts.RawCallback != nil {
		h.opts.RawCallback(userdata, domain, cbid, cbdata)
	}

	if h.handler != nil {
		evt := decodeEvent(userdata, domain, cbid, cbdata, reentrant)
		h.handler.HandleCallback(evt)
	}
}

type rawCallbackData struct {
	callbackSite        cupti.ApiCallbackSite
	functionName        *byte
	functionParams      unsafe.Pointer
	functionReturnValue unsafe.Pointer
	symbolName          *byte
	context             cupti.Context
	contextUid          uint32
	correlationData     *uint64
	correlationId       uint32
}

func decodeEvent(userdata uintptr, domain cupti.CallbackDomain, cbid uint32, cbdata unsafe.Pointer, reentrant bool) *Event {
	evt := &Event{
		Userdata:  userdata,
		Domain:    domain,
		ID:        cbid,
		Reentrant: reentrant,
		Raw:       cbdata,
	}

	if cbdata == nil {
		return evt
	}

	switch domain {
	case cupti.CbDomainDriverApi, cupti.CbDomainRuntimeApi:
		raw := (*rawCallbackData)(cbdata)
		api := &ApiData{
			CallbackSite:  raw.callbackSite,
			FunctionName:  goString(raw.functionName),
			SymbolName:    goString(raw.symbolName),
			Context:       raw.context,
			ContextUid:    raw.contextUid,
			CorrelationId: raw.correlationId,
		}
		if raw.correlationData != nil {
			api.CorrelationData = *raw.correlationData
		}
		evt.Api = api
		evt.Name = api.FunctionName

	case cupti.CbDomainResource:
		raw := (*cupti.ResourceData)(cbdata)
		evt.Res = &ResourceData{
			Context:            raw.Context,
			ResourceHandle:     raw.ResourceHandle[0],
			ResourceDescriptor: raw.ResourceDescriptor,
		}

	case cupti.CbDomainSynchronize:
		raw := (*cupti.SynchronizeData)(cbdata)
		evt.Sync = &SynchronizeData{
			Context: raw.Context,
			Stream:  raw.Stream,
		}

	case cupti.CbDomainNvtx:
		raw := (*struct {
			functionName *byte
		})(cbdata)
		nvtx := &NvtxData{
			FunctionName: goString(raw.functionName),
		}
		evt.Nvtx = nvtx
		evt.Name = nvtx.FunctionName

	case cupti.CbDomainState:
		raw := (*cupti.StateData)(cbdata)
		evt.State = &StateData{
			Notification: raw.Notification,
		}
	}

	return evt
}

func goString(p *byte) string {
	if p == nil {
		return ""
	}
	var b []byte
	for ptr := p; *ptr != 0; ptr = (*byte)(unsafe.Add(unsafe.Pointer(ptr), 1)) {
		b = append(b, *ptr)
	}
	return string(b)
}

// Subscription represents an active CUPTI callback subscription.
type Subscription struct {
	mu          sync.Mutex
	sub         cupti.Subscriber
	opts        Options
	closed      bool
	unsubscribe func() error
}

// Subscribe initializes and registers a CUPTI callback subscriber with the given options and handler.
//
// The native CUPTI API admits at most one active subscriber per process.
// If a subscriber is already active, Subscribe returns cupti.ErrMultipleSubscribersNotSupported.
func Subscribe(opts Options, handler Handler) (*Subscription, error) {
	if err := CheckVersion(); err != nil {
		return nil, err
	}
	if handler == nil && opts.RawCallback == nil {
		return nil, fmt.Errorf("cupticallback: subscribe: handler and RawCallback cannot both be nil")
	}

	subscribeMu.Lock()
	defer subscribeMu.Unlock()

	if handlerSlot.Load() != nil {
		return nil, fmt.Errorf("cupticallback: subscribe: subscriber already active in process: %w", cupti.ErrMultipleSubscribersNotSupported)
	}

	cb, err := getTrampoline()
	if err != nil {
		return nil, err
	}

	sub, err := cupti.Subscribe(cb, opts.Userdata)
	if err != nil {
		return nil, fmt.Errorf("cupticallback: subscribe: %w", err)
	}

	s := &Subscription{
		sub:         sub,
		unsubscribe: sub.Unsubscribe,
		opts:        opts,
	}

	handlerSlot.Store(&handlerState{
		opts:    opts,
		handler: handler,
	})

	return s, nil
}

// SubscribeFunc registers fn as a callback handler with default options.
func SubscribeFunc(fn func(evt *Event)) (*Subscription, error) {
	if fn == nil {
		return nil, fmt.Errorf("cupticallback: subscribe: nil handler function")
	}
	return Subscribe(Options{}, HandlerFunc(fn))
}

// Handle returns the raw CUPTI subscriber handle.
func (s *Subscription) Handle() cupti.Subscriber {
	return s.sub
}

// EnableDomain enables or disables every callback in domain.
func (s *Subscription) EnableDomain(domain cupti.CallbackDomain, enable bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var val uint32
	if enable {
		val = 1
	}
	if err := s.sub.EnableDomain(val, domain); err != nil {
		return fmt.Errorf("cupticallback: enable domain %v: %w", domain, err)
	}
	return nil
}

// EnableCallback enables or disables a specific callback in domain.
func (s *Subscription) EnableCallback(domain cupti.CallbackDomain, cbid uint32, enable bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var val uint32
	if enable {
		val = 1
	}
	if err := s.sub.EnableCallback(val, domain, cbid); err != nil {
		return fmt.Errorf("cupticallback: enable callback (%v, %d): %w", domain, cbid, err)
	}
	return nil
}

// EnableAllDomains enables or disables callbacks in all domains.
func (s *Subscription) EnableAllDomains(enable bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var val uint32
	if enable {
		val = 1
	}
	if err := s.sub.EnableAllDomains(val); err != nil {
		return fmt.Errorf("cupticallback: enable all domains: %w", err)
	}
	return nil
}

// CallbackState reports whether a callback in domain is enabled.
func (s *Subscription) CallbackState(domain cupti.CallbackDomain, cbid uint32) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.sub.GetCallbackState(domain, cbid)
	if err != nil {
		return false, fmt.Errorf("cupticallback: get callback state (%v, %d): %w", domain, cbid, err)
	}
	return state != 0, nil
}

// EnabledCallbacks returns the list of enabled callback IDs for the specified domain.
func (s *Subscription) EnabledCallbacks(domain cupti.CallbackDomain) ([]uint32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	c, err := s.sub.GetEnabledCallbacks(domain, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("cupticallback: get enabled callbacks: %w", err)
	}
	if c == 0 {
		return nil, nil
	}
	buf := make([]uint32, c)
	size := c * 4
	count, err := s.sub.GetEnabledCallbacks(domain, &buf[0], &size)
	if err != nil {
		return nil, fmt.Errorf("cupticallback: get enabled callbacks: %w", err)
	}
	return buf[:count], nil
}

// Unsubscribe removes the subscriber and drops the callback handler.
// Unsubscribe is idempotent after success. Failure retains the subscription
// and callback handler so cleanup can be retried.
func (s *Subscription) Unsubscribe() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	unsubscribe := s.unsubscribe
	if unsubscribe == nil {
		unsubscribe = s.sub.Unsubscribe
	}
	if err := unsubscribe(); err != nil {
		return fmt.Errorf("cupticallback: unsubscribe: %w", err)
	}
	handlerSlot.Store(nil)
	s.closed = true
	return nil
}
