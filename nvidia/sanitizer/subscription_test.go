package sanitizer

import (
	"strings"
	"sync"
	"testing"
	"unsafe"
)

// TestDispatchWithNoHandler checks the reentrancy guard: the C-callable
// trampoline outlives every subscription, because purego cannot free one, so
// an invocation arriving after Unsubscribe must return quietly rather than
// call through a nil function value.
//
// This exercises the Go side of that path directly. It does not establish
// anything about what the library does after unsubscribe.
func TestDispatchWithNoHandler(t *testing.T) {
	saved := handler.Load()
	t.Cleanup(func() { handler.Store(saved) })

	handler.Store(nil)
	dispatch(0, 0, 0, nil) // must not panic
}

// TestDispatchRoutesToHandler checks that the trampoline forwards every
// argument unchanged to the installed Go callback.
func TestDispatchRoutesToHandler(t *testing.T) {
	saved := handler.Load()
	t.Cleanup(func() { handler.Store(saved) })

	var got struct {
		userdata uintptr
		domain   CallbackDomain
		cbid     uint32
		cbdata   unsafe.Pointer
		calls    int
	}
	handler.Store(&handlerState{fn: func(userdata uintptr, domain CallbackDomain, cbid uint32, cbdata unsafe.Pointer) {
		got.userdata, got.domain, got.cbid, got.cbdata = userdata, domain, cbid, cbdata
		got.calls++
	}})

	cbdata := unsafe.Pointer(&got)
	dispatch(0xDEAD, CbDomainDriverApi, 7, cbdata)

	if got.calls != 1 {
		t.Fatalf("handler called %d times, want 1", got.calls)
	}
	if got.userdata != 0xDEAD || got.domain != CbDomainDriverApi || got.cbid != 7 || got.cbdata != cbdata {
		t.Errorf("handler got (%#x, %v, %d, %#x), want (0xdead, CbDomainDriverApi, 7, the pointer passed in)",
			got.userdata, got.domain, got.cbid, got.cbdata)
	}
}

// TestSubscribeFuncNilCallback checks that a nil callback is rejected in Go,
// before any trampoline exists, rather than installing a handler that would
// crash the process on the first event.
func TestSubscribeFuncNilCallback(t *testing.T) {
	sub, err := SubscribeFunc(nil, 0)
	if err == nil {
		if sub != nil {
			sub.Unsubscribe()
		}
		t.Fatal("SubscribeFunc(nil) returned no error")
	}
	if !strings.Contains(err.Error(), "nil callback") {
		t.Errorf("SubscribeFunc(nil) error = %v, want it to name the nil callback", err)
	}
}

// TestSubscribeFuncWithoutLibrary checks that subscribing on a machine with no
// Compute Sanitizer install returns the loader's error and does not allocate a
// purego callback slot. Those slots come from a small fixed pool and are never
// reclaimed, so burning one on a subscription that cannot happen is a leak.
//
// Where the library is present this test cannot observe that path and skips
// with a named reason; it never subscribes for real, because a real
// subscription is process-wide state and would have to precede every CUDA call
// to mean anything.
func TestSubscribeFuncWithoutLibrary(t *testing.T) {
	if err := checkLibSym("sanitizerSubscribe"); err == nil {
		t.Skip("library present: the no-library path is not observable here")
	}

	sub, err := SubscribeFunc(func(uintptr, CallbackDomain, uint32, unsafe.Pointer) {}, 0)
	if err == nil {
		sub.Unsubscribe()
		t.Fatal("SubscribeFunc succeeded with no library loaded")
	}
	if sub != nil {
		t.Error("SubscribeFunc returned a non-nil Subscription alongside an error")
	}
	if trampoline != 0 {
		t.Error("a purego callback slot was allocated even though subscribing could not proceed")
	}
	if handler.Load() != nil {
		t.Error("a handler was installed even though subscribing failed")
	}
}

// TestUnsubscribeIsIdempotent checks that Unsubscribe calls the library once
// and replays that result, so a deferred Unsubscribe next to an explicit one
// does not unsubscribe a subscriber that a later caller now owns.
//
// It drives the sync.Once directly, with no library involved: the underlying
// unsubscribe cannot run here.
func TestUnsubscribeIsIdempotent(t *testing.T) {
	saved := handler.Load()
	t.Cleanup(func() { handler.Store(saved) })
	handler.Store(&handlerState{fn: func(uintptr, CallbackDomain, uint32, unsafe.Pointer) {}})

	calls := 0
	s := &Subscription{}
	run := func() error {
		s.once.Do(func() {
			calls++
			handler.Store(nil)
		})
		return s.err
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() { defer wg.Done(); run() }()
	}
	wg.Wait()
	if calls != 1 {
		t.Errorf("teardown ran %d times, want 1", calls)
	}
	if handler.Load() != nil {
		t.Error("handler slot still populated after teardown")
	}
}

// TestSubscriptionHandleRoundTrip checks that Handle returns the handle the
// subscription holds, since the generated Subscriber methods are reached
// through it.
func TestSubscriptionHandleRoundTrip(t *testing.T) {
	s := &Subscription{sub: Subscriber(0x1234)}
	if got := s.Handle(); got != Subscriber(0x1234) {
		t.Errorf("Handle() = %#x, want 0x1234", uintptr(got))
	}
}

func TestBoolToUint32(t *testing.T) {
	if boolToUint32(true) != 1 {
		t.Error("boolToUint32(true) != 1")
	}
	if boolToUint32(false) != 0 {
		t.Error("boolToUint32(false) != 0")
	}
}

// TestDispatchTranslatesRuntimeCBID checks that a CUDA 12 library's runtime
// callback IDs reach the handler in this package's numbering, and that other
// domains' IDs pass through.
func TestDispatchTranslatesRuntimeCBID(t *testing.T) {
	saved := handler.Load()
	t.Cleanup(func() { handler.Store(saved) })

	var got uint32
	fn := func(_ uintptr, _ CallbackDomain, cbid uint32, _ unsafe.Pointer) { got = cbid }
	tests := []struct {
		major  int
		domain CallbackDomain
		cbid   uint32
		want   uint32
	}{
		{12, CbDomainRuntimeApi, 254, uint32(RuntimeCBIDCudaMemAdviseV1)},
		{12, CbDomainRuntimeApi, 448, uint32(RuntimeCBIDCudaMemAdvise)},
		{12, CbDomainDriverApi, 254, 254},
		{13, CbDomainRuntimeApi, 448, uint32(RuntimeCBIDCudaMemAdvise)},
	}
	for _, tt := range tests {
		handler.Store(&handlerState{fn: fn, major: tt.major})
		dispatch(0, tt.domain, tt.cbid, nil)
		if got != tt.want {
			t.Errorf("CUDA %d, %v, cbid %d: handler got %d, want %d", tt.major, tt.domain, tt.cbid, got, tt.want)
		}
	}
}

// TestDispatchAllocs checks that translating a CUDA 12 runtime callback ID
// costs no allocation on the callback path.
func TestDispatchAllocs(t *testing.T) {
	saved := handler.Load()
	t.Cleanup(func() { handler.Store(saved) })

	var sum uint32
	handler.Store(&handlerState{fn: func(_ uintptr, _ CallbackDomain, cbid uint32, _ unsafe.Pointer) { sum += cbid }, major: 12})
	n := testing.AllocsPerRun(100, func() {
		for cbid := uint32(0); cbid < 490; cbid++ {
			dispatch(0, CbDomainRuntimeApi, cbid, nil)
		}
	})
	if n != 0 {
		t.Errorf("dispatch allocates %v times per run", n)
	}
}

// TestSubscriptionLibCBID checks the translation EnableCallback and
// CallbackState apply before calling the library.
func TestSubscriptionLibCBID(t *testing.T) {
	tests := []struct {
		major  int
		domain CallbackDomain
		cbid   RuntimeAPICBID
		want   uint32
		ok     bool
	}{
		{12, CbDomainRuntimeApi, RuntimeCBIDCudaMemAdvise, 448, true},
		{12, CbDomainRuntimeApi, RuntimeCBIDCudaMemAdviseV1, 254, true},
		{13, CbDomainRuntimeApi, RuntimeCBIDCudaMemAdvise, 448, true},
		{13, CbDomainRuntimeApi, RuntimeCBIDCudaMemAdviseV1, 0, false},
		{12, CbDomainDriverApi, 900, 900, true},
	}
	for _, tt := range tests {
		s := &Subscription{major: tt.major}
		got, err := s.libCBID(tt.domain, uint32(tt.cbid))
		if got != tt.want || (err == nil) != tt.ok {
			t.Errorf("CUDA %d: libCBID(%v, %v) = %d, %v, want %d, ok %v", tt.major, tt.domain, tt.cbid, got, err, tt.want, tt.ok)
		}
	}
}
