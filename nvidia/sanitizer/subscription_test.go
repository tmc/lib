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
