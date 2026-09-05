package sanitizer

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/ebitengine/purego"
)

// CallbackFunc is the Go form of Sanitizer_CallbackFunc, the one host callback
// this API takes:
//
//	typedef void (SANITIZERAPI *Sanitizer_CallbackFunc)(void *userdata,
//	    Sanitizer_CallbackDomain domain, Sanitizer_CallbackId cbid,
//	    const void *cbdata);
//
// Register one with [SubscribeFunc], not with purego.NewCallback: this package
// creates exactly one C-callable trampoline for the whole process and
// dispatches through it. See [Subscription] for what a callback may do.
//
// userdata is the opaque value handed to [SubscribeFunc], returned unchanged.
// It is a uintptr because the library stores it across calls, where a Go
// pointer may not go.
//
// cbdata points at a per-domain struct in memory the library owns -- a
// [CallbackData] for the driver and runtime API domains, a [ResourceMemoryData]
// for a memory resource callback, and so on. It is valid only until this
// function returns, so anything worth keeping must be copied out here. It is
// nil for domains that carry no data.
//
// This type is written by hand rather than generated so that cbdata is an
// unsafe.Pointer. The generated form makes every callback pointer a uintptr,
// which is correct on the wire but pushes a uintptr-to-pointer conversion into
// every callback body, and go vet reports each one as a possible misuse of
// unsafe.Pointer. Nothing is lost: the pointer addresses library memory, which
// the collector does not manage either way.
type CallbackFunc func(userdata uintptr, domain CallbackDomain, cbid uint32, cbdata unsafe.Pointer)

// A Subscription is one registered callback subscriber.
//
// # One subscriber per process
//
// The native API admits at most one active subscriber at a time. A second
// sanitizerSubscribe while one is live fails with
// SANITIZER_ERROR_MAX_LIMIT_REACHED, reported here as [ErrMaxLimitReached].
// [SubscribeFunc] does not pre-check that condition and does not serialize
// callers behind a Go-side "is one already active" flag: the library is the
// authority on whether a subscriber exists, including subscribers registered
// by code this package never sees (a preloaded tool, a C library in the same
// process). A Go-side check would either duplicate that answer or contradict
// it, and contradicting it would replace the real error with an invented one.
//
// The single-subscriber rule is why this type exists at all rather than a bare
// [Subscriber] handle: a *Subscription is an owned resource with an
// [Subscription.Unsubscribe] that must be called, which is harder to misread
// as something you can freely make a second of.
//
// # Subscribe before CUDA
//
// For complete reporting, subscribe before the process makes any CUDA call.
// Callbacks describe events the sanitizer observes from the point of
// subscription onward; contexts, modules and allocations created earlier are
// not replayed. In a Go program that generally means subscribing from an init
// function, or from main before any package touches the driver. Note that
// another package's init can beat main to the driver.
//
// # Calling CUDA from inside a callback
//
// The headers state that there are restrictions on calling CUDA functions from
// within a callback, and supply a small set of sanitizer-provided replacements
// meant to be used instead: [StreamHandle.MemcpyDeviceToHost],
// [StreamHandle.MemcpyHostToDeviceAsync], [StreamHandle.Memset],
// [StreamHandle.Synchronize], [Alloc], [AllocHost], [Free] and [FreeHost],
// each documented as "can be called within a callback function".
//
// The pinned headers do not state the complete rule. They do not enumerate
// which CUDA entry points are safe, nor define the behavior of calling one
// that is not. This package therefore does not document a rule it cannot
// support: use the sanitizer-provided replacements above, treat any other CUDA
// call from inside a callback as unspecified, and consult the Sanitizer API
// reference manual for the release you are running against.
// [IsInsideRecursiveCallback] exists to let a callback detect that it was
// reached from within another callback.
//
// # Concurrency
//
// The headers state that the enable and query calls are not thread safe with
// respect to each other, and that concurrent use gives undefined results.
// The methods on *Subscription hold a mutex so that concurrent calls through
// one Subscription are serialized. That does not extend to calls made through
// the raw [Subscriber] handle from [Subscription.Handle], which are outside
// this type's control.
type Subscription struct {
	mu   sync.Mutex // serializes the enable/query calls, per the header's warning
	sub  Subscriber
	once sync.Once
	err  error
}

// handler holds the Go function the trampoline dispatches to.
//
// It is a pointer slot rather than a plain variable so that clearing it after
// unsubscribe is a single atomic store: a callback that the library is already
// executing loads nil and returns without touching a subscription that Go code
// has finished with. The Go func value itself lives here, reachable by the
// collector, for exactly as long as the subscription does.
var handler atomic.Pointer[handlerState]

type handlerState struct {
	fn CallbackFunc
}

var (
	trampolineOnce sync.Once
	trampoline     uintptr
	trampolineErr  error

	// subscribeMu serializes SubscribeFunc so two goroutines cannot both
	// believe they installed the handler. It does not attempt to enforce the
	// library's own one-subscriber rule; see [Subscription].
	subscribeMu sync.Mutex
)

// dispatch is the one C-callable entry point this package ever creates.
//
// No Go pointer crosses into C here. cbdata arrives as an unsafe.Pointer
// holding an address in the library's own memory, which the collector neither
// traces nor moves; purego writes the register value straight into the
// argument. userdata stays a uintptr because it is an opaque token the library
// stored earlier, not a pointer this package may follow.
func dispatch(userdata uintptr, domain CallbackDomain, cbid uint32, cbdata unsafe.Pointer) {
	h := handler.Load()
	if h == nil {
		// Unsubscribed, or a stale invocation racing teardown. The trampoline
		// address outlives every subscription (see newTrampoline), so this is
		// a reachable state and must be a no-op, not a nil call.
		return
	}
	h.fn(userdata, domain, cbid, cbdata)
}

// newTrampoline creates the process-wide C function pointer for dispatch.
//
// purego.NewCallback allocates from a small fixed pool (a couple of thousand
// entries) and never frees an entry, so a package that made one per
// subscription would leak a slot per subscribe/unsubscribe cycle and
// eventually exhaust the process. Exactly one is created here, on first
// successful subscribe, and it is reused for the life of the process; the
// handler slot above is what actually changes.
//
// It is created only after the library and the sanitizerSubscribe symbol have
// been confirmed, so a machine without Compute Sanitizer installed does not
// burn a slot on a subscription that cannot happen.
func newTrampoline() (uintptr, error) {
	trampolineOnce.Do(func() {
		defer func() {
			if r := recover(); r != nil {
				trampolineErr = fmt.Errorf("sanitizer: creating callback trampoline: %v", r)
			}
		}()
		trampoline = purego.NewCallback(dispatch)
	})
	if trampolineErr != nil {
		return 0, trampolineErr
	}
	return trampoline, nil
}

// SubscribeFunc registers fn as the process's callback subscriber and returns
// the resulting subscription.
//
// No callback is delivered until a domain is enabled; see
// [Subscription.EnableDomain] and [Subscription.EnableAllDomains]. Subscribe
// before the process makes any CUDA call, and read [Subscription] for the
// single-subscriber rule and the restrictions that apply inside fn.
//
// userdata is passed back to fn unchanged on every callback. It must not be a
// Go pointer. The library stores it and hands it back from a C thread, which
// is exactly the case Go's cgo pointer rules forbid: the collector cannot see
// the copy the library holds, so a Go pointer parked there can be moved or
// collected while the library still has it. Pass an index, a handle, or a
// pointer into memory the library or the C allocator owns, and keep the Go
// state in a package-level map keyed by that value.
//
// fn is kept alive by this package for the whole subscription and dropped by
// [Subscription.Unsubscribe]. Its cbdata argument points into memory the
// library owns and is valid only for the duration of the call: copy out
// anything needed afterwards. In particular the functionParams field of a
// driver or runtime API callback must not be dereferenced after fn returns.
//
// If the library is missing, SubscribeFunc returns the loader's error; if the
// symbol is missing it returns *SymbolError; if a subscriber already exists it
// returns the library's own [ErrMaxLimitReached]. It never panics.
func SubscribeFunc(fn CallbackFunc, userdata uintptr) (*Subscription, error) {
	if fn == nil {
		return nil, fmt.Errorf("sanitizer: SubscribeFunc: nil callback")
	}
	// Confirm the library and symbol before allocating a trampoline slot, so a
	// failed subscribe on a machine without the library costs nothing
	// permanent.
	if err := checkLibSym("sanitizerSubscribe"); err != nil {
		return nil, err
	}

	subscribeMu.Lock()
	defer subscribeMu.Unlock()

	cb, err := newTrampoline()
	if err != nil {
		return nil, err
	}

	sub, err := Subscribe(cb, userdata)
	if err != nil {
		// Leave the handler slot untouched. A second subscribe failing with
		// ErrMaxLimitReached must not disturb the subscription that caused the
		// failure.
		return nil, err
	}

	// Store the handler only after the library has accepted the subscription.
	// No callback can arrive before this point: subscribing enables nothing,
	// and enabling requires the handle being returned here.
	handler.Store(&handlerState{fn: fn})
	return &Subscription{sub: sub}, nil
}

// Handle returns the raw subscriber handle, for the generated
// [Subscriber] methods and for calls this type does not wrap.
//
// Calls made through the handle bypass the mutex documented on
// [Subscription], and unsubscribing through it leaves this Subscription
// believing it still owns a live subscriber.
func (s *Subscription) Handle() Subscriber { return s.sub }

// EnableDomain enables or disables every callback in domain.
func (s *Subscription) EnableDomain(enable bool, domain CallbackDomain) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return enableDomain(boolToUint32(enable), s.sub, domain)
}

// EnableAllDomains enables or disables every callback in every domain.
func (s *Subscription) EnableAllDomains(enable bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return enableAllDomains(boolToUint32(enable), s.sub)
}

// EnableCallback enables or disables one callback in one domain.
func (s *Subscription) EnableCallback(enable bool, domain CallbackDomain, cbid uint32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return enableCallback(boolToUint32(enable), s.sub, domain, cbid)
}

// CallbackState reports whether one callback in one domain is enabled.
func (s *Subscription) CallbackState(domain CallbackDomain, cbid uint32) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, err := getCallbackState(s.sub, domain, cbid)
	if err != nil {
		return false, err
	}
	return v != 0, nil
}

// Unsubscribe removes the subscriber and drops the callback.
//
// It is idempotent: the underlying sanitizerUnsubscribe runs once and every
// later call returns that first result. After it returns, the library issues
// no further callbacks; a callback still executing when it is called runs to
// completion, and one that somehow reaches the trampoline afterwards finds an
// empty handler slot and returns without calling Go code.
//
// The C-callable trampoline address is not freed, because purego cannot free
// one. What is released is the reference to the Go function, which is what
// keeps the callback's captured state alive.
func (s *Subscription) Unsubscribe() error {
	s.once.Do(func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.err = unsubscribe(s.sub)
		// Clear after the library has promised no further callbacks, so a
		// callback in flight during the call still reaches fn.
		handler.Store(nil)
	})
	return s.err
}

func boolToUint32(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}
