package dcgm

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	"github.com/ebitengine/purego"
)

const policyConditions = PolicyCondDbe | PolicyCondPci | PolicyCondMaxPagesRetired | PolicyCondThermal | PolicyCondPower | PolicyCondNvlink | PolicyCondXid

// PolicyNotifications owns one v2 policy registration. Close must succeed before
// its connection, group, host engine or library is closed. The zero value is not
// usable. The library is not unloaded by Close.
type PolicyNotifications struct {
	closeMu sync.Mutex
	mu      sync.Mutex
	handle  Handle
	group   GpuGrp
	mask    PolicyCondition
	token   uint64
	closed  bool
	queue   chan PolicyCallbackResponse
	dropped atomic.Uint64
}

var policyCallbacks = struct {
	sync.Mutex
	next   uint64
	active map[uint64]*PolicyNotifications
}{active: make(map[uint64]*PolicyNotifications)}

var policyCallbackLookup sync.Map // uint64 -> *PolicyNotifications

var policyTrampoline struct {
	once sync.Once
	ptr  FpRecvUpdates
	err  error
}

// NewPolicyNotifications registers a v2 callback without setting policy or
// triggering collection. Capacity must be between 1 and 4096. At most 64
// subscriptions can be active in a process. Overlapping conditions on the same
// handle and group are refused because DCGM unregisters by condition, not token.
// Raw registrations must not overlap a subscription owned by this helper.
func NewPolicyNotifications(handle Handle, group GpuGrp, condition PolicyCondition, capacity int) (*PolicyNotifications, error) {
	if handle == 0 || condition == 0 || condition & ^policyConditions != 0 || capacity < 1 || capacity > 4096 {
		return nil, fmt.Errorf("dcgm: invalid policy notification configuration")
	}
	policyTrampoline.once.Do(func() {
		// purego retains this single trampoline permanently. It captures no
		// subscription and receives only a numeric, never-reused token.
		defer func() {
			if v := recover(); v != nil {
				policyTrampoline.err = fmt.Errorf("dcgm: create policy callback: %v", v)
			}
		}()
		policyTrampoline.ptr = FpRecvUpdates(purego.NewCallback(policyNotificationCallback))
	})
	if policyTrampoline.err != nil {
		return nil, policyTrampoline.err
	}
	s := &PolicyNotifications{handle: handle, group: group, mask: condition, queue: make(chan PolicyCallbackResponse, capacity)}
	policyCallbacks.Lock()
	if len(policyCallbacks.active) >= 64 || policyCallbacks.next == math.MaxUint64 {
		policyCallbacks.Unlock()
		return nil, fmt.Errorf("dcgm: policy notification capacity exhausted")
	}
	for _, prior := range policyCallbacks.active {
		if prior.handle == handle && prior.group == group && prior.mask&condition != 0 {
			policyCallbacks.Unlock()
			return nil, fmt.Errorf("dcgm: policy conditions already registered")
		}
	}
	policyCallbacks.next++
	s.token = policyCallbacks.next
	policyCallbacks.active[s.token] = s
	policyCallbackLookup.Store(s.token, s)
	policyCallbacks.Unlock()
	if err := PolicyRegisterV2(handle, group, condition, policyTrampoline.ptr, s.token); err != nil {
		s.retire()
		return nil, fmt.Errorf("dcgm: register policy notifications: %w", err)
	}
	return s, nil
}

func policyNotificationCallback(response *PolicyCallbackResponse, token uint64) int32 {
	v, ok := policyCallbackLookup.Load(token)
	if !ok {
		return 0
	}
	s := v.(*PolicyNotifications)
	if !s.mu.TryLock() {
		s.dropped.Add(1)
		return 0
	}
	defer s.mu.Unlock()
	if s.closed {
		return 0
	}
	if response == nil {
		s.dropped.Add(1)
		return 0
	}
	// Copy while DCGM owns the callback argument. Preserve unknown versions and
	// condition payloads for the reader; no pointer into library memory escapes.
	select {
	case s.queue <- *response:
	default:
		s.dropped.Add(1)
	}
	return 0
}

// Notifications returns copied callback responses. Close closes the channel;
// queued responses remain available. Unknown response versions are retained.
func (s *PolicyNotifications) Notifications() <-chan PolicyCallbackResponse {
	if s == nil {
		return nil
	}
	return s.queue
}

// Dropped reports callbacks discarded because the queue was full, a concurrent
// callback or Close owned the queue, or the response pointer was nil. Late
// callbacks for retired tokens are ignored.
func (s *PolicyNotifications) Dropped() uint64 {
	if s == nil {
		return 0
	}
	return s.dropped.Load()
}

// Close unregisters the conditions and retires this subscription. An unregister
// error leaves it active so the caller can retry. Successful unregister does not
// prove native callback quiescence: the permanent trampoline ignores retired
// tokens, and an already-running callback is synchronized with channel close.
func (s *PolicyNotifications) Close() error {
	if s == nil || s.queue == nil {
		return fmt.Errorf("dcgm: invalid policy notifications")
	}
	s.closeMu.Lock()
	defer s.closeMu.Unlock()
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()
	if closed {
		return nil
	}
	if err := PolicyUnregister(s.handle, s.group, s.mask); err != nil {
		return fmt.Errorf("dcgm: unregister policy notifications: %w", err)
	}
	s.retire()
	return nil
}

func (s *PolicyNotifications) retire() {
	policyCallbacks.Lock()
	delete(policyCallbacks.active, s.token)
	policyCallbackLookup.Delete(s.token)
	policyCallbacks.Unlock()
	s.mu.Lock()
	if !s.closed {
		s.closed = true
		close(s.queue)
	}
	s.mu.Unlock()
}
