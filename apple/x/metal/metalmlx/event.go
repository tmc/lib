package metalmlx

import (
	"fmt"
	"sync"

	"github.com/tmc/apple/metal"
)

// Event is a shared event, which orders work between the CPU and the GPU by
// a monotonically increasing value. Port of metal::EventImpl in
// mlx/backend/metal/event.cpp, without the error poisoning MLX layers on top.
type Event struct {
	mu sync.RWMutex
	ev metal.MTLSharedEvent
}

// NewEvent creates a shared event with value zero.
func NewEvent(d *Device) (*Event, error) {
	var ev metal.MTLSharedEvent
	WithPool(func() { ev = d.raw.NewSharedEvent() })
	if !valid(ev) {
		return nil, fmt.Errorf("metalmlx: failed to create Metal shared event")
	}
	return &Event{ev: ev}, nil
}

// MTLEvent returns the underlying shared event. The returned object is borrowed
// and remains valid only until Destroy.
func (e *Event) MTLEvent() (metal.MTLSharedEvent, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !valid(e.ev) {
		return nil, fmt.Errorf("metalmlx: event is closed")
	}
	return e.ev, nil
}

// Value returns the value the event has been signaled with.
func (e *Event) Value() (uint64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !valid(e.ev) {
		return 0, fmt.Errorf("metalmlx: event is closed")
	}
	return e.ev.SignaledValue(), nil
}

// IsSignaled reports whether the event has reached value.
func (e *Event) IsSignaled(value uint64) (bool, error) {
	got, err := e.Value()
	return got >= value, err
}

// Signal sets the event value from the CPU.
func (e *Event) Signal(value uint64) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !valid(e.ev) {
		return fmt.Errorf("metalmlx: event is closed")
	}
	e.ev.SetSignaledValue(value)
	return nil
}

// Wait blocks the calling goroutine until the event reaches value. It never
// times out.
func (e *Event) Wait(value uint64) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !valid(e.ev) {
		return fmt.Errorf("metalmlx: event is closed")
	}
	e.ev.WaitUntilSignaledValueTimeoutMS(value, ^uint64(0))
	return nil
}

// Destroy releases the event. It is idempotent. The caller must not encode a
// use concurrently with Destroy and must synchronize every command buffer that
// refers to the event before destroying it. CPU Signal, Wait, and Value calls
// are serialized with Destroy and return an error after it.
func (e *Event) Destroy() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if !valid(e.ev) {
		return
	}
	release(e.ev)
	e.ev = nil
}
