package cudakv

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// Descriptor is a copied host command value. Generation identifies the ring
// lifetime; the ring does not interpret the other fields.
type Descriptor struct {
	SeqID       uint64
	Generation  uint64
	PromptToken int32
	Flags       uint32
}

// DeviceCommandRing is a bounded, mutex-protected host FIFO for one generation.
// It is not device-visible storage or a GPU publication protocol. The zero
// value cannot publish; use NewDeviceCommandRing.
type DeviceCommandRing struct {
	ring       []Descriptor
	head       int // next slot to publish
	tail       int // next slot to consume
	queued     int
	generation uint64
	closed     bool
	mu         sync.Mutex
}

var (
	errRingClosed = errors.New("cudakv: command ring closed")
	errRingFull   = errors.New("cudakv: command ring buffer full")
)

// NewDeviceCommandRing creates a ring with capacity slots pinned to generation.
// Capacity and generation must be nonzero, and the slot storage byte extent
// must fit in int. The caller chooses a generation for this ring's lifetime;
// a new generation requires a new ring. Closing the old ring permits its
// queued descriptors to drain and does not report completion of their work.
func NewDeviceCommandRing(capacity, generation uint64) (*DeviceCommandRing, error) {
	if generation == 0 {
		return nil, errors.New("cudakv: command ring generation must be nonzero")
	}
	maxCapacity := uint64(^uint(0)>>1) / uint64(unsafe.Sizeof(Descriptor{}))
	if capacity == 0 || capacity > maxCapacity {
		return nil, fmt.Errorf("cudakv: invalid command ring capacity %d", capacity)
	}
	return &DeviceCommandRing{
		ring:       make([]Descriptor, int(capacity)),
		generation: generation,
	}, nil
}

// Publish copies desc into the next slot. Its generation must match the ring's
// nonzero generation. A refusal leaves the queued values and order unchanged.
func (r *DeviceCommandRing) Publish(desc Descriptor) error {
	if r == nil {
		return errors.New("cudakv: nil command ring")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return errRingClosed
	}
	if r.generation == 0 {
		return errors.New("cudakv: command ring must be created with NewDeviceCommandRing")
	}
	if desc.Generation != r.generation {
		return fmt.Errorf("cudakv: descriptor generation %d does not match ring generation %d", desc.Generation, r.generation)
	}
	if r.queued == len(r.ring) {
		return errRingFull
	}
	r.ring[r.head] = desc
	r.head++
	if r.head == len(r.ring) {
		r.head = 0
	}
	r.queued++
	return nil
}

// Consume removes and returns the oldest descriptor. It reports ok=false when
// the ring is empty.
func (r *DeviceCommandRing) Consume() (desc Descriptor, ok bool) {
	if r == nil {
		return Descriptor{}, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.queued == 0 {
		return Descriptor{}, false
	}
	desc = r.ring[r.tail]
	r.ring[r.tail] = Descriptor{}
	r.tail++
	if r.tail == len(r.ring) {
		r.tail = 0
	}
	r.queued--
	return desc, true
}

// Close rejects future publications while allowing already published work to drain.
func (r *DeviceCommandRing) Close() {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.closed = true
	r.mu.Unlock()
}

// Len reports the number of descriptors currently queued.
func (r *DeviceCommandRing) Len() int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.queued
}
