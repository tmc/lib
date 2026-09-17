// Package cudakv implements host-light ring buffers and KV tier prefetching.
package cudakv

import (
	"errors"
	"fmt"
	"sync"
)

type Descriptor struct {
	SeqID       uint64
	Generation  uint64
	PromptToken int32
	Flags       uint32
}

type DeviceCommandRing struct {
	ring     []Descriptor
	head     uint64
	tail     uint64
	capacity uint64
	closed   bool
	mu       sync.Mutex
}

var errRingClosed = errors.New("cudakv: command ring closed")

func NewDeviceCommandRing(cap uint64) *DeviceCommandRing {
	if cap == 0 {
		cap = 1
	}
	return &DeviceCommandRing{
		ring:     make([]Descriptor, cap),
		capacity: cap,
	}
}

func (r *DeviceCommandRing) Publish(desc Descriptor) error {
	if r == nil {
		return errors.New("cudakv: nil command ring")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return errRingClosed
	}
	if r.head-r.tail >= r.capacity {
		return fmt.Errorf("cudakv: command ring buffer full")
	}
	r.ring[r.head%r.capacity] = desc
	r.head++
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
	if r.tail == r.head {
		return Descriptor{}, false
	}
	desc = r.ring[r.tail%r.capacity]
	r.ring[r.tail%r.capacity] = Descriptor{}
	r.tail++
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
	return int(r.head - r.tail)
}
