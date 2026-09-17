// Package cudakv implements host-light ring buffers and KV tier prefetching.
package cudakv

import (
	"fmt"
	"sync/atomic"
)

type Descriptor struct {
	SeqID       uint64
	PromptToken int32
	Flags       uint32
}

type DeviceCommandRing struct {
	ring     []Descriptor
	head     uint64
	tail     uint64
	capacity uint64
}

func NewDeviceCommandRing(cap uint64) *DeviceCommandRing {
	return &DeviceCommandRing{
		ring:     make([]Descriptor, cap),
		capacity: cap,
	}
}

func (r *DeviceCommandRing) Publish(desc Descriptor) error {
	nextHead := atomic.LoadUint64(&r.head) + 1
	if nextHead-atomic.LoadUint64(&r.tail) > r.capacity {
		return fmt.Errorf("cudakv: command ring buffer full")
	}
	r.ring[r.head%r.capacity] = desc
	atomic.StoreUint64(&r.head, nextHead)
	return nil
}
