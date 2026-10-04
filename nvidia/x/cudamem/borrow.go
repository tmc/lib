package cudamem

import (
	"errors"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// Device returns the allocation's actual owner, or nil after it is freed.
func (s Slice[T]) Device() *cuda.Device {
	if s.alloc == nil {
		return nil
	}
	s.alloc.mu.Lock()
	defer s.alloc.mu.Unlock()
	if s.alloc.ptr == 0 {
		return nil
	}
	return s.alloc.device
}

// Borrow validates the allocation owner and prevents Free through every alias
// until the returned release function is called. Release is safe to repeat.
// A caller that submits asynchronous work must retain the borrow until actual
// device completion; cancellation of a host wait does not release it.
func (s Slice[T]) Borrow(device *cuda.Device) (func(), error) {
	if device == nil {
		return nil, errors.New("cudamem: nil borrow device")
	}
	return s.retain(device)
}
