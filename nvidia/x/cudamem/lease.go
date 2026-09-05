package cudamem

import (
	"errors"
	"sync"

	"github.com/tmc/lib/nvidia/x/cuda"
)

// ErrInUse means an asynchronous transfer still retains an allocation.
// Establish completion before retrying Free.
var ErrInUse = errors.New("cudamem: allocation is in use")

// A native enqueue can report an earlier asynchronous failure. Do not infer
// that no work owns the buffers from that error; retain them until the stream
// establishes completion. Pre-enqueue validation failures own no GPU work.
func failedTransferCleanup(attempted bool, register func(func()), release func()) {
	if attempted {
		register(release)
	} else {
		release()
	}
}

// retain prevents explicit Free through any alias. Merely retaining a Go
// pointer to allocation would not prevent a caller from freeing native memory.
func (s Slice[T]) retain(device *cuda.Device) (func(), error) {
	a := s.alloc
	if a == nil {
		return nil, errors.New("cudamem: allocation has no ownership record")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.ptr == 0 {
		return nil, errors.New("cudamem: allocation is freed")
	}
	if a.device != device {
		return nil, errors.New("cudamem: allocation and stream devices differ")
	}
	a.users++
	var once sync.Once
	return func() {
		once.Do(func() { a.mu.Lock(); a.users--; a.mu.Unlock() })
	}, nil
}

func (h HostAllocation[T]) retain() (func(), error) {
	a := h.alloc
	if a == nil {
		return nil, errors.New("cudamem: host allocation has no ownership record")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.ptr == 0 {
		return nil, errors.New("cudamem: host allocation is freed")
	}
	a.users++
	var once sync.Once
	return func() { once.Do(func() { a.mu.Lock(); a.users--; a.mu.Unlock() }) }, nil
}
