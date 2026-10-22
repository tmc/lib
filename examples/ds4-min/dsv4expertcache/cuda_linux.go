//go:build linux

package dsv4expertcache

import (
	"context"
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// CUDAUpload owns one bounded pinned stage and three uniform device slabs.
// It is not integrated into production. Callers serialize access and complete
// every consumer lease before Close. Graph capture is initially unsupported.
type CUDAUpload struct {
	backend cudaUploadBackend
	strides [3]int
	slots   int
	closed  bool
	closing bool
	drained bool
	failed  bool
	starts  uint64
	events  map[cudaUploadEvent]bool
	lease   *Lease
}

// The backend seam exercises the same ownership transitions without a device.
// The real backend below retains all CUDA calls and partially freed handles.
type cudaUploadBackend interface {
	allocate(int, [3]int) error
	stage() []byte
	newEvent() (cudaUploadEvent, error)
	copy(int, int, []byte) error
	drain() error
	freeSlab(int) error
	freeStage() error
	destroyStream() error
}

type cudaUploadEvent interface {
	record() error
	wait() error
	destroy() error
}

// NewCUDAUpload allocates admitted uniform triplet storage within budget.
// On allocation failure with unsuccessful cleanup, it returns the retained
// owner and an error; the caller must retry Close rather than discard it.
func NewCUDAUpload(device *cuda.Device, slots int, strides [3]int, budget uint64) (*CUDAUpload, error) {
	if device == nil {
		return nil, errors.New("invalid CUDA upload device")
	}
	return newCUDAUpload(&cudaUploadDevice{device: device}, slots, strides, budget)
}

func newCUDAUpload(backend cudaUploadBackend, slots int, strides [3]int, budget uint64) (*CUDAUpload, error) {
	if slots <= 0 {
		return nil, errors.New("invalid CUDA upload slots")
	}
	var bytes uint64
	for _, n := range strides {
		if n <= 0 || uint64(slots) > uint64(^uint(0)>>1)/uint64(n) {
			return nil, errors.New("invalid CUDA triplet stride")
		}
		size := uint64(slots) * uint64(n)
		if size > budget-bytes {
			return nil, errors.New("CUDA expert slabs exceed budget")
		}
		bytes += size
	}
	u := &CUDAUpload{backend: backend, slots: slots, strides: strides, events: make(map[cudaUploadEvent]bool)}
	if err := backend.allocate(slots, strides); err != nil {
		if cleanup := u.Close(); cleanup != nil {
			return u, errors.Join(err, cleanup)
		}
		return nil, err
	}
	return u, nil
}

type cudaUploadCompletion struct {
	owner    *CUDAUpload
	event    cudaUploadEvent
	complete bool
}

func (e *cudaUploadCompletion) Wait() error {
	if e.complete {
		return nil
	}
	if e.event == nil {
		if err := e.owner.backend.drain(); err != nil {
			return err
		}
	} else if err := e.event.wait(); err != nil {
		return err
	}
	// Completion must remain successful and must not destroy the event. Close
	// owns event teardown after the consumer lease has ended.
	e.complete = true
	return nil
}

// Submit enqueues a component from this adapter's pinned stage. Its completion
// falls back to a stream drain if event recording fails after submission.
// Callers must Wait before reusing the pinned stage, including on submission
// error, and must complete every consumer before Close. Stage manages these
// upload waits and requires consumer completion through Lease.Release.
func (u *CUDAUpload) Submit(upload Upload, component int, src []byte) (Completion, error) {
	if u == nil || u.closed || u.closing || u.failed || component < 0 || component >= 3 || upload.Slot < 0 || upload.Slot >= u.slots || len(src) != u.strides[component] || len(src) == 0 {
		return nil, errors.New("invalid CUDA expert upload")
	}
	stage := u.backend.stage()
	if len(stage) < len(src) || unsafe.Pointer(&src[0]) != unsafe.Pointer(&stage[0]) {
		return nil, errors.New("invalid CUDA expert staging address")
	}
	e := &cudaUploadCompletion{owner: u}
	u.starts++
	var err error
	e.event, err = u.backend.newEvent()
	if e.event != nil {
		u.events[e.event] = true
	}
	if err != nil {
		u.failed = true
		// Creation can succeed before restoring the context fails. Retain
		// that event for teardown, but never wait on an unrecorded event.
		e.event = nil
		return e, err
	}
	if err = u.backend.copy(component, upload.Slot, src); err == nil {
		err = e.event.record()
	}
	if err != nil {
		u.failed = true
		if destroyErr := e.event.destroy(); destroyErr == nil {
			delete(u.events, e.event)
		} else {
			err = errors.Join(err, destroyErr)
		}
		// Even a failed async copy may have submitted work. Do not use an
		// unrecorded event as proof that staging storage can be reused.
		e.event = nil
	}
	return e, err
}

// Stage uses the owned pinned buffer with the backend-neutral source planner.
func (u *CUDAUpload) Stage(source *Source, lease *Lease, graphCapture bool) error {
	if u == nil || u.closed || u.closing || u.failed {
		return errors.New("CUDA upload closed")
	}
	if u.lease != nil && !u.lease.done {
		return errors.New("CUDA upload consumer lease active")
	}
	if u.lease != nil {
		// Event completion covers GPU work, but the stream owns copy-retention
		// callbacks. Drain them before starting another bounded lease.
		if err := u.backend.drain(); err != nil {
			return fmt.Errorf("release CUDA upload transfers: %w", err)
		}
		if err := u.destroyEvents(); err != nil {
			return fmt.Errorf("release CUDA upload events: %w", err)
		}
		u.lease = nil
	}
	starts := u.starts
	err := source.Stage(lease, u.backend.stage(), u, graphCapture)
	if err == nil {
		u.lease = lease
	} else if u.starts != starts {
		// Abort may fail to drain and close the cache. Do not let a new
		// cache overwrite staging bytes that remain in flight.
		u.failed = true
	}
	return err
}

// Close drains upload work before freeing storage. It must not race consumers.
// A failed drain retains allocations, since freeing active memory is unsafe.
func (u *CUDAUpload) Close() error {
	if u == nil || u.closed {
		return nil
	}
	u.closing = true
	if u.lease != nil && !u.lease.done {
		return errors.New("CUDA upload consumer lease active")
	}
	if !u.drained {
		if err := u.backend.drain(); err != nil {
			return fmt.Errorf("drain CUDA uploads: %w", err)
		}
		u.drained = true
	}
	if err := u.destroyEvents(); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if err := u.backend.freeSlab(i); err != nil {
			return err
		}
	}
	if err := u.backend.freeStage(); err != nil {
		return err
	}
	if err := u.backend.destroyStream(); err != nil {
		return err
	}
	u.closed = true
	return nil
}

func (u *CUDAUpload) destroyEvents() error {
	var err error
	for event := range u.events {
		if e := event.destroy(); e != nil {
			err = errors.Join(err, e)
		} else {
			delete(u.events, event)
		}
	}
	return err
}

type cudaUploadDevice struct {
	device  *cuda.Device
	stream  *cuda.Stream
	host    cudamem.HostAllocation[byte]
	slabs   [3]cudamem.Slice[byte]
	strides [3]int
}

func (b *cudaUploadDevice) allocate(slots int, strides [3]int) error {
	b.strides = strides
	return b.device.WithContext(func() error {
		var err error
		b.stream, err = b.device.NewStreamWithFlags(driver.StreamNonBlocking)
		if err != nil {
			return err
		}
		b.host, err = cudamem.AllocHost[byte](max(strides[0], strides[1], strides[2]))
		if err != nil {
			return err
		}
		for i, n := range strides {
			b.slabs[i], err = cudamem.Alloc[byte](b.device, slots*n)
			if err != nil {
				return err
			}
		}
		return nil
	})
}

func (b *cudaUploadDevice) stage() []byte { return b.host.Slice() }
func (b *cudaUploadDevice) newEvent() (cudaUploadEvent, error) {
	var event *cuda.Event
	err := b.device.WithContext(func() error {
		var err error
		event, err = b.device.NewEventWithFlags(driver.EventDisableTiming)
		return err
	})
	if event == nil {
		return nil, err
	}
	return &cudaUploadDeviceEvent{event, b.stream}, err
}
func (b *cudaUploadDevice) copy(component, slot int, src []byte) error {
	return b.device.WithContext(func() error {
		n := b.strides[component]
		return b.slabs[component].Slice(slot*n, (slot+1)*n).CopyFromHostAsync(src, b.stream)
	})
}
func (b *cudaUploadDevice) drain() error {
	if b.stream == nil {
		return nil
	}
	return b.stream.Synchronize(context.Background())
}
func (b *cudaUploadDevice) freeSlab(i int) error {
	if err := b.slabs[i].Free(); err != nil {
		return err
	}
	b.slabs[i] = cudamem.Slice[byte]{}
	return nil
}
func (b *cudaUploadDevice) freeStage() error {
	if err := b.host.Free(); err != nil {
		return err
	}
	b.host = cudamem.HostAllocation[byte]{}
	return nil
}
func (b *cudaUploadDevice) destroyStream() error {
	return b.destroyStreamWithOperations((*cuda.Stream).Close, func(s *cuda.Stream) bool {
		live := false
		_ = s.WithHandle(func(driver.Stream) error { live = true; return nil })
		return live
	})
}

func (b *cudaUploadDevice) destroyStreamWithOperations(destroy func(*cuda.Stream) error, live func(*cuda.Stream) bool) error {
	if b.stream == nil {
		return nil
	}
	err := destroy(b.stream)
	// Destroy can consume the raw handle before restoring the context fails.
	// WithHandle inspects wrapper lifetime without making another CUDA call.
	if err == nil || !live(b.stream) {
		b.stream = nil
	}
	return err
}

type cudaUploadDeviceEvent struct {
	event  *cuda.Event
	stream *cuda.Stream
}

func (e *cudaUploadDeviceEvent) record() error  { return e.event.Record(e.stream) }
func (e *cudaUploadDeviceEvent) wait() error    { return e.event.Synchronize(context.Background()) }
func (e *cudaUploadDeviceEvent) destroy() error { return e.event.Close() }
