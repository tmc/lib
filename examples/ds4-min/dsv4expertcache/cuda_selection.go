package dsv4expertcache

import (
	"errors"
	"io"
	"sync"
)

type cudaSelectionUpload interface {
	Stage(*Source, *Lease, bool) error
	io.Closer
}

type cudaConsumer interface {
	Completion
	close() error
}

// CUDASelections serializes a shared upload cache through final consumer
// completion. Its zero value is unusable. Close is retryable after failures.
type CUDASelections struct {
	mu              sync.Mutex
	source          *Source
	cache           *Cache
	upload          cudaSelectionUpload
	view            func(int, int) (uint64, uint64, error)
	active          *CUDASelection
	closing, closed bool
}

// CUDASelection borrows a committed route-ordered set of expert components.
// It does not own the device allocations and cannot free them. Finish on Linux
// records completion after the final consumer on its request stream.
type CUDASelection struct {
	owner    *CUDASelections
	lease    *Lease
	consumer cudaConsumer
	released bool
	failed   bool
}

// CUDAView is a bounded borrowed expert component, not a whole slab. Address
// refuses access after selection completion. Callers must not free the address
// or submit readers after Finish; retaining an address does not extend lifetime.
type CUDAView struct {
	selection      *CUDASelection
	address, bytes uint64
}

func cudaSelectionCache(source *Source, slots int, strides [3]int) (*Cache, error) {
	if source == nil || len(source.layers) == 0 || slots <= 0 {
		return nil, errors.New("invalid CUDA selection shape")
	}
	var bytes uint64
	for _, n := range strides {
		if n <= 0 || bytes > ^uint64(0)-uint64(n) {
			return nil, errors.New("CUDA stride overflow")
		}
		bytes += uint64(n)
	}
	for _, tr := range source.layers {
		for i, t := range []Tensor{tr.Gate, tr.Up, tr.Down} {
			if t.Stride != uint64(strides[i]) {
				return nil, errors.New("CUDA source stride differs")
			}
		}
	}
	if bytes == 0 || uint64(slots) > ^uint64(0)/bytes {
		return nil, errors.New("CUDA cache extent overflow")
	}
	return New(source.identity, slots, bytes, uint64(slots)*bytes)
}

// Address returns the live component's device address and exact byte extent.
func (v CUDAView) Address() (uint64, uint64, error) {
	if v.selection == nil || v.selection.owner == nil {
		return 0, 0, errors.New("invalid CUDA view")
	}
	o := v.selection.owner
	o.mu.Lock()
	defer o.mu.Unlock()
	if err := v.selection.check(); err != nil {
		return 0, 0, err
	}
	if v.selection.consumer != nil {
		return 0, 0, errors.New("CUDA consumer already recorded")
	}
	return v.address, v.bytes, nil
}

func (s *CUDASelection) check() error {
	if s == nil || s.owner == nil || s.lease == nil || s.released || s.failed || s.owner.active != s || !s.lease.committed {
		return errors.New("stale CUDA selection")
	}
	return s.lease.check()
}

func (s *CUDASelection) checkStream(checkDevice func() error) error {
	if s == nil || s.owner == nil {
		return errors.New("missing CUDA selection")
	}
	o := s.owner
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed || o.closing {
		return errors.New("CUDA selections closed")
	}
	if err := s.check(); err != nil {
		return err
	}
	if s.consumer != nil {
		return errors.New("CUDA consumer already recorded")
	}
	return checkDevice()
}

// Select uploads misses and returns one live committed selection. Concurrent
// requests are refused while another reader remains active; no slots are evicted.
// Graph capture is not supported. An owner returned with an error must be kept.
func (o *CUDASelections) Select(layer int, ids []int, graphCapture bool) (*CUDASelection, error) {
	if o == nil {
		return nil, errors.New("missing CUDA selections")
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.source == nil || o.cache == nil || o.upload == nil || o.view == nil || o.closing || o.closed || o.active != nil {
		return nil, errors.New("CUDA selections unavailable")
	}
	l, err := o.source.Select(o.cache, layer, ids, o.upload, graphCapture)
	if err != nil {
		if l != nil {
			o.active = &CUDASelection{owner: o, lease: l, failed: true}
			if l.committed {
				o.active.consumer = cudaUnexposedConsumer{}
			}
			o.closing = true
		}
		return o.active, err
	}
	s := &CUDASelection{owner: o, lease: l}
	o.active = s
	return s, nil
}

// No views were exposed when selection failed after upload commit. This is
// not an upload event being substituted for an actual device reader's event.
type cudaUnexposedConsumer struct{}

func (cudaUnexposedConsumer) Wait() error  { return nil }
func (cudaUnexposedConsumer) close() error { return nil }

// Views returns gate, up and down in the original requested route order,
// including duplicate IDs. Only a committed live selection exposes components.
func (s *CUDASelection) Views(route int) ([3]CUDAView, error) {
	var out [3]CUDAView
	if s == nil || s.owner == nil {
		return out, errors.New("missing CUDA selection")
	}
	o := s.owner
	o.mu.Lock()
	defer o.mu.Unlock()
	if err := s.check(); err != nil {
		return out, err
	}
	if s.consumer != nil {
		return out, errors.New("CUDA consumer already recorded")
	}
	if route < 0 || route >= len(s.lease.remap) {
		return out, errors.New("invalid CUDA route")
	}
	for component := range out {
		a, n, err := o.view(s.lease.remap[route], component)
		if err != nil || a == 0 || n == 0 || a > ^uint64(0)-n {
			return [3]CUDAView{}, errors.Join(errors.New("invalid CUDA component extent"), err)
		}
		out[component] = CUDAView{s, a, n}
	}
	return out, nil
}

func (s *CUDASelection) finish(newConsumer func() (cudaConsumer, error)) error {
	if s == nil || s.owner == nil {
		return errors.New("missing CUDA selection")
	}
	o := s.owner
	o.mu.Lock()
	defer o.mu.Unlock()
	if s.released && o.active == s && s.consumer != nil {
		if err := s.consumer.close(); err != nil {
			return err
		}
		o.active = nil
		return nil
	}
	if err := s.check(); err != nil {
		return err
	}
	if s.consumer == nil {
		c, err := newConsumer()
		s.consumer = c
		if err != nil {
			return err
		}
		if c == nil {
			return errors.New("missing CUDA consumer")
		}
	}
	if err := s.lease.Release(s.consumer); err != nil {
		return err
	}
	s.released = true
	if err := s.consumer.close(); err != nil {
		return err
	}
	o.active = nil
	return nil
}

// Close refuses a consumer without a recorded completion. Failed waits or
// frees retain owners for a later Close; no active device reader is freed.
func (o *CUDASelections) Close() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return nil
	}
	if o.upload == nil || o.cache == nil {
		return errors.New("invalid CUDA selections owner")
	}
	o.closing = true
	if s := o.active; s != nil {
		if s.lease.committed && !s.released {
			if s.consumer == nil {
				return errors.New("CUDA consumer completion required")
			}
			if err := s.lease.Release(s.consumer); err != nil {
				return err
			}
			s.released = true
		}
		if s.consumer != nil {
			if err := s.consumer.close(); err != nil {
				return err
			}
		}
		o.active = nil
	}
	if err := o.upload.Close(); err != nil {
		return err
	}
	if err := o.cache.Close(); err != nil {
		return err
	}
	o.closed = true
	return nil
}
