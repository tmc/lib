//go:build linux

package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type cudaShapeCache struct {
	r       *cudaFoundation
	plan    shapeCachePlan
	buffers []cudamem.Slice[float32]
	ops     shapeCacheOperations
	closed  bool
}

type shapeCacheOperations struct {
	available  func() (uint64, error)
	alloc      func(int) (cudamem.Slice[float32], error)
	initialize func(cudamem.Slice[float32], uint32) error
	drain      func() error
	free       func(*cudamem.Slice[float32]) error
}

func newCUDAShapeCache(r *cudaFoundation, c dsv4shape.Config, context, reserve uint64) (*cudaShapeCache, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if r.device == nil {
		return nil, errors.New("cuda shape cache: missing device")
	}
	ops := shapeCacheOperations{
		available: func() (uint64, error) {
			var free uint
			err := r.withContext(func() error { var err error; free, _, err = driver.MemGetInfo(); return err })
			return uint64(free), err
		},
		alloc: func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](r.device, n) },
		initialize: func(v cudamem.Slice[float32], bits uint32) error {
			return r.withContext(func() error {
				return r.stream().WithHandle(func(h driver.Stream) error { return v.Ptr().MemsetD32Async(bits, uint(v.Len()), h) })
			})
		},
		drain: func() error { return r.device.WithContext(func() error { return syncStream(r.stream()) }) },
		free:  func(v *cudamem.Slice[float32]) error { return v.Free() },
	}
	return newCUDAShapeCacheWithOperations(r, c, context, reserve, ops)
}

func newCUDAShapeCacheWithOperations(r *cudaFoundation, c dsv4shape.Config, context, reserve uint64, ops shapeCacheOperations) (*cudaShapeCache, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	available, err := ops.available()
	if err != nil {
		return nil, err
	}
	plan, err := planShapeCache(c, context, available, reserve)
	if err != nil {
		return nil, err
	}
	s := &cudaShapeCache{r: r, plan: plan, ops: ops, buffers: make([]cudamem.Slice[float32], len(plan.buffers))}
	fail := func(err error) (*cudaShapeCache, error) { return nil, finishBackendCleanup(err, s) }
	for i, p := range plan.buffers {
		s.buffers[i], err = ops.alloc(p.elements)
		if err != nil {
			return fail(fmt.Errorf("cuda shape cache: allocate layer %d %s: %w", p.layer, p.name, err))
		}
		if s.buffers[i].Ptr() == 0 || s.buffers[i].Len() != p.elements {
			return fail(fmt.Errorf("cuda shape cache: invalid allocation layer %d %s", p.layer, p.name))
		}
		if err := ops.initialize(s.buffers[i], p.initial); err != nil {
			return fail(fmt.Errorf("cuda shape cache: initialize layer %d %s: %w", p.layer, p.name, err))
		}
	}
	if err := ops.drain(); err != nil {
		return fail(err)
	}
	return s, nil
}

func (s *cudaShapeCache) Close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	owned := false
	for _, v := range s.buffers {
		owned = owned || v.Ptr() != 0
	}
	if !owned {
		return nil
	}
	if err := s.ops.drain(); err != nil {
		return err
	}
	var errs []error
	for i := range s.buffers {
		v := &s.buffers[i]
		if v.Ptr() == 0 {
			continue
		}
		if err := s.ops.free(v); err != nil {
			errs = append(errs, err)
		} else {
			*v = cudamem.Slice[float32]{}
		}
	}
	return errors.Join(errs...)
}
