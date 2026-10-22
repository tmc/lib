//go:build linux

package main

import (
	"context"

	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type cudaChildCleanupOperations struct {
	synchronize func(*cudaRuntime) error
	graph       func(*cudagraph.Exec) error
	bytes       func(*cudamem.Slice[byte]) error
	arena       func(*cudamem.Arena) error
	grouping    func(*cudaMoEGrouping) error
	hc          func(*cudaHC) error
	hcBatch     func(*cudaHCBatch) error
	batch       func(*cudaBatchResources) error
	foundation  func(*cudaFoundation) error
}

func defaultCUDAChildCleanupOperations() cudaChildCleanupOperations {
	return cudaChildCleanupOperations{
		synchronize: func(r *cudaRuntime) error {
			if r.borrowedFoundation {
				if r.requestOps != nil {
					return r.requestOps.synchronize(r)
				}
				if r.workStream == nil {
					return nil
				}
				return syncStream(r.workStream)
			}
			return syncStream(r.stream())
		},
		graph:      func(g *cudagraph.Exec) error { return g.Destroy() },
		bytes:      func(s *cudamem.Slice[byte]) error { return s.Free() },
		arena:      func(a *cudamem.Arena) error { return a.Free() },
		grouping:   func(g *cudaMoEGrouping) error { return g.Free() },
		hc:         func(h *cudaHC) error { return h.Close() },
		hcBatch:    func(h *cudaHCBatch) error { return h.Close() },
		batch:      func(b *cudaBatchResources) error { return b.Close() },
		foundation: func(f *cudaFoundation) error { return f.Close() },
	}
}

func (r *cudaRuntime) cleanupOperations() cudaChildCleanupOperations {
	if r.cleanupOps != nil {
		return *r.cleanupOps
	}
	return defaultCUDAChildCleanupOperations()
}

func (b *cudaBatchResources) cleanupOperations() cudaChildCleanupOperations {
	if b.cleanupOps != nil {
		return *b.cleanupOps
	}
	return defaultCUDAChildCleanupOperations()
}

func freeCUDAByteSlices(slices []*cudamem.Slice[byte], release func(*cudamem.Slice[byte]) error) error {
	for _, s := range slices {
		if s.Len() == 0 {
			continue
		}
		if err := release(s); err != nil {
			return err
		}
		*s = cudamem.Slice[byte]{}
	}
	return nil
}

type cudaHCCleanupOperations struct {
	synchronize func(*cudaHC) error
	destroy     func(*cudaHC, func()) error
	arena       func(*cudamem.Arena) error
}

func defaultCUDAHCCleanupOperations() cudaHCCleanupOperations {
	return cudaHCCleanupOperations{
		synchronize: func(h *cudaHC) error { return h.r.stream().Synchronize(context.Background()) },
		destroy: func(h *cudaHC, released func()) error {
			return h.r.device.WithContext(func() error {
				if err := h.blas.Close(); err != nil {
					return err
				}
				released()
				return nil
			})
		},
		arena: func(a *cudamem.Arena) error { return a.Free() },
	}
}

func (h *cudaHC) cleanupOperations() cudaHCCleanupOperations {
	if h.cleanupOps != nil {
		return *h.cleanupOps
	}
	return defaultCUDAHCCleanupOperations()
}
