//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"runtime"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// The projection follows pinned ds4_cuda.cu:15887-15927. The full 24-row mix
// feeds both HC transforms (ds4.c:24675-24704,25468-25482). Output HC uses a
// separate four-row call (26281-26305); cuBLAS reduction depends on that shape.
//
//go:embed cuda_hc.cu
var cudaHCSource string

// cudaHC owns one sequential projection and its cached split. Its foundation
// and stream must outlive it. A second pre transform invalidates the old split.
type cudaHC struct {
	r                      *cudaFoundation
	arena                  *cudamem.Arena
	blas                   cublas.Handle
	normalized, mix, split cudamem.Slice[float32]
	half                   cudamem.Slice[uint16]
	residual               driver.DevicePtr
	ready                  bool
	closing, closeDrained  bool
	cleanupOps             *cudaHCCleanupOperations
}

func newCUDAHC(r *cudaFoundation) (_ *cudaHC, err error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if r == nil || r.device == nil {
		return nil, errors.New("cuda hc: missing device")
	}
	h := &cudaHC{r: r, arena: cudamem.NewArena(r.device)}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, h)
		}
	}()
	h.normalized = cudamem.AllocIn[float32](h.arena, "hc-normalized", 16384)
	h.half = cudamem.AllocIn[uint16](h.arena, "hc-half", 16384)
	h.mix = cudamem.AllocIn[float32](h.arena, "hc-mix", 24)
	h.split = cudamem.AllocIn[float32](h.arena, "hc-split", 24)
	if err := h.arena.Err(); err != nil {
		return nil, err
	}
	if err := r.withContext(func() error {
		var err error
		h.blas, err = cublas.Create()
		if err != nil {
			return err
		}
		if err := h.blas.SetMathMode(cublas.TF32TensorOpMath); err != nil {
			return err
		}
		return r.stream().WithHandle(func(s driver.Stream) error { return h.blas.SetStream(driver.Stream(s)) })
	}); err != nil {
		return nil, fmt.Errorf("cuda hc: create cuBLAS: %w", err)
	}
	return h, nil
}

func (h *cudaHC) Close() error {
	if h == nil {
		return nil
	}
	h.closing, h.ready = true, false
	ops := h.cleanupOperations()
	if h.r != nil && h.r.device != nil {
		if !h.closeDrained {
			if err := ops.synchronize(h); err != nil {
				return err
			}
			h.closeDrained = true
		}
		if h.blas != 0 {
			if err := ops.destroy(h, func() { h.blas = 0 }); err != nil {
				return err
			}
		}
	}
	if h.arena != nil {
		if err := ops.arena(h.arena); err != nil {
			return err
		}
		h.arena = nil
	}
	return nil
}

func (h *cudaHC) project(state cudamem.Slice[float32], fn cudaModelTensor, rows int) error {
	if h != nil && h.closing {
		return dsv4.ErrClosed
	}
	if h == nil || h.blas == 0 {
		return errors.New("cuda hc: closed")
	}
	h.ready = false
	if state.Len() != 16384 || fn.typeID != 1 || len(fn.dimensions) != 2 ||
		fn.dimensions[0] != 16384 || fn.dimensions[1] != uint64(rows) ||
		(rows != 4 && rows != 24) || fn.bytes < uint64(rows)*16384*2 {
		return errors.New("cuda hc: invalid projection shape")
	}
	if err := h.r.encodeRMS(state, cudaModelTensor{}, h.normalized, 1, false); err != nil {
		return err
	}
	return h.r.withContext(func() error {
		if err := h.r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(h.r.hcHalfFn).Grid(64, 1, 1).Block(256, 1, 1).On(h.r.stream()).Launch(h.normalized, h.half); err != nil {
			return fmt.Errorf("cuda hc: convert activation: %w", err)
		}
		// cuBLAS reads the scalars through the pointers while the call runs.
		// Converting to uintptr hides them from the compiler, so pin them:
		// an unpinned local is free to move and the call would read garbage.
		var pin runtime.Pinner
		defer pin.Unpin()
		alpha, beta := float32(1), float32(0)
		pin.Pin(&alpha)
		pin.Pin(&beta)
		err := h.blas.GemmEx(cublas.OpT, cublas.OpN, int32(rows), 1, 16384, cublas.Float32Scalar(alpha), driver.DevicePtr(fn.device), cublas.R16F, 16384, driver.DevicePtr(h.half.Ptr()), cublas.R16F, 16384, cublas.Float32Scalar(beta), driver.DevicePtr(h.mix.Ptr()), cublas.R32F, int32(rows), cublas.Compute32F, cublas.GemmDefault)
		if err != nil {
			return fmt.Errorf("cuda hc: project %d rows: %w", rows, err)
		}
		return nil
	})
}

func (h *cudaHC) pre(state cudamem.Slice[float32], fn, scale, base, norm cudaModelTensor, out cudamem.Slice[float32], head bool) error {
	if h != nil && h.closing {
		return dsv4.ErrClosed
	}
	rows, nscale := 24, 3
	if head {
		rows, nscale = 4, 1
	}
	if out.Len() != 8192 || scale.typeID != 0 || len(scale.dimensions) != 1 || scale.dimensions[0] != uint64(nscale) || scale.bytes < uint64(nscale)*4 ||
		base.typeID != 0 || len(base.dimensions) != 1 || base.dimensions[0] != uint64(rows) || base.bytes < uint64(rows)*4 ||
		norm.typeID != 0 || len(norm.dimensions) != 1 || norm.dimensions[0] != 4096 || norm.bytes < 4096*4 {
		if h != nil {
			h.ready = false
		}
		return errors.New("cuda hc: invalid transform profile")
	}
	if err := h.project(state, fn, rows); err != nil {
		return err
	}
	headArg := uint32(0)
	if head {
		headArg = 1
	}
	if err := h.r.withContext(func() error {
		stream := h.r.stream()
		if err := cudalaunch.Kernel(h.r.hcSplitFn).Grid(1, 1, 1).Block(32, 1, 1).On(stream).Launch(h.mix, scale.device, base.device, h.split, headArg); err != nil {
			return fmt.Errorf("cuda hc: split: %w", err)
		}
		return cudalaunch.Kernel(h.r.hcCollapseFn).Grid(16, 1, 1).Block(256, 1, 1).On(stream).Launch(state, h.split, out)
	}); err != nil {
		return err
	}
	if err := h.r.encodeRMS(out.Slice(0, 4096), norm, out.Slice(4096, 8192), 1, true); err != nil {
		return err
	}
	h.ready, h.residual = !head, state.Ptr()
	return nil
}

func (h *cudaHC) post(state, block, out cudamem.Slice[float32]) error {
	if h != nil && h.closing {
		return dsv4.ErrClosed
	}
	if h == nil || h.blas == 0 || !h.ready {
		return errors.New("cuda hc: missing cached layer projection")
	}
	h.ready = false
	if state.Len() != 16384 || state.Ptr() != h.residual || block.Len() != 4096 || out.Len() != 16384 {
		return errors.New("cuda hc: invalid residual or output")
	}
	return h.r.withContext(func() error {
		if err := cudalaunch.Kernel(h.r.hcExpandFn).Grid(64, 1, 1).Block(256, 1, 1).On(h.r.stream()).Launch(state, block, h.split, out); err != nil {
			return fmt.Errorf("cuda hc: expand: %w", err)
		}
		return nil
	})
}

// cudaHCBatch holds token-major scratch for a resident group. Projection keeps
// the scalar cuBLAS shape: changing GemmEx N can change its reduction order.
// A successful pre must be followed by post on the same state and row count.
type cudaHCBatch struct {
	h                                                *cudaHC
	capacity                                         int
	half                                             cudamem.Slice[uint16]
	mix, split, norm                                 cudamem.Slice[float32]
	rmsHalfFn, splitFn, collapseFn, expandFn, normFn driver.Function
	residual                                         driver.DevicePtr
	ready                                            int
	preCalls                                         uint64 // successful batched pre dispatches, for qualification reach checks
}

func newCUDAHCBatch(r *cudaFoundation, capacity int) (_ *cudaHCBatch, err error) {
	if capacity < 1 || capacity > cudaPrefillGroupWidth {
		return nil, errors.New("cuda hc batch: invalid capacity")
	}
	h, err := newCUDAHC(r)
	if err != nil {
		return nil, err
	}
	b := &cudaHCBatch{h: h, capacity: capacity}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, b)
		}
	}()
	b.half = cudamem.AllocIn[uint16](h.arena, "hc-batch-half", capacity*16384)
	b.mix = cudamem.AllocIn[float32](h.arena, "hc-batch-mix", capacity*24)
	b.split = cudamem.AllocIn[float32](h.arena, "hc-batch-split", capacity*24)
	b.norm = cudamem.AllocIn[float32](h.arena, "hc-batch-norm", capacity*4096)
	if err := h.arena.Err(); err != nil {
		return nil, err
	}
	if err := r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		for _, f := range []struct {
			name string
			dst  *driver.Function
		}{
			{"ds4_hc_rms_half_batch", &b.rmsHalfFn},
			{"ds4_hc_cached_split_batch", &b.splitFn},
			{"ds4_hc_cached_collapse_batch", &b.collapseFn},
			{"ds4_hc_cached_expand_batch", &b.expandFn},
			{"ds4_hc_norm_batch", &b.normFn},
		} {
			fn, err := r.matrixMod.Function(f.name)
			if err != nil {
				return fmt.Errorf("cuda hc batch: find %s: %w", f.name, err)
			}
			*f.dst = fn
		}
		return nil
	}); err != nil {
		return nil, err
	}
	return b, nil
}

func (b *cudaHCBatch) Close() error {
	if b == nil {
		return nil
	}
	b.ready = 0
	return b.h.Close()
}

// pre writes n interleaved [collapsed, normalized] rows of 8192 floats.
// State contains n adjacent rows of 16384 floats. Output-head HC stays scalar.
func (b *cudaHCBatch) pre(state cudamem.Slice[float32], fn, scale, base, norm cudaModelTensor, out cudamem.Slice[float32], n int) error {
	if b != nil && b.h != nil && b.h.closing {
		return dsv4.ErrClosed
	}
	if b == nil || b.h == nil || b.h.blas == 0 {
		return errors.New("cuda hc batch: closed")
	}
	b.ready = 0
	if n < 1 || n > b.capacity || state.Len() != n*16384 || out.Len() != n*8192 ||
		fn.typeID != 1 || fn.device == 0 || len(fn.dimensions) != 2 || fn.dimensions[0] != 16384 || fn.dimensions[1] != 24 || fn.bytes < 24*16384*2 ||
		scale.typeID != 0 || scale.device == 0 || len(scale.dimensions) != 1 || scale.dimensions[0] != 3 || scale.bytes < 3*4 ||
		base.typeID != 0 || base.device == 0 || len(base.dimensions) != 1 || base.dimensions[0] != 24 || base.bytes < 24*4 ||
		norm.typeID != 0 || norm.device == 0 || len(norm.dimensions) != 1 || norm.dimensions[0] != 4096 || norm.bytes < 4096*4 {
		return errors.New("cuda hc batch: invalid transform profile")
	}
	r := b.h.r
	if err := r.withContext(func() error {
		if err := cudalaunch.Kernel(b.rmsHalfFn).Grid(uint32(n), 1, 1).Block(1024, 1, 1).On(r.stream()).Launch(state, b.half, uint32(16384)); err != nil {
			return fmt.Errorf("cuda hc batch: normalize activation: %w", err)
		}
		var pin runtime.Pinner
		defer pin.Unpin()
		alpha, beta := float32(1), float32(0)
		pin.Pin(&alpha)
		pin.Pin(&beta)
		// One GEMM over the whole group, rather than the group's own count of
		// M=24,N=1,K=16384 calls. The activations are already token-major and
		// 16384 apart, and mix is token-major and 24 apart, so widening N is the
		// only change: column n of the result is what call n used to write.
		//
		// This is the launch the prefill spends its host time on. At a group of
		// 2048 and two pre calls per layer it was 176,128 dispatches of a GEMM
		// with one column each -- two thirds of every kernel the prefill issues,
		// and about a quarter of the wall clock spent issuing them.
		//
		// cuBLAS picks its algorithm from the shape, so a wide N does not sum
		// each column the way a single-column call did. DS4_CUDA_DISABLE_LIBRARY_GEMM=1
		// restores the per-token loop.
		project := func(rowLo, rows int) error {
			return b.h.blas.GemmEx(cublas.OpT, cublas.OpN, 24, int32(rows), 16384, cublas.Float32Scalar(alpha), driver.DevicePtr(fn.device), cublas.R16F, 16384, driver.DevicePtr(b.half.Slice(rowLo*16384, (rowLo+rows)*16384).Ptr()), cublas.R16F, 16384, cublas.Float32Scalar(beta), driver.DevicePtr(b.mix.Slice(rowLo*24, (rowLo+rows)*24).Ptr()), cublas.R32F, 24,
				cublas.Compute32F, cublas.GemmDefault)
		}
		if r.disableLibraryGEMM {
			for row := range n {
				if err := project(row, 1); err != nil {
					return fmt.Errorf("cuda hc batch: project row %d: %w", row, err)
				}
			}
		} else if err := project(0, n); err != nil {
			return fmt.Errorf("cuda hc batch: project %d rows: %w", n, err)
		}
		if err := cudalaunch.Kernel(b.splitFn).Grid(1, uint32(n), 1).Block(32, 1, 1).On(r.stream()).Launch(b.mix, scale.device, base.device, b.split, uint32(0)); err != nil {
			return fmt.Errorf("cuda hc batch: split: %w", err)
		}
		// Collapse and normalize in place in out's interleaved rows. The
		// batched compressor reads the normalized rows packed, from b.norm.
		if err := cudalaunch.Kernel(b.collapseFn).Grid(16, uint32(n), 1).Block(256, 1, 1).On(r.stream()).Launch(state, b.split, out, uint32(8192)); err != nil {
			return fmt.Errorf("cuda hc batch: collapse: %w", err)
		}
		if err := cudalaunch.Kernel(b.normFn).Grid(uint32(n), 1, 1).Block(1024, 1, 1).On(r.stream()).Launch(out, norm.device, b.norm); err != nil {
			return fmt.Errorf("cuda hc batch: normalize: %w", err)
		}
		return nil
	}); err != nil {
		return err
	}
	b.ready, b.residual = n, state.Ptr()
	b.preCalls++
	return nil
}

func (b *cudaHCBatch) post(state, block, out cudamem.Slice[float32], n int) error {
	if b != nil && b.h != nil && b.h.closing {
		return dsv4.ErrClosed
	}
	if b == nil || b.h == nil || b.h.blas == 0 || b.ready == 0 {
		return errors.New("cuda hc batch: missing cached layer projection")
	}
	ready := b.ready
	b.ready = 0
	if n != ready || state.Len() != n*16384 || state.Ptr() != b.residual || block.Len() != n*4096 || out.Len() != n*16384 {
		return errors.New("cuda hc batch: invalid residual or output")
	}
	return b.h.r.withContext(func() error {
		if err := cudalaunch.Kernel(b.expandFn).Grid(64, uint32(n), 1).Block(256, 1, 1).On(b.h.r.stream()).Launch(state, block, b.split, out); err != nil {
			return fmt.Errorf("cuda hc batch: expand: %w", err)
		}
		return nil
	})
}
