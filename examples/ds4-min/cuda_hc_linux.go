//go:build linux

package main

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// The projection follows pinned ds4_cuda.cu:15887-15927. The full 24-row mix
// feeds both HC transforms (ds4.c:24675-24704,25468-25482). Output HC uses a
// separate four-row call (26281-26305); cuBLAS reduction depends on that shape.
const cudaHCSource = `
extern "C" __global__ void ds4_hc_half_input(const float *in, unsigned short *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 16384) out[i] = ds4_q81_half_bits(in[i]);
}
extern "C" __global__ void ds4_hc_cached_split(
    const float *mix, const float *scale, const float *base, float *split, unsigned head) {
  if (threadIdx.x || blockIdx.x) return;
  for (unsigned h = 0; h < 4; h++)
    split[h] = 1.0f / (1.0f + expf(-(mix[h] * scale[0] + base[h]))) + 1.0e-6f;
  if (!head) ds4_hc_split4(mix, scale, base, split + 4, split + 8);
}
extern "C" __global__ void ds4_hc_cached_collapse(
    const float *state, const float *split, float *out) {
  unsigned d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= 4096) return;
  float v = 0.0f;
  for (unsigned h = 0; h < 4; h++) v += state[h * 4096 + d] * split[h];
  out[d] = v;
}
extern "C" __global__ void ds4_hc_cached_expand(
    const float *state, const float *block, const float *split, float *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 16384) return;
  unsigned dst = i / 4096, d = i % 4096;
  float v = block[d] * split[4 + dst];
  for (unsigned src = 0; src < 4; src++)
    v += split[8 + dst + src * 4] * state[src * 4096 + d];
  out[i] = v;
}
`

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
}

func newCUDAHC(r *cudaFoundation) (_ *cudaHC, err error) {
	if r == nil || r.device == nil {
		return nil, errors.New("cuda hc: missing device")
	}
	h := &cudaHC{r: r, arena: cudamem.NewArena(r.device)}
	defer func() {
		if err != nil {
			err = errors.Join(err, h.Close())
		}
	}()
	h.normalized = cudamem.AllocIn[float32](h.arena, "hc-normalized", 16384)
	h.half = cudamem.AllocIn[uint16](h.arena, "hc-half", 16384)
	h.mix = cudamem.AllocIn[float32](h.arena, "hc-mix", 24)
	h.split = cudamem.AllocIn[float32](h.arena, "hc-split", 24)
	if err := h.arena.Err(); err != nil {
		return nil, err
	}
	if err := r.device.WithContext(func() error {
		var err error
		h.blas, err = cublas.Create()
		if err != nil {
			return err
		}
		if err := h.blas.SetMathMode(cublas.TF32TensorOpMath); err != nil {
			return err
		}
		return r.device.DefaultStream().WithHandle(func(s driver.Stream) error { return h.blas.SetStream(cublas.Stream(s)) })
	}); err != nil {
		return nil, fmt.Errorf("cuda hc: create cuBLAS: %w", err)
	}
	return h, nil
}

func (h *cudaHC) Close() error {
	if h == nil {
		return nil
	}
	var errs []error
	if h.r != nil && h.r.device != nil {
		errs = append(errs, h.r.device.DefaultStream().Synchronize(context.Background()))
		if h.blas != 0 {
			errs = append(errs, h.r.device.WithContext(h.blas.Destroy))
			h.blas = 0
		}
	}
	if h.arena != nil {
		errs = append(errs, h.arena.Free())
		h.arena = nil
	}
	h.ready = false
	return errors.Join(errs...)
}

func (h *cudaHC) project(state cudamem.Slice[float32], fn cudaModelTensor, rows int) error {
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
	return h.r.device.WithContext(func() error {
		if err := h.r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(h.r.hcHalfFn).Grid(64).Block(256).On(h.r.device.DefaultStream()).Launch(h.normalized, h.half); err != nil {
			return fmt.Errorf("cuda hc: convert activation: %w", err)
		}
		alpha, beta := float32(1), float32(0)
		err := h.blas.GemmEx(cublas.OpT, cublas.OpN, int32(rows), 1, 16384,
			uintptr(unsafe.Pointer(&alpha)), uintptr(fn.device), cublas.R16F, 16384,
			uintptr(h.half.Ptr()), cublas.R16F, 16384,
			uintptr(unsafe.Pointer(&beta)), uintptr(h.mix.Ptr()), cublas.R32F, int32(rows), cublas.Compute32F, cublas.GemmDefault)
		runtime.KeepAlive(&alpha)
		runtime.KeepAlive(&beta)
		if err != nil {
			return fmt.Errorf("cuda hc: project %d rows: %w", rows, err)
		}
		return nil
	})
}

func (h *cudaHC) pre(state cudamem.Slice[float32], fn, scale, base, norm cudaModelTensor, out cudamem.Slice[float32], head bool) error {
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
	if err := h.r.device.WithContext(func() error {
		stream := h.r.device.DefaultStream()
		if err := cudalaunch.Kernel(h.r.hcSplitFn).Grid(1).Block(1).On(stream).Launch(h.mix, scale.device, base.device, h.split, headArg); err != nil {
			return fmt.Errorf("cuda hc: split: %w", err)
		}
		return cudalaunch.Kernel(h.r.hcCollapseFn).Grid(16).Block(256).On(stream).Launch(state, h.split, out)
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
	if h == nil || h.blas == 0 || !h.ready {
		return errors.New("cuda hc: missing cached layer projection")
	}
	h.ready = false
	if state.Len() != 16384 || state.Ptr() != h.residual || block.Len() != 4096 || out.Len() != 16384 {
		return errors.New("cuda hc: invalid residual or output")
	}
	return h.r.device.WithContext(func() error {
		if err := cudalaunch.Kernel(h.r.hcExpandFn).Grid(64).Block(256).On(h.r.device.DefaultStream()).Launch(state, block, h.split, out); err != nil {
			return fmt.Errorf("cuda hc: expand: %w", err)
		}
		return nil
	})
}
