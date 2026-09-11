//go:build linux

package main

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// cudaQ80Source follows antirez/ds4 b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd
// ds4_cuda.cu:5037 (activation quantization) and :5152 (single-token matvec).
// Activations retain float32 scales; GGUF weight scales remain binary16.
const cudaQ80Source = cudaQ80QuantSource + cudaQ80MatvecBody

const cudaQ80Layout = `
struct ds4_q8_activation { float d; signed char qs[32]; };
static_assert(sizeof(ds4_q8_activation) == 36, "Q8 activation block size");

`

const cudaQ80QuantSource = cudaQ80Layout + `
extern "C" __global__ void ds4_quant_q8_0(const float *in, ds4_q8_activation *out) {
  const unsigned tid = threadIdx.x;
  const float *x = in + (unsigned long long)blockIdx.x * 32;
  ds4_q8_activation *q = out + blockIdx.x;
  __shared__ float magnitude[32];
  magnitude[tid] = fabsf(x[tid]);
  __syncthreads();
  for (unsigned stride = 16; stride; stride >>= 1) {
    if (tid < stride) magnitude[tid] = fmaxf(magnitude[tid], magnitude[tid + stride]);
    __syncthreads();
  }
  const float d = magnitude[0] / 127.0f;
  const float inverse = d != 0.0f ? 1.0f / d : 0.0f;
  int value = (int)lrintf(x[tid] * inverse);
  value = value > 127 ? 127 : value < -128 ? -128 : value;
  q->qs[tid] = (signed char)value;
  if (tid == 0) q->d = d;
}

`

const cudaQ80MatvecBody = `
extern "C" __global__ void ds4_q8_0_matvec_quantized(
    const unsigned char *weights, const ds4_q8_activation *x, float *out,
    unsigned blocks) {
  const unsigned row = blockIdx.x, lane = threadIdx.x;
  const unsigned char *weight_row = weights + (unsigned long long)row * blocks * 34;
  float sum = 0.0f;
  for (unsigned b = lane; b < blocks; b += 32) {
    const unsigned char *p = weight_row + (unsigned long long)b * 34;
    const signed char *w = reinterpret_cast<const signed char *>(p + 2);
    int dot = 0;
    for (unsigned i = 0; i < 32; i++) dot += int(w[i]) * int(x[b].qs[i]);
    const float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
    sum += d * x[b].d * float(dot);
  }
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  if (lane == 0) out[row] = sum;
}

extern "C" __global__ void ds4_q8_0_matvec_quantized_blockdiag(
    const unsigned char *weights, const ds4_q8_activation *x, float *out,
    unsigned blocks, unsigned rows) {
  const unsigned row = blockIdx.x, diag = blockIdx.y, lane = threadIdx.x;
  const unsigned char *weight_row = weights + ((unsigned long long)diag * rows + row) * blocks * 34;
  const ds4_q8_activation *xb = x + (unsigned long long)diag * blocks;
  float sum = 0.0f;
  for (unsigned b = lane; b < blocks; b += 32) {
    const unsigned char *p = weight_row + (unsigned long long)b * 34;
    const signed char *w = reinterpret_cast<const signed char *>(p + 2);
    int dot = 0;
    for (unsigned i = 0; i < 32; i++) dot += int(w[i]) * int(xb[b].qs[i]);
    const float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
    sum += d * xb[b].d * float(dot);
  }
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  if (lane == 0) out[(unsigned long long)diag * rows + row] = sum;
}
`

// encodeQuantQ80 appends the pinned CUDA Q8_0 activation quantizer. Each
// 32-element block occupies 36 bytes: one float32 scale and 32 signed bytes.
// The isolated module uses the pinned NVCC fast-math quantizer semantics; the
// matvec and other model kernels retain their existing compiler options.
func (r *cudaFoundation) encodeQuantQ80(in cudamem.Slice[float32], out cudamem.Slice[byte], cols int) error {
	if cols <= 0 || cols%32 != 0 || in.Len() == 0 || in.Len()%cols != 0 || uint64(in.Len()/32) > math.MaxInt32 || uint64(out.Len()) != uint64(in.Len()/32)*36 {
		return fmt.Errorf("cuda: Q8_0 activation quantization: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadQ80Quantizer(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.quantQ80Fn).Grid(uint32(in.Len()/32)).Block(32).On(r.device.DefaultStream()).Launch(in, out); err != nil {
			return fmt.Errorf("cuda: launch Q8_0 activation quantization: %w", err)
		}
		return nil
	})
}

// encodeQ80MatvecQuantized projects Q8_0 weights using quantized activations.
func (r *cudaFoundation) encodeQ80MatvecQuantized(t cudaModelTensor, in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.device == 0 {
		return fmt.Errorf("cuda: quantized Q8_0 matvec %q: invalid tensor", t.name)
	}
	cols, rows := t.dimensions[0], t.dimensions[1]
	if cols == 0 || cols%32 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 || uint64(in.Len()) != cols/32*36 || uint64(out.Len()) != rows || rows > t.bytes/(cols/32*34) {
		return fmt.Errorf("cuda: quantized Q8_0 matvec %q: invalid shape or extent", t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.q80QuantizedFn).Grid(uint32(rows)).Block(32).On(r.device.DefaultStream()).Launch(t.device, in, out, uint32(cols/32)); err != nil {
			return fmt.Errorf("cuda: launch quantized Q8_0 matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeQ80MatvecQuantizedBlockDiag projects block-diagonal Q8_0 weights using quantized activations.
func (r *cudaFoundation) encodeQ80MatvecQuantizedBlockDiag(t cudaModelTensor, in cudamem.Slice[byte], out cudamem.Slice[float32], diag int) error {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.device == 0 || diag <= 0 {
		return fmt.Errorf("cuda: quantized Q8_0 block-diagonal matvec %q: invalid tensor", t.name)
	}
	cols, totalRows := t.dimensions[0], t.dimensions[1]
	if totalRows%uint64(diag) != 0 {
		return fmt.Errorf("cuda: quantized Q8_0 block-diagonal matvec %q: total rows not divisible by diag", t.name)
	}
	rows := totalRows / uint64(diag)
	if cols == 0 || cols%32 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 ||
		uint64(in.Len()) != (cols/32*36)*uint64(diag) || uint64(out.Len()) != totalRows || totalRows > t.bytes/(cols/32*34) {
		return fmt.Errorf("cuda: quantized Q8_0 block-diagonal matvec %q: invalid shape or extent", t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.q80QuantizedBlockDiagFn).Grid(uint32(rows), uint32(diag)).Block(32).On(r.device.DefaultStream()).Launch(t.device, in, out, uint32(cols/32), uint32(rows)); err != nil {
			return fmt.Errorf("cuda: launch quantized Q8_0 block-diagonal matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// loadQ80Quantizer is called with the device context current. The module is
// owned immediately after load, but is usable only after function lookup. A
// failed cleanup retains the module for a later lookup or Close attempt.
func (r *cudaFoundation) loadQ80Quantizer() error {
	if r.quantQ80Mod != 0 && r.quantQ80Fn != 0 {
		return nil
	}
	if r.quantQ80Mod == 0 {
		compiled, err := nvrtcx.Compile(cudaQ80QuantSource, nvrtcx.WithName("ds4_quant_q8_0.cu"), nvrtcx.WithArch(r.major, r.minor), nvrtcx.WithFlag("--use_fast_math"))
		if err != nil {
			return fmt.Errorf("cuda: compile Q8_0 quantizer: %w", err)
		}
		mod, err := compiled.Load(r.device)
		if err != nil {
			return fmt.Errorf("cuda: load Q8_0 quantizer: %w", err)
		}
		r.quantQ80Mod = mod
	}
	fn, err := r.quantQ80Mod.GetFunction("ds4_quant_q8_0")
	if err != nil {
		unloadErr := r.quantQ80Mod.Unload()
		if unloadErr == nil {
			r.quantQ80Mod, r.quantQ80Fn = 0, 0
		}
		return errors.Join(fmt.Errorf("cuda: find Q8_0 quantizer: %w", err), unloadErr)
	}
	r.quantQ80Fn = fn
	return nil
}
