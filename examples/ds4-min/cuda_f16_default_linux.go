//go:build linux

package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaF16DefaultSource reproduces the binary16 activation boundary in the
// default single-token cuBLAS dispatch in pinned antirez/ds4 b0a147a,
// ds4_cuda.cu:15887-15927. Float64 accumulation avoids cancellation loss; it is not cuBLAS's implementation.
const cudaF16DefaultSource = `
extern "C" __global__ void ds4_f16_matvec_half_input(
    const unsigned short *weights, const float *x, float *out, unsigned cols) {
  const unsigned row = blockIdx.x, tid = threadIdx.x;
  const unsigned long long offset = (unsigned long long)row * cols;
  double sum = 0.0;
  for (unsigned i = tid; i < cols; i += blockDim.x)
    sum += double(ds4_half_to_float(weights[offset + i])) * double(ds4_half_round(x[i]));
  __shared__ double partial[256];
  partial[tid] = sum;
  __syncthreads();
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[row] = float(partial[0]);
}
extern "C" __global__ void ds4_f16_pair_ordered_chunks(
    const unsigned short *a, const unsigned short *b, const float *x,
    float *out_a, float *out_b, unsigned cols) {
  unsigned row = blockIdx.x, tid = threadIdx.x;
  unsigned long long offset = (unsigned long long)row * cols;
  unsigned long long chunk = ((unsigned long long)cols + 31) / 32;
  unsigned long long start = tid * chunk, end = start + chunk;
  if (end > cols) end = cols;
  float sum_a = 0.0f, sum_b = 0.0f;
  for (unsigned long long i = start; i < end; i++) {
    sum_a += ds4_half_to_float(a[offset + i]) * x[i];
    sum_b += ds4_half_to_float(b[offset + i]) * x[i];
  }
  __shared__ float partial_a[32], partial_b[32];
  partial_a[tid] = sum_a; partial_b[tid] = sum_b;
  __syncthreads();
  if (tid == 0) {
    float total_a = 0.0f, total_b = 0.0f;
    for (unsigned i = 0; i < 32; i++) { total_a += partial_a[i]; total_b += partial_b[i]; }
    out_a[row] = total_a; out_b[row] = total_b;
  }
}

`

// encodeF16MatvecDefault projects F16 weights after rounding input to F16,
// matching the default upstream cuBLAS operand types. It does not promise
// cuBLAS's accumulation order; the original raw-F32-input primitive is separate.
func (r *cudaFoundation) encodeF16MatvecDefault(t cudaModelTensor, in, out cudamem.Slice[float32]) error {
	if t.typeID != 1 || len(t.dimensions) != 2 || t.device == 0 {
		return fmt.Errorf("cuda: default F16 matvec %q: invalid tensor", t.name)
	}
	cols, rows := t.dimensions[0], t.dimensions[1]
	if cols == 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 || uint64(in.Len()) != cols || uint64(out.Len()) != rows || rows > t.bytes/(cols*2) {
		return fmt.Errorf("cuda: default F16 matvec %q: invalid shape or extent", t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.f16DefaultFn).Grid(uint32(rows)).Block(256).On(r.device.DefaultStream()).Launch(t.device, in, out, uint32(cols)); err != nil {
			return fmt.Errorf("cuda: launch default F16 matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeF16PairOrdered follows the raw-F32-input single-token pair dispatch in
// pinned ds4_cuda.cu:16194. Thirty-two contiguous partials are summed serially.
func (r *cudaFoundation) encodeF16PairOrdered(a, b cudaModelTensor, in, outA, outB cudamem.Slice[float32]) error {
	if a.typeID != 1 || b.typeID != 1 || len(a.dimensions) != 2 || len(b.dimensions) != 2 || a.device == 0 || b.device == 0 {
		return fmt.Errorf("cuda: ordered F16 pair: invalid tensor")
	}
	cols, rows := a.dimensions[0], a.dimensions[1]
	if cols == 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxInt32 || b.dimensions[0] != cols || b.dimensions[1] != rows || uint64(in.Len()) != cols || uint64(outA.Len()) != rows || outB.Len() != outA.Len() || rows > a.bytes/(cols*2) || rows > b.bytes/(cols*2) {
		return fmt.Errorf("cuda: ordered F16 pair: invalid shape or extent")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.f16PairOrderedFn).Grid(uint32(rows)).Block(32).On(r.device.DefaultStream()).Launch(a.device, b.device, in, outA, outB, uint32(cols)); err != nil {
			return fmt.Errorf("cuda: launch ordered F16 pair: %w", err)
		}
		return nil
	})
}
