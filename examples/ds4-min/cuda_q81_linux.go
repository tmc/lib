//go:build linux

package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaQ81Source follows antirez/ds4 b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd
// cuda/mmq/quantize.cu:5 and cuda/mmq/ds4_mmq.cu:3950.
// Activation scales and block sums are binary16; the dot does not consume sums.
const cudaQ81Source = `
struct ds4_q8_1_activation { unsigned short d, sum; signed char qs[32]; };
static_assert(sizeof(ds4_q8_1_activation) == 36, "Q8_1 block size");

// ds4_half_round already implements the binary16 nearest-even boundary. Encode
// its rounded float result without another rounding step or toolkit headers.
__device__ unsigned short ds4_q81_half_bits(float value) {
  unsigned bits = __float_as_uint(ds4_half_round(value));
  unsigned sign = (bits >> 16) & 0x8000, mag = bits & 0x7fffffff;
  if (mag >= 0x7f800000) return (unsigned short)(sign | 0x7c00 | ((mag & 0x7fffff) ? 0x200 : 0));
  if (mag == 0) return (unsigned short)sign;
  int exponent = int(mag >> 23) - 127;
  unsigned mantissa = mag & 0x7fffff;
  if (exponent >= -14) return (unsigned short)(sign | unsigned(exponent + 15) << 10 | mantissa >> 13);
  return (unsigned short)(sign | ((mantissa | 0x800000) >> (-exponent - 1)));
}

extern "C" __global__ void ds4_quant_q8_1(const float *in, ds4_q8_1_activation *out) {
  const unsigned tid = threadIdx.x;
  float value = in[(unsigned long long)blockIdx.x * 32 + tid];
  float magnitude = fabsf(value), sum = value;
  for (unsigned stride = 16; stride; stride >>= 1) {
    magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffff, magnitude, stride));
    sum += __shfl_xor_sync(0xffffffff, sum, stride);
  }
  float d = magnitude / 127.0f;
  out[blockIdx.x].qs[tid] = magnitude == 0.0f ? 0 : (signed char)roundf(value / d);
  if (tid == 0) { out[blockIdx.x].d = ds4_q81_half_bits(d); out[blockIdx.x].sum = ds4_q81_half_bits(sum); }
}

extern "C" __global__ void ds4_q8_0_matvec_q8_1(
    const unsigned char *weights, const ds4_q8_1_activation *x, float *out,
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
    sum += d * ds4_half_to_float(x[b].d) * float(dot);
  }
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  if (lane == 0) out[row] = sum;
}
`

// encodeQuantQ81 appends the pinned CUDA Q8_1 activation quantizer. Each
// 32-element block occupies 36 bytes: two binary16 values and 32 signed bytes.
func (r *cudaFoundation) encodeQuantQ81(in cudamem.Slice[float32], out cudamem.Slice[byte], cols int) error {
	if cols <= 0 || cols%32 != 0 || in.Len() == 0 || in.Len()%cols != 0 || uint64(in.Len()/32) > math.MaxInt32 || uint64(out.Len()) != uint64(in.Len()/32)*36 {
		return fmt.Errorf("cuda: Q8_1 activation quantization: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.quantQ81Fn).Grid(uint32(in.Len()/32)).Block(32).On(r.device.DefaultStream()).Launch(in, out); err != nil {
			return fmt.Errorf("cuda: launch Q8_1 activation quantization: %w", err)
		}
		return nil
	})
}

// encodeQ80MatvecQ81 projects Q8_0 weights using quantized activations.
func (r *cudaFoundation) encodeQ80MatvecQ81(t cudaModelTensor, in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
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
		if err := cudalaunch.Kernel(r.q81MatvecFn).Grid(uint32(rows)).Block(32).On(r.device.DefaultStream()).Launch(t.device, in, out, uint32(cols/32)); err != nil {
			return fmt.Errorf("cuda: launch quantized Q8_0 matvec %q: %w", t.name, err)
		}
		return nil
	})
}
