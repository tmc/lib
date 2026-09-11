//go:build linux

package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaExpertSource implements F32 selected-expert projections. Its codebooks
// are from antirez/ds4 b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd,
// metal/moe.metal; block layouts match lm/quant's pinned C oracle fixtures.
const cudaExpertSource = `
__device__ const unsigned char ds4_cuda_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

__device__ const unsigned long long ds4_cuda_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};
__device__ float ds4_iq2_block_f32(const unsigned char *p, const float *x) {
  float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
  float total = 0.0f;
  for (unsigned group = 0; group < 8; group++) {
    const unsigned char *q = p + 2 + group * 8;
    unsigned aux = unsigned(q[4]) | unsigned(q[5]) << 8 | unsigned(q[6]) << 16 | unsigned(q[7]) << 24;
    float scale = d * (0.5f + float(aux >> 28)) * 0.25f;
    for (unsigned lane = 0; lane < 4; lane++) {
      unsigned long long grid = ds4_cuda_iq2xxs_grid[q[lane]];
      unsigned char signs = ds4_cuda_ksigns_iq2xs[(aux >> (7 * lane)) & 127];
      for (unsigned i = 0; i < 8; i++) {
        float w = float((grid >> (8 * i)) & 255);
        if (signs & (1u << i)) w = -w;
        total += scale * w * x[group * 32 + lane * 8 + i];
      }
    }
  }
  return total;
}

extern "C" __global__ void ds4_iq2_xxs_pair_selected_f32(
    const unsigned char *gate, const unsigned char *up, const int *ids,
    const float *x, float *gate_out, float *up_out,
    unsigned cols, unsigned rows, unsigned experts) {
  unsigned row = blockIdx.x % rows, slot = blockIdx.x / rows;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (threadIdx.x == 0) {
      gate_out[blockIdx.x] = __int_as_float(0x7fffffff);
      up_out[blockIdx.x] = __int_as_float(0x7fffffff);
    }
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long offset = (static_cast<unsigned long long>(expert) * rows + row) * blocks * 66;
  float g = 0.0f, u = 0.0f;
  for (unsigned b = threadIdx.x; b < blocks; b += blockDim.x) {
    g += ds4_iq2_block_f32(gate + offset + (unsigned long long)b * 66, x + b * 256);
    u += ds4_iq2_block_f32(up + offset + (unsigned long long)b * 66, x + b * 256);
  }
  __shared__ float gs[256], us[256];
  gs[threadIdx.x] = g; us[threadIdx.x] = u;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) { gs[threadIdx.x] += gs[threadIdx.x + stride]; us[threadIdx.x] += us[threadIdx.x + stride]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { gate_out[blockIdx.x] = gs[0]; up_out[blockIdx.x] = us[0]; }
}

extern "C" __global__ void ds4_q2_k_selected_f32(
    const unsigned char *weights, const int *ids, const float *x, float *out,
    unsigned cols, unsigned rows, unsigned experts) {
  unsigned row = blockIdx.x % rows, slot = blockIdx.x / rows;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (threadIdx.x == 0) out[blockIdx.x] = __int_as_float(0x7fffffff);
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long offset = (static_cast<unsigned long long>(expert) * rows + row) * blocks * 84;
  const float *input = x + (unsigned long long)slot * cols;
  float sum = 0.0f;
  for (unsigned b = threadIdx.x; b < blocks; b += blockDim.x) {
    const unsigned char *p = weights + offset + (unsigned long long)b * 84;
    float d = ds4_half_to_float(unsigned(p[80]) | unsigned(p[81]) << 8);
    float dm = ds4_half_to_float(unsigned(p[82]) | unsigned(p[83]) << 8);
    float block_sum = 0.0f;
    for (unsigned i = 0; i < 256; i++) {
      unsigned group = i / 16, qbase = 32 * (group / 8) + 16 * (group & 1), shift = ((group / 2) & 3) * 2;
      unsigned sm = p[group];
      float w = d * float(sm & 15) * float((p[16 + qbase + i % 16] >> shift) & 3) - dm * float(sm >> 4);
      block_sum += w * input[b * 256 + i];
    }
    sum += block_sum;
  }
  __shared__ float partial[256]; partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = partial[0];
}
struct ds4_q8_k {
  float d;
  signed char qs[256];
  short bsums[16];
};
static_assert(sizeof(ds4_q8_k) == 292, "Q8_K block size");

// Pinned ds4_cuda.cu:19883: signed maximum, nearest-even rounding, then
// sixteen group sums. Tie winners follow the 256-lane reduction tree.
extern "C" __global__ void ds4_quant_q8_k(const float *in, ds4_q8_k *out) {
  unsigned tid = threadIdx.x;
  const float *x = in + (unsigned long long)blockIdx.x * 256;
  ds4_q8_k *q = out + blockIdx.x;
  __shared__ float magnitudes[256], values[256], inverse;
  float v = x[tid];
  magnitudes[tid] = fabsf(v); values[tid] = v;
  __syncthreads();
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (tid < stride && magnitudes[tid + stride] > magnitudes[tid]) {
      magnitudes[tid] = magnitudes[tid + stride];
      values[tid] = values[tid + stride];
    }
    __syncthreads();
  }
  if (magnitudes[0] == 0.0f) {
    if (tid == 0) q->d = 0.0f;
    q->qs[tid] = 0;
    if (tid < 16) q->bsums[tid] = 0;
    return;
  }
  if (tid == 0) inverse = -127.0f / values[0];
  __syncthreads();
  int value = (int)lrintf(inverse * x[tid]);
  value = value > 127 ? 127 : value < -128 ? -128 : value;
  q->qs[tid] = (signed char)value;
  __syncthreads();
  if (tid < 16) {
    int sum = 0;
    for (unsigned j = 0; j < 16; j++) sum += q->qs[tid * 16 + j];
    q->bsums[tid] = (short)sum;
  }
  if (tid == 0) q->d = 1.0f / inverse;
}

__device__ float ds4_iq2_block_q8_k(const unsigned char *p, const ds4_q8_k *x) {
  int total = 0;
  for (unsigned group = 0; group < 8; group++) {
    const unsigned char *q = p + 2 + group * 8;
    unsigned aux = unsigned(q[4]) | unsigned(q[5]) << 8 | unsigned(q[6]) << 16 | unsigned(q[7]) << 24;
    int sum = 0;
    for (unsigned lane = 0; lane < 4; lane++) {
      unsigned long long grid = ds4_cuda_iq2xxs_grid[q[lane]];
      unsigned char signs = ds4_cuda_ksigns_iq2xs[(aux >> (7 * lane)) & 127];
      for (unsigned i = 0; i < 8; i++) {
        int w = int((grid >> (8 * i)) & 255);
        if (signs & (1u << i)) w = -w;
        sum += w * int(x->qs[group * 32 + lane * 8 + i]);
      }
    }
    total += sum * int(2 * (aux >> 28) + 1);
  }
  float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8) * x->d;
  return 0.125f * d * float(total);
}

__device__ float ds4_q2_block_q8_k(const unsigned char *p, const ds4_q8_k *x) {
  int total = 0, minimum = 0;
  for (unsigned group = 0; group < 16; group++) {
    unsigned sm = p[group], qbase = 32 * (group / 8) + 16 * (group & 1), shift = ((group / 2) & 3) * 2;
    int sum = 0;
    for (unsigned i = 0; i < 16; i++) sum += int((p[16 + qbase + i] >> shift) & 3) * int(x->qs[group * 16 + i]);
    total += int(sm & 15) * sum;
    minimum += int(sm >> 4) * int(x->bsums[group]);
  }
  float d = x->d * ds4_half_to_float(unsigned(p[80]) | unsigned(p[81]) << 8);
  float dm = x->d * ds4_half_to_float(unsigned(p[82]) | unsigned(p[83]) << 8);
  return d * float(total) - dm * float(minimum);
}
extern "C" __global__ void ds4_iq2_xxs_pair_selected_q8_k(
    const unsigned char *gate, const unsigned char *up, const int *ids,
    const ds4_q8_k *x, float *gate_out, float *up_out,
    unsigned cols, unsigned rows, unsigned experts) {
  unsigned row = blockIdx.x % rows, slot = blockIdx.x / rows;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (threadIdx.x == 0) {
      gate_out[blockIdx.x] = __int_as_float(0x7fffffff);
      up_out[blockIdx.x] = __int_as_float(0x7fffffff);
    }
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long offset = (static_cast<unsigned long long>(expert) * rows + row) * blocks * 66;
  float g = 0.0f, u = 0.0f;
  for (unsigned b = threadIdx.x; b < blocks; b += blockDim.x) {
    g += ds4_iq2_block_q8_k(gate + offset + (unsigned long long)b * 66, x + b);
    u += ds4_iq2_block_q8_k(up + offset + (unsigned long long)b * 66, x + b);
  }
  __shared__ float gs[256], us[256];
  gs[threadIdx.x] = g; us[threadIdx.x] = u;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) { gs[threadIdx.x] += gs[threadIdx.x + stride]; us[threadIdx.x] += us[threadIdx.x + stride]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { gate_out[blockIdx.x] = gs[0]; up_out[blockIdx.x] = us[0]; }
}

extern "C" __global__ void ds4_q2_k_selected_q8_k(
    const unsigned char *weights, const int *ids, const ds4_q8_k *x, float *out,
    unsigned cols, unsigned rows, unsigned experts) {
  unsigned row = blockIdx.x % rows, slot = blockIdx.x / rows;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (threadIdx.x == 0) out[blockIdx.x] = __int_as_float(0x7fffffff);
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long offset = (static_cast<unsigned long long>(expert) * rows + row) * blocks * 84;
  const ds4_q8_k *input = x + (unsigned long long)slot * blocks;
  float sum = 0.0f;
  for (unsigned b = threadIdx.x; b < blocks; b += blockDim.x) {
    const unsigned char *p = weights + offset + (unsigned long long)b * 84;
    sum += ds4_q2_block_q8_k(p, input + b);
  }
  __shared__ float partial[256]; partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = partial[0];
}
`

// cudaExpertProfile checks both dimensions and the available weight extent.
func cudaExpertProfile(t cudaModelTensor, typeID uint32, blockBytes uint64) error {
	if t.typeID != typeID || len(t.dimensions) != 3 || t.device == 0 {
		return fmt.Errorf("cuda: expert tensor %q: invalid type or rank", t.name)
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	if cols == 0 || cols%256 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxUint32 || experts == 0 || experts > math.MaxInt32 {
		return fmt.Errorf("cuda: expert tensor %q: invalid dimensions", t.name)
	}
	rowBytes := cols / 256 * blockBytes
	if rows > t.bytes/rowBytes || experts > t.bytes/(rowBytes*rows) {
		return fmt.Errorf("cuda: expert tensor %q: truncated weights", t.name)
	}
	return nil
}

// encodeMatvecIQ2Selected appends paired F32-input IQ2_XXS projections.
// Outputs are slot-major. Invalid device IDs produce non-finite outputs.
func (r *cudaFoundation) encodeMatvecIQ2Selected(gate, up cudaModelTensor, ids cudamem.Slice[int32], in, gateOut, upOut cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertProfile(up, 16, 66); err != nil {
		return err
	}
	for i := range gate.dimensions {
		if gate.dimensions[i] != up.dimensions[i] {
			return fmt.Errorf("cuda: IQ2 selected pair: mismatched dimensions")
		}
	}
	cols, rows, experts := gate.dimensions[0], gate.dimensions[1], gate.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols || uint64(gateOut.Len()) != slots*rows || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("cuda: IQ2 selected pair: invalid input or output shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.iq2SelectedFn).Grid(uint32(rows*slots)).Block(256).On(r.device.DefaultStream()).Launch(gate.device, up.device, ids, in, gateOut, upOut, uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch IQ2 selected pair: %w", err)
		}
		return nil
	})
}

// encodeMatvecQ2KSelected appends Q2_K projections for GPU-selected experts.
// Inputs and outputs are slot-major. Invalid IDs produce non-finite outputs.
func (r *cudaFoundation) encodeMatvecQ2KSelected(t cudaModelTensor, ids cudamem.Slice[int32], in, out cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(t, 10, 84); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols*slots || uint64(out.Len()) != rows*slots {
		return fmt.Errorf("cuda: Q2_K selected: invalid input or output shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.q2KSelectedFn).Grid(uint32(rows*slots)).Block(256).On(r.device.DefaultStream()).Launch(t.device, ids, in, out, uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch Q2_K selected: %w", err)
		}
		return nil
	})
}

// encodeQuantQ8K appends CUDA-oracle Q8_K activation quantization. Each row
// contains cols elements and each 256-element block occupies 292 bytes.
func (r *cudaFoundation) encodeQuantQ8K(in cudamem.Slice[float32], out cudamem.Slice[byte], cols int) error {
	if cols <= 0 || cols%256 != 0 || in.Len() == 0 || in.Len()%cols != 0 || uint64(in.Len()/256) > math.MaxInt32 || uint64(out.Len()) != uint64(in.Len()/256)*292 {
		return fmt.Errorf("cuda: Q8_K quantization: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.quantQ8KFn).Grid(uint32(in.Len()/256)).Block(256).On(r.device.DefaultStream()).Launch(in, out); err != nil {
			return fmt.Errorf("cuda: launch Q8_K quantization: %w", err)
		}
		return nil
	})
}

// encodeMatvecIQ2SelectedQ8K appends paired projections using Q8_K activations.
func (r *cudaFoundation) encodeMatvecIQ2SelectedQ8K(gate, up cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], gateOut, upOut cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertProfile(up, 16, 66); err != nil {
		return err
	}
	for i := range gate.dimensions {
		if gate.dimensions[i] != up.dimensions[i] {
			return fmt.Errorf("cuda: IQ2 Q8_K selected pair: mismatched dimensions")
		}
	}
	cols, rows, experts := gate.dimensions[0], gate.dimensions[1], gate.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/256*292 || uint64(gateOut.Len()) != slots*rows || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("cuda: IQ2 Q8_K selected pair: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.iq2SelectedQ8KFn).Grid(uint32(rows*slots)).Block(256).On(r.device.DefaultStream()).Launch(gate.device, up.device, ids, in, gateOut, upOut, uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch IQ2 Q8_K selected pair: %w", err)
		}
		return nil
	})
}

// encodeMatvecQ2KSelectedQ8K appends down projections with slot-major Q8_K input.
func (r *cudaFoundation) encodeMatvecQ2KSelectedQ8K(t cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(t, 10, 84); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/256*292*slots || uint64(out.Len()) != rows*slots {
		return fmt.Errorf("cuda: Q2_K Q8_K selected: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.q2KSelectedQ8KFn).Grid(uint32(rows*slots)).Block(256).On(r.device.DefaultStream()).Launch(t.device, ids, in, out, uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch Q2_K Q8_K selected: %w", err)
		}
		return nil
	})
}
