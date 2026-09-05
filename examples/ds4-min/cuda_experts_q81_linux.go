//go:build linux

package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaExpertQ81Source follows the default aligned single-token MoE arithmetic
// in pinned antirez/ds4 b0a147a, cuda/mmq/ds4_mmq.cu:2554 and :4307. Reading raw
// GGUF weights avoids repacking but preserves the aligned kernels' lane mapping.
// IQ2 truncates the scaled integer dot before floating conversion. Q2 uses
// quantized codes for its minimum correction, not the stored half block sum.
const cudaExpertQ81Source = `
__device__ float ds4_iq2_part_q8_1(const unsigned char *p, const ds4_q8_1_activation *x, unsigned part) {
  const unsigned char *q = p + 2 + part * 8;
  unsigned aux = unsigned(q[4]) | unsigned(q[5]) << 8 | unsigned(q[6]) << 16 | unsigned(q[7]) << 24;
  int sum = 0;
  for (unsigned group = 0; group < 4; group++) {
    unsigned long long grid = ds4_cuda_iq2xxs_grid[q[group]];
    unsigned signs = ds4_cuda_ksigns_iq2xs[(aux >> (group * 7)) & 127];
    for (unsigned i = 0; i < 8; i++) {
      int w = int((grid >> (8 * i)) & 255);
      if (signs & (1u << i)) w = -w;
      sum += w * int(x->qs[group * 8 + i]);
    }
  }
  int ls = int(aux >> 27) | 1;
  float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
  return d * ds4_half_to_float(x->d) * float(sum * ls / 8);
}

extern "C" __global__ void ds4_iq2_xxs_pair_selected_q8_1(
    const unsigned char *gate, const unsigned char *up, const int *ids,
    const ds4_q8_1_activation *x, float *gate_out, float *up_out,
    unsigned cols, unsigned rows, unsigned experts) {
  unsigned row = blockIdx.x % rows, slot = blockIdx.x / rows, lane = threadIdx.x;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (lane == 0) { gate_out[blockIdx.x] = __int_as_float(0x7fffffff); up_out[blockIdx.x] = __int_as_float(0x7fffffff); }
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long offset = ((unsigned long long)expert * rows + row) * blocks * 66;
  float g = 0.0f, u = 0.0f;
  for (unsigned part = lane; part < cols / 32; part += 32) {
    unsigned b = part / 8, sub = part % 8;
    g += ds4_iq2_part_q8_1(gate + offset + (unsigned long long)b * 66, x + part, sub);
    u += ds4_iq2_part_q8_1(up + offset + (unsigned long long)b * 66, x + part, sub);
  }
  for (unsigned stride = 16; stride; stride >>= 1) {
    g += __shfl_down_sync(0xffffffff, g, stride);
    u += __shfl_down_sync(0xffffffff, u, stride);
  }
  if (lane == 0) { gate_out[blockIdx.x] = g; up_out[blockIdx.x] = u; }
}

__device__ float ds4_q2_window_q8_1(const unsigned char *p, const ds4_q8_1_activation *x, unsigned iqs) {
  unsigned qoffset = 16 + iqs * 4;
  unsigned v = unsigned(p[qoffset]) | unsigned(p[qoffset+1]) << 8 | unsigned(p[qoffset+2]) << 16 | unsigned(p[qoffset+3]) << 24;
  unsigned scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
  float sum_d = 0.0f, sum_m = 0.0f;
  for (unsigned i = 0; i < 4; i++) {
    const ds4_q8_1_activation *q = x + 4 * (iqs / 8) + i;
    unsigned scale = p[scale_offset + 2 * i];
    int dot = 0, minimum = 0;
    for (unsigned j = 0; j < 4; j++) {
      int value = int(q->qs[(iqs % 8) * 4 + j]);
      dot += int((v >> (8 * j + 2 * i)) & 3) * value;
      minimum += int(scale >> 4) * value;
    }
    float d8 = ds4_half_to_float(q->d);
    sum_d += d8 * float(dot * int(scale & 15));
    sum_m += d8 * float(minimum);
  }
  float d = ds4_half_to_float(unsigned(p[80]) | unsigned(p[81]) << 8);
  float dm = ds4_half_to_float(unsigned(p[82]) | unsigned(p[83]) << 8);
  return d * sum_d - dm * sum_m;
}

extern "C" __global__ void ds4_q2_k_selected_q8_1(
    const unsigned char *weights, const int *ids, const ds4_q8_1_activation *x,
    float *out, unsigned cols, unsigned rows, unsigned experts) {
  unsigned row = blockIdx.x % rows, slot = blockIdx.x / rows, lane = threadIdx.x;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (lane == 0) out[blockIdx.x] = __int_as_float(0x7fffffff);
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long offset = ((unsigned long long)expert * rows + row) * blocks * 84;
  const ds4_q8_1_activation *input = x + (unsigned long long)slot * (cols / 32);
  float sum = 0.0f;
  for (unsigned b = lane / 16; b < blocks; b += 2)
    sum += ds4_q2_window_q8_1(weights + offset + (unsigned long long)b * 84, input + b * 8, lane % 16);
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, stride);
  if (lane == 0) out[blockIdx.x] = sum;
}
`

func (r *cudaFoundation) encodeMatvecIQ2SelectedQ81(gate, up cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], gateOut, upOut cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertProfile(up, 16, 66); err != nil {
		return err
	}
	for i := range gate.dimensions {
		if gate.dimensions[i] != up.dimensions[i] {
			return fmt.Errorf("cuda: IQ2 Q8_1 selected pair: mismatched dimensions")
		}
	}
	cols, rows, experts := gate.dimensions[0], gate.dimensions[1], gate.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/32*36 || uint64(gateOut.Len()) != slots*rows || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("cuda: IQ2 Q8_1 selected pair: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.iq2SelectedQ81Fn).Grid(uint32(rows*slots)).Block(32).On(r.device.DefaultStream()).Launch(gate.device, up.device, ids, in, gateOut, upOut, uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch IQ2 Q8_1 selected pair: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) encodeMatvecQ2KSelectedQ81(t cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(t, 10, 84); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/32*36*slots || uint64(out.Len()) != rows*slots {
		return fmt.Errorf("cuda: Q2_K Q8_1 selected: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.q2KSelectedQ81Fn).Grid(uint32(rows*slots)).Block(32).On(r.device.DefaultStream()).Launch(t.device, ids, in, out, uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch Q2_K Q8_1 selected: %w", err)
		}
		return nil
	})
}
