
__device__ float ds4_iq2_part_q8_1_scalar(const unsigned char *p, const ds4_q8_1_activation *x, unsigned part) {
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

// ds4_iq2_part_q8_1_split evaluates one 32-weight part of an IQ2_XXS block
// whose 8 code bytes are at q and whose half scale is dbits. The raw GGUF
// block and the aligned artifact store the same bytes in different places,
// and both kernels below reach this one function, so they compute the same
// value from the same bytes.
__device__ float ds4_iq2_part_q8_1_split(
    const unsigned char *q,
    unsigned short dbits,
    const ds4_q8_1_activation *x,
    const unsigned int *grid_lo_table,
    const unsigned int *grid_hi_table,
    const unsigned char *ksigns_table) {
  unsigned aux = unsigned(*reinterpret_cast<const unsigned short *>(q + 4)) |
                 (unsigned(*reinterpret_cast<const unsigned short *>(q + 6)) << 16);
  const int *u = reinterpret_cast<const int *>(x->qs);
  int sum = 0;
  for (unsigned group = 0; group < 4; group++) {
    unsigned signs = ksigns_table[(aux >> (group * 7)) & 127] * 0x01010101u;
    unsigned grid_lo = grid_lo_table[q[group]];
    int signs0 = __vcmpne4(signs & 0x08040201u, 0);
    int grid0 = __vsub4(grid_lo ^ signs0, signs0);
    sum = __dp4a(grid0, u[group * 2 + 0], sum);
    unsigned grid_hi = grid_hi_table[q[group]];
    int signs1 = __vcmpne4(signs & 0x80402010u, 0);
    int grid1 = __vsub4(grid_hi ^ signs1, signs1);
    sum = __dp4a(grid1, u[group * 2 + 1], sum);
  }
  int ls = int(aux >> 27) | 1;
  float d = ds4_half_to_float(dbits);
  return d * ds4_half_to_float(x->d) * float(sum * ls / 8);
}

__device__ float ds4_iq2_part_q8_1(
    const unsigned char *p,
    const ds4_q8_1_activation *x,
    unsigned part,
    const unsigned int *grid_lo_table,
    const unsigned int *grid_hi_table,
    const unsigned char *ksigns_table) {
  return ds4_iq2_part_q8_1_split(p + 2 + part * 8, *reinterpret_cast<const unsigned short *>(p),
                                 x, grid_lo_table, grid_hi_table, ksigns_table);
}

struct ds4_iq2_hoisted_part {
  int grid[8];
  float d;
  int ls;
};

__device__ void ds4_iq2_hoist_part(
    const unsigned char *p,
    unsigned part,
    const unsigned int *grid_lo_table,
    const unsigned int *grid_hi_table,
    const unsigned char *ksigns_table,
    ds4_iq2_hoisted_part *h) {
  const unsigned char *q = p + 2 + part * 8;
  unsigned aux = unsigned(*reinterpret_cast<const unsigned short *>(q + 4)) |
                 (unsigned(*reinterpret_cast<const unsigned short *>(q + 6)) << 16);
  for (unsigned group = 0; group < 4; group++) {
    unsigned signs = ksigns_table[(aux >> (group * 7)) & 127] * 0x01010101u;
    unsigned grid_lo = grid_lo_table[q[group]];
    int signs0 = __vcmpne4(signs & 0x08040201u, 0);
    h->grid[group * 2 + 0] = __vsub4(grid_lo ^ signs0, signs0);
    unsigned grid_hi = grid_hi_table[q[group]];
    int signs1 = __vcmpne4(signs & 0x80402010u, 0);
    h->grid[group * 2 + 1] = __vsub4(grid_hi ^ signs1, signs1);
  }
  h->ls = int(aux >> 27) | 1;
  h->d = ds4_half_to_float(*reinterpret_cast<const unsigned short *>(p));
}

__device__ float ds4_iq2_eval_hoisted(
    const ds4_iq2_hoisted_part &h,
    const ds4_q8_1_activation *x) {
  const int *u = reinterpret_cast<const int *>(x->qs);
  int sum = 0;
  #pragma unroll
  for (unsigned i = 0; i < 8; i++) {
    sum = __dp4a(h.grid[i], u[i], sum);
  }
  return h.d * ds4_half_to_float(x->d) * float(sum * h.ls / 8);
}

extern "C" __global__ void ds4_iq2_xxs_pair_selected_q8_1(
    const unsigned char *gate, const unsigned char *up, const int *ids,
    const ds4_q8_1_activation *x, float *gate_out, float *up_out,
    unsigned cols, unsigned rows, unsigned slots, unsigned experts) {
  __shared__ unsigned int s_grid_lo[256];
  __shared__ unsigned int s_grid_hi[256];
  __shared__ unsigned char s_ksigns[128];

  for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) {
    unsigned long long grid = ds4_cuda_iq2xxs_grid[i];
    s_grid_lo[i] = unsigned(grid);
    s_grid_hi[i] = unsigned(grid >> 32);
  }
  for (unsigned i = threadIdx.x; i < 32; i += blockDim.x) {
    reinterpret_cast<unsigned int *>(s_ksigns)[i] = reinterpret_cast<const unsigned int *>(ds4_cuda_ksigns_iq2xs)[i];
  }
  __syncthreads();

  unsigned output = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (output >= rows * slots) return;
  unsigned row = output % rows, slot = output / rows, lane = threadIdx.x % 32;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (lane == 0) { gate_out[output] = __int_as_float(0x7fffffff); up_out[output] = __int_as_float(0x7fffffff); }
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long offset = ((unsigned long long)expert * rows + row) * blocks * 66;
  float g = 0.0f, u = 0.0f;
  for (unsigned part = lane; part < cols / 32; part += 32) {
    unsigned b = part / 8, sub = part % 8;
    unsigned long long block_offset = offset + (unsigned long long)b * 66;
    g += ds4_iq2_part_q8_1(gate + block_offset, x + part, sub, s_grid_lo, s_grid_hi, s_ksigns);
    u += ds4_iq2_part_q8_1(up + block_offset, x + part, sub, s_grid_lo, s_grid_hi, s_ksigns);
  }
  for (unsigned stride = 16; stride; stride >>= 1) {
    g += __shfl_down_sync(0xffffffff, g, stride);
    u += __shfl_down_sync(0xffffffff, u, stride);
  }
  if (lane == 0) { gate_out[output] = g; up_out[output] = u; }
}

extern "C" __global__ void ds4_iq2_xxs_pair_selected_q8_1_scalar(
    const unsigned char *gate, const unsigned char *up, const int *ids,
    const ds4_q8_1_activation *x, float *gate_out, float *up_out,
    unsigned cols, unsigned rows, unsigned slots, unsigned experts) {
  unsigned output = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (output >= rows * slots) return;
  unsigned row = output % rows, slot = output / rows, lane = threadIdx.x % 32;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (lane == 0) { gate_out[output] = __int_as_float(0x7fffffff); up_out[output] = __int_as_float(0x7fffffff); }
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long offset = ((unsigned long long)expert * rows + row) * blocks * 66;
  float g = 0.0f, u = 0.0f;
  for (unsigned part = lane; part < cols / 32; part += 32) {
    unsigned b = part / 8, sub = part % 8;
    g += ds4_iq2_part_q8_1_scalar(gate + offset + (unsigned long long)b * 66, x + part, sub);
    u += ds4_iq2_part_q8_1_scalar(up + offset + (unsigned long long)b * 66, x + part, sub);
  }
  for (unsigned stride = 16; stride; stride >>= 1) {
    g += __shfl_down_sync(0xffffffff, g, stride);
    u += __shfl_down_sync(0xffffffff, u, stride);
  }
  if (lane == 0) { gate_out[output] = g; up_out[output] = u; }
}

__device__ float ds4_q2_window_q8_1_scalar(const unsigned char *p, const ds4_q8_1_activation *x, unsigned iqs) {
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

// ds4_q2_window_q8_1_split evaluates one lane's window of a Q2_K block from
// its code word v, the four scale bytes it uses packed low byte first, and
// the block's d/dmin pair. As with ds4_iq2_part_q8_1_split, the raw and the
// aligned kernels differ only in where they load these three values from.
__device__ float ds4_q2_window_q8_1_split(
    unsigned v, unsigned scales, unsigned d_pair,
    const ds4_q8_1_activation *x, unsigned iqs) {
  float sum_d = 0.0f, sum_m = 0.0f;
  for (unsigned i = 0; i < 4; i++) {
    const ds4_q8_1_activation *q = x + 4 * (iqs / 8) + i;
    unsigned scale = (scales >> (8 * i)) & 255;
    int q_packed = *reinterpret_cast<const int *>(&q->qs[(iqs % 8) * 4]);
    int vi = (v >> (2 * i)) & 0x03030303;
    int dot = __dp4a(vi, q_packed, 0);
    int minimum = int(scale >> 4) * __dp4a(0x01010101, q_packed, 0);
    float d8 = ds4_half_to_float(q->d);
    sum_d += d8 * float(dot * int(scale & 15));
    sum_m += d8 * float(minimum);
  }
  float d = ds4_half_to_float(d_pair & 0xffff);
  float dm = ds4_half_to_float(d_pair >> 16);
  return d * sum_d - dm * sum_m;
}

__device__ float ds4_q2_window_q8_1(const unsigned char *p, const ds4_q8_1_activation *x, unsigned iqs) {
  unsigned v = *reinterpret_cast<const unsigned int *>(p + 16 + iqs * 4);
  const unsigned char *s = p + iqs - iqs % 8 + (iqs % 8) / 4;
  unsigned scales = unsigned(s[0]) | unsigned(s[2]) << 8 | unsigned(s[4]) << 16 | unsigned(s[6]) << 24;
  return ds4_q2_window_q8_1_split(v, scales, *reinterpret_cast<const unsigned int *>(p + 80), x, iqs);
}

__device__ float ds4_q2_window_q8_1_hoisted(
    unsigned v, const unsigned char *scales, float d, float dm,
    const ds4_q8_1_activation *x, unsigned iqs) {
  float sum_d = 0.0f, sum_m = 0.0f;
  for (unsigned i = 0; i < 4; i++) {
    const ds4_q8_1_activation *q = x + 4 * (iqs / 8) + i;
    unsigned scale = scales[i];
    int q_packed = *reinterpret_cast<const int *>(&q->qs[(iqs % 8) * 4]);
    int vi = (v >> (2 * i)) & 0x03030303;
    int dot = __dp4a(vi, q_packed, 0);
    int minimum = int(scale >> 4) * __dp4a(0x01010101, q_packed, 0);
    float d8 = ds4_half_to_float(q->d);
    sum_d += d8 * float(dot * int(scale & 15));
    sum_m += d8 * float(minimum);
  }
  return d * sum_d - dm * sum_m;
}

extern "C" __global__ void ds4_q2_k_selected_q8_1(
    const unsigned char *weights, const int *ids, const ds4_q8_1_activation *x,
    float *out, unsigned cols, unsigned rows, unsigned slots, unsigned experts) {
  unsigned output = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (output >= rows * slots) return;
  unsigned row = output % rows, slot = output / rows, lane = threadIdx.x % 32;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (lane == 0) out[output] = __int_as_float(0x7fffffff);
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
  if (lane == 0) out[output] = sum;
}

extern "C" __global__ void ds4_q2_k_selected_q8_1_scalar(
    const unsigned char *weights, const int *ids, const ds4_q8_1_activation *x,
    float *out, unsigned cols, unsigned rows, unsigned slots, unsigned experts) {
  unsigned output = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (output >= rows * slots) return;
  unsigned row = output % rows, slot = output / rows, lane = threadIdx.x % 32;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (lane == 0) out[output] = __int_as_float(0x7fffffff);
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long offset = ((unsigned long long)expert * rows + row) * blocks * 84;
  const ds4_q8_1_activation *input = x + (unsigned long long)slot * (cols / 32);
  float sum = 0.0f;
  for (unsigned b = lane / 16; b < blocks; b += 2)
    sum += ds4_q2_window_q8_1_scalar(weights + offset + (unsigned long long)b * 84, input + b * 8, lane % 16);
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, stride);
  if (lane == 0) out[output] = sum;
}

// ds4_iq2_xxs_pair_selected_q8_1_aligned is ds4_iq2_xxs_pair_selected_q8_1
// reading the aligned artifact ds4_repack_iq2_xxs_aligned writes: the half
// scale of raw block i at dq[i] and its 64 code bytes at qs[8*i..8*i+8).
// Blocks keep their raw order, so only the loads differ.
extern "C" __global__ void ds4_iq2_xxs_pair_selected_q8_1_aligned(
    const unsigned short *gate_dq, const uint2 *gate_qs,
    const unsigned short *up_dq, const uint2 *up_qs,
    const int *ids, const ds4_q8_1_activation *x, float *gate_out, float *up_out,
    unsigned cols, unsigned rows, unsigned slots, unsigned experts) {
  __shared__ unsigned int s_grid_lo[256];
  __shared__ unsigned int s_grid_hi[256];
  __shared__ unsigned char s_ksigns[128];

  for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) {
    unsigned long long grid = ds4_cuda_iq2xxs_grid[i];
    s_grid_lo[i] = unsigned(grid);
    s_grid_hi[i] = unsigned(grid >> 32);
  }
  for (unsigned i = threadIdx.x; i < 32; i += blockDim.x) {
    reinterpret_cast<unsigned int *>(s_ksigns)[i] = reinterpret_cast<const unsigned int *>(ds4_cuda_ksigns_iq2xs)[i];
  }
  __syncthreads();

  unsigned output = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (output >= rows * slots) return;
  unsigned row = output % rows, slot = output / rows, lane = threadIdx.x % 32;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (lane == 0) { gate_out[output] = __int_as_float(0x7fffffff); up_out[output] = __int_as_float(0x7fffffff); }
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long first = ((unsigned long long)expert * rows + row) * blocks;
  float g = 0.0f, u = 0.0f;
  for (unsigned part = lane; part < cols / 32; part += 32) {
    unsigned b = part / 8, sub = part % 8;
    unsigned long long blk = first + b;
    g += ds4_iq2_part_q8_1_split(reinterpret_cast<const unsigned char *>(gate_qs + blk * 8 + sub), gate_dq[blk],
                                 x + part, s_grid_lo, s_grid_hi, s_ksigns);
    u += ds4_iq2_part_q8_1_split(reinterpret_cast<const unsigned char *>(up_qs + blk * 8 + sub), up_dq[blk],
                                 x + part, s_grid_lo, s_grid_hi, s_ksigns);
  }
  for (unsigned stride = 16; stride; stride >>= 1) {
    g += __shfl_down_sync(0xffffffff, g, stride);
    u += __shfl_down_sync(0xffffffff, u, stride);
  }
  if (lane == 0) { gate_out[output] = g; up_out[output] = u; }
}

// ds4_q2_k_selected_q8_1_aligned is ds4_q2_k_selected_q8_1 reading the
// aligned artifact ds4_repack_q2_k_aligned writes. That layout interleaves
// rows r and r^1: raw block (expert, row, b) becomes pair block
// ((expert*rows/2 + row/2)*blocks + b) with parity row&1, and each of its
// code words, scale words and d/dmin pair is stored at that parity.
extern "C" __global__ void ds4_q2_k_selected_q8_1_aligned(
    const unsigned int *dm2, const unsigned int *sc4, const unsigned int *qs2,
    const int *ids, const ds4_q8_1_activation *x,
    float *out, unsigned cols, unsigned rows, unsigned slots, unsigned experts) {
  unsigned output = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (output >= rows * slots) return;
  unsigned row = output % rows, slot = output / rows, lane = threadIdx.x % 32;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (lane == 0) out[output] = __int_as_float(0x7fffffff);
    return;
  }
  unsigned blocks = cols / 256;
  unsigned parity = row & 1;
  unsigned long long first = ((unsigned long long)expert * (rows / 2) + row / 2) * blocks;
  const ds4_q8_1_activation *input = x + (unsigned long long)slot * (cols / 32);
  unsigned iqs = lane % 16;
  unsigned scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
  float sum = 0.0f;
  for (unsigned b = lane / 16; b < blocks; b += 2) {
    unsigned long long pblk = first + b;
    const unsigned char *sc = reinterpret_cast<const unsigned char *>(sc4 + pblk * 8 + parity * 2);
    unsigned scales = 0;
    for (unsigned i = 0; i < 4; i++) {
      unsigned k = scale_offset + 2 * i;
      unsigned w = k / 4;
      scales |= unsigned(sc[((w >> 1) * 4 + (w & 1)) * 4 + k % 4]) << (8 * i);
    }
    sum += ds4_q2_window_q8_1_split(qs2[pblk * 32 + iqs * 2 + parity], scales, dm2[pblk * 2 + parity],
                                    input + b * 8, iqs);
  }
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, stride);
  if (lane == 0) out[output] = sum;
}
