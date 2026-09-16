
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

__device__ float ds4_iq2_part_q8_1(
    const unsigned char *p,
    const ds4_q8_1_activation *x,
    unsigned part,
    const unsigned int *grid_lo_table,
    const unsigned int *grid_hi_table,
    const unsigned char *ksigns_table) {
  const unsigned char *q = p + 2 + part * 8;
  unsigned long long q_addr = reinterpret_cast<unsigned long long>(q);
  unsigned q_lo, aux;
  if ((q_addr & 7) == 0) {
    uint2 q_vec = *reinterpret_cast<const uint2 *>(q);
    q_lo = q_vec.x;
    aux = q_vec.y;
  } else if ((q_addr & 3) == 0) {
    const unsigned int *q32 = reinterpret_cast<const unsigned int *>(q);
    q_lo = q32[0];
    aux = q32[1];
  } else {
    const unsigned short *q16 = reinterpret_cast<const unsigned short *>(q);
    q_lo = unsigned(q16[0]) | (unsigned(q16[1]) << 16);
    aux = unsigned(q16[2]) | (unsigned(q16[3]) << 16);
  }
  const unsigned char q_bytes[4] = {
    (unsigned char)(q_lo & 0xff),
    (unsigned char)((q_lo >> 8) & 0xff),
    (unsigned char)((q_lo >> 16) & 0xff),
    (unsigned char)((q_lo >> 24) & 0xff),
  };
  const int *u = reinterpret_cast<const int *>(x->qs);
  int sum = 0;
  for (unsigned group = 0; group < 4; group++) {
    unsigned signs = ksigns_table[(aux >> (group * 7)) & 127] * 0x01010101u;
    unsigned grid_lo = grid_lo_table[q_bytes[group]];
    int signs0 = __vcmpne4(signs & 0x08040201u, 0);
    int grid0 = __vsub4(grid_lo ^ signs0, signs0);
    sum = __dp4a(grid0, u[group * 2 + 0], sum);
    unsigned grid_hi = grid_hi_table[q_bytes[group]];
    int signs1 = __vcmpne4(signs & 0x80402010u, 0);
    int grid1 = __vsub4(grid_hi ^ signs1, signs1);
    sum = __dp4a(grid1, u[group * 2 + 1], sum);
  }
  int ls = int(aux >> 27) | 1;
  float d = ds4_half_to_float(*reinterpret_cast<const unsigned short *>(p));
  return d * ds4_half_to_float(x->d) * float(sum * ls / 8);
}

struct __align__(16) ds4_iq2_hoisted_part {
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
  unsigned long long q_addr = reinterpret_cast<unsigned long long>(q);
  unsigned q_lo, aux;
  if ((q_addr & 7) == 0) {
    uint2 q_vec = *reinterpret_cast<const uint2 *>(q);
    q_lo = q_vec.x;
    aux = q_vec.y;
  } else if ((q_addr & 3) == 0) {
    const unsigned int *q32 = reinterpret_cast<const unsigned int *>(q);
    q_lo = q32[0];
    aux = q32[1];
  } else {
    const unsigned short *q16 = reinterpret_cast<const unsigned short *>(q);
    q_lo = unsigned(q16[0]) | (unsigned(q16[1]) << 16);
    aux = unsigned(q16[2]) | (unsigned(q16[3]) << 16);
  }
  const unsigned char q_bytes[4] = {
    (unsigned char)(q_lo & 0xff),
    (unsigned char)((q_lo >> 8) & 0xff),
    (unsigned char)((q_lo >> 16) & 0xff),
    (unsigned char)((q_lo >> 24) & 0xff),
  };
  #pragma unroll
  for (unsigned group = 0; group < 4; group++) {
    unsigned signs = ksigns_table[(aux >> (group * 7)) & 127] * 0x01010101u;
    unsigned grid_lo = grid_lo_table[q_bytes[group]];
    int signs0 = __vcmpne4(signs & 0x08040201u, 0);
    h->grid[group * 2 + 0] = __vsub4(grid_lo ^ signs0, signs0);
    unsigned grid_hi = grid_hi_table[q_bytes[group]];
    int signs1 = __vcmpne4(signs & 0x80402010u, 0);
    h->grid[group * 2 + 1] = __vsub4(grid_hi ^ signs1, signs1);
  }
  h->ls = int(aux >> 27) | 1;
  h->d = ds4_half_to_float(*reinterpret_cast<const unsigned short *>(p));
}

__device__ float ds4_iq2_eval_hoisted(
    const ds4_iq2_hoisted_part &h,
    const ds4_q8_1_activation *x) {
  unsigned long long addr = reinterpret_cast<unsigned long long>(x->qs);
  int sum = 0;
  if ((addr & 15) == 0) {
    const uint4 *p = reinterpret_cast<const uint4 *>(x->qs);
    uint4 a = p[0], b = p[1];
    sum = __dp4a(h.grid[0], int(a.x), sum);
    sum = __dp4a(h.grid[1], int(a.y), sum);
    sum = __dp4a(h.grid[2], int(a.z), sum);
    sum = __dp4a(h.grid[3], int(a.w), sum);
    sum = __dp4a(h.grid[4], int(b.x), sum);
    sum = __dp4a(h.grid[5], int(b.y), sum);
    sum = __dp4a(h.grid[6], int(b.z), sum);
    sum = __dp4a(h.grid[7], int(b.w), sum);
  } else if ((addr & 7) == 0) {
    const uint2 *p = reinterpret_cast<const uint2 *>(x->qs);
    uint2 a = p[0], b = p[1], c = p[2], d = p[3];
    sum = __dp4a(h.grid[0], int(a.x), sum);
    sum = __dp4a(h.grid[1], int(a.y), sum);
    sum = __dp4a(h.grid[2], int(b.x), sum);
    sum = __dp4a(h.grid[3], int(b.y), sum);
    sum = __dp4a(h.grid[4], int(c.x), sum);
    sum = __dp4a(h.grid[5], int(c.y), sum);
    sum = __dp4a(h.grid[6], int(d.x), sum);
    sum = __dp4a(h.grid[7], int(d.y), sum);
  } else {
    const int *u = reinterpret_cast<const int *>(x->qs);
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
      sum = __dp4a(h.grid[i], u[i], sum);
    }
  }
  return h.d * ds4_half_to_float(x->d) * float(sum * h.ls / 8);
}

__device__ void ds4_iq2_eval_hoisted_pair(
    const ds4_iq2_hoisted_part &hg,
    const ds4_iq2_hoisted_part &hu,
    const ds4_q8_1_activation *x,
    float *out_g,
    float *out_u) {
  unsigned long long addr = reinterpret_cast<unsigned long long>(x->qs);
  int u0, u1, u2, u3, u4, u5, u6, u7;
  if ((addr & 15) == 0) {
    const uint4 *p = reinterpret_cast<const uint4 *>(x->qs);
    uint4 a = p[0], b = p[1];
    u0 = int(a.x); u1 = int(a.y); u2 = int(a.z); u3 = int(a.w);
    u4 = int(b.x); u5 = int(b.y); u6 = int(b.z); u7 = int(b.w);
  } else if ((addr & 7) == 0) {
    const uint2 *p = reinterpret_cast<const uint2 *>(x->qs);
    uint2 a = p[0], b = p[1], c = p[2], d = p[3];
    u0 = int(a.x); u1 = int(a.y); u2 = int(b.x); u3 = int(b.y);
    u4 = int(c.x); u5 = int(c.y); u6 = int(d.x); u7 = int(d.y);
  } else {
    const int *p = reinterpret_cast<const int *>(x->qs);
    u0 = p[0]; u1 = p[1]; u2 = p[2]; u3 = p[3];
    u4 = p[4]; u5 = p[5]; u6 = p[6]; u7 = p[7];
  }
  float xd = ds4_half_to_float(x->d);
  int sum_g = 0, sum_u = 0;
  sum_g = __dp4a(hg.grid[0], u0, sum_g);
  sum_u = __dp4a(hu.grid[0], u0, sum_u);
  sum_g = __dp4a(hg.grid[1], u1, sum_g);
  sum_u = __dp4a(hu.grid[1], u1, sum_u);
  sum_g = __dp4a(hg.grid[2], u2, sum_g);
  sum_u = __dp4a(hu.grid[2], u2, sum_u);
  sum_g = __dp4a(hg.grid[3], u3, sum_g);
  sum_u = __dp4a(hu.grid[3], u3, sum_u);
  sum_g = __dp4a(hg.grid[4], u4, sum_g);
  sum_u = __dp4a(hu.grid[4], u4, sum_u);
  sum_g = __dp4a(hg.grid[5], u5, sum_g);
  sum_u = __dp4a(hu.grid[5], u5, sum_u);
  sum_g = __dp4a(hg.grid[6], u6, sum_g);
  sum_u = __dp4a(hu.grid[6], u6, sum_u);
  sum_g = __dp4a(hg.grid[7], u7, sum_g);
  sum_u = __dp4a(hu.grid[7], u7, sum_u);
  *out_g += hg.d * xd * float(sum_g * hg.ls / 8);
  *out_u += hu.d * xd * float(sum_u * hu.ls / 8);
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

__device__ float ds4_q2_window_q8_1(const unsigned char *p, const ds4_q8_1_activation *x, unsigned iqs) {
  unsigned qoffset = 16 + iqs * 4;
  unsigned v = *reinterpret_cast<const unsigned int *>(p + qoffset);
  unsigned scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
  float sum_d = 0.0f, sum_m = 0.0f;
  for (unsigned i = 0; i < 4; i++) {
    const ds4_q8_1_activation *q = x + 4 * (iqs / 8) + i;
    unsigned scale = p[scale_offset + 2 * i];
    int q_packed = *reinterpret_cast<const int *>(&q->qs[(iqs % 8) * 4]);
    int vi = (v >> (2 * i)) & 0x03030303;
    int dot = __dp4a(vi, q_packed, 0);
    int minimum = int(scale >> 4) * __dp4a(0x01010101, q_packed, 0);
    float d8 = ds4_half_to_float(q->d);
    sum_d += d8 * float(dot * int(scale & 15));
    sum_m += d8 * float(minimum);
  }
  unsigned d_pair = *reinterpret_cast<const unsigned int *>(p + 80);
  float d = ds4_half_to_float(d_pair & 0xffff);
  float dm = ds4_half_to_float(d_pair >> 16);
  return d * sum_d - dm * sum_m;
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
