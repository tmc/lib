// Each table entry names one committed, route-ordered expert component.
// This fragment follows cuda_probe.cu and cuda_experts.cu in its own module.
// No expert id is multiplied into a whole-tensor address.
extern "C" __global__ void ds4_iq2_xxs_pair_address_f32(
    const unsigned long long *gate, const unsigned long long *up,
    const unsigned long long *gate_bytes, const unsigned long long *up_bytes,
    const float *x, float *gate_out, float *up_out,
    unsigned cols, unsigned rows, unsigned slots,
    unsigned long long input_count, unsigned long long output_count) {
  if (!rows || !cols || cols % 256 || blockDim.x != 256) return;
  unsigned row = blockIdx.x % rows, slot = blockIdx.x / rows;
  if (slot >= slots || blockIdx.x >= output_count) return;
  unsigned blocks = cols / 256;
  unsigned long long need = (unsigned long long)rows * blocks * 66;
  if (!gate[slot] || !up[slot] || gate_bytes[slot] != need ||
      up_bytes[slot] != need || input_count != cols) {
    if (threadIdx.x == 0) {
      gate_out[blockIdx.x] = __int_as_float(0x7fffffff);
      up_out[blockIdx.x] = __int_as_float(0x7fffffff);
    }
    return;
  }
  const unsigned char *gptr = (const unsigned char *)gate[slot];
  const unsigned char *uptr = (const unsigned char *)up[slot];
  unsigned long long offset = (unsigned long long)row * blocks * 66;
  float g = 0, u = 0;
  for (unsigned b = threadIdx.x; b < blocks; b += blockDim.x) {
    g += ds4_iq2_block_f32(gptr + offset + (unsigned long long)b * 66, x + b * 256);
    u += ds4_iq2_block_f32(uptr + offset + (unsigned long long)b * 66, x + b * 256);
  }
  __shared__ float gs[256], us[256];
  gs[threadIdx.x] = g; us[threadIdx.x] = u;
  __syncthreads();
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) { gs[threadIdx.x] += gs[threadIdx.x + stride]; us[threadIdx.x] += us[threadIdx.x + stride]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { gate_out[blockIdx.x] = gs[0]; up_out[blockIdx.x] = us[0]; }
}

extern "C" __global__ void ds4_q2_k_address_f32(
    const unsigned long long *weights, const unsigned long long *weight_bytes,
    const float *x, float *out, unsigned cols, unsigned rows, unsigned slots,
    unsigned long long input_count, unsigned long long output_count) {
  if (!rows || !cols || cols % 256 || blockDim.x != 256) return;
  unsigned row = blockIdx.x % rows, slot = blockIdx.x / rows;
  if (slot >= slots || blockIdx.x >= output_count) return;
  unsigned blocks = cols / 256;
  unsigned long long need = (unsigned long long)rows * blocks * 84;
  if (!weights[slot] || weight_bytes[slot] != need ||
      input_count != (unsigned long long)cols * slots) {
    if (threadIdx.x == 0) out[blockIdx.x] = __int_as_float(0x7fffffff);
    return;
  }
  const unsigned char *base = (const unsigned char *)weights[slot];
  const float *input = x + (unsigned long long)slot * cols;
  unsigned long long offset = (unsigned long long)row * blocks * 84;
  float sum = 0;
  for (unsigned b = threadIdx.x; b < blocks; b += blockDim.x) {
    const unsigned char *p = base + offset + (unsigned long long)b * 84;
    float d = ds4_half_to_float(unsigned(p[80]) | unsigned(p[81]) << 8);
    float dm = ds4_half_to_float(unsigned(p[82]) | unsigned(p[83]) << 8);
    float block_sum = 0;
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
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = partial[0];
}
