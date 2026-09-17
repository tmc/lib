
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

// Batch entry points retain the scalar kernels above as the reference.

extern "C" __global__ void ds4_hc_half_input_batch(const float *in, unsigned short *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  in += (unsigned long long)blockIdx.y * 16384;
  out += (unsigned long long)blockIdx.y * 16384;
  if (i < 16384) out[i] = ds4_q81_half_bits(in[i]);
}
extern "C" __global__ void ds4_hc_cached_split_batch(
    const float *mix, const float *scale, const float *base, float *split, unsigned head) {
  if (threadIdx.x || blockIdx.x) return;
  mix += (unsigned long long)blockIdx.y * 24;
  split += (unsigned long long)blockIdx.y * 24;
  for (unsigned h = 0; h < 4; h++)
    split[h] = 1.0f / (1.0f + expf(-(mix[h] * scale[0] + base[h]))) + 1.0e-6f;
  if (!head) ds4_hc_split4(mix, scale, base, split + 4, split + 8);
}
extern "C" __global__ void ds4_hc_cached_collapse_batch(
    const float *state, const float *split, float *out) {
  unsigned d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= 4096) return;
  state += (unsigned long long)blockIdx.y * 16384;
  split += (unsigned long long)blockIdx.y * 24;
  out += (unsigned long long)blockIdx.y * 4096;
  float v = 0.0f;
  for (unsigned h = 0; h < 4; h++) v += state[h * 4096 + d] * split[h];
  out[d] = v;
}
extern "C" __global__ void ds4_hc_cached_expand_batch(
    const float *state, const float *block, const float *split, float *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 16384) return;
  state += (unsigned long long)blockIdx.y * 16384;
  block += (unsigned long long)blockIdx.y * 4096;
  split += (unsigned long long)blockIdx.y * 24;
  out += (unsigned long long)blockIdx.y * 16384;
  unsigned dst = i / 4096, d = i % 4096;
  float v = block[d] * split[4 + dst];
  for (unsigned src = 0; src < 4; src++)
    v += split[8 + dst + src * 4] * state[src * 4096 + d];
  out[i] = v;
}

// The runtime stores collapsed and normalized rows interleaved for projection.
extern "C" __global__ void ds4_hc_pack_batch(
    const float *collapsed, const float *normalized, float *out) {
  unsigned d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= 4096) return;
  unsigned long long row = blockIdx.y;
  out[row * 8192 + d] = collapsed[row * 4096 + d];
  out[row * 8192 + 4096 + d] = normalized[row * 4096 + d];
}
