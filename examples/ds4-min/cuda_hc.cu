
extern "C" __global__ void ds4_hc_half_input(const float *in, unsigned short *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 16384) out[i] = ds4_q81_half_bits(in[i]);
}
// Launched as one warp: ds4_hc_split4_warp needs every lane for its shuffles,
// so no lane may return early on the threadIdx.x test.
extern "C" __global__ void ds4_hc_cached_split(
    const float *mix, const float *scale, const float *base, float *split, unsigned head) {
  if (blockIdx.x) return;
  if (threadIdx.x < 4) {
    const unsigned h = threadIdx.x;
    split[h] = 1.0f / (1.0f + expf(-(mix[h] * scale[0] + base[h]))) + 1.0e-6f;
  }
  if (!head) ds4_hc_split4_warp(mix, scale, base, split + 4, split + 8);
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

// ds4_hc_rms_half_batch is the unweighted ds4_rms of one 16384-float row per
// block, stored as half. It repeats ds4_rms's general path statement for
// statement, reduction order and all, so each half is the one
// ds4_hc_half_input would make from ds4_rms's float output. Launch with 1024
// threads, as encodeRMS does for this width.
extern "C" __global__ void ds4_hc_rms_half_batch(
    const float *x, unsigned short *out, unsigned int width) {
  const unsigned int tid = threadIdx.x;
  const unsigned int warp_id = tid >> 5;
  const unsigned int lane = tid & 31;
  const unsigned long long offset = (unsigned long long)blockIdx.x * width;

  const unsigned int num_warps = blockDim.x >> 5;
  float sum = 0.0f;
  for (unsigned int i = tid; i < width; i += blockDim.x) {
    const float v = x[offset + i];
    sum += v * v;
  }

  #pragma unroll
  for (unsigned int stride = 16; stride != 0; stride >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  }

  __shared__ float s_warp[32];
  if (lane == 0 && warp_id < 32) {
    s_warp[warp_id] = sum;
  }
  __syncthreads();

  float block_sum = 0.0f;
  if (warp_id == 0) {
    block_sum = (lane < num_warps && lane < 32) ? s_warp[lane] : 0.0f;
    #pragma unroll
    for (unsigned int stride = 16; stride != 0; stride >>= 1) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, stride);
    }
  }

  __shared__ float s_inv;
  if (tid == 0) {
    s_inv = rsqrtf(block_sum / float(width) + 1.0e-6f);
  }
  __syncthreads();

  const float inv = s_inv;
  for (unsigned int i = tid; i < width; i += blockDim.x)
    out[offset + i] = ds4_q81_half_bits(x[offset + i] * inv * 1.0f);
}
// One warp per token; see ds4_hc_cached_split.
extern "C" __global__ void ds4_hc_cached_split_batch(
    const float *mix, const float *scale, const float *base, float *split, unsigned head) {
  if (blockIdx.x) return;
  mix += (unsigned long long)blockIdx.y * 24;
  split += (unsigned long long)blockIdx.y * 24;
  if (threadIdx.x < 4) {
    const unsigned h = threadIdx.x;
    split[h] = 1.0f / (1.0f + expf(-(mix[h] * scale[0] + base[h]))) + 1.0e-6f;
  }
  if (!head) ds4_hc_split4_warp(mix, scale, base, split + 4, split + 8);
}
// Row r of out starts at r*out_stride; pre passes 8192 to write the
// collapsed half of its interleaved output in place.
extern "C" __global__ void ds4_hc_cached_collapse_batch(
    const float *state, const float *split, float *out, unsigned out_stride) {
  unsigned d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= 4096) return;
  state += (unsigned long long)blockIdx.y * 16384;
  split += (unsigned long long)blockIdx.y * 24;
  out += (unsigned long long)blockIdx.y * out_stride;
  float v = 0.0f;
  for (unsigned h = 0; h < 4; h++) v += state[h * 4096 + d] * split[h];
  out[d] = v;
}

// Fuse the split and collapse for the prefill path.  The first warp computes
// the 24 HC coefficients, then the block consumes them while collapsing the
// four state rows.  Keeping this as a separate entry point leaves the scalar
// and diagnostic batch kernels available as references.
extern "C" __global__ void ds4_hc_cached_split_collapse_batch(
    const float *mix, const float *scale, const float *base,
    const float *state, float *splitOut, float *out) {
  unsigned row = blockIdx.y;
  mix += (unsigned long long)row * 24;
  state += (unsigned long long)row * 16384;
  out += (unsigned long long)row * 4096;
  __shared__ float split[24];
  if (threadIdx.x < 4) {
    unsigned h = threadIdx.x;
    split[h] = 1.0f / (1.0f + expf(-(mix[h] * scale[0] + base[h]))) + 1.0e-6f;
  }
  if (threadIdx.x < 32)
    ds4_hc_split4_warp(mix, scale, base, split + 4, split + 8);
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x < 24)
    splitOut[row * 24 + threadIdx.x] = split[threadIdx.x];
  unsigned d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= 4096) return;
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

// ds4_hc_norm_batch normalizes the collapsed half of each interleaved
// 8192-float row of io into its second half, and also into norm, 4096 floats
// per row, for consumers that read the normalized rows packed. It repeats the
// weighted width-4096 path of ds4_rms statement for statement, so both copies
// are bit-identical to that kernel's output. Launch with 1024 threads.
extern "C" __global__ void ds4_hc_norm_batch(
    float *io, const float *weight, float *norm) {
  const unsigned int tid = threadIdx.x;
  const unsigned int warp_id = tid >> 5;
  const unsigned int lane = tid & 31;
  const float *xg = io + (unsigned long long)blockIdx.x * 8192;
  float *outg = io + (unsigned long long)blockIdx.x * 8192 + 4096;
  float *normg = norm + (unsigned long long)blockIdx.x * 4096;

  float v0 = xg[tid];
  float v1 = xg[tid + 1024];
  float v2 = xg[tid + 2048];
  float v3 = xg[tid + 3072];
  float sum = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;

  #pragma unroll
  for (unsigned int stride = 16; stride != 0; stride >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  }

  __shared__ float s_warp[32];
  if (lane == 0) s_warp[warp_id] = sum;
  __syncthreads();

  float block_sum = 0.0f;
  if (warp_id == 0) {
    block_sum = s_warp[lane];
    #pragma unroll
    for (unsigned int stride = 16; stride != 0; stride >>= 1) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, stride);
    }
  }

  __shared__ float s_inv;
  if (tid == 0) {
    s_inv = rsqrtf(block_sum * 0.000244140625f + 1.0e-6f);
  }
  __syncthreads();

  const float inv = s_inv;
  const float o0 = v0 * inv * weight[tid];
  const float o1 = v1 * inv * weight[tid + 1024];
  const float o2 = v2 * inv * weight[tid + 2048];
  const float o3 = v3 * inv * weight[tid + 3072];
  outg[tid] = o0;
  outg[tid + 1024] = o1;
  outg[tid + 2048] = o2;
  outg[tid + 3072] = o3;
  normg[tid] = o0;
  normg[tid + 1024] = o1;
  normg[tid + 2048] = o2;
  normg[tid + 3072] = o3;
}
