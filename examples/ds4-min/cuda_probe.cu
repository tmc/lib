
extern "C" __global__ void ds4_probe(float *p) {
  if (blockIdx.x == 0 && threadIdx.x == 0) p[0] = 4.0f;
}

__device__ float ds4_half_to_float(unsigned short h) {
  const unsigned int sign = unsigned(h & 0x8000) << 16;
  unsigned int exp = (h >> 10) & 0x1f;
  unsigned int frac = h & 0x03ff;
  if (exp == 0) {
    if (frac == 0) return __uint_as_float(sign);
    int e = -14;
    while ((frac & 0x0400) == 0) { frac <<= 1; e--; }
    frac &= 0x03ff;
    return __uint_as_float(sign | unsigned(e + 127) << 23 | frac << 13);
  }
  if (exp == 0x1f)
    return __uint_as_float(sign | 0x7f800000 | frac << 13);
  return __uint_as_float(sign | (exp - 15 + 127) << 23 | frac << 13);
}

extern "C" __global__ void ds4_read_u16(const unsigned short *p, unsigned int *out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = p[0];
}

extern "C" __global__ void ds4_read_u8(const unsigned char *p, unsigned long long offset, unsigned int *out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = p[offset];
}

// ds4_q8_0_matvec computes one output row per CUDA block directly from GGUF
// Q8_0 blocks. The input and output are ordinary device allocations owned by
// the caller.
extern "C" __global__ void ds4_q8_0_matvec(
    const unsigned char *weights, const float *x, float *out,
    unsigned int cols, unsigned int rows) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float partial[256];
  float sum = 0.0f;
  const unsigned int blocks_per_row = cols / 32;
  const unsigned char *row_weights = weights + row * blocks_per_row * 34;
  for (unsigned int block = threadIdx.x; block < blocks_per_row; block += blockDim.x) {
    const unsigned char *p = row_weights + block * 34;
    const unsigned short scale_bits = unsigned(p[0]) | unsigned(p[1]) << 8;
    const float scale = ds4_half_to_float(scale_bits);
    const signed char *q = reinterpret_cast<const signed char *>(p + 2);
    for (unsigned int i = 0; i < 32; i++)
      sum += scale * float(q[i]) * x[block * 32 + i];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = partial[0];
}

extern "C" __global__ void ds4_rms(
    const float *x, const float *weight, float *out,
    unsigned int width, unsigned int groups, unsigned int weighted) {
  const unsigned int group = blockIdx.x;
  if (group >= groups) return;
  __shared__ float partial[256];
  const unsigned int offset = group * width;
  float sum = 0.0f;
  for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = x[offset + i];
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(partial[0] / float(width) + 1.0e-6f);
  for (unsigned int i = threadIdx.x; i < width; i += blockDim.x)
    out[offset + i] = x[offset + i] * inv * (weighted ? weight[i] : 1.0f);
}

extern "C" __global__ void ds4_f16_matvec(
    const unsigned short *weights, const float *x, float *out,
    unsigned int cols, unsigned int rows) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float partial[256];
  float sum = 0.0f;
  const unsigned long long offset = (unsigned long long)row * cols;
  for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x)
    sum += ds4_half_to_float(weights[offset + i]) * x[i];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = partial[0];
}

// ds4_hc_pre applies the four-stream hyper-connection input transform and
// writes the 4096-value current stream followed by its weighted RMS norm.
// Only the first four projection rows contribute to the input transform. The
// remaining rows describe the output transform and are consumed by HCPost.
__device__ float ds4_half_round(float value);

extern "C" __global__ void ds4_hc_pre(
    const float *state, const unsigned short *hc_fn, const float *scale,
    const float *base, const float *norm_weight, float *out, unsigned int half_input) {
  if (blockIdx.x != 0) return;
  __shared__ float partial[256];
  __shared__ float inv;
  __shared__ float pre[4];
  float ss = 0.0f;
  for (unsigned int i = threadIdx.x; i < 16384; i += blockDim.x)
    ss += state[i] * state[i];
  partial[threadIdx.x] = ss;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) inv = rsqrtf(partial[0] / 16384.0f + 1.0e-6f);
  __syncthreads();
  for (unsigned int row = 0; row < 4; row++) {
    float sum = 0.0f;
    const unsigned long long off = (unsigned long long)row * 16384;
    for (unsigned int i = threadIdx.x; i < 16384; i += blockDim.x)
      sum += half_input
        ? ds4_half_to_float(hc_fn[off + i]) * ds4_half_round(state[i] * inv)
        : ds4_half_to_float(hc_fn[off + i]) * state[i] * inv;
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
      if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      pre[row] = 1.0f / (1.0f + expf(-(partial[0] * scale[0] + base[row]))) + 1.0e-6f;
    __syncthreads();
  }
  float cur_ss = 0.0f;
  for (unsigned int d = threadIdx.x; d < 4096; d += blockDim.x) {
    float v = 0.0f;
    for (unsigned int h = 0; h < 4; h++) v += state[h * 4096 + d] * pre[h];
    out[d] = v;
    cur_ss += v * v;
  }
  partial[threadIdx.x] = cur_ss;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) inv = rsqrtf(partial[0] / 4096.0f + 1.0e-6f);
  __syncthreads();
  for (unsigned int d = threadIdx.x; d < 4096; d += blockDim.x)
    out[4096 + d] = out[d] * inv * norm_weight[d];
}

// ds4_hc_split4 derives the hyper-connection output gate and the Sinkhorn
// normalized 4x4 combination matrix from one mixing vector. It matches the
// pinned 20 Sinkhorn iterations of the CPU fixture.
__device__ void ds4_hc_split4(
    const float *mix, const float *scale, const float *base,
    float *post, float *comb) {
  for (unsigned int h = 0; h < 4; h++)
    post[h] = 2.0f / (1.0f + expf(-(mix[4 + h] * scale[1] + base[4 + h])));
  for (unsigned int dst = 0; dst < 4; dst++) {
    float mx = -3.402823466e+38f;
    for (unsigned int src = 0; src < 4; src++) {
      const unsigned int i = src + dst * 4;
      comb[i] = mix[8 + i] * scale[2] + base[8 + i];
      mx = fmaxf(mx, comb[i]);
    }
    float sum = 0.0f;
    for (unsigned int src = 0; src < 4; src++) {
      const unsigned int i = src + dst * 4;
      comb[i] = expf(comb[i] - mx);
      sum += comb[i];
    }
    for (unsigned int src = 0; src < 4; src++) {
      const unsigned int i = src + dst * 4;
      comb[i] = comb[i] / sum + 1.0e-6f;
    }
  }
  for (unsigned int src = 0; src < 4; src++) {
    float sum = 0.0f;
    for (unsigned int dst = 0; dst < 4; dst++) sum += comb[src + dst * 4];
    for (unsigned int dst = 0; dst < 4; dst++) comb[src + dst * 4] /= sum + 1.0e-6f;
  }
  for (unsigned int iter = 1; iter < 20; iter++) {
    for (unsigned int dst = 0; dst < 4; dst++) {
      float sum = 0.0f;
      for (unsigned int src = 0; src < 4; src++) sum += comb[src + dst * 4];
      for (unsigned int src = 0; src < 4; src++) comb[src + dst * 4] /= sum + 1.0e-6f;
    }
    for (unsigned int src = 0; src < 4; src++) {
      float sum = 0.0f;
      for (unsigned int dst = 0; dst < 4; dst++) sum += comb[src + dst * 4];
      for (unsigned int dst = 0; dst < 4; dst++) comb[src + dst * 4] /= sum + 1.0e-6f;
    }
  }
}

// ds4_hc_post writes the hyper-connection residual update: every output stream
// mixes the block output through its own gate and the four input streams
// through the combination matrix.
extern "C" __global__ void ds4_hc_post(
    const float *state, const float *block, const unsigned short *hc_fn,
    const float *scale, const float *base, float *out, unsigned int half_input) {
  if (blockIdx.x != 0) return;
  __shared__ float partial[256];
  __shared__ float mix[24];
  __shared__ float post[4];
  __shared__ float comb[16];
  __shared__ float inv;
  float ss = 0.0f;
  for (unsigned int i = threadIdx.x; i < 16384; i += blockDim.x)
    ss += state[i] * state[i];
  partial[threadIdx.x] = ss;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) inv = rsqrtf(partial[0] / 16384.0f + 1.0e-6f);
  __syncthreads();
  for (unsigned int row = 0; row < 24; row++) {
    float sum = 0.0f;
    const unsigned long long off = (unsigned long long)row * 16384;
    for (unsigned int i = threadIdx.x; i < 16384; i += blockDim.x)
      sum += half_input
        ? ds4_half_to_float(hc_fn[off + i]) * ds4_half_round(state[i] * inv)
        : ds4_half_to_float(hc_fn[off + i]) * state[i] * inv;
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
      if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) mix[row] = partial[0];
    __syncthreads();
  }
  if (threadIdx.x == 0) ds4_hc_split4(mix, scale, base, post, comb);
  __syncthreads();
  for (unsigned int i = threadIdx.x; i < 4 * 4096; i += blockDim.x) {
    const unsigned int dst = i / 4096, d = i % 4096;
    float v = block[d] * post[dst];
    for (unsigned int src = 0; src < 4; src++)
      v += comb[dst + src * 4] * state[src * 4096 + d];
    out[i] = v;
  }
}

// ds4_get_row_f16 widens one F16 matrix row, which is how a token embedding
// enters the layer loop.
extern "C" __global__ void ds4_get_row_f16(
    const unsigned short *weights, float *out, unsigned int row, unsigned int cols) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cols) out[i] = ds4_half_to_float(weights[(unsigned long long)row * cols + i]);
}

// ds4_half_round returns value after one round trip through IEEE binary16,
// which is how raw key-value rows are stored in the attention cache.
__device__ float ds4_half_round(float value) {
  const unsigned int x = __float_as_uint(value);
  const unsigned int sign = (x >> 16) & 0x8000u;
  const unsigned int mag = x & 0x7fffffffu;
  if (mag >= 0x7f800000u) return value;
  if (mag >= 0x477ff000u) return ds4_half_to_float((unsigned short)(sign | 0x7c00u));
  if (mag >= 0x38800000u) {
    const unsigned int bits = mag + 0x00000fffu + ((mag >> 13) & 1u);
    return ds4_half_to_float((unsigned short)(sign | (bits - 0x38000000u) >> 13));
  }
  if (mag >= 0x33000000u) {
    const int shift = 126 - (int)(mag >> 23);
    const unsigned int m = (mag & 0x007fffffu) | 0x00800000u;
    unsigned int h = m >> shift;
    const unsigned int rem = m & ((1u << shift) - 1u);
    const unsigned int mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (h & 1u))) h++;
    return ds4_half_to_float((unsigned short)(sign | h));
  }
  return ds4_half_to_float((unsigned short)sign);
}

__device__ float ds4_e4m3fn_value(int code) {
  const int e = (code >> 3) & 15, m = code & 7;
  return e == 0 ? (float)m * 0.001953125f : (1.0f + (float)m * 0.125f) * exp2f((float)(e - 7));
}

// ds4_e4m3fn_round rounds one value to the nearest representable E4M3FN
// magnitude, matching the checkpoint's FP8 key-value quantization.
__device__ float ds4_e4m3fn_round(float value) {
  const float sign = value < 0.0f ? -1.0f : 1.0f;
  const float ax = fminf(fabsf(value), 448.0f);
  int lo = 0, hi = 126;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (ds4_e4m3fn_value(mid) <= ax) lo = mid; else hi = mid - 1;
  }
  int best = lo;
  if (best < 126) {
    const float bd = fabsf(ax - ds4_e4m3fn_value(best));
    const float nd = fabsf(ax - ds4_e4m3fn_value(best + 1));
    if (nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0)) best++;
  }
  return sign * ds4_e4m3fn_value(best);
}

// ds4_rope_attention_raw is the layer 0 and 1 attention path. It rotates the
// query and key-value vectors, quantizes and stores the key-value row in the
// rolling raw cache, attends with an additive sink, and inverse rotates the
// head outputs. One thread per head. Out holds the rotated queries, the stored
// key-value row, and the head outputs, in that order.
extern "C" __global__ void ds4_rope_attention_raw(
    const float *q, const float *kv, const float *sinks, float *cache,
    float *scores, float *out, unsigned int position, unsigned int cache_cap) {
  const unsigned int head = threadIdx.x;
  if (blockIdx.x != 0 || head >= 64) return;
  const unsigned int qoff = head * 512;
  for (unsigned int i = 0; i < 512; i++) out[qoff + i] = q[qoff + i];
  if (head == 0)
    for (unsigned int i = 0; i < 512; i++) out[32768 + i] = kv[i];
  const float theta_scale = powf(10000.0f, -2.0f / 64.0f);
  float theta = (float)position;
  for (unsigned int i = 0; i < 64; i += 2) {
    const unsigned int qi = qoff + 448 + i;
    const float c = cosf(theta), s = sinf(theta);
    float a = out[qi], b = out[qi + 1];
    out[qi] = a * c - b * s;
    out[qi + 1] = a * s + b * c;
    if (head == 0) {
      const unsigned int ki = 32768 + 448 + i;
      a = out[ki];
      b = out[ki + 1];
      out[ki] = a * c - b * s;
      out[ki + 1] = a * s + b * c;
    }
    theta *= theta_scale;
  }
  __syncthreads();
  if (head == 0) {
    for (unsigned int off = 0; off < 448; off += 64) {
      float amax = 1.0e-4f;
      for (unsigned int i = 0; i < 64; i++) amax = fmaxf(amax, fabsf(out[32768 + off + i]));
      const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
      for (unsigned int i = 0; i < 64; i++)
        out[32768 + off + i] =
            ds4_e4m3fn_round(fminf(fmaxf(out[32768 + off + i] / scale, -448.0f), 448.0f)) * scale;
    }
    const unsigned int row = position % cache_cap;
    for (unsigned int i = 0; i < 512; i++)
      cache[row * 512 + i] = ds4_half_round(out[32768 + i]);
  }
  __syncthreads();
  const unsigned int count = min(position + 1, cache_cap);
  const unsigned int first = position + 1 - count;
  float maximum = sinks[head];
  for (unsigned int t = 0; t < count; t++) {
    const unsigned int row = (first + t) % cache_cap;
    float score = 0.0f;
    // Attention always reads the half-rounded cache, as the oracle does: it
    // pushes the row through f16 before attending, at every position.
    const float *values = cache + row * 512;
    // The oracle rounds BOTH operands to f16 before the dot: it loads Q from
    // f32 and casts it to half at kernel entry alongside the half KV, then
    // widens both for the accumulation. Rounding only KV leaves Q at full f32
    // and diverges from the capture at f16 ULP scale.
    for (unsigned int i = 0; i < 512; i++) score += ds4_half_round(out[qoff + i]) * values[i];
    score /= sqrtf(512.0f);
    scores[head * cache_cap + t] = score;
    maximum = fmaxf(maximum, score);
  }
  float denom = expf(sinks[head] - maximum);
  for (unsigned int t = 0; t < count; t++)
    denom += expf(scores[head * cache_cap + t] - maximum);
  for (unsigned int i = 0; i < 512; i++) {
    float value = 0.0f;
    for (unsigned int t = 0; t < count; t++) {
      const unsigned int row = (first + t) % cache_cap;
      value += expf(scores[head * cache_cap + t] - maximum) * cache[row * 512 + i];
    }
    out[33280 + qoff + i] = value / denom;
  }
  theta = (float)position;
  for (unsigned int i = 0; i < 64; i += 2) {
    const unsigned int oi = 33280 + qoff + 448 + i;
    const float c = cosf(theta), s = -sinf(theta);
    const float a = out[oi], b = out[oi + 1];
    out[oi] = a * c - b * s;
    out[oi + 1] = a * s + b * c;
    theta *= theta_scale;
  }
}

extern "C" __global__ void ds4_raw_attention_prepare(
    const float *q, const float *kv, float *cache, float *out,
    unsigned int position, unsigned int cache_cap) {
  const unsigned int head = threadIdx.x;
  if (blockIdx.x != 0 || head >= 64) return;
  const unsigned int qoff = head * 512;
  for (unsigned int i = 0; i < 512; i++) out[qoff + i] = q[qoff + i];
  if (head == 0)
    for (unsigned int i = 0; i < 512; i++) out[32768 + i] = kv[i];
  const float theta_scale = powf(10000.0f, -2.0f / 64.0f);
  float theta = (float)position;
  for (unsigned int i = 0; i < 64; i += 2) {
    const unsigned int qi = qoff + 448 + i;
    const float c = cosf(theta), s = sinf(theta);
    float a = out[qi], b = out[qi + 1];
    out[qi] = a * c - b * s;
    out[qi + 1] = a * s + b * c;
    if (head == 0) {
      const unsigned int ki = 32768 + 448 + i;
      a = out[ki];
      b = out[ki + 1];
      out[ki] = a * c - b * s;
      out[ki + 1] = a * s + b * c;
    }
    theta *= theta_scale;
  }
  __syncthreads();
  for (unsigned int b = 0; b < 7; b++) {
    const unsigned int off = b * 64;
    const float val = out[32768 + off + head];
    float amax = fabsf(val);
    __shared__ float s_amax[2];
    for (int offset = 16; offset > 0; offset /= 2) {
      amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
    }
    if ((head & 31) == 0) s_amax[head >> 5] = amax;
    __syncthreads();
    if (head == 0) s_amax[0] = fmaxf(1.0e-4f, fmaxf(s_amax[0], s_amax[1]));
    __syncthreads();
    const float group_amax = s_amax[0];
    const float scale = exp2f(ceilf(log2f(group_amax / 448.0f)));
    out[32768 + off + head] = ds4_e4m3fn_round(fminf(fmaxf(val / scale, -448.0f), 448.0f)) * scale;
    __syncthreads();
  }
  const unsigned int row = position % cache_cap;
  for (unsigned int i = head; i < 512; i += 64) {
    cache[row * 512 + i] = ds4_half_round(out[32768 + i]);
  }
}

extern "C" __global__ void ds4_raw_attention_eval(
    const float *sinks,
    const float *cache,
    float *scores,
    float *out,
    unsigned int position,
    unsigned int cache_cap) {
  const unsigned int h = blockIdx.x;
  if (h >= 64) return;
  const unsigned int tid = threadIdx.x;
  const unsigned int qoff = h * 512;
  const unsigned int count = min(position + 1, cache_cap);
  const unsigned int first = position + 1 - count;

  extern __shared__ float s_raw_dyn[];
  float *s_scores = s_raw_dyn;
  __shared__ float s_q[512];
  __shared__ float s_max;
  __shared__ float s_denom;

  s_q[tid] = ds4_half_round(out[qoff + tid]);
  __syncthreads();

  if (tid < count) {
    const unsigned int row = (first + tid) % cache_cap;
    const float *values = cache + row * 512;
    float dot = 0.0f;
    #pragma unroll 1
    for (unsigned int i = 0; i < 512; i++) {
      dot += s_q[i] * values[i];
    }
    const float score = dot / sqrtf(512.0f);
    s_scores[tid] = score;
    scores[h * cache_cap + tid] = score;
  }
  __syncthreads();

  if (tid == 0) {
    float mx = sinks[h];
    for (unsigned int t = 0; t < count; t++) mx = fmaxf(mx, s_scores[t]);
    s_max = mx;
  }
  __syncthreads();

  const float mx = s_max;
  if (tid < count) {
    s_scores[tid] = expf(s_scores[tid] - mx);
  }
  __syncthreads();

  if (tid == 0) {
    float denom = expf(sinks[h] - mx);
    for (unsigned int t = 0; t < count; t++) denom += s_scores[t];
    s_denom = denom;
  }
  __syncthreads();

  const float denom = s_denom;
  float acc = 0.0f;
  for (unsigned int t = 0; t < count; t++) {
    const unsigned int row = (first + t) % cache_cap;
    acc += s_scores[t] * cache[row * 512 + tid];
  }
  out[33280 + qoff + tid] = acc / denom;
  __syncthreads();

  if (tid == 0) {
    const float theta_scale = powf(10000.0f, -2.0f / 64.0f);
    float theta = (float)position;
    for (unsigned int i = 0; i < 64; i += 2) {
      const unsigned int oi = 33280 + qoff + 448 + i;
      const float c = cosf(theta), s = -sinf(theta);
      const float a = out[oi], b = out[oi + 1];
      out[oi] = a * c - b * s;
      out[oi + 1] = a * s + b * c;
      theta *= theta_scale;
    }
  }
}


__device__ float ds4_router_softplus(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return expf(x);
  return log1pf(expf(x));
}

// ds4_router_hash selects the six experts of a dense layer from the token's
// row of the identity map. Mixture weights use the unbiased scores.
extern "C" __global__ void ds4_router_hash(
    const float *logits, const int *table, float *out, int *selected,
    float *weights, unsigned int token) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (unsigned int i = 0; i < 256; i++) {
    const float x = logits[i];
    out[i] = sqrtf(ds4_router_softplus(x));
  }
  float sum = 0.0f;
  for (unsigned int i = 0; i < 6; i++) {
    const int id = table[(unsigned long long)token * 6 + i];
    selected[i] = id;
    sum += out[id];
  }
  sum = fmaxf(sum, 6.103515625e-5f);
  for (unsigned int i = 0; i < 6; i++) weights[i] = out[selected[i]] / sum * 1.5f;
}

// ds4_router_biased selects the six routed experts by biased score and weights
// them by their unbiased scores, as the contract requires.
extern "C" __global__ void ds4_router_biased(
    const float *logits, const float *bias, float *out, int *selected, float *weights) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float selection[256];
  for (unsigned int i = 0; i < 256; i++) {
    const float x = logits[i];
    out[i] = sqrtf(ds4_router_softplus(x));
    selection[i] = out[i] + bias[i];
  }
  for (unsigned int k = 0; k < 6; k++) {
    int best = -1;
    for (unsigned int i = 0; i < 256; i++) {
      bool used = false;
      for (unsigned int j = 0; j < k; j++) used = used || selected[j] == (int)i;
      if (!used && (best < 0 || selection[i] > selection[best])) best = (int)i;
    }
    selected[k] = best;
  }
  float sum = 0.0f;
  for (unsigned int k = 0; k < 6; k++) sum += out[selected[k]];
  sum = fmaxf(sum, 6.103515625e-5f);
  for (unsigned int k = 0; k < 6; k++) weights[k] = out[selected[k]] / sum * 1.5f;
}

// ds4_swiglu applies the clamped SwiGLU activation. Slots take either one
// scalar weight or a device vector holding one weight per slot.
extern "C" __global__ void ds4_swiglu(
    const float *gate, const float *up, float *out,
    unsigned int width, unsigned int slots, float weight, const float *weights) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * slots) return;
  const float x = fminf(gate[i], 10.0f);
  const float y = fminf(fmaxf(up[i], -10.0f), 10.0f);
  out[i] = (x / (1.0f + expf(-x))) * y * (weights ? weights[i / width] : weight);
}

// ds4_sum_vectors reduces count adjacent vectors of the same width.
extern "C" __global__ void ds4_sum_vectors(
    const float *x, float *out, unsigned int width, unsigned int count) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width) return;
  float v = 0.0f;
  for (unsigned int row = 0; row < count; row++) v += x[(unsigned long long)row * width + i];
  out[i] = v;
}

extern "C" __global__ void ds4_sum_routed_shared(
    const float *experts, const float *shared, float *out, unsigned int width) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width) return;
  float v = 0.0f;
  #pragma unroll
  for (unsigned int row = 0; row < 6; row++)
    v += experts[(unsigned long long)row * width + i];
  out[i] = v + shared[i];
}