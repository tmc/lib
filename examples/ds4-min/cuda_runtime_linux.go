//go:build linux

package main

import (
	"errors"
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const cudaMathSource = cudaProbeSource + cudaExpertSource + cudaCompressedSource + cudaQ80Layout + cudaQ80MatvecBody + cudaQ81Source + cudaF16DefaultSource + cudaExpertQ81Source + cudaHCSource

const cudaProbeSource = `
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
}`

const (
	cudaMemHostRegisterDeviceMap = 2
	cudaMaxRegisteredModelRange  = 8 << 20
)

type cudaModelRange struct {
	host       uintptr
	device     driver.DevicePtr
	bytes      uint
	fileOffset uint64
	view       []byte
}

type cudaModelTensor struct {
	name       string
	typeID     uint32
	dimensions []uint64
	device     driver.DevicePtr
	fileOffset uint64
	bytes      uint64
}

// cudaFoundation owns the resources needed to begin a CUDA model runtime.
// It deliberately does not implement sessionBackend until token evaluation is
// available on CUDA.
type cudaFoundation struct {
	device *cuda.Device
	file   *os.File
	model  []byte
	name   string
	major  int
	minor  int
	hmm    bool
	gguf   *gguf.File
	mapped cudaModelRange

	probeTensor      string
	probeHost        uint16
	probeDevice      uint16
	probeCols        uint32
	probeRows        uint32
	probeBytes       uint64
	probeOffset      uint64
	probePtr         driver.DevicePtr
	matrixMod        driver.Module
	quantQ80Mod      driver.Module
	matrixFn         driver.Function
	quantQ80Fn       driver.Function
	quantQ81Fn       driver.Function
	q81MatvecFn      driver.Function
	q80QuantizedFn   driver.Function
	rmsFn            driver.Function
	f16Fn            driver.Function
	f16DefaultFn     driver.Function
	f16PairOrderedFn driver.Function
	hcPreFn          driver.Function
	hcHalfFn         driver.Function
	hcSplitFn        driver.Function
	hcCollapseFn     driver.Function
	hcExpandFn       driver.Function

	hcPostFn              driver.Function
	rowFn                 driver.Function
	rawAttentionFn        driver.Function
	routerHashFn          driver.Function
	routerBiasedFn        driver.Function
	swigluFn              driver.Function
	sumVectorsFn          driver.Function
	iq2SelectedFn         driver.Function
	q2KSelectedFn         driver.Function
	quantQ8KFn            driver.Function
	iq2SelectedQ8KFn      driver.Function
	q2KSelectedQ8KFn      driver.Function
	iq2SelectedQ81Fn      driver.Function
	q2KSelectedQ81Fn      driver.Function
	compressorPoolFn      driver.Function
	compressorAttentionFn driver.Function
	compressorIndexerFn   driver.Function
	mixedAttentionFn      driver.Function
	indexerQueryFn        driver.Function
	indexerSelectFn       driver.Function
}

func newCUDAFoundation(path string) (_ *cudaFoundation, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cuda: open model: %w", err)
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("cuda: stat model: %w", err)
	}
	if info.Size() != pinnedModelBytes {
		return nil, fmt.Errorf("cuda: model size %d, want %d", info.Size(), int64(pinnedModelBytes))
	}
	g, err := gguf.Read(f, info.Size())
	if err != nil {
		return nil, fmt.Errorf("cuda: read model: %w", err)
	}
	if err := validatePinnedProfile(g); err != nil {
		return nil, fmt.Errorf("cuda: model profile: %w", err)
	}
	model, err := syscall.Mmap(int(f.Fd()), 0, int(info.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("cuda: map model: %w", err)
	}
	defer func() {
		if err != nil {
			syscall.Munmap(model)
		}
	}()

	dev, err := cuda.NewDevice(0)
	if err != nil {
		return nil, fmt.Errorf("cuda: device: %w", err)
	}
	defer func() {
		if err != nil {
			dev.Close()
		}
	}()
	name, err := dev.Name()
	if err != nil {
		return nil, err
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	hmm, err := cudaSupportsHostPageTables(dev)
	if err != nil {
		return nil, err
	}
	r := &cudaFoundation{device: dev, file: f, model: model, name: name, major: major, minor: minor, hmm: hmm, gguf: g}
	if err := r.probe(); err != nil {
		return nil, err
	}
	if err := r.probeModel(g); err != nil {
		if r.mapped.host != 0 {
			_ = dev.WithContext(func() error { return driver.MemHostUnregister(r.mapped.host) })
			_ = syscall.Munmap(r.mapped.view)
			r.mapped = cudaModelRange{}
		}
		return nil, err
	}
	return r, nil
}

// cudaSupportsHostPageTables reports whether ordinary system allocations use
// the same virtual addresses on the CPU and GPU. All four properties are
// required: accepting a host pointer based only on UVA is not sufficient.
func cudaSupportsHostPageTables(dev *cuda.Device) (bool, error) {
	attrs := [...]driver.DeviceAttribute{
		driver.DevAttrIntegrated,
		driver.DevAttrUnifiedAddressing,
		driver.DevAttrPageableMemoryAccess,
		driver.DevAttrPageableMemoryAccessUsesHostPageTables,
	}
	for _, attr := range attrs {
		value, err := dev.Attribute(attr)
		if err != nil {
			return false, fmt.Errorf("cuda: query HMM capability %v: %w", attr, err)
		}
		if value == 0 {
			return false, nil
		}
	}
	return true, nil
}

// modelRange returns a device view of bytes in the stable GGUF mapping. On an
// integrated device using the host page tables this is the mmap address itself
// and requires neither registration nor a CUDA allocation. Other devices use
// the bounded registered fallback.
func (r *cudaFoundation) modelRange(fileOffset, length uint64) (driver.DevicePtr, error) {
	if length == 0 {
		return 0, fmt.Errorf("cuda: empty model range")
	}
	if fileOffset > uint64(len(r.model)) || length > uint64(len(r.model))-fileOffset {
		return 0, fmt.Errorf("cuda: model range [%d,%d) outside mapping", fileOffset, fileOffset+length)
	}
	if r.hmm {
		return driver.DevicePtr(uintptr(unsafe.Pointer(&r.model[fileOffset]))), nil
	}
	return r.mapModelRange(fileOffset, length)
}

func (r *cudaFoundation) tensor(name string) (cudaModelTensor, error) {
	if r.gguf == nil {
		return cudaModelTensor{}, fmt.Errorf("cuda: model is closed")
	}
	for _, tensor := range r.gguf.Tensors {
		if tensor.Name != name {
			continue
		}
		bytes, err := tensor.Bytes()
		if err != nil {
			return cudaModelTensor{}, fmt.Errorf("cuda: size tensor %q: %w", name, err)
		}
		offset := r.gguf.DataOffset + tensor.Offset
		ptr, err := r.modelRange(offset, bytes)
		if err != nil {
			return cudaModelTensor{}, fmt.Errorf("cuda: bind tensor %q: %w", name, err)
		}
		return cudaModelTensor{name: name, typeID: tensor.TypeID, dimensions: append([]uint64(nil), tensor.Dimensions...), device: ptr, fileOffset: offset, bytes: bytes}, nil
	}
	return cudaModelTensor{}, fmt.Errorf("cuda: model has no tensor %q", name)
}

func (r *cudaFoundation) probe() error {
	compiled, err := nvrtcx.Compile(cudaProbeSource,
		nvrtcx.WithName("ds4_probe.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		return fmt.Errorf("cuda: compile probe: %w", err)
	}
	return r.device.WithContext(func() error {
		mod, err := compiled.Load(r.device)
		if err != nil {
			return fmt.Errorf("cuda: load probe: %w", err)
		}
		defer mod.Unload()
		fn, err := mod.GetFunction("ds4_probe")
		if err != nil {
			return fmt.Errorf("cuda: find probe: %w", err)
		}
		buf, err := cudamem.Alloc[float32](r.device, 1)
		if err != nil {
			return fmt.Errorf("cuda: allocate probe: %w", err)
		}
		defer buf.Free()
		if err := cudalaunch.Kernel(fn).Launch(buf); err != nil {
			return fmt.Errorf("cuda: launch probe: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize probe: %w", err)
		}
		var got [1]float32
		if err := buf.CopyToHost(got[:]); err != nil {
			return fmt.Errorf("cuda: read probe: %w", err)
		}
		if got[0] != 4 {
			return fmt.Errorf("cuda: probe result %g, want 4", got[0])
		}
		return nil
	})
}

// mapModelRange makes a small, page-aligned part of the GGUF directly
// addressable by CUDA. It uses a private writable mapping because GB10 does not
// advertise CUDA's read-only host-registration extension. Device writes cannot
// modify the model file. Large ranges are rejected: pinning the entire 86 GB
// model would deprive the host of pageable memory.
func (r *cudaFoundation) mapModelRange(fileOffset, length uint64) (driver.DevicePtr, error) {
	if length == 0 || length > cudaMaxRegisteredModelRange {
		return 0, fmt.Errorf("cuda: model range length %d exceeds limit %d", length, cudaMaxRegisteredModelRange)
	}
	if fileOffset > uint64(len(r.model)) || length > uint64(len(r.model))-fileOffset {
		return 0, fmt.Errorf("cuda: model range [%d,%d) outside mapping", fileOffset, fileOffset+length)
	}
	if r.mapped.host != 0 {
		return 0, fmt.Errorf("cuda: model range already mapped")
	}
	page := uint64(syscall.Getpagesize())
	start := fileOffset &^ (page - 1)
	end := (fileOffset + length + page - 1) &^ (page - 1)
	if end > uint64(len(r.model)) {
		end = uint64(len(r.model))
	}
	bytes := uint(end - start)
	view, err := syscall.Mmap(int(r.file.Fd()), int64(start), int(bytes), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE)
	if err != nil {
		return 0, fmt.Errorf("cuda: map private model range: %w", err)
	}
	host := uintptr(unsafe.Pointer(&view[0]))
	err = r.device.WithContext(func() error {
		if err := driver.MemHostRegister(host, bytes, cudaMemHostRegisterDeviceMap); err != nil {
			return fmt.Errorf("register model range: %w", err)
		}
		device, err := driver.MemHostGetDevicePointer(host, 0)
		if err != nil {
			_ = driver.MemHostUnregister(host)
			return fmt.Errorf("get model device pointer: %w", err)
		}
		r.mapped = cudaModelRange{host: host, device: device, bytes: bytes, fileOffset: start, view: view}
		return nil
	})
	if err != nil {
		_ = syscall.Munmap(view)
		return 0, fmt.Errorf("cuda: map model range: %w", err)
	}
	return r.mapped.device + driver.DevicePtr(fileOffset-start), nil
}

func (r *cudaFoundation) probeModel(g *gguf.File) error {
	var tensor *gguf.Tensor
	for i := range g.Tensors {
		if g.Tensors[i].Name == "blk.0.attn_kv.weight" {
			tensor = &g.Tensors[i]
			break
		}
	}
	if tensor == nil {
		return fmt.Errorf("cuda: model has no tensor %q", "blk.0.attn_kv.weight")
	}
	if tensor.TypeID != 8 || len(tensor.Dimensions) != 2 || tensor.Dimensions[0]%32 != 0 {
		return fmt.Errorf("cuda: tensor %q is not a Q8_0 matrix", tensor.Name)
	}
	tensorBytes, err := tensor.Bytes()
	if err != nil {
		return fmt.Errorf("cuda: size tensor %q: %w", tensor.Name, err)
	}
	if tensorBytes > cudaMaxRegisteredModelRange {
		return fmt.Errorf("cuda: Q8_0 tensor %q is %d bytes, limit %d", tensor.Name, tensorBytes, cudaMaxRegisteredModelRange)
	}
	offset := g.DataOffset + tensor.Offset
	want := uint16(r.model[offset]) | uint16(r.model[offset+1])<<8
	ptr, err := r.modelRange(offset, tensorBytes)
	if err != nil {
		return fmt.Errorf("cuda: bind tensor %q: %w", tensor.Name, err)
	}
	compiled, err := nvrtcx.Compile(cudaProbeSource,
		nvrtcx.WithName("ds4_model_probe.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		return fmt.Errorf("cuda: compile model probe: %w", err)
	}
	return r.device.WithContext(func() error {
		mod, err := compiled.Load(r.device)
		if err != nil {
			return fmt.Errorf("cuda: load model probe: %w", err)
		}
		defer mod.Unload()
		fn, err := mod.GetFunction("ds4_read_u16")
		if err != nil {
			return fmt.Errorf("cuda: find model probe: %w", err)
		}
		out, err := cudamem.Alloc[uint32](r.device, 1)
		if err != nil {
			return fmt.Errorf("cuda: allocate model probe: %w", err)
		}
		defer out.Free()
		if err := cudalaunch.Kernel(fn).Launch(ptr, out); err != nil {
			return fmt.Errorf("cuda: launch model probe: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize model probe: %w", err)
		}
		var got [1]uint32
		if err := out.CopyToHost(got[:]); err != nil {
			return fmt.Errorf("cuda: read model probe: %w", err)
		}
		if got[0] != uint32(want) {
			return fmt.Errorf("cuda: model probe result %#x, want %#x", got[0], want)
		}
		r.probeTensor = tensor.Name
		r.probeHost = want
		r.probeDevice = uint16(got[0])
		r.probeCols = uint32(tensor.Dimensions[0])
		r.probeRows = uint32(tensor.Dimensions[1])
		r.probeBytes = tensorBytes
		r.probeOffset = offset
		r.probePtr = ptr
		return nil
	})
}

// q80Matvec multiplies the registered Q8_0 matrix by x. The caller owns both
// x and out; the foundation owns only the bounded registered model tensor.
func (r *cudaFoundation) q80Matvec(x, out cudamem.Slice[float32]) error {
	if r.probePtr == 0 || r.probeCols == 0 || r.probeRows == 0 {
		return fmt.Errorf("cuda: Q8_0 matrix is not bound")
	}
	if x.Len() != int(r.probeCols) || out.Len() != int(r.probeRows) {
		return fmt.Errorf("cuda: Q8_0 matrix input/output lengths %d/%d, want %d/%d", x.Len(), out.Len(), r.probeCols, r.probeRows)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.matrixFn).Grid(r.probeRows).Block(256).Launch(r.probePtr, x, out, r.probeCols, r.probeRows); err != nil {
			return fmt.Errorf("cuda: launch Q8_0 matrix: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize Q8_0 matrix: %w", err)
		}
		return nil
	})
}

// encodeQ80Matvec appends a matrix-vector product to the foundation's CUDA
// stream. It does not synchronize or copy results to the host.
func (r *cudaFoundation) encodeQ80Matvec(t cudaModelTensor, x, out cudamem.Slice[float32]) error {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.dimensions[0]%32 != 0 ||
		uint64(x.Len()) != t.dimensions[0] || uint64(out.Len()) != t.dimensions[1] {
		return fmt.Errorf("cuda: matvec %q: invalid Q8_0 tensor or input", t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.matrixFn).Grid(uint32(t.dimensions[1])).Block(256).On(r.device.DefaultStream()).Launch(
			t.device, x, out, uint32(t.dimensions[0]), uint32(t.dimensions[1])); err != nil {
			return fmt.Errorf("cuda: launch matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeRMS appends grouped RMS normalization to the foundation's CUDA
// stream. Weight must be an F32 vector when weighted is true.
func (r *cudaFoundation) encodeRMS(x cudamem.Slice[float32], weight cudaModelTensor, out cudamem.Slice[float32], groups int, weighted bool) error {
	if x.Len() == 0 || out.Len() != x.Len() || groups <= 0 || x.Len()%groups != 0 ||
		(weighted && (weight.typeID != 0 || len(weight.dimensions) != 1 || weight.dimensions[0] != uint64(x.Len()/groups))) {
		return fmt.Errorf("cuda: RMS: invalid input, weight, or groups")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		var weightPtr driver.DevicePtr
		if weighted {
			weightPtr = weight.device
		}
		weightedArg := uint32(0)
		if weighted {
			weightedArg = 1
		}
		if err := cudalaunch.Kernel(r.rmsFn).Grid(uint32(groups)).Block(256).On(r.device.DefaultStream()).Launch(
			x, weightPtr, out, uint32(x.Len()/groups), uint32(groups), weightedArg); err != nil {
			return fmt.Errorf("cuda: launch RMS: %w", err)
		}
		return nil
	})
}

// encodeF16Matvec appends an F16 matrix-vector product to the default stream.
func (r *cudaFoundation) encodeF16Matvec(t cudaModelTensor, x, out cudamem.Slice[float32]) error {
	if t.typeID != 1 || len(t.dimensions) != 2 || uint64(x.Len()) != t.dimensions[0] || uint64(out.Len()) != t.dimensions[1] {
		return fmt.Errorf("cuda: matvec %q: invalid F16 tensor or input", t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.f16Fn).Grid(uint32(t.dimensions[1])).Block(256).On(r.device.DefaultStream()).Launch(
			t.device, x, out, uint32(t.dimensions[0]), uint32(t.dimensions[1])); err != nil {
			return fmt.Errorf("cuda: launch matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeHCPre appends the DS4 hyper-connection input transform. Out contains
// the current stream followed by its attention-normalized form.
func (r *cudaFoundation) encodeHCPre(state cudamem.Slice[float32], hcFn, scale, base, norm cudaModelTensor, out cudamem.Slice[float32]) error {
	return r.encodeHCPrePrecision(state, hcFn, scale, base, norm, out, false)
}

func (r *cudaFoundation) encodeHCPrePrecision(state cudamem.Slice[float32], hcFn, scale, base, norm cudaModelTensor, out cudamem.Slice[float32], halfInput bool) error {
	half := uint32(0)
	if halfInput {
		half = 1
	}
	if state.Len() != 4*4096 || out.Len() != 2*4096 || hcFn.typeID != 1 || len(hcFn.dimensions) != 2 || hcFn.dimensions[0] != 4*4096 || hcFn.dimensions[1] != 24 ||
		scale.typeID != 0 || base.typeID != 0 || norm.typeID != 0 || len(scale.dimensions) != 1 || scale.dimensions[0] != 3 ||
		len(base.dimensions) != 1 || base.dimensions[0] != 24 || len(norm.dimensions) != 1 || norm.dimensions[0] != 4096 {
		return fmt.Errorf("cuda: HC pre: invalid tensor profile")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.hcPreFn).Grid(1).Block(256).On(r.device.DefaultStream()).Launch(
			state, hcFn.device, scale.device, base.device, norm.device, out, half); err != nil {
			return fmt.Errorf("cuda: launch HC pre: %w", err)
		}
		return nil
	})
}

// cudaKernels names every kernel the layer loop needs, in load order.
// encodeHCPost appends the DS4 hyper-connection residual update, folding one
// block output back into the four-stream state.
func (r *cudaFoundation) encodeHCPost(state, block cudamem.Slice[float32], hcFn, scale, base cudaModelTensor, out cudamem.Slice[float32]) error {
	return r.encodeHCPostPrecision(state, block, hcFn, scale, base, out, false)
}

func (r *cudaFoundation) encodeHCPostPrecision(state, block cudamem.Slice[float32], hcFn, scale, base cudaModelTensor, out cudamem.Slice[float32], halfInput bool) error {
	half := uint32(0)
	if halfInput {
		half = 1
	}
	if state.Len() != 4*4096 || block.Len() != 4096 || out.Len() != state.Len() ||
		hcFn.typeID != 1 || len(hcFn.dimensions) != 2 || hcFn.dimensions[0] != 4*4096 || hcFn.dimensions[1] != 24 ||
		scale.typeID != 0 || len(scale.dimensions) != 1 || scale.dimensions[0] != 3 ||
		base.typeID != 0 || len(base.dimensions) != 1 || base.dimensions[0] != 24 {
		return fmt.Errorf("cuda: HC post: invalid tensor profile")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.hcPostFn).Grid(1).Block(256).On(r.device.DefaultStream()).Launch(
			state, block, hcFn.device, scale.device, base.device, out, half); err != nil {
			return fmt.Errorf("cuda: launch HC post: %w", err)
		}
		return nil
	})
}

// encodeOutputHead appends the final hyper-connection transform and the logits
// projection. Mixed receives the flattened stream followed by its normalized
// form; logits receives one score per vocabulary entry.
//
// The output head reuses the layer input transform: both collapse the four
// streams through the first four projection rows and normalize the result. The
// head's own gate and base vectors are shorter than a layer's, so its tensor
// profile is checked here rather than in encodeHCPre.
func (r *cudaFoundation) encodeOutputHead(state cudamem.Slice[float32], out cudaOutput, mixed, logits cudamem.Slice[float32]) error {
	if err := r.encodeOutputTransform(state, out, mixed, logits, false); err != nil {
		return err
	}
	return r.encodeQ80Matvec(out.weight, mixed.Slice(4096, 8192), logits)
}

func (r *cudaFoundation) encodeOutputTransform(state cudamem.Slice[float32], out cudaOutput, mixed, logits cudamem.Slice[float32], halfInput bool) error {
	half := uint32(0)
	if halfInput {
		half = 1
	}
	if state.Len() != 4*4096 || mixed.Len() != 2*4096 ||
		out.hcFn.typeID != 1 || len(out.hcFn.dimensions) != 2 || out.hcFn.dimensions[0] != 4*4096 || out.hcFn.dimensions[1] < 4 ||
		out.hcScale.typeID != 0 || len(out.hcScale.dimensions) != 1 || out.hcScale.dimensions[0] < 1 ||
		out.hcBase.typeID != 0 || len(out.hcBase.dimensions) != 1 || out.hcBase.dimensions[0] < 4 ||
		out.norm.typeID != 0 || len(out.norm.dimensions) != 1 || out.norm.dimensions[0] != 4096 ||
		out.weight.typeID != 8 || len(out.weight.dimensions) != 2 || out.weight.dimensions[0] != 4096 ||
		uint64(logits.Len()) != out.weight.dimensions[1] {
		return fmt.Errorf("cuda: output head: invalid tensor profile")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.hcPreFn).Grid(1).Block(256).On(r.device.DefaultStream()).Launch(
			state, out.hcFn.device, out.hcScale.device, out.hcBase.device, out.norm.device, mixed, half); err != nil {
			return fmt.Errorf("cuda: launch output transform: %w", err)
		}
		return nil
	})
}

// encodeEmbeddingRow appends the widening of one F16 embedding row.
func (r *cudaFoundation) encodeEmbeddingRow(t cudaModelTensor, row uint32, out cudamem.Slice[float32]) error {
	if t.typeID != 1 || len(t.dimensions) != 2 || uint64(out.Len()) != t.dimensions[0] || uint64(row) >= t.dimensions[1] {
		return fmt.Errorf("cuda: embedding row %d of %q: invalid tensor or row", row, t.name)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		cols := uint32(t.dimensions[0])
		if err := cudalaunch.Kernel(r.rowFn).Grid((cols+255)/256).Block(256).On(r.device.DefaultStream()).Launch(
			t.device, out, row, cols); err != nil {
			return fmt.Errorf("cuda: launch embedding row: %w", err)
		}
		return nil
	})
}

// encodeRawAttention appends the uncompressed rotary attention used by layers
// 0 and 1. Out holds the rotated queries, the stored key-value row, and the
// inverse-rotated head outputs, in that order.
func (r *cudaFoundation) encodeRawAttention(q, kv cudamem.Slice[float32], sinks cudaModelTensor, cache, scores, out cudamem.Slice[float32], position int) error {
	if q.Len() != 64*512 || kv.Len() != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 ||
		cache.Len() == 0 || cache.Len()%512 != 0 || scores.Len() != 64*(cache.Len()/512) ||
		out.Len() != 32768+512+32768 || position < 0 {
		return fmt.Errorf("cuda: raw attention: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.rawAttentionFn).Grid(1).Block(64).On(r.device.DefaultStream()).Launch(
			q, kv, sinks.device, cache, scores, out, uint32(position), uint32(cache.Len()/512)); err != nil {
			return fmt.Errorf("cuda: launch raw attention: %w", err)
		}
		return nil
	})
}

// encodeRouterHash appends the dense-layer expert selection, which reads the
// token's row of the identity map.
func (r *cudaFoundation) encodeRouterHash(logits cudamem.Slice[float32], table cudaModelTensor, scores cudamem.Slice[float32], selected cudamem.Slice[int32], weights cudamem.Slice[float32], token uint32) error {
	if logits.Len() != 256 || scores.Len() != 256 || selected.Len() != 6 || weights.Len() != 6 ||
		table.typeID != 26 || len(table.dimensions) != 2 || table.dimensions[0] != 6 || uint64(token) >= table.dimensions[1] {
		return fmt.Errorf("cuda: router hash: invalid tensor profile")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.routerHashFn).Grid(1).Block(1).On(r.device.DefaultStream()).Launch(
			logits, table.device, scores, selected, weights, token); err != nil {
			return fmt.Errorf("cuda: launch router hash: %w", err)
		}
		return nil
	})
}

// encodeRouterBiased appends the routed-layer expert selection. Bias picks the
// six experts; their unbiased scores set the mixture weights.
func (r *cudaFoundation) encodeRouterBiased(logits cudamem.Slice[float32], bias cudaModelTensor, scores cudamem.Slice[float32], selected cudamem.Slice[int32], weights cudamem.Slice[float32]) error {
	if logits.Len() != 256 || scores.Len() != 256 || selected.Len() != 6 || weights.Len() != 6 ||
		bias.typeID != 0 || len(bias.dimensions) != 1 || bias.dimensions[0] != 256 {
		return fmt.Errorf("cuda: router biased: invalid tensor profile")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.routerBiasedFn).Grid(1).Block(1).On(r.device.DefaultStream()).Launch(
			logits, bias.device, scores, selected, weights); err != nil {
			return fmt.Errorf("cuda: launch router biased: %w", err)
		}
		return nil
	})
}

// encodeSwiGLU appends the clamped SwiGLU activation scaled by one weight.
func (r *cudaFoundation) encodeSwiGLU(gate, up, out cudamem.Slice[float32], weight float32) error {
	return r.encodeSwiGLUSlots(gate, up, cudamem.Slice[float32]{}, out, 1, weight)
}

// encodeSwiGLUSlots appends the clamped SwiGLU activation over slots adjacent
// expert lanes. A non-empty weights vector supplies one weight per slot and
// keeps the mixture weights on the device.
func (r *cudaFoundation) encodeSwiGLUSlots(gate, up, weights, out cudamem.Slice[float32], slots int, weight float32) error {
	if slots <= 0 || gate.Len() == 0 || gate.Len()%slots != 0 || up.Len() != gate.Len() || out.Len() != gate.Len() ||
		(weights.Len() != 0 && weights.Len() != slots) {
		return fmt.Errorf("cuda: SwiGLU: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		n := uint32(gate.Len())
		if err := cudalaunch.Kernel(r.swigluFn).Grid((n+255)/256).Block(256).On(r.device.DefaultStream()).Launch(
			gate, up, out, n/uint32(slots), uint32(slots), weight, weights); err != nil {
			return fmt.Errorf("cuda: launch SwiGLU: %w", err)
		}
		return nil
	})
}

// encodeSumVectors appends a reduction across count adjacent vectors.
func (r *cudaFoundation) encodeSumVectors(in, out cudamem.Slice[float32], count int) error {
	if count <= 0 || in.Len() == 0 || in.Len()%count != 0 || out.Len() != in.Len()/count {
		return fmt.Errorf("cuda: sum vectors: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		width := uint32(out.Len())
		if err := cudalaunch.Kernel(r.sumVectorsFn).Grid((width+255)/256).Block(256).On(r.device.DefaultStream()).Launch(
			in, out, width, uint32(count)); err != nil {
			return fmt.Errorf("cuda: launch sum vectors: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) cudaKernels() []struct {
	name string
	dst  *driver.Function
} {
	return []struct {
		name string
		dst  *driver.Function
	}{
		{"ds4_q8_0_matvec", &r.matrixFn},
		{"ds4_quant_q8_1", &r.quantQ81Fn},
		{"ds4_q8_0_matvec_q8_1", &r.q81MatvecFn},
		{"ds4_q8_0_matvec_quantized", &r.q80QuantizedFn},
		{"ds4_rms", &r.rmsFn},
		{"ds4_f16_matvec", &r.f16Fn},
		{"ds4_f16_matvec_half_input", &r.f16DefaultFn},
		{"ds4_f16_pair_ordered_chunks", &r.f16PairOrderedFn},
		{"ds4_hc_pre", &r.hcPreFn},
		{"ds4_hc_half_input", &r.hcHalfFn},
		{"ds4_hc_cached_split", &r.hcSplitFn},
		{"ds4_hc_cached_collapse", &r.hcCollapseFn},
		{"ds4_hc_cached_expand", &r.hcExpandFn},
		{"ds4_hc_post", &r.hcPostFn},
		{"ds4_get_row_f16", &r.rowFn},
		{"ds4_rope_attention_raw", &r.rawAttentionFn},
		{"ds4_router_hash", &r.routerHashFn},
		{"ds4_router_biased", &r.routerBiasedFn},
		{"ds4_swiglu", &r.swigluFn},
		{"ds4_sum_vectors", &r.sumVectorsFn},
		{"ds4_iq2_xxs_pair_selected_f32", &r.iq2SelectedFn},
		{"ds4_q2_k_selected_f32", &r.q2KSelectedFn},
		{"ds4_quant_q8_k", &r.quantQ8KFn},
		{"ds4_iq2_xxs_pair_selected_q8_k", &r.iq2SelectedQ8KFn},
		{"ds4_q2_k_selected_q8_k", &r.q2KSelectedQ8KFn},
		{"ds4_iq2_xxs_pair_selected_q8_1", &r.iq2SelectedQ81Fn},
		{"ds4_q2_k_selected_q8_1", &r.q2KSelectedQ81Fn},
		{"ds4_compressor_pool", &r.compressorPoolFn},
		{"ds4_compressor_attention", &r.compressorAttentionFn},
		{"ds4_compressor_indexer", &r.compressorIndexerFn},
		{"ds4_attention_mixed", &r.mixedAttentionFn},
		{"ds4_indexer_query", &r.indexerQueryFn},
		{"ds4_indexer_select", &r.indexerSelectFn},
	}
}

func (r *cudaFoundation) loadMathKernels() error {
	if r.matrixMod != 0 {
		return nil
	}
	compiled, err := nvrtcx.Compile(cudaMathSource,
		nvrtcx.WithName("ds4_math.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		return fmt.Errorf("cuda: compile math kernels: %w", err)
	}
	mod, err := compiled.Load(r.device)
	if err != nil {
		return fmt.Errorf("cuda: load math kernels: %w", err)
	}
	for _, kernel := range r.cudaKernels() {
		fn, err := mod.GetFunction(kernel.name)
		if err != nil {
			_ = mod.Unload()
			r.clearMathKernels()
			return fmt.Errorf("cuda: find kernel %q: %w", kernel.name, err)
		}
		*kernel.dst = fn
	}
	r.matrixMod = mod
	return nil
}

func (r *cudaFoundation) clearMathKernels() {
	for _, kernel := range r.cudaKernels() {
		*kernel.dst = 0
	}
}

// q80Row computes the first registered Q8_0 tensor row. The caller owns x and
// out; this foundation owns only the bounded, registered model row.
func (r *cudaFoundation) q80Row(x, out cudamem.Slice[float32]) error {
	if r.probePtr == 0 || r.probeCols == 0 {
		return fmt.Errorf("cuda: Q8_0 row is not bound")
	}
	if x.Len() != int(r.probeCols) || out.Len() != 1 {
		return fmt.Errorf("cuda: Q8_0 row input/output lengths %d/%d, want %d/1", x.Len(), out.Len(), r.probeCols)
	}
	compiled, err := nvrtcx.Compile(cudaProbeSource,
		nvrtcx.WithName("ds4_q8_0_row.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		return fmt.Errorf("cuda: compile Q8_0 row: %w", err)
	}
	return r.device.WithContext(func() error {
		mod, err := compiled.Load(r.device)
		if err != nil {
			return fmt.Errorf("cuda: load Q8_0 row: %w", err)
		}
		defer mod.Unload()
		fn, err := mod.GetFunction("ds4_q8_0_matvec")
		if err != nil {
			return fmt.Errorf("cuda: find Q8_0 row: %w", err)
		}
		if err := cudalaunch.Kernel(fn).Grid(1).Block(256).Launch(r.probePtr, x, out, r.probeCols, uint32(1)); err != nil {
			return fmt.Errorf("cuda: launch Q8_0 row: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize Q8_0 row: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) Close() error {
	if r == nil {
		return nil
	}
	var errs []error
	if r.quantQ80Mod != 0 && r.device != nil {
		// A foundation may also be closed directly by a qualification harness.
		// Establish completion before unloading the separately owned quantizer.
		err := r.device.WithContext(func() error {
			if err := driver.CtxSynchronize(); err != nil {
				return fmt.Errorf("cuda: synchronize Q8_0 quantizer: %w", err)
			}
			if err := r.quantQ80Mod.Unload(); err != nil {
				return err
			}
			r.quantQ80Mod, r.quantQ80Fn = 0, 0
			return nil
		})
		if err != nil {
			return fmt.Errorf("cuda: close Q8_0 quantizer: %w", err)
		}
	}
	if r.matrixMod != 0 && r.device != nil {
		err := r.device.WithContext(func() error { return r.matrixMod.Unload() })
		errs = append(errs, err)
		r.matrixMod = 0
		r.clearMathKernels()
	}
	if r.mapped.host != 0 && r.device != nil {
		err := r.device.WithContext(func() error { return driver.MemHostUnregister(r.mapped.host) })
		errs = append(errs, err)
		errs = append(errs, syscall.Munmap(r.mapped.view))
		r.mapped = cudaModelRange{}
	}
	if r.device != nil {
		errs = append(errs, r.device.Close())
		r.device = nil
	}
	if r.model != nil {
		errs = append(errs, syscall.Munmap(r.model))
		r.model = nil
	}
	r.gguf = nil
	if r.file != nil {
		errs = append(errs, r.file.Close())
		r.file = nil
	}
	return errors.Join(errs...)
}
