// Inter-GEMM kernels of one fp32 Llama-style decoder layer, for the
// decoder-layer harness in decoder_native_test.go. The GEMMs between them are
// cuBLAS calls. Every kernel here is named chain_*, which is how the span
// accounting tells a chain kernel from a GEMM kernel.
//
// Layouts follow column-major cuBLAS: a [T][N] activation holds token t's N
// values contiguously. Per-head tensors are [H][T][hd] (q, attention output)
// and [H][S][hd] (key and value caches, S = L + T positions).
//
// Compile without --use_fast_math: the harness's error bounds assume the
// accurate expf, cosf, sinf, rsqrtf and IEEE division.

#define BLOCK 256

extern "C" __global__ void chain_embed(const int *ids, const float *table, float *x, int T, int D) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)T * D) return;
  size_t t = i / D, d = i % D;
  x[i] = table[(size_t)ids[t] * D + d];
}

// One block per row.
extern "C" __global__ void chain_rmsnorm(const float *x, const float *w, float *y, int D, float eps) {
  __shared__ float part[BLOCK];
  const float *row = x + (size_t)blockIdx.x * D;
  float *out = y + (size_t)blockIdx.x * D;
  float s = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) s += row[d] * row[d];
  part[threadIdx.x] = s;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) part[threadIdx.x] += part[threadIdx.x + k];
    __syncthreads();
  }
  float r = rsqrtf(part[0] / (float)D + eps);
  for (int d = threadIdx.x; d < D; d += blockDim.x) out[d] = row[d] * r * w[d];
}

// Rotary embedding of q and k (rotate-half pairing) at positions L..L+T-1,
// writing q to [H][T][hd] and k and v into the caches.
extern "C" __global__ void chain_rope_kv(const float *qkv, const float *inv_freq, float *q, float *kc, float *vc,
    int T, int L, int S, int H, int hd) {
  int half = hd / 2;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)T * H * half) return;
  int i = idx % half;
  int h = (idx / half) % H;
  int t = idx / ((size_t)half * H);
  int D = H * hd;
  const float *row = qkv + (size_t)t * 3 * D;
  int pos = L + t;
  float th = (float)pos * inv_freq[i];
  float c = cosf(th), s = sinf(th);
  float q1 = row[h * hd + i], q2 = row[h * hd + i + half];
  float *qo = q + ((size_t)h * T + t) * hd;
  qo[i] = q1 * c - q2 * s;
  qo[i + half] = q2 * c + q1 * s;
  float k1 = row[D + h * hd + i], k2 = row[D + h * hd + i + half];
  float *ko = kc + ((size_t)h * S + pos) * hd;
  ko[i] = k1 * c - k2 * s;
  ko[i + half] = k2 * c + k1 * s;
  float *vo = vc + ((size_t)h * S + pos) * hd;
  vo[i] = row[2 * D + h * hd + i];
  vo[i + half] = row[2 * D + h * hd + i + half];
}

// One block per (head, token) row of scores [H][T][S]. With causal set, token
// t attends to positions 0..L+t; the harness's planted defect clears it.
extern "C" __global__ void chain_softmax(const float *scores, float *probs, int T, int L, int S, float scale, int causal) {
  __shared__ float part[BLOCK];
  int t = blockIdx.x % T;
  const float *row = scores + (size_t)blockIdx.x * S;
  float *out = probs + (size_t)blockIdx.x * S;
  int allowed = causal ? L + t + 1 : S;
  float m = __int_as_float(0xff800000); // -inf
  for (int s = threadIdx.x; s < allowed; s += blockDim.x) m = fmaxf(m, scale * row[s]);
  part[threadIdx.x] = m;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) part[threadIdx.x] = fmaxf(part[threadIdx.x], part[threadIdx.x + k]);
    __syncthreads();
  }
  m = part[0];
  __syncthreads();
  float sum = 0.f;
  for (int s = threadIdx.x; s < allowed; s += blockDim.x) sum += expf(scale * row[s] - m);
  part[threadIdx.x] = sum;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) part[threadIdx.x] += part[threadIdx.x + k];
    __syncthreads();
  }
  sum = part[0];
  for (int s = threadIdx.x; s < S; s += blockDim.x) out[s] = s < allowed ? expf(scale * row[s] - m) / sum : 0.f;
}

// [H][T][hd] to [T][H*hd].
extern "C" __global__ void chain_merge_heads(const float *o, float *attn, int T, int H, int hd) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int D = H * hd;
  if (i >= (size_t)T * D) return;
  size_t t = i / D;
  int rem = i % D, h = rem / hd, d = rem % hd;
  attn[i] = o[((size_t)h * T + t) * hd + d];
}

extern "C" __global__ void chain_residual_add(const float *a, const float *b, float *out, int n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) out[i] = a[i] + b[i];
}

// gu is [T][2F], gate then up.
extern "C" __global__ void chain_silu_mul(const float *gu, float *h, int T, int F) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)T * F) return;
  size_t t = i / F, j = i % F;
  float g = gu[t * 2 * F + j], u = gu[t * 2 * F + F + j];
  h[i] = g / (1.f + expf(-g)) * u;
}
