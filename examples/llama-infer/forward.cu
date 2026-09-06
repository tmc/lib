#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Warp-level sum reduction using shuffle intrinsics

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Weight storage traits.
//
// Only the weight operand's storage narrows. Activations stay float32 and every
// accumulation stays float32, so a half kernel differs from its float32
// counterpart in exactly one place: the precision the weight was rounded to
// before the multiply. That keeps quantization error separate from
// implementation error, and it is what the CPU reference in halfref_test.go
// models.
//
// The conversions are written against PTX and raw bits rather than
// cuda_fp16.h/cuda_bf16.h, because NVRTC compiles this source without an
// include path and those headers would not resolve. cvt.f32.f16 is the
// hardware conversion instruction, so IEEE half subnormals, infinities and
// NaNs convert exactly as the hardware defines. bfloat16 is the top 16 bits of
// a float32, so widening it is a shift.

__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

// WF32 keeps the original float4 loads, so the float32 kernels below generate
// the same code they did before weights became a template parameter.
struct WF32 {
    typedef float storage_t;
    typedef float4 vec_t;
    static __device__ __forceinline__ float4 load4(const storage_t* p, int i) {
        return reinterpret_cast<const float4*>(p)[i];
    }
    static __device__ __forceinline__ float load1(const storage_t* p, int i) {
        return p[i];
    }
};

// WF16 and WBF16 load four consecutive 16-bit weights as one 64-bit access, so
// the narrower storage keeps a vectorized load rather than falling back to four
// scalar ones. This needs the same 4-element alignment the float4 path already
// requires of a row.
struct WF16 {
    typedef unsigned short storage_t;
    typedef ushort4 vec_t;
    static __device__ __forceinline__ float4 load4(const storage_t* p, int i) {
        ushort4 v = reinterpret_cast<const ushort4*>(p)[i];
        return make_float4(f16_to_f32(v.x), f16_to_f32(v.y), f16_to_f32(v.z), f16_to_f32(v.w));
    }
    static __device__ __forceinline__ float load1(const storage_t* p, int i) {
        return f16_to_f32(p[i]);
    }
};

struct WBF16 {
    typedef unsigned short storage_t;
    typedef ushort4 vec_t;
    static __device__ __forceinline__ float4 load4(const storage_t* p, int i) {
        ushort4 v = reinterpret_cast<const ushort4*>(p)[i];
        return make_float4(bf16_to_f32(v.x), bf16_to_f32(v.y), bf16_to_f32(v.z), bf16_to_f32(v.w));
    }
    static __device__ __forceinline__ float load1(const storage_t* p, int i) {
        return bf16_to_f32(p[i]);
    }
};

// WEIGHT_KERNELS defines one extern "C" entry point per weight dtype for a
// kernel template. The float32 name is unsuffixed, so existing callers and the
// float32 reference path are unaffected; the suffixed names are what the half
// dispatch selects.
#define WEIGHT_KERNELS(NAME, PARAMS, ARGS)                                    \
    extern "C" __global__ void NAME PARAMS(WF32) { NAME##Impl<WF32> ARGS; }    \
    extern "C" __global__ void NAME##F16 PARAMS(WF16) { NAME##Impl<WF16> ARGS; } \
    extern "C" __global__ void NAME##BF16 PARAMS(WBF16) { NAME##Impl<WBF16> ARGS; }

// 1. Embedding lookup on GPU
template <class W>
__device__ __forceinline__ void EmbeddingLookupImpl(
    const typename W::storage_t* __restrict__ embed_table,
    const int* __restrict__ token_id,
    float* __restrict__ out,
    int hidden_dim
) {
    int tok = *token_id;
    const typename W::storage_t* row = embed_table + tok * hidden_dim;
    int dim4 = hidden_dim / 4;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim4; i += blockDim.x * gridDim.x) {
        reinterpret_cast<float4*>(out)[i] = W::load4(row, i);
    }
    for (int i = dim4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < hidden_dim; i += blockDim.x * gridDim.x) {
        out[i] = W::load1(row, i);
    }
}

#define EMBED_PARAMS(W) (const W::storage_t* __restrict__ embed_table, \
    const int* __restrict__ token_id, float* __restrict__ out, int hidden_dim)
WEIGHT_KERNELS(EmbeddingLookup, EMBED_PARAMS, (embed_table, token_id, out, hidden_dim))

// 2. RMSNorm with 128-bit vectorized loads and warp reduction
template <class W>
__device__ __forceinline__ void RMSNormImpl(
    const float* __restrict__ x,
    const typename W::storage_t* __restrict__ weight,
    float* __restrict__ out,
    int hidden_dim,
    float eps
) {
    __shared__ float s_sum_sq[32];
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    int dim4 = hidden_dim / 4;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim4; i += blockDim.x) {
        float4 v = x4[i];
        sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int i = dim4 * 4 + tid; i < hidden_dim; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = warp_sum(sum_sq);

    if (lane == 0) {
        s_sum_sq[wid] = sum_sq;
    }
    __syncthreads();

    float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum_sq[lane] : 0.0f;
    if (wid == 0) {
        block_sum = warp_sum(block_sum);
    }
    __shared__ float s_rsqrt;
    if (tid == 0) {
        s_rsqrt = rsqrtf(block_sum / float(hidden_dim) + eps);
    }
    __syncthreads();

    float rsqrt_val = s_rsqrt;
    float4* out4 = reinterpret_cast<float4*>(out);

    for (int i = tid; i < dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 wv = weight ? W::load4(weight, i) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        out4[i] = make_float4(
            xv.x * rsqrt_val * wv.x,
            xv.y * rsqrt_val * wv.y,
            xv.z * rsqrt_val * wv.z,
            xv.w * rsqrt_val * wv.w
        );
    }
    for (int i = dim4 * 4 + tid; i < hidden_dim; i += blockDim.x) {
        float w = weight ? W::load1(weight, i) : 1.0f;
        out[i] = x[i] * rsqrt_val * w;
    }
}

#define RMSNORM_PARAMS(W) (const float* __restrict__ x, \
    const W::storage_t* __restrict__ weight, float* __restrict__ out, \
    int hidden_dim, float eps)
WEIGHT_KERNELS(RMSNorm, RMSNORM_PARAMS, (x, weight, out, hidden_dim, eps))

// 3. Matrix-Vector Multiplication (Linear Projection) with 128-bit loads
template <class W>
__device__ __forceinline__ void LinearGEMVImpl(
    const float* __restrict__ x,
    const typename W::storage_t* __restrict__ w,
    const typename W::storage_t* __restrict__ bias,
    float* __restrict__ out,
    int in_dim,
    int out_dim
) {
    int row = blockIdx.x; // each block computes one output row
    if (row >= out_dim) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ float s_sum[32];

    const typename W::storage_t* w_row = w + row * in_dim;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int in_dim4 = in_dim / 4;

    float sum = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 wv = W::load4(w_row, i);
        sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }
    for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        sum += x[i] * W::load1(w_row, i);
    }
    sum = warp_sum(sum);

    if (lane == 0) {
        s_sum[wid] = sum;
    }
    __syncthreads();

    float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
    if (wid == 0) {
        block_sum = warp_sum(block_sum);
    }

    if (tid == 0) {
        float b = bias ? W::load1(bias, row) : 0.0f;
        out[row] = block_sum + b;
    }
}

#define GEMV_PARAMS(W) (const float* __restrict__ x, \
    const W::storage_t* __restrict__ w, const W::storage_t* __restrict__ bias, \
    float* __restrict__ out, int in_dim, int out_dim)
WEIGHT_KERNELS(LinearGEMV, GEMV_PARAMS, (x, w, bias, out, in_dim, out_dim))

// 4. Rotary Position Embedding (RoPE)
extern "C" __global__ void ApplyRoPE(
    float* __restrict__ q,
    float* __restrict__ k,
    int pos,
    int q_heads,
    int kv_heads,
    int head_dim,
    float theta
) {
    int h = blockIdx.x; // head index
    int i = threadIdx.x; // index in 0..head_dim/2
    if (i >= head_dim / 2) return;

    float freq = 1.0f / powf(theta, (2.0f * i) / float(head_dim));
    float val = float(pos) * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    if (h < q_heads) {
        float* head_q = q + h * head_dim;
        float v0 = head_q[i];
        float v1 = head_q[i + head_dim / 2];
        head_q[i] = v0 * cos_val - v1 * sin_val;
        head_q[i + head_dim / 2] = v0 * sin_val + v1 * cos_val;
    }

    if (h < kv_heads) {
        float* head_k = k + h * head_dim;
        float v0 = head_k[i];
        float v1 = head_k[i + head_dim / 2];
        head_k[i] = v0 * cos_val - v1 * sin_val;
        head_k[i + head_dim / 2] = v0 * sin_val + v1 * cos_val;
    }
}

// 5. Store current K and V into KV Cache with float4 vectorized writes
extern "C" __global__ void StoreKVCache(
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    int pos,
    int num_kv_heads,
    int head_dim
) {
    int kv_dim = num_kv_heads * head_dim;
    float4* k_dest4 = reinterpret_cast<float4*>(k_cache + pos * kv_dim);
    float4* v_dest4 = reinterpret_cast<float4*>(v_cache + pos * kv_dim);
    const float4* k4 = reinterpret_cast<const float4*>(k);
    const float4* v4 = reinterpret_cast<const float4*>(v);
    int dim4 = kv_dim / 4;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim4; i += blockDim.x * gridDim.x) {
        k_dest4[i] = k4[i];
        v_dest4[i] = v4[i];
    }
    for (int i = dim4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < kv_dim; i += blockDim.x * gridDim.x) {
        (k_cache + pos * kv_dim)[i] = k[i];
        (v_cache + pos * kv_dim)[i] = v[i];
    }
}

// 6. SwiGLU Activation: out = silu(gate) * up with float4 vectorization
extern "C" __global__ void SwiGLU(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int intermediate_dim
) {
    const float4* g4 = reinterpret_cast<const float4*>(gate);
    const float4* u4 = reinterpret_cast<const float4*>(up);
    float4* out4 = reinterpret_cast<float4*>(out);
    int dim4 = intermediate_dim / 4;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim4; i += blockDim.x * gridDim.x) {
        float4 g = g4[i];
        float4 u = u4[i];
        out4[i] = make_float4(
            (g.x / (1.0f + expf(-g.x))) * u.x,
            (g.y / (1.0f + expf(-g.y))) * u.y,
            (g.z / (1.0f + expf(-g.z))) * u.z,
            (g.w / (1.0f + expf(-g.w))) * u.w
        );
    }
    for (int i = dim4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < intermediate_dim; i += blockDim.x * gridDim.x) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

// 7. Vector Add (Residual Connection): x += res with float4 vectorization
extern "C" __global__ void AddResidual(
    float* __restrict__ x,
    const float* __restrict__ res,
    int dim
) {
    float4* x4 = reinterpret_cast<float4*>(x);
    const float4* res4 = reinterpret_cast<const float4*>(res);
    int dim4 = dim / 4;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim4; i += blockDim.x * gridDim.x) {
        float4 xv = x4[i];
        float4 rv = res4[i];
        x4[i] = make_float4(xv.x + rv.x, xv.y + rv.y, xv.z + rv.z, xv.w + rv.w);
    }
    for (int i = dim4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += blockDim.x * gridDim.x) {
        x[i] += res[i];
    }
}

// 8. E-Graph Extracted Fused Megakernels with float4 vectorization


template <class W>
__device__ __forceinline__ void FusedRMSNormQKVImpl(
    const float* __restrict__ x,
    const typename W::storage_t* __restrict__ weight,
    const typename W::storage_t* __restrict__ W_q,
    const typename W::storage_t* __restrict__ W_k,
    const typename W::storage_t* __restrict__ W_v,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int in_dim, int q_dim, int kv_dim, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ float s_sum[32];

    const float4* x4 = reinterpret_cast<const float4*>(x);
    int in_dim4 = in_dim / 4;

    // 1. Compute RMS scale across vector x
    float local_sq = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        local_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    local_sq = warp_sum(local_sq);
    if (lane == 0) s_sum[wid] = local_sq;
    __syncthreads();

    float total_sq = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
    if (wid == 0) total_sq = warp_sum(total_sq);
    if (tid == 0) s_sum[0] = rsqrtf(total_sq / (float)in_dim + eps);
    __syncthreads();
    float inv_rms = s_sum[0];

    // 2. Compute Q, K, V projections with exact RMSNorm scaling
    if (row < q_dim) {
        const typename W::storage_t* wq_row = W_q + row * in_dim;
        float sum = 0.0f;
        for (int i = tid; i < in_dim4; i += blockDim.x) {
            float4 xv = x4[i];
            float4 wv = weight ? W::load4(weight, i) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 norm_xv = make_float4(xv.x * inv_rms * wv.x, xv.y * inv_rms * wv.y, xv.z * inv_rms * wv.z, xv.w * inv_rms * wv.w);
            float4 wrv = W::load4(wq_row, i);
            sum += norm_xv.x * wrv.x + norm_xv.y * wrv.y + norm_xv.z * wrv.z + norm_xv.w * wrv.w;
        }
        sum = warp_sum(sum);
        if (lane == 0) s_sum[wid] = sum;
        __syncthreads();

        float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
        if (wid == 0) block_sum = warp_sum(block_sum);
        if (tid == 0) out_q[row] = block_sum;
    }
    if (row < kv_dim) {
        const typename W::storage_t* wk_row = W_k + row * in_dim;
        const typename W::storage_t* wv_row = W_v + row * in_dim;
        float sumK = 0.0f;
        float sumV = 0.0f;
        for (int i = tid; i < in_dim4; i += blockDim.x) {
            float4 xv = x4[i];
            float4 wv = weight ? W::load4(weight, i) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 norm_xv = make_float4(xv.x * inv_rms * wv.x, xv.y * inv_rms * wv.y, xv.z * inv_rms * wv.z, xv.w * inv_rms * wv.w);
            float4 wkv = W::load4(wk_row, i);
            float4 wvv = W::load4(wv_row, i);
            sumK += norm_xv.x * wkv.x + norm_xv.y * wkv.y + norm_xv.z * wkv.z + norm_xv.w * wkv.w;
            sumV += norm_xv.x * wvv.x + norm_xv.y * wvv.y + norm_xv.z * wvv.z + norm_xv.w * wvv.w;
        }
        sumK = warp_sum(sumK);
        sumV = warp_sum(sumV);
        if (lane == 0) {
            s_sum[wid] = sumK;
            s_sum[wid + 16] = sumV;
        }
        __syncthreads();

        float bk = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
        float bv = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane + 16] : 0.0f;
        if (wid == 0) {
            bk = warp_sum(bk);
            bv = warp_sum(bv);
        }
        if (tid == 0) {
            out_k[row] = bk;
            out_v[row] = bv;
        }
    }
}

#define FQKV_PARAMS(W) (const float* __restrict__ x, \
    const W::storage_t* __restrict__ weight, const W::storage_t* __restrict__ W_q, \
    const W::storage_t* __restrict__ W_k, const W::storage_t* __restrict__ W_v, \
    float* __restrict__ out_q, float* __restrict__ out_k, float* __restrict__ out_v, \
    int in_dim, int q_dim, int kv_dim, float eps)
WEIGHT_KERNELS(FusedRMSNormQKV, FQKV_PARAMS,
    (x, weight, W_q, W_k, W_v, out_q, out_k, out_v, in_dim, q_dim, kv_dim, eps))

template <class W>
__device__ __forceinline__ void FusedSwiGLUDownImpl(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const typename W::storage_t* __restrict__ W_down,
    float* __restrict__ out,
    int hidden_dim, int intermediate_dim
) {
    int row = blockIdx.x;
    if (row >= hidden_dim) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ float s_sum[32];

    const float4* g4 = reinterpret_cast<const float4*>(gate);
    const float4* u4 = reinterpret_cast<const float4*>(up);
    const typename W::storage_t* w_row = W_down + row * intermediate_dim;
    int inter_dim4 = intermediate_dim / 4;

    float sum = 0.0f;
    for (int i = tid; i < inter_dim4; i += blockDim.x) {
        float4 g = g4[i];
        float4 u = u4[i];
        float4 act = make_float4(
            (g.x / (1.0f + expf(-g.x))) * u.x,
            (g.y / (1.0f + expf(-g.y))) * u.y,
            (g.z / (1.0f + expf(-g.z))) * u.z,
            (g.w / (1.0f + expf(-g.w))) * u.w
        );
        float4 wv = W::load4(w_row, i);
        sum += act.x * wv.x + act.y * wv.y + act.z * wv.z + act.w * wv.w;
    }
    sum = warp_sum(sum);

    if (lane == 0) s_sum[wid] = sum;
    __syncthreads();

    float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
    if (wid == 0) block_sum = warp_sum(block_sum);

    if (tid == 0) out[row] = block_sum;
}

#define FSWIGLU_PARAMS(W) (const float* __restrict__ gate, const float* __restrict__ up, \
    const W::storage_t* __restrict__ W_down, float* __restrict__ out, \
    int hidden_dim, int intermediate_dim)
WEIGHT_KERNELS(FusedSwiGLUDown, FSWIGLU_PARAMS,
    (gate, up, W_down, out, hidden_dim, intermediate_dim))

// The batch kernels use grid.y for the token row. A single launch therefore
// consumes each shared weight matrix for every live request in the wave.
template <class W>
__device__ __forceinline__ void BatchEmbeddingLookupImpl(
    const typename W::storage_t* table, const int* tokens,
    float* out, int hidden_dim, int batch) {
    int b = blockIdx.y;
    if (b >= batch) return;
    int tok = tokens[b];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden_dim; i += blockDim.x * gridDim.x)
        out[b * hidden_dim + i] = W::load1(table, tok * hidden_dim + i);
}

#define BEMBED_PARAMS(W) (const W::storage_t* table, const int* tokens, \
    float* out, int hidden_dim, int batch)
WEIGHT_KERNELS(BatchEmbeddingLookup, BEMBED_PARAMS, (table, tokens, out, hidden_dim, batch))

template <class W>
__device__ __forceinline__ void BatchRMSNormImpl(
    const float* x, const typename W::storage_t* weight,
    float* out, int hidden_dim, int batch, float eps) {
    int b = blockIdx.y;
    if (b >= batch) return;
    x += b * hidden_dim;
    out += b * hidden_dim;
    __shared__ float sums[32];
    int tid = threadIdx.x, lane = tid % WARP_SIZE, wid = tid / WARP_SIZE;
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) sum += x[i] * x[i];
    sum = warp_sum(sum);
    if (lane == 0) sums[wid] = sum;
    __syncthreads();
    float total = tid < blockDim.x / WARP_SIZE ? sums[lane] : 0.0f;
    if (wid == 0) total = warp_sum(total);
    if (tid == 0) sums[0] = rsqrtf(total / float(hidden_dim) + eps);
    __syncthreads();
    for (int i = tid; i < hidden_dim; i += blockDim.x)
        out[i] = x[i] * sums[0] * (weight ? W::load1(weight, i) : 1.0f);
}

#define BNORM_PARAMS(W) (const float* x, const W::storage_t* weight, \
    float* out, int hidden_dim, int batch, float eps)
WEIGHT_KERNELS(BatchRMSNorm, BNORM_PARAMS, (x, weight, out, hidden_dim, batch, eps))

template <class W>
__device__ __forceinline__ void BatchLinearGEMVImpl(
    const float* x, const typename W::storage_t* w,
    const typename W::storage_t* bias, float* out, int in_dim, int out_dim, int batch) {
    int row = blockIdx.x, b = blockIdx.y;
    if (row >= out_dim || b >= batch) return;
    x += b * in_dim;
    __shared__ float sums[32];
    int tid = threadIdx.x, lane = tid % WARP_SIZE, wid = tid / WARP_SIZE;
    float sum = 0.0f;
    for (int i = tid; i < in_dim; i += blockDim.x) sum += x[i] * W::load1(w, row * in_dim + i);
    sum = warp_sum(sum);
    if (lane == 0) sums[wid] = sum;
    __syncthreads();
    float total = tid < blockDim.x / WARP_SIZE ? sums[lane] : 0.0f;
    if (wid == 0) total = warp_sum(total);
    if (tid == 0) out[b * out_dim + row] = total + (bias ? W::load1(bias, row) : 0.0f);
}

#define BGEMV_PARAMS(W) (const float* x, const W::storage_t* w, \
    const W::storage_t* bias, float* out, int in_dim, int out_dim, int batch)
WEIGHT_KERNELS(BatchLinearGEMV, BGEMV_PARAMS, (x, w, bias, out, in_dim, out_dim, batch))

extern "C" __global__ void BatchSwiGLU(const float* gate, const float* up,
    float* out, int intermediate_dim, int batch) {
    int b = blockIdx.y;
    if (b >= batch) return;
    int base = b * intermediate_dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < intermediate_dim; i += blockDim.x * gridDim.x) {
        float g = gate[base + i];
        out[base + i] = (g / (1.0f + expf(-g))) * up[base + i];
    }
}

extern "C" __global__ void BatchAddResidual(float* x, const float* residual,
    int dim, int batch) {
    int b = blockIdx.y;
    if (b >= batch) return;
    int base = b * dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += blockDim.x * gridDim.x)
        x[base + i] += residual[base + i];
}

template <class W>
__device__ __forceinline__ void BatchFusedRMSNormQKVImpl(
    const float* x, const typename W::storage_t* weight,
    const typename W::storage_t* wq, const typename W::storage_t* wk,
    const typename W::storage_t* wv, float* q, float* k, float* v,
    int in_dim, int q_dim, int kv_dim, int batch, float eps) {
    int out = blockIdx.x, b = blockIdx.y;
    if (b >= batch) return;
    const float* bx = x + b * in_dim;
    __shared__ float sums[32];
    int tid = threadIdx.x, lane = tid % WARP_SIZE, wid = tid / WARP_SIZE;
    float sq = 0.0f;
    for (int i = tid; i < in_dim; i += blockDim.x) sq += bx[i] * bx[i];
    sq = warp_sum(sq);
    if (lane == 0) sums[wid] = sq;
    __syncthreads();
    float total = tid < blockDim.x / WARP_SIZE ? sums[lane] : 0.0f;
    if (wid == 0) total = warp_sum(total);
    if (tid == 0) sums[0] = rsqrtf(total / float(in_dim) + eps);
    __syncthreads();
    float inv = sums[0], qs = 0.0f, ks = 0.0f, vs = 0.0f;
    for (int i = tid; i < in_dim; i += blockDim.x) {
        float nx = bx[i] * inv * (weight ? W::load1(weight, i) : 1.0f);
        if (out < q_dim) qs += nx * W::load1(wq, out * in_dim + i);
        if (out < kv_dim) {
            ks += nx * W::load1(wk, out * in_dim + i);
            vs += nx * W::load1(wv, out * in_dim + i);
        }
    }
    qs = warp_sum(qs); ks = warp_sum(ks); vs = warp_sum(vs);
    if (lane == 0) { sums[wid] = qs; sums[wid + 8] = ks; sums[wid + 16] = vs; }
    __syncthreads();
    float qt = tid < blockDim.x / WARP_SIZE ? sums[lane] : 0.0f;
    float kt = tid < blockDim.x / WARP_SIZE ? sums[lane + 8] : 0.0f;
    float vt = tid < blockDim.x / WARP_SIZE ? sums[lane + 16] : 0.0f;
    if (wid == 0) { qt = warp_sum(qt); kt = warp_sum(kt); vt = warp_sum(vt); }
    if (tid == 0) {
        if (out < q_dim) q[b * q_dim + out] = qt;
        if (out < kv_dim) { k[b * kv_dim + out] = kt; v[b * kv_dim + out] = vt; }
    }
}

#define BFQKV_PARAMS(W) (const float* x, const W::storage_t* weight, \
    const W::storage_t* wq, const W::storage_t* wk, const W::storage_t* wv, \
    float* q, float* k, float* v, int in_dim, int q_dim, int kv_dim, int batch, float eps)
WEIGHT_KERNELS(BatchFusedRMSNormQKV, BFQKV_PARAMS,
    (x, weight, wq, wk, wv, q, k, v, in_dim, q_dim, kv_dim, batch, eps))

template <class W>
__device__ __forceinline__ void BatchFusedSwiGLUDownImpl(
    const float* gate, const float* up,
    const typename W::storage_t* down, float* out,
    int hidden_dim, int intermediate_dim, int batch) {
    int row = blockIdx.x, b = blockIdx.y;
    if (row >= hidden_dim || b >= batch) return;
    __shared__ float sums[32];
    int tid = threadIdx.x, lane = tid % WARP_SIZE, wid = tid / WARP_SIZE;
    int base = b * intermediate_dim;
    float sum = 0.0f;
    for (int i = tid; i < intermediate_dim; i += blockDim.x) {
        float g = gate[base + i];
        sum += (g / (1.0f + expf(-g))) * up[base + i] * W::load1(down, row * intermediate_dim + i);
    }
    sum = warp_sum(sum);
    if (lane == 0) sums[wid] = sum;
    __syncthreads();
    float total = tid < blockDim.x / WARP_SIZE ? sums[lane] : 0.0f;
    if (wid == 0) total = warp_sum(total);
    if (tid == 0) out[b * hidden_dim + row] = total;
}

#define BFSWIGLU_PARAMS(W) (const float* gate, const float* up, \
    const W::storage_t* down, float* out, int hidden_dim, int intermediate_dim, int batch)
WEIGHT_KERNELS(BatchFusedSwiGLUDown, BFSWIGLU_PARAMS,
    (gate, up, down, out, hidden_dim, intermediate_dim, batch))

extern "C" __global__ void BatchArgmax(const float* logits, int* tokens,
    int vocab, int batch) {
    int b = blockIdx.x;
    if (b >= batch) return;
    __shared__ float values[256];
    __shared__ int indices[256];
    int tid = threadIdx.x;
    const float* row = logits + b * vocab;
    float best = -3.402823466e+38F;
    int index = -1;
    for (int i = tid; i < vocab; i += blockDim.x) {
        if (row[i] > best || (row[i] == best && (index < 0 || i < index))) {
            best = row[i]; index = i;
        }
    }
    values[tid] = best; indices[tid] = index;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset >>= 1) {
        if (tid < offset) {
            float other = values[tid + offset];
            int otherIndex = indices[tid + offset];
            if (other > values[tid] ||
                (other == values[tid] && otherIndex >= 0 && (indices[tid] < 0 || otherIndex < indices[tid]))) {
                values[tid] = other; indices[tid] = otherIndex;
            }
        }
        __syncthreads();
    }
    if (tid == 0) tokens[b] = indices[0] < 0 ? 0 : indices[0];
}


// Add one projection bias to every prompt row after matrix multiplication.
template <class W>
__device__ __forceinline__ void BatchAddBiasImpl(
    float* out, const typename W::storage_t* bias, int dim, int rows) {
    int row = blockIdx.y;
    if (row >= rows) return;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out[row * dim + i] += W::load1(bias, i);
}

#define BBIAS_PARAMS(W) (float* out, const W::storage_t* bias, int dim, int rows)
WEIGHT_KERNELS(BatchAddBias, BBIAS_PARAMS, (out, bias, dim, rows))
