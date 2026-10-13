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

// Row-major matrices are read with float4 loads only when the row length is a
// multiple of 4: then every row base is 16-byte aligned (allocations are). For
// other lengths, odd rows start misaligned, so the whole row is read as
// scalars. A scalar tail alone is not enough.
__device__ __forceinline__ bool rows_vectorizable(int row_len) {
    return (row_len & 3) == 0;
}

// block_reduce_sum returns the sum of val over the block in thread 0. blockDim.x must
// be a multiple of WARP_SIZE. scratch holds 32 floats owned by this call; the
// caller must not reuse it before a following __syncthreads.
__device__ __forceinline__ float block_reduce_sum(float val, float* scratch) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warp_sum(val);
    if (lane == 0) scratch[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? scratch[lane] : 0.0f;
    if (wid == 0) val = warp_sum(val);
    return val;
}

// 1. Embedding lookup on GPU
extern "C" __global__ void EmbeddingLookup(
    const float* __restrict__ embed_table,
    const int* __restrict__ token_id,
    float* __restrict__ out,
    int hidden_dim
) {
    int tok = *token_id;
    const float* src = embed_table + tok * hidden_dim;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* out4 = reinterpret_cast<float4*>(out);
    int dim4 = rows_vectorizable(hidden_dim) ? hidden_dim / 4 : 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim4; i += blockDim.x * gridDim.x) {
        out4[i] = src4[i];
    }
    for (int i = dim4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < hidden_dim; i += blockDim.x * gridDim.x) {
        out[i] = src[i];
    }
}

// 2. RMSNorm with 128-bit float4 vectorized loads and warp reduction
extern "C" __global__ void RMSNorm(
    const float* __restrict__ x,
    const float* __restrict__ weight,
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
    const float4* w4 = weight ? reinterpret_cast<const float4*>(weight) : nullptr;

    for (int i = tid; i < dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 wv = w4 ? w4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        out4[i] = make_float4(
            xv.x * rsqrt_val * wv.x,
            xv.y * rsqrt_val * wv.y,
            xv.z * rsqrt_val * wv.z,
            xv.w * rsqrt_val * wv.w
        );
    }
    for (int i = dim4 * 4 + tid; i < hidden_dim; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        out[i] = x[i] * rsqrt_val * w;
    }
}

// 3. Matrix-Vector Multiplication (Linear Projection) with 128-bit float4 loads
extern "C" __global__ void LinearGEMV(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
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

    const float* w_row = w + row * in_dim;
    const float4* w_row4 = reinterpret_cast<const float4*>(w_row);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int in_dim4 = rows_vectorizable(in_dim) ? in_dim / 4 : 0;

    float sum = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 wv = w_row4[i];
        sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }
    for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        sum += x[i] * w_row[i];
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
        float b = bias ? bias[row] : 0.0f;
        out[row] = block_sum + b;
    }
}

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
    float* k_dest = k_cache + pos * kv_dim;
    float* v_dest = v_cache + pos * kv_dim;
    float4* k_dest4 = reinterpret_cast<float4*>(k_dest);
    float4* v_dest4 = reinterpret_cast<float4*>(v_dest);
    const float4* k4 = reinterpret_cast<const float4*>(k);
    const float4* v4 = reinterpret_cast<const float4*>(v);
    int dim4 = rows_vectorizable(kv_dim) ? kv_dim / 4 : 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim4; i += blockDim.x * gridDim.x) {
        k_dest4[i] = k4[i];
        v_dest4[i] = v4[i];
    }
    for (int i = dim4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < kv_dim; i += blockDim.x * gridDim.x) {
        k_dest[i] = k[i];
        v_dest[i] = v[i];
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

// 8. Fused kernels. These are not qualified on a device; gpu_model.go
// refuses -fuse-egraph until fused_parity_cuda_test.go passes on one.

// FusedRMSNormQKV computes one row of [Q; K; V] = W · RMSNorm(x) + bias per
// block, with grid = q_dim + 2*kv_dim: rows [0, q_dim) are Q, the next kv_dim
// are K, the last kv_dim are V. Each block recomputes the RMSNorm scale over x
// (in_dim floats) and applies it after the dot product, which reassociates the
// unfused RMSNorm then LinearGEMV sequence. weight and bias pointers may be
// null. blockDim.x must be a multiple of WARP_SIZE.
extern "C" __global__ void FusedRMSNormQKV(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ W_q,
    const float* __restrict__ W_k,
    const float* __restrict__ W_v,
    const float* __restrict__ b_q,
    const float* __restrict__ b_k,
    const float* __restrict__ b_v,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int in_dim, int q_dim, int kv_dim, float eps
) {
    int row = blockIdx.x;
    const float* W;
    const float* bias;
    float* out;
    if (row < q_dim) {
        W = W_q; bias = b_q; out = out_q;
    } else if (row < q_dim + kv_dim) {
        row -= q_dim;
        W = W_k; bias = b_k; out = out_k;
    } else if (row < q_dim + 2 * kv_dim) {
        row -= q_dim + kv_dim;
        W = W_v; bias = b_v; out = out_v;
    } else {
        return;
    }
    int tid = threadIdx.x;
    __shared__ float s_sq[32];
    __shared__ float s_dot[32];

    const float* w_row = W + row * in_dim;
    int in_dim4 = rows_vectorizable(in_dim) ? in_dim / 4 : 0;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* g4 = reinterpret_cast<const float4*>(weight);
    const float4* w_row4 = reinterpret_cast<const float4*>(w_row);

    float sq = 0.0f;
    float dot = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 gv = weight ? g4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 wv = w_row4[i];
        sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
        dot += xv.x * gv.x * wv.x + xv.y * gv.y * wv.y + xv.z * gv.z * wv.z + xv.w * gv.w * wv.w;
    }
    for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        float xv = x[i];
        float gv = weight ? weight[i] : 1.0f;
        sq += xv * xv;
        dot += xv * gv * w_row[i];
    }
    // Both sums are complete only in thread 0.
    sq = block_reduce_sum(sq, s_sq);
    dot = block_reduce_sum(dot, s_dot);
    if (tid == 0) {
        out[row] = dot * rsqrtf(sq / float(in_dim) + eps) + (bias ? bias[row] : 0.0f);
    }
}

// FusedSwiGLUDown computes out[row] = W_down[row] · (silu(gate) * up) per
// block, with grid = hidden_dim. Each block recomputes the activation.
// blockDim.x must be a multiple of WARP_SIZE.
extern "C" __global__ void FusedSwiGLUDown(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ W_down,
    float* __restrict__ out,
    int hidden_dim, int intermediate_dim
) {
    int row = blockIdx.x;
    if (row >= hidden_dim) return;

    int tid = threadIdx.x;
    __shared__ float s_sum[32];

    const float* w_row = W_down + row * intermediate_dim;
    const float4* g4 = reinterpret_cast<const float4*>(gate);
    const float4* u4 = reinterpret_cast<const float4*>(up);
    const float4* w_row4 = reinterpret_cast<const float4*>(w_row);
    int inter_dim4 = rows_vectorizable(intermediate_dim) ? intermediate_dim / 4 : 0;

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
        float4 wv = w_row4[i];
        sum += act.x * wv.x + act.y * wv.y + act.z * wv.z + act.w * wv.w;
    }
    for (int i = inter_dim4 * 4 + tid; i < intermediate_dim; i += blockDim.x) {
        float g = gate[i];
        sum += (g / (1.0f + expf(-g))) * up[i] * w_row[i];
    }
    sum = block_reduce_sum(sum, s_sum);
    if (tid == 0) out[row] = sum;
}
