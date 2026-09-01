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

// 1. Embedding lookup on GPU
extern "C" __global__ void EmbeddingLookup(
    const float* __restrict__ embed_table,
    const int* __restrict__ token_id,
    float* __restrict__ out,
    int hidden_dim
) {
    int tok = *token_id;
    const float* src = embed_table + tok * hidden_dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden_dim; i += blockDim.x * gridDim.x) {
        out[i] = src[i];
    }
}

// 2. RMSNorm with warp reduction
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

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
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
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        out[i] = x[i] * rsqrt_val * w;
    }
}

// 3. Matrix-Vector Multiplication (Linear Projection) with warp reduction
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
    float sum = 0.0f;
    for (int i = tid; i < in_dim; i += blockDim.x) {
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

// 5. Store current K and V into KV Cache
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

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kv_dim; i += blockDim.x * gridDim.x) {
        k_dest[i] = k[i];
        v_dest[i] = v[i];
    }
}

// 6. SwiGLU Activation: out = silu(gate) * up
extern "C" __global__ void SwiGLU(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int intermediate_dim
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < intermediate_dim; i += blockDim.x * gridDim.x) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

// 7. Vector Add (Residual Connection): x += res
extern "C" __global__ void AddResidual(
    float* __restrict__ x,
    const float* __restrict__ res,
    int dim
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += blockDim.x * gridDim.x) {
        x[i] += res[i];
    }
}

// 9. E-Graph Extracted Fused Megakernels

extern "C" __global__ void FusedRMSNormQKV(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ W_q,
    const float* __restrict__ W_k,
    const float* __restrict__ W_v,
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

    if (row < q_dim) {
        const float* w_row = W_q + row * in_dim;
        float sum = 0.0f;
        for (int i = tid; i < in_dim; i += blockDim.x) {
            sum += (x[i] * weight[i]) * w_row[i];
        }
        sum = warp_sum(sum);
        if (lane == 0) s_sum[wid] = sum;
        __syncthreads();

        float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
        if (wid == 0) block_sum = warp_sum(block_sum);
        if (tid == 0) out_q[row] = block_sum;
    }
    if (row < kv_dim) {
        const float* wk_row = W_k + row * in_dim;
        const float* wv_row = W_v + row * in_dim;
        float sumK = 0.0f;
        float sumV = 0.0f;
        for (int i = tid; i < in_dim; i += blockDim.x) {
            float normed = x[i] * weight[i];
            sumK += normed * wk_row[i];
            sumV += normed * wv_row[i];
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
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ float s_sum[32];

    const float* w_row = W_down + row * intermediate_dim;
    float sum = 0.0f;
    for (int i = tid; i < intermediate_dim; i += blockDim.x) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        float act = silu * up[i];
        sum += act * w_row[i];
    }
    sum = warp_sum(sum);

    if (lane == 0) s_sum[wid] = sum;
    __syncthreads();

    float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
    if (wid == 0) block_sum = warp_sum(block_sum);

    if (tid == 0) out[row] = block_sum;
}
