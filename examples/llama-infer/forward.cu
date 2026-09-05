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
    const float4* src4 = reinterpret_cast<const float4*>(embed_table + tok * hidden_dim);
    float4* out4 = reinterpret_cast<float4*>(out);
    int dim4 = hidden_dim / 4;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim4; i += blockDim.x * gridDim.x) {
        out4[i] = src4[i];
    }
    for (int i = dim4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < hidden_dim; i += blockDim.x * gridDim.x) {
        out[i] = (embed_table + tok * hidden_dim)[i];
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

    const float4* w_row4 = reinterpret_cast<const float4*>(w + row * in_dim);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int in_dim4 = in_dim / 4;

    float sum = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 wv = w_row4[i];
        sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }
    for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        sum += x[i] * (w + row * in_dim)[i];
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

    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* w4 = weight ? reinterpret_cast<const float4*>(weight) : nullptr;
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
        const float4* w_row4 = reinterpret_cast<const float4*>(W_q + row * in_dim);
        float sum = 0.0f;
        for (int i = tid; i < in_dim4; i += blockDim.x) {
            float4 xv = x4[i];
            float4 wv = w4 ? w4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 norm_xv = make_float4(xv.x * inv_rms * wv.x, xv.y * inv_rms * wv.y, xv.z * inv_rms * wv.z, xv.w * inv_rms * wv.w);
            float4 wrv = w_row4[i];
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
        const float4* wk_row4 = reinterpret_cast<const float4*>(W_k + row * in_dim);
        const float4* wv_row4 = reinterpret_cast<const float4*>(W_v + row * in_dim);
        float sumK = 0.0f;
        float sumV = 0.0f;
        for (int i = tid; i < in_dim4; i += blockDim.x) {
            float4 xv = x4[i];
            float4 wv = w4 ? w4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 norm_xv = make_float4(xv.x * inv_rms * wv.x, xv.y * inv_rms * wv.y, xv.z * inv_rms * wv.z, xv.w * inv_rms * wv.w);
            float4 wkv = wk_row4[i];
            float4 wvv = wv_row4[i];
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

    const float4* g4 = reinterpret_cast<const float4*>(gate);
    const float4* u4 = reinterpret_cast<const float4*>(up);
    const float4* w_row4 = reinterpret_cast<const float4*>(W_down + row * intermediate_dim);
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
        float4 wv = w_row4[i];
        sum += act.x * wv.x + act.y * wv.y + act.z * wv.z + act.w * wv.w;
    }
    sum = warp_sum(sum);

    if (lane == 0) s_sum[wid] = sum;
    __syncthreads();

    float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
    if (wid == 0) block_sum = warp_sum(block_sum);

    if (tid == 0) out[row] = block_sum;
}

// The batch kernels use grid.y for the token row. A single launch therefore
// consumes each shared weight matrix for every live request in the wave.
extern "C" __global__ void BatchEmbeddingLookup(const float* table, const int* tokens,
    float* out, int hidden_dim, int batch) {
    int b = blockIdx.y;
    if (b >= batch) return;
    int tok = tokens[b];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden_dim; i += blockDim.x * gridDim.x)
        out[b * hidden_dim + i] = table[tok * hidden_dim + i];
}

extern "C" __global__ void BatchRMSNorm(const float* x, const float* weight,
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
        out[i] = x[i] * sums[0] * (weight ? weight[i] : 1.0f);
}

extern "C" __global__ void BatchLinearGEMV(const float* x, const float* w,
    const float* bias, float* out, int in_dim, int out_dim, int batch) {
    int row = blockIdx.x, b = blockIdx.y;
    if (row >= out_dim || b >= batch) return;
    x += b * in_dim;
    __shared__ float sums[32];
    int tid = threadIdx.x, lane = tid % WARP_SIZE, wid = tid / WARP_SIZE;
    float sum = 0.0f;
    for (int i = tid; i < in_dim; i += blockDim.x) sum += x[i] * w[row * in_dim + i];
    sum = warp_sum(sum);
    if (lane == 0) sums[wid] = sum;
    __syncthreads();
    float total = tid < blockDim.x / WARP_SIZE ? sums[lane] : 0.0f;
    if (wid == 0) total = warp_sum(total);
    if (tid == 0) out[b * out_dim + row] = total + (bias ? bias[row] : 0.0f);
}

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

extern "C" __global__ void BatchFusedRMSNormQKV(const float* x, const float* weight,
    const float* wq, const float* wk, const float* wv, float* q, float* k, float* v,
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
        float nx = bx[i] * inv * (weight ? weight[i] : 1.0f);
        if (out < q_dim) qs += nx * wq[out * in_dim + i];
        if (out < kv_dim) {
            ks += nx * wk[out * in_dim + i];
            vs += nx * wv[out * in_dim + i];
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

extern "C" __global__ void BatchFusedSwiGLUDown(const float* gate, const float* up,
    const float* down, float* out, int hidden_dim, int intermediate_dim, int batch) {
    int row = blockIdx.x, b = blockIdx.y;
    if (row >= hidden_dim || b >= batch) return;
    __shared__ float sums[32];
    int tid = threadIdx.x, lane = tid % WARP_SIZE, wid = tid / WARP_SIZE;
    int base = b * intermediate_dim;
    float sum = 0.0f;
    for (int i = tid; i < intermediate_dim; i += blockDim.x) {
        float g = gate[base + i];
        sum += (g / (1.0f + expf(-g))) * up[base + i] * down[row * intermediate_dim + i];
    }
    sum = warp_sum(sum);
    if (lane == 0) sums[wid] = sum;
    __syncthreads();
    float total = tid < blockDim.x / WARP_SIZE ? sums[lane] : 0.0f;
    if (wid == 0) total = warp_sum(total);
    if (tid == 0) out[b * hidden_dim + row] = total;
}

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
extern "C" __global__ void BatchAddBias(float* out, const float* bias, int dim, int rows) {
    int row = blockIdx.y;
    if (row >= rows) return;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out[row * dim + i] += bias[i];
}
