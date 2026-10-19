// BF16-weight kernels. This file is compiled after forward.cu, in the same
// module, and uses its helpers. Weights are raw bfloat16 bits; activations,
// accumulation and outputs stay float32.

// bf16_to_f32 widens a bfloat16 bit pattern to float32. Widening is exact.
__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

// WeightConvertProbeBF16 widens n bfloat16 values, for checking bf16_to_f32
// against the host on every bit pattern.
extern "C" __global__ void WeightConvertProbeBF16(
    const unsigned short* __restrict__ in,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = bf16_to_f32(in[i]);
}

// LinearGEMVBF16 is LinearGEMV with bfloat16 weights. It keeps LinearGEMV's
// thread partition, vector-path gate and expression order, so that on widened
// weights the two can be compared bit for bit. A row of in_dim % 4 == 0
// bfloat16 values starts 8-byte aligned, as a ushort4 load requires.
extern "C" __global__ void LinearGEMVBF16(
    const float* __restrict__ x,
    const unsigned short* __restrict__ w,
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

    const unsigned short* w_row = w + (size_t)row * in_dim;
    const ushort4* w_row4 = reinterpret_cast<const ushort4*>(w_row);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int in_dim4 = rows_vectorizable(in_dim) ? in_dim / 4 : 0;

    float sum = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        ushort4 wb = w_row4[i];
        float4 wv = make_float4(bf16_to_f32(wb.x), bf16_to_f32(wb.y), bf16_to_f32(wb.z), bf16_to_f32(wb.w));
        sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }
    for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        sum += x[i] * bf16_to_f32(w_row[i]);
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

// EmbeddingLookupBF16 widens row *token_id of a bfloat16 table into out.
// The row is read as scalars; one row per token is not worth vectorizing.
extern "C" __global__ void EmbeddingLookupBF16(
    const unsigned short* __restrict__ embed_table,
    const int* __restrict__ token_id,
    float* __restrict__ out,
    int hidden_dim
) {
    const unsigned short* src = embed_table + (size_t)(*token_id) * hidden_dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden_dim; i += blockDim.x * gridDim.x) {
        out[i] = bf16_to_f32(src[i]);
    }
}

// FusedRMSNormQKVBF16 is FusedRMSNormQKV with bfloat16 projection matrices.
// Norm weights, biases, activations and outputs stay float32, and the thread
// partition, vector-path gate and expression order match FusedRMSNormQKV so
// that on widened weights the two can be compared bit for bit.
extern "C" __global__ void FusedRMSNormQKVBF16(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const unsigned short* __restrict__ W_q,
    const unsigned short* __restrict__ W_k,
    const unsigned short* __restrict__ W_v,
    const float* __restrict__ b_q,
    const float* __restrict__ b_k,
    const float* __restrict__ b_v,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int in_dim, int q_dim, int kv_dim, float eps
) {
    int row = blockIdx.x;
    const unsigned short* W;
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

    const unsigned short* w_row = W + (size_t)row * in_dim;
    int in_dim4 = rows_vectorizable(in_dim) ? in_dim / 4 : 0;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* g4 = reinterpret_cast<const float4*>(weight);
    const ushort4* w_row4 = reinterpret_cast<const ushort4*>(w_row);

    float sq = 0.0f;
    float dot = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 gv = weight ? g4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        ushort4 wqkv = w_row4[i];
        float4 wv = make_float4(bf16_to_f32(wqkv.x), bf16_to_f32(wqkv.y), bf16_to_f32(wqkv.z), bf16_to_f32(wqkv.w));
        sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
        dot += xv.x * gv.x * wv.x + xv.y * gv.y * wv.y + xv.z * gv.z * wv.z + xv.w * gv.w * wv.w;
    }
    for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        float xv = x[i];
        float gv = weight ? weight[i] : 1.0f;
        sq += xv * xv;
        dot += xv * gv * bf16_to_f32(w_row[i]);
    }
    // Both sums are complete only in thread 0.
    sq = block_reduce_sum(sq, s_sq);
    dot = block_reduce_sum(dot, s_dot);
    if (tid == 0) {
        out[row] = dot * rsqrtf(sq / float(in_dim) + eps) + (bias ? bias[row] : 0.0f);
    }
}

// FusedSwiGLUDownBF16 is FusedSwiGLUDown with bfloat16 down-projection
// weights. Activations and outputs stay float32, and the thread partition,
// vector-path gate and expression order match FusedSwiGLUDown so that on
// widened weights the two can be compared bit for bit.
extern "C" __global__ void FusedSwiGLUDownBF16(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const unsigned short* __restrict__ W_down,
    float* __restrict__ out,
    int hidden_dim, int intermediate_dim
) {
    int row = blockIdx.x;
    if (row >= hidden_dim) return;

    int tid = threadIdx.x;
    __shared__ float s_sum[32];

    const unsigned short* w_row = W_down + (size_t)row * intermediate_dim;
    const float4* g4 = reinterpret_cast<const float4*>(gate);
    const float4* u4 = reinterpret_cast<const float4*>(up);
    const ushort4* w_row4 = reinterpret_cast<const ushort4*>(w_row);
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
        ushort4 wd = w_row4[i];
        float4 wv = make_float4(bf16_to_f32(wd.x), bf16_to_f32(wd.y), bf16_to_f32(wd.z), bf16_to_f32(wd.w));
        sum += act.x * wv.x + act.y * wv.y + act.z * wv.z + act.w * wv.w;
    }
    for (int i = inter_dim4 * 4 + tid; i < intermediate_dim; i += blockDim.x) {
        float g = gate[i];
        sum += (g / (1.0f + expf(-g))) * up[i] * bf16_to_f32(w_row[i]);
    }
    sum = block_reduce_sum(sum, s_sum);
    if (tid == 0) out[row] = sum;
}
