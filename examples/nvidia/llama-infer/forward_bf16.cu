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

// bf16_pair_to_f32 unpacks two packed bfloat16 values from a 32-bit word
// (low 16 bits in .x, high 16 bits in .y).
__device__ __forceinline__ float2 bf16_pair_to_f32(unsigned int u) {
    float lo = __uint_as_float(u << 16);
    float hi = __uint_as_float(u & 0xffff0000u);
    return make_float2(lo, hi);
}

// rows_vectorizable8 reports whether every row of in_dim bfloat16 elements starts
// on a 16-byte boundary (as uint4 loads require) when row 0 does.
__device__ __forceinline__ bool rows_vectorizable8(int in_dim) {
    return (in_dim & 7) == 0;
}

// LinearGEMVBF16Wide computes two output rows per CTA using 16-byte (uint4 = 8
// bfloat16 elements) weight loads when (in_dim & 7) == 0, reusing the x activation
// loads across both rows, and falls back to scalar loads for unaligned tails or
// when in_dim is not a multiple of 8. Launch with gridDim.x = (out_dim + 1) / 2.
extern "C" __global__ void LinearGEMVBF16Wide(
    const float* __restrict__ x,
    const unsigned short* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int in_dim,
    int out_dim
) {
    int row0 = blockIdx.x * 2;
    if (row0 >= out_dim) return;
    int row1 = row0 + 1;
    bool has_row1 = (row1 < out_dim);

    int tid = threadIdx.x;
    __shared__ float s_sum0[32];
    __shared__ float s_sum1[32];

    const unsigned short* w_row0 = w + (size_t)row0 * in_dim;
    const unsigned short* w_row1 = has_row1 ? (w + (size_t)row1 * in_dim) : w_row0;
    const uint4* w_row0_8 = reinterpret_cast<const uint4*>(w_row0);
    const uint4* w_row1_8 = reinterpret_cast<const uint4*>(w_row1);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int in_dim8 = rows_vectorizable8(in_dim) ? in_dim / 8 : 0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int i = tid; i < in_dim8; i += blockDim.x) {
        float4 x_lo = x4[2 * i];
        float4 x_hi = x4[2 * i + 1];

        uint4 wb0 = w_row0_8[i];
        float2 p00 = bf16_pair_to_f32(wb0.x);
        float2 p01 = bf16_pair_to_f32(wb0.y);
        float2 p02 = bf16_pair_to_f32(wb0.z);
        float2 p03 = bf16_pair_to_f32(wb0.w);
        sum0 += x_lo.x * p00.x + x_lo.y * p00.y + x_lo.z * p01.x + x_lo.w * p01.y
              + x_hi.x * p02.x + x_hi.y * p02.y + x_hi.z * p03.x + x_hi.w * p03.y;

        if (has_row1) {
            uint4 wb1 = w_row1_8[i];
            float2 p10 = bf16_pair_to_f32(wb1.x);
            float2 p11 = bf16_pair_to_f32(wb1.y);
            float2 p12 = bf16_pair_to_f32(wb1.z);
            float2 p13 = bf16_pair_to_f32(wb1.w);
            sum1 += x_lo.x * p10.x + x_lo.y * p10.y + x_lo.z * p11.x + x_lo.w * p11.y
                  + x_hi.x * p12.x + x_hi.y * p12.y + x_hi.z * p13.x + x_hi.w * p13.y;
        }
    }
    for (int i = in_dim8 * 8 + tid; i < in_dim; i += blockDim.x) {
        float xv = x[i];
        sum0 += xv * bf16_to_f32(w_row0[i]);
        if (has_row1) {
            sum1 += xv * bf16_to_f32(w_row1[i]);
        }
    }

    sum0 = block_reduce_sum(sum0, s_sum0);
    if (has_row1) {
        sum1 = block_reduce_sum(sum1, s_sum1);
    }

    if (tid == 0) {
        float b0 = bias ? bias[row0] : 0.0f;
        out[row0] = sum0 + b0;
        if (has_row1) {
            float b1 = bias ? bias[row1] : 0.0f;
            out[row1] = sum1 + b1;
        }
    }
}

