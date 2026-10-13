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
