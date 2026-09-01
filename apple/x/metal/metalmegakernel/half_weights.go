//go:build darwin

package metalmegakernel

// HalfWeightKernelsMSL contains decoder kernel variants that read
// half-precision weights while keeping float32 activations and
// accumulation. Only kernels that touch weights have variants; RoPE,
// SiLU-mul, and attention operate on activations and are shared with
// the float32 path.
const HalfWeightKernelsMSL = `
// 1H. Embedding lookup over half weights
kernel void EmbeddingLookupH(
    device const half* embed_table     [[buffer(0)]],
    device float* x_out                [[buffer(1)]],
    constant uint& token_id            [[buffer(2)]],
    constant uint& dim                 [[buffer(3)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid < dim) {
        x_out[tid] = float(embed_table[token_id * dim + tid]);
    }
}

// 2H. Fast RMSNorm with half norm weights
kernel void ComputeRMSNormH(
    device const float* x              [[buffer(0)]],
    device const half* weight          [[buffer(1)]],
    device float* out_normed           [[buffer(2)]],
    constant uint& dim                 [[buffer(3)]],
    constant float& eps                [[buffer(4)]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += 128) {
        float v = x[i];
        local_sq += v * v;
    }
    local_sq = simd_sum(local_sq);
    threadgroup float s_sum[4];
    if (simd_lane == 0) s_sum[simd_id] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) total_sq = simd_sum(total_sq);

    threadgroup float s_rms;
    if (tid == 0) {
        s_rms = rsqrt(total_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < dim; i += 128) {
        out_normed[i] = x[i] * s_rms * float(weight[i]);
    }
}

// 3H. Fast GEMV over half weights, float accumulation
kernel void FastGEMVH(
    device const float* x              [[buffer(0)]], // [InDim]
    device const half* W               [[buffer(1)]], // [OutDim, InDim]
    device float* y                    [[buffer(2)]], // [OutDim]
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const half* w_row = W + row * in_dim;
        float acc = 0.0f;
        // half4 vector loads halve the instruction count on the hot path.
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;
        device const float4* x4 = (device const float4*)x;
        for (uint i = tid; i < in_dim4; i += 128) {
            float4 wv = float4(w_row4[i]);
            float4 xv = x4[i];
            acc += dot(xv, wv);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[4];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) y[row] = block_sum;
    }
}

// 4H. Fast GEMV with residual accumulation over half weights
kernel void FastGEMVResidualH(
    device const float* x              [[buffer(0)]], // [InDim]
    device const half* W               [[buffer(1)]], // [OutDim, InDim]
    device float* y_inout              [[buffer(2)]], // [OutDim]
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const half* w_row = W + row * in_dim;
        float acc = 0.0f;
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;
        device const float4* x4 = (device const float4*)x;
        for (uint i = tid; i < in_dim4; i += 128) {
            float4 wv = float4(w_row4[i]);
            float4 xv = x4[i];
            acc += dot(xv, wv);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[4];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) y_inout[row] += block_sum;
    }
}

// 5H. Fused RMSNorm + GEMV over half weights (e-graph extraction target)
kernel void FusedRMSNormGEMVH(
    device const float* x              [[buffer(0)]],
    device const half* weight          [[buffer(1)]],
    device const half* W_proj          [[buffer(2)]],
    device float* out                  [[buffer(3)]],
    constant uint& dim                 [[buffer(4)]],
    constant uint& out_dim             [[buffer(5)]],
    constant float& eps                [[buffer(6)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += 128) {
        float v = x[i];
        local_sq += v * v;
    }
    local_sq = simd_sum(local_sq);
    threadgroup float s_sum[4];
    if (simd_lane == 0) s_sum[simd_id] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) total_sq = simd_sum(total_sq);

    threadgroup float s_rms;
    if (tid == 0) {
        s_rms = rsqrt(total_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < out_dim) {
        device const half* w_row = W_proj + row * dim;
        float acc = 0.0f;
        for (uint i = tid; i < dim; i += 128) {
            float norm_x = (x[i] * s_rms) * float(weight[i]);
            acc += norm_x * float(w_row[i]);
        }
        acc = simd_sum(acc);
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) out[row] = block_sum;
    }
}
`
