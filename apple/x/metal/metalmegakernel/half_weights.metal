#include <metal_stdlib>
using namespace metal;

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
    device const float* x              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* y                    [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const half* w_row = W + row * in_dim;
        uint in_dim8 = in_dim / 8;
        device const half8* w_row8 = (device const half8*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim8; i += 128) {
            half8 w8 = w_row8[i];
            uint base = i * 8;
            acc += x[base+0] * float(w8[0]) +
                   x[base+1] * float(w8[1]) +
                   x[base+2] * float(w8[2]) +
                   x[base+3] * float(w8[3]) +
                   x[base+4] * float(w8[4]) +
                   x[base+5] * float(w8[5]) +
                   x[base+6] * float(w8[6]) +
                   x[base+7] * float(w8[7]);
        }
        for (uint i = in_dim8 * 8 + tid; i < in_dim; i += 128) {
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

// 3bH. Fast GEMV with Residual Accumulation over half weights
kernel void FastGEMVResidualH(
    device const float* x              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* y_inout              [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const half* w_row = W + row * in_dim;
        uint in_dim8 = in_dim / 8;
        device const half8* w_row8 = (device const half8*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim8; i += 128) {
            half8 w8 = w_row8[i];
            uint base = i * 8;
            acc += x[base+0] * float(w8[0]) +
                   x[base+1] * float(w8[1]) +
                   x[base+2] * float(w8[2]) +
                   x[base+3] * float(w8[3]) +
                   x[base+4] * float(w8[4]) +
                   x[base+5] * float(w8[5]) +
                   x[base+6] * float(w8[6]) +
                   x[base+7] * float(w8[7]);
        }
        for (uint i = in_dim8 * 8 + tid; i < in_dim; i += 128) {
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

// 4H. 64-thread variants over half weights
kernel void FastGEMV64H(
    device const float* x              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* y                    [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const half* w_row = W + row * in_dim;
        uint in_dim8 = in_dim / 8;
        device const half8* w_row8 = (device const half8*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim8; i += 64) {
            half8 w8 = w_row8[i];
            uint base = i * 8;
            acc += x[base+0] * float(w8[0]) +
                   x[base+1] * float(w8[1]) +
                   x[base+2] * float(w8[2]) +
                   x[base+3] * float(w8[3]) +
                   x[base+4] * float(w8[4]) +
                   x[base+5] * float(w8[5]) +
                   x[base+6] * float(w8[6]) +
                   x[base+7] * float(w8[7]);
        }
        for (uint i = in_dim8 * 8 + tid; i < in_dim; i += 64) {
            acc += x[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[2];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 2) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) y[row] = block_sum;
    }
}

kernel void FastGEMVResidual64H(
    device const float* x              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* y_inout              [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const half* w_row = W + row * in_dim;
        uint in_dim8 = in_dim / 8;
        device const half8* w_row8 = (device const half8*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim8; i += 64) {
            half8 w8 = w_row8[i];
            uint base = i * 8;
            acc += x[base+0] * float(w8[0]) +
                   x[base+1] * float(w8[1]) +
                   x[base+2] * float(w8[2]) +
                   x[base+3] * float(w8[3]) +
                   x[base+4] * float(w8[4]) +
                   x[base+5] * float(w8[5]) +
                   x[base+6] * float(w8[6]) +
                   x[base+7] * float(w8[7]);
        }
        for (uint i = in_dim8 * 8 + tid; i < in_dim; i += 64) {
            acc += x[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[2];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 2) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) y_inout[row] += block_sum;
    }
}
