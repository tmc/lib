#include <metal_stdlib>
using namespace metal;

#ifndef METAL_SIMD_SUM_DEFINED
#define METAL_SIMD_SUM_DEFINED
inline float simd_sum(float val) {
    #pragma unroll
    for (ushort offset = 16; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}
#endif

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

    float rms = s_rms;
    for (uint i = tid; i < dim; i += 128) {
        out_normed[i] = x[i] * rms * float(weight[i]);
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
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x[base+0] * float(w4[0]) +
                   x[base+1] * float(w4[1]) +
                   x[base+2] * float(w4[2]) +
                   x[base+3] * float(w4[3]);
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
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x[base+0] * float(w4[0]) +
                   x[base+1] * float(w4[1]) +
                   x[base+2] * float(w4[2]) +
                   x[base+3] * float(w4[3]);
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
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 64) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x[base+0] * float(w4[0]) +
                   x[base+1] * float(w4[1]) +
                   x[base+2] * float(w4[2]) +
                   x[base+3] * float(w4[3]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 64) {
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
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 64) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x[base+0] * float(w4[0]) +
                   x[base+1] * float(w4[1]) +
                   x[base+2] * float(w4[2]) +
                   x[base+3] * float(w4[3]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 64) {
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

// 5H. Batched embedding lookup over half weights
kernel void BatchedEmbeddingLookupH(
    device const half* embed_table     [[buffer(0)]],
    device const int* tokens           [[buffer(1)]],
    device float* x_out                [[buffer(2)]],
    constant uint& seq_len             [[buffer(3)]],
    constant uint& dim                 [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint total = seq_len * dim;
    if (gid < total) {
        uint token_idx = gid / dim;
        uint dim_idx = gid % dim;
        int tok = tokens[token_idx];
        x_out[gid] = float(embed_table[tok * dim + dim_idx]);
    }
}

// 6H. Batched RMSNorm over half norm weights
kernel void BatchedRMSNormH(
    device const float* x_in           [[buffer(0)]],
    device const half* weight          [[buffer(1)]],
    device float* x_out                [[buffer(2)]],
    constant uint& seq_len             [[buffer(3)]],
    constant uint& dim                 [[buffer(4)]],
    constant float& eps                [[buffer(5)]],
    uint token_idx                     [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (token_idx >= seq_len) return;
    device const float* x_row = x_in + token_idx * dim;
    device float* out_row = x_out + token_idx * dim;

    uint dim4 = dim / 4;
    device const float4* x4 = (device const float4*)x_row;

    float local_sq = 0.0f;
    for (uint i = tid; i < dim4; i += 128) {
        float4 v = x4[i];
        local_sq += dot(v, v);
    }
    for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
        float v = x_row[i];
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

    float rms = s_rms;
    for (uint i = tid; i < dim; i += 128) {
        out_row[i] = x_row[i] * rms * float(weight[i]);
    }
}

// 7H. Batched GEMM over half weights
kernel void BatchedGEMMH(
    device const float* X              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* Y                    [[buffer(2)]],
    constant uint& seq_len             [[buffer(3)]],
    constant uint& in_dim              [[buffer(4)]],
    constant uint& out_dim             [[buffer(5)]],
    uint group_idx                     [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint token_idx = group_idx / out_dim;
    uint out_col = group_idx % out_dim;
    if (token_idx < seq_len && out_col < out_dim) {
        device const float* x_row = X + token_idx * in_dim;
        device const half* w_row = W + out_col * in_dim;
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x_row[base+0] * float(w4[0]) +
                   x_row[base+1] * float(w4[1]) +
                   x_row[base+2] * float(w4[2]) +
                   x_row[base+3] * float(w4[3]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x_row[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[4];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) {
            Y[token_idx * out_dim + out_col] = block_sum;
        }
    }
}

// 8H. Batched GEMM Residual over half weights
kernel void BatchedGEMMResidualH(
    device const float* X              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* Y                    [[buffer(2)]],
    constant uint& seq_len             [[buffer(3)]],
    constant uint& in_dim              [[buffer(4)]],
    constant uint& out_dim             [[buffer(5)]],
    uint group_idx                     [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint token_idx = group_idx / out_dim;
    uint out_col = group_idx % out_dim;
    if (token_idx < seq_len && out_col < out_dim) {
        device const float* x_row = X + token_idx * in_dim;
        device const half* w_row = W + out_col * in_dim;
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x_row[base+0] * float(w4[0]) +
                   x_row[base+1] * float(w4[1]) +
                   x_row[base+2] * float(w4[2]) +
                   x_row[base+3] * float(w4[3]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x_row[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[4];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) {
            Y[token_idx * out_dim + out_col] += block_sum;
        }
    }
}
