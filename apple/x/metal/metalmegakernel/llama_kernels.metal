#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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

// 1. Embedding lookup
kernel void EmbeddingLookup(
    device const float* embed_table    [[buffer(0)]],
    device float* x_out                [[buffer(1)]],
    constant uint& token_id            [[buffer(2)]],
    constant uint& dim                 [[buffer(3)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid < dim) {
        x_out[tid] = embed_table[token_id * dim + tid];
    }
}

// 2. Fast RMSNorm
kernel void ComputeRMSNorm(
    device const float* x              [[buffer(0)]],
    device const float* weight         [[buffer(1)]],
    device float* out_normed           [[buffer(2)]],
    constant uint& dim                 [[buffer(3)]],
    constant float& eps                [[buffer(4)]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint dim4 = dim / 4;
    device const float4* x4 = (device const float4*)x;
    device const float4* w4 = (device const float4*)weight;
    device float4* out4 = (device float4*)out_normed;

    float local_sq = 0.0f;
    for (uint i = tid; i < dim4; i += 128) {
        float4 v = x4[i];
        local_sq += dot(v, v);
    }
    for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
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
    for (uint i = tid; i < dim4; i += 128) {
        out4[i] = x4[i] * rms * w4[i];
    }
    for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
        out_normed[i] = x[i] * rms * weight[i];
    }
}

// 3. Fast GEMV Matrix-Vector Multiplication (Row-parallel with float4 SIMD reduction)
kernel void FastGEMV(
    device const float* x              [[buffer(0)]], // [InDim]
    device const float* W              [[buffer(1)]], // [OutDim, InDim]
    device float* y                    [[buffer(2)]], // [OutDim]
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const float* w_row = W + row * in_dim;
        uint in_dim4 = in_dim / 4;
        device const float4* w_row4 = (device const float4*)w_row;
        device const float4* x4 = (device const float4*)x;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            acc += dot(x4[i], w_row4[i]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x[i] * w_row[i];
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

// 3b. Fast GEMV Matrix-Vector Multiplication with Residual Accumulation
kernel void FastGEMVResidual(
    device const float* x              [[buffer(0)]],
    device const float* W              [[buffer(1)]],
    device float* y_inout              [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const float* w_row = W + row * in_dim;
        uint in_dim4 = in_dim / 4;
        device const float4* w_row4 = (device const float4*)w_row;
        device const float4* x4 = (device const float4*)x;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            acc += dot(x4[i], w_row4[i]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x[i] * w_row[i];
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

// 4. Fast GEMV 64-thread Variant
kernel void FastGEMV64(
    device const float* x              [[buffer(0)]],
    device const float* W              [[buffer(1)]],
    device float* y                    [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const float* w_row = W + row * in_dim;
        uint in_dim4 = in_dim / 4;
        device const float4* w_row4 = (device const float4*)w_row;
        device const float4* x4 = (device const float4*)x;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 64) {
            acc += dot(x4[i], w_row4[i]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 64) {
            acc += x[i] * w_row[i];
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

// 4b. Fast GEMV 64-thread Variant with Residual Accumulation
kernel void FastGEMVResidual64(
    device const float* x              [[buffer(0)]],
    device const float* W              [[buffer(1)]],
    device float* y_inout              [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const float* w_row = W + row * in_dim;
        uint in_dim4 = in_dim / 4;
        device const float4* w_row4 = (device const float4*)w_row;
        device const float4* x4 = (device const float4*)x;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 64) {
            acc += dot(x4[i], w_row4[i]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 64) {
            acc += x[i] * w_row[i];
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

// 5. Fast RoPE & KV Store
kernel void ApplyRoPEStoreKV(
    device float* q               [[buffer(0)]],
    device float* k               [[buffer(1)]],
    device const float* v         [[buffer(2)]],
    device float* k_cache         [[buffer(3)]],
    device float* v_cache         [[buffer(4)]],
    constant uint& pos            [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& q_heads        [[buffer(7)]],
    constant uint& kv_heads       [[buffer(8)]],
    constant float& theta         [[buffer(9)]],
    uint tid                      [[thread_position_in_grid]]
) {
    uint total_q_pairs = (q_heads * head_dim) / 2;
    if (tid < total_q_pairs) {
        uint h = tid / (head_dim / 2);
        uint i = tid % (head_dim / 2);

        float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
        float val = float(pos) * freq;
        float cos_val = cos(val);
        float sin_val = sin(val);

        uint idx0 = h * head_dim + i;
        uint idx1 = h * head_dim + i + head_dim / 2;

        float v0 = q[idx0];
        float v1 = q[idx1];
        q[idx0] = v0 * cos_val - v1 * sin_val;
        q[idx1] = v0 * sin_val + v1 * cos_val;
    }

    uint total_kv_pairs = (kv_heads * head_dim) / 2;
    if (tid < total_kv_pairs) {
        uint h = tid / (head_dim / 2);
        uint i = tid % (head_dim / 2);

        float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
        float val = float(pos) * freq;
        float cos_val = cos(val);
        float sin_val = sin(val);

        uint idx0 = h * head_dim + i;
        uint idx1 = h * head_dim + i + head_dim / 2;

        float v0 = k[idx0];
        float v1 = k[idx1];
        float rot0 = v0 * cos_val - v1 * sin_val;
        float rot1 = v0 * sin_val + v1 * cos_val;
        k[idx0] = rot0;
        k[idx1] = rot1;

        uint kv_dim = kv_heads * head_dim;
        k_cache[pos * kv_dim + idx0] = rot0;
        k_cache[pos * kv_dim + idx1] = rot1;
        v_cache[pos * kv_dim + idx0] = v[idx0];
        v_cache[pos * kv_dim + idx1] = v[idx1];
    }
}

// 6. Fast SiLU Activation Fused with Up Elementwise Multiply (out[i] = silu(gate[i]) * up[i])
kernel void FastSiLUMul(
    device const float* gate      [[buffer(0)]],
    device const float* up        [[buffer(1)]],
    device float* out             [[buffer(2)]],
    constant uint& hidden_dim     [[buffer(3)]],
    uint tid                      [[thread_position_in_grid]]
) {
    uint dim4 = hidden_dim / 4;
    device const float4* g4 = (device const float4*)gate;
    device const float4* u4 = (device const float4*)up;
    device float4* o4 = (device float4*)out;

    if (tid < dim4) {
        float4 g = g4[tid];
        float4 silu = g / (float4(1.0f) + exp(-g));
        o4[tid] = silu * u4[tid];
    }
}

// 7. Fused RMSNorm + QKV Projection (Single pass over x, computes Q, K, V in parallel)
kernel void FusedRMSNormQKV(
    device const float* x              [[buffer(0)]],
    device const float* norm_weight    [[buffer(1)]],
    device const float* WQ             [[buffer(2)]],
    device const float* WK             [[buffer(3)]],
    device const float* WV             [[buffer(4)]],
    device float* q                    [[buffer(5)]],
    device float* k                    [[buffer(6)]],
    device float* v                    [[buffer(7)]],
    constant uint& dim                 [[buffer(8)]],
    constant uint& q_dim               [[buffer(9)]],
    constant uint& kv_dim              [[buffer(10)]],
    constant float& eps                [[buffer(11)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint dim4 = dim / 4;
    device const float4* x4 = (device const float4*)x;

    float local_sq = 0.0f;
    for (uint i = tid; i < dim4; i += 128) {
        float4 val = x4[i];
        local_sq += dot(val, val);
    }
    for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
        float val = x[i];
        local_sq += val * val;
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

    // Compute Q for this row
    if (row < q_dim) {
        device const float* wq_row = WQ + row * dim;
        device const float4* wq4 = (device const float4*)wq_row;
        device const float4* nw4 = (device const float4*)norm_weight;
        float acc = 0.0f;
        for (uint i = tid; i < dim4; i += 128) {
            float4 normed = x4[i] * rms * nw4[i];
            acc += dot(normed, wq4[i]);
        }
        for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
            float normed = x[i] * rms * norm_weight[i];
            acc += normed * wq_row[i];
        }
        acc = simd_sum(acc);
        threadgroup float s_acc[4];
        if (simd_lane == 0) s_acc[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float blk = (simd_id == 0 && simd_lane < 4) ? s_acc[simd_lane] : 0.0f;
        if (simd_id == 0) blk = simd_sum(blk);
        if (tid == 0) q[row] = blk;
    }

    // Compute K and V if row < kv_dim
    if (row < kv_dim) {
        device const float* wk_row = WK + row * dim;
        device const float4* wk4 = (device const float4*)wk_row;
        device const float4* nw4 = (device const float4*)norm_weight;
        float acc_k = 0.0f;
        for (uint i = tid; i < dim4; i += 128) {
            float4 normed = x4[i] * rms * nw4[i];
            acc_k += dot(normed, wk4[i]);
        }
        for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
            float normed = x[i] * rms * norm_weight[i];
            acc_k += normed * wk_row[i];
        }
        acc_k = simd_sum(acc_k);

        device const float* wv_row = WV + row * dim;
        device const float4* wv4 = (device const float4*)wv_row;
        float acc_v = 0.0f;
        for (uint i = tid; i < dim4; i += 128) {
            float4 normed = x4[i] * rms * nw4[i];
            acc_v += dot(normed, wv4[i]);
        }
        for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
            float normed = x[i] * rms * norm_weight[i];
            acc_v += normed * wv_row[i];
        }
        acc_v = simd_sum(acc_v);

        threadgroup float s_acc_k[4];
        threadgroup float s_acc_v[4];
        if (simd_lane == 0) {
            s_acc_k[simd_id] = acc_k;
            s_acc_v[simd_id] = acc_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float blk_k = (simd_id == 0 && simd_lane < 4) ? s_acc_k[simd_lane] : 0.0f;
        if (simd_id == 0) blk_k = simd_sum(blk_k);
        float blk_v = (simd_id == 0 && simd_lane < 4) ? s_acc_v[simd_lane] : 0.0f;
        if (simd_id == 0) blk_v = simd_sum(blk_v);
        if (tid == 0) {
            k[row] = blk_k;
            v[row] = blk_v;
        }
    }
}

// 8. Fused SwiGLU + Down Projection (Single pass over Gate and Up activations)
kernel void FusedSwiGLUDown(
    device const float* gate           [[buffer(0)]],
    device const float* up             [[buffer(1)]],
    device const float* WDown          [[buffer(2)]],
    device float* x_inout              [[buffer(3)]],
    constant uint& dim                 [[buffer(4)]],
    constant uint& inter_dim           [[buffer(5)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < dim) {
        device const float* wd_row = WDown + row * inter_dim;
        uint inter4 = inter_dim / 4;
        device const float4* wd4 = (device const float4*)wd_row;
        device const float4* g4 = (device const float4*)gate;
        device const float4* u4 = (device const float4*)up;

        float acc = 0.0f;
        for (uint i = tid; i < inter4; i += 128) {
            float4 g = g4[i];
            float4 silu = g / (float4(1.0f) + exp(-g));
            float4 act = silu * u4[i];
            acc += dot(act, wd4[i]);
        }
        for (uint i = inter4 * 4 + tid; i < inter_dim; i += 128) {
            float g = gate[i];
            float silu = g / (1.0f + exp(-g));
            float act = silu * up[i];
            acc += act * wd_row[i];
        }
        acc = simd_sum(acc);
        threadgroup float s_acc[4];
        if (simd_lane == 0) s_acc[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float blk = (simd_id == 0 && simd_lane < 4) ? s_acc[simd_lane] : 0.0f;
        if (simd_id == 0) blk = simd_sum(blk);
        if (tid == 0) x_inout[row] += blk;
    }
}
