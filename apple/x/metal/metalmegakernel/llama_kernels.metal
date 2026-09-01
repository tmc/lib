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
    threadgroup float s_sum[5];
    if (simd_lane == 0) s_sum[simd_id] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) total_sq = simd_sum(total_sq);

    if (tid == 0) {
        s_sum[4] = rsqrt(total_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = s_sum[4];
    for (uint i = tid; i < dim4; i += 128) {
        out4[i] = x4[i] * rms * w4[i];
    }
    for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
        out_normed[i] = x[i] * rms * weight[i];
    }
}

// 3. Fast GEMV Matrix-Vector Multiplication
kernel void FastGEMV(
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
        device const float4* x4 = (device const float4*)x;
        device const float4* w4 = (device const float4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            acc += dot(x4[i], w4[i]);
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

// 3b. Fast GEMV with Residual Accumulation
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
        device const float4* x4 = (device const float4*)x;
        device const float4* w4 = (device const float4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            acc += dot(x4[i], w4[i]);
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

// 4. FastGEMV 64-thread variants
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
        device const float4* x4 = (device const float4*)x;
        device const float4* w4 = (device const float4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 64) {
            acc += dot(x4[i], w4[i]);
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
        device const float4* x4 = (device const float4*)x;
        device const float4* w4 = (device const float4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 64) {
            acc += dot(x4[i], w4[i]);
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

// 5. Fast Rotary Position Embedding (RoPE) and KV-cache store
kernel void ApplyRoPEStoreKV(
    device float* q                    [[buffer(0)]],
    device float* k                    [[buffer(1)]],
    device float* v                    [[buffer(2)]],
    device float* k_cache              [[buffer(3)]],
    device float* v_cache              [[buffer(4)]],
    constant uint& pos                 [[buffer(5)]],
    constant uint& head_dim            [[buffer(6)]],
    constant uint& num_q_heads         [[buffer(7)]],
    constant uint& num_kv_heads        [[buffer(8)]],
    constant float& theta              [[buffer(9)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint total_q_dim = num_q_heads * head_dim;
    uint total_kv_dim = num_kv_heads * head_dim;
    uint half_head_dim = head_dim / 2;

    if (gid < total_q_dim) {
        uint h = gid / head_dim;
        uint i = gid % head_dim;
        if (i < half_head_dim) {
            float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
            float val = float(pos) * freq;
            float cos_val = cos(val);
            float sin_val = sin(val);

            uint idx0 = h * head_dim + i;
            uint idx1 = h * head_dim + i + half_head_dim;
            float q0 = q[idx0];
            float q1 = q[idx1];
            q[idx0] = q0 * cos_val - q1 * sin_val;
            q[idx1] = q0 * sin_val + q1 * cos_val;
        }
    }

    if (gid < total_kv_dim) {
        uint h = gid / head_dim;
        uint i = gid % head_dim;
        if (i < half_head_dim) {
            float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
            float val = float(pos) * freq;
            float cos_val = cos(val);
            float sin_val = sin(val);

            uint idx0 = h * head_dim + i;
            uint idx1 = h * head_dim + i + half_head_dim;
            float k0 = k[idx0];
            float k1 = k[idx1];
            float rk0 = k0 * cos_val - k1 * sin_val;
            float rk1 = k0 * sin_val + k1 * cos_val;
            k[idx0] = rk0;
            k[idx1] = rk1;
            k_cache[pos * total_kv_dim + idx0] = rk0;
            k_cache[pos * total_kv_dim + idx1] = rk1;
        }
        v_cache[pos * total_kv_dim + gid] = v[gid];
    }
}

// 6. Fast SiLU Multiplicative Activation
kernel void FastSiLUMul(
    device const float* gate           [[buffer(0)]],
    device const float* up             [[buffer(1)]],
    device float* out_act              [[buffer(2)]],
    constant uint& hidden_dim          [[buffer(3)]],
    uint tid                           [[thread_position_in_grid]]
) {
    uint hidden_dim4 = hidden_dim / 4;
    device const float4* g4 = (device const float4*)gate;
    device const float4* u4 = (device const float4*)up;
    device float4* o4 = (device float4*)out_act;

    if (tid < hidden_dim4) {
        float4 g = g4[tid];
        float4 silu_g = g / (float4(1.0f) + exp(-g));
        o4[tid] = silu_g * u4[tid];
    }
}

// 7. E-Graph Fused RMSNorm + QKV Projection Megakernel
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
    threadgroup float s_sum[5];
    if (simd_lane == 0) s_sum[simd_id] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) total_sq = simd_sum(total_sq);

    if (tid == 0) {
        s_sum[4] = rsqrt(total_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = s_sum[4];

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
        threadgroup float s_q[4];
        if (simd_lane == 0) s_q[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float blk = (simd_id == 0 && simd_lane < 4) ? s_q[simd_lane] : 0.0f;
        if (simd_id == 0) blk = simd_sum(blk);
        if (tid == 0) q[row] = blk;
    }

    if (row < kv_dim) {
        device const float* wk_row = WK + row * dim;
        device const float* wv_row = WV + row * dim;
        device const float4* wk4 = (device const float4*)wk_row;
        device const float4* wv4 = (device const float4*)wv_row;
        device const float4* nw4 = (device const float4*)norm_weight;

        float acc_k = 0.0f;
        float acc_v = 0.0f;
        for (uint i = tid; i < dim4; i += 128) {
            float4 normed = x4[i] * rms * nw4[i];
            acc_k += dot(normed, wk4[i]);
            acc_v += dot(normed, wv4[i]);
        }
        for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
            float normed = x[i] * rms * norm_weight[i];
            acc_k += normed * wk_row[i];
            acc_v += normed * wv_row[i];
        }
        acc_k = simd_sum(acc_k);
        acc_v = simd_sum(acc_v);

        threadgroup float s_k[4];
        threadgroup float s_v[4];
        if (simd_lane == 0) {
            s_k[simd_id] = acc_k;
            s_v[simd_id] = acc_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float blk_k = (simd_id == 0 && simd_lane < 4) ? s_k[simd_lane] : 0.0f;
        float blk_v = (simd_id == 0 && simd_lane < 4) ? s_v[simd_lane] : 0.0f;
        if (simd_id == 0) {
            blk_k = simd_sum(blk_k);
            blk_v = simd_sum(blk_v);
        }
        if (tid == 0) {
            k[row] = blk_k;
            v[row] = blk_v;
        }
    }
}

// 8. E-Graph Fused SwiGLU + Down Projection + Residual Add Megakernel
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

// =========================================================================
// Batched Prefill Kernels (Matrix-Matrix GEMM & FlashAttention Prefill)
// =========================================================================

kernel void BatchedEmbeddingLookup(
    device const float* embed_table    [[buffer(0)]],
    device const int* tokens           [[buffer(1)]],
    device float* x_out                [[buffer(2)]],
    constant uint& seq_len             [[buffer(3)]],
    constant uint& dim                 [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint token_idx = gid / dim;
    uint dim_idx = gid % dim;
    if (token_idx < seq_len && dim_idx < dim) {
        int tok = tokens[token_idx];
        x_out[token_idx * dim + dim_idx] = embed_table[tok * dim + dim_idx];
    }
}

kernel void BatchedRMSNorm(
    device const float* x              [[buffer(0)]],
    device const float* weight         [[buffer(1)]],
    device float* out_normed           [[buffer(2)]],
    constant uint& seq_len             [[buffer(3)]],
    constant uint& dim                 [[buffer(4)]],
    constant float& eps                [[buffer(5)]],
    uint token_idx                     [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (token_idx < seq_len) {
        device const float* x_row = x + token_idx * dim;
        device float* out_row = out_normed + token_idx * dim;
        uint dim4 = dim / 4;
        device const float4* x4 = (device const float4*)x_row;
        device const float4* w4 = (device const float4*)weight;
        device float4* out4 = (device float4*)out_row;

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
        threadgroup float s_sum[5];
        if (simd_lane == 0) s_sum[simd_id] = local_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_sq = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) total_sq = simd_sum(total_sq);

        if (tid == 0) {
            s_sum[4] = rsqrt(total_sq / float(dim) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rms = s_sum[4];
        for (uint i = tid; i < dim4; i += 128) {
            out4[i] = x4[i] * rms * w4[i];
        }
        for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
            out_row[i] = x_row[i] * rms * weight[i];
        }
    }
}

kernel void BatchedGEMM(
    device const float* X              [[buffer(0)]],
    device const float* W              [[buffer(1)]],
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
        device const float* w_row = W + out_col * in_dim;
        uint in_dim4 = in_dim / 4;
        device const float4* x4 = (device const float4*)x_row;
        device const float4* w4 = (device const float4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            acc += dot(x4[i], w4[i]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x_row[i] * w_row[i];
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

kernel void BatchedGEMMResidual(
    device const float* X              [[buffer(0)]],
    device const float* W              [[buffer(1)]],
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
        device const float* w_row = W + out_col * in_dim;
        uint in_dim4 = in_dim / 4;
        device const float4* x4 = (device const float4*)x_row;
        device const float4* w4 = (device const float4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            acc += dot(x4[i], w4[i]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x_row[i] * w_row[i];
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

kernel void BatchedRoPE(
    device float* q_all                [[buffer(0)]],
    device float* k_all                [[buffer(1)]],
    device float* v_all                [[buffer(2)]],
    device float* k_cache              [[buffer(3)]],
    device float* v_cache              [[buffer(4)]],
    constant uint& seq_len             [[buffer(5)]],
    constant uint& start_pos           [[buffer(6)]],
    constant uint& head_dim            [[buffer(7)]],
    constant uint& num_q_heads         [[buffer(8)]],
    constant uint& num_kv_heads        [[buffer(9)]],
    constant float& theta              [[buffer(10)]],
    uint t                             [[thread_position_in_grid]]
) {
    if (t < seq_len) {
        uint actual_pos = start_pos + t;
        uint total_dim = num_q_heads * head_dim;
        uint kv_dim = num_kv_heads * head_dim;
        uint half_hd = head_dim / 2;

        device float* q = q_all + t * total_dim;
        device float* k = k_all + t * kv_dim;
        device float* v = v_all + t * kv_dim;
        device float* kc = k_cache + actual_pos * kv_dim;
        device float* vc = v_cache + actual_pos * kv_dim;

        for (uint h = 0; h < num_q_heads; h++) {
            for (uint i = 0; i < half_hd; i++) {
                float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
                float val = float(actual_pos) * freq;
                float cos_val = cos(val);
                float sin_val = sin(val);

                uint idx0 = h * head_dim + i;
                uint idx1 = h * head_dim + i + half_hd;
                float q0 = q[idx0];
                float q1 = q[idx1];
                q[idx0] = q0 * cos_val - q1 * sin_val;
                q[idx1] = q0 * sin_val + q1 * cos_val;
            }
        }

        for (uint h = 0; h < num_kv_heads; h++) {
            for (uint i = 0; i < half_hd; i++) {
                float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
                float val = float(actual_pos) * freq;
                float cos_val = cos(val);
                float sin_val = sin(val);

                uint idx0 = h * head_dim + i;
                uint idx1 = h * head_dim + i + half_hd;
                float k0 = k[idx0];
                float k1 = k[idx1];
                float rk0 = k0 * cos_val - k1 * sin_val;
                float rk1 = k0 * sin_val + k1 * cos_val;
                k[idx0] = rk0;
                k[idx1] = rk1;
                kc[idx0] = rk0;
                kc[idx1] = rk1;
            }
            for (uint i = 0; i < head_dim; i++) {
                uint idx = h * head_dim + i;
                vc[idx] = v[idx];
            }
        }
    }
}

kernel void BatchedCausalFlashAttention(
    device const float* q_all          [[buffer(0)]],
    device const float* k_cache        [[buffer(1)]],
    device const float* v_cache        [[buffer(2)]],
    device float* out_all              [[buffer(3)]],
    constant uint& seq_len             [[buffer(4)]],
    constant uint& start_pos           [[buffer(5)]],
    constant uint& num_q_heads         [[buffer(6)]],
    constant uint& num_kv_heads        [[buffer(7)]],
    constant uint& head_dim            [[buffer(8)]],
    constant float& scale              [[buffer(9)]],
    uint group_idx                     [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint t = group_idx / num_q_heads;
    uint q_head = group_idx % num_q_heads;
    if (t < seq_len && q_head < num_q_heads) {
        uint actual_pos = start_pos + t;
        uint kv_group = num_q_heads / num_kv_heads;
        uint kv_head = q_head / kv_group;

        uint q_offset = t * (num_q_heads * head_dim) + q_head * head_dim;
        device const float* q_vec = q_all + q_offset;
        device float* out_vec = out_all + q_offset;

        float m_prev = -1e30f;
        float l_prev = 0.0f;
        float acc_v[128] = {0.0f};

        for (uint p = 0; p <= actual_pos; p++) {
            device const float* k_vec = k_cache + p * (num_kv_heads * head_dim) + kv_head * head_dim;
            float score = 0.0f;
            for (uint d = tid; d < head_dim; d += 128) {
                score += q_vec[d] * k_vec[d];
            }
            score = simd_sum(score);
            threadgroup float s_score[4];
            if (simd_lane == 0) s_score[simd_id] = score;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float total_score = (simd_id == 0 && simd_lane < 4) ? s_score[simd_lane] : 0.0f;
            if (simd_id == 0) total_score = simd_sum(total_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            total_score *= scale;

            float m_curr = max(m_prev, total_score);
            float exp_prev = exp(m_prev - m_curr);
            float exp_curr = exp(total_score - m_curr);
            float l_curr = l_prev * exp_prev + exp_curr;

            device const float* v_vec = v_cache + p * (num_kv_heads * head_dim) + kv_head * head_dim;
            for (uint d = tid; d < head_dim; d += 128) {
                acc_v[d] = acc_v[d] * exp_prev + exp_curr * v_vec[d];
            }
            m_prev = m_curr;
            l_prev = l_curr;
        }

        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        for (uint d = tid; d < head_dim; d += 128) {
            out_vec[d] = acc_v[d] * inv_l;
        }
    }
}

kernel void BatchedSiLUMul(
    device const float* gate           [[buffer(0)]],
    device const float* up             [[buffer(1)]],
    device float* out_act              [[buffer(2)]],
    constant uint& total_elements      [[buffer(3)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid < total_elements) {
        float g = gate[tid];
        float silu_g = g / (1.0f + exp(-g));
        out_act[tid] = silu_g * up[tid];
    }
}
