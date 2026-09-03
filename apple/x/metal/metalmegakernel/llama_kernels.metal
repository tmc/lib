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

kernel void BatchedQKVProjection(
    device const float* X [[buffer(0)]], device const float* WQ [[buffer(1)]],
    device const float* WK [[buffer(2)]], device const float* WV [[buffer(3)]],
    device float* Q [[buffer(4)]], device float* K [[buffer(5)]], device float* V [[buffer(6)]],
    constant uint& rows [[buffer(7)]], constant uint& in_dim [[buffer(8)]],
    constant uint& q_dim [[buffer(9)]], constant uint& kv_dim [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    uint span = q_dim + 2 * kv_dim;
    uint row = group / span, col = group % span;
    if (row >= rows) return;
    device const float* weight;
    device float* output;
    uint out_dim, out_col;
    if (col < q_dim) { weight=WQ; output=Q; out_dim=q_dim; out_col=col; }
    else if (col < q_dim+kv_dim) { weight=WK; output=K; out_dim=kv_dim; out_col=col-q_dim; }
    else { weight=WV; output=V; out_dim=kv_dim; out_col=col-q_dim-kv_dim; }
    device const float* x = X + row * in_dim;
    device const float* w = weight + out_col * in_dim;
    uint n4=in_dim/4; device const float4* x4=(device const float4*)x; device const float4* w4=(device const float4*)w;
    float acc=0;
    for(uint i=tid;i<n4;i+=128) acc += dot(x4[i],w4[i]);
    for(uint i=n4*4+tid;i<in_dim;i+=128) acc += x[i]*w[i];
    acc=simd_sum(acc); threadgroup float sums[4]; if(lane==0) sums[simd]=acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=(simd==0 && lane<4)?sums[lane]:0; if(simd==0) total=simd_sum(total);
    if(tid==0) output[row*out_dim+out_col]=total;
}

kernel void BatchedGateUpProjection(
    device const float* X [[buffer(0)]], device const float* WG [[buffer(1)]],
    device const float* WU [[buffer(2)]], device float* G [[buffer(3)]], device float* U [[buffer(4)]],
    constant uint& rows [[buffer(5)]], constant uint& in_dim [[buffer(6)]], constant uint& out_dim [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    uint span=2*out_dim, row=group/span, col=group%span; if(row>=rows) return;
    bool up=col>=out_dim; uint out_col=up?col-out_dim:col;
    device const float* w=(up?WU:WG)+out_col*in_dim; device const float* x=X+row*in_dim;
    uint n4=in_dim/4; device const float4* x4=(device const float4*)x; device const float4* w4=(device const float4*)w;
    float acc=0; for(uint i=tid;i<n4;i+=128) acc+=dot(x4[i],w4[i]);
    for(uint i=n4*4+tid;i<in_dim;i+=128) acc+=x[i]*w[i];
    acc=simd_sum(acc); threadgroup float sums[4]; if(lane==0) sums[simd]=acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=(simd==0 && lane<4)?sums[lane]:0; if(simd==0) total=simd_sum(total);
    if(tid==0) (up?U:G)[row*out_dim+out_col]=total;
}

kernel void BatchedArgmaxSampling(
    device const float* logits          [[buffer(0)]],
    device int* token_out               [[buffer(1)]],
    constant uint& batch                [[buffer(2)]],
    constant uint& vocab_size           [[buffer(3)]],
    uint row                             [[threadgroup_position_in_grid]],
    uint tid                             [[thread_position_in_threadgroup]],
    uint tg_size                         [[threads_per_threadgroup]]
) {
    // The host contract dispatches exactly 128 threads. Fixing the width keeps
    // the reduction layout explicit and prevents partial reductions.
    if (row >= batch || vocab_size == 0 || tg_size != 128) return;
    threadgroup float values[128];
    threadgroup int indices[128];
    device const float* row_logits = logits + row * vocab_size;
    float best_value = -INFINITY;
    int best_index = -1;
    for (uint i = tid; i < vocab_size; i += tg_size) {
        float value = row_logits[i];
        // NaNs are ignored. All-NaN and all-negative-infinity rows return 0.
        if (!isnan(value) && (value > best_value ||
            (value == best_value && (best_index < 0 || int(i) < best_index)))) {
            best_value = value;
            best_index = int(i);
        }
    }
    values[tid] = best_value;
    indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = tg_size / 2; width > 0; width /= 2) {
        if (tid < width) {
            float other = values[tid + width];
            int other_index = indices[tid + width];
            if (other > values[tid] || (other == values[tid] && other_index < indices[tid])) {
                values[tid] = other;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) token_out[row] = max(indices[0], 0);
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
    if (t >= seq_len || q_head >= num_q_heads) {
        return;
    }

    uint actual_pos = start_pos + t;
    uint kv_group = num_q_heads / num_kv_heads;
    uint kv_head = q_head / kv_group;

    uint q_offset = t * (num_q_heads * head_dim) + q_head * head_dim;
    device const float* q_vec = q_all + q_offset;
    device float* out_vec = out_all + q_offset;

    threadgroup float s_red[4];
    threadgroup float s_score_bcast;

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
        if (simd_lane == 0) {
            s_red[simd_id] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = s_red[0] + s_red[1] + s_red[2] + s_red[3];
            s_score_bcast = total * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_score = s_score_bcast;

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

kernel void RowIndexedRoPEStoreKV(
    device float* q [[buffer(0)]], device float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device float* kc [[buffer(3)]],
    device float* vc [[buffer(4)]], device const uint* positions [[buffer(5)]],
    device const uint* rows [[buffer(6)]], constant uint& batch [[buffer(7)]],
    constant uint& max_seq [[buffer(8)]], constant uint& head_dim [[buffer(9)]],
    constant uint& q_heads [[buffer(10)]], constant uint& kv_heads [[buffer(11)]],
    constant float& theta [[buffer(12)]], constant uint& row_capacity [[buffer(13)]],
    uint b [[thread_position_in_grid]]) {
    if (b >= batch || head_dim == 0 || head_dim > 128 || (head_dim & 1) != 0 ||
        q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 || theta <= 0) return;
    uint pos=positions[b], row=rows[b];
    if (pos >= max_seq || row >= row_capacity) return;
    uint qdim=q_heads*head_dim, kdim=kv_heads*head_dim, half_dim=head_dim/2;
    for (uint h=0; h<q_heads; h++) for (uint i=0; i<half_dim; i++) {
        float a=float(pos)/pow(theta,float(2*i)/float(head_dim)), c=cos(a), s=sin(a);
        uint x=b*qdim+h*head_dim+i, y=x+half_dim; float x0=q[x], x1=q[y]; q[x]=x0*c-x1*s; q[y]=x0*s+x1*c;
    }
    uint base=(row*max_seq+pos)*kdim;
    for (uint h=0; h<kv_heads; h++) {
        for (uint i=0; i<half_dim; i++) { float a=float(pos)/pow(theta,float(2*i)/float(head_dim)), c=cos(a), s=sin(a); uint x=b*kdim+h*head_dim+i,y=x+half_dim; float x0=k[x],x1=k[y]; k[x]=x0*c-x1*s;k[y]=x0*s+x1*c;kc[base+h*head_dim+i]=k[x];kc[base+h*head_dim+i+half_dim]=k[y]; }
        for (uint i=0;i<head_dim;i++) vc[base+h*head_dim+i]=v[b*kdim+h*head_dim+i];
    }
}

kernel void RowIndexedCausalAttention(
    device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]],
    device const float* vc [[buffer(2)]], device float* out [[buffer(3)]],
    device const uint* positions [[buffer(4)]], device const uint* rows [[buffer(5)]],
    constant uint& batch [[buffer(6)]], constant uint& max_seq [[buffer(7)]],
    constant uint& q_heads [[buffer(8)]], constant uint& kv_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]], constant float& scale [[buffer(11)]],
    constant uint& row_capacity [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    if (q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 ||
        head_dim == 0 || head_dim > 128 || (head_dim & 1) != 0 ||
        scale <= 0) return;
    uint b=group/q_heads, qh=group%q_heads; if (b>=batch) return;
    uint pos=positions[b], row=rows[b]; if (pos>=max_seq || row>=row_capacity) return;
    uint kh=qh/(q_heads/kv_heads), kdim=kv_heads*head_dim;
    device const float* qv=q+b*q_heads*head_dim+qh*head_dim;
    if (tg_size != 128) return;
    threadgroup float red[4]; threadgroup float score;
    float m=-INFINITY,l=0,acc=0;
    for(uint p=0;p<=pos;p++) {
        uint base=(row*max_seq+p)*kdim+kh*head_dim;
        float z=tid<head_dim ? qv[tid]*kc[base+tid] : 0;
        z=simd_sum(z); if(lane==0) red[simd]=z;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0) { float total=0; for(uint i=0;i<4;i++) total+=red[i]; score=total*scale; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float nm=max(m,score),ep=exp(m-nm),ec=exp(score-nm);
        if(tid<head_dim) acc=acc*ep+ec*vc[base+tid]; l=l*ep+ec; m=nm;
    }
    if(tid<head_dim) out[b*q_heads*head_dim+qh*head_dim+tid]=l>0 ? acc/l : 0;
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
