//go:build darwin

package metalmegakernel

// LlamaDecoderKernelsMSL contains hand-optimized, SIMD-vectorized Metal compute shaders
// for 100% GPU-resident transformer forward pass execution.
const LlamaDecoderKernelsMSL = `
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float simd_reduce_sum(float val) {
    #pragma unroll
    for (ushort offset = 16; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

// 1. Embedding lookup kernel on GPU
kernel void EmbeddingLookup(
    device const float* embed_table    [[buffer(0)]], // [VocabSize, Dim]
    device float* x_out                [[buffer(1)]], // [Dim]
    constant uint& token_id            [[buffer(2)]],
    constant uint& dim                 [[buffer(3)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid < dim) {
        x_out[tid] = embed_table[token_id * dim + tid];
    }
}

// 2. Fused RMSNorm + QKV Projection Kernel
kernel void FusedRMSNormQKV(
    device const float* x          [[buffer(0)]], // [Dim]
    device const float* norm_w     [[buffer(1)]], // [Dim]
    device const float* W_q        [[buffer(2)]], // [Heads*HeadDim, Dim]
    device const float* W_k        [[buffer(3)]], // [KVHeads*HeadDim, Dim]
    device const float* W_v        [[buffer(4)]], // [KVHeads*HeadDim, Dim]
    device float* out_q            [[buffer(5)]], // [Heads*HeadDim]
    device float* out_k            [[buffer(6)]], // [KVHeads*HeadDim]
    device float* out_v            [[buffer(7)]], // [KVHeads*HeadDim]
    constant uint& dim             [[buffer(8)]],
    constant uint& q_dim           [[buffer(9)]],
    constant uint& kv_dim          [[buffer(10)]],
    constant float& eps            [[buffer(11)]],
    uint row                       [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += 128) {
        float v = x[i];
        local_sq += v * v;
    }
    local_sq = simd_reduce_sum(local_sq);
    threadgroup float s_sum[4];
    if (simd_lane == 0) s_sum[simd_id] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) total_sq = simd_reduce_sum(total_sq);

    threadgroup float s_rms;
    if (tid == 0) {
        s_rms = rsqrt(total_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Q Projection
    if (row < q_dim) {
        device const float* w_row = W_q + row * dim;
        float acc = 0.0f;
        for (uint i = tid; i < dim; i += 128) {
            acc += (x[i] * s_rms * norm_w[i]) * w_row[i];
        }
        acc = simd_reduce_sum(acc);
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_reduce_sum(block_sum);
        if (tid == 0) out_q[row] = block_sum;
    }

    // K & V Projections
    if (row < kv_dim) {
        device const float* wk_row = W_k + row * dim;
        device const float* wv_row = W_v + row * dim;
        float acc_k = 0.0f;
        float acc_v = 0.0f;
        for (uint i = tid; i < dim; i += 128) {
            float normed = x[i] * s_rms * norm_w[i];
            acc_k += normed * wk_row[i];
            acc_v += normed * wv_row[i];
        }
        acc_k = simd_reduce_sum(acc_k);
        acc_v = simd_reduce_sum(acc_v);

        threadgroup float s_sum_k[4];
        threadgroup float s_sum_v[4];
        if (simd_lane == 0) {
            s_sum_k[simd_id] = acc_k;
            s_sum_v[simd_id] = acc_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float bk_sum = (simd_id == 0 && simd_lane < 4) ? s_sum_k[simd_lane] : 0.0f;
        float bv_sum = (simd_id == 0 && simd_lane < 4) ? s_sum_v[simd_lane] : 0.0f;
        if (simd_id == 0) {
            bk_sum = simd_reduce_sum(bk_sum);
            bv_sum = simd_reduce_sum(bv_sum);
        }
        if (tid == 0) {
            out_k[row] = bk_sum;
            out_v[row] = bv_sum;
        }
    }
}

// 3. Fused RoPE + KV Cache Storage
kernel void FusedRoPEStoreKV(
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

// 4. Attention Output projection & residual add
kernel void OutProjResidual(
    device const float* attn_out  [[buffer(0)]],
    device const float* W_o       [[buffer(1)]],
    device float* x_inout         [[buffer(2)]],
    constant uint& dim            [[buffer(3)]],
    uint row                      [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    if (row < dim) {
        device const float* w_row = W_o + row * dim;
        float acc = 0.0f;
        for (uint i = tid; i < dim; i += 128) {
            acc += attn_out[i] * w_row[i];
        }
        acc = simd_reduce_sum(acc);
        threadgroup float s_sum[4];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_reduce_sum(block_sum);
        if (tid == 0) x_inout[row] += block_sum;
    }
}

// 5. Fused SwiGLU FFN + Residual Add Megakernel
kernel void FusedSwiGLUResidual(
    device float* x               [[buffer(0)]],
    device const float* norm_w    [[buffer(1)]],
    device const float* W_gate    [[buffer(2)]],
    device const float* W_up      [[buffer(3)]],
    device const float* W_down    [[buffer(4)]],
    device float* s_ffn_act       [[buffer(5)]],
    constant uint& dim            [[buffer(6)]],
    constant uint& hidden_dim     [[buffer(7)]],
    constant float& eps           [[buffer(8)]],
    uint row                      [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += 128) {
        float v = x[i];
        local_sq += v * v;
    }
    local_sq = simd_reduce_sum(local_sq);
    threadgroup float s_sum[4];
    if (simd_lane == 0) s_sum[simd_id] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) total_sq = simd_reduce_sum(total_sq);

    threadgroup float s_rms;
    if (tid == 0) {
        s_rms = rsqrt(total_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < hidden_dim; j += 128) {
        float gate = 0.0f;
        float up = 0.0f;
        device const float* wg_row = W_gate + j * dim;
        device const float* wu_row = W_up + j * dim;
        for (uint k = 0; k < dim; k++) {
            float normed = x[k] * s_rms * norm_w[k];
            gate += normed * wg_row[k];
            up += normed * wu_row[k];
        }
        float silu = gate / (1.0f + exp(-gate));
        s_ffn_act[j] = silu * up;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < dim) {
        device const float* wd_row = W_down + row * hidden_dim;
        float acc = 0.0f;
        for (uint j = tid; j < hidden_dim; j += 128) {
            acc += s_ffn_act[j] * wd_row[j];
        }
        acc = simd_reduce_sum(acc);
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_reduce_sum(block_sum);
        if (tid == 0) x[row] += block_sum;
    }
}

// 6. Final RMSNorm + LM Head Logits Projection
kernel void FusedFinalNormLMHead(
    device const float* x          [[buffer(0)]],
    device const float* norm_w     [[buffer(1)]],
    device const float* W_lm       [[buffer(2)]],
    device float* out_logits       [[buffer(3)]],
    constant uint& dim             [[buffer(4)]],
    constant uint& vocab_size      [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
    uint row                       [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += 128) {
        float v = x[i];
        local_sq += v * v;
    }
    local_sq = simd_reduce_sum(local_sq);
    threadgroup float s_sum[4];
    if (simd_lane == 0) s_sum[simd_id] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) total_sq = simd_reduce_sum(total_sq);

    threadgroup float s_rms;
    if (tid == 0) {
        s_rms = rsqrt(total_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < vocab_size) {
        device const float* w_row = W_lm + row * dim;
        float acc = 0.0f;
        for (uint i = tid; i < dim; i += 128) {
            acc += (x[i] * s_rms * norm_w[i]) * w_row[i];
        }
        acc = simd_reduce_sum(acc);
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_reduce_sum(block_sum);
        if (tid == 0) out_logits[row] = block_sum;
    }
}
`
