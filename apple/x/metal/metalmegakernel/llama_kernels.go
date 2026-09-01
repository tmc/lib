//go:build darwin

package metalmegakernel

// LlamaDecoderKernelsMSL contains high-throughput 2D tiled GEMV compute shaders
// for fast GPU-resident transformer inference on Apple Silicon.
const LlamaDecoderKernelsMSL = `
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float simd_sum(float val) {
    #pragma unroll
    for (ushort offset = 16; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

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

// 4. Fast GEMV with Residual Accumulation (y[row] += W[row] * x)
kernel void FastGEMVResidual(
    device const float* x              [[buffer(0)]], // [InDim]
    device const float* W              [[buffer(1)]], // [OutDim, InDim]
    device float* y_inout              [[buffer(2)]], // [OutDim]
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

// 4b. 64-thread GEMV variants used by the runtime PGO tuner.
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
`
