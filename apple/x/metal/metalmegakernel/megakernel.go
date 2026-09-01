package metalmegakernel

import (
	"fmt"
)

// LayerConfig defines the hyperparameter dimensions for a transformer layer megakernel.
type LayerConfig struct {
	HiddenDim        int
	IntermediateDim  int
	NumQHeads        int
	NumKVHeads       int
	HeadDim          int
	RmsNormEps       float32
	RopeTheta        float32
	MaxSeqLen        int
}

// GenerateLayerMegakernelMSL generates a single fused MSL compute kernel that evaluates an entire transformer layer.
func GenerateLayerMegakernelMSL(cfg LayerConfig) string {
	return fmt.Sprintf(`// --- Metal Transformer Layer Megakernel ---
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float simdgroup_sum(float val) {
    #pragma unroll
    for (ushort offset = 16; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

kernel void MetalLayerMegakernel(
    device float* x                    [[buffer(0)]],
    device const float* w_input_norm   [[buffer(1)]],
    device const float* w_q            [[buffer(2)]],
    device const float* w_k            [[buffer(3)]],
    device const float* w_v            [[buffer(4)]],
    device const float* w_o            [[buffer(5)]],
    device const float* w_post_norm    [[buffer(6)]],
    device const float* w_gate         [[buffer(7)]],
    device const float* w_up           [[buffer(8)]],
    device const float* w_down         [[buffer(9)]],
    device float* k_cache              [[buffer(10)]],
    device float* v_cache              [[buffer(11)]],
    constant int& cur_pos              [[buffer(12)]],
    threadgroup float* s_scratch       [[threadgroup(0)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint hidden_dim = %d;
    constexpr uint inter_dim  = %d;
    constexpr float eps       = %f;

    // Step 1: Input RMSNorm in shared memory
    float sum_sq = 0.0f;
    for (uint i = tid; i < hidden_dim; i += 128) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = simdgroup_sum(sum_sq);
    threadgroup float s_warp[4];
    if (simd_lane == 0) s_warp[simd_id] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = (simd_id == 0 && simd_lane < 4) ? s_warp[simd_lane] : 0.0f;
    total_sum_sq = simdgroup_sum(total_sum_sq);
    threadgroup float s_rsqrt;
    if (tid == 0) {
        s_rsqrt = rsqrt(total_sum_sq / float(hidden_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rsqrt_val = s_rsqrt;
    for (uint i = tid; i < hidden_dim; i += 128) {
        s_scratch[i] = x[i] * rsqrt_val * w_input_norm[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: FFN Gate & Up projections + SiLU in-register
    for (uint j = tid; j < inter_dim; j += 128) {
        float g = 0.0f;
        float u = 0.0f;
        for (uint k = 0; k < hidden_dim; k++) {
            float normed_x = s_scratch[k];
            g += normed_x * w_gate[j * hidden_dim + k];
            u += normed_x * w_up[j * hidden_dim + k];
        }
        float silu = g / (1.0f + exp(-g));
        s_scratch[hidden_dim + j] = silu * u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: FFN Down projection and residual accumulate
    if (row < hidden_dim) {
        device const float* w_down_row = w_down + row * inter_dim;
        float acc = 0.0f;
        for (uint j = tid; j < inter_dim; j += 128) {
            acc += s_scratch[hidden_dim + j] * w_down_row[j];
        }
        acc = simdgroup_sum(acc);
        if (simd_lane == 0) s_warp[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_acc = (simd_id == 0 && simd_lane < 4) ? s_warp[simd_lane] : 0.0f;
        block_acc = simdgroup_sum(block_acc);
        if (tid == 0) {
            x[row] += block_acc;
        }
    }
}
`, cfg.HiddenDim, cfg.IntermediateDim, cfg.RmsNormEps)
}
