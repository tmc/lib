package metalmegakernel

import (
	"fmt"
	"math"
)

// FusedFFNConfig configures the generated RMSNorm and feed-forward kernel.
type FusedFFNConfig struct {
	HiddenDim       int
	IntermediateDim int
	RMSNormEpsilon  float32
}

// FusedFFNThreads is the required number of threads in the single threadgroup
// used to dispatch MetalFusedFFN.
const FusedFFNThreads = 128

// ScratchBytes reports the threadgroup-memory requirement of the generated
// kernel.
func (c FusedFFNConfig) ScratchBytes() int {
	return 4 * (c.HiddenDim + c.IntermediateDim)
}

// GenerateFusedFFNMSL generates an MSL kernel that fuses RMS normalization,
// SwiGLU, the down projection, and residual addition for one vector.
// It does not implement attention or a complete transformer layer.
// Dispatch it as one threadgroup of FusedFFNThreads threads and bind
// cfg.ScratchBytes() bytes of threadgroup memory at index 0.
func GenerateFusedFFNMSL(cfg FusedFFNConfig) (string, error) {
	if cfg.HiddenDim <= 0 {
		return "", fmt.Errorf("metalmegakernel: hidden dimension must be positive")
	}
	if cfg.IntermediateDim <= 0 {
		return "", fmt.Errorf("metalmegakernel: intermediate dimension must be positive")
	}
	if cfg.RMSNormEpsilon <= 0 || math.IsNaN(float64(cfg.RMSNormEpsilon)) || math.IsInf(float64(cfg.RMSNormEpsilon), 0) {
		return "", fmt.Errorf("metalmegakernel: RMS normalization epsilon must be finite and positive")
	}
	return generateFusedFFNMSL(cfg), nil
}

// LayerConfig is the legacy configuration for GenerateLayerMegakernelMSL.
//
// Deprecated: use FusedFFNConfig. Attention-related fields have never affected
// the generated kernel.
type LayerConfig struct {
	HiddenDim       int
	IntermediateDim int
	NumQHeads       int
	NumKVHeads      int
	HeadDim         int
	RmsNormEps      float32
	RopeTheta       float32
	MaxSeqLen       int
}

// GenerateLayerMegakernelMSL generates the historical RMSNorm and feed-forward
// kernel. It returns an empty string for invalid dimensions or epsilon.
//
// Deprecated: use GenerateFusedFFNMSL. This function has never generated a
// complete transformer layer.
func GenerateLayerMegakernelMSL(cfg LayerConfig) string {
	msl, _ := GenerateFusedFFNMSL(FusedFFNConfig{
		HiddenDim:       cfg.HiddenDim,
		IntermediateDim: cfg.IntermediateDim,
		RMSNormEpsilon:  cfg.RmsNormEps,
	})
	return msl
}

func generateFusedFFNMSL(cfg FusedFFNConfig) string {
	return fmt.Sprintf(`// --- Metal Fused RMSNorm and Feed-Forward Kernel ---
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

kernel void MetalFusedFFN(
    device float* x                    [[buffer(0)]],
    device const float* w_input_norm   [[buffer(1)]],
    device const float* w_gate         [[buffer(2)]],
    device const float* w_up           [[buffer(3)]],
    device const float* w_down         [[buffer(4)]],
    threadgroup float* s_scratch       [[threadgroup(0)]],
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
    for (uint row = 0; row < hidden_dim; row++) {
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
`, cfg.HiddenDim, cfg.IntermediateDim, cfg.RMSNormEpsilon)
}
