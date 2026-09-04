//go:build darwin

package metalbench

import "fmt"

// generateTiledGEMVMSL returns MSL source for a GEMV kernel named kernelName,
// unrolled unrollFactor times over a tile of tileWidth elements. The harnesses
// compile several of these to compare dispatch shapes on the same math.
func generateTiledGEMVMSL(kernelName string, unrollFactor int) string {
	return fmt.Sprintf(`
#include <metal_stdlib>
using namespace metal;

inline float simd_sum(float val) {
    #pragma unroll
    for (ushort offset = 16; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

kernel void %s(
    device const float* x      [[buffer(0)]],
    device const float* W      [[buffer(1)]],
    device float* out          [[buffer(2)]],
    constant uint& dim         [[buffer(3)]],
    constant uint& out_dim     [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= out_dim) return;

    device const float* w_row = W + row * dim;
    float acc = 0.0f;

    uint i = tid * %d;
    for (; i + %d <= dim; i += 128 * %d) {
        #pragma unroll
        for (int k = 0; k < %d; k++) {
            acc += x[i + k] * w_row[i + k];
        }
    }
    for (; i < dim; i++) {
        acc += x[i] * w_row[i];
    }

    acc = simd_sum(acc);
    threadgroup float s_sum[4];
    if (simd_lane == 0) s_sum[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) block_sum = simd_sum(block_sum);
    if (tid == 0) out[row] = block_sum;
}
`, kernelName, unrollFactor, unrollFactor, unrollFactor, unrollFactor)
}
