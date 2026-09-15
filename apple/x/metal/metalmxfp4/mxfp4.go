package metalmxfp4

import (
	"fmt"
	"math"
)

// DefaultGroupSize is the standard OCP MXFP4 block size (32 elements).
const DefaultGroupSize = 32

// E2M1Table maps each 4-bit nibble (0..15) to its exact floating-point value.
var E2M1Table = [16]float32{
	0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
	-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
}

// MXFP4KernelsMSL contains the Metal Shading Language compute shader for fast 4-bit matrix operations.
const MXFP4KernelsMSL = `
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant float c_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Intra-SIMD reduction helper using Apple Silicon 32-wide SIMD-groups
inline float simdgroup_sum(float val) {
    #pragma unroll
    for (ushort offset = 16; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

// MXFP4Dequantize unpacks packed uint8 pairs and scales into full float32 output
kernel void MXFP4Dequantize(
    device const uchar* packed         [[buffer(0)]],
    device const float* scales         [[buffer(1)]],
    device float* out                  [[buffer(2)]],
    constant uint& total_elements      [[buffer(3)]],
    constant uint& group_size          [[buffer(4)]],
    uint idx                           [[thread_position_in_grid]]
) {
    if (idx * 2 >= total_elements) return;

    uchar byte_val = packed[idx];
    uint group_idx = (idx * 2) / group_size;
    float scale = scales[group_idx];

    uint n0 = byte_val & 0x0F;
    uint n1 = (byte_val >> 4) & 0x0F;

    out[idx * 2]     = c_e2m1_lut[n0] * scale;
    out[idx * 2 + 1] = c_e2m1_lut[n1] * scale;
}

// MXFP4LinearGEMV computes y = W_mxfp4 * x in a single high-throughput fused pass on Apple GPU
kernel void MXFP4LinearGEMV(
    device const float* x              [[buffer(0)]],
    device const uchar* packed_w       [[buffer(1)]],
    device const float* scales         [[buffer(2)]],
    device float* out_y                [[buffer(3)]],
    constant uint& in_dim              [[buffer(4)]],
    constant uint& out_dim             [[buffer(5)]],
    constant uint& group_size          [[buffer(6)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row >= out_dim) return;

    uint pairs_per_row = in_dim / 2;
    uint groups_per_row = in_dim / group_size;
    device const uchar* row_packed = packed_w + row * pairs_per_row;
    device const float* row_scales = scales + row * groups_per_row;

    float local_sum = 0.0f;
    for (uint p = tid; p < pairs_per_row; p += 128) {
        uchar b = row_packed[p];
        uint g = (p * 2) / group_size;
        float scale = row_scales[g];

        uint n0 = b & 0x0F;
        uint n1 = (b >> 4) & 0x0F;

        float w0 = c_e2m1_lut[n0] * scale;
        float w1 = c_e2m1_lut[n1] * scale;

        local_sum += x[p * 2] * w0 + x[p * 2 + 1] * w1;
    }

    local_sum = simdgroup_sum(local_sum);

    threadgroup float s_warp_sums[4];
    if (simd_lane == 0) {
        s_warp_sums[simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float val = (simd_lane < 4) ? s_warp_sums[simd_lane] : 0.0f;
        val = simdgroup_sum(val);
        if (simd_lane == 0) {
            out_y[row] = val;
        }
    }
}
`

// Quantize compresses a float32 slice into packed MXFP4 nibbles and scales.
func Quantize(src []float32, groupSize int) ([]byte, []float32, error) {
	if groupSize <= 0 || groupSize%2 != 0 {
		return nil, nil, fmt.Errorf("metalmxfp4: groupSize must be positive even integer, got %d", groupSize)
	}
	if len(src)%groupSize != 0 {
		return nil, nil, fmt.Errorf("metalmxfp4: src length (%d) must be multiple of groupSize (%d)", len(src), groupSize)
	}

	numGroups := len(src) / groupSize
	packed := make([]byte, len(src)/2)
	scales := make([]float32, numGroups)

	for g := 0; g < numGroups; g++ {
		groupStart := g * groupSize
		group := src[groupStart : groupStart+groupSize]

		maxAbs := float32(0.0)
		for _, v := range group {
			abs := float32(math.Abs(float64(v)))
			if abs > maxAbs {
				maxAbs = abs
			}
		}

		scale := maxAbs / 6.0
		if scale == 0 {
			scale = 1.0
		}
		scales[g] = scale
		invScale := 1.0 / scale

		packedStart := g * (groupSize / 2)
		for i := 0; i < groupSize; i += 2 {
			v0 := group[i] * invScale
			v1 := group[i+1] * invScale

			n0 := quantizeE2M1(v0)
			n1 := quantizeE2M1(v1)

			packed[packedStart+i/2] = byte(n0 | (n1 << 4))
		}
	}

	return packed, scales, nil
}

// Dequantize unpacks MXFP4 bytes and scales into float32 destination slice.
func Dequantize(packed []byte, scales []float32, groupSize int, dst []float32) error {
	if groupSize <= 0 || groupSize%2 != 0 {
		return fmt.Errorf("metalmxfp4: groupSize must be positive even integer, got %d", groupSize)
	}
	expectedLen := len(scales) * groupSize
	if len(dst) < expectedLen {
		return fmt.Errorf("metalmxfp4: dst slice too short (%d < %d)", len(dst), expectedLen)
	}
	if len(packed) < expectedLen/2 {
		return fmt.Errorf("metalmxfp4: packed slice too short (%d < %d)", len(packed), expectedLen/2)
	}

	for g := 0; g < len(scales); g++ {
		scale := scales[g]
		groupStart := g * groupSize
		packedStart := g * (groupSize / 2)

		for i := 0; i < groupSize/2; i++ {
			b := packed[packedStart+i]
			n0 := b & 0x0F
			n1 := (b >> 4) & 0x0F

			dst[groupStart+i*2] = E2M1Table[n0] * scale
			dst[groupStart+i*2+1] = E2M1Table[n1] * scale
		}
	}

	return nil
}

func quantizeE2M1(val float32) uint8 {
	isNeg := false
	if math.Signbit(float64(val)) {
		isNeg = true
		val = -val
	}

	bestIdx := 0
	bestDist := float32(math.Abs(float64(val - E2M1Table[0])))
	for i := 1; i < 8; i++ {
		dist := float32(math.Abs(float64(val - E2M1Table[i])))
		if dist < bestDist {
			bestDist = dist
			bestIdx = i
		}
	}

	if isNeg {
		bestIdx += 8
	}
	return uint8(bestIdx)
}
