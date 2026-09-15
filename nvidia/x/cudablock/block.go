package cudablock

// Header is the embedded CUDA C++ header defining warp and block collective primitives.
const Header = `
#pragma once

#ifndef CUDABLOCK_CUH
#define CUDABLOCK_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Warp-level sum reduction using shuffle intrinsics
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Warp-level max reduction using shuffle intrinsics
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(FULL_MASK, val, offset);
        val = (other > val) ? other : val;
    }
    return val;
}

// Block-level sum reduction using shared memory for warp leaders
template <typename T, int BLOCK_THREADS>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared_scratch) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum<T>(val);

    if (lane == 0) {
        shared_scratch[wid] = val;
    }
    __syncthreads();

    const int NUM_WARPS = (BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    T block_sum = (threadIdx.x < NUM_WARPS) ? shared_scratch[lane] : T(0);

    if (wid == 0) {
        block_sum = warp_reduce_sum<T>(block_sum);
    }
    return block_sum;
}

// Block-level cooperative 2D tile loader from global to shared memory
template <typename T, int TILE_M, int TILE_N, int BLOCK_THREADS>
__device__ __forceinline__ void tile_load_2d(
    T shared_tile[TILE_M][TILE_N],
    const T* __restrict__ global_src,
    int stride,
    int max_rows,
    int max_cols
) {
    const int TOTAL_ELEMENTS = TILE_M * TILE_N;
    #pragma unroll
    for (int idx = threadIdx.x; idx < TOTAL_ELEMENTS; idx += BLOCK_THREADS) {
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        if (r < max_rows && c < max_cols) {
            shared_tile[r][c] = global_src[r * stride + c];
        } else {
            shared_tile[r][c] = T(0);
        }
    }
}

// INT4 dequantization to float32 using scale and zero-point
__device__ __forceinline__ float dequant_int4(unsigned char packed_byte, int nibble_idx, float scale, float zero) {
    int val = (nibble_idx == 0) ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);
    return (float(val) - zero) * scale;
}

// Fast FP16 vector load
__device__ __forceinline__ float2 load_half2_as_float2(const __half2* ptr) {
    __half2 h = *ptr;
    return make_float2(__half2float(h.x), __half2float(h.y));
}

#endif // CUDABLOCK_CUH
`

// TileConfig defines dimensions for cooperative threadblock staging.
type TileConfig struct {
	TileM        int
	TileN        int
	TileK        int
	BlockThreads int
	ElemBytes    int
}

// SharedMemBytes calculates the required shared memory scratchpad size in bytes for A and B tiles.
func (tc TileConfig) SharedMemBytes() int {
	elem := tc.ElemBytes
	if elem == 0 {
		elem = 4
	}
	return (tc.TileM*tc.TileK + tc.TileK*tc.TileN) * elem
}


// DefaultGEMMTile returns standard shared-memory tile dimensions for GEMM/GEMV.
func DefaultGEMMTile() TileConfig {
	return TileConfig{
		TileM:        128,
		TileN:        128,
		TileK:        32,
		BlockThreads: 256,
		ElemBytes:    4,
	}
}
