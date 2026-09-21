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

// Number of lanes a launch of block_threads threads populates in warp wid. The
// final warp is partial whenever block_threads is not a multiple of WARP_SIZE,
// and the lanes past its end hold no value: a shuffle still returns something
// for them, so a reduction has to discard it rather than fold it in.
__device__ __forceinline__ int warp_lane_count(int block_threads, int wid) {
    int rest = block_threads - wid * WARP_SIZE;
    return (rest < WARP_SIZE) ? rest : WARP_SIZE;
}

// Block-level sum reduction using shared memory for warp leaders.
//
// shared_scratch must hold at least ceil(BLOCK_THREADS/WARP_SIZE) elements, and
// BLOCK_THREADS must equal blockDim.x. Every thread of the block must call this,
// because it synchronizes. The block sum is returned in lane 0 of warp 0, i.e.
// the thread with threadIdx.x == 0; what every other thread receives is
// unspecified. Callers that need the result blockwide must broadcast it
// themselves.
template <typename T, int BLOCK_THREADS>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared_scratch) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    const int warp_lanes = warp_lane_count(BLOCK_THREADS, wid);
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(FULL_MASK, val, offset);
        if (lane + offset < warp_lanes) {
            val += other;
        }
    }

    if (lane == 0) {
        shared_scratch[wid] = val;
    }
    __syncthreads();

    const int NUM_WARPS = (BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    T block_sum = (threadIdx.x < NUM_WARPS) ? shared_scratch[lane] : T(0);

    if (wid == 0) {
        const int TAIL_LANES = (NUM_WARPS < WARP_SIZE) ? NUM_WARPS : WARP_SIZE;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(FULL_MASK, block_sum, offset);
            if (lane + offset < TAIL_LANES) {
                block_sum += other;
            }
        }
    }
    return block_sum;
}

// Block-level max reduction using shared memory for warp leaders.
//
// The contract matches block_reduce_sum: shared_scratch must hold at least
// ceil(BLOCK_THREADS/WARP_SIZE) elements, BLOCK_THREADS must equal blockDim.x,
// every thread of the block must call this because it synchronizes, and the
// block maximum is returned in lane 0 of warp 0, i.e. the thread with
// threadIdx.x == 0. What every other thread receives is unspecified.
//
// No identity element is needed. The lanes that have no warp of their own in
// the second stage repeat shared_scratch[0], which is a real warp maximum, so
// folding it in again leaves the result unchanged. That keeps this correct for
// integer and half types, which have no portable -INFINITY.
template <typename T, int BLOCK_THREADS>
__device__ __forceinline__ T block_reduce_max(T val, T* shared_scratch) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    const int warp_lanes = warp_lane_count(BLOCK_THREADS, wid);
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(FULL_MASK, val, offset);
        if (lane + offset < warp_lanes && other > val) {
            val = other;
        }
    }

    if (lane == 0) {
        shared_scratch[wid] = val;
    }
    __syncthreads();

    const int NUM_WARPS = (BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    T block_max = (threadIdx.x < NUM_WARPS) ? shared_scratch[lane] : shared_scratch[0];

    if (wid == 0) {
        const int TAIL_LANES = (NUM_WARPS < WARP_SIZE) ? NUM_WARPS : WARP_SIZE;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(FULL_MASK, block_max, offset);
            if (lane + offset < TAIL_LANES && other > block_max) {
                block_max = other;
            }
        }
    }
    return block_max;
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
