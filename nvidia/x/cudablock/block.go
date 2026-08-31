package cudablock

// Header is the embedded CUDA C++ header defining warp and block collective primitives.
const Header = `
#pragma once

#ifndef CUDABLOCK_CUH
#define CUDABLOCK_CUH

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
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int idx = tid; idx < TILE_M * TILE_N; idx += BLOCK_THREADS) {
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        if (r < max_rows && c < max_cols) {
            shared_tile[r][c] = global_src[r * stride + c];
        } else {
            shared_tile[r][c] = T(0);
        }
    }
    __syncthreads();
}

#endif // CUDABLOCK_CUH
`

// ReductionOp specifies an associative collective reduction operation.
type ReductionOp int

const (
	OpSum ReductionOp = iota
	OpMax
	OpMin
	OpProd
)

// String returns the string representation of the reduction op.
func (op ReductionOp) String() string {
	switch op {
	case OpSum:
		return "Sum"
	case OpMax:
		return "Max"
	case OpMin:
		return "Min"
	case OpProd:
		return "Prod"
	default:
		return "Unknown"
	}
}

// TileConfig describes block dimensions and shared-memory tile dimensions.
type TileConfig struct {
	BlockThreads int
	TileM        int
	TileN        int
	TileK        int
	ElemBytes    int
}

// SharedMemBytes calculates the required dynamic shared memory in bytes for the tile configuration.
func (tc TileConfig) SharedMemBytes() int {
	return (tc.TileM*tc.TileK + tc.TileK*tc.TileN) * tc.ElemBytes
}
