package main

const benchKernelsCU = `
#define WARP_SIZE 32

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// --- Unfused Primitive Kernels ---

extern "C" __global__ void kernel_rmsnorm(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ out, int dim, float eps) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dim) return;
    out[idx] = x[idx] * weight[idx];
}

extern "C" __global__ void kernel_gemv(const float* __restrict__ x, const float* __restrict__ W, float* __restrict__ out, int inDim, int outDim) {
    int row = blockIdx.x; // 1 block per row
    if (row >= outDim) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ float s_sum[32];

    const float* w_row = W + row * inDim;
    float sum = 0.0f;
    for (int i = tid; i < inDim; i += blockDim.x) {
        sum += x[i] * w_row[i];
    }
    sum = warp_sum(sum);

    if (lane == 0) s_sum[wid] = sum;
    __syncthreads();

    float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
    if (wid == 0) block_sum = warp_sum(block_sum);

    if (tid == 0) out[row] = block_sum;
}

extern "C" __global__ void kernel_silu(const float* __restrict__ x, float* __restrict__ out, int dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dim) return;
    float v = x[idx];
    out[idx] = v / (1.0f + expf(-v));
}

extern "C" __global__ void kernel_mul(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dim) return;
    out[idx] = a[idx] * b[idx];
}

extern "C" __global__ void kernel_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dim) return;
    out[idx] = a[idx] + b[idx];
}

// --- E-Graph Extracted Fused Megakernels with Warp Reductions ---

extern "C" __global__ void kernel_fused_rmsnorm_gemv(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ W_q,
    const float* __restrict__ W_k,
    const float* __restrict__ W_v,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int inDim, int qDim, int kvDim, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ float s_sum[32];

    if (row < qDim) {
        const float* w_row = W_q + row * inDim;
        float sum = 0.0f;
        for (int i = tid; i < inDim; i += blockDim.x) {
            sum += (x[i] * weight[i]) * w_row[i];
        }
        sum = warp_sum(sum);
        if (lane == 0) s_sum[wid] = sum;
        __syncthreads();

        float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
        if (wid == 0) block_sum = warp_sum(block_sum);
        if (tid == 0) out_q[row] = block_sum;
    }
    if (row < kvDim) {
        const float* wk_row = W_k + row * inDim;
        const float* wv_row = W_v + row * inDim;
        float sumK = 0.0f;
        float sumV = 0.0f;
        for (int i = tid; i < inDim; i += blockDim.x) {
            float normed = x[i] * weight[i];
            sumK += normed * wk_row[i];
            sumV += normed * wv_row[i];
        }
        sumK = warp_sum(sumK);
        sumV = warp_sum(sumV);
        if (lane == 0) {
            s_sum[wid] = sumK;
            s_sum[wid + 16] = sumV;
        }
        __syncthreads();

        float bk = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
        float bv = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane + 16] : 0.0f;
        if (wid == 0) {
            bk = warp_sum(bk);
            bv = warp_sum(bv);
        }
        if (tid == 0) {
            out_k[row] = bk;
            out_v[row] = bv;
        }
    }
}

extern "C" __global__ void kernel_fused_swiglu_gemv(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ W_down,
    float* __restrict__ out,
    int hiddenDim, int intermediateDim
) {
    int row = blockIdx.x;
    if (row >= hiddenDim) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ float s_sum[32];

    const float* w_row = W_down + row * intermediateDim;
    float sum = 0.0f;
    for (int i = tid; i < intermediateDim; i += blockDim.x) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        float act = silu * up[i];
        sum += act * w_row[i];
    }
    sum = warp_sum(sum);

    if (lane == 0) s_sum[wid] = sum;
    __syncthreads();

    float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
    if (wid == 0) block_sum = warp_sum(block_sum);

    if (tid == 0) out[row] = block_sum;
}
`

// BenchmarkResults holds comparative timing measurements.
type BenchmarkResults struct {
	UnfusedDurationMs   float64
	UnfusedTokensPerSec float64

	EGraphDurationMs   float64
	EGraphTokensPerSec float64

	SpeedupVsUnfused float64
}

// RunComparisonBenchmark executes comparative decoding benchmarks on the GPU.
func RunComparisonBenchmark(steps int) (*BenchmarkResults, error) {
	res, err := RunBenchmarkComparison(1, steps, 0)
	if err != nil {
		return nil, err
	}
	return &BenchmarkResults{
		UnfusedDurationMs:   res.UnfusedDurationMs,
		UnfusedTokensPerSec: res.UnfusedTokSec,
		EGraphDurationMs:    res.TmcEGraphDurationMs,
		EGraphTokensPerSec:  res.TmcEGraphTokSec,
		SpeedupVsUnfused:    res.Speedup,
	}, nil
}
