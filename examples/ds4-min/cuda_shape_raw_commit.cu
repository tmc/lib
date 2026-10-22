// Standalone NVRTC source: no host system headers are required for integers.
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
static_assert(sizeof(uint32_t) == 4 && sizeof(uint64_t) == 8, "integer widths");
static_assert(uint32_t(-1) > 0 && uint64_t(-1) > 0, "integer signedness");
#include <cuda_fp16.h>

__device__ static float dsv4_e4m3fn_value_dev(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * dsv4_e4m3fn_value_dev(best);
}

__device__ static void fp8_kv_quantize_row(
        float    *xr,
        uint32_t  head_dim,
        uint32_t  n_rot,
        float    *scratch) {
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            xr[off + tid] = q;
        }
        __syncthreads();
    }
}


extern "C" __global__ void ds4_shape_raw_commit(float *kv, float *raw, uint32_t raw_cap, uint32_t position, uint32_t head_dim, uint32_t n_rot) {
    __shared__ float scratch[64];
    fp8_kv_quantize_row(kv, head_dim, n_rot, scratch);
    uint32_t slot = position % raw_cap;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        raw[(uint64_t)slot * head_dim + d] = __half2float(__float2half(kv[d]));
    }
}
