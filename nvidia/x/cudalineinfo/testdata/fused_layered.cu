#include <cuda_runtime.h>

extern "C" __global__ void fused_layered(
#line 38 "x/cudalineinfo/testdata/wiring/kernels.go"
    const float* __restrict__ x,
#line 39 "x/cudalineinfo/testdata/wiring/kernels.go"
    const float* __restrict__ w,
#line 50 "x/cudalineinfo/testdata/wiring/kernels.go"
    float* __restrict__ out_div_4,
#line 7 "fused_layered.cu"
    int n
) {
    if (n <= 0 || blockDim.x == 0 || gridDim.x == 0 || blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;
    for (unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x; idx < (unsigned int)n; idx += (unsigned long long)blockDim.x * gridDim.x) {
#line 38 "x/cudalineinfo/testdata/wiring/kernels.go"
        float v_x = x[idx];
#line 39 "x/cudalineinfo/testdata/wiring/kernels.go"
        float v_w = w[idx];
#line 45 "x/cudalineinfo/testdata/wiring/kernels.go"
        float v_silu_2 = v_x / (1.0f + expf(-v_x));
#line 49 "x/cudalineinfo/testdata/wiring/kernels.go"
        float v_sqrt_3 = sqrtf(v_w);
#line 50 "x/cudalineinfo/testdata/wiring/kernels.go"
        float v_div_4 = v_silu_2 / v_sqrt_3;
#line 50 "x/cudalineinfo/testdata/wiring/kernels.go"
        out_div_4[idx] = v_div_4;
#line 17 "fused_layered.cu"
    }
}