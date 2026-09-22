#include <cuda_runtime.h>

extern "C" __global__ void fused_layered(
#line 38 "x/examples/source-wiring/kernels.go"
    const float* __restrict__ x,
#line 39 "x/examples/source-wiring/kernels.go"
    const float* __restrict__ w,
#line 50 "x/examples/source-wiring/kernels.go"
    float* __restrict__ out_div_4,
#line 7 "fused_layered.cu"
    int n
) {
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
#line 38 "x/examples/source-wiring/kernels.go"
        float v_x = x[idx];
#line 39 "x/examples/source-wiring/kernels.go"
        float v_w = w[idx];
#line 45 "x/examples/source-wiring/kernels.go"
        float v_silu_2 = v_x / (1.0f + expf(-v_x));
#line 49 "x/examples/source-wiring/kernels.go"
        float v_sqrt_3 = sqrtf(v_w);
#line 50 "x/examples/source-wiring/kernels.go"
        float v_div_4 = v_silu_2 / v_sqrt_3;
#line 50 "x/examples/source-wiring/kernels.go"
        out_div_4[idx] = v_div_4;
#line 16 "fused_layered.cu"
    }
}