#include <cuda_runtime.h>

extern "C" __global__ void fused_gate(
#line 17 "kernels.go"
    const float* __restrict__ x,
#line 18 "kernels.go"
    const float* __restrict__ w,
#line 19 "kernels.go"
    float* __restrict__ out_mul_3,
#line 7 "fused_gate.cu"
    int n
) {
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
#line 17 "kernels.go"
        float v_x = x[idx];
#line 18 "kernels.go"
        float v_w = w[idx];
#line 19 "kernels.go"
        float v_silu_2 = v_x / (1.0f + expf(-v_x));
#line 19 "kernels.go"
        float v_mul_3 = v_silu_2 * v_w;
#line 19 "kernels.go"
        out_mul_3[idx] = v_mul_3;
#line 15 "fused_gate.cu"
    }
}