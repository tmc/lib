#include <cuda_runtime.h>

extern "C" __global__ void fused_gate(
#line 12 "x/cudalineinfo/testdata/wiring/kernels.go"
    const float* __restrict__ x,
#line 13 "x/cudalineinfo/testdata/wiring/kernels.go"
    const float* __restrict__ w,
#line 15 "x/cudalineinfo/testdata/wiring/kernels.go"
    float* __restrict__ out_mul_3,
#line 7 "fused_gate.cu"
    int n
) {
    if (n <= 0 || blockDim.x == 0 || gridDim.x == 0 || blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;
    for (unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x; idx < (unsigned int)n; idx += (unsigned long long)blockDim.x * gridDim.x) {
#line 12 "x/cudalineinfo/testdata/wiring/kernels.go"
        float v_x = x[idx];
#line 13 "x/cudalineinfo/testdata/wiring/kernels.go"
        float v_w = w[idx];
#line 14 "x/cudalineinfo/testdata/wiring/kernels.go"
        float v_silu_2 = v_x / (1.0f + expf(-v_x));
#line 15 "x/cudalineinfo/testdata/wiring/kernels.go"
        float v_mul_3 = v_silu_2 * v_w;
#line 15 "x/cudalineinfo/testdata/wiring/kernels.go"
        out_mul_3[idx] = v_mul_3;
#line 16 "fused_gate.cu"
    }
}