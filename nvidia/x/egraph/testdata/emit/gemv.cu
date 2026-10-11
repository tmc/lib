// expression: "(gemv a matrix)"
// fingerprint: "(gemv a{string:a} matrix{string:matrix})"
// cost model: "AstSizeCost", cost: 3
// rule suites: []
// extraction stats: work=0 nodes=0 exhausted=false

typedef unsigned char uint8_t;
template <typename T, int N> struct tvec {
    T values[N];
    __device__ T& operator[](int i) { return values[i]; }
    __device__ const T& operator[](int i) const { return values[i]; }
};

// e-class 0: "a", from "input"
__device__ auto eg_c0_a(const tvec<float, 9>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 3> {
    tvec<float, 3> output;
    for (int i = 0; i < 3; ++i) {
        output[i] = input[0 + i];
    }
    return output;
}

// e-class 2: "matrix", from "input"
__device__ auto eg_c2_matrix(const tvec<float, 9>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 6> {
    tvec<float, 6> output;
    for (int i = 0; i < 6; ++i) {
        output[i] = input[3 + i];
    }
    return output;
}

// e-class 3: "gemv", from "input"
__device__ auto eg_c3_gemv(const tvec<float, 9>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 2> {
    auto arg0 = eg_c0_a(input, params, fwd_ctx);
    auto arg1 = eg_c2_matrix(input, params, fwd_ctx);
    tvec<float, 2> output;
    for (int row = 0; row < 2; ++row) {
        float dot = 0.0f;
        for (int i = 0; i < 3; ++i) {
            dot += arg1[row * 3 + i] * arg0[i];
        }
        output[row] = dot;
    }
    return output;
}

extern "C" __global__ void chain(const float* a, const float* matrix, float* eg_output) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 || threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }
    tvec<float, 9> eg_input;
    for (int i = 0; i < 3; ++i) {
        eg_input[0 + i] = a[i];
    }
    for (int i = 0; i < 6; ++i) {
        eg_input[3 + i] = matrix[i];
    }
    auto eg_result = eg_c3_gemv(eg_input, nullptr, nullptr);
    for (int i = 0; i < 2; ++i) {
        eg_output[i] = eg_result[i];
    }
}
