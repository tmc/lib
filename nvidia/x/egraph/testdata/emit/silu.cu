// expression: "(silu a)"
// fingerprint: "(silu a{string:a})"
// cost model: "AstSizeCost", cost: 2
// rule suites: []
// extraction stats: work=0 nodes=0 exhausted=false

typedef unsigned char uint8_t;
template <typename T, int N> struct tvec {
    T values[N];
    __device__ T& operator[](int i) { return values[i]; }
    __device__ const T& operator[](int i) const { return values[i]; }
};

// e-class 0: "a", from "input"
__device__ auto eg_c0_a(const tvec<float, 3>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 3> {
    tvec<float, 3> output;
    for (int i = 0; i < 3; ++i) {
        output[i] = input[0 + i];
    }
    return output;
}

// e-class 2: "silu", from "input"
__device__ auto eg_c2_silu(const tvec<float, 3>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 3> {
    auto arg0 = eg_c0_a(input, params, fwd_ctx);
    tvec<float, 3> output;
    for (int i = 0; i < 3; ++i) {
        output[i] = arg0[i] / (1.0f + expf(-arg0[i]));
    }
    return output;
}

extern "C" __global__ void chain(const float* a, float* eg_output) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 || threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }
    tvec<float, 3> eg_input;
    for (int i = 0; i < 3; ++i) {
        eg_input[0 + i] = a[i];
    }
    auto eg_result = eg_c2_silu(eg_input, nullptr, nullptr);
    for (int i = 0; i < 3; ++i) {
        eg_output[i] = eg_result[i];
    }
}
