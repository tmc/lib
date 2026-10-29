package qualify

import (
	"fmt"
	"strings"
)

// DeterministicCUDA returns CUDA C++ source for the deterministic-mode
// kernels, written so that cudahostemu and nvcc both build it:
//
//   - qd_stochastic_round: the twin of [EmitDeterministicStochasticRoundPTX]
//     with the rounding step done in software as [StochasticRoundE2M1x4]
//     models it, so it runs on targets without cvt.rs (sm_121a);
//   - qd_det_gemv: y[r][c] = x[r]·w[c] with K split into kchunk-element
//     chunks spread over blockDim.x threads (the split count) and the chunk
//     partials added in the [FixedTreeSum] tree, so the result is the same
//     for every split count; grid (N, rows), at most 256 chunks;
//   - qd_naive_splitk_gemv: the same product with each thread summing a
//     contiguous run of chunks and adding its total with a float atomicAdd,
//     the split-K reduction deterministic mode forbids. y must start zeroed.
//
// All products and sums use __fmul_rn and __fadd_rn, so the compiler cannot
// contract them and results match the Go reference bit for bit.
func DeterministicCUDA(cfg DeterministicConfig) (string, error) {
	rounds := cfg.rounds()
	if rounds < 1 || rounds > 16 {
		return "", fmt.Errorf("deterministic cuda: PhiloxRounds %d out of range 1..16", rounds)
	}
	key := StochasticRoundKey(cfg.GlobalSeed, 0)
	src := strings.NewReplacer(
		"$ROUNDS", fmt.Sprint(rounds),
		"$KEY0", fmt.Sprintf("%#xu", key[0]),
		"$KEY1", fmt.Sprintf("%#xu", key[1]),
	).Replace(deterministicCUDASrc)
	return src, nil
}

const deterministicCUDASrc = `
#include <cstdint>

__device__ __forceinline__ uint32_t qd_mulhi(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * b) >> 32);
}

// qd_philox runs Philox4x32 in place on c with key (k0, k1).
__device__ __forceinline__ void qd_philox(uint32_t c[4], uint32_t k0, uint32_t k1) {
    for (int r = 0; r < $ROUNDS; ++r) {
        if (r) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        uint32_t hi0 = qd_mulhi(0xD2511F53u, c[0]), lo0 = 0xD2511F53u * c[0];
        uint32_t hi1 = qd_mulhi(0xCD9E8D57u, c[2]), lo1 = 0xCD9E8D57u * c[2];
        uint32_t n0 = hi1 ^ c[1] ^ k0, n2 = hi0 ^ c[3] ^ k1;
        c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
    }
}

// qd_sr_e2m1 rounds v to E2M1 stochastically with 8 random bits r.
__device__ __forceinline__ uint32_t qd_sr_e2m1(float v, uint32_t r) {
    if (v != v) return 7u;
    uint32_t sign = (__float_as_uint(v) >> 31) ? 8u : 0u;
    float a = __uint_as_float(__float_as_uint(v) & 0x7fffffffu);
    if (a >= 6.0f) return sign | 7u;
    float ulp = a >= 4.0f ? 2.0f : (a >= 2.0f ? 1.0f : 0.5f);
    float s = a / ulp;
    float q = floorf(s);
    uint32_t frac8 = (uint32_t)floorf((s - q) * 256.0f);
    if (frac8 + r >= 256u) q += 1.0f;
    float m = q * ulp;
    uint32_t code = m < 2.0f ? (uint32_t)(m * 2.0f) : (m < 4.0f ? (uint32_t)m + 2u : (m < 6.0f ? 6u : 7u));
    return code == 0 ? 0u : (sign | code);
}

__global__ void qd_stochastic_round(const float* in, uint16_t* out, uint32_t layer, uint32_t token0, uint32_t tokens, uint32_t cols) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t groups = cols / 4;
    if (i >= groups * tokens) return;
    uint32_t token = i / groups, col = (i % groups) * 4;
    uint32_t c[4] = {token0 + token, col, 0u, 0u};
    qd_philox(c, $KEY0 ^ layer, $KEY1);
    uint32_t rb = c[0];
    const float* v = in + token * cols + col;
    uint32_t d = qd_sr_e2m1(v[0], rb >> 24) << 12 | qd_sr_e2m1(v[1], (rb >> 8) & 0xffu) << 8 |
                 qd_sr_e2m1(v[2], (rb >> 16) & 0xffu) << 4 | qd_sr_e2m1(v[3], rb & 0xffu);
    out[i] = (uint16_t)d;
}

__device__ __forceinline__ float qd_chunk(const float* x, const float* w, int lo, int hi) {
    float acc = 0.0f;
    for (int i = lo; i < hi; ++i) acc = __fadd_rn(acc, __fmul_rn(x[i], w[i]));
    return acc;
}

__global__ void qd_det_gemv(const float* x, const float* w, float* y, int K, int N, int kchunk) {
    __shared__ float part[256];
    int c = blockIdx.x, r = blockIdx.y;
    int n = (K + kchunk - 1) / kchunk;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        int hi = (j + 1) * kchunk < K ? (j + 1) * kchunk : K;
        part[j] = qd_chunk(x + (size_t)r * K, w + (size_t)c * K, j * kchunk, hi);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int s = 1; s < n; s *= 2)
            for (int i = 0; i + s < n; i += 2 * s) part[i] = __fadd_rn(part[i], part[i + s]);
        y[(size_t)r * N + c] = part[0];
    }
}

__global__ void qd_naive_splitk_gemv(const float* x, const float* w, float* y, int K, int N, int kchunk) {
    int c = blockIdx.x, r = blockIdx.y;
    int n = (K + kchunk - 1) / kchunk;
    int per = (n + blockDim.x - 1) / blockDim.x;
    float acc = 0.0f;
    for (int j = threadIdx.x * per; j < n && j < (int)(threadIdx.x + 1) * per; ++j) {
        int hi = (j + 1) * kchunk < K ? (j + 1) * kchunk : K;
        acc = __fadd_rn(acc, qd_chunk(x + (size_t)r * K, w + (size_t)c * K, j * kchunk, hi));
    }
    atomicAdd(&y[(size_t)r * N + c], acc);
}
`
