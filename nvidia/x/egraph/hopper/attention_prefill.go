package hopper

import (
	"fmt"
	"math"
	"strings"
)

// A PrefillAttention is causal attention for a chunk of prompt rows over a
// paged BF16 KV cache, on mma.sync tensor cores (sm_80 and later). The
// chunk's n rows sit at positions start .. start+n-1, and row i attends the
// cache's positions 0 .. start+i, which must already hold the chunk's own
// keys and values.
//
// The kernel takes
//
//	q        BF16 [n, NumHeads, HeadDim]
//	k_cache  BF16 [slots, KVHeads, HeadDim]
//	v_cache  BF16 [slots, KVHeads, HeadDim]
//	page_table int32 [MaxPages]: position p is in slot
//	         page_table[p / PageSize]·PageSize + p % PageSize
//	start, n int32
//	out      BF16 [n, NumHeads, HeadDim]
//
// Each block of 128 threads serves 64 rows of one query head, 16 per warp.
// It walks the keys up to its last row 64 at a time: S = Q·Kᵀ on mma.sync,
// an online softmax in registers, and O += P·V with P reused as the A
// operand. Q, K and V pass through 48 KB of static shared memory in 128-byte
// XOR-swizzled rows, so ldmatrix reads are free of bank conflicts.
type PrefillAttention struct {
	Name     string
	NumHeads int // query heads, a multiple of KVHeads
	KVHeads  int
	HeadDim  int     // 64 or 128
	PageSize int     // positions per page
	MaxPages int     // entries in page_table
	Scale    float32 // softmax scale; zero means 1/sqrt(HeadDim)
}

// prefillAttnRows is the query rows each PrefillAttention block serves, and
// the keys it takes per step.
const prefillAttnRows = 64

func (a PrefillAttention) norm() (PrefillAttention, error) {
	if a.Name == "" {
		return a, fmt.Errorf("prefill attention: no name")
	}
	if a.KVHeads < 1 || a.NumHeads < a.KVHeads || a.NumHeads%a.KVHeads != 0 {
		return a, fmt.Errorf("prefill attention %s: %d query heads over %d KV heads", a.Name, a.NumHeads, a.KVHeads)
	}
	if a.HeadDim != 64 && a.HeadDim != 128 {
		return a, fmt.Errorf("prefill attention %s: head dim %d, want 64 or 128", a.Name, a.HeadDim)
	}
	if a.PageSize < 1 || a.MaxPages < 1 {
		return a, fmt.Errorf("prefill attention %s: page size %d, %d pages", a.Name, a.PageSize, a.MaxPages)
	}
	if a.Scale == 0 {
		a.Scale = float32(1 / math.Sqrt(float64(a.HeadDim)))
	}
	return a, nil
}

// Grid returns the launch shape for a chunk of n rows.
func (a PrefillAttention) Grid(n int) (grid, block [3]int) {
	return [3]int{(n + prefillAttnRows - 1) / prefillAttnRows, a.NumHeads, 1}, [3]int{128, 1, 1}
}

// CUDA returns the kernel's source, which needs [Prelude].
func (a PrefillAttention) CUDA() (string, error) {
	a, err := a.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, `// %[1]s: causal prefill attention, %[2]d query heads over %[3]d KV heads of %[4]d, mma.sync.
__global__ void __launch_bounds__(128) %[1]s(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    int start,
    int n,
    uint16_t* __restrict__ out) {
    constexpr int HQ = %[2]d, HKV = %[3]d, G = HQ / HKV, D = %[4]d, PAGE = %[5]d;
    constexpr int BQ = 64, BKV = 64, DC = D / 8, KS = D / 16, NTD = D / 8;
    constexpr float SCALE_LOG2 = %[6]v * 1.4426950408889634f;
    const int h = blockIdx.y, kvh = h / G, r0 = blockIdx.x * BQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (r0 >= n) return;
    const int rows = min(BQ, n - r0);
    const int kv_end = start + r0 + rows; // keys of positions < kv_end
    alignas(128) __shared__ uint16_t sq[BQ * D];
    alignas(128) __shared__ uint16_t sk[BKV * D];
    alignas(128) __shared__ uint16_t sv[BKV * D];
    // Row r's 16-byte chunk c lives at chunk c ^ (r & 7).
    auto at = [](int r, int c) { return r * D + ((c ^ (r & 7)) * 8); };

    for (int i = tid; i < BQ * DC; i += 128) {
        const int r = i / DC, c = i %% DC;
        const int row = r0 + min(r, rows - 1);
        eg_cp_async16(&sq[at(r, c)], q + ((size_t)row * HQ + h) * D + c * 8);
    }
    eg_cp_async_commit();
    eg_cp_async_wait<0>();
    __syncthreads();
    uint32_t qf[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) eg_ldmatrix_x4(qf[ks], &sq[at(warp * 16 + lane %% 16, ks * 2 + lane / 16)]);

    // This thread holds rows g and g + 8 of the warp's 16.
    const int g = lane / 4, t = lane %% 4;
    const int pos0 = start + r0 + warp * 16 + g, pos1 = pos0 + 8;
    const float NEG_INF = -__uint_as_float(0x7f800000u);
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
    float o[NTD][4] = {};
    for (int kv0 = 0; kv0 < kv_end; kv0 += BKV) {
        __syncthreads();
        for (int i = tid; i < BKV * DC; i += 128) {
            const int r = i / DC, c = i %% DC;
            const int p = min(kv0 + r, kv_end - 1);
            const size_t off = ((size_t)(page_table[p / PAGE] * PAGE + p %% PAGE) * HKV + kvh) * D + c * 8;
            eg_cp_async16(&sk[at(r, c)], k_cache + off);
            eg_cp_async16(&sv[at(r, c)], v_cache + off);
        }
        eg_cp_async_commit();
        eg_cp_async_wait<0>();
        __syncthreads();

        float s[BKV / 8][4] = {};
#pragma unroll
        for (int ks = 0; ks < KS; ++ks)
#pragma unroll
            for (int np = 0; np < BKV / 16; ++np) {
                uint32_t t4[4];
                eg_ldmatrix_x4(t4, &sk[at(np * 16 + lane %% 8 + 8 * (lane / 16), ks * 2 + (lane / 8) %% 2)]);
                const uint32_t b0[2] = {t4[0], t4[1]}, b1[2] = {t4[2], t4[3]};
                eg_mma_bf16(s[2 * np], qf[ks], b0, s[2 * np]);
                eg_mma_bf16(s[2 * np + 1], qf[ks], b1, s[2 * np + 1]);
            }

        // Mask keys after each row's position; scale into log2 units.
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nt = 0; nt < BKV / 8; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int key = kv0 + nt * 8 + 2 * t + (i & 1);
                const bool hi = i >= 2;
                const float v = key <= (hi ? pos1 : pos0) ? s[nt][i] * SCALE_LOG2 : NEG_INF;
                s[nt][i] = v;
                if (hi) mx1 = fmaxf(mx1, v); else mx0 = fmaxf(mx0, v);
            }
#pragma unroll
        for (int x = 1; x < 4; x <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, x));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, x));
        }
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        // A row with no visible key yet keeps base 0 so exp2 stays finite.
        const float b0 = n0 == NEG_INF ? 0.0f : n0, b1 = n1 == NEG_INF ? 0.0f : n1;
        const float c0 = exp2f(m0 - b0), c1 = exp2f(m1 - b1);
        m0 = n0, m1 = n1;
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < BKV / 8; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - b0), s[nt][1] = exp2f(s[nt][1] - b0);
            s[nt][2] = exp2f(s[nt][2] - b1), s[nt][3] = exp2f(s[nt][3] - b1);
            sum0 += s[nt][0] + s[nt][1], sum1 += s[nt][2] + s[nt][3];
        }
        l0 = l0 * c0 + sum0, l1 = l1 * c1 + sum1;
#pragma unroll
        for (int nt = 0; nt < NTD; ++nt) o[nt][0] *= c0, o[nt][1] *= c0, o[nt][2] *= c1, o[nt][3] *= c1;

        // O += P·V: P's accumulators for keys 16kk .. 16kk+15 are an A fragment.
#pragma unroll
        for (int kk = 0; kk < BKV / 16; ++kk) {
            const uint32_t pa[4] = {
                eg_pack_bf16x2(s[2 * kk][0], s[2 * kk][1]), eg_pack_bf16x2(s[2 * kk][2], s[2 * kk][3]),
                eg_pack_bf16x2(s[2 * kk + 1][0], s[2 * kk + 1][1]), eg_pack_bf16x2(s[2 * kk + 1][2], s[2 * kk + 1][3]),
            };
#pragma unroll
            for (int np = 0; np < NTD / 2; ++np) {
                uint32_t t4[4];
                eg_ldmatrix_x4_trans(t4, &sv[at(kk * 16 + lane %% 8 + 8 * ((lane / 8) %% 2), np * 2 + lane / 16)]);
                const uint32_t v0[2] = {t4[0], t4[1]}, v1[2] = {t4[2], t4[3]};
                eg_mma_bf16(o[2 * np], pa, v0, o[2 * np]);
                eg_mma_bf16(o[2 * np + 1], pa, v1, o[2 * np + 1]);
            }
        }
    }
#pragma unroll
    for (int x = 1; x < 4; x <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, x);
        l1 += __shfl_xor_sync(0xffffffffu, l1, x);
    }
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    const int row0 = r0 + warp * 16 + g, row1 = row0 + 8;
#pragma unroll
    for (int nt = 0; nt < NTD; ++nt) {
        const int d = nt * 8 + 2 * t;
        if (row0 < n) *reinterpret_cast<uint32_t*>(out + ((size_t)row0 * HQ + h) * D + d) = eg_pack_bf16x2(o[nt][0] * i0, o[nt][1] * i0);
        if (row1 < n) *reinterpret_cast<uint32_t*>(out + ((size_t)row1 * HQ + h) * D + d) = eg_pack_bf16x2(o[nt][2] * i1, o[nt][3] * i1);
    }
}
`, a.Name, a.NumHeads, a.KVHeads, a.HeadDim, a.PageSize, a.Scale)
	return b.String(), nil
}
