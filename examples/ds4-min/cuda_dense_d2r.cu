#include <cuda_fp16.h>

namespace ggml_cuda_mma {
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
    }
}

namespace {
namespace dq8 {

constexpr int kDqMTile   = 128;
constexpr int kDqNTile   = 128;
constexpr int kDqRowWarps = 8;
constexpr int kDqColWarps = 2;
constexpr int kDqWarps   = kDqRowWarps * kDqColWarps;
constexpr int kDqThreads = 32 * kDqWarps;
constexpr int kDqStages  = 3;
constexpr int kDqRowPad  = 80;  // 64 B payload + 16 B: 20-int ldmatrix stride rotates banks
constexpr int kDqNFrag   = kDqNTile / 8;
constexpr int kDqNFragPerWarp = kDqNFrag / kDqColWarps;

constexpr size_t kDqSmemWQBytes  = (size_t)kDqStages * kDqMTile * kDqRowPad;
constexpr size_t kDqSmemQ8QBytes = (size_t)kDqStages * kDqNTile * kDqRowPad;
constexpr size_t kDqSmemQ8HBytes = (size_t)kDqStages * kDqNTile * 16;
constexpr size_t kDqSmemWDBytes  = (size_t)kDqStages * kDqMTile * sizeof(uint32_t);
constexpr size_t kDqSmemWQOff  = 0;
constexpr size_t kDqSmemQ8QOff = kDqSmemWQOff + kDqSmemWQBytes;
constexpr size_t kDqSmemQ8HOff = kDqSmemQ8QOff + kDqSmemQ8QBytes;
constexpr size_t kDqSmemWDOff  = kDqSmemQ8HOff + kDqSmemQ8HBytes;
constexpr size_t kDqSmemTotalBytes = kDqSmemWDOff + kDqSmemWDBytes;
static_assert(kDqSmemTotalBytes <= 99ull * 1024ull, "dense D2R dynamic smem exceeds sm limit");

struct DenseQ8Params {
    const char *wd_row0;       // dq plane at cta_row0 (row stride nb*2 = K/16)
    const char *wq_row0;       // qs plane at cta_row0 (row stride K)
    const char *q8_tile;       // q8 blocks at col_lo
    uint32_t q8_k128_stride_bytes;
    uint32_t wq_row_stride;    // K bytes
    int k64_iters;
    int k128_iters;
};

__device__ __forceinline__ void dq8_cp_async_4B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"(smem), "l"(src));
    } else {
        *reinterpret_cast<uint32_t *>(dst) = 0u;
    }
#else
    if (pred) {
        *reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(src);
    } else {
        *reinterpret_cast<uint32_t *>(dst) = 0u;
    }
#endif
}

__device__ __forceinline__ void dq8_issue_q8_qs(
        int8_t (&s_q8q)[kDqStages][kDqNTile][kDqRowPad],
        const DenseQ8Params &p, int k64_iter) {
    const bool pred = k64_iter < p.k64_iters;
    const char *base = p.q8_tile +
                       (uint64_t)(k64_iter >> 1) * p.q8_k128_stride_bytes +
                       16u + (uint64_t)(k64_iter & 1) * 64u;
    const int buf = k64_iter % kDqStages;
    const int chunk = d2r_tid();
    const int col = chunk >> 2;
    const int h = chunk & 3;
    void *dst = &s_q8q[buf][col][h * 16];
    const void *src = base + (uint64_t)col * sizeof(block_q8_1_mmq) + (uint64_t)h * 16u;
    cp_async_16B(dst, src, pred);
}

__device__ __forceinline__ void dq8_issue_q8_hdr(
        float (&s_q8h)[kDqStages][kDqNTile][4],
        const DenseQ8Params &p, int k128_iter) {
    if (d2r_tid() >= kDqNTile) {
        return;
    }
    const bool pred = k128_iter < p.k128_iters;
    const char *base = p.q8_tile + (uint64_t)k128_iter * p.q8_k128_stride_bytes;
    const int col = d2r_tid();
    void *dst = &s_q8h[k128_iter % kDqStages][col][0];
    const void *src = base + (uint64_t)col * sizeof(block_q8_1_mmq);
    cp_async_16B(dst, src, pred);
}

__device__ __forceinline__ void dq8_issue_w(
        int8_t (&s_wq)[kDqStages][kDqMTile][kDqRowPad],
        uint32_t (&s_wd)[kDqStages][kDqMTile],
        const DenseQ8Params &p, int k64_iter) {
    const bool pred = k64_iter < p.k64_iters;
    const int wst = k64_iter % kDqStages;
    const char *wq = p.wq_row0 + (uint64_t)k64_iter * 64u;
    const char *wd = p.wd_row0 + (uint64_t)k64_iter * 4u;
    const uint32_t wq_row_stride = p.wq_row_stride;
    const uint32_t wd_row_stride = p.wq_row_stride >> 4;  // nb*2 = K/16
    const int chunk = d2r_tid();
    const int row = chunk >> 2;
    const int h = chunk & 3;
    void *dst = &s_wq[wst][row][h * 16];
    const void *src = wq + (uint64_t)row * wq_row_stride + (uint64_t)h * 16u;
    cp_async_16B(dst, src, pred);
    if (d2r_tid() < kDqMTile) {
        void *ddst = &s_wd[wst][d2r_tid()];
        const void *dsrc = wd + (uint64_t)d2r_tid() * wd_row_stride;
        dq8_cp_async_4B(ddst, dsrc, pred);
    }
}

template <typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void dq8_mainloop(
        float (&acc)[kDqNFragPerWarp][4],
        float (&s_q8h)[kDqStages][kDqNTile][4],
        int8_t (&s_q8q)[kDqStages][kDqNTile][kDqRowPad],
        int8_t (&s_wq)[kDqStages][kDqMTile][kDqRowPad],
        uint32_t (&s_wd)[kDqStages][kDqMTile],
        const DenseQ8Params &p) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int k64_iters = p.k64_iters;
    const int lane = d2r_lane();
    const int group = lane >> 2;
    const int wrow = (d2r_warp() >> 1) * 16;
    const int nf0 = (d2r_warp() & 1) * kDqNFragPerWarp;
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);

    dq8_issue_w(s_wq, s_wd, p, 0);
    dq8_issue_q8_qs(s_q8q, p, 0);
    dq8_issue_q8_hdr(s_q8h, p, 0);
    cp_async_commit();
    dq8_issue_w(s_wq, s_wd, p, 1);
    dq8_issue_q8_qs(s_q8q, p, 1);
    dq8_issue_q8_hdr(s_q8h, p, 1);
    cp_async_commit();

    for (int i = 0; i < k64_iters; ++i) {
        // wait -> barrier -> issue: the barrier both publishes stage i CTA-wide
        // and proves all warps finished stage i-1, whose buffer ((i-1)%3 ==
        // (i+2)%3) the issues below overwrite.
        cp_async_wait_group<1>();
        __syncthreads();
        dq8_issue_w(s_wq, s_wd, p, i + 2);
        dq8_issue_q8_qs(s_q8q, p, i + 2);
        if ((i & 1) == 0) {
            dq8_issue_q8_hdr(s_q8h, p, (i >> 1) + 2);
        }
        cp_async_commit();

        const int wst = i % kDqStages;
        const int qbuf = i % kDqStages;
        const int hbuf = (i >> 1) % kDqStages;
        const uint32_t dw0_bits = s_wd[wst][wrow + group];
        const uint32_t dw1_bits = s_wd[wst][wrow + group + 8];
        const float2 dw0 = __half22float2(*reinterpret_cast<const half2 *>(&dw0_bits));
        const float2 dw1 = __half22float2(*reinterpret_cast<const half2 *>(&dw1_bits));

#pragma unroll
        for (int t = 0; t < 2; ++t) {
            TileA A;
            ggml_cuda_mma::load_ldmatrix(
                A, reinterpret_cast<const int *>(&s_wq[wst][wrow][t * 32]),
                kDqRowPad / (int)sizeof(int));
            const int tq8 = (i & 1) * 2 + t;
            const float dwr0 = t ? dw0.y : dw0.x;
            const float dwr1 = t ? dw1.y : dw1.x;
#pragma unroll
            for (int nf = 0; nf < kDqNFragPerWarp; ++nf) {
                TileB B;
                TileC C;
                ggml_cuda_mma::load_ldmatrix(
                    B, reinterpret_cast<const int *>(&s_q8q[qbuf][(nf0 + nf) * 8][t * 32]),
                    kDqRowPad / (int)sizeof(int));
                ggml_cuda_mma::mma(C, A, B);
                const float da0 = s_q8h[hbuf][(nf0 + nf) * 8 + c0][tq8];
                const float da1 = s_q8h[hbuf][(nf0 + nf) * 8 + c1][tq8];
                acc[nf][0] = fmaf((float)C.x[0], dwr0 * da0, acc[nf][0]);
                acc[nf][1] = fmaf((float)C.x[1], dwr0 * da1, acc[nf][1]);
                acc[nf][2] = fmaf((float)C.x[2], dwr1 * da0, acc[nf][2]);
                acc[nf][3] = fmaf((float)C.x[3], dwr1 * da1, acc[nf][3]);
            }
        }
    }
}

} // namespace dq8
} // anonymous namespace

extern "C" __global__ __launch_bounds__(dq8::kDqThreads, 1)
void ds4_dense_q8_d2r(const char * __restrict__ wd_plane,
                      const char * __restrict__ wq_plane,
                      const block_q8_1_mmq * __restrict__ q8,
                      float * __restrict__ out,
                      int M, int N, int K, int group_m, int out_stride) {
#if defined(TURING_MMA_AVAILABLE)
    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    extern __shared__ __align__(16) char dq8_smem[];
    auto &s_wq  = *reinterpret_cast<int8_t (*)[dq8::kDqStages][dq8::kDqMTile][dq8::kDqRowPad]>(dq8_smem + dq8::kDqSmemWQOff);
    auto &s_q8q = *reinterpret_cast<int8_t (*)[dq8::kDqStages][dq8::kDqNTile][dq8::kDqRowPad]>(dq8_smem + dq8::kDqSmemQ8QOff);
    auto &s_q8h = *reinterpret_cast<float (*)[dq8::kDqStages][dq8::kDqNTile][4]>(dq8_smem + dq8::kDqSmemQ8HOff);
    auto &s_wd  = *reinterpret_cast<uint32_t (*)[dq8::kDqStages][dq8::kDqMTile]>(dq8_smem + dq8::kDqSmemWDOff);

    if (out_stride <= 0) {
        out_stride = M;
    }

    const int num_m = M / dq8::kDqMTile;
    const int num_n = (N + dq8::kDqNTile - 1) / dq8::kDqNTile;
    const int width = group_m * num_n;
    const int g = (int)blockIdx.x / width;
    const int rem = (int)blockIdx.x - g * width;
    int gsize = num_m - g * group_m;
    if (gsize > group_m) {
        gsize = group_m;
    }
    const int pid_m = g * group_m + rem % gsize;
    const int pid_n = rem / gsize;

    const int cta_row0 = pid_m * dq8::kDqMTile;
    const int col_lo = pid_n * dq8::kDqNTile;

    dq8::DenseQ8Params p;
    p.wd_row0 = wd_plane + (uint64_t)cta_row0 * (uint64_t)(K >> 4);
    p.wq_row0 = wq_plane + (uint64_t)cta_row0 * (uint64_t)K;
    p.q8_tile = reinterpret_cast<const char *>(q8) +
                (uint64_t)col_lo * sizeof(block_q8_1_mmq);
    p.q8_k128_stride_bytes = (uint32_t)((uint64_t)N * sizeof(block_q8_1_mmq));
    p.wq_row_stride = (uint32_t)K;
    p.k64_iters = K >> 6;
    p.k128_iters = K >> 7;

    float acc[dq8::kDqNFragPerWarp][tile_C::ne] = {};

    dq8::dq8_mainloop<tile_A, tile_B, tile_C>(acc, s_q8h, s_q8q, s_wq, s_wd, p);

    // Column-major out [N][M]; rows always in range (M % 128 == 0 validated).
    // The isfinite guard preserves the sanitize contract of the mmq path.
    const int out_col_lo = col_lo + (d2r_warp() & 1) * (dq8::kDqNFragPerWarp * 8);
    const int out_row0 = cta_row0 + ((d2r_warp() >> 1) << 4);
#pragma unroll
    for (int nf = 0; nf < dq8::kDqNFragPerWarp; ++nf) {
        const int col_frag0 = out_col_lo + nf * 8;
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = out_row0 + tile_C::get_i(l);
            const int col = col_frag0 + tile_C::get_j(l);
            if (col < N) {
                const float v = isfinite(acc[nf][l]) ? acc[nf][l] : 0.0f;
                out[(uint64_t)col * (uint64_t)out_stride + (uint64_t)row] = v;
            }
        }
    }
#else
    (void)wd_plane; (void)wq_plane; (void)q8; (void)out;
    (void)M; (void)N; (void)K; (void)group_m; (void)out_stride;
#endif
}

extern "C" __global__ void ds4_repack_q8_0_aligned(
        __half *dq,
        unsigned char *qs,
        const unsigned char *raw,
        uint64_t nblk) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 2ull) return;
    const uint64_t blk = i >> 1;
    const uint32_t p = (uint32_t)(i & 1u);
    const unsigned char *src = raw + blk * 34ull;
    if (p == 0u) {
        uint16_t h;
        memcpy(&h, src, 2u);
        dq[blk] = __ushort_as_half(h);
    }
    memcpy(qs + blk * 32ull + p * 16ull, src + 2u + p * 16ull, 16u);
}
