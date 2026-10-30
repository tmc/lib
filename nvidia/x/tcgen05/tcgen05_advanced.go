package tcgen05

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// TCGen05AdvancedPrelude is the self-contained device preamble emitted for
// advanced sm_100a/sm_103a tcgen05 kernels (2-CTA cta_group::2 cooperative MMA,
// tcgen05.cp scale staging, TMEM-sourced operand A FlashAttention, and
// 4-warp 32-row epilogue datapath access).
const TCGen05AdvancedPrelude = `#ifndef EG_TCGEN05_ADVANCED_PRELUDE
#define EG_TCGEN05_ADVANCED_PRELUDE

#if (defined(EG_HOST_EMU) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000 || __CUDA_ARCH__ >= 1200))) && !defined(EG_EMU)
#define EG_EMU 1
#endif

#if defined(EG_HOST_EMU)
#define EG_TCGEN05_SHARED(T, name, dims) EG_EMU_SHARED(T, name, dims)
#define EG_TCGEN05_TMEM_BUF(name) EG_EMU_SHARED(float, name, [128 * 512])
#define EG_REGISTER_CLUSTER2(fn) \
    static const bool eg_reg_cluster2_##fn = eg_emu_register_cluster_kernel(fn, 2);
#elif defined(EG_EMU)
#define EG_TCGEN05_SHARED(T, name, dims) alignas(16) __shared__ T name dims
__device__ float eg_emu_dev_tmem_backing[128 * 512];
#define EG_TCGEN05_TMEM_BUF(name) float* name = eg_emu_dev_tmem_backing
#define EG_REGISTER_CLUSTER2(fn)
#else
#define EG_TCGEN05_SHARED(T, name, dims) alignas(16) __shared__ T name dims
#define EG_TCGEN05_TMEM_BUF(name) float* name = nullptr
#define EG_REGISTER_CLUSTER2(fn)
#endif

EG_DEV int eg_tcgen05_cluster_rank() {
#if defined(EG_HOST_EMU)
    return eg_emu_cluster_rank;
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint32_t r;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return (int)r;
#else
    return (int)(blockIdx.x & 1u);
#endif
}

EG_DEV void eg_tcgen05_cluster_sync() {
#if defined(EG_HOST_EMU)
    eg_emu_cluster_sync();
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("barrier.cluster.arrive.aligned;\n"
                 "barrier.cluster.wait.aligned;" ::: "memory");
#else
    __syncthreads();
#endif
}

template <typename T>
EG_DEV T* eg_tcgen05_map_shared_rank(T* p, int r) {
#if defined(EG_HOST_EMU)
    return eg_emu_map_shared_rank(p, r);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    T* m;
    asm("mapa.u64 %0, %1, %2;" : "=l"(m) : "l"(p), "r"((uint32_t)r));
    return m;
#else
    (void)r;
    return p;
#endif
}

// eg_elect_one_sync elects a single leader thread in the warp (PTX elect.sync).
EG_DEV bool eg_elect_one_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xffffffff;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}" : "=r"(pred));
    return pred != 0;
#else
    return (threadIdx.x & 31u) == 0u;
#endif
}

// eg_tcgen05_check_warp_row enforces the sm_100a/sm_103a 4-warp 32-row TMEM
// datapath partitioning rule: warp w in an epilogue warpgroup may only access
// rows [32*(w%4) .. 32*(w%4)+31].
EG_DEV void eg_tcgen05_check_warp_row(int warp, uint32_t row) {
#if defined(EG_HOST_EMU)
    const uint32_t lo = (uint32_t)(32 * (warp & 3));
    const uint32_t hi = lo + 32u;
    if (row < lo || row >= hi) {
        std::fprintf(stderr, "tcgen05: warp %d accessed row %u outside partition [%u, %u)\n",
                     warp, row, lo, hi);
        std::fflush(stderr);
        std::_Exit(4);
    }
#else
    (void)warp;
    (void)row;
#endif
}

EG_DEV void eg_tcgen05_alloc_cta1(float* tmem, uint32_t* slot, uint32_t offset, uint32_t cols) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)offset;
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(slot);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(s), "r"(cols) : "memory");
#else
    __syncwarp();
    if ((threadIdx.x & 31u) == 0u) *slot = offset;
    __syncwarp();
    for (uint32_t idx = threadIdx.x & 31u; idx < 128u * cols; idx += 32u) {
        const uint32_t r = idx / cols, c = idx % cols;
        tmem[r * 512u + offset + c] = 0.0f;
    }
    __syncwarp();
#endif
}

EG_DEV void eg_tcgen05_alloc_cta2(float* tmem, uint32_t* slot, uint32_t offset, uint32_t cols) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)offset;
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(slot);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(s), "r"(cols) : "memory");
#else
    __syncwarp();
    if ((threadIdx.x & 31u) == 0u) *slot = offset;
    __syncwarp();
    for (uint32_t idx = threadIdx.x & 31u; idx < 128u * cols; idx += 32u) {
        const uint32_t r = idx / cols, c = idx % cols;
        tmem[r * 512u + offset + c] = 0.0f;
    }
    __syncwarp();
#endif
}

EG_DEV void eg_tcgen05_dealloc_cta1(uint32_t taddr, uint32_t cols) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 :: "r"(taddr), "r"(cols));
#else
    (void)taddr;
    (void)cols;
    __syncwarp();
#endif
}

EG_DEV void eg_tcgen05_dealloc_cta2(uint32_t taddr, uint32_t cols) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
                 "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"
                 :: "r"(taddr), "r"(cols));
#else
    (void)taddr;
    (void)cols;
    __syncwarp();
#endif
}

// eg_tcgen05_ld_row loads one 32-bit float cell for each of the 32 lanes in the
// calling warp from TMEM column col, enforcing the 4-warp 32-row partition.
EG_DEV float eg_tcgen05_ld_row(const float* tmem, int warp, int lane, uint32_t col) {
    const uint32_t row = (uint32_t)(32 * (warp & 3) + lane);
    eg_tcgen05_check_warp_row(warp, row);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    uint32_t bits;
    const uint32_t taddr = (row << 16) | (col & 0xffffu);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=r"(bits) : "r"(taddr));
    return __uint_as_float(bits);
#else
    return tmem[row * 512u + col];
#endif
}

// eg_tcgen05_st_row stores one 32-bit float cell for each of the 32 lanes in
// the calling warp into TMEM column col, enforcing the 4-warp 32-row partition.
EG_DEV void eg_tcgen05_st_row(float* tmem, int warp, int lane, uint32_t col, float val) {
    const uint32_t row = (uint32_t)(32 * (warp & 3) + lane);
    eg_tcgen05_check_warp_row(warp, row);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    const uint32_t bits = __float_as_uint(val);
    const uint32_t taddr = (row << 16) | (col & 0xffffu);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                 :: "r"(taddr), "r"(bits) : "memory");
#else
    tmem[row * 512u + col] = val;
#endif
}

EG_DEV void eg_tcgen05_wait_ld() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#else
    __syncwarp();
#endif
}

EG_DEV void eg_tcgen05_wait_st() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#else
    __syncwarp();
#endif
}

EG_DEV void eg_tcgen05_commit_cta1(uint64_t* mbar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(s) : "memory");
#else
    (void)mbar;
#endif
}

EG_DEV void eg_tcgen05_commit_cta2(uint64_t* mbar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0], 3;"
                 :: "r"(s) : "memory");
#else
    (void)mbar;
#endif
}

// eg_tcgen05_cp_scales_32x128b copies staged UE4M3 block scales from shared
// memory into dedicated TMEM columns at scale_taddr via tcgen05.cp.cta_group::1.32x128b.warpx4.
EG_DEV void eg_tcgen05_cp_scales_32x128b(
    float* tmem, uint32_t scale_taddr,
    const uint8_t* smem_a_scales, const uint8_t* smem_b_scales,
    int m_rows, int n_cols, int k_blocks) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)smem_b_scales;
    (void)m_rows;
    (void)n_cols;
    (void)k_blocks;
    const uint64_t desc = eg_wgmma_desc(smem_a_scales, 16, 128);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(scale_taddr), "l"(desc) : "memory");
#else
    for (int r = 0; r < m_rows; ++r) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            tmem[r * 512 + scale_taddr + kb] = eg_e4m3_to_f32(smem_a_scales[r * k_blocks + kb]);
        }
    }
    for (int n = 0; n < n_cols; ++n) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            tmem[n * 512 + scale_taddr + 16 + kb] = eg_e4m3_to_f32(smem_b_scales[n * k_blocks + kb]);
        }
    }
#endif
}

// eg_tcgen05_decode_e2m1 decodes a 4-bit E2M1 nibble in [0..15] to FP32.
EG_DEV float eg_tcgen05_decode_e2m1(uint32_t code) {
    uint32_t out[2];
    eg_e2m1x4_to_bf16x4(code & 0xfu, out);
    return eg_bf16_to_f32(out[0]);
}

// eg_tcgen05_mma_cta2_f32 issues a 2-CTA cooperative tcgen05.mma.cta_group::2
// from CTA 0's elected thread with the .kind::tf32 instruction descriptor idesc.
EG_DEV void eg_tcgen05_mma_cta2_f32(
    float* local_tmem, uint32_t d_taddr,
    float* local_smem_a, float* local_smem_b, int k_dim, uint32_t idesc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)local_tmem;
    (void)k_dim;
    const uint64_t desc_a = eg_wgmma_desc(local_smem_a, 128, 1024);
    const uint64_t desc_b = eg_wgmma_desc(local_smem_b, 128, 1024);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p;\n"
        "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
#else
    (void)idesc;
    const float* b0 = eg_tcgen05_map_shared_rank(local_smem_b, 0);
    const float* b1 = eg_tcgen05_map_shared_rank(local_smem_b, 1);
    for (int cta = 0; cta < 2; ++cta) {
        const float* a_cta = eg_tcgen05_map_shared_rank(local_smem_a, cta);
        float* tmem_cta = eg_tcgen05_map_shared_rank(local_tmem, cta);
        for (int r = 0; r < 128; ++r) {
            for (int n = 0; n < 128; ++n) {
                const float* b_src = (n < 64) ? b0 : b1;
                const int b_col = (n < 64) ? n : (n - 64);
                float acc = tmem_cta[r * 512 + d_taddr + n];
                for (int k = 0; k < k_dim; ++k) {
                    acc += a_cta[r * k_dim + k] * b_src[k * 64 + b_col];
                }
                tmem_cta[r * 512 + d_taddr + n] = acc;
            }
        }
    }
#endif
}

// eg_tcgen05_mma_block_scale_nvfp4 issues tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale
// with instruction descriptor idesc, reading UE4M3 block scales previously
// staged into TMEM at scale_taddr by tcgen05.cp.
EG_DEV void eg_tcgen05_mma_block_scale_nvfp4(
    float* tmem, uint32_t d_taddr, uint32_t scale_taddr,
    const uint8_t* smem_a_packed, const uint8_t* smem_b_packed,
    int m_rows, int n_cols, int k_dim, uint32_t idesc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)m_rows;
    (void)n_cols;
    (void)k_dim;
    const uint64_t desc_a = eg_wgmma_desc(smem_a_packed, 32, 256);
    const uint64_t desc_b = eg_wgmma_desc(smem_b_packed, 32, 256);
    const uint32_t sfa_taddr = scale_taddr;
    const uint32_t sfb_taddr = scale_taddr + 16u;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;\n"
        "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(sfa_taddr), "r"(sfb_taddr) : "memory");
#else
    (void)idesc;
    const int k_blocks = k_dim / 16;
    const int k_bytes = k_dim / 2;
    for (int r = 0; r < m_rows; ++r) {
        for (int n = 0; n < n_cols; ++n) {
            float acc = tmem[r * 512 + d_taddr + n];
            for (int kb = 0; kb < k_blocks; ++kb) {
                const float sa = tmem[r * 512 + scale_taddr + kb];
                const float sb = tmem[n * 512 + scale_taddr + 16 + kb];
                float block_dot = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t ba = smem_a_packed[r * k_bytes + kb * 8 + j];
                    const uint8_t bb = smem_b_packed[n * k_bytes + kb * 8 + j];
                    const float a0 = eg_tcgen05_decode_e2m1(ba & 0xfu);
                    const float a1 = eg_tcgen05_decode_e2m1((ba >> 4) & 0xfu);
                    const float b0 = eg_tcgen05_decode_e2m1(bb & 0xfu);
                    const float b1 = eg_tcgen05_decode_e2m1((bb >> 4) & 0xfu);
                    block_dot += a0 * b0 + a1 * b1;
                }
                acc += block_dot * (sa * sb);
            }
            tmem[r * 512 + d_taddr + n] = acc;
        }
    }
#endif
}

// eg_tcgen05_mma_block_scale_nvfp4_4x issues sm_103a (GB300 / B300)
// tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X with
// instruction descriptor idesc and E4M3 block scales (one per 16 FP4
// elements) staged in TMEM.
EG_DEV void eg_tcgen05_mma_block_scale_nvfp4_4x(
    float* tmem, uint32_t d_taddr, uint32_t scale_taddr,
    const uint8_t* smem_a_packed, const uint8_t* smem_b_packed,
    int m_rows, int n_cols, int k_dim, uint32_t idesc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)m_rows;
    (void)n_cols;
    (void)k_dim;
    const uint64_t desc_a = eg_wgmma_desc(smem_a_packed, 64, 512);
    const uint64_t desc_b = eg_wgmma_desc(smem_b_packed, 64, 512);
    const uint32_t sfa_taddr = scale_taddr;
    const uint32_t sfb_taddr = scale_taddr + 16u;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n"
        "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(sfa_taddr), "r"(sfb_taddr) : "memory");
#else
    (void)idesc;
    const int k_blocks = k_dim / 16;
    const int k_bytes = k_dim / 2;
    for (int r = 0; r < m_rows; ++r) {
        for (int n = 0; n < n_cols; ++n) {
            float acc = tmem[r * 512 + d_taddr + n];
            for (int kb = 0; kb < k_blocks; ++kb) {
                const float sa = tmem[r * 512 + scale_taddr + kb];
                const float sb = tmem[n * 512 + scale_taddr + 16 + kb];
                float block_dot = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t ba = smem_a_packed[r * k_bytes + kb * 8 + j];
                    const uint8_t bb = smem_b_packed[n * k_bytes + kb * 8 + j];
                    const float a0 = eg_tcgen05_decode_e2m1(ba & 0xfu);
                    const float a1 = eg_tcgen05_decode_e2m1((ba >> 4) & 0xfu);
                    const float b0 = eg_tcgen05_decode_e2m1(bb & 0xfu);
                    const float b1 = eg_tcgen05_decode_e2m1((bb >> 4) & 0xfu);
                    block_dot += a0 * b0 + a1 * b1;
                }
                acc += block_dot * (sa * sb);
            }
            tmem[r * 512 + d_taddr + n] = acc;
        }
    }
#endif
}

// eg_tcgen05_mma_smem_ab issues tcgen05.mma.cta_group::1.kind::tf32 with
// instruction descriptor idesc and both operands A and B in shared memory
// (e.g., S = Q * K^T into TMEM [d_taddr]).
EG_DEV void eg_tcgen05_mma_smem_ab(
    float* tmem, uint32_t d_taddr,
    const float* smem_a, const float* smem_b_trans,
    int m_rows, int n_cols, int k_dim, uint32_t idesc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)m_rows;
    (void)n_cols;
    (void)k_dim;
    const uint64_t desc_a = eg_wgmma_desc(smem_a, 128, 1024);
    const uint64_t desc_b = eg_wgmma_desc(smem_b_trans, 128, 1024);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n"
        "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
#else
    (void)idesc;
    for (int r = 0; r < m_rows; ++r) {
        for (int n = 0; n < n_cols; ++n) {
            float acc = tmem[r * 512 + d_taddr + n];
            for (int k = 0; k < k_dim; ++k) {
                acc += smem_a[r * k_dim + k] * smem_b_trans[n * k_dim + k];
            }
            tmem[r * 512 + d_taddr + n] = acc;
        }
    }
#endif
}

// eg_tcgen05_mma_tmem_a issues tcgen05.mma.cta_group::1.kind::tf32 with
// instruction descriptor idesc, operand A read directly from TMEM [a_taddr],
// and operand B read from shared memory (e.g., O += P * V without spilling P
// back to shared memory).
EG_DEV void eg_tcgen05_mma_tmem_a(
    float* tmem, uint32_t d_taddr, uint32_t a_taddr,
    const float* smem_v, int m_rows, int n_cols, int k_dim, uint32_t idesc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)m_rows;
    (void)n_cols;
    (void)k_dim;
    const uint64_t desc_v = eg_wgmma_desc(smem_v, 128, 1024);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p;\n"
        "}" :: "r"(d_taddr), "r"(a_taddr), "l"(desc_v), "r"(idesc) : "memory");
#else
    (void)idesc;
    for (int r = 0; r < m_rows; ++r) {
        for (int d = 0; d < n_cols; ++d) {
            float acc = tmem[r * 512 + d_taddr + d];
            for (int k = 0; k < k_dim; ++k) {
                const float p_val = tmem[r * 512 + a_taddr + k];
                acc += p_val * smem_v[k * n_cols + d];
            }
            tmem[r * 512 + d_taddr + d] = acc;
        }
    }
#endif
}

#endif // EG_TCGEN05_ADVANCED_PRELUDE
`

// TCGen05Cluster2GEMM emits an sm_100a/sm_103a 2-CTA cooperative (__cluster_dims__(2,1,1))
// M=256, N=128 GEMM (128x128 per CTA) using tcgen05.mma.cta_group::2,
// distributed shared memory via eg_emu_map_shared_rank, and 4-warp 32-row
// partitioned tcgen05.ld epilogue loads.
type TCGen05Cluster2GEMM struct {
	Name string
	M    int // defaults to 256 (must be 256)
	N    int // defaults to 128 (must be 128)
	K    int // defaults to 32 (must be a positive multiple of 16)
}

func (k TCGen05Cluster2GEMM) norm() (TCGen05Cluster2GEMM, error) {
	if k.Name == "" {
		k.Name = "tcgen05_cluster2_gemm"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 cluster2 gemm: bad kernel name %q", k.Name)
	}
	if k.M == 0 {
		k.M = 256
	}
	if k.N == 0 {
		k.N = 128
	}
	if k.K == 0 {
		k.K = 32
	}
	if k.M != 256 || k.N != 128 || k.K <= 0 || k.K%16 != 0 {
		return k, fmt.Errorf("tcgen05 cluster2 gemm %s: invalid shape M=%d N=%d K=%d (want M=256, N=128, K=16*m)", k.Name, k.M, k.N, k.K)
	}
	return k, nil
}

// Grid returns the 2-CTA cluster grid and 128-thread (4-warp) block dimensions.
func (k TCGen05Cluster2GEMM) Grid() (grid, block [3]int) {
	return [3]int{2, 1, 1}, [3]int{128, 1, 1}
}

// TraceEvents returns the formal sm_100a/sm_103a tcgen05 hardware protocol event trace
// for verification with cudatmem.VerifyTrace.
func (k TCGen05Cluster2GEMM) TraceEvents() []cudatmem.TraceEvent {
	events := []cudatmem.TraceEvent{
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventAlloc, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventClusterSync, CTAGroup: 2},
		{Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventWait, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
	}
	for cta := range 2 {
		for w := range 4 {
			events = append(events, cudatmem.TraceEvent{
				Kind:          cudatmem.EventLd,
				CTA:           cta,
				Warp:          w,
				ActiveThreads: 32,
				CTAGroup:      2,
				Offset:        0,
				Columns:       128,
				RowStart:      uint32(32 * w),
				Rows:          32,
			})
		}
	}
	events = append(events,
		cudatmem.TraceEvent{Kind: cudatmem.EventClusterSync, CTAGroup: 2},
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
	)
	return events
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05Cluster2GEMM) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	idesc, err := tf32InstDesc(k.M, k.N)
	if err != nil {
		return "", fmt.Errorf("tcgen05 cluster2 gemm %s: %w", k.Name, err)
	}
	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: 2-CTA cooperative (cta_group::2) M=%d, N=%d, K=%d tcgen05 GEMM.\n", k.Name, k.M, k.N, k.K)
	fmt.Fprintf(&b, `__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) %s(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out) {
    constexpr int K = %d;
    constexpr uint32_t IDESC = 0x%08xu;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rank = eg_tcgen05_cluster_rank();

    EG_TCGEN05_SHARED(float, smem_a, [128 * K]);
    EG_TCGEN05_SHARED(float, smem_b, [K * 64]);
    EG_TCGEN05_SHARED(float, tmem, [128 * 512]);
    EG_TCGEN05_SHARED(uint32_t, alloc_slot, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    // 1. Both CTAs allocate 128 TMEM columns under cta_group::2.
    if (warp == 0) {
        eg_tcgen05_alloc_cta2(tmem, &alloc_slot[0], 0u, 128u);
    }
    __syncthreads();
    const uint32_t taddr = alloc_slot[0];

    // 2. CTA rank loads its 128xK half of A and its Kx64 half of B into shared memory.
    for (int i = tid; i < 128 * K; i += 128) {
        const int r = i / K, c = i %% K;
        smem_a[i] = a[(rank * 128 + r) * K + c];
    }
    for (int i = tid; i < K * 64; i += 128) {
        const int r = i / 64, c = i %% 64;
        smem_b[i] = b[r * 128 + rank * 64 + c];
    }
    eg_tcgen05_cluster_sync();

    // 3. Single elected thread on CTA 0 issues tcgen05.mma.cta_group::2 + commit.
    if (rank == 0 && warp == 0 && eg_elect_one_sync()) {
        eg_tcgen05_mma_cta2_f32(tmem, taddr, smem_a, smem_b, K, IDESC);
        eg_tcgen05_commit_cta2(&mbar[0]);
    }
    eg_tcgen05_cluster_sync();

    // 4. 4-warp 32-row-per-warp partitioned tcgen05.ld epilogue (warp w loads rows 32*w..32*w+31).
    const int global_row = rank * 128 + 32 * warp + lane;
    for (int col = 0; col < 128; ++col) {
        const float v = eg_tcgen05_ld_row(tmem, warp, lane, taddr + (uint32_t)col);
        out[global_row * 128 + col] = v;
    }
    eg_tcgen05_wait_ld();

    // 5. Cluster barrier before cooperative cta_group::2 deallocation.
    eg_tcgen05_cluster_sync();
    if (warp == 0) {
        eg_tcgen05_dealloc_cta2(taddr, 128u);
    }
}
EG_REGISTER_CLUSTER2(%s)
`, k.Name, k.K, idesc, k.Name)
	return b.String(), nil
}

// TCGen05BlockScaledGEMM emits an sm_100a/sm_103a NVFP4 block-scaled GEMM that stages
// UE4M3 block scales into shared memory, copies them into dedicated TMEM
// columns via tcgen05.cp.cta_group::1.32x128b, issues
// tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale (or with .scale_vec::4X
// when ScaleVec4X is true on sm_103a), and reads the accumulated tile via
// 4-warp 32-row partitioned tcgen05.ld.
type TCGen05BlockScaledGEMM struct {
	Name       string
	M          int  // defaults to 128 (must be 128)
	N          int  // defaults to 128 (must be 128)
	K          int  // defaults to 64 (or 128 when ScaleVec4X is true; multiple of 16 in 16..256)
	ScaleVec4X bool // when true, emits sm_103a scale_vec::4X (K multiple of 64, 4 E4M3 scales per step)
}

func (k TCGen05BlockScaledGEMM) norm() (TCGen05BlockScaledGEMM, error) {
	if k.Name == "" {
		if k.ScaleVec4X {
			k.Name = "tcgen05_scale_vec_4x_gemm"
		} else {
			k.Name = "tcgen05_block_scaled_gemm"
		}
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 block scaled gemm: bad kernel name %q", k.Name)
	}
	if k.M == 0 {
		k.M = 128
	}
	if k.N == 0 {
		k.N = 128
	}
	if k.K == 0 {
		if k.ScaleVec4X {
			k.K = 128
		} else {
			k.K = 64
		}
	}
	if k.M != 128 || k.N != 128 || k.K < 16 || k.K > 256 || k.K%16 != 0 {
		return k, fmt.Errorf("tcgen05 block scaled gemm %s: invalid shape M=%d N=%d K=%d (want M=128, N=128, K in 16..256 multiple of 16)", k.Name, k.M, k.N, k.K)
	}
	if k.ScaleVec4X && k.K%64 != 0 {
		return k, fmt.Errorf("tcgen05 block scaled gemm %s: ScaleVec4X requires K=%d to be a multiple of 64", k.Name, k.K)
	}
	return k, nil
}

// Grid returns the single-CTA grid and 128-thread (4-warp) block dimensions.
func (k TCGen05BlockScaledGEMM) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

// TraceEvents returns the formal sm_100a/sm_103a tcgen05 hardware protocol event trace
// for verification with cudatmem.VerifyTrace.
func (k TCGen05BlockScaledGEMM) TraceEvents() []cudatmem.TraceEvent {
	events := []cudatmem.TraceEvent{
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		{Kind: cudatmem.EventCp, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		{Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128, HasScaleInTMEM: true, ScaleOffset: 128},
		{Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
	}
	for w := range 4 {
		events = append(events, cudatmem.TraceEvent{
			Kind:          cudatmem.EventLd,
			CTA:           0,
			Warp:          w,
			ActiveThreads: 32,
			CTAGroup:      1,
			Offset:        0,
			Columns:       128,
			RowStart:      uint32(32 * w),
			Rows:          32,
		})
	}
	events = append(events,
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
	)
	return events
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05BlockScaledGEMM) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	mmaHelper := "eg_tcgen05_mma_block_scale_nvfp4"
	if k.ScaleVec4X {
		mmaHelper = "eg_tcgen05_mma_block_scale_nvfp4_4x"
	}
	// One mxf4nvf4 instruction covers dense K=64 on sm_100a/sm_103a
	// (K=128 needs sm_107f), for both scale_vec::2X and scale_vec::4X.
	idesc, err := InstDesc{
		Kind:      KindMXF4NVF4,
		AType:     ElemE2M1,
		BType:     ElemE2M1,
		M:         k.M,
		N:         k.N,
		K:         64,
		ScaleType: ScaleUE4M3,
	}.Encode()
	if err != nil {
		return "", fmt.Errorf("tcgen05 block scaled gemm %s: %w", k.Name, err)
	}
	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: NVFP4 block-scaled M=%d, N=%d, K=%d GEMM with tcgen05.cp scale staging.\n", k.Name, k.M, k.N, k.K)
	fmt.Fprintf(&b, `__global__ void __launch_bounds__(128) %s(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ b_packed,
    const uint8_t* __restrict__ b_scales,
    float* __restrict__ out) {
    constexpr int M = %d, N = %d, K = %d;
    constexpr int K_BYTES = K / 2, K_BLOCKS = K / 16;
    constexpr uint32_t IDESC = 0x%08xu;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    EG_TCGEN05_SHARED(uint8_t, smem_a_packed, [M * K_BYTES]);
    EG_TCGEN05_SHARED(uint8_t, smem_b_packed, [N * K_BYTES]);
    EG_TCGEN05_SHARED(uint8_t, smem_a_scales, [M * K_BLOCKS]);
    EG_TCGEN05_SHARED(uint8_t, smem_b_scales, [N * K_BLOCKS]);
    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot_d, [1]);
    EG_TCGEN05_SHARED(uint32_t, slot_s, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    // 1. Allocate 128 accumulator columns (offset 0) then 32 scale columns (offset 128).
    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot_d[0], 0u, 128u);
        eg_tcgen05_alloc_cta1(tmem, &slot_s[0], 128u, 32u);
    }
    __syncthreads();
    const uint32_t d_taddr = slot_d[0];
    const uint32_t s_taddr = slot_s[0];

    // 2. Stage NVFP4 packed E2M1 operands and UE4M3 block scales in shared memory.
    for (int i = tid; i < M * K_BYTES; i += 128) smem_a_packed[i] = a_packed[i];
    for (int i = tid; i < N * K_BYTES; i += 128) smem_b_packed[i] = b_packed[i];
    for (int i = tid; i < M * K_BLOCKS; i += 128) smem_a_scales[i] = a_scales[i];
    for (int i = tid; i < N * K_BLOCKS; i += 128) smem_b_scales[i] = b_scales[i];
    __syncthreads();

    // 3. Single elected thread stages scales into TMEM via tcgen05.cp.cta_group::1.32x128b
    //    and issues block-scaled tcgen05.mma.
    if (warp == 0 && eg_elect_one_sync()) {
        eg_tcgen05_cp_scales_32x128b(tmem, s_taddr, smem_a_scales, smem_b_scales, M, N, K_BLOCKS);
        %s(tmem, d_taddr, s_taddr, smem_a_packed, smem_b_packed, M, N, K, IDESC);
        eg_tcgen05_commit_cta1(&mbar[0]);
    }
    __syncthreads();

    // 4. 4-warp 32-row partitioned tcgen05.ld epilogue.
    const int row = 32 * warp + lane;
    for (int col = 0; col < N; ++col) {
        out[row * N + col] = eg_tcgen05_ld_row(tmem, warp, lane, d_taddr + (uint32_t)col);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 5. Release TMEM allocations.
    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(d_taddr, 128u);
        eg_tcgen05_dealloc_cta1(s_taddr, 32u);
    }
}
`, k.Name, k.M, k.N, k.K, idesc, mmaHelper)
	return b.String(), nil
}

// TCGen05TMEMFlashAttention emits an sm_100a/sm_103a fused attention kernel where
// S = Q * K^T is accumulated into TMEM [tmem_s], rescaled in-place in TMEM
// (P = softmax(S * Scale)) using 4-warp 32-row partitioned tcgen05.ld /
// tcgen05.st, and O = P * V is issued via tcgen05.mma with operand A (P) read
// directly from TMEM ([tmem_o], [tmem_p], desc_v) without spilling P to shared
// memory.
type TCGen05TMEMFlashAttention struct {
	Name  string
	M     int     // query rows; defaults to 128 (must be 128)
	KV    int     // key/value rows; defaults to 64 (must be 64)
	D     int     // head dimension; defaults to 64 (must be 64)
	Scale float32 // softmax logit scale; defaults to 1/sqrt(D)
}

func (k TCGen05TMEMFlashAttention) norm() (TCGen05TMEMFlashAttention, error) {
	if k.Name == "" {
		k.Name = "tcgen05_tmem_flash_attention"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 tmem flash attention: bad kernel name %q", k.Name)
	}
	if k.M == 0 {
		k.M = 128
	}
	if k.KV == 0 {
		k.KV = 64
	}
	if k.D == 0 {
		k.D = 64
	}
	if k.M != 128 || k.KV != 64 || k.D != 64 {
		return k, fmt.Errorf("tcgen05 tmem flash attention %s: invalid shape M=%d KV=%d D=%d (want 128, 64, 64)", k.Name, k.M, k.KV, k.D)
	}
	if k.Scale == 0 {
		k.Scale = 1.0 / float32(math.Sqrt(float64(k.D)))
	}
	return k, nil
}

// Grid returns the single-CTA grid and 128-thread (4-warp) block dimensions.
func (k TCGen05TMEMFlashAttention) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

// TraceEvents returns the formal sm_100a/sm_103a tcgen05 hardware protocol event trace
// for verification with cudatmem.VerifyTrace.
func (k TCGen05TMEMFlashAttention) TraceEvents() []cudatmem.TraceEvent {
	events := []cudatmem.TraceEvent{
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128},
		{Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
		{Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
		{Kind: cudatmem.EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
	}
	for w := range 4 {
		events = append(events,
			cudatmem.TraceEvent{
				Kind: cudatmem.EventLd, CTA: 0, Warp: w, ActiveThreads: 32, CTAGroup: 1,
				Offset: 0, Columns: 64, RowStart: uint32(32 * w), Rows: 32,
			},
			cudatmem.TraceEvent{
				Kind: cudatmem.EventSt, CTA: 0, Warp: w, ActiveThreads: 32, CTAGroup: 1,
				Offset: 0, Columns: 64, RowStart: uint32(32 * w), Rows: 32,
			},
		)
	}
	events = append(events,
		cudatmem.TraceEvent{
			Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1,
			Offset: 64, Columns: 64, Rows: 128, SourceAFromTMEM: true, SourceAOffset: 0,
		},
		cudatmem.TraceEvent{Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128},
		cudatmem.TraceEvent{Kind: cudatmem.EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128},
	)
	for w := range 4 {
		events = append(events, cudatmem.TraceEvent{
			Kind: cudatmem.EventLd, CTA: 0, Warp: w, ActiveThreads: 32, CTAGroup: 1,
			Offset: 64, Columns: 64, RowStart: uint32(32 * w), Rows: 32,
		})
	}
	events = append(events,
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 64, Rows: 128},
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 64, Columns: 64, Rows: 128},
	)
	return events
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05TMEMFlashAttention) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	idescS, err := tf32InstDesc(k.M, k.KV)
	if err != nil {
		return "", fmt.Errorf("tcgen05 tmem flash attention %s: %w", k.Name, err)
	}
	idescO, err := tf32InstDesc(k.M, k.D)
	if err != nil {
		return "", fmt.Errorf("tcgen05 tmem flash attention %s: %w", k.Name, err)
	}
	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: TMEM-sourced operand A FlashAttention (M=%d, KV=%d, D=%d).\n", k.Name, k.M, k.KV, k.D)
	fmt.Fprintf(&b, `__global__ void __launch_bounds__(128) %s(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out) {
    constexpr int M = %d, KV = %d, D = %d;
    constexpr float SCALE = %s;
    constexpr uint32_t IDESC_S = 0x%08xu, IDESC_O = 0x%08xu;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    EG_TCGEN05_SHARED(float, smem_q, [M * D]);
    EG_TCGEN05_SHARED(float, smem_k, [KV * D]);
    EG_TCGEN05_SHARED(float, smem_v, [KV * D]);
    EG_TCGEN05_SHARED(float, tmem, [128 * 512]);
    EG_TCGEN05_SHARED(uint32_t, slot_s, [1]);
    EG_TCGEN05_SHARED(uint32_t, slot_o, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    // 1. Allocate TMEM slices: [tmem_s / tmem_p] (64 cols at offset 0) and [tmem_o] (64 cols at offset 64).
    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot_s[0], 0u, 64u);
        eg_tcgen05_alloc_cta1(tmem, &slot_o[0], 64u, 64u);
    }
    __syncthreads();
    const uint32_t tmem_s = slot_s[0];
    const uint32_t tmem_o = slot_o[0];

    // 2. Stage Q, K, V in shared memory (P is never spilled to shared memory).
    for (int i = tid; i < M * D; i += 128) smem_q[i] = q[i];
    for (int i = tid; i < KV * D; i += 128) smem_k[i] = k[i];
    for (int i = tid; i < KV * D; i += 128) smem_v[i] = v[i];
    __syncthreads();

    // 3. First MMA: S = Q * K^T accumulated into TMEM [tmem_s].
    if (warp == 0 && eg_elect_one_sync()) {
        eg_tcgen05_mma_smem_ab(tmem, tmem_s, smem_q, smem_k, M, KV, D, IDESC_S);
        eg_tcgen05_commit_cta1(&mbar[0]);
    }
    __syncthreads();

    // 4. Rescale S -> P = softmax(S * SCALE) in TMEM using 4-warp 32-row partitioned ld/st.
    float s_row[KV];
    float row_max = -__uint_as_float(0x7f800000u);
    for (int j = 0; j < KV; ++j) {
        const float val = eg_tcgen05_ld_row(tmem, warp, lane, tmem_s + (uint32_t)j) * SCALE;
        s_row[j] = val;
        row_max = fmaxf(row_max, val);
    }
    eg_tcgen05_wait_ld();

    float row_sum = 0.0f;
    for (int j = 0; j < KV; ++j) {
        s_row[j] = expf(s_row[j] - row_max);
        row_sum += s_row[j];
    }
    const float inv_sum = 1.0f / row_sum;
    for (int j = 0; j < KV; ++j) {
        eg_tcgen05_st_row(tmem, warp, lane, tmem_s + (uint32_t)j, s_row[j] * inv_sum);
    }
    eg_tcgen05_wait_st();
    __syncthreads();

    // 5. Second MMA: O = P * V reading operand A (P) directly from TMEM [tmem_s].
    if (warp == 0 && eg_elect_one_sync()) {
        eg_tcgen05_mma_tmem_a(tmem, tmem_o, tmem_s, smem_v, M, D, KV, IDESC_O);
        eg_tcgen05_commit_cta1(&mbar[0]);
    }
    __syncthreads();

    // 6. 4-warp 32-row partitioned tcgen05.ld epilogue from [tmem_o] to global out.
    const int row = 32 * warp + lane;
    for (int d = 0; d < D; ++d) {
        out[row * D + d] = eg_tcgen05_ld_row(tmem, warp, lane, tmem_o + (uint32_t)d);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 7. Release TMEM slices.
    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(tmem_s, 64u);
        eg_tcgen05_dealloc_cta1(tmem_o, 64u);
    }
}
`, k.Name, k.M, k.KV, k.D, floatLiteral(k.Scale), idescS, idescO)
	return b.String(), nil
}

// tf32InstDesc returns the dense .kind::tf32 instruction descriptor
// (TF32 x TF32 -> F32, K=8) for an M x N MMA.
func tf32InstDesc(m, n int) (uint32, error) {
	return InstDesc{
		Kind:  KindTF32,
		DType: ElemF32,
		AType: ElemTF32,
		BType: ElemTF32,
		M:     m,
		N:     n,
		K:     8,
	}.Encode()
}
