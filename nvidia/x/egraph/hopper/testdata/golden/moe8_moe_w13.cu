#ifndef EG_PRELUDE
#define EG_PRELUDE
#if !defined(EG_HOST_EMU)
#if defined(__CUDACC_RTC__)
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif
#endif

#define EG_DEV static __device__ __forceinline__

// EG_LAUNCH launches kernel; host launchers use it so the emulator can run them.
#if defined(EG_HOST_EMU)
#define EG_LAUNCH(kernel, grid, block, smem, stream, ...) eg_emu_launch(grid, block, smem, kernel, __VA_ARGS__)
#else
#define EG_LAUNCH(kernel, grid, block, smem, stream, ...) kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)
#endif

struct alignas(16) eg_u32x4 { uint32_t x, y, z, w; };

EG_DEV float eg_bf16_to_f32(uint32_t b) { return __uint_as_float((b & 0xffffu) << 16); }
EG_DEV float eg_bf16_hi_to_f32(uint32_t b) { return __uint_as_float(b & 0xffff0000u); }

// eg_f32_to_bf16 rounds to nearest even (cvt.rn.bf16.f32).
EG_DEV uint16_t eg_f32_to_bf16(float f) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_bf16(f);
#else
    uint16_t r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
#endif
}

// eg_pack_bf16x2 packs lo into bits 0-15 and hi into bits 16-31.
EG_DEV uint32_t eg_pack_bf16x2(float lo, float hi) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_bf16(lo) | ((uint32_t)eg_emu_f32_to_bf16(hi) << 16);
#else
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#endif
}

// eg_e4m3_to_f32 decodes float8_e4m3fn bits; 0x7f and 0xff are NaN.
EG_DEV float eg_e4m3_to_f32(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0) v = (float)m * (1.0f / 512.0f);
    else if (e == 15 && m == 7) v = __uint_as_float(0x7fc00000u);
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}

// eg_f32x2_to_e4m3x2 rounds to nearest even, saturating to ±448
// (cvt.rn.satfinite.e4m3x2.f32); lo lands in bits 0-7.
EG_DEV uint16_t eg_f32x2_to_e4m3x2(float lo, float hi) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_e4m3(lo) | ((uint16_t)eg_emu_f32_to_e4m3(hi) << 8);
#else
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
#endif
}

EG_DEV uint32_t eg_prmt(uint32_t a, uint32_t b, uint32_t sel) {
#if defined(EG_HOST_EMU)
    return eg_emu_prmt(a, b, sel);
#else
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
#endif
}

// eg_e2m1x4_to_bf16x4 decodes the four E2M1 codes in the low 16 bits of q
// (code i in bits 4i..4i+3) to bfloat16: out[0] holds codes 0 and 1, out[1]
// codes 2 and 3. Each magnitude's two bytes come from a byte-permute table
// lookup and the sign is bit 3 of the code.
EG_DEV void eg_e2m1x4_to_bf16x4(uint32_t q, uint32_t out[2]) {
    // bf16 bytes of 0, 0.5, 1, 1.5, 2, 3, 4, 6.
    const uint32_t lo_a = 0xc0800000u, lo_b = 0xc0804000u;
    const uint32_t hi_a = 0x3f3f3f00u, hi_b = 0x40404040u;
    const uint32_t idx = q & 0x7777u;
    const uint32_t lo = eg_prmt(lo_a, lo_b, idx), hi = eg_prmt(hi_a, hi_b, idx);
    out[0] = eg_prmt(lo, hi, 0x5140u) | ((q & 0x8u) << 12) | ((q & 0x80u) << 24);
    out[1] = eg_prmt(lo, hi, 0x7362u) | ((q & 0x800u) << 4) | ((q & 0x8000u) << 16);
}

EG_DEV float eg_warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

EG_DEV float eg_warp_max(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// eg_ldg_stream loads 16 bytes that are read once (weights), bypassing L1.
EG_DEV eg_u32x4 eg_ldg_stream(const void* p) {
#if defined(EG_HOST_EMU)
    eg_u32x4 r;
    memcpy(&r, p, 16);
    return r;
#else
    eg_u32x4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
#endif
}

EG_DEV eg_u32x4 eg_ldg16(const void* p) { return *reinterpret_cast<const eg_u32x4*>(p); }

// eg_ld_cg loads through L2 only, to read another block's fenced writes.
EG_DEV float eg_ld_cg(const float* p) {
#if defined(EG_HOST_EMU)
    return *(volatile const float*)p;
#else
    float r;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(r) : "l"(p));
    return r;
#endif
}

// eg_cp_async16 copies 16 bytes global to shared without staging in registers.
EG_DEV void eg_cp_async16(void* smem, const void* gmem) {
#if defined(EG_HOST_EMU)
    memcpy(smem, gmem, 16);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
#endif
}

// eg_cp_async4 copies 4 bytes global to shared (through L1).
EG_DEV void eg_cp_async4(void* smem, const void* gmem) {
#if defined(EG_HOST_EMU)
    memcpy(smem, gmem, 4);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(gmem));
#endif
}

EG_DEV void eg_cp_async_commit() {
#if !defined(EG_HOST_EMU)
    asm volatile("cp.async.commit_group;");
#endif
}

// eg_cp_async_wait waits until at most n committed groups are pending.
template <int n> EG_DEV void eg_cp_async_wait() {
#if !defined(EG_HOST_EMU)
    asm volatile("cp.async.wait_group %0;" :: "n"(n));
#endif
}

EG_DEV void eg_ldmatrix_x4(uint32_t r[4], const void* row) {
#if defined(EG_HOST_EMU)
    eg_emu_ldmatrix_x4(r, row, false);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
#endif
}

EG_DEV void eg_ldmatrix_x4_trans(uint32_t r[4], const void* row) {
#if defined(EG_HOST_EMU)
    eg_emu_ldmatrix_x4(r, row, true);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
#endif
}

// eg_mma_bf16 is d = a·b + c for one m16n8k16 tile, BF16 inputs, FP32 accumulate.
EG_DEV void eg_mma_bf16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
#if defined(EG_HOST_EMU)
    eg_emu_mma_bf16_16816(d, a, b, c);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
}

// eg_mma_e4m3 is d = a·b + c for one m16n8k32 tile, E4M3 inputs (four per
// register, k ascending from the low byte), FP32 accumulate (sm_89+).
EG_DEV void eg_mma_e4m3(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
#if defined(EG_HOST_EMU)
    eg_emu_mma_e4m3_16832(d, a, b, c);
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
}

// Programmatic dependent launch: wait for the previous kernel's writes, and
// let the next kernel start its preamble.
EG_DEV void eg_pdl_wait() {
#if !defined(EG_HOST_EMU)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

EG_DEV void eg_pdl_launch_dependents() {
#if !defined(EG_HOST_EMU)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

#if defined(EG_HOST_EMU)
#define EG_DYNAMIC_SMEM(T, name) T* name = reinterpret_cast<T*>(eg_emu_dynamic_smem())
#else
#define EG_DYNAMIC_SMEM(T, name) extern __shared__ __align__(16) unsigned char eg_smem_raw[]; T* name = reinterpret_cast<T*>(eg_smem_raw)
#endif

struct alignas(8) eg_u32x2 { uint32_t x, y; };

// eg_load converts V contiguous elements at p (aligned to their width) to FP32.
template <int V> EG_DEV void eg_load(const float* p, float* o) {
    if constexpr (V % 4 == 0) {
#pragma unroll
        for (int i = 0; i < V; i += 4) {
            const eg_u32x4 u = *reinterpret_cast<const eg_u32x4*>(p + i);
            o[i] = __uint_as_float(u.x); o[i + 1] = __uint_as_float(u.y);
            o[i + 2] = __uint_as_float(u.z); o[i + 3] = __uint_as_float(u.w);
        }
    } else {
#pragma unroll
        for (int i = 0; i < V; i += 2) {
            const eg_u32x2 u = *reinterpret_cast<const eg_u32x2*>(p + i);
            o[i] = __uint_as_float(u.x); o[i + 1] = __uint_as_float(u.y);
        }
    }
}

template <int V> EG_DEV void eg_load(const uint16_t* p, float* o) {
    uint32_t w[V / 2];
    if constexpr (V % 8 == 0) {
#pragma unroll
        for (int i = 0; i < V / 2; i += 4) {
            const eg_u32x4 u = *reinterpret_cast<const eg_u32x4*>(p + 2 * i);
            w[i] = u.x; w[i + 1] = u.y; w[i + 2] = u.z; w[i + 3] = u.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < V / 2; ++i) w[i] = reinterpret_cast<const uint32_t*>(p)[i];
    }
#pragma unroll
    for (int i = 0; i < V / 2; ++i) { o[2 * i] = eg_bf16_to_f32(w[i]); o[2 * i + 1] = eg_bf16_hi_to_f32(w[i]); }
}

template <int V> EG_DEV void eg_load(const uint8_t* p, float* o) {
#pragma unroll
    for (int i = 0; i < V; i += 2) {
        const uint32_t u = reinterpret_cast<const uint16_t*>(p)[i / 2];
        o[i] = eg_e4m3_to_f32(u & 0xffu); o[i + 1] = eg_e4m3_to_f32(u >> 8);
    }
}

// eg_store rounds V FP32 values to the element type at p.
template <int V> EG_DEV void eg_store(float* p, const float* v) {
    if constexpr (V % 4 == 0) {
#pragma unroll
        for (int i = 0; i < V; i += 4)
            *reinterpret_cast<eg_u32x4*>(p + i) = eg_u32x4{__float_as_uint(v[i]), __float_as_uint(v[i + 1]), __float_as_uint(v[i + 2]), __float_as_uint(v[i + 3])};
    } else {
#pragma unroll
        for (int i = 0; i < V; i += 2) *reinterpret_cast<eg_u32x2*>(p + i) = eg_u32x2{__float_as_uint(v[i]), __float_as_uint(v[i + 1])};
    }
}

template <int V> EG_DEV void eg_store(uint16_t* p, const float* v) {
    if constexpr (V % 8 == 0) {
#pragma unroll
        for (int i = 0; i < V; i += 8)
            *reinterpret_cast<eg_u32x4*>(p + i) = eg_u32x4{eg_pack_bf16x2(v[i], v[i + 1]), eg_pack_bf16x2(v[i + 2], v[i + 3]),
                                                           eg_pack_bf16x2(v[i + 4], v[i + 5]), eg_pack_bf16x2(v[i + 6], v[i + 7])};
    } else {
#pragma unroll
        for (int i = 0; i < V; i += 2) reinterpret_cast<uint32_t*>(p)[i / 2] = eg_pack_bf16x2(v[i], v[i + 1]);
    }
}

template <int V> EG_DEV void eg_store(uint8_t* p, const float* v) {
#pragma unroll
    for (int i = 0; i < V; i += 2) reinterpret_cast<uint16_t*>(p)[i / 2] = eg_f32x2_to_e4m3x2(v[i], v[i + 1]);
}

// eg_block_combine combines one warp-reduced value per warp (OP 0: sum,
// 1: max) in warp order; every thread gets the result. red holds 32 floats.
template <int OP> EG_DEV float eg_block_combine(float v, float* red) {
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = red[0];
    for (int i = 1; i < (int)(blockDim.x >> 5); ++i) s = OP ? fmaxf(s, red[i]) : s + red[i];
    __syncthreads();
    return s;
}

EG_DEV float eg_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
EG_DEV float eg_silu(float x) { return x / (1.0f + expf(-x)); }
#endif // EG_PRELUDE

// moe8_moe_w13: out = (mul (silu $0) $1) over x[M, 4096] · W[8192, 4096]ᵀ (nvfp4), 128x128 tiles, 3 stages.
__global__ void __launch_bounds__(256) moe8_moe_w13(
    int m_total,
    const uint16_t* __restrict__ x,
    const int32_t* __restrict__ rows,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ w_sf,
    const float* __restrict__ w_gs,
    const int32_t* __restrict__ offsets,
    uint16_t* __restrict__ out) {
    constexpr int N = 8192, K = 4096, BM = 128, BN = 128, BK = 64, STAGES = 3, LD = 72;
    constexpr int NOUT = N / 2;
    constexpr int WM = BM / 2, WN = BN / 4, MT = WM / 16, NT = WN / 8, KT = K / BK;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / 4, wn = warp % 4;
    const int e = blockIdx.z, row0 = offsets[e], row1 = offsets[e + 1];
    const int bm0 = row0 + blockIdx.y * BM, bn0 = blockIdx.x * BN;
    if (bm0 >= row1) return;
    EG_DYNAMIC_SMEM(uint16_t, sa); // [STAGES][BM][LD], then the W ring
    // Tile row j: warp column j / WN holds WN/2 gate rows, then the matching up rows.
    auto wrow = [&](int j) {
        const int o = j % WN, c = blockIdx.x * (BN / 2) + (j / WN) * (WN / 2) + o % (WN / 2);
        return o < WN / 2 ? c : NOUT + c;
    };
    uint8_t* const sbq = reinterpret_cast<uint8_t*>(sa + STAGES * BM * LD); // [STAGES][BN][48]
    uint8_t* const sbs = sbq + STAGES * BN * 48;                                 // [STAGES][BN][4]
    const uint8_t* const we = w + (size_t)e * N * (K / 2);
    const uint8_t* const se = w_sf + (size_t)e * N * (K / 16);
    auto load = [&](int st, int kt) {
        for (int c = tid; c < BM * 8; c += 256) {
            const int r = c / 8, ch = c % 8, gm = min(bm0 + r, row1 - 1);
            eg_cp_async16(&sa[(st * BM + r) * LD + ch * 8], x + (size_t)rows[gm] * K + kt * BK + ch * 8);
        }
        for (int c = tid; c < BN * 2; c += 256) {
            const int r = c / 2, ch = c % 2;
            eg_cp_async16(&sbq[(st * BN + r) * 48 + ch * 16], we + (size_t)wrow(r) * (K / 2) + kt * 32 + ch * 16);
        }
        for (int r = tid; r < BN; r += 256) eg_cp_async4(&sbs[(st * BN + r) * 4], se + (size_t)wrow(r) * (K / 16) + kt * 4);
    };
    float acc[MT][NT][4] = {};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        eg_cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        eg_cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        eg_cp_async_commit();
        const int st = kt % STAGES;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            uint32_t af[MT][4], bf[NT][2];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                eg_ldmatrix_x4(af[mt], &sa[(st * BM + wm * WM + mt * 16 + lane % 16) * LD + ks * 16 + (lane / 16) * 8]);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                // b0 holds k = 2t, 2t+1 (byte t of the k16 step), b1 k + 8 (byte t + 4).
                const int r = st * BN + wn * WN + nt * 8 + lane / 4;
                const uint8_t* const qb = &sbq[r * 48 + ks * 8 + lane % 4];
                uint32_t h[2];
                eg_e2m1x4_to_bf16x4(qb[0] | (uint32_t)qb[4] << 8, h);
                const float sf = eg_e4m3_to_f32(sbs[r * 4 + ks]);
                bf[nt][0] = eg_pack_bf16x2(eg_bf16_to_f32(h[0]) * sf, eg_bf16_hi_to_f32(h[0]) * sf);
                bf[nt][1] = eg_pack_bf16x2(eg_bf16_to_f32(h[1]) * sf, eg_bf16_hi_to_f32(h[1]) * sf);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) eg_mma_bf16(acc[mt][nt], af[mt], bf[nt], acc[mt][nt]);
        }
    }
    eg_cp_async_wait<0>();
    const float gs = w_gs[e];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT / 2; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = bm0 + wm * WM + mt * 16 + lane / 4 + 8 * (i / 2);
                const int n = blockIdx.x * (BN / 2) + wn * (WN / 2) + nt * 8 + 2 * (lane % 4) + (i & 1);
                if (m >= row1) continue;
                const float v = acc[mt][nt][i] * gs;
                const float u = acc[mt][nt + NT / 2][i] * gs;
                out[(size_t)m * NOUT + n] = eg_f32_to_bf16((eg_silu((v)) * (u)));
            }
}

inline void moe8_moe_w13_launch(int m, int max_rows, int experts, const uint16_t* __restrict__ x, const int32_t* __restrict__ rows, const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_sf, const float* __restrict__ w_gs, const int32_t* __restrict__ offsets, uint16_t* __restrict__ out, cudaStream_t stream) {
    constexpr int smem = 75264;
    static const cudaError_t attr = cudaFuncSetAttribute(moe8_moe_w13, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)attr;
    const dim3 grid(64, (max_rows + 127) / 128, experts), block(256);
    if (grid.y > 0 && grid.z > 0) EG_LAUNCH(moe8_moe_w13, grid, block, smem, stream, m, x, rows, w, w_sf, w_gs, offsets, out);
}
