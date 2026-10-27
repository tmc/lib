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

// eg_pack_f16x2 rounds lo and hi to F16 (nearest even) into bits 0-15 and 16-31.
EG_DEV uint32_t eg_pack_f16x2(float lo, float hi) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_f16(lo) | ((uint32_t)eg_emu_f32_to_f16(hi) << 16);
#else
    uint32_t r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#endif
}

// eg_e4m3x2_to_f16x2 widens the two E4M3 bytes in bits 0-15 of x to F16
// (exactly; cvt.rn.f16x2.e4m3x2, sm_89+), the low byte into bits 0-15.
EG_DEV uint32_t eg_e4m3x2_to_f16x2(uint32_t x) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_f16(eg_emu_e4m3(x & 0xffu)) | ((uint32_t)eg_emu_f32_to_f16(eg_emu_e4m3((x >> 8) & 0xffu)) << 16);
#else
    uint32_t r;
    asm("{\n.reg .b16 lo;\ncvt.u16.u32 lo, %1;\ncvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(r) : "r"(x));
    return r;
#endif
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
    return __ldcg(p);
#endif
}

// eg_ld_cg2 loads two floats (8-byte aligned) through L2 only.
EG_DEV float2 eg_ld_cg2(const float* p) {
#if defined(EG_HOST_EMU)
    const volatile float* q = p;
    return float2{q[0], q[1]};
#else
    return __ldcg(reinterpret_cast<const float2*>(p));
#endif
}

// eg_cp_async16 copies 16 bytes global to shared without staging in registers.
EG_DEV void eg_cp_async16(void* smem, const void* gmem) {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async(smem, gmem, 16);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
#endif
}

// eg_cp_async4 copies 4 bytes global to shared (through L1).
EG_DEV void eg_cp_async4(void* smem, const void* gmem) {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async(smem, gmem, 4);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(gmem));
#endif
}

EG_DEV void eg_cp_async_commit() {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async_commit();
#else
    asm volatile("cp.async.commit_group;");
#endif
}

// eg_cp_async_wait waits until at most n committed groups are pending.
template <int n> EG_DEV void eg_cp_async_wait() {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async_wait(n);
#else
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

// eg_mma_f16 is d = a·b + c for one m16n8k16 tile, F16 inputs, FP32 accumulate.
EG_DEV void eg_mma_f16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
#if defined(EG_HOST_EMU)
    eg_emu_mma_f16_16816(d, a, b, c);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
}

// eg_wgmma_desc returns a wgmma shared-memory matrix descriptor (no
// swizzle) for the matrix at p with leading and stride byte offsets.
EG_DEV uint64_t eg_wgmma_desc(const void* p, uint32_t lbo, uint32_t sbo) {
#if defined(EG_HOST_EMU)
    const uint64_t a = (uint64_t)((const unsigned char*)p - eg_emu_dynamic_smem());
#else
    const uint64_t a = (uint32_t)__cvta_generic_to_shared(p);
#endif
    return ((a & 0x3ffffu) >> 4) | (uint64_t)((lbo & 0x3ffffu) >> 4) << 16 | (uint64_t)((sbo & 0x3ffffu) >> 4) << 32;
}

// eg_fence_proxy_async orders this thread's generic-proxy shared memory
// writes (st.shared, cp.async) before later async-proxy reads (wgmma).
EG_DEV void eg_fence_proxy_async() {
#if !defined(EG_HOST_EMU)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

EG_DEV void eg_wgmma_fence() {
#if defined(EG_HOST_EMU)
    eg_emu_wgmma_fence();
#else
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
}

EG_DEV void eg_wgmma_commit() {
#if defined(EG_HOST_EMU)
    eg_emu_wgmma_commit();
#else
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#endif
}

// eg_wgmma_wait waits until at most n wgmma groups are pending.
template <int n> EG_DEV void eg_wgmma_wait() {
#if defined(EG_HOST_EMU)
    eg_emu_wgmma_wait(n);
#else
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(n) : "memory");
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

#if defined(EG_HOST_EMU)
EG_DEV uint32_t __funnelshift_l(uint32_t lo, uint32_t hi, uint32_t shift) {
    return (uint32_t)((((uint64_t)hi << 32) | (uint64_t)lo) << (shift & 31u) >> 32);
}
#endif
// eg_dequant_1w: 9 instructions (bitopt).
__device__ __forceinline__ void eg_dequant_1w(const uint32_t (&q)[1], uint32_t (&a)[4]) {
    const uint32_t t1 = q[0] << 4;
    const uint32_t t2 = __funnelshift_l(q[0], q[0], 10);
    const uint32_t t3 = q[0] << 6;
    const uint32_t t4 = q[0] >> 2;
    a[0] = (q[0] & 0x81c081c0u);
    a[1] = (t1 & 0x81c081c0u);
    a[2] = (t2 & 0x81c081c0u);
    a[3] = (t3 & 0x80c080c0u) | (t4 & 0x01000100u);
}

// eg_dequant_2w: 16 instructions (bitopt).
__device__ __forceinline__ void eg_dequant_2w(const uint32_t (&q)[2], uint32_t (&a)[8]) {
    const uint32_t t0 = q[0] << 2;
    const uint32_t t1 = q[0] << 6;
    const uint32_t t2 = __funnelshift_l(q[0], q[0], 12);
    const uint32_t t3 = q[1] << 4;
    const uint32_t t4 = __funnelshift_l(q[1], q[1], 10);
    const uint32_t t5 = __funnelshift_l(q[1], q[1], 14);
    const uint32_t t6 = __byte_perm(q[0], q[1], 0x4163);
    a[0] = (t0 & 0x81c081c0u);
    a[1] = (t1 & 0x81c081c0u);
    a[2] = (t2 & 0x81c081c0u);
    a[3] = (t3 & 0x81c081c0u);
    a[4] = (t4 & 0x81c081c0u);
    a[5] = (t5 & 0x81c081c0u);
    a[6] = (t6 & 0x81c081c0u);
    a[7] = (q[0] & 0x01800180u) | (q[1] & 0x80408040u);
}

// eg_dequant9 unpacks a dequant9-repacked 32-bit word into four bf16x2
// registers holding unscaled E2M1 subnormals (v * 2^-126 as bf16) in 9
// integer instructions via bitopt's eg_dequant_1w.
EG_DEV void eg_dequant9(uint32_t q, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
    const uint32_t qv[1] = {q};
    uint32_t av[4];
    eg_dequant_1w(qv, av);
    a0 = av[0]; a1 = av[1]; a2 = av[2]; a3 = av[3];
}

// eg_scale_e4m3_bf16x2 converts an E4M3 byte b to bf16x2 of (sf * 2^118, sf * 2^118):
// positive normal E4M3 bytes take the 1-instruction b * 16 + 0x7700 fast path;
// zero, subnormals, negatives and 0x7f/0xff NaN take the exact fallback.
EG_DEV uint32_t eg_scale_e4m3_bf16x2(uint32_t b) {
    const uint32_t byte = b & 0xffu;
    if ((byte & 0x80u) == 0u && ((byte >> 3) & 15u) != 0u && byte != 0x7fu) {
        const uint32_t x = byte | (byte << 16);
        return x * 16u + 0x77007700u;
    }
    if ((byte & 0x7fu) == 0x7fu) return 0x7fc07fc0u;
    const float v = eg_e4m3_to_f32(byte) * __uint_as_float(0x7a800000u); // 2^118
    const uint32_t u = eg_f32_to_bf16(v);
    return u | (u << 16);
}

EG_DEV uint32_t eg_mul_bf16x2(uint32_t a, uint32_t b) {
#if defined(EG_HOST_EMU)
    return eg_pack_bf16x2(eg_bf16_to_f32(a) * eg_bf16_to_f32(b), eg_bf16_hi_to_f32(a) * eg_bf16_hi_to_f32(b));
#else
    uint32_t r;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
#endif
}

EG_DEV uint32_t eg_wgmma_decode_nibble(uint32_t q, int j) {
    const uint32_t x = q >> (4 * j);
    return ((x << 6) & 0x01C001C0u) | ((x << 12) & 0x80008000u);
}

// eg_wgmma_desc_128b returns a 128B-swizzled wgmma shared-memory matrix
// descriptor (layout mode 1, bits [63:62] = 1, LBO = 16) for the matrix at p
// with stride byte offset sbo.
EG_DEV uint64_t eg_wgmma_desc_128b(const void* p, uint32_t sbo) {
#if defined(EG_HOST_EMU)
    return eg_emu_wgmma_desc_128b(p, sbo);
#else
    const uint64_t a = (uint32_t)__cvta_generic_to_shared(p);
    return ((a & 0x3ffffu) >> 4) | ((uint64_t)1u << 16) | ((uint64_t)((sbo & 0x3ffffu) >> 4) << 32) | ((uint64_t)1u << 62);
#endif
}

EG_DEV uint32_t eg_swizzle_128b(uint32_t offset) {
    return offset ^ (((offset >> 7) & 7u) << 4);
}

#if !defined(EG_HOST_EMU)
EG_DEV uint32_t eg_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#endif

// eg_mbar_init initializes the mbarrier at bar to complete each phase after count arrivals.
EG_DEV void eg_mbar_init(uint64_t* bar, uint32_t count) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_init(bar, (int)count);
#else
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(eg_smem_u32(bar)), "r"(count) : "memory");
#endif
}

// eg_fence_mbar_init makes initialized mbarriers visible to the async proxy.
EG_DEV void eg_fence_mbar_init() {
#if !defined(EG_HOST_EMU)
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

EG_DEV void eg_mbar_arrive(uint64_t* bar) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_arrive(bar);
#else
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(eg_smem_u32(bar)) : "memory");
#endif
}

// eg_mbar_arrive_expect_tx arrives on bar and adds bytes to the transaction
// bytes its phase waits for.
EG_DEV void eg_mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_arrive_expect_tx(bar, bytes);
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(eg_smem_u32(bar)), "r"(bytes) : "memory");
#endif
}

// eg_mbar_wait returns once the phase of bar with parity parity has completed.
EG_DEV void eg_mbar_wait(uint64_t* bar, uint32_t parity) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_wait(bar, (int)parity);
#else
    asm volatile(
        "{\n.reg .pred p;\nEG_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra EG_MBAR_WAIT_%=;\n}\n" ::"r"(eg_smem_u32(bar)), "r"(parity) : "memory");
#endif
}

// eg_bulk_copy copies bytes, a multiple of 16, from global to shared memory
// with cp.async.bulk, completing them on bar's transaction bytes.
EG_DEV void eg_bulk_copy(void* smem, const void* gmem, uint32_t bytes, uint64_t* bar) {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async_bulk(smem, gmem, bytes, bar);
#else
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(eg_smem_u32(smem)), "l"(gmem), "r"(bytes), "r"(eg_smem_u32(bar)) : "memory");
#endif
}

// eg_smem_align1024 rounds p up to a 1024-byte aligned shared address, as
// 128B-swizzled wgmma operands need.
EG_DEV unsigned char* eg_smem_align1024(unsigned char* p) {
#if defined(EG_HOST_EMU)
    return p + (1024 - (size_t)(p - eg_emu_dynamic_smem()) % 1024) % 1024;
#else
    return p + ((1024u - (eg_smem_u32(p) & 1023u)) & 1023u);
#endif
}

// eg_setmaxnreg_dec and eg_setmaxnreg_inc set the calling warpgroup's
// registers per thread to R.
template <int R> EG_DEV void eg_setmaxnreg_dec() {
#if !defined(EG_HOST_EMU)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R));
#endif
}

template <int R> EG_DEV void eg_setmaxnreg_inc() {
#if !defined(EG_HOST_EMU)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R));
#endif
}

EG_DEV void moe8_moe_w13_ws_wgmma(float* d, const uint32_t a[4], uint64_t db) {
#if defined(EG_HOST_EMU)
    eg_emu_wgmma_bf16(d, a, db, 128);
#else
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
        "}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
#endif
}

// moe8_moe_w13_ws: out = (mul (silu $0) $1) over x[M, 4096] · W[8192, 4096]ᵀ (nvfp4), 128x128 tiles, 4 stages, warp-specialized wgmma.
__global__ void __launch_bounds__(384, 1) moe8_moe_w13_ws(
    int m_total,
    const uint16_t* __restrict__ x,
    const int32_t* __restrict__ rows,
    const uint32_t* __restrict__ w,
    const uint8_t* __restrict__ w_sf,
    const float* __restrict__ w_gs,
    const int32_t* __restrict__ offsets,
    uint16_t* __restrict__ out) {
    constexpr int N = 8192, K = 4096, BM = 128, BN = 128, BK = 64, STAGES = 4, KT = K / BK;
    constexpr int NOUT = N / 2;
    // Stage: x [BM][64] BF16, then 256 threads' fragment words and scales.
    constexpr int XBYTES = BM * 128, WBYTES = 256 * 16, SBYTES = 256 * 8;
    constexpr int TX = XBYTES + WBYTES + SBYTES, STAGE = (TX + 1023) / 1024 * 1024;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int e = blockIdx.z, row0 = offsets[e], row1 = offsets[e + 1];
    const int bm0 = row0 + blockIdx.y * BM, bn0 = blockIdx.x * BN;
    if (bm0 >= row1) return;
    EG_DYNAMIC_SMEM(unsigned char, smem_raw);
    unsigned char* const smem = eg_smem_align1024(smem_raw); // [STAGES][STAGE], then full[STAGES], empty[STAGES]
    uint64_t* const full = reinterpret_cast<uint64_t*>(smem + STAGES * STAGE);
    uint64_t* const empty = full + STAGES;
    const uint32_t* const we = w + ((size_t)e * (N / 128) + blockIdx.x) * KT * 256 * 4;
    const uint8_t* const se = w_sf + ((size_t)e * (N / 128) + blockIdx.x) * KT * 256 * 8;
    const float gs = w_gs[e] * 256.0f;
    __shared__ int srow[BM];
    for (int r = tid; r < BM; r += 384) srow[r] = rows[min(bm0 + r, row1 - 1)];
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) {
            eg_mbar_init(&full[s], 1);
            eg_mbar_init(&empty[s], 8);
        }
        eg_fence_mbar_init();
    }
    __syncthreads();

    if (warp < 4) {
        // Producer warpgroup.
        eg_setmaxnreg_dec<40>();
        for (int kt = 0; kt < KT; ++kt) {
            const int s = kt % STAGES;
            if (kt >= STAGES) eg_mbar_wait(&empty[s], (kt / STAGES - 1) & 1);
            unsigned char* const sa = smem + s * STAGE;
            if (tid == 0) eg_mbar_arrive_expect_tx(&full[s], TX);
            for (int c = tid; c < BM * 8; c += 128) {
                const int r = c / 8, ch = c % 8;
                eg_bulk_copy(sa + eg_swizzle_128b(r * 128 + ch * 16), x + (size_t)srow[r] * K + kt * BK + ch * 8, 16, &full[s]);
            }
            if (tid == 0) {
                eg_bulk_copy(sa + XBYTES, we + (size_t)kt * 256 * 4, WBYTES, &full[s]);
                eg_bulk_copy(sa + XBYTES + WBYTES, se + (size_t)kt * 256 * 8, SBYTES, &full[s]);
            }
        }
        return;
    }

    // Consumer warpgroups.
    eg_setmaxnreg_inc<232>();
    const int wgi = warp / 4 - 1;
    const int ct = tid - 128; // 0..255 over both consumer warpgroups, as RepackNVFP4WGMMA numbers threads
    float d[BM / 2] = {};
    uint32_t a[2][4][4]; // two steps' A fragments: a step's stay untouched while its wgmma runs
    // step decodes stage s's fragments into av and issues its wgmmas.
    auto step = [&](int kt, uint32_t (&av)[4][4]) {
        const int s = kt % STAGES;
        eg_mbar_wait(&full[s], (kt / STAGES) & 1);
        const unsigned char* const sa = smem + s * STAGE;
        const eg_u32x4 q = *reinterpret_cast<const eg_u32x4*>(sa + XBYTES + ct * 16);
        const eg_u32x2 sc = *reinterpret_cast<const eg_u32x2*>(sa + XBYTES + WBYTES + ct * 8);
        const uint32_t qs[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // Scale byte 2*ks + h is row (lane/4) + 8h's scale for this 16 k.
            const uint32_t sw = (ks < 2 ? sc.x : sc.y) >> (16 * (ks & 1));
            const uint32_t s0 = eg_scale_e4m3_bf16x2(sw), s1 = eg_scale_e4m3_bf16x2(sw >> 8);
#pragma unroll
            for (int j = 0; j < 4; ++j) av[ks][j] = eg_mul_bf16x2(eg_wgmma_decode_nibble(qs[ks], j), (j & 1) ? s1 : s0);
        }
        eg_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) moe8_moe_w13_ws_wgmma(d, av[ks], eg_wgmma_desc_128b(sa + ks * 32, 1024));
        eg_wgmma_commit();
        // The previous step's group has retired: release its stage.
        eg_wgmma_wait<1>();
        if (kt > 0) {
            __syncwarp();
            if (lane == 0) eg_mbar_arrive(&empty[(kt - 1) % STAGES]);
        }
    };
    for (int kt = 0; kt < KT; kt += 2) {
        step(kt, a[0]);
        if (kt + 1 < KT) step(kt + 1, a[1]);
    }
    eg_wgmma_wait<0>();
    __syncwarp();
    if (lane == 0) eg_mbar_arrive(&empty[(KT - 1) % STAGES]);
    // Accumulator d[nt*4 + i]: fragment row 64*wgi + (warp % 4)*16 + lane/4 + 8*(i/2), token nt*8 + 2*(lane%4) + (i&1).
    const bool even = (lane / 4) % 2 == 0;
#pragma unroll
    for (int nt = 0; nt < BM / 8; ++nt) {
        const int m0 = bm0 + nt * 8 + 2 * (lane % 4);
        float y[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float v = d[nt * 4 + i] * gs, u = d[nt * 4 + 2 + i] * gs;
            const int m = m0 + (i & 1), n = blockIdx.x * 64 + (wgi * 4 + warp % 4) * 8 + lane / 4;
            y[i] = m < row1 ? (eg_silu((v)) * (u)) : 0.0f;
        }
        {
            const float mine = even ? y[0] : y[1];
            const float theirs = __shfl_xor_sync(0xffffffffu, even ? y[1] : y[0], 4);
            const float yv[2] = {even ? mine : theirs, even ? theirs : mine};
            const int m = m0 + (even ? 0 : 1);
            if (m < row1) eg_store<2>(out + (size_t)m * NOUT + blockIdx.x * 64 + (wgi * 4 + warp % 4) * 8 + lane / 4 - (even ? 0 : 1), yv);
        }
    }
}

inline void moe8_moe_w13_ws_launch(int m, int max_rows, int experts, const uint16_t* __restrict__ x, const int32_t* __restrict__ rows, const uint32_t* __restrict__ w, const uint8_t* __restrict__ w_sf, const float* __restrict__ w_gs, const int32_t* __restrict__ offsets, uint16_t* __restrict__ out, cudaStream_t stream) {
    constexpr int smem = 91200;
    static const cudaError_t attr = cudaFuncSetAttribute(moe8_moe_w13_ws, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)attr;
    const dim3 grid(64, (max_rows + 127) / 128, experts), block(384);
    if (grid.y > 0 && grid.z > 0) EG_LAUNCH(moe8_moe_w13_ws, grid, block, smem, stream, m, x, rows, w, w_sf, w_gs, offsets, out);
}
