package hopper

// nvfp4TMADevice is a direct port of the retained v27 device implementation.
// Origin SHA256: 570ec10423625b76fa891b2ef02a7cc9b56a3b8e529b125c98ae3d4989883989
// Device-prefix SHA256 before comment backticks become apostrophes: fc7b155374f29c9d6271a5e08f3c6ec4f5343ef3dc2c1c47d4cf5998b70352ef
// Only the namespace is substituted when emitted. No CUDA compiler or GPU
// qualification is implied by the host tests of this port.
const nvfp4TMADevice = `
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace pm {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra W%=;\n"
      "}\n" ::"r"(bar), "r"(parity) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 r;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a));
  return r;
}

__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 r;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "r"(a));
  return r;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(0u), "r"(sel));
  return r;
}

__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// (s, s) as bf16x2 with s = sf * 2^118, sf the E4M3 byte 'b' (0-7) of the 8-byte scale pair.
template <bool FAST>
__device__ __forceinline__ uint32_t scale2(uint2 sc, int b) {
  uint32_t w = b < 4 ? sc.x : sc.y;
  uint32_t sel = 0x4040u | ((uint32_t)(b & 3) << 8) | (uint32_t)(b & 3);
  uint32_t x = prmt(w, sel);
  if (FAST) {
    // Positive normal E4M3: exponent/mantissa into BF16's, rebiased by 238 - 127 + 7 = 118.
    return x * 16u + 0x77007700u;
  } else {
    uint32_t byte = x & 0xffu;
    uint32_t e = (byte >> 3) & 0xfu, m = byte & 7u;
    float v = e ? ldexpf(1.0f + m * 0.125f, (int)e - 7) : ldexpf((float)m, -9);
    if (byte & 0x80u) v = -v;
    __nv_bfloat16 h = __float2bfloat16_rn(ldexpf(v, 118));
    uint32_t u = *reinterpret_cast<uint16_t*>(&h);
    return u | (u << 16);
  }
}

// dequant16 (egraph-expander's bitopt, proved optimal in its model): a pair of fragment words to
// their 8 registers as BF16 * 2^-126 in 16 instructions, for the pair pack of CODE_BITS in
// moe_wgmma.py. Register k of the pair is register 4 * (k / 4) + k % 4 of k-steps 2p, 2p + 1.
__device__ __forceinline__ void dequant16(uint32_t q0, uint32_t q1, uint32_t (&a)[8]) {
  constexpr uint32_t M = 0x81C081C0u;
  a[0] = (q0 << 2) & M;
  a[1] = (q0 << 6) & M;
  a[2] = __funnelshift_l(q0, q0, 12) & M;
  a[3] = (q1 << 4) & M;
  a[4] = __funnelshift_l(q1, q1, 10) & M;
  a[5] = __funnelshift_l(q1, q1, 14) & M;
  a[6] = __byte_perm(q0, q1, 0x4163) & M;
  a[7] = (q0 & 0x01800180u) | (q1 & 0x80408040u);
}

__device__ __forceinline__ uint64_t desc_b(uint32_t addr) {
  // K-major, 128B swizzle: start >> 4, LBO 1 (unused), SBO 1024 B >> 4, layout 1 (128B).
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) |
         ((uint64_t)1 << 62);
}

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ float silu(float v) { return v / (1.0f + __expf(-v)); }

}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_32(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_48(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_64(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_80(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, {%40, %41, %42, %43}, %44, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_96(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%48, %49, %50, %51}, %52, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_128(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_160(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n160k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79}, {%80, %81, %82, %83}, %84, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_192(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, {%96, %97, %98, %99}, %100, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_224(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n224k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111}, {%112, %113, %114, %115}, %116, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
__device__ __forceinline__ void wgmma_256(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));
}
}  // namespace pm

namespace pm {
template <int NN>
__device__ __forceinline__ void wgmma_n(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {
  if constexpr (NN == 16) wgmma_16(d, a0, a1, a2, a3, b);
  if constexpr (NN == 32) wgmma_32(d, a0, a1, a2, a3, b);
  if constexpr (NN == 48) wgmma_48(d, a0, a1, a2, a3, b);
  if constexpr (NN == 64) wgmma_64(d, a0, a1, a2, a3, b);
  if constexpr (NN == 80) wgmma_80(d, a0, a1, a2, a3, b);
  if constexpr (NN == 96) wgmma_96(d, a0, a1, a2, a3, b);
  if constexpr (NN == 128) wgmma_128(d, a0, a1, a2, a3, b);
  if constexpr (NN == 160) wgmma_160(d, a0, a1, a2, a3, b);
  if constexpr (NN == 192) wgmma_192(d, a0, a1, a2, a3, b);
  if constexpr (NN == 224) wgmma_224(d, a0, a1, a2, a3, b);
  if constexpr (NN == 256) wgmma_256(d, a0, a1, a2, a3, b);
}
}  // namespace pm

namespace pm {

constexpr int BK = 64;

// One CTA tile's k-loop with a wgmma N of NN (<= BM) activation rows: d[u] accumulates weight tile u.
// PROBE (lab only): 1 skips the decode (raw words as A, one LOP3 per register), 2 skips the wgmma
// (decode kept live), 3 skips both (the data pipeline alone).
template <int NN, int BM, int NSUB, int STAGES, bool FAST, int PROBE>
__device__ __forceinline__ void tile_loop(float (&d)[NSUB][BM / 2], uint32_t it, int KB, uint32_t x0,
                                          uint32_t full0, uint32_t empty0, int ct, int lt) {
  constexpr int XBYTES = BM * BK * 2, WBYTES = 256 * 16, SBYTES = 64 * 8;
  constexpr int SX = XBYTES + NSUB * (WBYTES + SBYTES), SB = (SX + 1023) / 1024 * 1024;  // bytes, stride
  uint32_t a[2][NSUB][16];
  uint4 q[NSUB];
  uint2 sc[NSUB];
  // Stage and parity of step k, carried from step to step (no division in the loop).
  uint32_t s = it % STAGES, ph = (it / STAGES) & 1, sprev = 0;
  auto fetch = [&](uint32_t s, uint32_t ph) {
    mbar_wait(full0 + 8 * s, ph);
#pragma unroll
    for (int u = 0; u < NSUB; u++) {
      const uint32_t st = x0 + s * SB + XBYTES + u * (WBYTES + SBYTES);
      q[u] = lds128(st + 16 * ct);
      sc[u] = lds64(st + WBYTES + 8 * (ct / 4));  // a lane quad shares its rows' scales
    }
  };
  auto dec = [&](uint32_t (&ab)[NSUB][16]) {
#pragma unroll
    for (int u = 0; u < NSUB; u++) {
      const uint32_t qs[4] = {q[u].x, q[u].y, q[u].z, q[u].w};
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        if constexpr (PROBE == 1 || PROBE == 3) {
          // One LOP3 per register, so that the A registers are not the lds destinations
          // (which would serialize the wgmmas, C7513).
#pragma unroll
          for (int r = 0; r < 4; r++) ab[u][4 * ks + r] = qs[ks] | (0x00010001u << r);
          continue;
        }
        if (ks % 2) continue;
        uint32_t r[8];
        dequant16(qs[ks], qs[ks + 1], r);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          const int kk = ks + k / 4, j = k % 4;  // k-step, register within it
          ab[u][4 * kk + j] = bmul(r[k], scale2<FAST>(sc[u], 2 * kk + j % 2));
        }
      }
    }
  };
  fetch(s, ph);
  dec(a[0]);
  for (int kb = 0; kb < KB; kb += 2) {
#pragma unroll
    for (int v = 0; v < 2; v++) {
      const int k = kb + v;
      const uint64_t db = desc_b(x0 + s * SB);  // + 2 per 16 k (32 bytes)
      wg_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int u = 0; u < NSUB; u++)
          if constexpr (PROBE == 2 || PROBE == 3)
            asm volatile("" ::"r"(a[v][u][4 * ks]), "r"(a[v][u][4 * ks + 1]), "r"(a[v][u][4 * ks + 2]),
                         "r"(a[v][u][4 * ks + 3]));
          else
            wgmma_n<NN>(d[u], a[v][u][4 * ks], a[v][u][4 * ks + 1], a[v][u][4 * ks + 2], a[v][u][4 * ks + 3],
                        db + 2 * ks);
      wg_commit();
      if (k + 1 < KB) {
        uint32_t sn = s + 1, pn = ph;
        if (sn == STAGES) sn = 0, pn ^= 1;
        fetch(sn, pn);
        wg_wait<1>();
        if (k > 0 && lt == 0) mbar_arrive(empty0 + 8 * sprev);
        dec(a[v ^ 1]);
        sprev = s, s = sn, ph = pn;
      }
    }
  }
  wg_wait<0>();
  if (lt == 0) {
    if (KB > 1) mbar_arrive(empty0 + 8 * sprev);
    mbar_arrive(empty0 + 8 * s);
  }
}

// wq: [E][NT][K/64][256][4] uint32 fragment words; ws: [E][NT][K/64][256] 8-byte scale sets.
// nprod: producer threads (1-4).
// bvalid[mb]: rows of block mb that are real (the rest is padding); the wgmma N is that, rounded
// up to NSTEP. out: [rows, OUTN] BF16, OUTN = 64 * NT when SILU else 128 * NT.
// A CTA tile is NSUB consecutive 128-row weight tiles against one BM-row activation tile.
// VARN (a BLOCK_M in the generator's NN table, FAST scales): the k-loop's wgmma N and the activation rows loaded (NSTEP-row TMA boxes) follow
// bvalid; above that every tile runs the full BM, since nearly all blocks are full there and the
// per-N k-loop copies cost more than the padding. FAST=false (a fallback
// the judge's weights do not take) also runs the full BM, to halve the compile.
template <int BM, int NSUB, int STAGES, bool SILU, bool FAST, int PROBE = 0>
__global__ void __launch_bounds__(384, 1)
gemm_kernel(const __grid_constant__ CUtensorMap tmx, const uint4* __restrict__ wq, const uint2* __restrict__ ws,
            const float* __restrict__ gs, const int* __restrict__ eids, const int* __restrict__ bvalid,
            const int* __restrict__ npost, __nv_bfloat16* __restrict__ out, int K, int NT, int group_m, int nprod) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  // Stage: activation tile (TMA, 128B swizzle), then per weight tile this k-block's words and scales.
  constexpr int XBYTES = BM * BK * 2, WBYTES = 256 * 16, SBYTES = 64 * 8;
  constexpr int SX = XBYTES + NSUB * (WBYTES + SBYTES), SB = (SX + 1023) / 1024 * 1024;  // bytes, stride
  uint64_t* bars = (uint64_t*)(smem + STAGES * SB);
  const uint32_t x0 = smem_u32(smem);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + STAGES);
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 2);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int KB = K / BK;
  const int NTC = NT / NSUB;
  const int num_mb = npost[0] / BM;
  const int total = num_mb * NTC;
  const int wg = tid / 128;
  constexpr bool VARN = ((BM == 64 && NSUB == 1) || (BM == 128 && NSUB == 1) || (BM == 256 && NSUB == 1)) && FAST;
  constexpr int NSTEP = (BM == 64 && NSUB == 1 ? 16 : BM == 128 && NSUB == 1 ? 16 : BM == 256 && NSUB == 1 ? 32 : BM);

  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    // Lane 0 of each of the first nprod warps issues every nprod-th stage: one thread's issue rate
    // limits the weight stream at small N.
    if (tid % 32 == 0 && tid / 32 < nprod) {
      const int pid = tid / 32;
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmx) : "memory");
      uint32_t it = 0;
      for (int t = blockIdx.x; t < total; t += gridDim.x) {
        const int in_group = group_m * NTC;
        const int first = (t / in_group) * group_m;
        const int gsz = min(num_mb - first, group_m);
        const int mb = first + (t % in_group) % gsz;
        const int ntc = (t % in_group) / gsz;
        const size_t wbase = (size_t)(eids[mb] * NT + ntc * NSUB) * KB * 256;
        const int nb = VARN ? min(BM, bvalid[mb] + NSTEP - 1) / NSTEP : 1;
        const uint32_t bytes = VARN ? nb * NSTEP * BK * 2 + NSUB * (WBYTES + SBYTES) : SX;
        // This thread's first k-step of the tile: the one with it + kb == pid (mod nprod).
        const int kb0 = (pid + nprod - (int)(it % nprod)) % nprod;
        for (int kb = kb0; kb < KB; kb += nprod) {
          const uint32_t i = it + kb;
          const uint32_t s = i % STAGES;
          const uint32_t ph = (i / STAGES) & 1;
          const uint32_t st = x0 + s * SB;
          mbar_wait(empty0 + 8 * s, ph ^ 1);
          mbar_expect_tx(full0 + 8 * s, bytes);
          for (int b = 0; b < nb; b++)
            tma_load_2d(st + b * (VARN ? NSTEP * BK * 2 : 0), &tmx, full0 + 8 * s, kb * BK, mb * BM + NSTEP * b);
#pragma unroll
          for (int u = 0; u < NSUB; u++) {
            const size_t off = wbase + ((size_t)u * KB + kb) * 256;
            const uint32_t dst = st + XBYTES + u * (WBYTES + SBYTES);
            bulk_load(dst, wq + off, WBYTES, full0 + 8 * s);
            bulk_load(dst + WBYTES, ws + off / 4, SBYTES, full0 + 8 * s);
          }
        }
        it += KB;
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");

  const int ct = tid - 128;          // 0..255 over both consumer warpgroups
  const int lt = ct & 127;           // within the warpgroup
  const int warp = lt / 32, lane = lt % 32, g = lane / 4, tq = lane % 4;
  const int cw = ct / 128;           // consumer warpgroup: rows 64 * cw ... of each weight tile
  float d[NSUB][BM / 2];
  uint32_t it = 0;
  for (int t = blockIdx.x; t < total; t += gridDim.x) {
    const int in_group = group_m * NTC;
    const int first = (t / in_group) * group_m;
    const int gsz = min(num_mb - first, group_m);
    const int mb = first + (t % in_group) % gsz;
    const int ntc = (t % in_group) / gsz;
    const int e = eids[mb];
    const int nn = VARN ? min(BM, (bvalid[mb] + NSTEP - 1) / NSTEP * NSTEP) : BM;
#pragma unroll
    for (int u = 0; u < NSUB; u++)
#pragma unroll
      for (int i = 0; i < BM / 2; i++) d[u][i] = 0.f;
    if constexpr (VARN) {
      if constexpr (BM == 64 && NSUB == 1) {
        switch (nn / NSTEP) { case 1: tile_loop<16, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 2: tile_loop<32, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 3: tile_loop<48, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 4: tile_loop<64, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; }
      } else if constexpr (BM == 128 && NSUB == 1) {
        switch (nn / NSTEP) { case 1: tile_loop<16, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 2: tile_loop<32, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 3: tile_loop<48, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 4: tile_loop<64, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 5: tile_loop<80, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 6: tile_loop<96, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 7: case 8: tile_loop<128, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; }
      } else if constexpr (BM == 256 && NSUB == 1) {
        switch (nn / NSTEP) { case 1: case 2: case 3: case 4: case 5: tile_loop<160, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 6: tile_loop<192, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; case 7: case 8: tile_loop<256, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt); break; }
      }
    } else {
      tile_loop<BM, BM, NSUB, STAGES, FAST, PROBE>(d, it, KB, x0, full0, empty0, ct, lt);
    }
    it += KB;

    const float gsv = gs[e];
    const int p0 = mb * BM;
#pragma unroll
    for (int u = 0; u < NSUB; u++) {
      const int nt = ntc * NSUB + u;
      if (SILU) {
        const int ncol = nt * 64 + (cw * 4 + warp) * 8 + g;
        const int OUTN = NT * 64;
#pragma unroll
        for (int j = 0; j < BM / 8; j++) {
          if (8 * j >= nn) break;
          const int p = p0 + 8 * j + 2 * tq;
          const float g0 = d[u][4 * j] * gsv, g1 = d[u][4 * j + 1] * gsv;
          const float u0 = d[u][4 * j + 2] * gsv, u1 = d[u][4 * j + 3] * gsv;
          out[(size_t)p * OUTN + ncol] = __float2bfloat16_rn(silu(g0) * u0);
          out[(size_t)(p + 1) * OUTN + ncol] = __float2bfloat16_rn(silu(g1) * u1);
        }
      } else {
        const int n0 = nt * 128 + cw * 64 + warp * 16 + g;
        const int OUTN = NT * 128;
#pragma unroll
        for (int j = 0; j < BM / 8; j++) {
          if (8 * j >= nn) break;
          const int p = p0 + 8 * j + 2 * tq;
          out[(size_t)p * OUTN + n0] = __float2bfloat16_rn(d[u][4 * j] * gsv);
          out[(size_t)(p + 1) * OUTN + n0] = __float2bfloat16_rn(d[u][4 * j + 1] * gsv);
          out[(size_t)p * OUTN + n0 + 8] = __float2bfloat16_rn(d[u][4 * j + 2] * gsv);
          out[(size_t)(p + 1) * OUTN + n0 + 8] = __float2bfloat16_rn(d[u][4 * j + 3] * gsv);
        }
      }
    }
  }
}

}  // namespace pm
`
