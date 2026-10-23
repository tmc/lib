// Host emulation of the CUDA subset that hopper emits.
//
// Each CUDA thread is a std::thread; blocks run one after another, so a
// function-local static stands in for __shared__ memory. All of them live
// in one linker section, which each block starts with filled with 0xff
// bytes (NaN as FP32, BF16, F16 and E4M3), as is dynamic shared memory: a
// kernel that reads shared memory it did not write in that block reads NaN
// here, as it may read garbage on a GPU. Define EG_EMU_ZERO_SMEM to start
// from zeros instead. Warp collectives
// (shuffles, ldmatrix, mma.sync) exchange operands through a per-warp buffer
// between two barrier phases and follow the PTX ISA fragment layouts. A
// thread that returns early drops out of its barriers, so early exit never
// deadlocks the emulation (on a GPU it is the kernel's responsibility).
#pragma once

#define EG_HOST_EMU 1

#include <atomic>
#include <barrier>
#include <algorithm>
#include <mutex>
#include <cmath>
#include <math.h> // expf and the other float functions, global in C and libstdc++
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#if defined(__APPLE__)
#define __shared__ static __attribute__((section("__DATA,__eg_shared")))
extern unsigned char eg_emu_shared_start[] __asm("section$start$__DATA$__eg_shared");
extern unsigned char eg_emu_shared_end[] __asm("section$end$__DATA$__eg_shared");
#else
#define __shared__ static __attribute__((section("eg_shared")))
extern unsigned char __start_eg_shared[], __stop_eg_shared[];
#define eg_emu_shared_start __start_eg_shared
#define eg_emu_shared_end __stop_eg_shared
#endif

// The anchor keeps the section, and its bounds, defined in a program
// without shared memory.
__attribute__((used)) __shared__ unsigned char eg_emu_shared_anchor;

#if defined(EG_EMU_ZERO_SMEM)
constexpr unsigned char eg_emu_smem_fill = 0;
#else
constexpr unsigned char eg_emu_smem_fill = 0xff;
#endif
#define __launch_bounds__(...)

struct dim3 {
    unsigned x = 1, y = 1, z = 1;
    dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};

struct alignas(8) float2 { float x, y; };

inline thread_local dim3 threadIdx, blockIdx;
inline dim3 blockDim, gridDim;
constexpr int warpSize = 32;

struct eg_emu_warp {
    std::barrier<> bar;
    uint64_t val[32];
    const void* ptr[32];
    uint32_t regs[32][8];
    explicit eg_emu_warp(int n) : bar(n) {}
};

struct eg_emu_block {
    std::barrier<> bar;
    std::vector<std::unique_ptr<eg_emu_warp>> warps;
    std::vector<unsigned char> smem;
    explicit eg_emu_block(int n) : bar(n) {}
};

inline thread_local eg_emu_block* eg_emu_cur_block;
inline thread_local eg_emu_warp* eg_emu_cur_warp;
inline thread_local int eg_emu_lane;

inline void __syncthreads() { eg_emu_cur_block->bar.arrive_and_wait(); }
inline void __syncwarp(unsigned = 0xffffffffu) { eg_emu_cur_warp->bar.arrive_and_wait(); }
inline void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }

inline float __uint_as_float(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline uint32_t __float_as_uint(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float __expf(float x) { return std::exp(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline void sincosf(float x, float* s, float* c) { *s = std::sin(x); *c = std::cos(x); }
using std::min; using std::max;

template <typename T> T __ldg(const T* p) { return *p; }

template <typename T> T eg_emu_exchange(T v, int src) {
    static_assert(sizeof(T) <= 8, "shuffle operand too large");
    eg_emu_warp* w = eg_emu_cur_warp;
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    w->val[eg_emu_lane] = bits;
    w->bar.arrive_and_wait();
    uint64_t got = w->val[src & 31];
    w->bar.arrive_and_wait();
    T r;
    std::memcpy(&r, &got, sizeof(T));
    return r;
}
template <typename T> T __shfl_xor_sync(unsigned, T v, int m, int = 32) { return eg_emu_exchange(v, eg_emu_lane ^ m); }
template <typename T> T __shfl_sync(unsigned, T v, int src, int width = 32) {
    return eg_emu_exchange(v, (eg_emu_lane & ~(width - 1)) | (src & (width - 1)));
}
template <typename T> T __shfl_down_sync(unsigned, T v, int d, int = 32) {
    int src = eg_emu_lane + d;
    return eg_emu_exchange(v, src < 32 ? src : eg_emu_lane);
}

inline unsigned __ballot_sync(unsigned, int pred) {
    eg_emu_warp* w = eg_emu_cur_warp;
    w->val[eg_emu_lane] = pred != 0;
    w->bar.arrive_and_wait();
    unsigned r = 0;
    for (int i = 0; i < 32; ++i) r |= (unsigned)(w->val[i] & 1) << i;
    w->bar.arrive_and_wait();
    return r;
}
inline int __popc(unsigned v) { return __builtin_popcount(v); }

// cp.async is modeled at its most adversarial legal timing: issue poisons
// the destination with 0xff (a BF16/FP32 NaN, so a read before the matching
// wait_group, or an overwrite of a slot still being read, shows up as NaN),
// and the data lands only when cp.async.wait_group retires its group.
struct eg_emu_cp {
    void* dst;
    unsigned char src[16];
    int n, group;
};
inline thread_local std::vector<eg_emu_cp> eg_emu_cp_pending;
inline thread_local int eg_emu_cp_committed;

inline void eg_emu_cp_async(void* dst, const void* src, int n) {
    eg_emu_cp c{dst, {}, n, -1};
    std::memcpy(c.src, src, n);
    std::memset(dst, 0xff, n);
    eg_emu_cp_pending.push_back(c);
}
inline void eg_emu_cp_async_commit() {
    for (auto& c : eg_emu_cp_pending)
        if (c.group < 0) c.group = eg_emu_cp_committed;
    ++eg_emu_cp_committed;
}
// eg_emu_cp_async_wait retires every committed group except the newest n.
inline void eg_emu_cp_async_wait(int n) {
    int keep = eg_emu_cp_committed - n;
    std::vector<eg_emu_cp> rest;
    for (auto& c : eg_emu_cp_pending) {
        if (c.group >= 0 && c.group < keep) std::memcpy(c.dst, c.src, c.n);
        else rest.push_back(c);
    }
    eg_emu_cp_pending.swap(rest);
}

template <typename T> T atomicAdd(T* p, T v) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    T old = *p;
    *p = old + v;
    return old;
}

// --- prelude primitives ---------------------------------------------------

inline uint16_t eg_emu_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0 | ((u >> 16) & 0x8000);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// eg_emu_f32_to_f16 rounds to nearest even (cvt.rn.f16.f32).
inline uint16_t eg_emu_f32_to_f16(float f) {
    const uint32_t u = __float_as_uint(f), s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return s | 0x7e00u;
    if (a >= 0x477ff000u) return s | 0x7c00u;
    if (a < 0x38800000u) return s | (uint16_t)std::nearbyint(__uint_as_float(a) * 16777216.0f);
    uint32_t r = (((a >> 23) - 112u) << 10) | ((a >> 13) & 0x3ffu);
    const uint32_t rem = a & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    return s | r;
}

inline float eg_emu_f16_value(uint32_t h) {
    const int e = (h >> 10) & 31, m = h & 0x3ff;
    const float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp((float)(m | 0x400), e - 25);
    return h & 0x8000u ? -v : v;
}

inline uint8_t eg_emu_f32_to_e4m3(float f) {
    uint32_t bits = __float_as_uint(f);
    uint8_t sign = (bits >> 24) & 0x80;
    uint32_t a = bits & 0x7fffffffu;
    if (a > 0x7f800000u) return sign | 0x7f;
    if (a >= 0x43e00000u) return sign | 0x7e;
    if (a < 0x3c800000u) return sign | (uint8_t)std::nearbyint(__uint_as_float(a) * 512.0f);
    int e = (int)(a >> 23) - 127 + 7;
    uint32_t m = (a >> 20) & 7u, rem = a & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) {
        if (++m == 8) { m = 0; ++e; }
    }
    if (e > 15 || (e == 15 && m == 7)) return sign | 0x7e;
    return sign | (uint8_t)(e << 3) | (uint8_t)m;
}

inline uint32_t eg_emu_prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint8_t bytes[8];
    for (int i = 0; i < 4; ++i) { bytes[i] = a >> (8 * i); bytes[4 + i] = b >> (8 * i); }
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t s = (sel >> (4 * i)) & 15u;
        uint8_t v = bytes[s & 7];
        if (s & 8) v = (v & 0x80) ? 0xff : 0x00;
        r |= (uint32_t)v << (8 * i);
    }
    return r;
}

// ldmatrix .m8n8.x4 .b16: lane l supplies the row address of matrix l/8,
// row l%8; register j of lane l receives matrix j, row l/4, columns
// 2(l%4) and 2(l%4)+1.
inline void eg_emu_ldmatrix_x4(uint32_t r[4], const void* row, bool trans) {
    eg_emu_warp* w = eg_emu_cur_warp;
    w->ptr[eg_emu_lane] = row;
    w->bar.arrive_and_wait();
    for (int j = 0; j < 4; ++j) {
        uint16_t h[2];
        for (int k = 0; k < 2; ++k) {
            int rr = eg_emu_lane / 4, cc = 2 * (eg_emu_lane % 4) + k;
            if (trans) std::swap(rr, cc);
            std::memcpy(&h[k], (const uint16_t*)w->ptr[8 * j + rr] + cc, 2);
        }
        r[j] = h[0] | ((uint32_t)h[1] << 16);
    }
    w->bar.arrive_and_wait();
}

inline float eg_emu_bf16(uint32_t reg, int half) { return __uint_as_float((half ? reg >> 16 : reg & 0xffffu) << 16); }
inline float eg_emu_f16(uint32_t reg, int half) { return eg_emu_f16_value(half ? reg >> 16 : reg & 0xffffu); }

// mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}, fragments per the PTX ISA:
// g = lane/4, t = lane%4; a0 (g, 2t..), a1 (g+8, 2t..), a2 (g, 2t+8..), a3 (g+8, 2t+8..);
// b0 (k 2t.., n g), b1 (k 2t+8.., n g); c0,c1 (g, 2t..), c2,c3 (g+8, 2t..).
template <float (*eg_emu_elem)(uint32_t, int)>
inline void eg_emu_mma_16816(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    eg_emu_warp* w = eg_emu_cur_warp;
    uint32_t* mine = w->regs[eg_emu_lane];
    for (int i = 0; i < 4; ++i) mine[i] = a[i];
    mine[4] = b[0];
    mine[5] = b[1];
    w->bar.arrive_and_wait();
    auto A = [&](int row, int col) {
        int lane = (row % 8) * 4 + (col % 8) / 2;
        int reg = (row / 8) + 2 * (col / 8);
        return eg_emu_elem(w->regs[lane][reg], col & 1);
    };
    auto B = [&](int k, int col) {
        int lane = col * 4 + (k % 8) / 2;
        return eg_emu_elem(w->regs[lane][4 + k / 8], k & 1);
    };
    int g = eg_emu_lane / 4, t = eg_emu_lane % 4;
    float out[4];
    for (int i = 0; i < 4; ++i) {
        int row = g + 8 * (i / 2), col = 2 * t + (i & 1);
        float acc = c[i];
        for (int k = 0; k < 16; ++k) acc += A(row, k) * B(k, col);
        out[i] = acc;
    }
    w->bar.arrive_and_wait();
    for (int i = 0; i < 4; ++i) d[i] = out[i];
}

inline void eg_emu_mma_bf16_16816(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    eg_emu_mma_16816<eg_emu_bf16>(d, a, b, c);
}
inline void eg_emu_mma_f16_16816(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    eg_emu_mma_16816<eg_emu_f16>(d, a, b, c);
}

inline float eg_emu_e4m3(uint32_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    float v = e == 15 && m == 7 ? NAN : e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return b & 0x80 ? -v : v;
}

// mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32: as m16n8k16 with four
// k values per register (low byte first): a0 (g, 4t..), a1 (g+8, 4t..),
// a2 (g, 4t+16..), a3 (g+8, 4t+16..); b0 (k 4t.., n g), b1 (k 4t+16.., n g).
inline void eg_emu_mma_e4m3_16832(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    eg_emu_warp* w = eg_emu_cur_warp;
    uint32_t* mine = w->regs[eg_emu_lane];
    for (int i = 0; i < 4; ++i) mine[i] = a[i];
    mine[4] = b[0];
    mine[5] = b[1];
    w->bar.arrive_and_wait();
    auto A = [&](int row, int col) {
        int lane = (row % 8) * 4 + (col % 16) / 4;
        int reg = (row / 8) + 2 * (col / 16);
        return eg_emu_e4m3((w->regs[lane][reg] >> (8 * (col % 4))) & 0xff);
    };
    auto B = [&](int k, int col) {
        int lane = col * 4 + (k % 16) / 4;
        return eg_emu_e4m3((w->regs[lane][4 + k / 16] >> (8 * (k % 4))) & 0xff);
    };
    int g = eg_emu_lane / 4, t = eg_emu_lane % 4;
    float out[4];
    for (int i = 0; i < 4; ++i) {
        int row = g + 8 * (i / 2), col = 2 * t + (i & 1);
        float acc = c[i];
        for (int k = 0; k < 32; ++k) acc += A(row, k) * B(k, col);
        out[i] = acc;
    }
    w->bar.arrive_and_wait();
    for (int i = 0; i < 4; ++i) d[i] = out[i];
}

// wgmma.mma_async m64nNk16 f32.bf16.bf16, D += A·B, A and B K-major in
// shared memory with no-swizzle descriptors: element (r, k) of a matrix
// is at start + (r/8)·SBO + (k/8)·LBO + (r%8)·16 + (k%8)·2. Thread i of
// the warpgroup holds d[nt*4 + j] = D[(i/32%4)*16 + lane/4 + 8(j/2)][nt*8 +
// 2(lane%4) + (j&1)]. Operands come from shared memory, so each thread
// computes its own elements; the emulation completes synchronously.
inline unsigned char* eg_emu_dynamic_smem();
inline void eg_emu_wgmma_bf16(float* d, uint64_t da, uint64_t db, int n) {
    auto at = [](uint64_t desc, int r, int k) {
        const unsigned char* base = eg_emu_dynamic_smem() + ((desc & 0x3fff) << 4);
        const uint64_t lbo = ((desc >> 16) & 0x3fff) << 4, sbo = ((desc >> 32) & 0x3fff) << 4;
        uint16_t h;
        std::memcpy(&h, base + (r / 8) * sbo + (k / 8) * lbo + (r % 8) * 16 + (k % 8) * 2, 2);
        return eg_emu_bf16(h, 0);
    };
    const int wig = (threadIdx.x / 32) % 4, g = eg_emu_lane / 4, t = eg_emu_lane % 4;
    for (int nt = 0; nt < n / 8; ++nt)
        for (int i = 0; i < 4; ++i) {
            const int row = wig * 16 + g + 8 * (i / 2), col = nt * 8 + 2 * t + (i & 1);
            float acc = d[nt * 4 + i];
            for (int k = 0; k < 16; ++k) acc += at(da, row, k) * at(db, col, k);
            d[nt * 4 + i] = acc;
        }
}

// --- launch ---------------------------------------------------------------

inline unsigned char* eg_emu_dynamic_smem() { return eg_emu_cur_block->smem.data(); }

typedef struct eg_emu_stream* cudaStream_t;
typedef int cudaError_t;
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
template <typename K> cudaError_t cudaFuncSetAttribute(K, cudaFuncAttribute, int) { return 0; }

// The emulated device: eg_emu_sms multiprocessors of eg_emu_blocks_per_sm
// resident blocks each, which launchers that size grids from the device
// see. Set them before the first launch.
inline int eg_emu_sms = 4, eg_emu_blocks_per_sm = 2;
enum cudaDeviceAttr { cudaDevAttrMultiProcessorCount = 16 };
inline cudaError_t cudaGetDevice(int* dev) { *dev = 0; return 0; }
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr, int) { *v = eg_emu_sms; return 0; }
template <typename K> cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, K, int, size_t) {
    *n = eg_emu_blocks_per_sm;
    return 0;
}

// eg_emu_last_grid is the grid of the latest launch, for tests of
// launchers that size it.
inline dim3 eg_emu_last_grid;

template <typename K, typename... Args>
void eg_emu_launch(dim3 grid, dim3 block, size_t smem, K kernel, Args... args) {
    eg_emu_last_grid = grid;
    gridDim = grid;
    blockDim = block;
    int n = block.x * block.y * block.z;
    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned bx = 0; bx < grid.x; ++bx) {
        eg_emu_block b(n);
        b.smem.assign(smem + 16, eg_emu_smem_fill);
        std::memset(eg_emu_shared_start, eg_emu_smem_fill, eg_emu_shared_end - eg_emu_shared_start);
        for (int w = 0; w * 32 < n; ++w) b.warps.emplace_back(new eg_emu_warp(std::min(32, n - 32 * w)));
        std::vector<std::thread> threads;
        for (int i = 0; i < n; ++i) {
            threads.emplace_back([&, i] {
                threadIdx = dim3(i % block.x, (i / block.x) % block.y, i / (block.x * block.y));
                blockIdx = dim3(bx, by, bz);
                eg_emu_cur_block = &b;
                eg_emu_cur_warp = b.warps[i / 32].get();
                eg_emu_lane = i % 32;
                kernel(args...);
                eg_emu_cur_warp->bar.arrive_and_drop();
                b.bar.arrive_and_drop();
            });
        }
        for (auto& t : threads) t.join();
    }
}

inline std::vector<unsigned char> eg_emu_read(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror(path); std::exit(2); }
    std::vector<unsigned char> buf;
    unsigned char chunk[1 << 16];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
    std::fclose(f);
    buf.resize(buf.size() + 16);  // slack for vector over-reads past the end
    return buf;
}

inline void eg_emu_write(const char* path, const void* p, size_t n) {
    FILE* f = std::fopen(path, "wb");
    if (!f || std::fwrite(p, 1, n, f) != n) { std::perror(path); std::exit(2); }
    std::fclose(f);
}
