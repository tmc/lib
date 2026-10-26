// cudahostemu.h: host emulation of a CUDA kernel subset, for testing kernel
// text on a CPU with a C++20 compiler.
//
// Each CUDA thread is a std::thread. Blocks run one after another, except
// that the blocks of one thread-block cluster run at once. A function-local
// static stands in for __shared__ memory; all of them live in one linker
// section, which each block (each cluster, when clusters are used) starts
// with filled with 0xff bytes, NaN as FP32, BF16, F16 and E4M3, as is
// dynamic shared memory: a kernel that reads shared memory it did not write
// in that block reads NaN here, as it may read garbage on a GPU. Define
// EG_EMU_ZERO_SMEM to start from zeros instead. cudaMalloc memory starts
// as 0xff bytes for the same reason.
//
// Warp collectives (shuffles, ballot/any/all, ldmatrix, mma.sync) exchange
// operands through a per-warp buffer between two mask-aware barrier phases
// and follow the PTX ISA fragment layouts; disjoint sub-warp masks (such as
// 0xff << 8k) synchronize independently without deadlocking the rest of the
// warp. Named CTA barriers (eg_emu_bar_sync / bar_sync, ids 0..15) synchronize
// a multiple-of-32 thread subset and fail loudly on invalid counts or
// incomplete barriers. cp.async is modeled at its latest legal timing:
// issue fills the destination with 0xff and the data lands only when
// cp.async.wait_group retires its group. Hopper mbarriers and bulk copies
// (eg_emu_mbar_*, eg_emu_cp_async_bulk, eg_emu_cp_async_bulk_tensor_2d) are
// modeled as adversarially: a copy's data lands only when a wait observes
// its phase, and a satisfied wait returns only when no other thread of the
// block can run. wgmma (bf16, f16 and e4m3, with
// shared-A or register-A, K-major or MN-major, and no/32B/64B/128B swizzle)
// is modeled the same way once a thread calls eg_emu_wgmma_fence: the
// accumulator reads NaN, and shared memory is read, only at the
// wgmma.wait_group that retires the op. A thread that returns early drops
// out of its barriers, so early exit never deadlocks the emulation (on a
// GPU it is the kernel's responsibility). Warps may be partial.
//
// Clusters: run_cluster_grid, or eg_emu_launch_cluster, runs the blocks of
// each cluster at once. Static __shared__ storage is one object shared by
// those blocks, so per-block shared memory in a cluster kernel must come
// from MOE8_SHARED (or CUDAHOSTEMU_SHARED), which is per block and starts
// zeroed, or from dynamic shared memory. cooperative_groups::this_cluster()
// provides sync, block_rank, num_blocks and map_shared_rank for both.
//
// The runtime subset (cudaMalloc, cudaMemcpy, cudaMemset, streams,
// cudaGetLastError) works on host memory and completes synchronously.
// Kernels launch with eg_emu_launch or CUDAHOSTEMU_LAUNCH, since <<<>>> is
// not C++.
//
// The eg_emu_* functions are the primitives egraph's hopper prelude calls
// under EG_HOST_EMU; run_grid, run_cluster_grid, MOE8_SHARED, read_file and
// write_file are moe8ops's. Their names and meanings are a contract with
// that generated code.
#pragma once

#define EG_HOST_EMU 1
#define CUDAHOSTEMU 1

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <math.h>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(...)
#define __cluster_dims__(...)
#define __grid_constant__
#ifndef __align__
#define __align__(n) alignas(n)
#endif

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

// --- types ----------------------------------------------------------------

struct dim3 {
    unsigned x = 1, y = 1, z = 1;
    dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};

struct alignas(8) float2 { float x, y; };
struct alignas(16) float4 { float x, y, z, w; };
struct alignas(8) uint2 { unsigned x, y; };
struct alignas(16) uint4 { unsigned x, y, z, w; };
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline uint2 make_uint2(unsigned x, unsigned y) { return {x, y}; }
inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return {x, y, z, w}; }

struct __nv_bfloat16 { uint16_t b; };
struct __nv_bfloat162 { __nv_bfloat16 x, y; };

inline thread_local dim3 threadIdx, blockIdx;
inline dim3 blockDim, gridDim;
constexpr int warpSize = 32;

// --- blocks, warps and clusters --------------------------------------------

inline void eg_emu_wake_delayers();

struct eg_emu_warp {
    std::mutex mu;
    std::condition_variable cv;
    uint32_t active_mask;
    uint32_t arrived_mask = 0;
    uint32_t target_mask[32] = {};
    uint32_t released_eff[32] = {};
    unsigned lane_gen[32] = {};
    uint64_t val[32] = {};
    const void* ptr[32] = {};
    uint32_t regs[32][8] = {};
    std::atomic<int>* stalled = nullptr;  // the block's count of lanes blocked here

    explicit eg_emu_warp(int n)
        : active_mask(n >= 32 ? 0xffffffffu : ((1u << n) - 1u)) {}

    uint32_t sync(uint32_t mask, int lane) {
        std::unique_lock<std::mutex> lock(mu);
        const uint32_t bit = 1u << lane;
        if (!(mask & bit)) {
            std::fprintf(stderr, "cudahostemu: lane %d not in warp sync mask %#x\n", lane, mask);
            std::fflush(stderr);
            std::_Exit(3);
        }
        uint32_t eff = mask & active_mask;
        for (uint32_t m = eff & arrived_mask; m; m &= m - 1) {
            int k = __builtin_ctz(m);
            if ((target_mask[k] & active_mask) != eff) {
                std::fprintf(stderr, "cudahostemu: conflicting warp sync masks %#x (lane %d) and %#x (lane %d)\n",
                             mask, lane, target_mask[k], k);
                std::fflush(stderr);
                std::_Exit(3);
            }
        }
        arrived_mask |= bit;
        target_mask[lane] = mask;
        unsigned g = lane_gen[lane];
        if ((arrived_mask & eff) == eff) {
            arrived_mask &= ~eff;
            for (uint32_t m = eff; m; m &= m - 1) {
                int k = __builtin_ctz(m);
                target_mask[k] = 0;
                released_eff[k] = eff;
                ++lane_gen[k];
            }
            if (stalled) stalled->fetch_sub(__builtin_popcount(eff) - 1);
            cv.notify_all();
            return eff;
        }
        if (stalled) {
            stalled->fetch_add(1);
            eg_emu_wake_delayers();
        }
        cv.wait(lock, [&] { return lane_gen[lane] != g; });
        return released_eff[lane];
    }

    void drop(int lane) {
        std::lock_guard<std::mutex> lock(mu);
        active_mask &= ~(1u << lane);
        bool woke = false;
        for (int k = 0; k < 32; ++k) {
            if (!(arrived_mask & (1u << k))) continue;
            uint32_t eff = target_mask[k] & active_mask;
            if (eff && (arrived_mask & eff) == eff) {
                arrived_mask &= ~eff;
                for (uint32_t m = eff; m; m &= m - 1) {
                    int j = __builtin_ctz(m);
                    target_mask[j] = 0;
                    released_eff[j] = eff;
                    ++lane_gen[j];
                }
                if (stalled) stalled->fetch_sub(__builtin_popcount(eff));
                woke = true;
            }
        }
        if (woke) cv.notify_all();
    }
};

struct eg_emu_named_bar {
    std::condition_variable cv;
    int arrived = 0;
    int expected = 0;
    unsigned gen = 0;
};

// An mbarrier (eg_emu_mbar_*). phase is the current, incomplete phase; it
// completes when pending arrivals and tx transaction bytes both reach zero.
struct eg_emu_mbar_waiter;
struct eg_emu_mbar {
    int count = 0;
    int pending = 0;
    int64_t tx = 0;
    int64_t phase = 0;
    std::vector<std::pair<unsigned char*, uint64_t>> copies;  // undelivered bulk copies: dst, seq
    std::vector<eg_emu_mbar_waiter*> waiters;
};

// A thread blocked in eg_emu_mbar_wait until phase target completes.
struct eg_emu_mbar_waiter {
    int64_t target;
    uint64_t gen = 0;  // the block's delay generation when target completed
    bool done = false;
};

// A cp.async.bulk copy whose data has not been delivered to its destination.
struct eg_emu_bulk {
    uint64_t* bar;
    int64_t phase;  // the phase of bar whose completion the copy signals
    uint64_t seq;
    std::vector<unsigned char> data;
};

struct eg_emu_block {
    int nthreads;
    std::mutex bar_mu;
    int live_threads;
    // Threads blocked in each kind of wait, for deadlock detection and for
    // the mbarrier wait's late release. warp_stalled is exact: the lane that
    // releases a warp sync subtracts the lanes it releases.
    int waiting_named = 0, waiting_sync = 0, waiting_mbar = 0;
    std::atomic<int> delaying{0}, warp_stalled{0}, cluster_stalled{0};
    eg_emu_named_bar named[16];
    int sync_arrived = 0;
    unsigned sync_gen = 0;
    std::condition_variable sync_cv;
    std::condition_variable mbar_cv;
    uint64_t delay_gen = 0, bulk_seq = 0;
    std::map<uint64_t*, eg_emu_mbar> mbars;
    std::map<unsigned char*, eg_emu_bulk> bulk;  // by destination
    std::vector<std::unique_ptr<eg_emu_warp>> warps;
    std::vector<unsigned char> smem;  // dynamic shared memory

    // Keyed per-block shared memory (MOE8_SHARED / EG_EMU_SHARED).
    std::mutex mu;
    std::map<std::string, std::vector<unsigned char>> keyed;

    explicit eg_emu_block(int n) : nthreads(n), live_threads(n) {}
    void* get(const std::string& key, size_t n, unsigned char fill = 0) {
        std::lock_guard<std::mutex> l(mu);
        auto& b = keyed[key];
        if (b.empty()) b.assign(n, fill);
        return b.data();
    }
};

struct eg_emu_cluster {
    std::barrier<> bar;
    std::vector<eg_emu_block*> blocks;
    explicit eg_emu_cluster(int threads) : bar(threads) {}
};

inline thread_local eg_emu_block* eg_emu_cur_block;
inline thread_local eg_emu_warp* eg_emu_cur_warp;
inline thread_local eg_emu_cluster* eg_emu_cur_cluster;
inline thread_local int eg_emu_lane;
inline thread_local int eg_emu_rank;
inline thread_local int& eg_emu_cluster_rank = eg_emu_rank;

inline void eg_emu_wake_delayers() {
    if (eg_emu_cur_block && eg_emu_cur_block->delaying.load() > 0) eg_emu_cur_block->mbar_cv.notify_all();
}

// eg_emu_quiescent reports whether no thread of b can run until a delayed
// mbarrier wait returns. b->bar_mu must be held.
inline bool eg_emu_quiescent(eg_emu_block* b) {
    return b->waiting_named + b->waiting_sync + b->waiting_mbar + b->delaying +
               b->warp_stalled.load() + b->cluster_stalled.load() >= b->live_threads;
}

// eg_emu_release_delayers returns every delayed mbarrier wait once b is
// quiescent. b->bar_mu must be held.
inline void eg_emu_release_delayers(eg_emu_block* b) {
    if (b->delaying > 0 && eg_emu_quiescent(b)) {
        ++b->delay_gen;
        b->delaying = 0;
        b->mbar_cv.notify_all();
    }
}

// eg_emu_check_deadlock fails if every live thread of b waits for another
// one. b->bar_mu must be held.
inline void eg_emu_check_deadlock(eg_emu_block* b, const char* what) {
    if (b->live_threads > 0 && b->delaying == 0 &&
        b->waiting_named + b->waiting_sync + b->waiting_mbar + b->warp_stalled.load() == b->live_threads) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) deadlock in %s: %d threads in named barriers, %d in __syncthreads, %d in mbarrier waits, %d in warp syncs\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, what, b->waiting_named, b->waiting_sync, b->waiting_mbar, b->warp_stalled.load());
        std::fflush(stderr);
        std::_Exit(3);
    }
}

// eg_emu_sync_release completes a __syncthreads that waiters threads wait in.
inline void eg_emu_sync_release(eg_emu_block* b, int waiters) {
    b->waiting_sync -= waiters;
    b->sync_arrived = 0;
    ++b->sync_gen;
    b->sync_cv.notify_all();
}

inline void __syncthreads() {
    eg_emu_block* b = eg_emu_cur_block;
    std::unique_lock<std::mutex> lock(b->bar_mu);
    if (++b->sync_arrived == b->live_threads) {
        eg_emu_sync_release(b, b->sync_arrived - 1);
        return;
    }
    ++b->waiting_sync;
    eg_emu_check_deadlock(b, "__syncthreads");
    eg_emu_release_delayers(b);
    const unsigned g = b->sync_gen;
    b->sync_cv.wait(lock, [&] { return b->sync_gen != g; });
}
inline void __syncwarp(unsigned mask = 0xffffffffu) { eg_emu_cur_warp->sync(mask, eg_emu_lane); }
inline void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void __threadfence_block() { std::atomic_thread_fence(std::memory_order_seq_cst); }

// eg_emu_bar_sync emulates PTX bar.sync id, count for named barriers 0..15.
// count must be a positive multiple of 32 not exceeding the block size.
inline void eg_emu_bar_sync(int id, int count) {
    eg_emu_block* b = eg_emu_cur_block;
    if (id < 0 || id >= 16 || count <= 0 || count % 32 != 0 || count > b->nthreads) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_emu_bar_sync(%d, %d): invalid barrier id or thread count (block has %d threads)\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, id, count, b->nthreads);
        std::fflush(stderr);
        std::_Exit(3);
    }
    std::unique_lock<std::mutex> lock(b->bar_mu);
    eg_emu_named_bar& nb = b->named[id];
    if (nb.arrived == 0) {
        nb.expected = count;
    } else if (nb.expected != count) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_emu_bar_sync(%d, %d): count mismatch (expected %d)\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, id, count, nb.expected);
        std::fflush(stderr);
        std::_Exit(3);
    }
    ++nb.arrived;
    ++b->waiting_named;
    if (nb.arrived == nb.expected) {
        b->waiting_named -= nb.expected;
        nb.arrived = 0;
        nb.expected = 0;
        ++nb.gen;
        nb.cv.notify_all();
        return;
    }
    for (int j = 0; j < 16; ++j) {
        auto& m = b->named[j];
        if (m.arrived > 0 && (b->live_threads < m.expected || b->waiting_named == b->live_threads)) {
            std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) named barrier %d incomplete (%d of %d threads arrived)\n",
                         blockIdx.x, blockIdx.y, blockIdx.z, j, m.arrived, m.expected);
            std::fflush(stderr);
            std::_Exit(3);
        }
    }
    eg_emu_check_deadlock(b, "named barrier");
    eg_emu_release_delayers(b);
    unsigned g = nb.gen;
    nb.cv.wait(lock, [&] { return nb.gen != g; });
}
inline void bar_sync(int id, int count) { eg_emu_bar_sync(id, count); }

template <class A> A* eg_emu_keyed_shared(const char* key, unsigned char fill = 0) {
    return (A*)eg_emu_cur_block->get(key, sizeof(A), fill);
}
template <class A> A* eg_emu_shared(const char* key) {
    return eg_emu_keyed_shared<A>(key, eg_emu_smem_fill);
}
#define EG_EMU_STR2(x) #x
#define EG_EMU_STR(x) EG_EMU_STR2(x)
#define CUDAHOSTEMU_SHARED(T, name, dims) T (&name) dims = *eg_emu_keyed_shared<T dims>(#name ":" EG_EMU_STR(__LINE__))
#define MOE8_SHARED(T, name, dims) CUDAHOSTEMU_SHARED(T, name, dims)
#define EG_EMU_SHARED(T, name, dims) T (&name) dims = *eg_emu_shared<T dims>(#name ":" EG_EMU_STR(__LINE__))

inline unsigned char* eg_emu_dynamic_smem() { return eg_emu_cur_block->smem.data(); }

inline void eg_emu_cluster_sync() {
    if (!eg_emu_cur_cluster) {
        std::fprintf(stderr, "cudahostemu: cluster_sync outside cluster\n");
        std::fflush(stderr);
        std::_Exit(3);
    }
    eg_emu_cur_block->cluster_stalled.fetch_add(1);
    eg_emu_wake_delayers();
    eg_emu_cur_cluster->bar.arrive_and_wait();
    eg_emu_cur_block->cluster_stalled.fetch_sub(1);
}

namespace cooperative_groups {
struct cluster_group {
    void sync() const { eg_emu_cluster_sync(); }
    unsigned block_rank() const { return (unsigned)eg_emu_rank; }
    unsigned num_blocks() const { return (unsigned)eg_emu_cur_cluster->blocks.size(); }
    // map_shared_rank returns the address of the same shared object, at the
    // same offset, in block r of the cluster. p must point into keyed or
    // dynamic shared memory; static __shared__ storage is not per block.
    template <class T> T* map_shared_rank(T* p, int r) const {
        if (!eg_emu_cur_cluster || r < 0 || r >= (int)eg_emu_cur_cluster->blocks.size()) {
            std::fprintf(stderr, "cudahostemu: map_shared_rank(%d) outside cluster\n", r);
            std::fflush(stderr);
            std::_Exit(3);
        }
        eg_emu_block* own = eg_emu_cur_block;
        eg_emu_block* other = eg_emu_cur_cluster->blocks.at(r);
        const unsigned char* q = (const unsigned char*)p;
        if (q >= own->smem.data() && q < own->smem.data() + own->smem.size())
            return (T*)(other->smem.data() + (q - own->smem.data()));
        std::string key;
        size_t size = 0, off = 0;
        {
            std::lock_guard<std::mutex> l(own->mu);
            for (auto& [k, buf] : own->keyed)
                if (q >= buf.data() && q < buf.data() + buf.size()) key = k, size = buf.size(), off = q - buf.data();
        }
        // Lock one block at a time: two blocks may map each other at once.
        if (size) return (T*)((unsigned char*)other->get(key, size, eg_emu_smem_fill) + off);
        std::fprintf(stderr, "cudahostemu: map_shared_rank of memory that is not per-block shared memory\n");
        std::exit(2);
    }
};
inline cluster_group this_cluster() { return {}; }
struct thread_block {
    void sync() const { __syncthreads(); }
};
inline thread_block this_thread_block() { return {}; }
}  // namespace cooperative_groups

template <typename T>
T* eg_emu_map_shared_rank(T* p, int r) {
    return cooperative_groups::this_cluster().map_shared_rank(p, r);
}

// --- arithmetic ------------------------------------------------------------

inline float __uint_as_float(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline uint32_t __float_as_uint(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float __int_as_float(int i) { float f; std::memcpy(&f, &i, 4); return f; }
inline int __float_as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
inline float __expf(float x) { return std::exp(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float __frcp_rn(float x) { return 1.0f / x; }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline void sincosf(float x, float* s, float* c) { *s = std::sin(x); *c = std::cos(x); }
// Separate functions, so the compiler cannot contract them into an FMA.
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
// expf, sqrtf and the rest come from <math.h> in the global namespace;
// libstdc++ does not declare them in std.
using std::min; using std::max;

template <typename T> T __ldg(const T* p) { return *p; }
template <typename T> T __ldcg(const T* p) { return *p; }
template <typename T> T __ldca(const T* p) { return *p; }
template <typename T> T __ldcs(const T* p) { return *p; }
template <class T> inline void __stwb(T* p, T v) { *p = v; }
template <class T> inline void __stcg(T* p, T v) { *p = v; }
template <class T> inline void __stcs(T* p, T v) { *p = v; }
template <class T> inline void __stwt(T* p, T v) { *p = v; }
inline int __popc(unsigned v) { return __builtin_popcount(v); }

inline std::mutex eg_emu_atomic_mu;
template <typename T> T atomicAdd(T* p, T v) {
    std::lock_guard<std::mutex> lock(eg_emu_atomic_mu);
    if constexpr (std::is_integral_v<T>) {
        return std::atomic_ref<T>(*p).fetch_add(v, std::memory_order_acq_rel);
    } else {
        T old = *p;
        *p = old + v;
        return old;
    }
}
inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) {
    std::lock_guard<std::mutex> lock(eg_emu_atomic_mu);
    return std::atomic_ref<unsigned long long>(*p).fetch_add(v, std::memory_order_acq_rel);
}
template <typename T> T atomicMax(T* p, T v) {
    std::lock_guard<std::mutex> lock(eg_emu_atomic_mu);
    if constexpr (std::is_integral_v<T>) {
        std::atomic_ref<T> a(*p);
        T old = a.load(std::memory_order_acquire);
        while (old < v && !a.compare_exchange_weak(old, v, std::memory_order_acq_rel, std::memory_order_acquire)) {}
        return old;
    } else {
        T old = *p;
        *p = old < v ? v : old;
        return old;
    }
}
inline unsigned long long atomicMax(unsigned long long* p, unsigned long long v) {
    std::lock_guard<std::mutex> lock(eg_emu_atomic_mu);
    std::atomic_ref<unsigned long long> a(*p);
    unsigned long long old = a.load(std::memory_order_acquire);
    while (old < v && !a.compare_exchange_weak(old, v, std::memory_order_acq_rel, std::memory_order_acquire)) {}
    return old;
}
template <typename T> T atomicMin(T* p, T v) {
    std::lock_guard<std::mutex> lock(eg_emu_atomic_mu);
    T old = *p;
    *p = v < old ? v : old;
    return old;
}
template <typename T> T atomicExch(T* p, T v) {
    std::lock_guard<std::mutex> lock(eg_emu_atomic_mu);
    T old = *p;
    *p = v;
    return old;
}
template <typename T> T atomicCAS(T* p, T cmp, T v) {
    std::lock_guard<std::mutex> lock(eg_emu_atomic_mu);
    T old = *p;
    if (old == cmp) *p = v;
    return old;
}
inline void __nanosleep(unsigned) { std::this_thread::yield(); }

// --- epoch-tagged ready counters -------------------------------------------

struct eg_emu_ready_buf {
    unsigned char* dst;
    size_t size;
    std::vector<unsigned char> data;
};
struct eg_emu_ready_entry {
    unsigned int epoch = 0;
    unsigned int expected = 0;
    bool poisoned = false;
    bool delivered = false;
    std::vector<eg_emu_ready_buf> bufs;
};
inline std::mutex eg_emu_ready_mu;
inline std::map<const unsigned long long*, eg_emu_ready_entry> eg_emu_ready_map;

// eg_emu_ready_stage registers [ptr, ptr+bytes) as produced by expected units
// on *slot in epoch. Once *slot reaches (epoch << 32) | expected,
// eg_emu_ready_signal snapshots the registered buffers and fills them with
// 0xff (NaN) until eg_emu_ready_wait(slot, epoch, need) observes the epoch
// with need >= expected.
inline void eg_emu_ready_stage(const unsigned long long* slot, unsigned int epoch,
                               unsigned int expected, void* ptr, size_t bytes) {
    if (!ptr || bytes == 0 || expected == 0) return;
    std::lock_guard<std::mutex> lock(eg_emu_ready_mu);
    eg_emu_ready_entry& e = eg_emu_ready_map[slot];
    if (e.epoch != epoch) {
        e = eg_emu_ready_entry{};
        e.epoch = epoch;
        e.expected = expected;
    }
    for (auto& b : e.bufs) {
        if (b.dst == (unsigned char*)ptr && b.size == bytes) return;
    }
    e.bufs.push_back({(unsigned char*)ptr, bytes, {}});
}

inline void eg_emu_ready_signal(unsigned long long* slot, unsigned int epoch, unsigned int units = 1) {
    if (!slot || epoch == 0) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_ready_signal(%p, epoch=%u, units=%u): bad slot or epoch 0\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, (void*)slot, epoch, units);
        std::fflush(stderr);
        std::_Exit(3);
    }
    std::atomic_thread_fence(std::memory_order_release);
    atomicMax(slot, (unsigned long long)epoch << 32);
    const unsigned long long prev = atomicAdd(slot, (unsigned long long)units);
    const unsigned long long cur = std::max(prev, (unsigned long long)epoch << 32) + units;
    std::lock_guard<std::mutex> lock(eg_emu_ready_mu);
    auto it = eg_emu_ready_map.find(slot);
    if (it != eg_emu_ready_map.end() && it->second.epoch == epoch && !it->second.poisoned) {
        const unsigned long long want = ((unsigned long long)epoch << 32) | it->second.expected;
        if (cur >= want) {
            it->second.poisoned = true;
            for (auto& b : it->second.bufs) {
                b.data.assign(b.dst, b.dst + b.size);
                std::memset(b.dst, 0xff, b.size);
            }
        }
    }
}

inline void eg_emu_ready_wait(const unsigned long long* slot, unsigned int epoch, unsigned int need) {
    if (!slot || epoch == 0) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_ready_wait(%p, epoch=%u, need=%u): bad slot or epoch 0\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, (const void*)slot, epoch, need);
        std::fflush(stderr);
        std::_Exit(3);
    }
    if (need == 0) return;
    const unsigned long long want = ((unsigned long long)epoch << 32) | need;
    std::atomic_ref<unsigned long long> a(*const_cast<unsigned long long*>(slot));
    const auto t0 = std::chrono::steady_clock::now();
    while (a.load(std::memory_order_acquire) < want) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(500)) {
            const unsigned long long got = a.load(std::memory_order_acquire);
            std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_ready_wait(%p, epoch=%u, need=%u): counter %#llx (epoch=%u, count=%u) did not reach %#llx\n",
                         blockIdx.x, blockIdx.y, blockIdx.z, (const void*)slot, epoch, need,
                         got, (unsigned)(got >> 32), (unsigned)got, want);
            std::fflush(stderr);
            std::_Exit(3);
        }
        std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    std::lock_guard<std::mutex> lock(eg_emu_ready_mu);
    auto it = eg_emu_ready_map.find(slot);
    if (it != eg_emu_ready_map.end() && it->second.epoch == epoch && it->second.poisoned && !it->second.delivered) {
        if (need < it->second.expected) {
            std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_ready_wait(%p, epoch=%u, need=%u): need < expected %u\n",
                         blockIdx.x, blockIdx.y, blockIdx.z, (const void*)slot, epoch, need, it->second.expected);
            std::fflush(stderr);
            std::_Exit(3);
        }
        it->second.delivered = true;
        for (auto& b : it->second.bufs) {
            std::memcpy(b.dst, b.data.data(), b.size);
        }
    }
}

// --- warp collectives ------------------------------------------------------

template <typename T> T eg_emu_exchange(unsigned mask, T v, int src) {
    static_assert(sizeof(T) <= 8, "shuffle operand too large");
    eg_emu_warp* w = eg_emu_cur_warp;
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    w->val[eg_emu_lane] = bits;
    w->sync(mask, eg_emu_lane);
    uint64_t got = w->val[src & 31];
    w->sync(mask, eg_emu_lane);
    T r;
    std::memcpy(&r, &got, sizeof(T));
    return r;
}
template <typename T> T __shfl_xor_sync(unsigned mask, T v, int m, int = 32) {
    return eg_emu_exchange(mask, v, eg_emu_lane ^ m);
}
template <typename T> T __shfl_sync(unsigned mask, T v, int src, int width = 32) {
    return eg_emu_exchange(mask, v, (eg_emu_lane & ~(width - 1)) | (src & (width - 1)));
}
template <typename T> T __shfl_down_sync(unsigned mask, T v, int d, int width = 32) {
    int sub = eg_emu_lane & (width - 1);
    int src = sub + d < width ? eg_emu_lane + d : eg_emu_lane;
    return eg_emu_exchange(mask, v, src);
}
template <typename T> T __shfl_up_sync(unsigned mask, T v, int d, int width = 32) {
    int sub = eg_emu_lane & (width - 1);
    int src = sub - d >= 0 ? eg_emu_lane - d : eg_emu_lane;
    return eg_emu_exchange(mask, v, src);
}

inline unsigned __ballot_sync(unsigned mask, int pred) {
    eg_emu_warp* w = eg_emu_cur_warp;
    w->val[eg_emu_lane] = pred != 0;
    uint32_t eff = w->sync(mask, eg_emu_lane);
    unsigned r = 0;
    for (uint32_t m = eff; m; m &= m - 1) {
        int i = __builtin_ctz(m);
        r |= (unsigned)(w->val[i] & 1) << i;
    }
    w->sync(mask, eg_emu_lane);
    return r;
}
inline int __any_sync(unsigned mask, int pred) {
    return __ballot_sync(mask, pred) != 0;
}
inline int __all_sync(unsigned mask, int pred) {
    eg_emu_warp* w = eg_emu_cur_warp;
    w->val[eg_emu_lane] = pred != 0;
    uint32_t eff = w->sync(mask, eg_emu_lane);
    int r = 1;
    for (uint32_t m = eff; m; m &= m - 1) {
        int i = __builtin_ctz(m);
        if (!(w->val[i] & 1)) r = 0;
    }
    w->sync(mask, eg_emu_lane);
    return r;
}

// --- cp.async --------------------------------------------------------------

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

// --- mbarrier and cp.async.bulk --------------------------------------------

// Hopper mbarriers and bulk copies are modeled adversarially, in the spirit
// of cp.async above. A phase completes when its pending arrivals and its
// transaction bytes (added by arrive.expect_tx, removed by each copy) both
// reach zero, as on the GPU. Issuing a copy fills its destination with 0xff
// and supersedes any undelivered copy it overlaps; the data lands only when
// a thread's eg_emu_mbar_wait (mbarrier.try_wait.parity in a loop) or
// eg_emu_mbar_test_wait (test_wait.parity) observes the completion of the
// copy's phase or a later one. A wait for the wrong parity, which returns at
// once for a phase that has not completed, therefore delivers nothing, and
// a consumer that reads without waiting reads NaN. A wait that is satisfied
// still returns as late as it legally can: only when every other live
// thread of the block is blocked too (or has exited). A producer that does
// not wait for a stage to be released therefore overwrites it, with 0xff,
// before its consumer reads. A thread that waits forever, and an exit with
// transaction bytes outstanding, fail loudly. With EG_EMU_SYNC_MBAR defined,
// copies land at issue and waits return when satisfied.

inline eg_emu_mbar& eg_emu_mbar_get(eg_emu_block* b, uint64_t* bar, const char* op) {
    auto it = b->mbars.find(bar);
    if (it == b->mbars.end()) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) %s: uninitialized mbarrier %p\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, op, (void*)bar);
        std::fflush(stderr);
        std::_Exit(3);
    }
    return it->second;
}

// eg_emu_mbar_advance completes m's phase if nothing is pending. b->bar_mu
// must be held.
inline void eg_emu_mbar_advance(eg_emu_block* b, eg_emu_mbar& m) {
    if (m.pending > 0 || m.tx != 0) return;
    ++m.phase;
    m.pending = m.count;
    std::vector<eg_emu_mbar_waiter*> rest;
    for (eg_emu_mbar_waiter* w : m.waiters) {
        if (w->target >= m.phase) {
            rest.push_back(w);
            continue;
        }
        w->done = true;
        w->gen = b->delay_gen;
        --b->waiting_mbar;
#if !defined(EG_EMU_SYNC_MBAR)
        ++b->delaying;
#endif
    }
    m.waiters.swap(rest);
    b->mbar_cv.notify_all();
}

// eg_emu_mbar_deliver copies the data of m's copies of phase c and earlier
// to their destinations. b->bar_mu must be held.
inline void eg_emu_mbar_deliver(eg_emu_block* b, eg_emu_mbar& m, int64_t c) {
    std::vector<std::pair<unsigned char*, uint64_t>> rest;
    for (auto& [dst, seq] : m.copies) {
        auto it = b->bulk.find(dst);
        if (it == b->bulk.end() || it->second.seq != seq) continue;  // superseded
        if (it->second.phase > c) {
            rest.push_back({dst, seq});
            continue;
        }
        std::memcpy(dst, it->second.data.data(), it->second.data.size());
        b->bulk.erase(it);
    }
    m.copies.swap(rest);
}

// eg_emu_mbar_init is mbarrier.init: count arrivals complete each phase.
inline void eg_emu_mbar_init(uint64_t* bar, int count) {
    eg_emu_block* b = eg_emu_cur_block;
    if (!bar || (uintptr_t)bar % 8 != 0 || count < 1 || count >= (1 << 20)) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_emu_mbar_init(%p, %d): bad barrier address or count\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, (void*)bar, count);
        std::fflush(stderr);
        std::_Exit(3);
    }
    std::lock_guard<std::mutex> lock(b->bar_mu);
    auto it = b->mbars.find(bar);
    if (it != b->mbars.end()) {
        if (!it->second.waiters.empty()) {
            std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_emu_mbar_init: mbarrier %p reinitialized with threads waiting on it\n",
                         blockIdx.x, blockIdx.y, blockIdx.z, (void*)bar);
            std::fflush(stderr);
            std::_Exit(3);
        }
        for (auto& [dst, seq] : it->second.copies) {
            auto c = b->bulk.find(dst);
            if (c != b->bulk.end() && c->second.seq == seq) b->bulk.erase(c);
        }
    }
    *bar = 0;
    eg_emu_mbar& m = b->mbars[bar];
    m = eg_emu_mbar{};
    m.count = count;
    m.pending = count;
}

inline void eg_emu_mbar_arrive_tx(uint64_t* bar, uint32_t tx, const char* op) {
    eg_emu_block* b = eg_emu_cur_block;
    std::lock_guard<std::mutex> lock(b->bar_mu);
    eg_emu_mbar& m = eg_emu_mbar_get(b, bar, op);
    if (m.pending == 0) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) %s: more than %d arrivals in phase %lld\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, op, m.count, (long long)m.phase);
        std::fflush(stderr);
        std::_Exit(3);
    }
    m.tx += tx;
    --m.pending;
    eg_emu_mbar_advance(b, m);
}

// eg_emu_mbar_arrive is mbarrier.arrive.
inline void eg_emu_mbar_arrive(uint64_t* bar) { eg_emu_mbar_arrive_tx(bar, 0, "eg_emu_mbar_arrive"); }

// eg_emu_mbar_arrive_expect_tx is mbarrier.arrive.expect_tx: the phase
// also waits for tx more bytes of copies.
inline void eg_emu_mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx) {
    eg_emu_mbar_arrive_tx(bar, tx, "eg_emu_mbar_arrive_expect_tx");
}

// eg_emu_mbar_wait is a loop on mbarrier.try_wait.parity: it returns once
// the phase with parity phase_parity (the current phase, if its parity is
// phase_parity, else the one before it) has completed.
inline void eg_emu_mbar_wait(uint64_t* bar, int phase_parity) {
    eg_emu_block* b = eg_emu_cur_block;
    if (phase_parity != 0 && phase_parity != 1) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_emu_mbar_wait: phase parity %d is not 0 or 1\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, phase_parity);
        std::fflush(stderr);
        std::_Exit(3);
    }
    std::unique_lock<std::mutex> lock(b->bar_mu);
    eg_emu_mbar& m = eg_emu_mbar_get(b, bar, "eg_emu_mbar_wait");
    int64_t c;  // the completed phase the wait observes
    uint64_t gen;
    if ((m.phase & 1) != phase_parity) {
        c = m.phase - 1;
        gen = b->delay_gen;
#if !defined(EG_EMU_SYNC_MBAR)
        ++b->delaying;
#endif
    } else {
        eg_emu_mbar_waiter w{m.phase};
        m.waiters.push_back(&w);
        ++b->waiting_mbar;
        eg_emu_check_deadlock(b, "mbarrier wait");
        eg_emu_release_delayers(b);
        while (!w.done) {
            b->mbar_cv.wait_for(lock, std::chrono::microseconds(200));
            if (!w.done) eg_emu_check_deadlock(b, "mbarrier wait");
        }
        c = w.target;
        gen = w.gen;
    }
#if !defined(EG_EMU_SYNC_MBAR)
    while (b->delay_gen == gen) {
        if (eg_emu_quiescent(b)) {
            ++b->delay_gen;
            b->delaying = 0;
            b->mbar_cv.notify_all();
            break;
        }
        b->mbar_cv.wait_for(lock, std::chrono::microseconds(200));
    }
#endif
    eg_emu_mbar_deliver(b, m, c);
}

// eg_emu_mbar_test_wait is mbarrier.test_wait.parity: it reports, without
// blocking or delay, whether the phase with parity phase_parity has
// completed.
inline bool eg_emu_mbar_test_wait(uint64_t* bar, int phase_parity) {
    eg_emu_block* b = eg_emu_cur_block;
    std::lock_guard<std::mutex> lock(b->bar_mu);
    eg_emu_mbar& m = eg_emu_mbar_get(b, bar, "eg_emu_mbar_test_wait");
    if ((m.phase & 1) == (phase_parity & 1)) return false;
    eg_emu_mbar_deliver(b, m, m.phase - 1);
    return true;
}

// eg_emu_bulk_issue queues data for dst against bar's current phase.
inline void eg_emu_bulk_issue(unsigned char* dst, std::vector<unsigned char> data, uint64_t* bar, const char* op) {
    eg_emu_block* b = eg_emu_cur_block;
    const size_t n = data.size();
    std::lock_guard<std::mutex> lock(b->bar_mu);
    eg_emu_mbar& m = eg_emu_mbar_get(b, bar, op);
    auto it = b->bulk.lower_bound(dst);
    if (it != b->bulk.begin()) {
        auto p = std::prev(it);
        if (p->first + p->second.data.size() > dst) it = p;
    }
    while (it != b->bulk.end() && it->first < dst + n) it = b->bulk.erase(it);
    // The bytes complete one at a time: if the phase's count runs out
    // within the copy, the phase completes there and the copy, not fully
    // landed, belongs to the next one.
    int64_t rest = (int64_t)n;
    if (m.pending == 0 && m.tx > 0 && m.tx < rest) {
        rest -= m.tx;
        m.tx = 0;
        eg_emu_mbar_advance(b, m);
    }
#if defined(EG_EMU_SYNC_MBAR)
    std::memcpy(dst, data.data(), n);
#else
    std::memset(dst, 0xff, n);
    const uint64_t seq = ++b->bulk_seq;
    b->bulk.emplace(dst, eg_emu_bulk{bar, m.phase, seq, std::move(data)});
    m.copies.push_back({dst, seq});
#endif
    m.tx -= rest;
    eg_emu_mbar_advance(b, m);
}

// eg_emu_cp_async_bulk is cp.async.bulk.shared::cluster.global.mbarrier::
// complete_tx::bytes: size bytes, a multiple of 16, between 16-byte aligned
// addresses.
inline void eg_emu_cp_async_bulk(void* smem_dst, const void* gmem_src, uint32_t size, uint64_t* bar) {
    if (size == 0 || size % 16 != 0 || (uintptr_t)smem_dst % 16 != 0 || (uintptr_t)gmem_src % 16 != 0) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_emu_cp_async_bulk(%p, %p, %u): size must be a positive multiple of 16 and addresses 16-byte aligned\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, smem_dst, gmem_src, size);
        std::fflush(stderr);
        std::_Exit(3);
    }
    const unsigned char* src = (const unsigned char*)gmem_src;
    eg_emu_bulk_issue((unsigned char*)smem_dst, std::vector<unsigned char>(src, src + size), bar, "eg_emu_cp_async_bulk");
}

// An eg_emu_tensor_map stands in for a 2D CUtensorMap made by
// cuTensorMapEncodeTiled: a row-major global tensor of dims[1] rows of
// dims[0] elements, rows stride bytes apart, read box[1] rows of box[0]
// elements at a time, with no swizzle or the 128-byte swizzle.
struct eg_emu_tensor_map {
    const unsigned char* base;
    uint32_t elem_bytes;
    uint64_t dims[2];
    uint64_t stride;
    uint32_t box[2];
    bool swizzle_128b;
};

inline eg_emu_tensor_map eg_emu_make_tensor_map_2d(const void* base, uint32_t elem_bytes, uint64_t inner, uint64_t outer,
                                                   uint64_t stride_bytes, uint32_t box_inner, uint32_t box_outer, bool swizzle_128b) {
    const uint64_t row = (uint64_t)box_inner * elem_bytes;
    if (!base || (uintptr_t)base % 16 != 0 || stride_bytes % 16 != 0 || row % 16 != 0 ||
        box_inner < 1 || box_inner > 256 || box_outer < 1 || box_outer > 256 || (swizzle_128b && row != 128)) {
        std::fprintf(stderr, "cudahostemu: eg_emu_make_tensor_map_2d: bad tensor map (box row %llu bytes, stride %llu)\n",
                     (unsigned long long)row, (unsigned long long)stride_bytes);
        std::fflush(stderr);
        std::_Exit(3);
    }
    return {(const unsigned char*)base, elem_bytes, {inner, outer}, stride_bytes, {box_inner, box_outer}, swizzle_128b};
}

// eg_emu_cp_async_bulk_tensor_2d is cp.async.bulk.tensor.2d.shared::cluster.
// global.tile.mbarrier::complete_tx::bytes: the box at element (c0, c1)
// lands at smem_dst, rows box[0] elements apart, elements outside the tensor
// zero, and with the 128-byte swizzle 16-byte chunk j of row r at chunk
// j ^ (r % 8). A swizzled box must start 1024-byte aligned in dynamic
// shared memory, so that the swizzle matches wgmma's.
inline void eg_emu_cp_async_bulk_tensor_2d(void* smem_dst, const eg_emu_tensor_map* map, int c0, int c1, uint64_t* bar) {
    // Alignment is relative to the start of dynamic shared memory, which the
    // emulation treats as 1024-byte aligned.
    unsigned char* dst = (unsigned char*)smem_dst;
    const unsigned char* smem = eg_emu_dynamic_smem();
    const bool dynamic = dst >= smem && dst < smem + eg_emu_cur_block->smem.size();
    const uintptr_t addr = dynamic ? (uintptr_t)(dst - smem) : (uintptr_t)dst;
    if (addr % 128 != 0 || (map->swizzle_128b && (!dynamic || addr % 1024 != 0))) {
        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) eg_emu_cp_async_bulk_tensor_2d: destination %p misaligned for the box\n",
                     blockIdx.x, blockIdx.y, blockIdx.z, smem_dst);
        std::fflush(stderr);
        std::_Exit(3);
    }
    const uint32_t eb = map->elem_bytes, row = map->box[0] * eb;
    std::vector<unsigned char> data((size_t)row * map->box[1], 0);
    for (uint32_t r = 0; r < map->box[1]; ++r) {
        const int64_t y = (int64_t)c1 + r;
        for (uint32_t i = 0; i < map->box[0]; ++i) {
            const int64_t x = (int64_t)c0 + i;
            if (x < 0 || y < 0 || (uint64_t)x >= map->dims[0] || (uint64_t)y >= map->dims[1]) continue;
            uint32_t o = r * row + i * eb;
            if (map->swizzle_128b) o ^= ((o >> 7) & 7u) << 4;
            std::memcpy(&data[o], map->base + (uint64_t)y * map->stride + (uint64_t)x * eb, eb);
        }
    }
    eg_emu_bulk_issue(dst, std::move(data), bar, "eg_emu_cp_async_bulk_tensor_2d");
}

// --- conversions -----------------------------------------------------------

// eg_emu_f32_to_bf16 rounds to nearest even (cvt.rn.bf16.f32); a NaN keeps
// its sign.
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

// eg_emu_f32_to_e4m3 converts to float8_e4m3fn, saturating to finite and
// rounding to nearest even.
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

inline float eg_emu_e4m3(uint32_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    float v = e == 15 && m == 7 ? NAN : e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return b & 0x80 ? -v : v;
}

// moe8ops's names. f2bf, unlike eg_emu_f32_to_bf16, returns a positive NaN.
inline float bf2f(__nv_bfloat16 v) { return __uint_as_float((uint32_t)v.b << 16); }
inline __nv_bfloat16 f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return {0x7fc0};
    u += 0x7fffu + ((u >> 16) & 1u);
    return {(uint16_t)(u >> 16)};
}
inline uint8_t e4m3(float f) { return eg_emu_f32_to_e4m3(f); }

inline float2 __bfloat1622float2(__nv_bfloat162 v) { return {bf2f(v.x), bf2f(v.y)}; }
inline __nv_bfloat162 __float22bfloat162_rn(float2 v) { return {f2bf(v.x), f2bf(v.y)}; }
inline __nv_bfloat162 __floats2bfloat162_rn(float a, float b) { return {f2bf(a), f2bf(b)}; }
inline float __bfloat162float(__nv_bfloat16 v) { return bf2f(v); }
inline __nv_bfloat16 __float2bfloat16_rn(float f) { return f2bf(f); }

enum { __NV_SATFINITE = 1, __NV_E4M3 = 1 };
inline uint16_t __nv_cvt_float2_to_fp8x2(float2 v, int, int) {
    return (uint16_t)e4m3(v.x) | ((uint16_t)e4m3(v.y) << 8);
}

// eg_emu_e4m3x2_to_f16x2 converts the low two E4M3 bytes of x to packed F16x2 (cvt.rn.f16x2.e4m3x2).
inline uint32_t eg_emu_e4m3x2_to_f16x2(uint32_t x) {
    uint16_t lo = eg_emu_f32_to_f16(eg_emu_e4m3(x & 0xffu));
    uint16_t hi = eg_emu_f32_to_f16(eg_emu_e4m3((x >> 8) & 0xffu));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
inline uint32_t cvt_e4m3x2_to_f16x2(uint32_t x) { return eg_emu_e4m3x2_to_f16x2(x); }
inline uint32_t e4m3x2_to_f16x2(uint32_t x) { return eg_emu_e4m3x2_to_f16x2(x); }

inline uint32_t __byte_perm(uint32_t a, uint32_t b, uint32_t s) {
    uint8_t bytes[8];
    for (int i = 0; i < 4; ++i) {
        bytes[i] = (a >> (8 * i)) & 0xffu;
        bytes[4 + i] = (b >> (8 * i)) & 0xffu;
    }
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) r |= (uint32_t)bytes[(s >> (4 * i)) & 7u] << (8 * i);
    return r;
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

// --- matrix instructions ---------------------------------------------------

// ldmatrix .m8n8.x4 .b16: lane l supplies the row address of matrix l/8,
// row l%8; register j of lane l receives matrix j, row l/4, columns
// 2(l%4) and 2(l%4)+1.
inline void eg_emu_ldmatrix_x4(uint32_t r[4], const void* row, bool trans) {
    eg_emu_warp* w = eg_emu_cur_warp;
    w->ptr[eg_emu_lane] = row;
    w->sync(0xffffffffu, eg_emu_lane);
    for (int j = 0; j < 4; ++j) {
        uint16_t h[2];
        for (int k = 0; k < 2; ++k) {
            int rr = eg_emu_lane / 4, cc = 2 * (eg_emu_lane % 4) + k;
            if (trans) std::swap(rr, cc);
            std::memcpy(&h[k], (const uint16_t*)w->ptr[8 * j + rr] + cc, 2);
        }
        r[j] = h[0] | ((uint32_t)h[1] << 16);
    }
    w->sync(0xffffffffu, eg_emu_lane);
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
    w->sync(0xffffffffu, eg_emu_lane);
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
    w->sync(0xffffffffu, eg_emu_lane);
    for (int i = 0; i < 4; ++i) d[i] = out[i];
}

inline void eg_emu_mma_bf16_16816(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    eg_emu_mma_16816<eg_emu_bf16>(d, a, b, c);
}
inline void eg_emu_mma_f16_16816(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    eg_emu_mma_16816<eg_emu_f16>(d, a, b, c);
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
    w->sync(0xffffffffu, eg_emu_lane);
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
    w->sync(0xffffffffu, eg_emu_lane);
    for (int i = 0; i < 4; ++i) d[i] = out[i];
}

enum eg_emu_wg_kind {
    eg_emu_wg_kind_bf16 = 0,
    eg_emu_wg_kind_f16 = 1,
    eg_emu_wg_kind_e4m3 = 2,
};

inline uint32_t eg_emu_wgmma_swizzle(uint32_t offset, uint64_t desc) {
    switch ((desc >> 62) & 3u) {
    case 1: return offset ^ (((offset >> 7) & 7u) << 4);  // 128B swizzle
    case 2: return offset ^ (((offset >> 7) & 3u) << 4);  // 64B swizzle
    case 3: return offset ^ (((offset >> 7) & 1u) << 4);  // 32B swizzle
    default: return offset;                               // no swizzle
    }
}

inline uint64_t eg_emu_wgmma_desc(size_t start, size_t lbo, size_t sbo, uint64_t swizzle = 0) {
    return (uint64_t)((start >> 4) & 0x3fff) | ((uint64_t)((lbo >> 4) & 0x3fff) << 16) |
           ((uint64_t)((sbo >> 4) & 0x3fff) << 32) | ((swizzle & 3u) << 62);
}

inline uint64_t eg_emu_wgmma_desc_128b(const void* smem_ptr, uint32_t sbo) {
    const size_t start = (const unsigned char*)smem_ptr - eg_emu_dynamic_smem();
    return eg_emu_wgmma_desc(start, 16, sbo, 1);
}

inline float eg_emu_wgmma_smem_elem(eg_emu_wg_kind kind, uint64_t desc, int r, int k, bool mn_major) {
    const unsigned char* smem = eg_emu_dynamic_smem();
    const uint32_t start = (uint32_t)(desc & 0x3fff) << 4;
    const uint32_t lbo = (uint32_t)((desc >> 16) & 0x3fff) << 4;
    const uint32_t sbo = (uint32_t)((desc >> 32) & 0x3fff) << 4;
    const uint32_t swz = (uint32_t)((desc >> 62) & 3u);
    const uint32_t w = swz ? (128u >> (swz - 1)) : 16u;
    const uint32_t eb = (kind == eg_emu_wg_kind_e4m3) ? 1u : 2u;
    uint32_t off;
    if (!mn_major) {
        const uint32_t b = (uint32_t)k * eb;
        off = start + (uint32_t)(r / 8) * sbo + (b / w) * lbo + (uint32_t)(r % 8) * w + (b % w);
    } else {
        const uint32_t b = (uint32_t)r * eb;
        off = start + (uint32_t)(k / 8) * sbo + (b / w) * lbo + (uint32_t)(k % 8) * w + (b % w);
    }
    off = eg_emu_wgmma_swizzle(off, desc);
    if (kind == eg_emu_wg_kind_e4m3) return eg_emu_e4m3(smem[off]);
    uint16_t h;
    std::memcpy(&h, smem + off, 2);
    return (kind == eg_emu_wg_kind_f16) ? eg_emu_f16(h, 0) : eg_emu_bf16(h, 0);
}

// wgmma is asynchronous on the GPU: mma_async returns at once, and the
// accumulator registers and the shared-memory operands are in use until
// wgmma.wait_group retires the op's group. The emulation models the latest
// legal timing, as it does for cp.async. A thread that calls
// eg_emu_wgmma_fence (wgmma.fence, which the PTX ISA requires before the
// first mma_async) is in async mode for the rest of the launch: each
// mma_async fills its accumulator with NaN, and the product is computed
// only when eg_emu_wgmma_wait retires its group, from shared memory as it
// is then. A read of the accumulator before the wait therefore sees NaN,
// and an operand overwritten before the wait changes the result. Ops on
// the same accumulator chain, as a k-loop does, in issue order. A thread
// that returns with ops pending is an error. Without eg_emu_wgmma_fence, or
// with EG_EMU_SYNC_WGMMA defined, mma_async completes at once.
struct eg_emu_wg_op {
    eg_emu_wg_kind kind;
    float* d;
    uint64_t da, db;
    int n, group;
    bool trans_a, trans_b;
    int scale_d;
    bool has_reg_a;
    float a_rows[2][32];
};
struct eg_emu_wg_acc {
    float* d;
    std::vector<float> value;  // the accumulator as the hardware would hold it
    int pending;
};
inline thread_local bool eg_emu_wg_async;
inline thread_local std::vector<eg_emu_wg_op> eg_emu_wg_pending;
inline thread_local std::vector<eg_emu_wg_acc> eg_emu_wg_accs;
inline thread_local int eg_emu_wg_committed;

// eg_emu_wgmma_math computes wgmma.mma_async m64nNk16 (bf16/f16) or m64nNk32
// (e4m3) for one thread's elements. Thread i of the warpgroup holds
// d[nt*4 + j] = D[(i/32%4)*16 + lane/4 + 8(j/2)][nt*8 + 2(lane%4) + (j&1)],
// n/2 values.
inline void eg_emu_wgmma_math(float* d, const eg_emu_wg_op& op) {
    const int wig = (threadIdx.x / 32) % 4, g = eg_emu_lane / 4, t = eg_emu_lane % 4;
    const int K = (op.kind == eg_emu_wg_kind_e4m3) ? 32 : 16;
    for (int nt = 0; nt < op.n / 8; ++nt)
        for (int i = 0; i < 4; ++i) {
            const int row = wig * 16 + g + 8 * (i / 2), col = nt * 8 + 2 * t + (i & 1);
            float acc = op.scale_d ? d[nt * 4 + i] : 0.0f;
            for (int k = 0; k < K; ++k) {
                float av = op.has_reg_a ? op.a_rows[i / 2][k]
                                        : eg_emu_wgmma_smem_elem(op.kind, op.da, row, k, op.trans_a);
                float bv = eg_emu_wgmma_smem_elem(op.kind, op.db, col, k, op.trans_b);
                acc += av * bv;
            }
            d[nt * 4 + i] = acc;
        }
}

inline void eg_emu_wgmma_bf16_math(float* d, uint64_t da, uint64_t db, int n) {
    eg_emu_wg_op op{eg_emu_wg_kind_bf16, d, da, db, n, -1, false, false, 1, false, {}};
    eg_emu_wgmma_math(d, op);
}

inline void eg_emu_wgmma_fence() {
#if !defined(EG_EMU_SYNC_WGMMA)
    eg_emu_wg_async = true;
#endif
}

inline void eg_emu_wgmma_issue(eg_emu_wg_op op) {
    if (!eg_emu_wg_async) {
        eg_emu_wgmma_math(op.d, op);
        return;
    }
    eg_emu_wg_acc* a = nullptr;
    for (auto& x : eg_emu_wg_accs)
        if (x.d == op.d) a = &x;
    if (!a) {
        eg_emu_wg_accs.push_back({op.d, std::vector<float>(op.d, op.d + op.n / 2), 0});
        a = &eg_emu_wg_accs.back();
    }
    ++a->pending;
    std::fill(op.d, op.d + op.n / 2, NAN);
    op.group = -1;
    eg_emu_wg_pending.push_back(op);
}

inline void eg_emu_wgmma_fill_reg_a(eg_emu_wg_op& op, const uint32_t a[4]) {
    eg_emu_warp* w = eg_emu_cur_warp;
    for (int i = 0; i < 4; ++i) w->regs[eg_emu_lane][i] = a[i];
    w->sync(0xffffffffu, eg_emu_lane);
    const int g = eg_emu_lane / 4;
    if (op.kind == eg_emu_wg_kind_e4m3) {
        for (int r = 0; r < 2; ++r)
            for (int k = 0; k < 32; ++k) {
                int lane = g * 4 + (k % 16) / 4;
                int reg = r + 2 * (k / 16);
                op.a_rows[r][k] = eg_emu_e4m3((w->regs[lane][reg] >> (8 * (k % 4))) & 0xffu);
            }
    } else {
        for (int r = 0; r < 2; ++r)
            for (int k = 0; k < 16; ++k) {
                int lane = g * 4 + (k % 8) / 2;
                int reg = r + 2 * (k / 8);
                uint32_t word = w->regs[lane][reg];
                op.a_rows[r][k] = (op.kind == eg_emu_wg_kind_f16) ? eg_emu_f16(word, k & 1) : eg_emu_bf16(word, k & 1);
            }
    }
    w->sync(0xffffffffu, eg_emu_lane);
}

inline void eg_emu_wgmma_bf16(float* d, uint64_t da, uint64_t db, int n, int trans_a = 0, int trans_b = 0, int scale_d = 1) {
    eg_emu_wgmma_issue({eg_emu_wg_kind_bf16, d, da, db, n, -1, trans_a != 0, trans_b != 0, scale_d, false, {}});
}
inline void eg_emu_wgmma_f16(float* d, uint64_t da, uint64_t db, int n, int trans_a = 0, int trans_b = 0, int scale_d = 1) {
    eg_emu_wgmma_issue({eg_emu_wg_kind_f16, d, da, db, n, -1, trans_a != 0, trans_b != 0, scale_d, false, {}});
}
inline void eg_emu_wgmma_e4m3(float* d, uint64_t da, uint64_t db, int n, int scale_d = 1) {
    eg_emu_wgmma_issue({eg_emu_wg_kind_e4m3, d, da, db, n, -1, false, false, scale_d, false, {}});
}

inline void eg_emu_wgmma_rs_bf16(float* d, const uint32_t a[4], uint64_t db, int n, int trans_b = 0, int scale_d = 1) {
    eg_emu_wg_op op{eg_emu_wg_kind_bf16, d, 0, db, n, -1, false, trans_b != 0, scale_d, true, {}};
    eg_emu_wgmma_fill_reg_a(op, a);
    eg_emu_wgmma_issue(op);
}
inline void eg_emu_wgmma_rs_f16(float* d, const uint32_t a[4], uint64_t db, int n, int trans_b = 0, int scale_d = 1) {
    eg_emu_wg_op op{eg_emu_wg_kind_f16, d, 0, db, n, -1, false, trans_b != 0, scale_d, true, {}};
    eg_emu_wgmma_fill_reg_a(op, a);
    eg_emu_wgmma_issue(op);
}
inline void eg_emu_wgmma_rs_e4m3(float* d, const uint32_t a[4], uint64_t db, int n, int scale_d = 1) {
    eg_emu_wg_op op{eg_emu_wg_kind_e4m3, d, 0, db, n, -1, false, false, scale_d, true, {}};
    eg_emu_wgmma_fill_reg_a(op, a);
    eg_emu_wgmma_issue(op);
}

template <typename Ptr, std::enable_if_t<std::is_pointer_v<Ptr>, int> = 0>
inline void eg_emu_wgmma_bf16(float* d, Ptr a, uint64_t db, int n, int trans_b = 0, int scale_d = 1) {
    eg_emu_wgmma_rs_bf16(d, a, db, n, trans_b, scale_d);
}
template <typename Ptr, std::enable_if_t<std::is_pointer_v<Ptr>, int> = 0>
inline void eg_emu_wgmma_f16(float* d, Ptr a, uint64_t db, int n, int trans_b = 0, int scale_d = 1) {
    eg_emu_wgmma_rs_f16(d, a, db, n, trans_b, scale_d);
}
template <typename Ptr, std::enable_if_t<std::is_pointer_v<Ptr>, int> = 0>
inline void eg_emu_wgmma_e4m3(float* d, Ptr a, uint64_t db, int n, int scale_d = 1) {
    eg_emu_wgmma_rs_e4m3(d, a, db, n, scale_d);
}

inline void eg_emu_wgmma_commit() {
    for (auto& op : eg_emu_wg_pending)
        if (op.group < 0) op.group = eg_emu_wg_committed;
    ++eg_emu_wg_committed;
}

// eg_emu_wgmma_wait retires every committed group except the newest n.
inline void eg_emu_wgmma_wait(int n) {
    const int keep = eg_emu_wg_committed - n;
    std::vector<eg_emu_wg_op> rest;
    for (auto& op : eg_emu_wg_pending) {
        if (op.group < 0 || op.group >= keep) {
            rest.push_back(op);
            continue;
        }
        auto a = std::find_if(eg_emu_wg_accs.begin(), eg_emu_wg_accs.end(), [&](auto& x) { return x.d == op.d; });
        eg_emu_wgmma_math(a->value.data(), op);
        if (--a->pending == 0) {
            std::copy(a->value.begin(), a->value.end(), op.d);
            eg_emu_wg_accs.erase(a);
        }
    }
    eg_emu_wg_pending.swap(rest);
}

// --- runtime ---------------------------------------------------------------

typedef struct eg_emu_stream* cudaStream_t;
typedef int cudaError_t;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorInvalidConfiguration = 9;
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };

inline thread_local cudaError_t eg_emu_last_error = cudaSuccess;
inline cudaError_t eg_emu_fail(cudaError_t e) { return eg_emu_last_error = e; }
inline cudaError_t cudaGetLastError() { cudaError_t e = eg_emu_last_error; eg_emu_last_error = cudaSuccess; return e; }
inline cudaError_t cudaPeekAtLastError() { return eg_emu_last_error; }
inline const char* cudaGetErrorString(cudaError_t e) {
    switch (e) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorInvalidConfiguration: return "invalid configuration argument";
    }
    return "unknown error";
}

// cudaMalloc returns 256-byte aligned memory filled with 0xff bytes, as
// fresh device memory holds whatever was there before.
inline cudaError_t cudaMalloc(void** p, size_t n) {
    if (!p) return eg_emu_fail(cudaErrorInvalidValue);
    *p = std::aligned_alloc(256, (n + 255) / 256 * 256 + 256);
    if (!*p) return eg_emu_fail(cudaErrorMemoryAllocation);
    std::memset(*p, 0xff, n);
    return cudaSuccess;
}
template <class T> cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (n && (!dst || !src)) return eg_emu_fail(cudaErrorInvalidValue);
    std::memmove(dst, src, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind k, cudaStream_t = nullptr) {
    return cudaMemcpy(dst, src, n, k);
}
inline cudaError_t cudaMemset(void* p, int v, size_t n) {
    if (n && !p) return eg_emu_fail(cudaErrorInvalidValue);
    std::memset(p, v, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t = nullptr) { return cudaMemset(p, v, n); }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
template <typename K> cudaError_t cudaFuncSetAttribute(K, cudaFuncAttribute, int) { return cudaSuccess; }

// The emulated device: eg_emu_sms multiprocessors of eg_emu_blocks_per_sm
// resident blocks each, which launchers that size grids from the device
// see. Set them before the first launch.
inline int eg_emu_sms = 4, eg_emu_blocks_per_sm = 2;
enum cudaDeviceAttr { cudaDevAttrMultiProcessorCount = 16 };
inline cudaError_t cudaGetDevice(int* dev) { *dev = 0; return cudaSuccess; }
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr, int) { *v = eg_emu_sms; return cudaSuccess; }
template <typename K> cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, K, int, size_t) {
    *n = eg_emu_blocks_per_sm;
    return cudaSuccess;
}

// --- launch ----------------------------------------------------------------

// eg_emu_last_grid is the grid of the latest launch, valid or not, for
// tests of launchers that size it.
inline dim3 eg_emu_last_grid;

// eg_emu_run runs body on every thread of grid × block, cluster blocks at a
// time along x. It reports an invalid configuration through
// cudaGetLastError and runs nothing.
template <typename F>
void eg_emu_run(dim3 grid, dim3 block, size_t smem, int cluster, F body) {
    eg_emu_last_grid = grid;
    const long n = (long)block.x * block.y * block.z;
    if (n <= 0 || n > 1024 || grid.x == 0 || grid.y == 0 || grid.z == 0 ||
        cluster < 1 || cluster > 16 || grid.x % cluster != 0) {
        eg_emu_fail(cudaErrorInvalidConfiguration);
        return;
    }
    gridDim = grid;
    blockDim = block;
    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned cx = 0; cx < grid.x; cx += cluster) {
        std::memset(eg_emu_shared_start, eg_emu_smem_fill, eg_emu_shared_end - eg_emu_shared_start);
        eg_emu_cluster cl((int)(n * cluster));
        std::vector<std::unique_ptr<eg_emu_block>> blocks;
        for (int r = 0; r < cluster; ++r) {
            blocks.emplace_back(new eg_emu_block((int)n));
            eg_emu_block& b = *blocks.back();
            b.smem.assign(smem + 16, eg_emu_smem_fill);
            for (int w = 0; w * 32 < n; ++w) {
                b.warps.emplace_back(new eg_emu_warp(std::min(32, (int)n - 32 * w)));
                b.warps.back()->stalled = &b.warp_stalled;
            }
            cl.blocks.push_back(&b);
        }
        std::vector<std::thread> threads;
        for (int r = 0; r < cluster; ++r)
            for (int i = 0; i < n; ++i)
                threads.emplace_back([&, r, i] {
                    eg_emu_block& b = *blocks[r];
                    threadIdx = dim3(i % block.x, (i / block.x) % block.y, i / (block.x * block.y));
                    blockIdx = dim3(cx + r, by, bz);
                    eg_emu_cur_block = &b;
                    eg_emu_cur_warp = b.warps[i / 32].get();
                    eg_emu_cur_cluster = &cl;
                    eg_emu_lane = i % 32;
                    eg_emu_rank = r;
                    eg_emu_cp_pending.clear();
                    eg_emu_cp_committed = 0;
                    eg_emu_wg_async = false;
                    eg_emu_wg_pending.clear();
                    eg_emu_wg_accs.clear();
                    eg_emu_wg_committed = 0;
                    body();
                    if (!eg_emu_wg_pending.empty()) {
                        std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) thread %d returned with %zu wgmma ops pending (no wgmma.wait_group)\n",
                                     blockIdx.x, blockIdx.y, blockIdx.z, i, eg_emu_wg_pending.size());
                        std::fflush(stderr);
                        std::_Exit(3);
                    }
                    // Drop out of the warp first, so lanes waiting for this one are not
                    // counted as stalled once it is no longer live.
                    eg_emu_cur_warp->drop(eg_emu_lane);
                    {
                        std::lock_guard<std::mutex> lock(b.bar_mu);
                        --b.live_threads;
                        for (int j = 0; j < 16; ++j) {
                            auto& m = b.named[j];
                            if (m.arrived > 0 && (b.live_threads < m.expected || b.waiting_named == b.live_threads)) {
                                std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) named barrier %d incomplete (%d of %d threads arrived before exit)\n",
                                             blockIdx.x, blockIdx.y, blockIdx.z, j, m.arrived, m.expected);
                                std::fflush(stderr);
                                std::_Exit(3);
                            }
                        }
                        if (b.sync_arrived > 0 && b.sync_arrived == b.live_threads) eg_emu_sync_release(&b, b.sync_arrived);
                        eg_emu_check_deadlock(&b, "thread exit");
                        eg_emu_release_delayers(&b);
                    }
                    cl.bar.arrive_and_drop();
                });
        for (auto& t : threads) t.join();
        for (int r = 0; r < cluster; ++r) {
            eg_emu_block& b = *blocks[r];
            for (auto& [bar, m] : b.mbars) {
                bool inflight = m.tx != 0;
                for (auto& [dst, seq] : m.copies) {
                    auto it = b.bulk.find(dst);
                    if (it != b.bulk.end() && it->second.seq == seq && it->second.phase >= m.phase) inflight = true;
                }
                if (inflight) {
                    std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) exited with mbarrier %p phase %lld incomplete: %lld transaction bytes outstanding\n",
                                 cx + r, by, bz, (void*)bar, (long long)m.phase, (long long)m.tx);
                    std::fflush(stderr);
                    std::_Exit(3);
                }
            }
        }
    }
}

// eg_emu_cluster_kernels (aliased as shim_cluster_kernels) holds kernels that
// eg_emu_launch and sgl_mock.h's LaunchKernel dispatch as thread-block clusters
// (8 CTAs by default, or the size passed to eg_emu_register_cluster_kernel).
inline std::set<const void*> eg_emu_cluster_kernels;
inline std::set<const void*>& shim_cluster_kernels = eg_emu_cluster_kernels;
inline std::map<const void*, int> eg_emu_cluster_sizes;

template <typename K>
inline bool eg_emu_register_cluster_kernel(K kernel, int cluster = 8) {
    const void* p = reinterpret_cast<const void*>(kernel);
    eg_emu_cluster_kernels.insert(p);
    eg_emu_cluster_sizes[p] = cluster;
    return true;
}

template <typename K, typename... Args>
void eg_emu_launch_cluster(dim3 grid, dim3 block, size_t smem, int cluster, K kernel, Args... args) {
    eg_emu_run(grid, block, smem, cluster, [&] { kernel(args...); });
}
template <typename K, typename... Args>
void eg_emu_launch(dim3 grid, dim3 block, size_t smem, K kernel, Args... args) {
    if constexpr (std::is_pointer_v<std::decay_t<K>>) {
        const void* p = reinterpret_cast<const void*>(kernel);
        if (eg_emu_cluster_kernels.count(p)) {
            auto it = eg_emu_cluster_sizes.find(p);
            int cluster = (it != eg_emu_cluster_sizes.end()) ? it->second : 8;
            eg_emu_launch_cluster(grid, block, smem, cluster, kernel, args...);
            return;
        }
    }
    eg_emu_run(grid, block, smem, 1, [&] { kernel(args...); });
}

// CUDAHOSTEMU_LAUNCH stands in for kernel<<<grid, block, smem, stream>>>(...).
#define CUDAHOSTEMU_LAUNCH(kernel, grid, block, smem, stream, ...) eg_emu_launch(grid, block, smem, kernel __VA_OPT__(,) __VA_ARGS__)

// moe8ops's launchers: F is a callable run by every thread.
inline void eg_emu_check_launch(const char* what) {
    if (cudaPeekAtLastError() == cudaErrorInvalidConfiguration) {
        std::fprintf(stderr, "cudahostemu: %s: invalid launch configuration\n", what);
        std::exit(2);
    }
}
template <class F> void run_grid(dim3 grid, dim3 block, F kernel) {
    eg_emu_run(grid, block, 0, 1, kernel);
    eg_emu_check_launch("run_grid");
}
template <class F> void run_cluster_grid(dim3 grid, dim3 block, int n, F kernel) {
    eg_emu_run(grid, block, 0, n, kernel);
    eg_emu_check_launch("run_cluster_grid");
}

// --- files -----------------------------------------------------------------

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

// read_file reads dir/name.bin; write_file writes dir/name.out.
inline std::vector<char> read_file(const char* dir, const char* name) {
    std::string path = std::string(dir) + "/" + name + ".bin";
    std::vector<unsigned char> b = eg_emu_read(path.c_str());
    std::vector<char> out(b.begin(), b.end() - 16);
    out.reserve(out.size() + 16);
    return out;
}
inline void write_file(const char* dir, const char* name, const std::vector<char>& b) {
    std::string path = std::string(dir) + "/" + name + ".out";
    eg_emu_write(path.c_str(), b.data(), b.size());
}
