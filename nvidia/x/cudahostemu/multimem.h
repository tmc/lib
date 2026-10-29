// multimem.h: host emulation of multi-GPU NVLink symmetric memory (nvlshm /
// multimem) collectives, peer pointers, and cross-rank barriers for cudahostemu.
#pragma once

#include "cudahostemu.h"

#ifndef EG_EMU
#define EG_EMU 1
#endif

inline thread_local int eg_emu_gpu_rank = 0;
inline int eg_emu_num_ranks_val = 1;

struct eg_emu_rank_accessor_t {
    int operator()() const noexcept { return eg_emu_gpu_rank; }
    operator int() const noexcept { return eg_emu_gpu_rank; }
    eg_emu_rank_accessor_t& operator=(int r) noexcept {
        eg_emu_gpu_rank = r;
        return *this;
    }
};

inline constexpr eg_emu_rank_accessor_t eg_emu_gpu_rank_obj{};
#define eg_emu_rank eg_emu_gpu_rank_obj

inline int eg_emu_num_ranks() noexcept { return eg_emu_num_ranks_val; }

struct eg_emu_sym_buf_entry {
    unsigned char* base = nullptr;
    size_t bytes_per_rank = 0;
    size_t total_bytes = 0;
    int num_ranks = 1;
    uint32_t synced_epoch = 0;
    bool dirty = false;
    bool remote_read_since_sync = false;
    int remote_reader_rank = -1;
    std::string remote_read_op;
    std::vector<unsigned char> synced_data;
};

struct eg_emu_nvlink_bar_key {
    const void* flag_ptr;
    unsigned bx, by, bz;
    uint32_t epoch;
    bool operator<(const eg_emu_nvlink_bar_key& o) const noexcept {
        if (flag_ptr != o.flag_ptr) return flag_ptr < o.flag_ptr;
        if (bx != o.bx) return bx < o.bx;
        if (by != o.by) return by < o.by;
        if (bz != o.bz) return bz < o.bz;
        return epoch < o.epoch;
    }
};

struct eg_emu_nvlink_bar_state {
    int arrived = 0;
    bool completed = false;
    std::condition_variable cv;
};

struct eg_emu_cta_nvbar_state {
    int arrived = 0;
    unsigned gen = 0;
    std::condition_variable cv;
};

inline std::mutex eg_emu_multimem_mu;
inline std::vector<eg_emu_sym_buf_entry> eg_emu_sym_bufs;
inline std::map<eg_emu_nvlink_bar_key, std::unique_ptr<eg_emu_nvlink_bar_state>> eg_emu_nvlink_bars;
inline std::map<eg_emu_block*, std::unique_ptr<eg_emu_cta_nvbar_state>> eg_emu_cta_nvbars;
inline std::vector<uint32_t> eg_emu_rank_epoch;
inline std::atomic<int> eg_emu_nvlink_violations{0};

inline void eg_emu_multimem_reset(int num_ranks) {
    std::lock_guard<std::mutex> lock(eg_emu_multimem_mu);
    eg_emu_num_ranks_val = num_ranks > 0 ? num_ranks : 1;
    eg_emu_sym_bufs.clear();
    eg_emu_nvlink_bars.clear();
    eg_emu_cta_nvbars.clear();
    eg_emu_rank_epoch.assign(eg_emu_num_ranks_val, 0);
    eg_emu_nvlink_violations.store(0);
}

inline void eg_emu_register_sym_buffer(void* base, size_t total_bytes, int num_ranks) {
    if (!base || total_bytes == 0) return;
    std::lock_guard<std::mutex> lock(eg_emu_multimem_mu);
    const int nr = num_ranks > 0 ? num_ranks : 1;
    const size_t bpr = (total_bytes % (size_t)nr == 0) ? (total_bytes / (size_t)nr) : total_bytes;
    unsigned char* p = static_cast<unsigned char*>(base);
    for (auto& e : eg_emu_sym_bufs) {
        if (e.base == p) {
            e.bytes_per_rank = bpr;
            e.total_bytes = total_bytes;
            e.num_ranks = nr;
            e.synced_epoch = 0;
            e.dirty = false;
            e.remote_read_since_sync = false;
            e.remote_reader_rank = -1;
            e.synced_data.assign(p, p + total_bytes);
            return;
        }
    }
    eg_emu_sym_buf_entry e;
    e.base = p;
    e.bytes_per_rank = bpr;
    e.total_bytes = total_bytes;
    e.num_ranks = nr;
    e.synced_data.assign(p, p + total_bytes);
    eg_emu_sym_bufs.push_back(std::move(e));
}

inline eg_emu_sym_buf_entry* eg_emu_find_sym_buf_locked(const void* ptr) {
    const unsigned char* p = static_cast<const unsigned char*>(ptr);
    for (auto& e : eg_emu_sym_bufs) {
        if (p >= e.base && p < e.base + e.total_bytes) {
            return &e;
        }
    }
    return nullptr;
}

inline bool eg_emu_buf_modified_locked(const eg_emu_sym_buf_entry& e) {
    if (e.dirty) return true;
    if (e.synced_data.size() == e.total_bytes &&
        std::memcmp(e.base, e.synced_data.data(), e.total_bytes) != 0) {
        return true;
    }
    return false;
}

inline bool eg_emu_ranks_epoch_diverged_locked() {
    if (eg_emu_rank_epoch.size() <= 1) return false;
    const uint32_t e0 = eg_emu_rank_epoch[0];
    for (size_t r = 1; r < eg_emu_rank_epoch.size(); ++r) {
        if (eg_emu_rank_epoch[r] != e0) return true;
    }
    return false;
}

[[noreturn]] inline void eg_emu_report_nvlink_violation_locked(const char* op, int rank, const char* detail) {
    eg_emu_nvlink_violations.fetch_add(1);
    std::fprintf(stderr, "cudahostemu: NVLink ordering violation: rank %d in %s: %s\n", rank, op, detail);
    std::fflush(stderr);
    std::_Exit(3);
}

inline void eg_emu_check_remote_access_locked(eg_emu_sym_buf_entry* e, const char* op, int my_rank) {
    if (eg_emu_ranks_epoch_diverged_locked()) {
        eg_emu_report_nvlink_violation_locked(op, my_rank,
            "remote NVLink access while ranks are at different eg_emu_nvlink_barrier epochs");
    }
    if (!e) return;
    if (eg_emu_buf_modified_locked(*e)) {
        eg_emu_report_nvlink_violation_locked(op, my_rank,
            "remote NVLink read/reduce after symmetric memory write without eg_emu_nvlink_barrier");
    }
    e->remote_read_since_sync = true;
    e->remote_reader_rank = my_rank;
    e->remote_read_op = op;
}

inline void eg_emu_verify_epoch_clean_locked(const char* where) {
    for (const auto& e : eg_emu_sym_bufs) {
        if (e.remote_read_since_sync && eg_emu_buf_modified_locked(e)) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                "symmetric buffer was both written and remotely read (%s) before %s without eg_emu_nvlink_barrier",
                e.remote_read_op.c_str(), where);
            eg_emu_report_nvlink_violation_locked(e.remote_read_op.c_str(), e.remote_reader_rank, msg);
        }
    }
}

// eg_emu_local_ptr returns this rank's slice of base when base is a multi-rank
// buffer of at least num_ranks * elems_per_rank elements, or base otherwise.
template <typename T>
inline T* eg_emu_local_ptr(T* base, size_t elems_per_rank) {
    if (!base || elems_per_rank == 0) return base;
    std::lock_guard<std::mutex> lock(eg_emu_multimem_mu);
    eg_emu_sym_buf_entry* e = eg_emu_find_sym_buf_locked(base);
    const size_t need_bytes = elems_per_rank * sizeof(T) * (size_t)eg_emu_num_ranks_val;
    if (e && e->total_bytes >= need_bytes) {
        const size_t slice_off = (size_t)(reinterpret_cast<const unsigned char*>(base) - e->base) % e->bytes_per_rank;
        return reinterpret_cast<T*>(e->base + (size_t)eg_emu_gpu_rank * e->bytes_per_rank + slice_off);
    }
    return base;
}

// eg_emu_peer_ptr maps a symmetric pointer into target rank's address space.
template <typename T>
inline T* eg_emu_peer_ptr(T* base, int rank, int my_rank, size_t bytes_per_rank) {
    if (!base) return nullptr;
    std::lock_guard<std::mutex> lock(eg_emu_multimem_mu);
    if (rank < 0 || rank >= eg_emu_num_ranks_val) {
        eg_emu_report_nvlink_violation_locked("eg_emu_peer_ptr", my_rank, "target rank out of range");
    }
    eg_emu_sym_buf_entry* e = eg_emu_find_sym_buf_locked(base);
    if (rank != my_rank) {
        eg_emu_check_remote_access_locked(e, "eg_emu_peer_ptr", my_rank);
    }
    if (e && bytes_per_rank > 0 && e->total_bytes >= (size_t)eg_emu_num_ranks_val * bytes_per_rank) {
        const size_t slice_off = (size_t)(reinterpret_cast<const unsigned char*>(base) - e->base) % bytes_per_rank;
        return reinterpret_cast<T*>(e->base + (size_t)rank * bytes_per_rank + slice_off);
    }
    using BytePtr = std::conditional_t<std::is_const_v<T>, const unsigned char*, unsigned char*>;
    return reinterpret_cast<T*>(reinterpret_cast<BytePtr>(base) +
                                ((ptrdiff_t)rank - (ptrdiff_t)my_rank) * (ptrdiff_t)bytes_per_rank);
}

// eg_emu_multimem_ld_reduce_add_f32 sums element elem_idx across all ranks'
// slices of the symmetric buffer sym_base.
inline float eg_emu_multimem_ld_reduce_add_f32(const float* sym_base, size_t elem_idx = 0) {
    const float* target = sym_base + elem_idx;
    std::lock_guard<std::mutex> lock(eg_emu_multimem_mu);
    eg_emu_sym_buf_entry* e = eg_emu_find_sym_buf_locked(target);
    eg_emu_check_remote_access_locked(e, "eg_emu_multimem_ld_reduce_add_f32", eg_emu_gpu_rank);
    if (!e) {
        return *target * (float)eg_emu_num_ranks_val;
    }
    const size_t slice_off = (size_t)(reinterpret_cast<const unsigned char*>(target) - e->base) % e->bytes_per_rank;
    float sum = 0.0f;
    for (int r = 0; r < e->num_ranks; ++r) {
        float v = 0.0f;
        std::memcpy(&v, e->base + (size_t)r * e->bytes_per_rank + slice_off, sizeof(float));
        sum += v;
    }
    return sum;
}

struct eg_emu_bf16_reduced {
    uint16_t b;
    float f;
    constexpr operator float() const noexcept { return f; }
    constexpr operator __nv_bfloat16() const noexcept { return {b}; }
};

// eg_emu_multimem_ld_reduce_add_bf16 sums BF16 element elem_idx across all
// ranks' slices of the symmetric buffer sym_base.
inline eg_emu_bf16_reduced eg_emu_multimem_ld_reduce_add_bf16(const __nv_bfloat16* sym_base, size_t elem_idx = 0) {
    const __nv_bfloat16* target = sym_base + elem_idx;
    std::lock_guard<std::mutex> lock(eg_emu_multimem_mu);
    eg_emu_sym_buf_entry* e = eg_emu_find_sym_buf_locked(target);
    eg_emu_check_remote_access_locked(e, "eg_emu_multimem_ld_reduce_add_bf16", eg_emu_gpu_rank);
    if (!e) {
        const float v = bf2f(*target) * (float)eg_emu_num_ranks_val;
        return {eg_emu_f32_to_bf16(v), v};
    }
    const size_t slice_off = (size_t)(reinterpret_cast<const unsigned char*>(target) - e->base) % e->bytes_per_rank;
    float sum = 0.0f;
    for (int r = 0; r < e->num_ranks; ++r) {
        uint16_t bits = 0;
        std::memcpy(&bits, e->base + (size_t)r * e->bytes_per_rank + slice_off, sizeof(uint16_t));
        sum += __uint_as_float((uint32_t)bits << 16);
    }
    return {eg_emu_f32_to_bf16(sum), sum};
}

inline eg_emu_bf16_reduced eg_emu_multimem_ld_reduce_add_bf16(const uint16_t* sym_base, size_t elem_idx = 0) {
    return eg_emu_multimem_ld_reduce_add_bf16(reinterpret_cast<const __nv_bfloat16*>(sym_base), elem_idx);
}

inline float2 eg_emu_multimem_ld_reduce_add_bf16x2(const void* sym_base, size_t elem_idx = 0) {
    const auto* p = reinterpret_cast<const __nv_bfloat16*>(sym_base);
    const float lo = (float)eg_emu_multimem_ld_reduce_add_bf16(p, elem_idx);
    const float hi = (float)eg_emu_multimem_ld_reduce_add_bf16(p, elem_idx + 1);
    return {lo, hi};
}

// eg_emu_multimem_st broadcasts val to element elem_idx across all ranks'
// slices of the symmetric buffer sym_base.
template <typename T>
inline void eg_emu_multimem_st(T* sym_base, size_t elem_idx, T val) {
    T* target = sym_base + elem_idx;
    std::lock_guard<std::mutex> lock(eg_emu_multimem_mu);
    eg_emu_sym_buf_entry* e = eg_emu_find_sym_buf_locked(target);
    if (!e) {
        *target = val;
        return;
    }
    const size_t slice_off = (size_t)(reinterpret_cast<const unsigned char*>(target) - e->base) % e->bytes_per_rank;
    for (int r = 0; r < e->num_ranks; ++r) {
        std::memcpy(e->base + (size_t)r * e->bytes_per_rank + slice_off, &val, sizeof(T));
    }
    e->dirty = true;
}

inline void eg_emu_multimem_st_f32(float* sym_base, size_t elem_idx, float val) {
    eg_emu_multimem_st<float>(sym_base, elem_idx, val);
}

inline void eg_emu_multimem_st_bf16(__nv_bfloat16* sym_base, size_t elem_idx, __nv_bfloat16 val) {
    eg_emu_multimem_st<__nv_bfloat16>(sym_base, elem_idx, val);
}

inline void eg_emu_multimem_st_bf16(uint16_t* sym_base, size_t elem_idx, uint16_t val) {
    eg_emu_multimem_st<uint16_t>(sym_base, elem_idx, val);
}

// eg_emu_nvlink_barrier synchronizes all num_ranks GPUs at epoch on flag_ptr.
inline void eg_emu_nvlink_barrier(void* flag_ptr, int num_ranks, uint32_t epoch = 1) {
    if (epoch == 0) {
        std::lock_guard<std::mutex> lock(eg_emu_multimem_mu);
        eg_emu_report_nvlink_violation_locked("eg_emu_nvlink_barrier", eg_emu_gpu_rank, "epoch must be > 0");
    }
    eg_emu_block* b = eg_emu_cur_block;
    bool is_cta_leader = true;
    unsigned my_cta_gen = 0;
    eg_emu_cta_nvbar_state* cta_st = nullptr;

    if (b) {
        std::unique_lock<std::mutex> mlock(eg_emu_multimem_mu);
        auto& ptr = eg_emu_cta_nvbars[b];
        if (!ptr) ptr = std::make_unique<eg_emu_cta_nvbar_state>();
        cta_st = ptr.get();
        my_cta_gen = cta_st->gen;
        const int arrived = ++cta_st->arrived;
        is_cta_leader = (arrived == 1);
        if (!is_cta_leader) {
            cta_st->cv.notify_all();
            cta_st->cv.wait(mlock, [&] { return cta_st->gen != my_cta_gen; });
            std::atomic_thread_fence(std::memory_order_seq_cst);
            return;
        }
        mlock.unlock();

        // Wait until every live thread in this CTA has either entered
        // eg_emu_nvlink_barrier or parked in a CTA barrier (__syncthreads /
        // bar_sync / mbarrier), so all prior local writes in the CTA have
        // completed before the cross-rank barrier snapshots symmetric memory.
        const auto t0 = std::chrono::steady_clock::now();
        while (true) {
            int live = 1, parked = 0;
            {
                std::lock_guard<std::mutex> block_lock(b->bar_mu);
                live = b->live_threads;
                parked = b->waiting_sync + b->waiting_named + b->waiting_mbar + b->warp_stalled.load();
            }
            std::unique_lock<std::mutex> mlock2(eg_emu_multimem_mu);
            if (cta_st->arrived + parked >= live) {
                break;
            }
            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(50)) {
                break;
            }
            cta_st->cv.wait_for(mlock2, std::chrono::microseconds(100));
        }
    }

    {
        std::unique_lock<std::mutex> lock(eg_emu_multimem_mu);
        const int want_ranks = num_ranks > 0 ? num_ranks : eg_emu_num_ranks_val;
        const eg_emu_nvlink_bar_key key{flag_ptr, blockIdx.x, blockIdx.y, blockIdx.z, epoch};
        auto& st_ptr = eg_emu_nvlink_bars[key];
        if (!st_ptr) st_ptr = std::make_unique<eg_emu_nvlink_bar_state>();
        eg_emu_nvlink_bar_state* st = st_ptr.get();

        // Verify no rank read remote symmetric memory while another rank wrote
        // in the pre-barrier epoch.
        eg_emu_verify_epoch_clean_locked("eg_emu_nvlink_barrier");

        if (++st->arrived == want_ranks) {
            for (size_t r = 0; r < eg_emu_rank_epoch.size(); ++r) {
                eg_emu_rank_epoch[r] = epoch;
            }
            if (flag_ptr) {
                uint32_t v = epoch;
                std::memcpy(flag_ptr, &v, sizeof(uint32_t));
            }
            for (auto& e : eg_emu_sym_bufs) {
                e.synced_data.assign(e.base, e.base + e.total_bytes);
                e.dirty = false;
                e.remote_read_since_sync = false;
                e.remote_reader_rank = -1;
                e.remote_read_op.clear();
                e.synced_epoch = epoch;
            }
            st->completed = true;
            st->cv.notify_all();
        } else {
            const bool ok = st->cv.wait_for(lock, std::chrono::milliseconds(1500), [&] { return st->completed; });
            if (!ok) {
                char msg[256];
                std::snprintf(msg, sizeof(msg),
                    "timeout waiting at eg_emu_nvlink_barrier(epoch=%u): only %d of %d ranks arrived",
                    epoch, st->arrived, want_ranks);
                eg_emu_report_nvlink_violation_locked("eg_emu_nvlink_barrier", eg_emu_gpu_rank, msg);
            }
        }
        if (cta_st) {
            cta_st->arrived = 0;
            ++cta_st->gen;
            cta_st->cv.notify_all();
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

// eg_emu_multirank_run executes body across num_ranks simulated GPUs
// simultaneously, running each CTA/cluster coordinate across all ranks in
// parallel with eg_emu_rank() == r and eg_emu_num_ranks() == num_ranks.
template <typename F>
void eg_emu_multirank_run(int num_ranks, dim3 grid, dim3 block, size_t smem, int cluster, F body) {
    eg_emu_last_grid = grid;
    const long n = (long)block.x * block.y * block.z;
    if (num_ranks < 1 || n <= 0 || n > 1024 || grid.x == 0 || grid.y == 0 || grid.z == 0 ||
        cluster < 1 || cluster > 16 || grid.x % cluster != 0) {
        eg_emu_fail(cudaErrorInvalidConfiguration);
        return;
    }
    gridDim = grid;
    blockDim = block;
    eg_emu_num_ranks_val = num_ranks;
    {
        std::lock_guard<std::mutex> lock(eg_emu_multimem_mu);
        eg_emu_rank_epoch.assign(num_ranks, 0);
        eg_emu_nvlink_bars.clear();
        eg_emu_cta_nvbars.clear();
    }

    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned cx = 0; cx < grid.x; cx += cluster) {
        std::memset(eg_emu_shared_start, eg_emu_smem_fill, eg_emu_shared_end - eg_emu_shared_start);
        std::vector<std::thread> rank_threads;
        rank_threads.reserve(num_ranks);
        for (int gpu_r = 0; gpu_r < num_ranks; ++gpu_r) {
            rank_threads.emplace_back([&, gpu_r, bz, by, cx] {
                eg_emu_gpu_rank = gpu_r;
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
                threads.reserve((size_t)(cluster * n));
                for (int r = 0; r < cluster; ++r) {
                    for (int i = 0; i < n; ++i) {
                        threads.emplace_back([&, gpu_r, r, i] {
                            eg_emu_block& b = *blocks[r];
                            threadIdx = dim3(i % block.x, (i / block.x) % block.y, i / (block.x * block.y));
                            blockIdx = dim3(cx + r, by, bz);
                            eg_emu_cur_block = &b;
                            eg_emu_cur_warp = b.warps[i / 32].get();
                            eg_emu_cur_cluster = &cl;
                            eg_emu_lane = i % 32;
                            eg_emu_cluster_rank = r;
                            eg_emu_gpu_rank = gpu_r;
                            eg_emu_cp_pending.clear();
                            eg_emu_cp_committed = 0;
                            eg_emu_wg_async = false;
                            eg_emu_wg_pending.clear();
                            eg_emu_wg_accs.clear();
                            eg_emu_wg_committed = 0;
                            body();
                            if (!eg_emu_wg_pending.empty()) {
                                std::fprintf(stderr, "cudahostemu: rank %d block (%u,%u,%u) thread %d returned with %zu wgmma ops pending\n",
                                             gpu_r, blockIdx.x, blockIdx.y, blockIdx.z, i, eg_emu_wg_pending.size());
                                std::fflush(stderr);
                                std::_Exit(3);
                            }
                            eg_emu_cur_warp->drop(eg_emu_lane);
                            {
                                std::lock_guard<std::mutex> lock(b.bar_mu);
                                --b.live_threads;
                                for (int j = 0; j < 16; ++j) {
                                    auto& m = b.named[j];
                                    if (m.arrived > 0 && (b.live_threads < m.expected || b.waiting_named == b.live_threads)) {
                                        std::fprintf(stderr, "cudahostemu: rank %d block (%u,%u,%u) named barrier %d incomplete (%d of %d threads arrived before exit)\n",
                                                     gpu_r, blockIdx.x, blockIdx.y, blockIdx.z, j, m.arrived, m.expected);
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
                    }
                }
                for (auto& t : threads) t.join();
            });
        }
        for (auto& rt : rank_threads) rt.join();

        std::lock_guard<std::mutex> lock(eg_emu_multimem_mu);
        if (eg_emu_ranks_epoch_diverged_locked()) {
            eg_emu_report_nvlink_violation_locked("eg_emu_nvlink_barrier", 0,
                "ranks exited CTA with mismatched eg_emu_nvlink_barrier epochs");
        }
        eg_emu_verify_epoch_clean_locked("CTA exit");
        for (auto& e : eg_emu_sym_bufs) {
            e.remote_read_since_sync = false;
            e.remote_reader_rank = -1;
            e.remote_read_op.clear();
        }
    }
}

template <typename K, typename... Args>
void eg_emu_multirank_launch_cluster(dim3 grid, dim3 block, size_t smem, int cluster, K kernel, Args... args) {
    eg_emu_multirank_run(eg_emu_num_ranks_val, grid, block, smem, cluster, [&] { kernel(args...); });
}

template <typename K, typename... Args>
void eg_emu_multirank_launch(dim3 grid, dim3 block, size_t smem, K kernel, Args... args) {
    if constexpr (std::is_pointer_v<std::decay_t<K>>) {
        const void* p = reinterpret_cast<const void*>(kernel);
        if (eg_emu_cluster_kernels.count(p)) {
            auto it = eg_emu_cluster_sizes.find(p);
            int cluster = (it != eg_emu_cluster_sizes.end()) ? it->second : 8;
            eg_emu_multirank_launch_cluster(grid, block, smem, cluster, kernel, args...);
            return;
        }
    }
    eg_emu_multirank_run(eg_emu_num_ranks_val, grid, block, smem, 1, [&] { kernel(args...); });
}

#define eg_emu_launch eg_emu_multirank_launch
#define eg_emu_launch_cluster eg_emu_multirank_launch_cluster
#undef CUDAHOSTEMU_LAUNCH
#define CUDAHOSTEMU_LAUNCH(kernel, grid, block, smem, stream, ...) eg_emu_multirank_launch(grid, block, smem, kernel __VA_OPT__(,) __VA_ARGS__)
