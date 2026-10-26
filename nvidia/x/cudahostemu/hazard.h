// hazard.h: deterministic host-CPU concurrency, barrier, DSMEM, and WGMMA
// hazard sanitizer for cudahostemu.
//
// Included after cudahostemu.h when CheckHazards compiles kernel source with
// EG_EMU_HAZARD_CHECK enabled.
#pragma once

#ifndef EG_EMU_HAZARD_CHECK
#define EG_EMU_HAZARD_CHECK 1
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

struct eg_emu_hazard_record {
    std::string kind;
    int cta;
    int thread;
    int peer_tid;
    int offset;
    std::string detail;
};

inline std::mutex eg_emu_hazard_mu;
inline std::vector<eg_emu_hazard_record> eg_emu_hazard_list;

inline void eg_emu_record_hazard(const char* kind, int cta, int thread, int peer_tid, int offset, const std::string& detail) {
    std::lock_guard<std::mutex> lock(eg_emu_hazard_mu);
    for (const auto& h : eg_emu_hazard_list) {
        if (h.kind == kind && h.cta == cta && h.thread == thread && h.peer_tid == peer_tid && h.offset == offset) {
            return;
        }
        // Deduplicate warpgroup-level WGMMA divergence and barrier-level count mismatches per CTA/offset.
        if (h.kind == kind && h.cta == cta && h.offset == offset &&
            (std::strcmp(kind, "wgmma_warpgroup_divergence") == 0 ||
             std::strcmp(kind, "named_barrier_count_mismatch") == 0)) {
            return;
        }
    }
    eg_emu_hazard_list.push_back({kind, cta, thread, peer_tid, offset, detail});
}

inline void eg_emu_flush_hazards() {
#ifdef EG_EMU_HAZARD_FILE
    std::lock_guard<std::mutex> lock(eg_emu_hazard_mu);
    std::stable_sort(eg_emu_hazard_list.begin(), eg_emu_hazard_list.end(),
                     [](const eg_emu_hazard_record& a, const eg_emu_hazard_record& b) {
                         if (a.cta != b.cta) return a.cta < b.cta;
                         if (a.kind != b.kind) return a.kind < b.kind;
                         if (a.offset != b.offset) return a.offset < b.offset;
                         if (a.thread != b.thread) return a.thread < b.thread;
                         return a.peer_tid < b.peer_tid;
                     });
    FILE* f = std::fopen(EG_EMU_HAZARD_FILE, "wb");
    if (!f) return;
    for (const auto& h : eg_emu_hazard_list) {
        std::fprintf(f, "%s\t%d\t%d\t%d\t%d\t%s\n",
                     h.kind.c_str(), h.cta, h.thread, h.peer_tid, h.offset, h.detail.c_str());
    }
    std::fclose(f);
#endif
}

struct eg_emu_shadow_byte {
    int last_writer_tid = -1;
    uint32_t last_write_epoch = 0;
    int last_reader_tid = -1;
    uint32_t last_read_epoch = 0;
    uint32_t reader_warp_mask = 0;
    int reader_tid_by_warp[32];

    eg_emu_shadow_byte() {
        for (int i = 0; i < 32; ++i) reader_tid_by_warp[i] = -1;
    }
};

struct eg_emu_hazard_cluster;

struct eg_emu_hazard_block {
    eg_emu_block* block = nullptr;
    int cta_id = 0;
    int rank = 0;
    int nthreads = 0;
    int num_warpgroups = 0;

    std::mutex smem_mu;
    uint32_t epoch = 1;
    std::map<uintptr_t, eg_emu_shadow_byte> shadow;

    std::vector<int> wgmma_step;
    std::mutex wgmma_mu;
    std::vector<bool> wg_reported;

    int named_first_tid[16] = {};
    bool named_reported[16] = {};

    eg_emu_hazard_block(eg_emu_block* b, int cta, int r, int n)
        : block(b),
          cta_id(cta),
          rank(r),
          nthreads(n),
          num_warpgroups((n + 127) / 128),
          wgmma_step(((n + 127) / 128) * 128, 0),
          wg_reported((n + 127) / 128, false) {
        for (int i = 0; i < 16; ++i) {
            named_first_tid[i] = -1;
            named_reported[i] = false;
        }
    }

    bool owns_ptr(const unsigned char* p) const {
        if (!block) return false;
        if (!block->smem.empty() && p >= block->smem.data() && p < block->smem.data() + block->smem.size()) {
            return true;
        }
        std::lock_guard<std::mutex> l(block->mu);
        for (const auto& [k, buf] : block->keyed) {
            if (!buf.empty() && p >= buf.data() && p < buf.data() + buf.size()) {
                return true;
            }
        }
        return false;
    }

    int offset_of(const void* ptr) const {
        const unsigned char* p = static_cast<const unsigned char*>(ptr);
        if (block && !block->smem.empty()) {
            const unsigned char* base = block->smem.data();
            if (p >= base && p < base + block->smem.size()) {
                return static_cast<int>(p - base);
            }
        }
        if (block) {
            std::lock_guard<std::mutex> l(block->mu);
            for (const auto& [k, buf] : block->keyed) {
                if (!buf.empty() && p >= buf.data() && p < buf.data() + buf.size()) {
                    return static_cast<int>(p - buf.data());
                }
            }
        }
        if (p >= eg_emu_shared_start && p < eg_emu_shared_end) {
            return static_cast<int>(p - eg_emu_shared_start);
        }
        return 0;
    }

    void advance_epoch() {
        std::lock_guard<std::mutex> lock(smem_mu);
        ++epoch;
    }

    void record_wgmma_step(int tid) {
        std::lock_guard<std::mutex> lock(wgmma_mu);
        if (tid >= 0 && tid < static_cast<int>(wgmma_step.size())) {
            ++wgmma_step[tid];
        }
    }

    void check_wgmma_uniformity(const char* where) {
        std::lock_guard<std::mutex> lock(wgmma_mu);
        for (int wg = 0; wg < num_warpgroups; ++wg) {
            if (wg_reported[wg]) continue;
            const int base = wg * 128;
            const int ref_steps = wgmma_step[base];
            bool any_active = (ref_steps > 0);
            int div_tid = -1;
            int div_steps = 0;
            for (int i = 1; i < 128; ++i) {
                const int tid = base + i;
                const int s = wgmma_step[tid];
                if (s > 0) any_active = true;
                if (s != ref_steps && div_tid == -1) {
                    div_tid = tid;
                    div_steps = s;
                }
            }
            if (any_active && div_tid != -1) {
                wg_reported[wg] = true;
                std::ostringstream oss;
                oss << "warpgroup " << wg << " WGMMA divergence (C7519) at " << where
                    << ": thread " << div_tid << " executed " << div_steps
                    << " wgmma ops, thread " << base << " executed " << ref_steps;
                eg_emu_record_hazard("wgmma_warpgroup_divergence", cta_id, div_tid, base, wg, oss.str());
            }
        }
    }
};

struct eg_emu_hazard_cluster {
    int cluster_size = 1;
    std::mutex mu;
    std::condition_variable cv;
    unsigned gen = 0;
    std::vector<eg_emu_hazard_block*> blocks;
    std::vector<int> arrive_count;
    std::vector<int> wait_count;
    std::vector<int> waiting_threads;
    std::vector<bool> exited;

    struct dsmem_access {
        int src_cta;
        int src_rank;
        int src_tid;
        int peer_rank;
        int offset;
        int peer_wait_count_at_access;
    };
    std::vector<dsmem_access> accesses_since_wait;

    explicit eg_emu_hazard_cluster(int c)
        : cluster_size(c),
          arrive_count(c, 0),
          wait_count(c, 0),
          waiting_threads(c, 0),
          exited(c, false) {}
};

inline thread_local int eg_emu_tid = 0;
inline thread_local eg_emu_hazard_block* eg_emu_cur_hb = nullptr;
inline thread_local eg_emu_hazard_cluster* eg_emu_cur_hc = nullptr;

inline eg_emu_hazard_block* eg_emu_owner_hb(const void* ptr) {
    const unsigned char* p = static_cast<const unsigned char*>(ptr);
    if (eg_emu_cur_hc) {
        for (eg_emu_hazard_block* hb : eg_emu_cur_hc->blocks) {
            if (hb && hb->owns_ptr(p)) return hb;
        }
    }
    return eg_emu_cur_hb;
}

// --- 1. Shared memory race tracking ----------------------------------------

inline void eg_emu_smem_write(const void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return;
    eg_emu_hazard_block* hb = eg_emu_owner_hb(ptr);
    if (!hb) return;
    std::lock_guard<std::mutex> lock(hb->smem_mu);
    const uint32_t ep = hb->epoch;
    const int tid = eg_emu_tid;
    const int warp = tid / 32;
    const int base_off = hb->offset_of(ptr);
    bool reported = false;
    for (size_t i = 0; i < bytes; ++i) {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr) + i;
        eg_emu_shadow_byte& sb = hb->shadow[addr];
        if (!reported && sb.last_writer_tid != -1 && sb.last_write_epoch == ep && (sb.last_writer_tid / 32) != warp) {
            std::ostringstream oss;
            oss << "write-after-write shared memory race at byte offset " << (base_off + static_cast<int>(i))
                << " between thread " << tid << " (warp " << warp << ") and thread "
                << sb.last_writer_tid << " (warp " << (sb.last_writer_tid / 32) << ") in epoch " << ep;
            eg_emu_record_hazard("shared_memory_race", hb->cta_id, tid, sb.last_writer_tid, base_off + static_cast<int>(i), oss.str());
            reported = true;
        }
        if (!reported && sb.last_reader_tid != -1 && sb.last_read_epoch == ep) {
            const uint32_t other_warps = sb.reader_warp_mask & ~(1u << (warp & 31));
            if (other_warps != 0) {
                const int ow = __builtin_ctz(other_warps);
                const int peer = sb.reader_tid_by_warp[ow];
                std::ostringstream oss;
                oss << "write-after-read shared memory race at byte offset " << (base_off + static_cast<int>(i))
                    << " between writer thread " << tid << " (warp " << warp << ") and reader thread "
                    << peer << " (warp " << ow << ") in epoch " << ep;
                eg_emu_record_hazard("shared_memory_race", hb->cta_id, tid, peer, base_off + static_cast<int>(i), oss.str());
                reported = true;
            }
        }
        sb.last_writer_tid = tid;
        sb.last_write_epoch = ep;
    }
}

inline void eg_emu_smem_read(const void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return;
    eg_emu_hazard_block* hb = eg_emu_owner_hb(ptr);
    if (!hb) return;
    std::lock_guard<std::mutex> lock(hb->smem_mu);
    const uint32_t ep = hb->epoch;
    const int tid = eg_emu_tid;
    const int warp = tid / 32;
    const int base_off = hb->offset_of(ptr);
    bool reported = false;
    for (size_t i = 0; i < bytes; ++i) {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr) + i;
        eg_emu_shadow_byte& sb = hb->shadow[addr];
        if (!reported && sb.last_writer_tid != -1 && sb.last_write_epoch == ep && (sb.last_writer_tid / 32) != warp) {
            std::ostringstream oss;
            oss << "read-after-write shared memory race at byte offset " << (base_off + static_cast<int>(i))
                << " between reader thread " << tid << " (warp " << warp << ") and writer thread "
                << sb.last_writer_tid << " (warp " << (sb.last_writer_tid / 32) << ") in epoch " << ep;
            eg_emu_record_hazard("shared_memory_race", hb->cta_id, tid, sb.last_writer_tid, base_off + static_cast<int>(i), oss.str());
            reported = true;
        }
        if (sb.last_read_epoch != ep) {
            sb.reader_warp_mask = 0;
            sb.last_read_epoch = ep;
        }
        sb.last_reader_tid = tid;
        sb.reader_warp_mask |= (1u << (warp & 31));
        sb.reader_tid_by_warp[warp & 31] = tid;
    }
}

template <typename T>
inline void eg_emu_smem_write(const T* ptr) {
    eg_emu_smem_write(static_cast<const void*>(ptr), sizeof(T));
}

template <typename T>
inline void eg_emu_smem_read(const T* ptr) {
    eg_emu_smem_read(static_cast<const void*>(ptr), sizeof(T));
}

// Smart shared-memory element reference and pointer wrappers so direct
// smem[idx] reads and writes are automatically tracked.
template <typename T>
struct eg_emu_smem_ref {
    T* ptr_;
    explicit eg_emu_smem_ref(T* p) : ptr_(p) {}

    operator T() const {
        eg_emu_smem_read(ptr_, sizeof(T));
        return *ptr_;
    }
    eg_emu_smem_ref& operator=(const T& v) {
        eg_emu_smem_write(ptr_, sizeof(T));
        *ptr_ = v;
        return *this;
    }
    eg_emu_smem_ref& operator=(const eg_emu_smem_ref& o) {
        T v = static_cast<T>(o);
        return *this = v;
    }
    eg_emu_smem_ref& operator+=(const T& v) {
        eg_emu_smem_read(ptr_, sizeof(T));
        eg_emu_smem_write(ptr_, sizeof(T));
        *ptr_ += v;
        return *this;
    }
    eg_emu_smem_ref& operator-=(const T& v) {
        eg_emu_smem_read(ptr_, sizeof(T));
        eg_emu_smem_write(ptr_, sizeof(T));
        *ptr_ -= v;
        return *this;
    }
    eg_emu_smem_ref& operator*=(const T& v) {
        eg_emu_smem_read(ptr_, sizeof(T));
        eg_emu_smem_write(ptr_, sizeof(T));
        *ptr_ *= v;
        return *this;
    }
    eg_emu_smem_ref& operator/=(const T& v) {
        eg_emu_smem_read(ptr_, sizeof(T));
        eg_emu_smem_write(ptr_, sizeof(T));
        *ptr_ /= v;
        return *this;
    }
    eg_emu_smem_ref& operator|=(const T& v) {
        eg_emu_smem_read(ptr_, sizeof(T));
        eg_emu_smem_write(ptr_, sizeof(T));
        *ptr_ |= v;
        return *this;
    }
    eg_emu_smem_ref& operator&=(const T& v) {
        eg_emu_smem_read(ptr_, sizeof(T));
        eg_emu_smem_write(ptr_, sizeof(T));
        *ptr_ &= v;
        return *this;
    }
    eg_emu_smem_ref& operator^=(const T& v) {
        eg_emu_smem_read(ptr_, sizeof(T));
        eg_emu_smem_write(ptr_, sizeof(T));
        *ptr_ ^= v;
        return *this;
    }
    T* operator&() const { return ptr_; }
};

template <typename T, size_t N>
struct eg_emu_smem_ref<T[N]> {
    using Sub = T[N];
    Sub* ptr_;
    explicit eg_emu_smem_ref(Sub* p) : ptr_(p) {}
    eg_emu_smem_ref<T> operator[](ptrdiff_t i) const {
        return eg_emu_smem_ref<T>(&((*ptr_)[i]));
    }
    operator T*() const { return *ptr_; }
    Sub* operator&() const { return ptr_; }
};

template <typename T>
struct eg_emu_smem_ptr {
    T* ptr_;
    eg_emu_smem_ptr() : ptr_(nullptr) {}
    eg_emu_smem_ptr(T* p) : ptr_(p) {}
    explicit eg_emu_smem_ptr(void* p) : ptr_( static_cast<T*>(p)) {}

    T* get() const { return ptr_; }
    operator T*() const { return ptr_; }
    operator const T*() const { return ptr_; }
    operator void*() const { return static_cast<void*>(ptr_); }
    operator const void*() const { return static_cast<const void*>(ptr_); }

    eg_emu_smem_ref<T> operator[](ptrdiff_t i) const { return eg_emu_smem_ref<T>(ptr_ + i); }
    eg_emu_smem_ref<T> operator*() const { return eg_emu_smem_ref<T>(ptr_); }
    eg_emu_smem_ptr<T> operator+(ptrdiff_t i) const { return eg_emu_smem_ptr<T>(ptr_ + i); }
    eg_emu_smem_ptr<T> operator-(ptrdiff_t i) const { return eg_emu_smem_ptr<T>(ptr_ - i); }
    eg_emu_smem_ptr<T>& operator+=(ptrdiff_t i) { ptr_ += i; return *this; }
    eg_emu_smem_ptr<T>& operator-=(ptrdiff_t i) { ptr_ -= i; return *this; }
};

template <typename T>
inline void eg_emu_smem_write(eg_emu_smem_ptr<T> ptr, size_t bytes = sizeof(T)) {
    eg_emu_smem_write(static_cast<const void*>(ptr.get()), bytes);
}

template <typename T>
inline void eg_emu_smem_read(eg_emu_smem_ptr<T> ptr, size_t bytes = sizeof(T)) {
    eg_emu_smem_read(static_cast<const void*>(ptr.get()), bytes);
}

// --- 2. Barriers and Named Barrier Count Checking --------------------------

inline void eg_emu_release_incomplete_named_barriers(eg_emu_block* b, eg_emu_hazard_block* hb, const char* context) {
    for (int j = 0; j < 16; ++j) {
        auto& m = b->named[j];
        if (m.arrived > 0 && m.arrived < m.expected) {
            if (!hb->named_reported[j]) {
                hb->named_reported[j] = true;
                std::ostringstream oss;
                oss << "named barrier " << j << " incomplete at " << context << " ("
                    << m.arrived << " of " << m.expected << " threads arrived)";
                eg_emu_record_hazard("named_barrier_count_mismatch", hb->cta_id,
                                     hb->named_first_tid[j] >= 0 ? hb->named_first_tid[j] : eg_emu_tid,
                                     eg_emu_tid, j, oss.str());
            }
            b->waiting_named -= m.arrived;
            m.arrived = 0;
            m.expected = 0;
            ++m.gen;
            m.cv.notify_all();
        }
    }
}

inline void eg_emu_hazard_syncthreads() {
    eg_emu_block* b = eg_emu_cur_block;
    eg_emu_hazard_block* hb = eg_emu_cur_hb;
    std::unique_lock<std::mutex> lock(b->bar_mu);
    ++b->sync_arrived;
    if (b->sync_arrived + b->waiting_named >= b->live_threads && b->waiting_named > 0) {
        eg_emu_release_incomplete_named_barriers(b, hb, "__syncthreads");
    }
    if (b->sync_arrived >= b->live_threads) {
        if (hb) {
            hb->check_wgmma_uniformity("__syncthreads");
            hb->advance_epoch();
        }
        eg_emu_sync_release(b, b->sync_arrived - 1);
        return;
    }
    ++b->waiting_sync;
    eg_emu_check_deadlock(b, "__syncthreads");
    eg_emu_release_delayers(b);
    const unsigned g = b->sync_gen;
    b->sync_cv.wait(lock, [&] { return b->sync_gen != g; });
}

inline void eg_emu_hazard_bar_sync(int id, int count) {
    eg_emu_block* b = eg_emu_cur_block;
    eg_emu_hazard_block* hb = eg_emu_cur_hb;
    if (id < 0 || id >= 16 || count <= 0 || count % 32 != 0 || count > b->nthreads) {
        if (hb) {
            std::ostringstream oss;
            oss << "eg_emu_bar_sync(" << id << ", " << count
                << "): invalid barrier id or non-multiple-of-32 thread count (block has " << b->nthreads << " threads)";
            eg_emu_record_hazard("named_barrier_count_mismatch", hb->cta_id, eg_emu_tid, -1, id, oss.str());
        }
        return;
    }
    std::unique_lock<std::mutex> lock(b->bar_mu);
    if (hb && hb->named_reported[id]) {
        return;
    }
    eg_emu_named_bar& nb = b->named[id];
    if (nb.arrived == 0) {
        nb.expected = count;
        if (hb) hb->named_first_tid[id] = eg_emu_tid;
    } else if (nb.expected != count) {
        if (hb && !hb->named_reported[id]) {
            hb->named_reported[id] = true;
            std::ostringstream oss;
            oss << "eg_emu_bar_sync(" << id << ", " << count
                << "): count mismatch (expected " << nb.expected << ")";
            eg_emu_record_hazard("named_barrier_count_mismatch", hb->cta_id, eg_emu_tid,
                                 hb->named_first_tid[id], id, oss.str());
        }
        b->waiting_named -= nb.arrived;
        nb.arrived = 0;
        nb.expected = 0;
        ++nb.gen;
        nb.cv.notify_all();
        return;
    }
    ++nb.arrived;
    ++b->waiting_named;
    if (nb.arrived == nb.expected) {
        b->waiting_named -= nb.expected;
        nb.arrived = 0;
        nb.expected = 0;
        ++nb.gen;
        if (hb) {
            hb->advance_epoch();
        }
        nb.cv.notify_all();
        return;
    }
    if (b->waiting_named + b->waiting_sync >= b->live_threads) {
        eg_emu_release_incomplete_named_barriers(b, hb, "named barrier");
        if (b->sync_arrived > 0 && b->sync_arrived >= b->live_threads) {
            if (hb) {
                hb->check_wgmma_uniformity("__syncthreads");
                hb->advance_epoch();
            }
            eg_emu_sync_release(b, b->sync_arrived);
        }
        return;
    }
    eg_emu_release_delayers(b);
    const unsigned g = nb.gen;
    nb.cv.wait(lock, [&] { return nb.gen != g; });
}

// --- 3. MBarrier wrappers (epoch advancement) ------------------------------

inline void eg_emu_hazard_mbar_wait(uint64_t* bar, int phase_parity) {
    eg_emu_mbar_wait(bar, phase_parity);
    if (eg_emu_cur_hb) eg_emu_cur_hb->advance_epoch();
}

inline bool eg_emu_hazard_mbar_test_wait(uint64_t* bar, int phase_parity) {
    bool ok = eg_emu_mbar_test_wait(bar, phase_parity);
    if (ok && eg_emu_cur_hb) eg_emu_cur_hb->advance_epoch();
    return ok;
}

inline void eg_emu_mbarrier_init(uint64_t* bar, int count) { eg_emu_mbar_init(bar, count); }
inline void eg_emu_mbarrier_arrive(uint64_t* bar) { eg_emu_mbar_arrive(bar); }
inline void eg_emu_mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx) { eg_emu_mbar_arrive_expect_tx(bar, tx); }
inline void eg_emu_mbarrier_wait(uint64_t* bar, int phase_parity) { eg_emu_hazard_mbar_wait(bar, phase_parity); }
inline bool eg_emu_mbarrier_try_wait(uint64_t* bar, int phase_parity) { return eg_emu_hazard_mbar_test_wait(bar, phase_parity); }
inline bool eg_emu_mbarrier_test_wait(uint64_t* bar, int phase_parity) { return eg_emu_hazard_mbar_test_wait(bar, phase_parity); }

// --- 4. Cluster barriers and DSMEM tracking --------------------------------

inline bool eg_emu_cluster_ready_locked(eg_emu_hazard_cluster* hc) {
    for (int r = 0; r < hc->cluster_size; ++r) {
        eg_emu_hazard_block* hb = hc->blocks[r];
        if (!hb || !hb->block) continue;
        eg_emu_block* b = hb->block;
        const int live = b->live_threads;
        if (live <= 0) continue;
        if (hc->waiting_threads[r] == 0) return false;
        const int blocked = hc->waiting_threads[r] + b->waiting_sync + b->waiting_named +
                            b->waiting_mbar + b->warp_stalled.load();
        if (blocked < live) return false;
    }
    return true;
}

inline void eg_emu_cluster_arrive() {
    if (!eg_emu_cur_hc) return;
    std::lock_guard<std::mutex> lock(eg_emu_cur_hc->mu);
    ++eg_emu_cur_hc->arrive_count[eg_emu_rank];
}

inline void eg_emu_cluster_wait() {
    if (!eg_emu_cur_cluster || !eg_emu_cur_hc) {
        std::fprintf(stderr, "cudahostemu: cluster_wait outside cluster\n");
        std::fflush(stderr);
        std::_Exit(3);
    }
    eg_emu_hazard_cluster* hc = eg_emu_cur_hc;
    eg_emu_block* b = eg_emu_cur_block;
    b->cluster_stalled.fetch_add(1);
    eg_emu_wake_delayers();
    {
        std::unique_lock<std::mutex> lock(hc->mu);
        if (hc->waiting_threads[eg_emu_rank] == 0) {
            ++hc->wait_count[eg_emu_rank];
        }
        ++hc->waiting_threads[eg_emu_rank];
        const unsigned g = hc->gen;
        if (eg_emu_cluster_ready_locked(hc)) {
            hc->accesses_since_wait.clear();
            for (int r = 0; r < hc->cluster_size; ++r) {
                hc->waiting_threads[r] = 0;
                if (hc->blocks[r]) {
                    hc->blocks[r]->check_wgmma_uniformity("cluster_wait");
                    hc->blocks[r]->advance_epoch();
                }
            }
            ++hc->gen;
            hc->cv.notify_all();
        } else {
            hc->cv.wait(lock, [&] { return hc->gen != g; });
        }
    }
    b->cluster_stalled.fetch_sub(1);
}

inline void eg_emu_cluster_reset_wait() {
    if (!eg_emu_cur_hc) return;
    std::lock_guard<std::mutex> lock(eg_emu_cur_hc->mu);
    eg_emu_cur_hc->wait_count[eg_emu_rank] = 0;
}

inline void eg_emu_hazard_cluster_sync() {
    eg_emu_cluster_arrive();
    eg_emu_cluster_wait();
}

template <typename T>
inline T* eg_emu_hazard_map_shared_rank(T* p, int r) {
    T* mapped = cooperative_groups::this_cluster().map_shared_rank(p, r);
    eg_emu_hazard_cluster* hc = eg_emu_cur_hc;
    eg_emu_hazard_block* hb = eg_emu_cur_hb;
    if (hc && hb && r != eg_emu_rank) {
        const int off = hb->offset_of(p);
        std::lock_guard<std::mutex> lock(hc->mu);
        if (hc->wait_count[eg_emu_rank] == 0 || hc->wait_count[r] == 0 || hc->exited[r]) {
            std::ostringstream oss;
            oss << "CTA " << hb->cta_id << " (rank " << eg_emu_rank << ", thread " << eg_emu_tid
                << ") mapped peer CTA rank " << r << " shared memory at offset " << off
                << " before cluster_wait (wait_count=" << hc->wait_count[eg_emu_rank]
                << ", peer_wait_count=" << hc->wait_count[r]
                << ", peer_exited=" << (hc->exited[r] ? 1 : 0) << ")";
            eg_emu_record_hazard("dsmem_before_cluster_wait", hb->cta_id, eg_emu_tid, r, off, oss.str());
        } else {
            hc->accesses_since_wait.push_back({hb->cta_id, eg_emu_rank, eg_emu_tid, r, off, hc->wait_count[r]});
        }
    }
    return mapped;
}

template <typename T>
inline eg_emu_smem_ptr<T> eg_emu_hazard_map_shared_rank(eg_emu_smem_ptr<T> p, int r) {
    return eg_emu_smem_ptr<T>(eg_emu_hazard_map_shared_rank(p.get(), r));
}

namespace eg_emu_hazard_cg {
struct cluster_group {
    void sync() const { eg_emu_hazard_cluster_sync(); }
    void arrive() const { eg_emu_cluster_arrive(); }
    void wait() const { eg_emu_cluster_wait(); }
    unsigned block_rank() const { return static_cast<unsigned>(eg_emu_rank); }
    unsigned num_blocks() const { return static_cast<unsigned>(eg_emu_cur_cluster->blocks.size()); }
    template <class T>
    T* map_shared_rank(T* p, int r) const {
        return eg_emu_hazard_map_shared_rank(p, r);
    }
    template <class T>
    eg_emu_smem_ptr<T> map_shared_rank(eg_emu_smem_ptr<T> p, int r) const {
        return eg_emu_hazard_map_shared_rank(p, r);
    }
};
inline cluster_group this_cluster() { return {}; }
struct thread_block {
    void sync() const { eg_emu_hazard_syncthreads(); }
};
inline thread_block this_thread_block() { return {}; }
}  // namespace eg_emu_hazard_cg

inline void eg_emu_hazard_cp_async(void* dst, const void* src, int n) {
    eg_emu_smem_write(dst, static_cast<size_t>(n));
    eg_emu_cp_async(dst, src, n);
}

inline void eg_emu_hazard_cp_async_bulk(void* smem_dst, const void* gmem_src, uint32_t size, uint64_t* bar) {
    eg_emu_smem_write(smem_dst, size);
    eg_emu_cp_async_bulk(smem_dst, gmem_src, size, bar);
}

inline void eg_emu_hazard_ldmatrix_x4(uint32_t r[4], const void* row, bool trans) {
    eg_emu_smem_read(row, 16);
    eg_emu_ldmatrix_x4(r, row, trans);
}

// --- 5. WGMMA warpgroup divergence tracking (C7519 detector) ---------------

inline void eg_emu_wgmma_record_op() {
    if (eg_emu_cur_hb) {
        eg_emu_cur_hb->record_wgmma_step(eg_emu_tid);
    }
}

inline void eg_emu_hazard_wgmma_fence() {
    eg_emu_wgmma_record_op();
    eg_emu_wgmma_fence();
}

inline void eg_emu_hazard_wgmma_commit() {
    eg_emu_wgmma_record_op();
    eg_emu_wgmma_commit();
}

inline void eg_emu_hazard_wgmma_wait(int n = 0) {
    eg_emu_wgmma_record_op();
    eg_emu_wgmma_wait(n);
}

inline void eg_emu_wgmma_Fence() { eg_emu_hazard_wgmma_fence(); }
inline void eg_emu_wgmma_commit_group() { eg_emu_hazard_wgmma_commit(); }
inline void eg_emu_wgmma_CommitGroup() { eg_emu_hazard_wgmma_commit(); }
inline void eg_emu_wgmma_wait_group(int n = 0) { eg_emu_hazard_wgmma_wait(n); }
inline void eg_emu_wgmma_WaitGroup(int n = 0) { eg_emu_hazard_wgmma_wait(n); }

inline void eg_emu_hazard_wgmma_bf16(float* d, uint64_t da, uint64_t db, int n = 64, int trans_a = 0, int trans_b = 0, int scale_d = 1) {
    eg_emu_wgmma_record_op();
    if (d) eg_emu_wgmma_bf16(d, da, db, n, trans_a, trans_b, scale_d);
}

inline void eg_emu_hazard_wgmma_f16(float* d, uint64_t da, uint64_t db, int n = 64, int trans_a = 0, int trans_b = 0, int scale_d = 1) {
    eg_emu_wgmma_record_op();
    if (d) eg_emu_wgmma_f16(d, da, db, n, trans_a, trans_b, scale_d);
}

inline void eg_emu_hazard_wgmma_e4m3(float* d, uint64_t da, uint64_t db, int n = 64, int scale_d = 1) {
    eg_emu_wgmma_record_op();
    if (d) eg_emu_wgmma_e4m3(d, da, db, n, scale_d);
}

inline void eg_emu_hazard_wgmma_rs_bf16(float* d, const uint32_t a[4], uint64_t db, int n = 64, int trans_b = 0, int scale_d = 1) {
    eg_emu_wgmma_record_op();
    if (d && a) eg_emu_wgmma_rs_bf16(d, a, db, n, trans_b, scale_d);
}

inline void eg_emu_hazard_wgmma_rs_f16(float* d, const uint32_t a[4], uint64_t db, int n = 64, int trans_b = 0, int scale_d = 1) {
    eg_emu_wgmma_record_op();
    if (d && a) eg_emu_wgmma_rs_f16(d, a, db, n, trans_b, scale_d);
}

inline void eg_emu_hazard_wgmma_rs_e4m3(float* d, const uint32_t a[4], uint64_t db, int n = 64, int scale_d = 1) {
    eg_emu_wgmma_record_op();
    if (d && a) eg_emu_wgmma_rs_e4m3(d, a, db, n, scale_d);
}

template <typename Ptr, std::enable_if_t<std::is_pointer_v<Ptr>, int> = 0>
inline void eg_emu_hazard_wgmma_bf16(float* d, Ptr a, uint64_t db, int n = 64, int trans_b = 0, int scale_d = 1) {
    eg_emu_hazard_wgmma_rs_bf16(d, a, db, n, trans_b, scale_d);
}

template <typename Ptr, std::enable_if_t<std::is_pointer_v<Ptr>, int> = 0>
inline void eg_emu_hazard_wgmma_f16(float* d, Ptr a, uint64_t db, int n = 64, int trans_b = 0, int scale_d = 1) {
    eg_emu_hazard_wgmma_rs_f16(d, a, db, n, trans_b, scale_d);
}

template <typename Ptr, std::enable_if_t<std::is_pointer_v<Ptr>, int> = 0>
inline void eg_emu_hazard_wgmma_e4m3(float* d, Ptr a, uint64_t db, int n = 64, int scale_d = 1) {
    eg_emu_hazard_wgmma_rs_e4m3(d, a, db, n, scale_d);
}

// Explicit m64n64k16 / m64n64k32 helpers (accepting 0 args or descriptor/register args).
inline void eg_emu_wgmma_m64n64k16_f16() { eg_emu_wgmma_record_op(); }
inline void eg_emu_wgmma_m64n64k16_f16(float* d, uint64_t da, uint64_t db, int trans_a = 0, int trans_b = 0, int scale_d = 1) {
    eg_emu_hazard_wgmma_f16(d, da, db, 64, trans_a, trans_b, scale_d);
}
template <typename Ptr, std::enable_if_t<std::is_pointer_v<Ptr>, int> = 0>
inline void eg_emu_wgmma_m64n64k16_f16(float* d, Ptr a, uint64_t db, int trans_b = 0, int scale_d = 1) {
    eg_emu_hazard_wgmma_rs_f16(d, a, db, 64, trans_b, scale_d);
}

inline void eg_emu_wgmma_m64n64k16_bf16() { eg_emu_wgmma_record_op(); }
inline void eg_emu_wgmma_m64n64k16_bf16(float* d, uint64_t da, uint64_t db, int trans_a = 0, int trans_b = 0, int scale_d = 1) {
    eg_emu_hazard_wgmma_bf16(d, da, db, 64, trans_a, trans_b, scale_d);
}
template <typename Ptr, std::enable_if_t<std::is_pointer_v<Ptr>, int> = 0>
inline void eg_emu_wgmma_m64n64k16_bf16(float* d, Ptr a, uint64_t db, int trans_b = 0, int scale_d = 1) {
    eg_emu_hazard_wgmma_rs_bf16(d, a, db, 64, trans_b, scale_d);
}

inline void eg_emu_wgmma_m64n64k16_e4m3() { eg_emu_wgmma_record_op(); }
inline void eg_emu_wgmma_m64n64k16_e4m3(float* d, uint64_t da, uint64_t db, int scale_d = 1) {
    eg_emu_hazard_wgmma_e4m3(d, da, db, 64, scale_d);
}
template <typename Ptr, std::enable_if_t<std::is_pointer_v<Ptr>, int> = 0>
inline void eg_emu_wgmma_m64n64k16_e4m3(float* d, Ptr a, uint64_t db, int scale_d = 1) {
    eg_emu_hazard_wgmma_rs_e4m3(d, a, db, 64, scale_d);
}

inline void eg_emu_wgmma_m64n64k32_e4m3() { eg_emu_wgmma_record_op(); }
inline void eg_emu_wgmma_m64n64k32_e4m3(float* d, uint64_t da, uint64_t db, int scale_d = 1) {
    eg_emu_hazard_wgmma_e4m3(d, da, db, 64, scale_d);
}
template <typename Ptr, std::enable_if_t<std::is_pointer_v<Ptr>, int> = 0>
inline void eg_emu_wgmma_m64n64k32_e4m3(float* d, Ptr a, uint64_t db, int scale_d = 1) {
    eg_emu_hazard_wgmma_rs_e4m3(d, a, db, 64, scale_d);
}

// --- 6. Hazard-aware kernel runner -----------------------------------------

template <typename F>
void eg_emu_hazard_run(dim3 grid, dim3 block, size_t smem, int cluster, F body) {
    eg_emu_last_grid = grid;
    const long n = static_cast<long>(block.x) * block.y * block.z;
    if (n <= 0 || n > 1024 || grid.x == 0 || grid.y == 0 || grid.z == 0 ||
        cluster < 1 || cluster > 16 || grid.x % cluster != 0) {
        eg_emu_fail(cudaErrorInvalidConfiguration);
        return;
    }
    gridDim = grid;
    blockDim = block;
    const size_t alloc_smem = std::max<size_t>(smem, 65536) + 16;
    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned cx = 0; cx < grid.x; cx += cluster) {
        std::memset(eg_emu_shared_start, eg_emu_smem_fill, eg_emu_shared_end - eg_emu_shared_start);
        eg_emu_cluster cl(static_cast<int>(n * cluster));
        eg_emu_hazard_cluster hc(cluster);
        std::vector<std::unique_ptr<eg_emu_block>> blocks;
        std::vector<std::unique_ptr<eg_emu_hazard_block>> hblocks;
        for (int r = 0; r < cluster; ++r) {
            blocks.emplace_back(new eg_emu_block(static_cast<int>(n)));
            eg_emu_block& b = *blocks.back();
            b.smem.assign(alloc_smem, eg_emu_smem_fill);
            for (int w = 0; w * 32 < n; ++w) {
                b.warps.emplace_back(new eg_emu_warp(std::min(32, static_cast<int>(n) - 32 * w)));
                b.warps.back()->stalled = &b.warp_stalled;
            }
            cl.blocks.push_back(&b);
            const int cta_id = static_cast<int>((bz * grid.y + by) * grid.x + cx + r);
            hblocks.emplace_back(new eg_emu_hazard_block(&b, cta_id, r, static_cast<int>(n)));
            hc.blocks.push_back(hblocks.back().get());
        }
        std::vector<std::thread> threads;
        for (int r = 0; r < cluster; ++r) {
            for (int i = 0; i < n; ++i) {
                threads.emplace_back([&, r, i] {
                    eg_emu_block& b = *blocks[r];
                    eg_emu_hazard_block& hb = *hblocks[r];
                    threadIdx = dim3(i % block.x, (i / block.x) % block.y, i / (block.x * block.y));
                    blockIdx = dim3(cx + r, by, bz);
                    eg_emu_cur_block = &b;
                    eg_emu_cur_warp = b.warps[i / 32].get();
                    eg_emu_cur_cluster = &cl;
                    eg_emu_lane = i % 32;
                    eg_emu_rank = r;
                    eg_emu_tid = i;
                    eg_emu_cur_hb = &hb;
                    eg_emu_cur_hc = &hc;
                    eg_emu_cp_pending.clear();
                    eg_emu_cp_committed = 0;
                    eg_emu_wg_async = false;
                    eg_emu_wg_pending.clear();
                    eg_emu_wg_accs.clear();
                    eg_emu_wg_committed = 0;

                    body();

                    // Clear any uncommitted/unwaited async wgmma ops so divergent warpgroup
                    // kernels report HazardWGMMADivergence instead of aborting with _Exit(3).
                    eg_emu_wg_pending.clear();
                    eg_emu_wg_accs.clear();

                    eg_emu_cur_warp->drop(eg_emu_lane);
                    int rem_live = 0;
                    {
                        std::lock_guard<std::mutex> lock(b.bar_mu);
                        rem_live = --b.live_threads;
                        if (b.waiting_named + b.waiting_sync >= b.live_threads && b.waiting_named > 0) {
                            eg_emu_release_incomplete_named_barriers(&b, &hb, "thread exit");
                        }
                        if (b.sync_arrived > 0 && b.sync_arrived >= b.live_threads) {
                            hb.check_wgmma_uniformity("__syncthreads");
                            hb.advance_epoch();
                            eg_emu_sync_release(&b, b.sync_arrived);
                        }
                        eg_emu_release_delayers(&b);
                    }
                    {
                        std::lock_guard<std::mutex> lock(hc.mu);
                        if (rem_live == 0) {
                            hc.exited[r] = true;
                        }
                        if (eg_emu_cluster_ready_locked(&hc)) {
                            bool any_waiting = false;
                            for (int k = 0; k < cluster; ++k) {
                                if (hc.waiting_threads[k] > 0) any_waiting = true;
                            }
                            if (any_waiting) {
                                hc.accesses_since_wait.clear();
                                for (int k = 0; k < cluster; ++k) {
                                    hc.waiting_threads[k] = 0;
                                    if (hc.blocks[k]) {
                                        hc.blocks[k]->check_wgmma_uniformity("cluster_wait");
                                        hc.blocks[k]->advance_epoch();
                                    }
                                }
                                ++hc.gen;
                                hc.cv.notify_all();
                            }
                        }
                    }
                    cl.bar.arrive_and_drop();
                });
            }
        }
        for (auto& t : threads) t.join();
        for (int r = 0; r < cluster; ++r) {
            hblocks[r]->check_wgmma_uniformity("kernel exit");
        }
        // Check if any CTA accessed peer DSMEM when the peer had fewer cluster_wait synchronizations.
        {
            std::lock_guard<std::mutex> lock(hc.mu);
            for (const auto& acc : hc.accesses_since_wait) {
                if (hc.wait_count[acc.peer_rank] < hc.wait_count[acc.src_rank]) {
                    std::ostringstream oss;
                    oss << "CTA " << acc.src_cta << " accessed peer CTA rank " << acc.peer_rank
                        << " DSMEM at offset " << acc.offset << " after peer exited without matching cluster_wait";
                    eg_emu_record_hazard("dsmem_before_cluster_wait", acc.src_cta, acc.src_tid, acc.peer_rank, acc.offset, oss.str());
                }
            }
        }
    }
    eg_emu_flush_hazards();
}

template <typename K, typename... Args>
void eg_emu_hazard_launch_cluster(dim3 grid, dim3 block, size_t smem, int cluster, K kernel, Args... args) {
    eg_emu_hazard_run(grid, block, smem, cluster, [&] { kernel(args...); });
}

template <typename K, typename... Args>
void eg_emu_hazard_launch(dim3 grid, dim3 block, size_t smem, K kernel, Args... args) {
    if constexpr (std::is_pointer_v<std::decay_t<K>>) {
        const void* p = reinterpret_cast<const void*>(kernel);
        if (eg_emu_cluster_kernels.count(p)) {
            auto it = eg_emu_cluster_sizes.find(p);
            int cluster = (it != eg_emu_cluster_sizes.end()) ? it->second : 8;
            eg_emu_hazard_launch_cluster(grid, block, smem, cluster, kernel, args...);
            return;
        }
    }
    eg_emu_hazard_run(grid, block, smem, 1, [&] { kernel(args...); });
}

template <class F>
void eg_emu_hazard_run_grid(dim3 grid, dim3 block, F kernel) {
    eg_emu_hazard_run(grid, block, 0, 1, kernel);
    eg_emu_check_launch("run_grid");
}

template <class F>
void eg_emu_hazard_run_cluster_grid(dim3 grid, dim3 block, int n, F kernel) {
    eg_emu_hazard_run(grid, block, 0, n, kernel);
    eg_emu_check_launch("run_cluster_grid");
}

// --- 7. Macro redirections for kernel source -------------------------------

inline void eg_emu_syncthreads() { eg_emu_hazard_syncthreads(); }

#define __syncthreads eg_emu_hazard_syncthreads
#define eg_emu_bar_sync eg_emu_hazard_bar_sync
#define bar_sync eg_emu_hazard_bar_sync
#define eg_emu_mbar_wait eg_emu_hazard_mbar_wait
#define eg_emu_mbar_test_wait eg_emu_hazard_mbar_test_wait
#define eg_emu_cluster_sync eg_emu_hazard_cluster_sync
#define eg_emu_map_shared_rank eg_emu_hazard_map_shared_rank
#define cooperative_groups eg_emu_hazard_cg
#define eg_emu_cp_async eg_emu_hazard_cp_async
#define eg_emu_cp_async_bulk eg_emu_hazard_cp_async_bulk
#define eg_emu_ldmatrix_x4 eg_emu_hazard_ldmatrix_x4

#define eg_emu_wgmma_fence eg_emu_hazard_wgmma_fence
#define eg_emu_wgmma_commit eg_emu_hazard_wgmma_commit
#define eg_emu_wgmma_wait eg_emu_hazard_wgmma_wait
#define eg_emu_wgmma_bf16 eg_emu_hazard_wgmma_bf16
#define eg_emu_wgmma_f16 eg_emu_hazard_wgmma_f16
#define eg_emu_wgmma_e4m3 eg_emu_hazard_wgmma_e4m3
#define eg_emu_wgmma_rs_bf16 eg_emu_hazard_wgmma_rs_bf16
#define eg_emu_wgmma_rs_f16 eg_emu_hazard_wgmma_rs_f16
#define eg_emu_wgmma_rs_e4m3 eg_emu_hazard_wgmma_rs_e4m3

#define eg_emu_run eg_emu_hazard_run
#define eg_emu_launch eg_emu_hazard_launch
#define eg_emu_launch_cluster eg_emu_hazard_launch_cluster
#define run_grid eg_emu_hazard_run_grid
#define run_cluster_grid eg_emu_hazard_run_cluster_grid
#undef CUDAHOSTEMU_LAUNCH
#define CUDAHOSTEMU_LAUNCH(kernel, grid, block, smem, stream, ...) eg_emu_hazard_launch(grid, block, smem, kernel __VA_OPT__(,) __VA_ARGS__)

#undef CUDAHOSTEMU_SHARED
#undef MOE8_SHARED
#undef EG_EMU_SHARED
#define CUDAHOSTEMU_SHARED(T, name, dims) eg_emu_smem_ptr<std::remove_extent_t<T dims>> name(reinterpret_cast<std::remove_extent_t<T dims>*>(eg_emu_keyed_shared<T dims>(#name ":" EG_EMU_STR(__LINE__), 0)))
#define MOE8_SHARED(T, name, dims) CUDAHOSTEMU_SHARED(T, name, dims)
#define EG_EMU_SHARED(T, name, dims) eg_emu_smem_ptr<std::remove_extent_t<T dims>> name(reinterpret_cast<std::remove_extent_t<T dims>*>(eg_emu_shared<T dims>(#name ":" EG_EMU_STR(__LINE__))))
