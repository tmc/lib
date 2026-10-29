// clc.h: Blackwell sm_100a clusterlaunchcontrol (PTX ISA 8.6+) dynamic
// work-stealing emulation and async mbarrier hazard detector for cudahostemu.
//
// Included after cudahostemu.h when RunWithCLC compiles kernel source.
#pragma once

#ifndef EG_EMU
#define EG_EMU 1
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct eg_emu_clc_mbar_state {
    uint4* dst = nullptr;
    uint4 response = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    uint64_t issue_seq = 0;
    uint64_t wait_seq = 0;
    int last_wait_parity = -1;
};

inline std::atomic<uint32_t> eg_emu_clc_active_ctas{0};
inline std::atomic<uint32_t> eg_emu_clc_total_tiles{0};
inline std::atomic<uint32_t> eg_emu_clc_next_tile{0};
inline std::atomic<int> eg_emu_clc_errors{0};

inline std::mutex eg_emu_clc_mu;
inline std::condition_variable eg_emu_clc_cv;
inline std::vector<std::string> eg_emu_clc_error_msgs;
inline std::map<uint64_t*, eg_emu_clc_mbar_state> eg_emu_clc_mbars;

// eg_emu_clc_init initializes the gigathread work queue for a persistent
// kernel with active_ctas resident CTAs (0 .. active_ctas-1) and total_tiles
// logical work tiles (resident CTAs steal active_ctas .. total_tiles-1).
inline void eg_emu_clc_init(int active_ctas, int total_tiles) {
    std::lock_guard<std::mutex> lock(eg_emu_clc_mu);
    const uint32_t active = active_ctas > 0 ? static_cast<uint32_t>(active_ctas) : 0u;
    const uint32_t total = total_tiles > 0 ? static_cast<uint32_t>(total_tiles) : 0u;
    eg_emu_clc_active_ctas.store(active, std::memory_order_release);
    eg_emu_clc_total_tiles.store(total, std::memory_order_release);
    eg_emu_clc_next_tile.store(active, std::memory_order_release);
    eg_emu_clc_errors.store(0, std::memory_order_release);
    eg_emu_clc_error_msgs.clear();
    eg_emu_clc_mbars.clear();
}

// eg_emu_clc_error_count returns the number of clusterlaunchcontrol hazard
// violations observed since eg_emu_clc_init.
inline int eg_emu_clc_error_count() {
    return eg_emu_clc_errors.load(std::memory_order_acquire);
}

inline void eg_emu_clc_record_error(const char* op, const char* detail) {
    eg_emu_clc_errors.fetch_add(1, std::memory_order_acq_rel);
    std::lock_guard<std::mutex> lock(eg_emu_clc_mu);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "cudahostemu: clc hazard in block (%u,%u,%u) thread %u: %s: %s",
                  blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, op, detail);
    eg_emu_clc_error_msgs.emplace_back(buf);
}

// eg_emu_clc_check exits with a non-zero status and descriptive diagnostic if
// any clusterlaunchcontrol hazard was recorded during the kernel launch.
inline void eg_emu_clc_check() {
    const int n = eg_emu_clc_error_count();
    if (n <= 0) return;
    std::lock_guard<std::mutex> lock(eg_emu_clc_mu);
    for (const auto& msg : eg_emu_clc_error_msgs) {
        std::fprintf(stderr, "%s\n", msg.c_str());
    }
    std::fprintf(stderr, "cudahostemu: %d clusterlaunchcontrol mbarrier hazard(s) detected\n", n);
    std::fflush(stderr);
    std::_Exit(4);
}

// eg_emu_clc_try_cancel_async emulates
// clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128.
// It atomically steals the next uncancelled tile ID from the gigathread work
// queue, immediately poisons *response_smem with 0xffffffff, and defers
// materializing the 16-byte response until *mbar is awaited.
inline void eg_emu_clc_try_cancel_async(uint4* response_smem, uint64_t* mbar) {
    if (!response_smem || !mbar) {
        eg_emu_clc_record_error("eg_emu_clc_try_cancel_async", "null response_smem or mbar pointer");
        return;
    }
    const uint32_t tile = eg_emu_clc_next_tile.fetch_add(1u, std::memory_order_acq_rel);
    const uint32_t total = eg_emu_clc_total_tiles.load(std::memory_order_acquire);
    uint4 real_resp;
    if (tile < total) {
        real_resp = make_uint4(tile, 0u, 0u, 1u);
    } else {
        real_resp = make_uint4(0u, 0u, 0u, 0u);
    }
    *response_smem = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
    {
        std::lock_guard<std::mutex> lock(eg_emu_clc_mu);
        eg_emu_clc_mbar_state& st = eg_emu_clc_mbars[mbar];
        st.dst = response_smem;
        st.response = real_resp;
        ++st.issue_seq;
    }
    eg_emu_clc_cv.notify_all();

    if (eg_emu_cur_block) {
        std::lock_guard<std::mutex> lock(eg_emu_cur_block->bar_mu);
        auto it = eg_emu_cur_block->mbars.find(mbar);
        if (it != eg_emu_cur_block->mbars.end()) {
            it->second.tx = 0;
            it->second.pending = 0;
            eg_emu_mbar_advance(eg_emu_cur_block, it->second);
        }
    }
}

// eg_emu_clc_wait awaits the mbarrier associated with an asynchronous
// clusterlaunchcontrol.try_cancel request and materializes the 16-byte
// response into *response_smem.
inline void eg_emu_clc_wait(uint64_t* mbar, int phase = 0) {
    if (!mbar) {
        eg_emu_clc_record_error("eg_emu_clc_wait", "null mbar pointer");
        return;
    }
    {
        std::unique_lock<std::mutex> lock(eg_emu_clc_mu);
        const auto t0 = std::chrono::steady_clock::now();
        while (true) {
            auto it = eg_emu_clc_mbars.find(mbar);
            if (it != eg_emu_clc_mbars.end()) {
                eg_emu_clc_mbar_state& st = it->second;
                if (st.issue_seq > st.wait_seq) {
                    break;
                }
                if (st.issue_seq > 0 && st.last_wait_parity == (phase & 1)) {
                    return;
                }
            }
            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(500)) {
                eg_emu_clc_record_error("eg_emu_clc_wait", "timed out waiting for eg_emu_clc_try_cancel_async");
                return;
            }
            eg_emu_clc_cv.wait_for(lock, std::chrono::microseconds(200));
        }
    }

    // Adversarial delay: wait until all other live threads in the block are
    // blocked (or up to a short bound) so any unsynchronized reader observes
    // the 0xffffffff poison before the response lands.
    if (eg_emu_cur_block) {
        eg_emu_block* b = eg_emu_cur_block;
        std::unique_lock<std::mutex> lock(b->bar_mu);
        const uint64_t gen = b->delay_gen;
        ++b->delaying;
        const auto t0 = std::chrono::steady_clock::now();
        while (b->delay_gen == gen) {
            if (eg_emu_quiescent(b)) {
                ++b->delay_gen;
                b->delaying = 0;
                b->mbar_cv.notify_all();
                break;
            }
            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(5)) {
                --b->delaying;
                break;
            }
            b->mbar_cv.wait_for(lock, std::chrono::microseconds(200));
        }
    }

    {
        std::lock_guard<std::mutex> lock(eg_emu_clc_mu);
        auto it = eg_emu_clc_mbars.find(mbar);
        if (it != eg_emu_clc_mbars.end()) {
            eg_emu_clc_mbar_state& st = it->second;
            if (st.issue_seq > st.wait_seq && st.dst != nullptr) {
                *st.dst = st.response;
                st.wait_seq = st.issue_seq;
                st.last_wait_parity = phase & 1;
            }
        }
    }
    eg_emu_clc_cv.notify_all();
}

// eg_emu_clc_query_is_canceled emulates
// clusterlaunchcontrol.query_cancel.is_canceled.pred.b128.
// Reading a poisoned (0xffffffff) response before awaiting mbar increments
// eg_emu_clc_error_count() and returns false.
inline bool eg_emu_clc_query_is_canceled(uint4 resp) {
    if (resp.w == 0xffffffffu || resp.x == 0xffffffffu) {
        eg_emu_clc_record_error(
            "eg_emu_clc_query_is_canceled",
            "read poisoned response_smem (0xffffffff) before awaiting mbarrier");
        return false;
    }
    return resp.w != 0u;
}

inline bool eg_emu_clc_query_is_canceled(const uint4* resp) {
    if (!resp) {
        eg_emu_clc_record_error("eg_emu_clc_query_is_canceled", "null response_smem pointer");
        return false;
    }
    return eg_emu_clc_query_is_canceled(*resp);
}

// eg_emu_clc_query_first_ctaid_x emulates
// clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128.
// Reading a poisoned (0xffffffff) response before awaiting mbar increments
// eg_emu_clc_error_count() and returns 0xffffffffu.
inline uint32_t eg_emu_clc_query_first_ctaid_x(uint4 resp) {
    if (resp.x == 0xffffffffu || resp.w == 0xffffffffu) {
        eg_emu_clc_record_error(
            "eg_emu_clc_query_first_ctaid_x",
            "read poisoned response_smem (0xffffffff) before awaiting mbarrier");
        return 0xffffffffu;
    }
    return resp.x;
}

inline uint32_t eg_emu_clc_query_first_ctaid_x(const uint4* resp) {
    if (!resp) {
        eg_emu_clc_record_error("eg_emu_clc_query_first_ctaid_x", "null response_smem pointer");
        return 0xffffffffu;
    }
    return eg_emu_clc_query_first_ctaid_x(*resp);
}

inline uint32_t eg_emu_clc_query_first_ctaid_y(uint4 resp) {
    if (resp.y == 0xffffffffu || resp.w == 0xffffffffu) {
        eg_emu_clc_record_error(
            "eg_emu_clc_query_first_ctaid_y",
            "read poisoned response_smem (0xffffffff) before awaiting mbarrier");
        return 0xffffffffu;
    }
    return resp.y;
}

inline uint32_t eg_emu_clc_query_first_ctaid_z(uint4 resp) {
    if (resp.z == 0xffffffffu || resp.w == 0xffffffffu) {
        eg_emu_clc_record_error(
            "eg_emu_clc_query_first_ctaid_z",
            "read poisoned response_smem (0xffffffff) before awaiting mbarrier");
        return 0xffffffffu;
    }
    return resp.z;
}

// --- MBarrier wrappers for CLC integration ---------------------------------

inline void eg_emu_clc_mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx) {
    bool is_clc = false;
    {
        std::lock_guard<std::mutex> lock(eg_emu_clc_mu);
        is_clc = (eg_emu_clc_mbars.find(bar) != eg_emu_clc_mbars.end());
    }
    if (is_clc) {
        return;
    }
    if (eg_emu_cur_block) {
        std::lock_guard<std::mutex> lock(eg_emu_cur_block->bar_mu);
        auto it = eg_emu_cur_block->mbars.find(bar);
        if (it != eg_emu_cur_block->mbars.end() && it->second.copies.empty() && tx == 16) {
            return;
        }
    }
    eg_emu_mbar_arrive_expect_tx(bar, tx);
}

inline void eg_emu_clc_mbar_wait(uint64_t* bar, int phase_parity) {
    bool is_clc = false;
    {
        std::lock_guard<std::mutex> lock(eg_emu_clc_mu);
        is_clc = (eg_emu_clc_mbars.find(bar) != eg_emu_clc_mbars.end());
    }
    if (!is_clc && eg_emu_cur_block) {
        std::lock_guard<std::mutex> lock(eg_emu_cur_block->bar_mu);
        auto it = eg_emu_cur_block->mbars.find(bar);
        if (it == eg_emu_cur_block->mbars.end() || (it->second.copies.empty() && it->second.tx == 0)) {
            is_clc = true;
        }
    }
    if (is_clc) {
        eg_emu_clc_wait(bar, phase_parity);
        return;
    }
    eg_emu_mbar_wait(bar, phase_parity);
}

inline bool eg_emu_clc_mbar_test_wait(uint64_t* bar, int phase_parity) {
    bool is_clc = false;
    {
        std::lock_guard<std::mutex> lock(eg_emu_clc_mu);
        is_clc = (eg_emu_clc_mbars.find(bar) != eg_emu_clc_mbars.end());
    }
    if (is_clc) {
        eg_emu_clc_wait(bar, phase_parity);
        return true;
    }
    return eg_emu_mbar_test_wait(bar, phase_parity);
}

#define eg_emu_mbar_arrive_expect_tx eg_emu_clc_mbar_arrive_expect_tx
#define eg_emu_mbar_wait eg_emu_clc_mbar_wait
#define eg_emu_mbar_test_wait eg_emu_clc_mbar_test_wait
