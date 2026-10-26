// pdl.h: adversarial Hopper/Blackwell Programmatic Dependent Launch (PDL)
// emulation (griddepcontrol.launch_dependents / griddepcontrol.wait) for
// cudahostemu.
#ifndef EG_EMU_PDL_H
#define EG_EMU_PDL_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

enum eg_emu_pdl_phase {
    EG_PDL_REGISTERED = 0,
    EG_PDL_POISONED = 1,
    EG_PDL_DELIVERED = 2,
};

struct eg_emu_pdl_buf {
    unsigned char* ptr = nullptr;
    size_t bytes = 0;
    std::vector<unsigned char> produced_data;
};

struct eg_emu_pdl_stage {
    uint32_t stage_id = 0;
    eg_emu_pdl_phase phase = EG_PDL_REGISTERED;
    std::vector<eg_emu_pdl_buf> bufs;
    std::set<int> triggered_ctas;
    int expected_producer_ctas = 0;
    bool producer_ok = false;
    int consumer_ctas_waited = 0;
    int consumer_ctas_missed = 0;
};

struct eg_emu_pdl_violation {
    uint32_t stage_id = 0;
    int cta = 0;
    int thread = 0;
    std::string detail;
};

inline std::mutex eg_emu_pdl_mu;
inline std::map<uint32_t, eg_emu_pdl_stage> eg_emu_pdl_stages;
inline std::vector<eg_emu_pdl_violation> eg_emu_pdl_violation_list;
inline std::atomic<int> eg_emu_pdl_violations{0};

inline int eg_emu_pdl_cur_cta = 0;
inline thread_local int eg_emu_pdl_tid = 0;
inline bool eg_emu_pdl_in_kernel = false;
inline std::map<uint32_t, bool> eg_emu_pdl_cta_waited;
inline std::set<uint32_t> eg_emu_pdl_kernel_triggered_stages;
inline std::set<uint32_t> eg_emu_pdl_kernel_waited_stages;

inline void eg_emu_pdl_record_violation_locked(uint32_t stage_id, int cta, int tid, const std::string& detail) {
    eg_emu_pdl_violations.fetch_add(1, std::memory_order_relaxed);
    for (const auto& v : eg_emu_pdl_violation_list) {
        if (v.stage_id == stage_id && v.cta == cta && v.detail == detail) {
            return;
        }
    }
    eg_emu_pdl_violation_list.push_back({stage_id, cta, tid, detail});
}

inline void eg_emu_pdl_record_violation(uint32_t stage_id, int cta, int tid, const std::string& detail) {
    std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
    eg_emu_pdl_record_violation_locked(stage_id, cta, tid, detail);
}

inline int eg_emu_pdl_violation_count() {
    return eg_emu_pdl_violations.load(std::memory_order_relaxed);
}

inline int eg_emu_pdl_triggered_ctas(uint32_t stage_id = 1) {
    std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
    auto it = eg_emu_pdl_stages.find(stage_id);
    if (it == eg_emu_pdl_stages.end()) return 0;
    return static_cast<int>(it->second.triggered_ctas.size());
}

inline int eg_emu_pdl_waited_ctas(uint32_t stage_id = 1) {
    std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
    auto it = eg_emu_pdl_stages.find(stage_id);
    if (it == eg_emu_pdl_stages.end()) return 0;
    return it->second.consumer_ctas_waited;
}

inline void eg_emu_pdl_reset() {
    std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
    eg_emu_pdl_stages.clear();
    eg_emu_pdl_violation_list.clear();
    eg_emu_pdl_violations.store(0, std::memory_order_relaxed);
    eg_emu_pdl_cta_waited.clear();
    eg_emu_pdl_kernel_triggered_stages.clear();
    eg_emu_pdl_kernel_waited_stages.clear();
}

inline void eg_emu_pdl_flush_violations() {
#ifdef EG_EMU_PDL_VIOLATION_FILE
    std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
    FILE* f = std::fopen(EG_EMU_PDL_VIOLATION_FILE, "wb");
    if (!f) return;
    for (const auto& v : eg_emu_pdl_violation_list) {
        std::fprintf(f, "%u\t%d\t%d\t%s\n", v.stage_id, v.cta, v.thread, v.detail.c_str());
    }
    std::fclose(f);
#endif
}

// eg_emu_pdl_register_buffer registers [ptr, ptr+bytes) as produced by stage_id.
// The buffer starts poisoned with 0xff (NaN) until the producer stage completes
// its stores and triggers eg_emu_pdl_launch_dependents(stage_id), and the
// consumer stage calls eg_emu_pdl_wait(stage_id).
inline void eg_emu_pdl_register_buffer(void* ptr, size_t bytes, uint32_t stage_id = 1) {
    if (!ptr || bytes == 0) return;
    std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
    eg_emu_pdl_stage& st = eg_emu_pdl_stages[stage_id];
    if (st.stage_id != stage_id || st.phase == EG_PDL_DELIVERED) {
        st = eg_emu_pdl_stage{};
        st.stage_id = stage_id;
    }
    unsigned char* p = static_cast<unsigned char*>(ptr);
    for (auto& b : st.bufs) {
        if (b.ptr == p) {
            b.bytes = bytes;
            return;
        }
    }
    std::vector<unsigned char> snap(p, p + bytes);
    std::memset(p, 0xff, bytes);
    st.bufs.push_back({p, bytes, std::move(snap)});
}

// eg_emu_pdl_launch_dependents models PTX griddepcontrol.launch_dependents for stage_id.
inline void eg_emu_pdl_launch_dependents(uint32_t stage_id = 1) {
    std::atomic_thread_fence(std::memory_order_release);
    std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
    eg_emu_pdl_stage& st = eg_emu_pdl_stages[stage_id];
    st.stage_id = stage_id;
    st.triggered_ctas.insert(eg_emu_pdl_cur_cta + eg_emu_rank);
    eg_emu_pdl_kernel_triggered_stages.insert(stage_id);
    if (!eg_emu_pdl_in_kernel) {
        st.producer_ok = true;
        st.phase = EG_PDL_POISONED;
        for (auto& b : st.bufs) {
            b.produced_data.assign(b.ptr, b.ptr + b.bytes);
            std::memset(b.ptr, 0xff, b.bytes);
        }
    }
}

// eg_emu_pdl_wait models PTX griddepcontrol.wait for stage_id.
// It delays unpoisoning until the calling CTA is quiescent so any thread in the
// CTA that reads the buffer before calling eg_emu_pdl_wait observes 0xff (NaN).
inline void eg_emu_pdl_wait(uint32_t stage_id = 1) {
    if (eg_emu_block* b = eg_emu_cur_block) {
        std::unique_lock<std::mutex> lock(b->bar_mu);
        const uint64_t g = b->delay_gen;
        ++b->delaying;
        eg_emu_release_delayers(b);
        if (!b->mbar_cv.wait_for(lock, std::chrono::milliseconds(50), [&] { return b->delay_gen != g; })) {
            if (b->delay_gen == g && b->delaying > 0) {
                --b->delaying;
                eg_emu_release_delayers(b);
            }
        }
    }
    std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
    eg_emu_pdl_kernel_waited_stages.insert(stage_id);
    auto it = eg_emu_pdl_stages.find(stage_id);
    if (it == eg_emu_pdl_stages.end()) {
        std::ostringstream oss;
        oss << "CTA " << eg_emu_pdl_cur_cta << " called eg_pdl_wait(" << stage_id
            << ") on unregistered stage";
        eg_emu_pdl_record_violation_locked(stage_id, eg_emu_pdl_cur_cta, eg_emu_pdl_tid, oss.str());
        return;
    }
    eg_emu_pdl_stage& st = it->second;
    if (st.phase == EG_PDL_REGISTERED || !st.producer_ok || st.triggered_ctas.empty()) {
        std::ostringstream oss;
        oss << "CTA " << eg_emu_pdl_cur_cta << " called eg_pdl_wait(" << stage_id
            << ") before producer triggered griddepcontrol.launch_dependents";
        eg_emu_pdl_record_violation_locked(stage_id, eg_emu_pdl_cur_cta, eg_emu_pdl_tid, oss.str());
        for (auto& b : st.bufs) {
            std::memset(b.ptr, 0xff, b.bytes);
        }
        return;
    }
    if (!eg_emu_pdl_cta_waited[stage_id]) {
        eg_emu_pdl_cta_waited[stage_id] = true;
        for (auto& b : st.bufs) {
            std::memcpy(b.ptr, b.produced_data.data(), b.bytes);
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}

#ifndef EG_PDL_DEVICE_HELPERS
#define EG_PDL_DEVICE_HELPERS 1
static inline void eg_pdl_trigger(uint32_t stage = 1) {
#if !defined(EG_NO_PDL_TRIGGER)
    eg_emu_pdl_launch_dependents(stage);
#else
    (void)stage;
#endif
}

static inline void eg_pdl_wait(uint32_t stage) {
#if !defined(EG_NO_PDL_WAIT)
    eg_emu_pdl_wait(stage);
#else
    (void)stage;
#endif
}

static inline void eg_pdl_launch_dependents(uint32_t stage) {
#if !defined(EG_NO_PDL_TRIGGER)
    eg_emu_pdl_launch_dependents(stage);
#else
    (void)stage;
#endif
}
#endif

template <typename T = uint32_t, std::enable_if_t<std::is_integral_v<T>, int> = 0>
static inline void eg_pdl_wait(T stage) {
#if !defined(EG_NO_PDL_WAIT)
    eg_emu_pdl_wait(static_cast<uint32_t>(stage));
#else
    (void)stage;
#endif
}

template <typename T = uint32_t, std::enable_if_t<std::is_integral_v<T>, int> = 0>
static inline void eg_pdl_launch_dependents(T stage) {
#if !defined(EG_NO_PDL_TRIGGER)
    eg_emu_pdl_launch_dependents(static_cast<uint32_t>(stage));
#else
    (void)stage;
#endif
}

struct eg_emu_pdl_out_arg {
    unsigned char* ptr = nullptr;
    size_t elem_size = 0;
};

template <typename A>
inline void eg_emu_pdl_collect_out_arg(std::vector<eg_emu_pdl_out_arg>& out, A arg) {
    if constexpr (std::is_pointer_v<A>) {
        using Elem = std::remove_pointer_t<A>;
        if constexpr (!std::is_const_v<Elem> && !std::is_void_v<Elem>) {
            if (arg != nullptr) {
                out.push_back({reinterpret_cast<unsigned char*>(arg), sizeof(Elem)});
            }
        }
    }
}

inline bool eg_emu_pdl_all_ff(const unsigned char* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (p[i] != 0xff) return false;
    }
    return true;
}

inline bool eg_emu_pdl_has_nan_elem(const eg_emu_pdl_out_arg& a) {
    if (!a.ptr) return false;
    if (a.elem_size == 4) {
        uint32_t u = 0;
        std::memcpy(&u, a.ptr, 4);
        if (u == 0xffffffffu) return true;
        float f = 0.0f;
        std::memcpy(&f, a.ptr, 4);
        return std::isnan(f);
    }
    if (a.elem_size == 2) {
        uint16_t u = 0;
        std::memcpy(&u, a.ptr, 2);
        return u == 0xffffu || (u & 0x7f80u) == 0x7f80u && (u & 0x007fu) != 0;
    }
    return false;
}

template <typename F>
void eg_emu_pdl_run(dim3 grid, dim3 block, size_t smem, int cluster, F body,
                    const std::vector<eg_emu_pdl_out_arg>& out_args = {}) {
    eg_emu_last_grid = grid;
    const long n = static_cast<long>(block.x) * block.y * block.z;
    if (n <= 0 || n > 1024 || grid.x == 0 || grid.y == 0 || grid.z == 0 ||
        cluster < 1 || cluster > 16 || grid.x % cluster != 0) {
        eg_emu_fail(cudaErrorInvalidConfiguration);
        return;
    }
    gridDim = grid;
    blockDim = block;
    const int total_ctas = static_cast<int>(grid.x * grid.y * grid.z);

    std::vector<uint32_t> poisoned_at_start;
    std::vector<uint32_t> registered_at_start;
    {
        std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
        eg_emu_pdl_in_kernel = true;
        eg_emu_pdl_kernel_triggered_stages.clear();
        eg_emu_pdl_kernel_waited_stages.clear();
        for (auto& [sid, st] : eg_emu_pdl_stages) {
            if (st.phase == EG_PDL_POISONED) {
                poisoned_at_start.push_back(sid);
                st.consumer_ctas_waited = 0;
                st.consumer_ctas_missed = 0;
            } else if (st.phase == EG_PDL_REGISTERED) {
                registered_at_start.push_back(sid);
            }
        }
    }

    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned cx = 0; cx < grid.x; cx += cluster) {
        const int base_cta = static_cast<int>((bz * grid.y + by) * grid.x + cx);
        {
            std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
            eg_emu_pdl_cur_cta = base_cta;
            eg_emu_pdl_cta_waited.clear();
            for (uint32_t sid : poisoned_at_start) {
                auto& st = eg_emu_pdl_stages[sid];
                for (auto& b : st.bufs) {
                    std::memset(b.ptr, 0xff, b.bytes);
                }
            }
        }

        std::memset(eg_emu_shared_start, eg_emu_smem_fill, eg_emu_shared_end - eg_emu_shared_start);
        eg_emu_cluster cl(static_cast<int>(n * cluster));
        std::vector<std::unique_ptr<eg_emu_block>> blocks;
        for (int r = 0; r < cluster; ++r) {
            blocks.emplace_back(new eg_emu_block(static_cast<int>(n)));
            eg_emu_block& b = *blocks.back();
            b.smem.assign(smem + 16, eg_emu_smem_fill);
            for (int w = 0; w * 32 < n; ++w) {
                b.warps.emplace_back(new eg_emu_warp(std::min(32, static_cast<int>(n) - 32 * w)));
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
                    eg_emu_pdl_tid = i;
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

        {
            std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
            for (uint32_t sid : poisoned_at_start) {
                auto& st = eg_emu_pdl_stages[sid];
                if (eg_emu_pdl_cta_waited[sid]) {
                    st.consumer_ctas_waited += cluster;
                    for (auto& b : st.bufs) {
                        b.produced_data.assign(b.ptr, b.ptr + b.bytes);
                        std::memset(b.ptr, 0xff, b.bytes);
                    }
                } else {
                    st.consumer_ctas_missed += cluster;
                }
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(eg_emu_pdl_mu);
        eg_emu_pdl_in_kernel = false;

        // Finalize stages that were in POISONED state when this kernel began.
        for (uint32_t sid : poisoned_at_start) {
            auto& st = eg_emu_pdl_stages[sid];
            if (st.consumer_ctas_waited > 0 && st.consumer_ctas_missed == 0 && st.producer_ok) {
                for (auto& b : st.bufs) {
                    std::memcpy(b.ptr, b.produced_data.data(), b.bytes);
                }
                // Check if any output buffer of this consumer kernel acquired NaN poison
                // because the kernel read a PDL buffer before calling eg_pdl_wait(sid).
                bool src_not_all_ff = false;
                for (const auto& b : st.bufs) {
                    if (!eg_emu_pdl_all_ff(b.produced_data.data(), b.bytes)) {
                        src_not_all_ff = true;
                        break;
                    }
                }
                if (src_not_all_ff) {
                    for (const auto& oa : out_args) {
                        bool is_stage_buf = false;
                        for (const auto& [other_sid, other_st] : eg_emu_pdl_stages) {
                            (void)other_sid;
                            if (other_st.phase == EG_PDL_REGISTERED) {
                                for (const auto& sb : other_st.bufs) {
                                    if (sb.ptr == oa.ptr) is_stage_buf = true;
                                }
                            }
                        }
                        if (!is_stage_buf && eg_emu_pdl_has_nan_elem(oa)) {
                            std::ostringstream oss;
                            oss << "consumer of stage " << sid
                                << " read poisoned PDL buffer before calling eg_pdl_wait(" << sid << ")";
                            eg_emu_pdl_record_violation_locked(sid, 0, 0, oss.str());
                        }
                    }
                }
                st.phase = EG_PDL_DELIVERED;
            } else {
                std::ostringstream oss;
                oss << "consumer kernel read stage " << sid
                    << " buffer without calling eg_pdl_wait(" << sid
                    << ") on all CTAs (waited=" << st.consumer_ctas_waited
                    << ", missed=" << st.consumer_ctas_missed << ")";
                eg_emu_pdl_record_violation_locked(sid, 0, 0, oss.str());
                for (auto& b : st.bufs) {
                    std::memset(b.ptr, 0xff, b.bytes);
                }
            }
        }

        // Finalize stages that were in REGISTERED state when this kernel began.
        for (uint32_t sid : registered_at_start) {
            auto& st = eg_emu_pdl_stages[sid];
            bool triggered = eg_emu_pdl_kernel_triggered_stages.count(sid) > 0;
            bool modified = false;
            for (const auto& b : st.bufs) {
                if (!eg_emu_pdl_all_ff(b.ptr, b.bytes)) {
                    modified = true;
                    break;
                }
            }
            if (triggered || modified) {
                st.expected_producer_ctas = total_ctas;
                for (auto& b : st.bufs) {
                    b.produced_data.assign(b.ptr, b.ptr + b.bytes);
                    std::memset(b.ptr, 0xff, b.bytes);
                }
                if (static_cast<int>(st.triggered_ctas.size()) >= total_ctas) {
                    st.producer_ok = true;
                } else {
                    st.producer_ok = false;
                    std::ostringstream oss;
                    oss << "producer kernel for stage " << sid
                        << " did not call eg_pdl_launch_dependents(" << sid
                        << ") on all CTAs (" << st.triggered_ctas.size()
                        << " of " << total_ctas << " triggered)";
                    eg_emu_pdl_record_violation_locked(sid, 0, 0, oss.str());
                }
                st.phase = EG_PDL_POISONED;
            }
        }
    }
    eg_emu_pdl_flush_violations();
}

template <typename K, typename... Args>
void eg_emu_pdl_launch_cluster(dim3 grid, dim3 block, size_t smem, int cluster, K kernel, Args... args) {
    std::vector<eg_emu_pdl_out_arg> out_args;
    (eg_emu_pdl_collect_out_arg(out_args, args), ...);
    eg_emu_pdl_run(grid, block, smem, cluster, [&] { kernel(args...); }, out_args);
}

template <typename K, typename... Args>
void eg_emu_pdl_launch(dim3 grid, dim3 block, size_t smem, K kernel, Args... args) {
    if constexpr (std::is_pointer_v<std::decay_t<K>>) {
        const void* p = reinterpret_cast<const void*>(kernel);
        if (eg_emu_cluster_kernels.count(p)) {
            auto it = eg_emu_cluster_sizes.find(p);
            int cluster = (it != eg_emu_cluster_sizes.end()) ? it->second : 8;
            eg_emu_pdl_launch_cluster(grid, block, smem, cluster, kernel, args...);
            return;
        }
    }
    std::vector<eg_emu_pdl_out_arg> out_args;
    (eg_emu_pdl_collect_out_arg(out_args, args), ...);
    eg_emu_pdl_run(grid, block, smem, 1, [&] { kernel(args...); }, out_args);
}

#ifndef EG_EMU_PDL_LAUNCH_TYPES
#define EG_EMU_PDL_LAUNCH_TYPES 1
enum cudaLaunchAttributeID {
    cudaLaunchAttributeProgrammaticStreamSerialization = 9,
};
union cudaLaunchAttributeValue {
    int programmaticStreamSerializationAllowed;
};
struct cudaLaunchAttribute {
    cudaLaunchAttributeID id;
    cudaLaunchAttributeValue val;
};
struct cudaLaunchConfig_t {
    dim3 gridDim;
    dim3 blockDim;
    size_t dynamicSmemBytes;
    cudaStream_t stream;
    cudaLaunchAttribute* attrs;
    unsigned int numAttrs;
};
template <typename K, typename... Args>
inline cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* cfg, K kernel, Args... args) {
    eg_emu_pdl_launch(cfg->gridDim, cfg->blockDim, cfg->dynamicSmemBytes, kernel, args...);
    return cudaSuccess;
}
#endif

#define eg_emu_run eg_emu_pdl_run
#define eg_emu_launch eg_emu_pdl_launch
#define eg_emu_launch_cluster eg_emu_pdl_launch_cluster
#undef CUDAHOSTEMU_LAUNCH
#define CUDAHOSTEMU_LAUNCH(kernel, grid, block, smem, stream, ...) eg_emu_pdl_launch(grid, block, smem, kernel __VA_OPT__(,) __VA_ARGS__)

#endif // EG_EMU_PDL_H
