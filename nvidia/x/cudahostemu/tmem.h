// tmem.h: host-CPU emulation of Blackwell (sm_100a) Tensor Memory (TMEM) and
// 5th-generation tcgen05 instructions with adversarial lifecycle and
// synchronization hazard tracking.
//
// Included after cudahostemu.h when RunWithTMEM compiles kernel source.
#pragma once

#ifndef EG_EMU
#define EG_EMU 1
#endif
#ifndef EG_HOST_EMU
#define EG_HOST_EMU 1
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

constexpr uint32_t eg_emu_tmem_max_rows = 128;
constexpr uint32_t eg_emu_tmem_max_cols = 512;
constexpr uint32_t eg_emu_tmem_nan_bits = 0x7fc00000u;

enum eg_emu_tmem_col_state : uint8_t {
    eg_emu_tmem_col_unallocated = 0,
    eg_emu_tmem_col_allocated_unwritten = 1,
    eg_emu_tmem_col_uncommitted = 2,
    eg_emu_tmem_col_committed_unwaited = 3,
    eg_emu_tmem_col_ready = 4,
};

struct eg_emu_tmem_alloc_range {
    uint32_t addr = 0;
    uint32_t cols = 0;
    bool active = false;
};

struct eg_emu_tmem_cta {
    std::mutex mu;
    int cta_id = 0;
    uint32_t next_col = 0;
    uint32_t leaked_cols = 0;
    bool relinquished = false;
    bool finalized = false;

    uint32_t live_bits[eg_emu_tmem_max_rows][eg_emu_tmem_max_cols];
    float staged[eg_emu_tmem_max_rows][eg_emu_tmem_max_cols];
    eg_emu_tmem_col_state col_state[eg_emu_tmem_max_cols];

    std::vector<eg_emu_tmem_alloc_range> alloc_history;
    std::vector<eg_emu_tmem_alloc_range> dealloc_history;
    int mma_executed_seq = 0;
    int commit_executed_seq = 0;
    int wait_executed_seq = 0;

    int thread_alloc_seq[1024] = {};
    int thread_dealloc_seq[1024] = {};
    int thread_mma_seq[1024] = {};
    int thread_commit_seq[1024] = {};
    int thread_wait_seq[1024] = {};

    explicit eg_emu_tmem_cta(int id) : cta_id(id) {
        for (uint32_t r = 0; r < eg_emu_tmem_max_rows; ++r) {
            for (uint32_t c = 0; c < eg_emu_tmem_max_cols; ++c) {
                live_bits[r][c] = eg_emu_tmem_nan_bits;
                staged[r][c] = __uint_as_float(eg_emu_tmem_nan_bits);
            }
        }
        for (uint32_t c = 0; c < eg_emu_tmem_max_cols; ++c) {
            col_state[c] = eg_emu_tmem_col_unallocated;
        }
    }
};

inline std::mutex eg_emu_tmem_global_mu;
inline int eg_emu_tmem_errors = 0;
inline std::vector<std::string> eg_emu_tmem_error_msgs;
inline std::map<eg_emu_block*, std::shared_ptr<eg_emu_tmem_cta>> eg_emu_tmem_ctas;
inline thread_local eg_emu_tmem_cta* eg_emu_cur_tmem_cta = nullptr;
inline thread_local int eg_emu_tmem_tid = 0;

inline void eg_emu_tmem_record_error(const std::string& msg) {
    std::lock_guard<std::mutex> lock(eg_emu_tmem_global_mu);
    ++eg_emu_tmem_errors;
    eg_emu_tmem_error_msgs.push_back(msg);
}

inline eg_emu_tmem_cta* eg_emu_get_tmem_cta() {
    if (eg_emu_cur_tmem_cta) return eg_emu_cur_tmem_cta;
    std::lock_guard<std::mutex> lock(eg_emu_tmem_global_mu);
    auto& ptr = eg_emu_tmem_ctas[eg_emu_cur_block];
    if (!ptr) {
        int id = static_cast<int>((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
        ptr = std::make_shared<eg_emu_tmem_cta>(id);
    }
    eg_emu_cur_tmem_cta = ptr.get();
    return eg_emu_cur_tmem_cta;
}

inline int eg_emu_tmem_cur_tid() {
    if (eg_emu_tmem_tid >= 0 && eg_emu_tmem_tid < 1024) {
        return eg_emu_tmem_tid;
    }
    int tid = static_cast<int>((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);
    return (tid >= 0 && tid < 1024) ? tid : 0;
}

inline void eg_emu_tmem_finalize_cta(eg_emu_tmem_cta* cta) {
    if (!cta) return;
    uint32_t leaked = 0;
    {
        std::lock_guard<std::mutex> lock(cta->mu);
        if (cta->finalized) return;
        cta->finalized = true;
        leaked = cta->leaked_cols;
    }
    if (leaked > 0) {
        std::ostringstream oss;
        oss << "CTA " << cta->cta_id << " exited with " << leaked
            << " TMEM columns leaked (missing eg_emu_tmem_dealloc)";
        eg_emu_tmem_record_error(oss.str());
    }
}

inline int eg_emu_tmem_error_count() {
    std::vector<std::shared_ptr<eg_emu_tmem_cta>> ctas;
    {
        std::lock_guard<std::mutex> lock(eg_emu_tmem_global_mu);
        for (auto& kv : eg_emu_tmem_ctas) {
            if (kv.second) ctas.push_back(kv.second);
        }
    }
    for (auto& cta : ctas) {
        eg_emu_tmem_finalize_cta(cta.get());
    }
    std::lock_guard<std::mutex> lock(eg_emu_tmem_global_mu);
    return eg_emu_tmem_errors;
}

inline void eg_emu_tmem_reset_errors() {
    std::lock_guard<std::mutex> lock(eg_emu_tmem_global_mu);
    eg_emu_tmem_errors = 0;
    eg_emu_tmem_error_msgs.clear();
    eg_emu_tmem_ctas.clear();
}

inline void eg_emu_tmem_flush_errors() {
    (void)eg_emu_tmem_error_count();
#ifdef EG_EMU_TMEM_ERROR_FILE
    std::lock_guard<std::mutex> lock(eg_emu_tmem_global_mu);
    FILE* f = std::fopen(EG_EMU_TMEM_ERROR_FILE, "wb");
    if (!f) return;
    std::fprintf(f, "%d\n", eg_emu_tmem_errors);
    for (const auto& msg : eg_emu_tmem_error_msgs) {
        std::fprintf(f, "%s\n", msg.c_str());
    }
    std::fclose(f);
#endif
}

// eg_emu_tmem_alloc validates num_cols is a power of two in [32, 512] and
// allocates columns from the CTA's 512-column TMEM pool, initializing
// allocated cells to NaN (0x7fc00000) until a tcgen05.mma writes and
// commits+waits them.
inline void eg_emu_tmem_alloc(uint32_t* tmem_addr_out, uint32_t num_cols) {
    eg_emu_tmem_cta* cta = eg_emu_get_tmem_cta();
    const int tid = eg_emu_tmem_cur_tid();
    std::unique_lock<std::mutex> lock(cta->mu);
    const int seq = cta->thread_alloc_seq[tid]++;
    if (seq < static_cast<int>(cta->alloc_history.size())) {
        const auto& prev = cta->alloc_history[seq];
        if (prev.cols == num_cols) {
            if (tmem_addr_out) *tmem_addr_out = prev.addr;
            return;
        }
    }
    if (!tmem_addr_out) {
        lock.unlock();
        eg_emu_tmem_record_error("eg_emu_tmem_alloc: null tmem_addr_out pointer");
        return;
    }
    const bool valid_pow2 = (num_cols >= 32 && num_cols <= 512 && (num_cols & (num_cols - 1)) == 0);
    if (!valid_pow2) {
        *tmem_addr_out = 0xffffffffu;
        lock.unlock();
        std::ostringstream oss;
        oss << "eg_emu_tmem_alloc: invalid num_cols=" << num_cols
            << " (must be a power of two in [32, 512])";
        eg_emu_tmem_record_error(oss.str());
        return;
    }
    if (cta->relinquished) {
        *tmem_addr_out = 0xffffffffu;
        lock.unlock();
        eg_emu_tmem_record_error("eg_emu_tmem_alloc: allocation after permit relinquished");
        return;
    }
    // Search for a free contiguous range of num_cols columns aligned to num_cols (or packed from 0).
    uint32_t found_addr = 0xffffffffu;
    for (uint32_t base = 0; base + num_cols <= eg_emu_tmem_max_cols; base += 32) {
        bool free_range = true;
        for (uint32_t c = base; c < base + num_cols; ++c) {
            if (cta->col_state[c] != eg_emu_tmem_col_unallocated) {
                free_range = false;
                break;
            }
        }
        if (free_range) {
            found_addr = base;
            break;
        }
    }
    if (found_addr == 0xffffffffu) {
        *tmem_addr_out = 0xffffffffu;
        lock.unlock();
        std::ostringstream oss;
        oss << "eg_emu_tmem_alloc: exhausted 512-column TMEM pool requesting " << num_cols << " columns";
        eg_emu_tmem_record_error(oss.str());
        return;
    }
    for (uint32_t r = 0; r < eg_emu_tmem_max_rows; ++r) {
        for (uint32_t c = found_addr; c < found_addr + num_cols; ++c) {
            cta->live_bits[r][c] = eg_emu_tmem_nan_bits;
            cta->staged[r][c] = __uint_as_float(eg_emu_tmem_nan_bits);
        }
    }
    for (uint32_t c = found_addr; c < found_addr + num_cols; ++c) {
        cta->col_state[c] = eg_emu_tmem_col_allocated_unwritten;
    }
    cta->leaked_cols += num_cols;
    cta->alloc_history.push_back({found_addr, num_cols, true});
    *tmem_addr_out = found_addr;
}

// eg_emu_tmem_dealloc frees the allocated TMEM range [tmem_addr, tmem_addr + num_cols).
inline void eg_emu_tmem_dealloc(uint32_t tmem_addr, uint32_t num_cols) {
    eg_emu_tmem_cta* cta = eg_emu_get_tmem_cta();
    const int tid = eg_emu_tmem_cur_tid();
    std::unique_lock<std::mutex> lock(cta->mu);
    const int seq = cta->thread_dealloc_seq[tid]++;
    if (seq < static_cast<int>(cta->dealloc_history.size())) {
        const auto& prev = cta->dealloc_history[seq];
        if (prev.addr == tmem_addr && prev.cols == num_cols) {
            return;
        }
    }
    int match_idx = -1;
    for (int i = 0; i < static_cast<int>(cta->alloc_history.size()); ++i) {
        if (cta->alloc_history[i].active &&
            cta->alloc_history[i].addr == tmem_addr &&
            cta->alloc_history[i].cols == num_cols) {
            match_idx = i;
            break;
        }
    }
    if (match_idx < 0) {
        lock.unlock();
        std::ostringstream oss;
        oss << "eg_emu_tmem_dealloc: invalid or mismatched deallocation (addr="
            << tmem_addr << ", num_cols=" << num_cols << ")";
        eg_emu_tmem_record_error(oss.str());
        return;
    }
    cta->alloc_history[match_idx].active = false;
    cta->dealloc_history.push_back({tmem_addr, num_cols, false});
    for (uint32_t r = 0; r < eg_emu_tmem_max_rows; ++r) {
        for (uint32_t c = tmem_addr; c < tmem_addr + num_cols; ++c) {
            cta->live_bits[r][c] = eg_emu_tmem_nan_bits;
            cta->staged[r][c] = __uint_as_float(eg_emu_tmem_nan_bits);
        }
    }
    for (uint32_t c = tmem_addr; c < tmem_addr + num_cols; ++c) {
        cta->col_state[c] = eg_emu_tmem_col_unallocated;
    }
    if (cta->leaked_cols >= num_cols) {
        cta->leaked_cols -= num_cols;
    } else {
        cta->leaked_cols = 0;
    }
}

inline void eg_emu_tmem_relinquish_alloc_permit() {
    eg_emu_tmem_cta* cta = eg_emu_get_tmem_cta();
    std::lock_guard<std::mutex> lock(cta->mu);
    cta->relinquished = true;
}

inline bool eg_emu_tmem_validate_mma_locked(eg_emu_tmem_cta* cta, uint32_t tmem_d, int M, int N, int K,
                                            const char* op, std::string* err_out) {
    if (M <= 0 || M > static_cast<int>(eg_emu_tmem_max_rows) ||
        N <= 0 || N > static_cast<int>(eg_emu_tmem_max_cols) || K <= 0 ||
        tmem_d + static_cast<uint32_t>(N) > eg_emu_tmem_max_cols) {
        std::ostringstream oss;
        oss << op << ": invalid tile dimensions M=" << M << " N=" << N << " K=" << K
            << " at tmem_d=" << tmem_d;
        *err_out = oss.str();
        return false;
    }
    for (uint32_t c = tmem_d; c < tmem_d + static_cast<uint32_t>(N); ++c) {
        if (cta->col_state[c] == eg_emu_tmem_col_unallocated) {
            std::ostringstream oss;
            oss << op << ": TMEM column " << c << " in [" << tmem_d << ", " << (tmem_d + N)
                << ") is not allocated";
            *err_out = oss.str();
            return false;
        }
    }
    return true;
}

// eg_emu_tcgen05_mma_bf16 computes D[M, N] = (accum_enable ? D : 0) + A[M, K] * B[N, K]^T
// for BF16 shared-memory tiles into the staged TMEM buffer and poisons live TMEM
// at [tmem_d, tmem_d + N) with NaN until eg_emu_tcgen05_commit() + eg_emu_tcgen05_wait().
inline void eg_emu_tcgen05_mma_bf16(uint32_t tmem_d, const void* smem_a, const void* smem_b,
                                    int M, int N, int K, int accum_enable) {
    eg_emu_tmem_cta* cta = eg_emu_get_tmem_cta();
    const int tid = eg_emu_tmem_cur_tid();
    std::unique_lock<std::mutex> lock(cta->mu);
    const int seq = cta->thread_mma_seq[tid]++;
    if (seq < cta->mma_executed_seq) {
        return;
    }
    cta->mma_executed_seq++;
    if (!smem_a || !smem_b) {
        lock.unlock();
        eg_emu_tmem_record_error("eg_emu_tcgen05_mma_bf16: null shared-memory operand pointer");
        return;
    }
    std::string err;
    if (!eg_emu_tmem_validate_mma_locked(cta, tmem_d, M, N, K, "eg_emu_tcgen05_mma_bf16", &err)) {
        lock.unlock();
        eg_emu_tmem_record_error(err);
        return;
    }
    const uint16_t* a = static_cast<const uint16_t*>(smem_a);
    const uint16_t* b = static_cast<const uint16_t*>(smem_b);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const uint32_t col = tmem_d + static_cast<uint32_t>(n);
            float acc = accum_enable ? cta->staged[m][col] : 0.0f;
            for (int k = 0; k < K; ++k) {
                const float av = eg_emu_bf16(a[m * K + k], 0);
                const float bv = eg_emu_bf16(b[n * K + k], 0);
                acc += av * bv;
            }
            cta->staged[m][col] = acc;
            cta->live_bits[m][col] = eg_emu_tmem_nan_bits;
        }
    }
    for (uint32_t r = static_cast<uint32_t>(M); r < eg_emu_tmem_max_rows; ++r) {
        for (int n = 0; n < N; ++n) {
            cta->live_bits[r][tmem_d + static_cast<uint32_t>(n)] = eg_emu_tmem_nan_bits;
        }
    }
    for (int n = 0; n < N; ++n) {
        cta->col_state[tmem_d + static_cast<uint32_t>(n)] = eg_emu_tmem_col_uncommitted;
    }
}

// eg_emu_tcgen05_mma_e4m3 computes D[M, N] = (accum_enable ? D : 0) + A[M, K] * B[N, K]^T
// for E4M3 (FP8) shared-memory tiles into the staged TMEM buffer and poisons live TMEM
// at [tmem_d, tmem_d + N) with NaN until eg_emu_tcgen05_commit() + eg_emu_tcgen05_wait().
inline void eg_emu_tcgen05_mma_e4m3(uint32_t tmem_d, const void* smem_a, const void* smem_b,
                                    int M, int N, int K, int accum_enable) {
    eg_emu_tmem_cta* cta = eg_emu_get_tmem_cta();
    const int tid = eg_emu_tmem_cur_tid();
    std::unique_lock<std::mutex> lock(cta->mu);
    const int seq = cta->thread_mma_seq[tid]++;
    if (seq < cta->mma_executed_seq) {
        return;
    }
    cta->mma_executed_seq++;
    if (!smem_a || !smem_b) {
        lock.unlock();
        eg_emu_tmem_record_error("eg_emu_tcgen05_mma_e4m3: null shared-memory operand pointer");
        return;
    }
    std::string err;
    if (!eg_emu_tmem_validate_mma_locked(cta, tmem_d, M, N, K, "eg_emu_tcgen05_mma_e4m3", &err)) {
        lock.unlock();
        eg_emu_tmem_record_error(err);
        return;
    }
    const uint8_t* a = static_cast<const uint8_t*>(smem_a);
    const uint8_t* b = static_cast<const uint8_t*>(smem_b);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const uint32_t col = tmem_d + static_cast<uint32_t>(n);
            float acc = accum_enable ? cta->staged[m][col] : 0.0f;
            for (int k = 0; k < K; ++k) {
                const float av = eg_emu_e4m3(a[m * K + k]);
                const float bv = eg_emu_e4m3(b[n * K + k]);
                acc += av * bv;
            }
            cta->staged[m][col] = acc;
            cta->live_bits[m][col] = eg_emu_tmem_nan_bits;
        }
    }
    for (uint32_t r = static_cast<uint32_t>(M); r < eg_emu_tmem_max_rows; ++r) {
        for (int n = 0; n < N; ++n) {
            cta->live_bits[r][tmem_d + static_cast<uint32_t>(n)] = eg_emu_tmem_nan_bits;
        }
    }
    for (int n = 0; n < N; ++n) {
        cta->col_state[tmem_d + static_cast<uint32_t>(n)] = eg_emu_tmem_col_uncommitted;
    }
}

// eg_emu_tcgen05_mma_nvfp4 computes block-scaled NVFP4 (E2M1 with E4M3 block-16 scales)
// D[M, N] = (accum_enable ? D : 0) + A[M, K] * B[N, K]^T into the staged TMEM buffer
// and poisons live TMEM at [tmem_d, tmem_d + N) with NaN until commit + wait.
inline void eg_emu_tcgen05_mma_nvfp4(uint32_t tmem_d,
                                     const uint8_t* smem_a_fp4, const uint8_t* smem_a_scale,
                                     const uint8_t* smem_b_fp4, const uint8_t* smem_b_scale,
                                     int M, int N, int K, int accum_enable) {
    eg_emu_tmem_cta* cta = eg_emu_get_tmem_cta();
    const int tid = eg_emu_tmem_cur_tid();
    std::unique_lock<std::mutex> lock(cta->mu);
    const int seq = cta->thread_mma_seq[tid]++;
    if (seq < cta->mma_executed_seq) {
        return;
    }
    cta->mma_executed_seq++;
    if (!smem_a_fp4 || !smem_b_fp4 || K % 16 != 0) {
        lock.unlock();
        eg_emu_tmem_record_error("eg_emu_tcgen05_mma_nvfp4: null fp4 pointer or K not a multiple of 16");
        return;
    }
    std::string err;
    if (!eg_emu_tmem_validate_mma_locked(cta, tmem_d, M, N, K, "eg_emu_tcgen05_mma_nvfp4", &err)) {
        lock.unlock();
        eg_emu_tmem_record_error(err);
        return;
    }
    const int k_packed = K / 2;
    const int k_scales = K / 16;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const uint32_t col = tmem_d + static_cast<uint32_t>(n);
            float acc = accum_enable ? cta->staged[m][col] : 0.0f;
            for (int k = 0; k < K; ++k) {
                const uint8_t a_byte = smem_a_fp4[m * k_packed + (k >> 1)];
                const uint8_t b_byte = smem_b_fp4[n * k_packed + (k >> 1)];
                const uint32_t a_code = (a_byte >> (4 * (k & 1))) & 0xfu;
                const uint32_t b_code = (b_byte >> (4 * (k & 1))) & 0xfu;
                const float a_sf = smem_a_scale ? eg_emu_e4m3(smem_a_scale[m * k_scales + (k >> 4)]) : 1.0f;
                const float b_sf = smem_b_scale ? eg_emu_e4m3(smem_b_scale[n * k_scales + (k >> 4)]) : 1.0f;
                const float av = eg_emu_e2m1(a_code) * a_sf;
                const float bv = eg_emu_e2m1(b_code) * b_sf;
                acc += av * bv;
            }
            cta->staged[m][col] = acc;
            cta->live_bits[m][col] = eg_emu_tmem_nan_bits;
        }
    }
    for (uint32_t r = static_cast<uint32_t>(M); r < eg_emu_tmem_max_rows; ++r) {
        for (int n = 0; n < N; ++n) {
            cta->live_bits[r][tmem_d + static_cast<uint32_t>(n)] = eg_emu_tmem_nan_bits;
        }
    }
    for (int n = 0; n < N; ++n) {
        cta->col_state[tmem_d + static_cast<uint32_t>(n)] = eg_emu_tmem_col_uncommitted;
    }
}

inline void eg_emu_tcgen05_mma_nvfp4(uint32_t tmem_d,
                                     const void* smem_a_fp4, const void* smem_a_scale,
                                     const void* smem_b_fp4, const void* smem_b_scale,
                                     int M, int N, int K, int accum_enable) {
    eg_emu_tcgen05_mma_nvfp4(tmem_d,
                             static_cast<const uint8_t*>(smem_a_fp4),
                             static_cast<const uint8_t*>(smem_a_scale),
                             static_cast<const uint8_t*>(smem_b_fp4),
                             static_cast<const uint8_t*>(smem_b_scale),
                             M, N, K, accum_enable);
}

// eg_emu_tcgen05_commit commits all pending uncommitted tcgen05.mma operations
// in the CTA. Cells remain poisoned in live TMEM until eg_emu_tcgen05_wait().
inline void eg_emu_tcgen05_commit() {
    eg_emu_tmem_cta* cta = eg_emu_get_tmem_cta();
    const int tid = eg_emu_tmem_cur_tid();
    std::lock_guard<std::mutex> lock(cta->mu);
    const int seq = cta->thread_commit_seq[tid]++;
    if (seq < cta->commit_executed_seq) {
        return;
    }
    cta->commit_executed_seq++;
    for (uint32_t c = 0; c < eg_emu_tmem_max_cols; ++c) {
        if (cta->col_state[c] == eg_emu_tmem_col_uncommitted) {
            cta->col_state[c] = eg_emu_tmem_col_committed_unwaited;
        }
    }
}

// eg_emu_tcgen05_wait waits for committed tcgen05.mma operations to retire,
// publishing their staged FP32 accumulators into live TMEM.
inline void eg_emu_tcgen05_wait() {
    eg_emu_tmem_cta* cta = eg_emu_get_tmem_cta();
    const int tid = eg_emu_tmem_cur_tid();
    std::lock_guard<std::mutex> lock(cta->mu);
    const int seq = cta->thread_wait_seq[tid]++;
    if (seq < cta->wait_executed_seq) {
        return;
    }
    cta->wait_executed_seq++;
    for (uint32_t c = 0; c < eg_emu_tmem_max_cols; ++c) {
        if (cta->col_state[c] == eg_emu_tmem_col_committed_unwaited) {
            for (uint32_t r = 0; r < eg_emu_tmem_max_rows; ++r) {
                cta->live_bits[r][c] = __float_as_uint(cta->staged[r][c]);
            }
            cta->col_state[c] = eg_emu_tmem_col_ready;
        }
    }
}

// eg_emu_tcgen05_ld_f32 loads count float32 accumulators for lane row from
// TMEM columns [tmem_addr + col_offset, tmem_addr + col_offset + count).
// If called before eg_emu_tcgen05_wait(), it reads the poisoned NaN and
// increments eg_emu_tmem_error_count().
inline void eg_emu_tcgen05_ld_f32(float* reg_out, uint32_t tmem_addr, int row, int col_offset, int count) {
    if (!reg_out || count <= 0) return;
    eg_emu_tmem_cta* cta = eg_emu_get_tmem_cta();
    std::unique_lock<std::mutex> lock(cta->mu);
    const uint32_t start_col = tmem_addr + static_cast<uint32_t>(col_offset);
    if (row < 0 || row >= static_cast<int>(eg_emu_tmem_max_rows) ||
        col_offset < 0 || start_col + static_cast<uint32_t>(count) > eg_emu_tmem_max_cols) {
        for (int i = 0; i < count; ++i) {
            reg_out[i] = __uint_as_float(eg_emu_tmem_nan_bits);
        }
        lock.unlock();
        std::ostringstream oss;
        oss << "eg_emu_tcgen05_ld_f32: out-of-bounds access (row=" << row
            << ", tmem_addr=" << tmem_addr << ", col_offset=" << col_offset
            << ", count=" << count << ")";
        eg_emu_tmem_record_error(oss.str());
        return;
    }
    bool hazard = false;
    uint32_t bad_col = start_col;
    eg_emu_tmem_col_state bad_state = eg_emu_tmem_col_ready;
    for (int i = 0; i < count; ++i) {
        const uint32_t c = start_col + static_cast<uint32_t>(i);
        const eg_emu_tmem_col_state st = cta->col_state[c];
        if (st != eg_emu_tmem_col_ready && !hazard) {
            hazard = true;
            bad_col = c;
            bad_state = st;
        }
        reg_out[i] = __uint_as_float(cta->live_bits[row][c]);
    }
    lock.unlock();
    if (hazard) {
        std::ostringstream oss;
        oss << "CTA " << cta->cta_id << " row " << row
            << " eg_emu_tcgen05_ld_f32 read TMEM column " << bad_col
            << " before eg_emu_tcgen05_wait() (state=" << static_cast<int>(bad_state) << ")";
        eg_emu_tmem_record_error(oss.str());
    }
}

// --- TMEM-aware kernel runner ----------------------------------------------

template <typename F>
void eg_emu_tmem_run(dim3 grid, dim3 block, size_t smem, int cluster, F body) {
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
        std::vector<std::unique_ptr<eg_emu_block>> blocks;
        std::vector<std::shared_ptr<eg_emu_tmem_cta>> tctas;
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
            auto tcta = std::make_shared<eg_emu_tmem_cta>(cta_id);
            {
                std::lock_guard<std::mutex> lock(eg_emu_tmem_global_mu);
                eg_emu_tmem_ctas[&b] = tcta;
            }
            tctas.push_back(tcta);
        }
        std::vector<std::thread> threads;
        for (int r = 0; r < cluster; ++r) {
            for (int i = 0; i < n; ++i) {
                threads.emplace_back([&, r, i] {
                    eg_emu_block& b = *blocks[r];
                    threadIdx = dim3(i % block.x, (i / block.x) % block.y, i / (block.x * block.y));
                    blockIdx = dim3(cx + r, by, bz);
                    eg_emu_cur_block = &b;
                    eg_emu_cur_warp = b.warps[i / 32].get();
                    eg_emu_cur_cluster = &cl;
                    eg_emu_lane = i % 32;
                    eg_emu_rank = r;
                    eg_emu_tmem_tid = i;
                    eg_emu_cur_tmem_cta = tctas[r].get();
                    eg_emu_cp_pending.clear();
                    eg_emu_cp_committed = 0;
                    eg_emu_wg_async = false;
                    eg_emu_wg_pending.clear();
                    eg_emu_wg_accs.clear();
                    eg_emu_wg_committed = 0;

                    body();

                    eg_emu_cur_warp->drop(eg_emu_lane);
                    {
                        std::lock_guard<std::mutex> lock(b.bar_mu);
                        --b.live_threads;
                        if (b.sync_arrived > 0 && b.sync_arrived == b.live_threads) {
                            eg_emu_sync_release(&b, b.sync_arrived);
                        }
                        eg_emu_check_deadlock(&b, "thread exit");
                        eg_emu_release_delayers(&b);
                    }
                    cl.bar.arrive_and_drop();
                });
            }
        }
        for (auto& t : threads) t.join();
        for (int r = 0; r < cluster; ++r) {
            eg_emu_tmem_finalize_cta(tctas[r].get());
        }
    }
    eg_emu_tmem_flush_errors();
}

template <typename K, typename... Args>
void eg_emu_tmem_launch_cluster(dim3 grid, dim3 block, size_t smem, int cluster, K kernel, Args... args) {
    eg_emu_tmem_run(grid, block, smem, cluster, [&] { kernel(args...); });
}

template <typename K, typename... Args>
void eg_emu_tmem_launch(dim3 grid, dim3 block, size_t smem, K kernel, Args... args) {
    if constexpr (std::is_pointer_v<std::decay_t<K>>) {
        const void* p = reinterpret_cast<const void*>(kernel);
        if (eg_emu_cluster_kernels.count(p)) {
            auto it = eg_emu_cluster_sizes.find(p);
            int cluster = (it != eg_emu_cluster_sizes.end()) ? it->second : 8;
            eg_emu_tmem_launch_cluster(grid, block, smem, cluster, kernel, args...);
            return;
        }
    }
    eg_emu_tmem_run(grid, block, smem, 1, [&] { kernel(args...); });
}

#define eg_emu_run eg_emu_tmem_run
#define eg_emu_launch eg_emu_tmem_launch
#define eg_emu_launch_cluster eg_emu_tmem_launch_cluster
#undef CUDAHOSTEMU_LAUNCH
#define CUDAHOSTEMU_LAUNCH(kernel, grid, block, smem, stream, ...) eg_emu_tmem_launch(grid, block, smem, kernel __VA_OPT__(,) __VA_ARGS__)
