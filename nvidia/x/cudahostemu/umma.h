// umma.h: host emulation of Blackwell (sm_100a, sm_103a) tcgen05 matrix
// multiplies as the hardware sees them: through 64-bit shared-memory matrix
// descriptors and 32-bit instruction descriptors, with tensor memory (TMEM)
// written by tcgen05.mma and tcgen05.cp and made visible by tcgen05.commit.
//
// Include it after cudahostemu.h. The descriptor layouts are CUTLASS's
// cute::UMMA::SmemDescriptor, InstrDescriptor and InstrDescriptorBlockScaled
// (cute/arch/mma_sm100_desc.hpp). Shared addresses in descriptors are byte
// offsets from the start of dynamic shared memory, which the emulation treats
// as 1024-byte aligned, as eg_emu_cp_async_bulk_tensor_2d does; the 128-byte
// swizzle is applied to those offsets, so a TMA box landed with the swizzle
// reads back through a SWIZZLE_128B descriptor.
//
// The model is adversarial in the way cudahostemu's mbarriers are:
//
//   - Allocated TMEM reads as NaN until written.
//   - tcgen05.mma computes at issue, reading shared memory then, but its
//     result is invisible: the accumulator columns read as NaN, and a
//     tcgen05.ld of them is an error, until a tcgen05.commit covering the
//     MMA has arrived on an mbarrier and a thread's wait has observed that
//     phase complete.
//   - tcgen05.ld from a warp outside the 32-lane quadrant it addresses is an
//     error (warp w may access lanes [32*(w%4), 32*(w%4)+32)).
//   - Descriptor fields the emulation does not model (MN-major operands,
//     swizzles other than none and 128B, M other than 128, sparse or
//     sub-byte scale ids) are errors, not silently misread.
//
// Errors print and exit with status 3, as the rest of cudahostemu does.
//
// The block-scaled scale-factor placement is a model, not a measurement:
// tcgen05.cp.32x128b.warpx4 copies 32 rows of 16 bytes into lanes r, r+32,
// r+64, r+96 of four columns, and the scale of operand row i, 16-element
// block j of the 64-element MMA K step, is byte j of TMEM cell
// [lane i][scale column + i/32]. That is how CUTLASS's Sm1xxBlockScaledConfig
// SfKMajorAtom lays a 128-row chunk out (byte 16*(i%32) + 4*(i/32) + j) and
// copies it with SM100_UTCCP_4x32dp128bit; whether the tensor core reads it
// that way has not been checked on hardware.
#pragma once

#include <functional>

constexpr int eg_umma_lanes = 128;
constexpr int eg_umma_cols = 512;

enum : uint8_t {
    eg_umma_free = 0,      // not allocated
    eg_umma_live = 1,      // allocated; reads see cell
    eg_umma_pending = 2,   // written by an uncommitted MMA
    eg_umma_committed = 3, // committed, not yet observed complete
};

// An eg_umma_tmem is one CTA's tensor memory. It lives in the block's keyed
// shared storage, zero-filled, so a zero eg_umma_tmem is empty.
struct eg_umma_tmem {
    uint32_t cell[eg_umma_lanes][eg_umma_cols];   // visible bits
    float staged[eg_umma_lanes][eg_umma_cols];    // MMA results, visible or not
    uint8_t state[eg_umma_cols];
    uint8_t relinquished;
};

inline std::mutex eg_umma_mu;

[[noreturn]] inline void eg_umma_fail(const char* op, const std::string& msg) {
    std::fprintf(stderr, "cudahostemu: block (%u,%u,%u) thread %u %s: %s\n",
                 blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, op, msg.c_str());
    std::fflush(stderr);
    std::_Exit(3);
}

inline eg_umma_tmem* eg_umma_cur() {
    return static_cast<eg_umma_tmem*>(eg_emu_cur_block->get("eg_umma_tmem", sizeof(eg_umma_tmem), 0));
}

// eg_emu_umma_alloc is tcgen05.alloc.cta_group::1: ncols, a power of two in
// [32, 512], columns aligned to ncols. One thread stands for the warp.
inline void eg_emu_umma_alloc(uint32_t* slot, uint32_t ncols) {
    if (ncols < 32 || ncols > 512 || (ncols & (ncols - 1)) != 0) {
        eg_umma_fail("tcgen05.alloc", "column count " + std::to_string(ncols) + " is not a power of two in [32, 512]");
    }
    eg_umma_tmem* t = eg_umma_cur();
    std::lock_guard<std::mutex> lock(eg_umma_mu);
    if (t->relinquished) eg_umma_fail("tcgen05.alloc", "allocation after relinquish_alloc_permit");
    for (uint32_t base = 0; base + ncols <= eg_umma_cols; base += ncols) {
        bool free = true;
        for (uint32_t c = base; c < base + ncols && free; ++c) free = t->state[c] == eg_umma_free;
        if (!free) continue;
        for (uint32_t c = base; c < base + ncols; ++c) {
            t->state[c] = eg_umma_live;
            for (int r = 0; r < eg_umma_lanes; ++r) {
                t->cell[r][c] = 0x7fc00000u;
                t->staged[r][c] = __uint_as_float(0x7fc00000u);
            }
        }
        *slot = base;
        return;
    }
    eg_umma_fail("tcgen05.alloc", "no " + std::to_string(ncols) + " free columns");
}

inline void eg_emu_umma_relinquish() {
    eg_umma_tmem* t = eg_umma_cur();
    std::lock_guard<std::mutex> lock(eg_umma_mu);
    t->relinquished = 1;
}

// eg_emu_umma_dealloc is tcgen05.dealloc.cta_group::1.
inline void eg_emu_umma_dealloc(uint32_t taddr, uint32_t ncols) {
    eg_umma_tmem* t = eg_umma_cur();
    std::lock_guard<std::mutex> lock(eg_umma_mu);
    const uint32_t col = taddr & 0xffffu;
    if ((taddr >> 16) != 0 || col + ncols > eg_umma_cols) eg_umma_fail("tcgen05.dealloc", "bad address");
    for (uint32_t c = col; c < col + ncols; ++c) {
        if (t->state[c] == eg_umma_free) eg_umma_fail("tcgen05.dealloc", "column " + std::to_string(c) + " not allocated");
        if (t->state[c] != eg_umma_live) eg_umma_fail("tcgen05.dealloc", "column " + std::to_string(c) + " has an MMA in flight");
        t->state[c] = eg_umma_free;
    }
}

// An eg_umma_operand decodes a K-major shared-memory matrix descriptor.
struct eg_umma_operand {
    uint32_t start, lbo, sbo;
    bool swizzle128;

    // byte returns byte b of the K extent of row r.
    unsigned char byte(int r, int b, const char* op) const {
        uint32_t off;
        if (swizzle128) {
            off = start + (uint32_t)(r / 8) * sbo + (uint32_t)(r % 8) * 128 + (uint32_t)b;
            off ^= ((off >> 7) & 7u) << 4;
        } else {
            off = start + (uint32_t)(r / 8) * sbo + (uint32_t)(b / 16) * lbo + (uint32_t)(r % 8) * 16 + (uint32_t)(b % 16);
        }
        if (off >= eg_emu_cur_block->smem.size()) eg_umma_fail(op, "descriptor addresses shared byte " + std::to_string(off) + ", outside dynamic shared memory");
        return eg_emu_dynamic_smem()[off];
    }
};

inline eg_umma_operand eg_umma_decode_sdesc(uint64_t d, int kbytes, const char* op) {
    eg_umma_operand o;
    o.start = (uint32_t)(d & 0x3fffu) << 4;
    o.lbo = (uint32_t)((d >> 16) & 0x3fffu) << 4;
    o.sbo = (uint32_t)((d >> 32) & 0x3fffu) << 4;
    const uint32_t version = (d >> 46) & 3u, base_offset = (d >> 49) & 7u, lbo_mode = (d >> 52) & 1u, layout = (d >> 61) & 7u;
    if (version != 1) eg_umma_fail(op, "shared descriptor version " + std::to_string(version) + ", want 1");
    if (base_offset != 0 || lbo_mode != 0) eg_umma_fail(op, "shared descriptor base offset or LBO mode set");
    switch (layout) {
    case 0:
        o.swizzle128 = false;
        break;
    case 2:
        o.swizzle128 = true;
        if (o.start % 128 + (uint32_t)kbytes > 128) eg_umma_fail(op, "128B-swizzled K step crosses its 128-byte row");
        break;
    default:
        eg_umma_fail(op, "shared descriptor layout type " + std::to_string(layout) + " is not modeled");
    }
    return o;
}

inline float eg_umma_e2m1(unsigned v) {
    static const float g[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    return (v & 8) ? -g[v & 7] : g[v & 7];
}

inline float eg_umma_ue4m3(unsigned v) {
    const unsigned e = (v >> 3) & 15, m = v & 7;
    if (e == 15 && m == 7) return __uint_as_float(0x7fc00000u);
    return e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + (float)m / 8.0f, (int)e - 7);
}

inline float eg_umma_half(uint16_t h) {
    const unsigned s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1.0f + (float)m / 1024.0f, (int)e - 15);
    return s ? -v : v;
}

enum { eg_umma_kind_f16 = 0, eg_umma_kind_mxf4nvf4 = 1 };

// eg_emu_umma_mma is tcgen05.mma.cta_group::1.kind::f16 (kind
// eg_umma_kind_f16) or .kind::mxf4nvf4.block_scale.scale_vec::4X
// (eg_umma_kind_mxf4nvf4, scale columns sfa and sfb): D[128, N] =
// (accumulate ? D : 0) + A[128, K] · B[N, K]ᵀ, one K step (16 F16/BF16 or 64
// E2M1 elements, 32 bytes of each row), with D at TMEM address d.
inline void eg_emu_umma_mma(int kind, uint32_t d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, int accumulate, uint32_t sfa, uint32_t sfb) {
    const char* op = kind == eg_umma_kind_f16 ? "tcgen05.mma.kind::f16" : "tcgen05.mma.kind::mxf4nvf4";
    const uint32_t c_format = (idesc >> 4) & 3u, a_format = (idesc >> 7) & 7u, b_format = (idesc >> 10) & 7u;
    const bool a_neg = (idesc >> 13) & 1u, b_neg = (idesc >> 14) & 1u;
    const uint32_t a_major = (idesc >> 15) & 1u, b_major = (idesc >> 16) & 1u;
    const int n = (int)((idesc >> 17) & 63u) << 3, m = (int)((idesc >> 24) & 31u) << 4;
    if ((idesc & 7u) != 0) eg_umma_fail(op, "sparse MMA is not modeled");
    if (a_major || b_major) eg_umma_fail(op, "MN-major operands are not modeled");
    if (m != 128) eg_umma_fail(op, "M " + std::to_string(m) + " is not modeled (want 128)");
    if (kind == eg_umma_kind_f16) {
        if (c_format != 1) eg_umma_fail(op, "accumulator format " + std::to_string(c_format) + ", want F32 (1)");
        if (a_format > 1 || b_format != a_format) eg_umma_fail(op, "A/B formats must both be F16 (0) or BF16 (1)");
        if (n < 16 || n > 256 || n % 16 != 0) eg_umma_fail(op, "N " + std::to_string(n) + " invalid for M 128");
        if ((idesc >> 30) != 0) eg_umma_fail(op, "max shift is not modeled");
    } else {
        if (a_format != 1 || b_format != 1) eg_umma_fail(op, "A/B formats must be E2M1 (1)");
        if ((idesc >> 23) & 1u) eg_umma_fail(op, "UE8M0 scales are not modeled");
        if (((idesc >> 4) & 3u) || ((idesc >> 29) & 3u)) eg_umma_fail(op, "nonzero scale-factor ids are not modeled");
        if (idesc >> 31) eg_umma_fail(op, "K 96 is not modeled");
        if (n < 8 || n > 128 || n % 8 != 0) eg_umma_fail(op, "N " + std::to_string(n) + " is not modeled (want a multiple of 8 up to 128)");
        if ((sfa >> 16) || (sfb >> 16)) eg_umma_fail(op, "scale-factor addresses must be lane 0");
    }
    const eg_umma_operand a = eg_umma_decode_sdesc(adesc, 32, op), b = eg_umma_decode_sdesc(bdesc, 32, op);
    const uint32_t dcol = d & 0xffffu;
    if ((d >> 16) != 0 || dcol + (uint32_t)n > eg_umma_cols) eg_umma_fail(op, "bad accumulator address");

    // Gather both operands (and scales) as floats before touching TMEM.
    const int kn = kind == eg_umma_kind_f16 ? 16 : 64;
    std::vector<float> av((size_t)m * kn), bv((size_t)n * kn);
    auto load = [&](const eg_umma_operand& o, int rows, uint32_t fmt, std::vector<float>& v) {
        for (int r = 0; r < rows; ++r) {
            for (int k = 0; k < kn; ++k) {
                float x;
                if (kind == eg_umma_kind_f16) {
                    const uint16_t h = (uint16_t)(o.byte(r, 2 * k, op) | (o.byte(r, 2 * k + 1, op) << 8));
                    x = fmt == 1 ? __uint_as_float((uint32_t)h << 16) : eg_umma_half(h);
                } else {
                    const unsigned q = o.byte(r, k / 2, op);
                    x = eg_umma_e2m1(k % 2 ? q >> 4 : q & 15);
                }
                v[(size_t)r * kn + k] = x;
            }
        }
    };
    load(a, m, a_format, av);
    load(b, n, b_format, bv);

    eg_umma_tmem* t = eg_umma_cur();
    std::lock_guard<std::mutex> lock(eg_umma_mu);
    for (uint32_t c = dcol; c < dcol + (uint32_t)n; ++c) {
        if (t->state[c] == eg_umma_free) eg_umma_fail(op, "accumulator column " + std::to_string(c) + " not allocated");
    }
    if (kind == eg_umma_kind_mxf4nvf4) {
        auto scale = [&](uint32_t col, int row, int j) -> float {
            const uint32_t c = col + (uint32_t)(row / 32);
            if (c >= eg_umma_cols || t->state[c] != eg_umma_live) eg_umma_fail(op, "scale column " + std::to_string(c) + " not written by tcgen05.cp");
            return eg_umma_ue4m3((t->cell[row][c] >> (8 * j)) & 0xffu);
        };
        for (int r = 0; r < m; ++r)
            for (int k = 0; k < kn; ++k) av[(size_t)r * kn + k] *= scale(sfa & 0xffffu, r, k / 16);
        for (int r = 0; r < n; ++r)
            for (int k = 0; k < kn; ++k) bv[(size_t)r * kn + k] *= scale(sfb & 0xffffu, r, k / 16);
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const uint32_t c = dcol + (uint32_t)j;
            float acc = accumulate ? t->staged[i][c] : 0.0f;
            for (int k = 0; k < kn; ++k) {
                float x = av[(size_t)i * kn + k], y = bv[(size_t)j * kn + k];
                if (a_neg) x = -x;
                if (b_neg) y = -y;
                acc += x * y;
            }
            t->staged[i][c] = acc;
        }
    }
    for (uint32_t c = dcol; c < dcol + (uint32_t)n; ++c) {
        t->state[c] = eg_umma_pending;
        for (int r = 0; r < eg_umma_lanes; ++r) t->cell[r][c] = 0x7fc00000u;
    }
}

// eg_emu_umma_cp_32x128b_warpx4 is tcgen05.cp.cta_group::1.32x128b.warpx4:
// 32 rows of 16 bytes, described by the no-swizzle descriptor sdesc, land in
// columns [col, col+4) of lanes r, r+32, r+64 and r+96.
inline void eg_emu_umma_cp_32x128b_warpx4(uint32_t taddr, uint64_t sdesc) {
    const char* op = "tcgen05.cp.32x128b.warpx4";
    const eg_umma_operand s = eg_umma_decode_sdesc(sdesc, 16, op);
    if (s.swizzle128) eg_umma_fail(op, "swizzled source is not modeled");
    const uint32_t col = taddr & 0xffffu;
    if ((taddr >> 16) != 0 || col + 4 > eg_umma_cols) eg_umma_fail(op, "bad TMEM address");
    unsigned char rows[32][16];
    for (int r = 0; r < 32; ++r)
        for (int b = 0; b < 16; ++b) rows[r][b] = s.byte(r, b, op);
    eg_umma_tmem* t = eg_umma_cur();
    std::lock_guard<std::mutex> lock(eg_umma_mu);
    for (uint32_t c = col; c < col + 4; ++c) {
        if (t->state[c] != eg_umma_live) eg_umma_fail(op, "column " + std::to_string(c) + " not allocated or has an MMA in flight");
    }
    for (int r = 0; r < 32; ++r) {
        for (int i = 0; i < 4; ++i) {
            uint32_t w;
            std::memcpy(&w, &rows[r][4 * i], 4);
            for (int q = 0; q < 4; ++q) t->cell[r + 32 * q][col + (uint32_t)i] = w;
        }
    }
}

// eg_emu_umma_commit is tcgen05.commit.cta_group::1.mbarrier::arrive::one:
// it arrives on bar, and the results of this CTA's MMAs issued before it,
// including those an earlier commit covered, become visible when a wait
// observes that arrival's phase complete.
inline void eg_emu_umma_commit(uint64_t* bar) {
    eg_umma_tmem* t = eg_umma_cur();
    std::vector<uint32_t> cols;
    {
        std::lock_guard<std::mutex> lock(eg_umma_mu);
        for (uint32_t c = 0; c < eg_umma_cols; ++c) {
            if (t->state[c] == eg_umma_pending || t->state[c] == eg_umma_committed) {
                t->state[c] = eg_umma_committed;
                cols.push_back(c);
            }
        }
    }
    eg_emu_mbar_arrive_effect(bar, [t, cols] {
        std::lock_guard<std::mutex> lock(eg_umma_mu);
        for (uint32_t c : cols) {
            // A later MMA may have taken the column back to pending; its
            // own commit publishes it.
            if (t->state[c] != eg_umma_committed) continue;
            for (int r = 0; r < eg_umma_lanes; ++r) t->cell[r][c] = __float_as_uint(t->staged[r][c]);
            t->state[c] = eg_umma_live;
        }
    });
}

// eg_emu_umma_ld is tcgen05.ld.sync.aligned.32x32b.xN for the calling
// thread: n consecutive 32-bit columns of lane (taddr>>16) + lane.
inline void eg_emu_umma_ld(uint32_t* r, uint32_t taddr, int n) {
    const char* op = "tcgen05.ld.32x32b";
    const uint32_t lane0 = taddr >> 16, col = taddr & 0xffffu;
    const uint32_t warp = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) / 32;
    if (lane0 != 32 * (warp % 4)) {
        eg_umma_fail(op, "warp " + std::to_string(warp) + " addresses lanes from " + std::to_string(lane0) + ", outside its quadrant");
    }
    eg_umma_tmem* t = eg_umma_cur();
    std::lock_guard<std::mutex> lock(eg_umma_mu);
    for (int i = 0; i < n; ++i) {
        const uint32_t c = col + (uint32_t)i;
        if (c >= eg_umma_cols || t->state[c] == eg_umma_free) eg_umma_fail(op, "column " + std::to_string(c) + " not allocated");
        if (t->state[c] != eg_umma_live) {
            eg_umma_fail(op, "column " + std::to_string(c) + " read before the commit of the MMA writing it was waited on");
        }
        r[i] = t->cell[lane0 + (uint32_t)eg_emu_lane][c];
    }
}
