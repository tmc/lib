package tcgen05

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// tmPrelude defines the device helpers shared by the TMEM-resident kernels
// in ssm.go, ttt.go, lora.go and multimodal.go.
//
// On sm_100a, sm_103a and sm_107a (and their f variants) the helpers issue
// tcgen05.alloc, tcgen05.mma.cta_group::1.kind::tf32 with shared-memory
// descriptors (A from shared memory or from TMEM), tcgen05.commit to an
// mbarrier, tcgen05.ld/st 32x32b.x1, and tcgen05.shift.down.
//
// Everywhere else EG_EMU is defined and TMEM is a [128][512] float array:
// per block in host emulation (cudahostemu), and one slice of a global
// array per resident CTA on devices without tcgen05 (sm_121a and older).
// The emulated MMA multiplies in FP32, not TF32. Emulated TMEM starts as
// NaN so a read of a cell nothing wrote shows up in the output.
//
// Operand tiles in shared memory use the canonical no-swizzle K-major
// layout: 8-row by 16-byte core matrices, K-adjacent core matrices 128
// bytes apart (leading byte offset) and adjacent 8-row groups K/4*128 bytes
// apart (stride byte offset). canonicalIndex is its Go twin.
// M is always 128: A tiles are padded with zero rows to 128, and D row m
// lands in TMEM lane m.
const tmPrelude = `#ifndef EG_TCGEN05_TM_PRELUDE
#define EG_TCGEN05_TM_PRELUDE

#if (defined(EG_HOST_EMU) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000 || __CUDA_ARCH__ >= 1200))) && !defined(EG_EMU)
#define EG_EMU 1
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
#define EG_TM_HW 1
#endif

#if defined(EG_HOST_EMU)
#define EG_TM_SHARED(T, name, dims) EG_EMU_SHARED(T, name, dims)
#define EG_TM_BUF(name) EG_EMU_SHARED(float, name##_cells, [128 * 512]); float* const name = name##_cells
#else
#define EG_TM_SHARED(T, name, dims) alignas(16) __shared__ T name dims
#ifndef EG_TM_EMU_CTAS
#define EG_TM_EMU_CTAS 8
#endif
// eg_tm_emu_backing holds emulated TMEM per CTA, indexed by linear block
// index. CTAs whose indices agree modulo EG_TM_EMU_CTAS share cells, so a
// launch must either be at most EG_TM_EMU_CTAS CTAs or define it larger.
// It is declared in every nvcc pass, not only the EG_EMU device pass: the
// host pass registers device globals and must see the same declaration.
static __device__ float eg_tm_emu_backing[EG_TM_EMU_CTAS * 128 * 512];
#if defined(EG_EMU)
#define EG_TM_BUF(name) float* const name = eg_tm_emu_backing + (size_t)((blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) % EG_TM_EMU_CTAS) * 128 * 512
#else
#define EG_TM_BUF(name) float* const name = nullptr
#endif
#endif

// eg_cm_idx returns the float index of element (r, k) of an R x K tile in the
// canonical no-swizzle K-major layout.
EG_DEV int eg_cm_idx(int r, int k, int K) {
    return ((((r >> 3) * (K >> 2)) + (k >> 2)) << 5) | ((r & 7) << 2) | (k & 3);
}

#if defined(EG_TM_HW)
// eg_tm_sdesc encodes a tcgen05 shared-memory descriptor: no swizzle,
// version 1 (bit 46), 16-byte-unit start address, LBO and SBO.
EG_DEV uint64_t eg_tm_sdesc(const float* p, uint32_t lbo, uint32_t sbo) {
    const uint64_t a = (uint32_t)__cvta_generic_to_shared(p);
    return ((a & 0x3ffffu) >> 4) | ((uint64_t)((lbo >> 4) & 0x3fffu) << 16) |
           ((uint64_t)((sbo >> 4) & 0x3fffu) << 32) | (1ull << 46);
}

// eg_tm_idesc_tf32 encodes the kind::tf32 instruction descriptor: F32
// accumulator, TF32 A and B, both K-major, M = 128.
EG_DEV uint32_t eg_tm_idesc_tf32(int n) {
    return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(n >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
}
#endif

// eg_tm_alloc allocates cols TMEM columns and stores the base column in *slot.
// All 32 threads of one warp call it.
EG_DEV void eg_tm_alloc(float* tm, uint32_t* slot, uint32_t cols) {
#if defined(EG_TM_HW)
    (void)tm;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(slot)), "r"(cols) : "memory");
#else
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t i = lane; i < 128u * cols; i += 32u) tm[(i / cols) * 512u + i % cols] = __uint_as_float(0x7fc00000u);
    if (lane == 0) *slot = 0;
#endif
}

// eg_tm_dealloc frees cols TMEM columns at base. All 32 threads of the
// allocating warp call it.
EG_DEV void eg_tm_dealloc(uint32_t base, uint32_t cols) {
#if defined(EG_TM_HW)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(base), "r"(cols) : "memory");
#else
    (void)base;
    (void)cols;
#endif
}

// eg_tm_ld reads TMEM lane row (the calling thread's own lane) at column col
// and waits for the load.
EG_DEV float eg_tm_ld(const float* tm, int row, uint32_t col) {
#if defined(EG_TM_HW)
    (void)tm;
    uint32_t v;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n"
                 "tcgen05.wait::ld.sync.aligned;"
                 : "=r"(v) : "r"(((uint32_t)(row & ~31) << 16) | col) : "memory");
    return __uint_as_float(v);
#else
    return tm[row * 512 + col];
#endif
}

// eg_tm_st writes v to TMEM lane row (the calling thread's own lane) at column col.
EG_DEV void eg_tm_st(float* tm, int row, uint32_t col, float v) {
#if defined(EG_TM_HW)
    (void)tm;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                 :: "r"(((uint32_t)(row & ~31) << 16) | col), "r"(__float_as_uint(v)) : "memory");
#else
    tm[row * 512 + col] = v;
#endif
}

// eg_tm_sync orders this CTA's tcgen05.st and shared-memory operand writes
// before tcgen05 operations issued after it. All threads call it.
EG_DEV void eg_tm_sync() {
    eg_fence_proxy_async();
#if defined(EG_TM_HW)
    asm volatile("tcgen05.wait::st.sync.aligned;\n"
                 "tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#else
    __syncthreads();
#endif
}

EG_DEV void eg_tm_mbar_init(uint64_t* mbar) {
#if defined(EG_TM_HW)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 "fence.mbarrier_init.release.cluster;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(mbar)) : "memory");
#else
    (void)mbar;
#endif
}

// eg_tm_mma_ss issues D[128, n] = (acc ? D : 0) + A[128, k] * B[n, k]^T with
// D at TMEM column d and A, B in shared memory. One thread calls it.
EG_DEV void eg_tm_mma_ss(float* tm, uint32_t d, const float* a, const float* b, int n, int k, int acc) {
#if defined(EG_TM_HW)
    (void)tm;
    const uint32_t idesc = eg_tm_idesc_tf32(n);
    for (int kk = 0; kk < k; kk += 8) {
        const uint64_t da = eg_tm_sdesc(a + (kk >> 2) * 32, 128u, (uint32_t)(k >> 2) * 128u);
        const uint64_t db = eg_tm_sdesc(b + (kk >> 2) * 32, 128u, (uint32_t)(k >> 2) * 128u);
        const int en = (kk > 0 || acc) ? 1 : 0;
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n}"
                     :: "r"(d), "l"(da), "l"(db), "r"(idesc), "r"(en) : "memory");
    }
#else
    for (int m = 0; m < 128; ++m) {
        for (int j = 0; j < n; ++j) {
            float s = acc ? tm[m * 512 + d + j] : 0.0f;
            for (int kk = 0; kk < k; ++kk) s += a[eg_cm_idx(m, kk, k)] * b[eg_cm_idx(j, kk, k)];
            tm[m * 512 + d + j] = s;
        }
    }
#endif
}

// eg_tm_mma_ts is eg_tm_mma_ss with A[128, k] read from TMEM columns
// [a, a + k).
EG_DEV void eg_tm_mma_ts(float* tm, uint32_t d, uint32_t a, const float* b, int n, int k, int acc) {
#if defined(EG_TM_HW)
    (void)tm;
    const uint32_t idesc = eg_tm_idesc_tf32(n);
    for (int kk = 0; kk < k; kk += 8) {
        const uint64_t db = eg_tm_sdesc(b + (kk >> 2) * 32, 128u, (uint32_t)(k >> 2) * 128u);
        const int en = (kk > 0 || acc) ? 1 : 0;
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p;\n}"
                     :: "r"(d), "r"(a + (uint32_t)kk), "l"(db), "r"(idesc), "r"(en) : "memory");
    }
#else
    float row[512];
    for (int m = 0; m < 128; ++m) {
        for (int kk = 0; kk < k; ++kk) row[kk] = tm[m * 512 + a + kk];
        for (int j = 0; j < n; ++j) {
            float s = acc ? tm[m * 512 + d + j] : 0.0f;
            for (int kk = 0; kk < k; ++kk) s += row[kk] * b[eg_cm_idx(j, kk, k)];
            tm[m * 512 + d + j] = s;
        }
    }
#endif
}

// eg_tm_shift_down issues tcgen05.shift.cta_group::1.down for each 8-column
// (32-byte) group in [col, col + ncols): rows 0..126 move down one lane and
// lane 0 keeps its value. One thread calls it.
EG_DEV void eg_tm_shift_down(float* tm, uint32_t col, int ncols) {
#if defined(EG_TM_HW)
    (void)tm;
    for (int c = 0; c < ncols; c += 8) {
        asm volatile("tcgen05.shift.cta_group::1.down [%0];" :: "r"(col + (uint32_t)c) : "memory");
    }
#else
    for (int r = 127; r > 0; --r) {
        for (int c = 0; c < ncols; ++c) tm[r * 512 + col + c] = tm[(r - 1) * 512 + col + c];
    }
#endif
}

// eg_tm_commit makes the mbarrier track completion of this thread's prior
// tcgen05.mma and tcgen05.shift operations. One thread calls it.
EG_DEV void eg_tm_commit(uint64_t* mbar) {
#if defined(EG_TM_HW)
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(mbar)) : "memory");
#else
    (void)mbar;
#endif
}

// eg_tm_wait waits for the mbarrier phase completed by the last eg_tm_commit.
// All threads call it.
EG_DEV void eg_tm_wait(uint64_t* mbar, uint32_t* phase) {
#if defined(EG_TM_HW)
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t done = 0;
    while (!done) {
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(s), "r"(*phase) : "memory");
    }
    *phase ^= 1u;
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#else
    (void)mbar;
    (void)phase;
    __syncthreads();
#endif
}
#endif // EG_TCGEN05_TM_PRELUDE
`

// canonicalIndex is the Go twin of eg_cm_idx: the float index of element
// (r, k) of an R x K tile in the canonical no-swizzle K-major layout.
func canonicalIndex(r, k, K int) int {
	return ((r>>3)*(K>>2)+(k>>2))<<5 | (r&7)<<2 | k&3
}

// checkTMTarget reports whether target is one the tmPrelude kernels emit
// tcgen05 instructions for.
func checkTMTarget(what, target string) error {
	switch target {
	case "sm_100a", "sm_103a", "sm_107a", "sm_100f", "sm_103f", "sm_107f":
		return nil
	}
	return fmt.Errorf("%s: unsupported target %q (want sm_100a, sm_103a, sm_107a or an f variant)", what, target)
}

// allocColumns returns the power-of-two TMEM allocation width covering cols.
func allocColumns(cols int) uint32 {
	n := uint32(32)
	for int(n) < cols {
		n <<= 1
	}
	return n
}

// maxDynamicShared is the dynamic shared memory limit used by these schedules.
// TestMaxDynamicSharedAgreesWithProfiles checks the sm_100a and sm_103a
// profiles. Targets without a profile retain the same planning limit.
var maxDynamicShared = int(mustProfile("sm_100a").SharedPerBlockOptin.Value)

// mustProfile returns the cudadevicetable profile named name. The table is
// embedded, so a miss is a programming error, not a runtime condition.
func mustProfile(name string) cudadevicetable.Profile {
	p, err := cudadevicetable.LookupProfile(name)
	if err != nil {
		panic("tcgen05: " + err.Error())
	}
	return p
}

// tmTrace builds the cudatmem events of a single-CTA kernel that allocates
// one TMEM range from warp 0.
type tmTrace struct {
	events []cudatmem.TraceEvent
	cols   uint32
}

func newTMTrace(cols uint32) *tmTrace {
	t := &tmTrace{cols: cols}
	t.events = append(t.events, cudatmem.TraceEvent{
		Kind: cudatmem.EventAlloc, ActiveThreads: 32, CTAGroup: 1, Columns: cols, Rows: 128,
	})
	return t
}

func (t *tmTrace) mma(col, n uint32, fromTMEM bool, aCol uint32) {
	t.events = append(t.events, cudatmem.TraceEvent{
		Kind: cudatmem.EventMMA, ActiveThreads: 1, CTAGroup: 1, Offset: col, Columns: n, Rows: 128,
		SourceAFromTMEM: fromTMEM, SourceAOffset: aCol,
	})
}

func (t *tmTrace) shift(col, n uint32) {
	t.events = append(t.events, cudatmem.TraceEvent{
		Kind: cudatmem.EventShift, ActiveThreads: 1, CTAGroup: 1, Offset: col, Columns: n, Rows: 128,
	})
}

func (t *tmTrace) commitWait() {
	t.events = append(t.events,
		cudatmem.TraceEvent{Kind: cudatmem.EventCommit, ActiveThreads: 1, CTAGroup: 1},
		cudatmem.TraceEvent{Kind: cudatmem.EventWait, ActiveThreads: 128, CTAGroup: 1},
	)
}

// rows records a tcgen05.ld or tcgen05.st over lanes [0, rows) by the warps
// that own them.
func (t *tmTrace) rows(kind cudatmem.EventKind, col, n uint32, rows int) {
	for w := 0; w*32 < rows; w++ {
		t.events = append(t.events, cudatmem.TraceEvent{
			Kind: kind, Warp: w, ActiveThreads: 32, CTAGroup: 1,
			Offset: col, Columns: n, RowStart: uint32(32 * w), Rows: 32,
		})
	}
}

func (t *tmTrace) done() []cudatmem.TraceEvent {
	return append(t.events, cudatmem.TraceEvent{
		Kind: cudatmem.EventDealloc, ActiveThreads: 32, CTAGroup: 1, Columns: t.cols, Rows: 128,
	})
}
