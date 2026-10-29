package hopper

import (
	"fmt"
	"math"
	"strings"
)

// ummaHelpers are the device and host helpers of [UMMAGEMM]: mbarriers, 2D
// TMA loads and tensor-map encoding, tensor-memory allocation, sm_100
// shared-memory and instruction descriptors, tcgen05.mma, tcgen05.cp,
// tcgen05.commit and tcgen05.ld. Under EG_HOST_EMU each one is the
// descriptor-level emulation in cudahostemu's umma.h.
//
// The descriptor layouts follow CUTLASS cute::UMMA (cute/arch/mma_sm100_desc.hpp):
// a shared descriptor holds the start address, leading and stride byte
// offsets in 16-byte units at bits 0, 16 and 32, version 1 at bit 46 and the
// layout type at bit 61 (2 is the 128-byte swizzle); an instruction
// descriptor holds the accumulator format at bit 4 (kind::f16 only), A and B
// formats at bits 7 and 10, N/8 at bit 17 and M/16 at bit 24.
const ummaHelpers = `#ifndef EG_UMMA_HELPERS
#define EG_UMMA_HELPERS
#if defined(EG_HOST_EMU)
#include "umma.h"
typedef eg_emu_tensor_map eg_tmap;
#define EG_TMAP const eg_tmap
#else
#include <cuda.h>
typedef CUtensorMap eg_tmap;
#define EG_TMAP const __grid_constant__ eg_tmap
#endif

// eg_umma_saddr returns p's address in the shared window, as descriptors
// and mbarrier operands take it.
EG_DEV uint32_t eg_umma_saddr(const void* p) {
#if defined(EG_HOST_EMU)
    return (uint32_t)((const unsigned char*)p - eg_emu_dynamic_smem());
#else
    return (uint32_t)__cvta_generic_to_shared(p);
#endif
}

// eg_umma_align1024 rounds p up to a 1024-byte aligned shared address, as
// the 128-byte swizzle needs.
EG_DEV unsigned char* eg_umma_align1024(unsigned char* p) {
    return p + ((1024u - (eg_umma_saddr(p) & 1023u)) & 1023u);
}

EG_DEV void eg_umma_bar_init(uint64_t* bar, uint32_t count) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_init(bar, (int)count);
#else
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(eg_umma_saddr(bar)), "r"(count) : "memory");
#endif
}

// eg_umma_bar_fence_init makes initialized mbarriers visible to the async proxy.
EG_DEV void eg_umma_bar_fence_init() {
#if !defined(EG_HOST_EMU)
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

// eg_umma_bar_expect arrives on bar and adds bytes to the transaction bytes
// its phase waits for.
EG_DEV void eg_umma_bar_expect(uint64_t* bar, uint32_t bytes) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_arrive_expect_tx(bar, bytes);
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(eg_umma_saddr(bar)), "r"(bytes) : "memory");
#endif
}

// eg_umma_bar_wait returns once the phase of bar with parity parity has completed.
EG_DEV void eg_umma_bar_wait(uint64_t* bar, uint32_t parity) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_wait(bar, (int)parity);
#else
    asm volatile(
        "{\n.reg .pred p;\nEG_UMMA_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra EG_UMMA_WAIT_%=;\n}\n" :: "r"(eg_umma_saddr(bar)), "r"(parity) : "memory");
#endif
}

// eg_umma_tma loads the box of map at element (c0, c1) into dst, completing
// its bytes on bar.
EG_DEV void eg_umma_tma(void* dst, const eg_tmap* map, int c0, int c1, uint64_t* bar) {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async_bulk_tensor_2d(dst, map, c0, c1, bar);
#else
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(eg_umma_saddr(dst)), "l"(map), "r"(c0), "r"(c1), "r"(eg_umma_saddr(bar)) : "memory");
#endif
}

// eg_umma_alloc allocates cols tensor-memory columns, storing the base
// address in *slot, and gives up the right to allocate more. One warp calls it.
EG_DEV void eg_umma_alloc(uint32_t* slot, uint32_t cols) {
#if defined(EG_HOST_EMU)
    if ((threadIdx.x & 31) == 0) {
        eg_emu_umma_alloc(slot, cols);
        eg_emu_umma_relinquish();
    }
#else
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 :: "r"(eg_umma_saddr(slot)), "r"(cols) : "memory");
#endif
}

// eg_umma_dealloc frees the cols columns at taddr. One warp calls it.
EG_DEV void eg_umma_dealloc(uint32_t taddr, uint32_t cols) {
#if defined(EG_HOST_EMU)
    if ((threadIdx.x & 31) == 0) eg_emu_umma_dealloc(taddr, cols);
#else
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(cols) : "memory");
#endif
}

// eg_umma_fence_before and eg_umma_fence_after order tcgen05 operations
// around a thread synchronization.
EG_DEV void eg_umma_fence_before() {
#if !defined(EG_HOST_EMU)
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
#endif
}

EG_DEV void eg_umma_fence_after() {
#if !defined(EG_HOST_EMU)
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#endif
}

// eg_umma_desc_sw128 describes a K-major operand at p, 1024-byte aligned up
// to a K step's byte offset within the row: rows of 128 bytes with the
// 128-byte swizzle, 8-row groups 1024 bytes apart.
EG_DEV uint64_t eg_umma_desc_sw128(const void* p) {
    return (uint64_t)((eg_umma_saddr(p) >> 4) & 0x3fffu) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61);
}

// eg_umma_desc_plain describes an unswizzled K-major operand at p: 8-row by
// 16-byte core matrices, lbo bytes apart along K and sbo bytes apart along rows.
EG_DEV uint64_t eg_umma_desc_plain(const void* p, uint32_t lbo, uint32_t sbo) {
    return (uint64_t)((eg_umma_saddr(p) >> 4) & 0x3fffu) | ((uint64_t)((lbo >> 4) & 0x3fffu) << 16) |
           ((uint64_t)((sbo >> 4) & 0x3fffu) << 32) | (1ull << 46);
}

// eg_umma_f16 issues D (+)= A·Bᵀ, one 16-element K step of BF16 or F16
// operands, FP32 accumulator in tensor memory at d. One thread calls it.
EG_DEV void eg_umma_f16(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
#if defined(EG_HOST_EMU)
    eg_emu_umma_mma(eg_umma_kind_f16, d, a, b, idesc, (int)accumulate, 0, 0);
#else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                 :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
#endif
}

// eg_umma_nvfp4 issues D (+)= (A·sfa)·(B·sfb)ᵀ, one 64-element K step of
// E2M1 operands with UE4M3 scales per 16 elements in tensor memory at sfa
// and sfb. One thread calls it.
EG_DEV void eg_umma_nvfp4(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate, uint32_t sfa, uint32_t sfb) {
#if defined(EG_HOST_EMU)
    eg_emu_umma_mma(eg_umma_kind_mxf4nvf4, d, a, b, idesc, (int)accumulate, sfa, sfb);
#else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p;\n}"
                 :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(accumulate), "r"(sfa), "r"(sfb) : "memory");
#endif
}

// eg_umma_cp_scales copies a 128-row scale chunk, 32 rows of 16 bytes
// described by s, into four columns at taddr of every lane quadrant.
EG_DEV void eg_umma_cp_scales(uint32_t taddr, uint64_t s) {
#if defined(EG_HOST_EMU)
    eg_emu_umma_cp_32x128b_warpx4(taddr, s);
#else
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s) : "memory");
#endif
}

// eg_umma_commit arrives on bar once this thread's earlier tcgen05
// operations complete.
EG_DEV void eg_umma_commit(uint64_t* bar) {
#if defined(EG_HOST_EMU)
    eg_emu_umma_commit(bar);
#else
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(eg_umma_saddr(bar)) : "memory");
#endif
}

// eg_umma_ld16 loads 16 consecutive 32-bit columns of the calling thread's
// tensor-memory lane and waits for them. The warp's 32 threads call it.
EG_DEV void eg_umma_ld16(uint32_t r[16], uint32_t taddr) {
#if defined(EG_HOST_EMU)
    eg_emu_umma_ld(r, taddr, 16);
#else
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
                 "tcgen05.wait::ld.sync.aligned;"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(taddr) : "memory");
#endif
}

// eg_umma_e2m1 rounds f to the nearest E2M1 code, ties to even, saturating.
EG_DEV uint32_t eg_umma_e2m1(float f) {
    const float a = fabsf(f);
    const uint32_t c = a <= 0.25f ? 0u : a < 0.75f ? 1u : a <= 1.25f ? 2u : a < 1.75f ? 3u : a <= 2.5f ? 4u : a < 3.5f ? 5u : a <= 5.0f ? 6u : 7u;
    return c | (f < 0.0f && c != 0u ? 8u : 0u);
}

// eg_umma_tmap_2d encodes a 2D tensor map over a row-major [outer, inner]
// tensor of elem_bytes elements (BF16 for 2, bytes for 1), rows stride
// bytes apart, read in boxes of box_inner by box_outer elements.
static inline eg_tmap eg_umma_tmap_2d(const void* base, int elem_bytes, uint64_t inner, uint64_t outer, uint64_t stride,
                                      uint32_t box_inner, uint32_t box_outer, bool swizzle128) {
#if defined(EG_HOST_EMU)
    return eg_emu_make_tensor_map_2d(base, (uint32_t)elem_bytes, inner, outer, stride, box_inner, box_outer, swizzle128);
#else
    typedef CUresult (*encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static encode_fn encode = nullptr;
    if (!encode) {
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&encode, 12000, cudaEnableDefault, &q);
    }
    CUtensorMap m;
    const cuuint64_t dims[2] = {inner, outer}, strides[1] = {stride};
    const cuuint32_t box[2] = {box_inner, box_outer}, elem[2] = {1, 1};
    encode(&m, elem_bytes == 2 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base),
           dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
           swizzle128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return m;
#endif
}
#endif // EG_UMMA_HELPERS
`

// UMMAGEMM computes out[m, n] = Epilogue(Σ_k x[m, k] · W[n, k], extras...)
// on the datacenter Blackwell tensor cores of sm_100a and sm_103a with
// tcgen05.mma.cta_group::1, accumulating in tensor memory (TMEM).
//
// One CTA of four warps computes a 128-row by NQ-column accumulator tile.
// Warp 0's first lane loads K steps of both operands with 2D TMA, 128 bytes
// of K per row with the 128-byte swizzle, into a ring of Stages stages, each
// guarded by a full and an empty mbarrier. Warp 1's first lane issues four
// MMAs per stage and commits each stage back to its empty barrier, and the
// last to an accumulator barrier, on which all four warps wait before each
// reads its 32-lane quadrant of TMEM with tcgen05.ld and applies the
// epilogue.
//
// With MaxM in 1..128 the GEMM has the decode form: A is a 128-row tile of
// W and B is all of x, padded to NQ = MaxM rounded up to 16 rows (TMA fills
// the rows past m with zeros), so a CTA covers 128 columns of out and every
// row of it, and the epilogue's stores are coalesced across lanes. With
// MaxM 0 it has the prefill form: A is 128 rows of x and B is BN rows of W,
// with a grid over both.
//
// With WeightNVFP4 the operands are E2M1 with UE4M3 scales per 16 elements:
// both W (w, w_sf, w_gs as for [SkinnyGEMM]) and x, which a [NVFP4Quant]
// kernel produces (xq, xsf) from BF16 rows. The MMA is
// kind::mxf4nvf4.block_scale.scale_vec::4X, 64 elements per K step and 256
// per stage; each stage's scales are loaded by TMA too, rearranged by warp 1
// into CUTLASS's 128-row chunk layout and copied into TMEM with
// tcgen05.cp.32x128b.warpx4, double-buffered. The accumulator is scaled by
// w_gs before the epilogue. This is W4A4: it quantizes activations, which
// the W4A16 [SkinnyGEMM] and [TileGEMM] forms do not.
//
// N must be a multiple of 128 (decode form) or BN (prefill form), and K a
// multiple of 64 (BF16) or 256 (NVFP4). The launcher encodes its tensor maps
// on every call with cuTensorMapEncodeTiled, through the runtime's driver
// entry point.
type UMMAGEMM struct {
	Name     string
	N, K     int
	Weight   WeightFormat // WeightBF16 or WeightNVFP4
	Out      DType        // BF16 or F32
	Epilogue Lambda       // $0 is the accumulator; zero value means $0
	Extras   []Operand

	MaxM   int // 1..128: decode form for m ≤ MaxM; 0: prefill form
	BN     int // prefill form: 64, 128 or 256 (128 at most for NVFP4); 0 means 128
	Stages int // 2..8; 0 means as many as fit in 200 KiB
}

// ummaSharedBudget is the dynamic shared memory UMMAGEMM fills with stages,
// below sm_100's 227 KiB per block.
const ummaSharedBudget = 200 * 1024

func (g UMMAGEMM) norm() (UMMAGEMM, error) {
	if !cIdent(g.Name) {
		return g, fmt.Errorf("umma gemm: bad kernel name %q", g.Name)
	}
	if g.Weight != WeightBF16 && g.Weight != WeightNVFP4 {
		return g, fmt.Errorf("umma gemm %s: weight must be bf16 or nvfp4, not %v", g.Name, g.Weight)
	}
	if g.Out != BF16 && g.Out != F32 {
		return g, fmt.Errorf("umma gemm %s: output must be bf16 or f32, not %v", g.Name, g.Out)
	}
	if g.Epilogue.root == nil {
		g.Epilogue = Identity
	}
	if g.Epilogue.Arity() > 1+len(g.Extras) {
		return g, fmt.Errorf("umma gemm %s: epilogue %s needs %d operands, have %d", g.Name, g.Epilogue, g.Epilogue.Arity(), 1+len(g.Extras))
	}
	for _, e := range g.Extras {
		if !cIdent(e.Name) || (e.DType != F32 && e.DType != BF16) || e.Broadcast < PerCol || e.Broadcast > Full {
			return g, fmt.Errorf("umma gemm %s: bad extra operand %+v", g.Name, e)
		}
	}
	if g.MaxM < 0 || g.MaxM > 128 {
		return g, fmt.Errorf("umma gemm %s: MaxM %d outside 0..128", g.Name, g.MaxM)
	}
	if g.MaxM > 0 {
		g.BN = 0
		if g.N%128 != 0 {
			return g, fmt.Errorf("umma gemm %s: decode form needs N %d a multiple of 128", g.Name, g.N)
		}
	} else {
		if g.BN == 0 {
			g.BN = 128
		}
		max := 256
		if g.Weight == WeightNVFP4 {
			max = 128
		}
		if g.BN != 64 && g.BN != 128 && g.BN != 256 || g.BN > max {
			return g, fmt.Errorf("umma gemm %s: BN %d not 64, 128 or 256 (at most %d for %v)", g.Name, g.BN, max, g.Weight)
		}
		if g.N%g.BN != 0 {
			return g, fmt.Errorf("umma gemm %s: N %d not a multiple of BN %d", g.Name, g.N, g.BN)
		}
	}
	if g.N <= 0 || g.K <= 0 || g.K%g.kStep() != 0 {
		return g, fmt.Errorf("umma gemm %s: N %d, K %d: K must be a positive multiple of %d", g.Name, g.N, g.K, g.kStep())
	}
	if g.Stages == 0 {
		g.Stages = min(8, (ummaSharedBudget-g.fixedShared())/g.stageBytes())
	}
	if g.Stages < 2 || g.Stages > 8 {
		return g, fmt.Errorf("umma gemm %s: stages %d outside 2..8", g.Name, g.Stages)
	}
	if g.SharedBytes() > 227*1024 {
		return g, fmt.Errorf("umma gemm %s: %d bytes of shared memory, more than 227 KiB", g.Name, g.SharedBytes())
	}
	return g, nil
}

// kStep is the K extent of one stage: 128 bytes of each row.
func (g UMMAGEMM) kStep() int {
	if g.Weight == WeightNVFP4 {
		return 256
	}
	return 64
}

// nq is the accumulator's column count, the MMA's N.
func (g UMMAGEMM) nq() int {
	if g.MaxM > 0 {
		return max(16, (g.MaxM+15)/16*16)
	}
	return g.BN
}

// stageBytes is the shared memory of one stage, rounded up to the 1024-byte
// alignment of the swizzled tiles.
func (g UMMAGEMM) stageBytes() int {
	b := (128 + g.nq()) * 128
	if g.Weight == WeightNVFP4 {
		b += (128 + g.nq()) * 16 // raw scale boxes
	}
	return (b + 1023) &^ 1023
}

// fixedShared is the shared memory outside the stages: barriers and
// alignment slack, and the double-buffered scale chunks of NVFP4.
func (g UMMAGEMM) fixedShared() int {
	b := 2048
	if g.Weight == WeightNVFP4 {
		b += 2 * 4096
	}
	return b
}

// tmemCols is the tensor memory a CTA allocates: the accumulator and, for
// NVFP4, two buffers of 32 scale columns.
func (g UMMAGEMM) tmemCols() int {
	c := g.nq()
	if g.Weight == WeightNVFP4 {
		c += 64
	}
	n := 32
	for n < c {
		n *= 2
	}
	return n
}

// SharedBytes returns the dynamic shared memory one CTA uses.
func (g UMMAGEMM) SharedBytes() int {
	return g.fixedShared() + g.Stages*g.stageBytes()
}

// idesc returns the instruction descriptor of g's MMA.
func (g UMMAGEMM) idesc() uint32 {
	d := uint32(1)<<7 | uint32(1)<<10 | uint32(g.nq()>>3)<<17 | uint32(128>>4)<<24
	if g.Weight == WeightBF16 {
		d |= 1 << 4 // F32 accumulator
	}
	return d
}

// Grid returns the launch dimensions for m rows.
func (g UMMAGEMM) Grid(m int) (grid, block [3]int) {
	if g.MaxM > 0 {
		return [3]int{g.N / 128, 1, 1}, [3]int{128, 1, 1}
	}
	return [3]int{g.N / g.BN, (m + 127) / 128, 1}, [3]int{128, 1, 1}
}

func (g UMMAGEMM) params() []string {
	p := []string{"int m_total", "EG_TMAP tm_a", "EG_TMAP tm_b"}
	if g.Weight == WeightNVFP4 {
		p = append(p, "EG_TMAP tm_sfa", "EG_TMAP tm_sfb", "const float* __restrict__ w_gs")
	}
	for _, e := range g.Extras {
		p = append(p, "const "+e.DType.cType()+"* __restrict__ "+e.Name)
	}
	return append(p, g.Out.cType()+"* __restrict__ out")
}

// CUDA returns the kernel source without [Prelude].
func (g UMMAGEMM) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	fp4 := g.Weight == WeightNVFP4
	nq := g.nq()
	var b strings.Builder
	b.WriteString(ummaHelpers)
	form := "prefill"
	if g.MaxM > 0 {
		form = "decode"
	}
	fmt.Fprintf(&b, "\n// %s: tcgen05 %s GEMM, %v, N=%d K=%d, 128x%d accumulator, %d stages.\n", g.Name, form, g.Weight, g.N, g.K, nq, g.Stages)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n", g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, S = %d, KT = %d, NQ = %d, TCOLS = %d;\n", g.N, g.Stages, g.K/g.kStep(), nq, g.tmemCols())
	fmt.Fprintf(&b, "    constexpr uint32_t IDESC = %#xu;\n", g.idesc())
	b.WriteString("    constexpr uint32_t A_BYTES = 128 * 128, B_BYTES = NQ * 128;\n")
	if fp4 {
		b.WriteString("    constexpr uint32_t SFA_BYTES = 128 * 16, SFB_BYTES = NQ * 16;\n")
		b.WriteString("    constexpr uint32_t TX = A_BYTES + B_BYTES + SFA_BYTES + SFB_BYTES;\n")
	} else {
		b.WriteString("    constexpr uint32_t TX = A_BYTES + B_BYTES;\n")
	}
	fmt.Fprintf(&b, "    constexpr uint32_t STAGE = %d; // TX rounded up to 1024 bytes\n", g.stageBytes())
	b.WriteString(`    EG_DYNAMIC_SMEM(unsigned char, smem_raw);
    uint64_t* const full = reinterpret_cast<uint64_t*>(smem_raw);
    uint64_t* const empty = full + S;
    uint64_t* const accum = empty + S;
    uint32_t* const tslot = reinterpret_cast<uint32_t*>(accum + 1);
    unsigned char* const tiles = eg_umma_align1024(smem_raw + 1024);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
`)
	if fp4 {
		b.WriteString("    unsigned char* const chunks = tiles + S * STAGE;\n")
	}
	if g.MaxM > 0 {
		b.WriteString("    const int a_row0 = blockIdx.x * 128, b_row0 = 0; // A: W rows n, B: x rows m\n")
	} else {
		b.WriteString("    const int a_row0 = blockIdx.y * 128, b_row0 = blockIdx.x * NQ; // A: x rows m, B: W rows n\n")
	}
	b.WriteString(`    if (warp == 0) {
        if (lane == 0) {
            for (int s = 0; s < S; ++s) {
                eg_umma_bar_init(&full[s], 1);
                eg_umma_bar_init(&empty[s], 1);
            }
            eg_umma_bar_init(accum, 1);
            eg_umma_bar_fence_init();
        }
        eg_umma_alloc(tslot, TCOLS);
    }
    eg_umma_fence_before();
    __syncthreads();
    eg_umma_fence_after();
    const uint32_t tmem = *tslot;

    if (warp == 0 && lane == 0) {
        // Producer: a stage is free once the MMAs reading it commit.
        for (int kt = 0; kt < KT; ++kt) {
            const int s = kt % S;
            if (kt >= S) eg_umma_bar_wait(&empty[s], (kt / S - 1) & 1);
            unsigned char* const st = tiles + s * STAGE;
            eg_umma_bar_expect(&full[s], TX);
`)
	kc := "kt * 64"
	if fp4 {
		kc = "kt * 128"
	}
	fmt.Fprintf(&b, "            eg_umma_tma(st, &tm_a, %s, a_row0, &full[s]);\n", kc)
	fmt.Fprintf(&b, "            eg_umma_tma(st + A_BYTES, &tm_b, %s, b_row0, &full[s]);\n", kc)
	if fp4 {
		b.WriteString("            eg_umma_tma(st + A_BYTES + B_BYTES, &tm_sfa, kt * 16, a_row0, &full[s]);\n")
		b.WriteString("            eg_umma_tma(st + A_BYTES + B_BYTES + SFA_BYTES, &tm_sfb, kt * 16, b_row0, &full[s]);\n")
	}
	b.WriteString(`        }
    } else if (warp == 1) {
        // MMA issuer.
        for (int kt = 0; kt < KT; ++kt) {
            const int s = kt % S;
            unsigned char* const st = tiles + s * STAGE;
            eg_umma_bar_wait(&full[s], (kt / S) & 1);
`)
	if fp4 {
		// Scale chunk c of buffer kt&1 holds the scales of MMA K step c
		// (16-element blocks 4c..4c+3) for 128 rows: row r's four bytes at
		// 16*(r%32) + 4*(r/32). Before rewriting a buffer, wait until the
		// MMAs of stage kt-2, which read it, have completed.
		b.WriteString(`            if (kt >= 2) eg_umma_bar_wait(&empty[(kt - 2) % S], ((kt - 2) / S) & 1);
            unsigned char* const ch = chunks + (kt & 1) * 4096;
            const unsigned char* const sfa = st + A_BYTES + B_BYTES;
            const unsigned char* const sfb = sfa + SFA_BYTES;
            for (int i = lane; i < 128 * 16; i += 32) {
                const int r = i / 16, j = i % 16;
                const int o = (j / 4) * 512 + 16 * (r % 32) + 4 * (r / 32) + j % 4;
                ch[o] = sfa[i];
                ch[2048 + o] = r < NQ ? sfb[i] : (unsigned char)0;
            }
            eg_fence_proxy_async();
            __syncwarp();
`)
	}
	b.WriteString(`            eg_umma_fence_after();
            if (lane == 0) {
`)
	if fp4 {
		b.WriteString(`                const uint32_t sf = tmem + NQ + (kt & 1) * 32;
                for (int c = 0; c < 4; ++c) {
                    eg_umma_cp_scales(sf + 4 * c, eg_umma_desc_plain(ch + c * 512, 128, 128));
                    eg_umma_cp_scales(sf + 16 + 4 * c, eg_umma_desc_plain(ch + 2048 + c * 512, 128, 128));
                }
`)
	}
	b.WriteString(`                for (int kk = 0; kk < 4; ++kk) {
                    const uint64_t da = eg_umma_desc_sw128(st + kk * 32), db = eg_umma_desc_sw128(st + A_BYTES + kk * 32);
`)
	if fp4 {
		b.WriteString("                    eg_umma_nvfp4(tmem, da, db, IDESC, (kt | kk) != 0, sf + 4 * kk, sf + 16 + 4 * kk);\n")
	} else {
		b.WriteString("                    eg_umma_f16(tmem, da, db, IDESC, (kt | kk) != 0);\n")
	}
	b.WriteString(`                }
                eg_umma_commit(&empty[s]);
            }
            __syncwarp();
        }
        if (lane == 0) eg_umma_commit(accum);
        __syncwarp();
    }

    // Epilogue: warp w reads TMEM lanes [32w, 32w + 32), row p of A.
    eg_umma_bar_wait(accum, 0);
    __syncwarp();
    eg_umma_fence_after();
    const int p = a_row0 + 32 * warp + lane;
`)
	if fp4 {
		b.WriteString("    const float gs = w_gs[0];\n")
	}
	b.WriteString(`    for (int c0 = 0; c0 < NQ; c0 += 16) {
        uint32_t r[16];
        eg_umma_ld16(r, tmem + ((uint32_t)(32 * warp) << 16) + (uint32_t)c0);
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
`)
	if g.MaxM > 0 {
		b.WriteString("            const int m = c0 + j, n = p;\n")
	} else {
		b.WriteString("            const int m = p, n = b_row0 + c0 + j;\n")
	}
	b.WriteString("            if (m >= m_total) continue;\n")
	b.WriteString("            float v = __uint_as_float(r[j]);\n")
	if fp4 {
		b.WriteString("            v *= gs;\n")
	}
	args := []string{"v"}
	for _, e := range g.Extras {
		args = append(args, e.load("m", "n", "N"))
	}
	expr, err := g.Epilogue.CUDA(args...)
	if err != nil {
		return "", err
	}
	if g.Out == BF16 {
		fmt.Fprintf(&b, "            out[(size_t)m * N + n] = eg_f32_to_bf16(%s);\n", expr)
	} else {
		fmt.Fprintf(&b, "            out[(size_t)m * N + n] = %s;\n", expr)
	}
	b.WriteString(`        }
    }
    eg_umma_fence_before();
    __syncthreads();
    if (warp == 0) {
        eg_umma_fence_after();
        eg_umma_dealloc(tmem, TCOLS);
    }
}
`)
	return b.String(), nil
}

// LaunchParams returns the launcher's parameters after m: the operands it
// reads, the extras and the output, in [Plan] argument order.
func (g UMMAGEMM) launchParams() []string {
	var p []string
	if g.Weight == WeightNVFP4 {
		p = []string{"const uint8_t* xq", "const uint8_t* xsf", "const uint8_t* w", "const uint8_t* w_sf", "const float* w_gs"}
	} else {
		p = []string{"const uint16_t* x", "const uint16_t* w"}
	}
	for _, e := range g.Extras {
		p = append(p, "const "+e.DType.cType()+"* "+e.Name)
	}
	return append(p, g.Out.cType()+"* out")
}

// Launcher returns a host function
//
//	void Name_launch(int m, <operands...>, <extras...>, out, cudaStream_t stream)
//
// whose operands are x and w for WeightBF16, and xq, xsf, w, w_sf and w_gs
// for WeightNVFP4. In the decode form m must be at most MaxM.
func (g UMMAGEMM) Launcher() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(int m, %s, cudaStream_t stream) {\n", g.Name, strings.Join(g.launchParams(), ", "))
	if g.MaxM > 0 {
		fmt.Fprintf(&b, "    if (m <= 0 || m > %d) return;\n", g.MaxM)
	} else {
		b.WriteString("    if (m <= 0) return;\n")
	}
	fmt.Fprintf(&b, "    constexpr int smem = %d, N = %d, K = %d;\n", g.SharedBytes(), g.N, g.K)
	fmt.Fprintf(&b, "    static const cudaError_t attr = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr;\n", g.Name)
	aRows, bRows := "m", "N"
	aBox, bBox := 128, g.nq()
	if g.MaxM > 0 {
		aRows, bRows = "N", "m"
	}
	var args []string
	if g.Weight == WeightNVFP4 {
		a, bb, sa, sb := "xq", "w", "xsf", "w_sf"
		if g.MaxM > 0 {
			a, bb, sa, sb = "w", "xq", "w_sf", "xsf"
		}
		fmt.Fprintf(&b, "    const eg_tmap ta = eg_umma_tmap_2d(%s, 1, K / 2, %s, K / 2, 128, %d, true);\n", a, aRows, aBox)
		fmt.Fprintf(&b, "    const eg_tmap tb = eg_umma_tmap_2d(%s, 1, K / 2, %s, K / 2, 128, %d, true);\n", bb, bRows, bBox)
		fmt.Fprintf(&b, "    const eg_tmap tsa = eg_umma_tmap_2d(%s, 1, K / 16, %s, K / 16, 16, %d, false);\n", sa, aRows, aBox)
		fmt.Fprintf(&b, "    const eg_tmap tsb = eg_umma_tmap_2d(%s, 1, K / 16, %s, K / 16, 16, %d, false);\n", sb, bRows, bBox)
		args = []string{"m", "ta", "tb", "tsa", "tsb", "w_gs"}
	} else {
		a, bb := "x", "w"
		if g.MaxM > 0 {
			a, bb = "w", "x"
		}
		fmt.Fprintf(&b, "    const eg_tmap ta = eg_umma_tmap_2d(%s, 2, K, %s, K * 2, 64, %d, true);\n", a, aRows, aBox)
		fmt.Fprintf(&b, "    const eg_tmap tb = eg_umma_tmap_2d(%s, 2, K, %s, K * 2, 64, %d, true);\n", bb, bRows, bBox)
		args = []string{"m", "ta", "tb"}
	}
	for _, e := range g.Extras {
		args = append(args, e.Name)
	}
	args = append(args, "out")
	if g.MaxM > 0 {
		b.WriteString("    const dim3 grid(N / 128, 1), block(128);\n")
	} else {
		fmt.Fprintf(&b, "    const dim3 grid(N / %d, (m + 127) / 128), block(128);\n", g.BN)
	}
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, smem, stream, %s);\n}\n", g.Name, strings.Join(args, ", "))
	return b.String(), nil
}

// Source returns [Prelude], the kernel and its launcher.
func (g UMMAGEMM) Source() (string, error) {
	src, err := g.CUDA()
	if err != nil {
		return "", err
	}
	l, err := g.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + src + "\n" + l, nil
}

// NVFP4Quant quantizes BF16 rows x [M, K] to NVFP4 for [UMMAGEMM]: xq
// [M, K/2] E2M1 codes, element 2i in the low nibble, and xsf [M, K/16]
// UE4M3 scales, with no global scale. Each 16-element block's scale is
// amax/6 rounded to E4M3 (saturating at 448), and each element is x/scale
// rounded to the nearest E2M1 value, ties to even, saturating at 6.
type NVFP4Quant struct {
	Name string
	K    int // a positive multiple of 16
}

func (q NVFP4Quant) norm() error {
	if !cIdent(q.Name) {
		return fmt.Errorf("nvfp4 quant: bad kernel name %q", q.Name)
	}
	if q.K <= 0 || q.K%16 != 0 {
		return fmt.Errorf("nvfp4 quant %s: K %d not a positive multiple of 16", q.Name, q.K)
	}
	return nil
}

// CUDA returns the kernel source without [Prelude].
func (q NVFP4Quant) CUDA() (string, error) {
	if err := q.norm(); err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(ummaHelpers)
	fmt.Fprintf(&b, `
// %s: BF16 rows to NVFP4, one thread per 16-element block.
__global__ void __launch_bounds__(256) %s(int m_total, const uint16_t* __restrict__ x, uint8_t* __restrict__ xq, uint8_t* __restrict__ xsf) {
    constexpr int KB = %d;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= m_total * KB) return;
    float v[16], amax = 0.0f;
    for (int j = 0; j < 16; ++j) {
        v[j] = eg_bf16_to_f32(x[(size_t)i * 16 + j]);
        amax = fmaxf(amax, fabsf(v[j]));
    }
    const uint32_t sc = eg_f32x2_to_e4m3x2(amax / 6.0f, 0.0f) & 0xffu;
    const float s = eg_e4m3_to_f32(sc);
    for (int j = 0; j < 16; j += 2) {
        const uint32_t lo = s > 0.0f ? eg_umma_e2m1(v[j] / s) : 0u, hi = s > 0.0f ? eg_umma_e2m1(v[j + 1] / s) : 0u;
        xq[(size_t)i * 8 + j / 2] = (uint8_t)(lo | hi << 4);
    }
    xsf[i] = (uint8_t)sc;
}
`, q.Name, q.Name, q.K/16)
	return b.String(), nil
}

// Launcher returns a host function
//
//	void Name_launch(int m, const uint16_t* x, uint8_t* xq, uint8_t* xsf, cudaStream_t stream)
func (q NVFP4Quant) Launcher() (string, error) {
	if err := q.norm(); err != nil {
		return "", err
	}
	return fmt.Sprintf(`inline void %s_launch(int m, const uint16_t* x, uint8_t* xq, uint8_t* xsf, cudaStream_t stream) {
    const int blocks = (m * %d + 255) / 256;
    if (blocks > 0) EG_LAUNCH(%s, dim3(blocks), dim3(256), 0, stream, m, x, xq, xsf);
}
`, q.Name, q.K/16, q.Name), nil
}

// Reference returns what the kernel computes from m rows of x.
func (q NVFP4Quant) Reference(m int, x []uint16) (xq, xsf []uint8, err error) {
	if err := q.norm(); err != nil {
		return nil, nil, err
	}
	if len(x) != m*q.K {
		return nil, nil, fmt.Errorf("nvfp4 quant %s: have %d values, want %d", q.Name, len(x), m*q.K)
	}
	xq = make([]uint8, m*q.K/2)
	xsf = make([]uint8, m*q.K/16)
	for i := range xsf {
		var v [16]float32
		var amax float32
		for j := range v {
			v[j] = FromBF16Bits(x[i*16+j])
			amax = max(amax, float32(math.Abs(float64(v[j]))))
		}
		xsf[i] = E4M3Bits(amax / 6)
		s := FromE4M3Bits(xsf[i])
		for j := 0; j < 16; j += 2 {
			var lo, hi uint8
			if s > 0 {
				lo, hi = e2m1Even(v[j]/s), e2m1Even(v[j+1]/s)
			}
			xq[i*8+j/2] = lo | hi<<4
		}
	}
	return xq, xsf, nil
}

// e2m1Even is eg_umma_e2m1: f rounded to the nearest E2M1 code, ties to
// even, saturating at 6.
func e2m1Even(f float32) uint8 {
	a := float32(math.Abs(float64(f)))
	var c uint8
	switch {
	case a <= 0.25:
		c = 0
	case a < 0.75:
		c = 1
	case a <= 1.25:
		c = 2
	case a < 1.75:
		c = 3
	case a <= 2.5:
		c = 4
	case a < 3.5:
		c = 5
	case a <= 5:
		c = 6
	default:
		c = 7
	}
	if f < 0 && c != 0 {
		c |= 8
	}
	return c
}
