package tcgen05

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// TCGen05BlackwellTMAPrelude provides device helpers for Blackwell sm_100a/sm_103a
// L2 cache eviction policy descriptors (createpolicy.fractional.L2::evict_last /
// L2::evict_first), 2D TMA 4-row non-contiguous gather/scatter (.gather4 /
// .scatter4), and 2D TMA cluster multicast (.multicast::cluster).
const TCGen05BlackwellTMAPrelude = `#ifndef EG_TCGEN05_BLACKWELL_TMA_PRELUDE
#define EG_TCGEN05_BLACKWELL_TMA_PRELUDE

// eg_tma_createpolicy_evict_last constructs a 64-bit L2 cache policy descriptor
// prioritizing persistent residency (L2::evict_last) for shared expert weights.
EG_DEV uint64_t eg_tma_createpolicy_evict_last(float fraction) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_EMU)
    uint64_t policy = 0;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, %1;"
                 : "=l"(policy) : "f"(fraction));
    return policy;
#else
    (void)fraction;
    return 0x1ULL;
#endif
}

// eg_tma_createpolicy_evict_first constructs a 64-bit L2 cache policy descriptor
// marking streaming gathered token activations as L2::evict_first so they do not
// thrash resident expert weights in L2.
EG_DEV uint64_t eg_tma_createpolicy_evict_first(float fraction) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(EG_EMU)
    uint64_t policy = 0;
    asm volatile("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, %1;"
                 : "=l"(policy) : "f"(fraction));
    return policy;
#else
    (void)fraction;
    return 0x2ULL;
#endif
}

// eg_tma_gather4_2d_f32 gathers 4 non-contiguous global rows (r0, r1, r2, r3) of
// width col_dim floats into dst_smem [4, col_dim] using a single Blackwell TMA
// .gather4 instruction with an L2::cache_hint policy descriptor.
EG_DEV void eg_tma_gather4_2d_f32(
    float* dst_smem, const float* src_global, int col_dim,
    int32_t r0, int32_t r1, int32_t r2, int32_t r3,
    uint64_t* mbar, uint64_t l2_policy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    if (threadIdx.x == 0) {
        const uint32_t smem_ptr = (uint32_t)__cvta_generic_to_shared(dst_smem);
        const uint32_t mbar_ptr = (uint32_t)__cvta_generic_to_shared(mbar);
        const int32_t c0 = 0;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.gather4.mbarrier::complete_tx::bytes.L2::cache_hint "
            "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
            :: "r"(smem_ptr), "l"(src_global),
               "r"(c0), "r"(r0), "r"(r1), "r"(r2), "r"(r3),
               "r"(mbar_ptr), "l"(l2_policy)
            : "memory");
    }
#else
    (void)mbar;
    (void)l2_policy;
    const int32_t rows[4] = {r0, r1, r2, r3};
    for (int i = (int)threadIdx.x; i < 4 * col_dim; i += (int)blockDim.x) {
        const int row_slot = i / col_dim;
        const int col = i % col_dim;
        dst_smem[i] = src_global[(int64_t)rows[row_slot] * (int64_t)col_dim + (int64_t)col];
    }
#endif
}

// eg_tma_scatter4_2d_f32 scatters 4 contiguous shared-memory rows [4, col_dim]
// to 4 non-contiguous global rows (r0, r1, r2, r3) using Blackwell TMA .scatter4.
EG_DEV void eg_tma_scatter4_2d_f32(
    float* dst_global, const float* src_smem, int col_dim,
    int32_t r0, int32_t r1, int32_t r2, int32_t r3,
    uint64_t l2_policy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    if (threadIdx.x == 0) {
        const uint32_t smem_ptr = (uint32_t)__cvta_generic_to_shared(src_smem);
        const int32_t c0 = 0;
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.scatter4.L2::cache_hint "
            "[%0, {%1, %2, %3, %4, %5}], [%6], %7;"
            :: "l"(dst_global),
               "r"(c0), "r"(r0), "r"(r1), "r"(r2), "r"(r3),
               "r"(smem_ptr), "l"(l2_policy)
            : "memory");
    }
#else
    (void)l2_policy;
    const int32_t rows[4] = {r0, r1, r2, r3};
    for (int i = (int)threadIdx.x; i < 4 * col_dim; i += (int)blockDim.x) {
        const int row_slot = i / col_dim;
        const int col = i % col_dim;
        dst_global[(int64_t)rows[row_slot] * (int64_t)col_dim + (int64_t)col] = src_smem[i];
    }
#endif
}

// eg_tma_multicast_2d_f32 broadcasts a [rows, cols] tile from global memory into
// the shared memory of all cluster CTAs selected by cta_mask (e.g., 0x3 for a
// 2-CTA cta_group::2 pair) using cp.async.bulk.tensor.2d...multicast::cluster.
EG_DEV void eg_tma_multicast_2d_f32(
    float* local_smem_b, const float* src_global_b,
    int rows, int cols, uint16_t cta_mask,
    uint64_t* mbar, uint64_t l2_policy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    if (threadIdx.x == 0) {
        const uint32_t smem_ptr = (uint32_t)__cvta_generic_to_shared(local_smem_b);
        const uint32_t mbar_ptr = (uint32_t)__cvta_generic_to_shared(mbar);
        const int32_t c0 = 0, c1 = 0;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint "
            "[%0], [%1, {%2, %3}], [%4], %5, %6;"
            :: "r"(smem_ptr), "l"(src_global_b),
               "r"(c0), "r"(c1),
               "r"(mbar_ptr), "h"(cta_mask), "l"(l2_policy)
            : "memory");
    }
#else
    (void)mbar;
    (void)l2_policy;
    const int total = rows * cols;
    for (int dst_rank = 0; dst_rank < 2; ++dst_rank) {
        if ((cta_mask & (uint16_t)(1u << dst_rank)) != 0u) {
            float* dst = eg_tcgen05_map_shared_rank(local_smem_b, dst_rank);
            for (int i = (int)threadIdx.x; i < total; i += (int)blockDim.x) {
                dst[i] = src_global_b[i];
            }
        }
    }
#endif
}

// eg_tcgen05_mma_cta2_multicast_f32 issues a 2-CTA cooperative tcgen05.mma.cta_group::2
// where each CTA holds its gathered 128xK activation slice in local_smem_a and
// the multicast-replicated Kx128 weight tile in local_smem_b, using the
// .kind::tf32 instruction descriptor idesc.
EG_DEV void eg_tcgen05_mma_cta2_multicast_f32(
    float* local_tmem, uint32_t d_taddr,
    float* local_smem_a, float* local_smem_b, int k_dim, uint32_t idesc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)local_tmem;
    (void)k_dim;
    const uint64_t desc_a = eg_wgmma_desc(local_smem_a, 128, 1024);
    const uint64_t desc_b = eg_wgmma_desc(local_smem_b, 128, 1024);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p;\n"
        "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
#else
    (void)idesc;
    for (int cta = 0; cta < 2; ++cta) {
        const float* a_cta = eg_tcgen05_map_shared_rank(local_smem_a, cta);
        const float* b_cta = eg_tcgen05_map_shared_rank(local_smem_b, cta);
        float* tmem_cta = eg_tcgen05_map_shared_rank(local_tmem, cta);
        for (int r = 0; r < 128; ++r) {
            for (int n = 0; n < 128; ++n) {
                float acc = tmem_cta[r * 512 + d_taddr + n];
                for (int k = 0; k < k_dim; ++k) {
                    acc += a_cta[r * k_dim + k] * b_cta[k * 128 + n];
                }
                tmem_cta[r * 512 + d_taddr + n] = acc;
            }
        }
    }
#endif
}

#endif // EG_TCGEN05_BLACKWELL_TMA_PRELUDE
`

// TCGen05TMAMulticastGatherGEMM emits a 2-CTA (__cluster_dims__(2, 1, 1),
// cta_group::2) Blackwell sm_100a/sm_103a routed GEMM kernel that:
//   - gathers non-contiguous token rows from global memory into each CTA's shared
//     memory in 4-row batches via TMA .gather4 with an L2::evict_first cache hint,
//   - multicasts the shared weight tile B from CTA 0 into both CTAs' shared
//     memory simultaneously via TMA .multicast::cluster (cta_mask = 0x3) with an
//     L2::evict_last residency hint,
//   - issues a cooperative 2-CTA tcgen05.mma.cta_group::2.kind::tf32 and
//     tcgen05.commit.cta_group::2 from CTA 0's elected leader thread, and
//   - reads each CTA's 128-row TMEM partition via 4-warp 32-row partitioned
//     tcgen05.ld (and optionally stores the output rows in 4-row batches via TMA
//     .scatter4 when ScatterOutput is true).
type TCGen05TMAMulticastGatherGEMM struct {
	Name          string
	M             int    // total gathered token rows across the 2-CTA cluster; defaults to 256 (128 per CTA)
	N             int    // output columns; defaults to 128 (must be 128)
	K             int    // reduction dimension; defaults to 32 (positive multiple of 16)
	Target        string // PTX target ("sm_100a", "sm_103a", or "sm_100f"; defaults to "sm_100a")
	ScatterOutput bool   // when true, stages epilogue rows in shared memory and stores via TMA .scatter4
}

func (k TCGen05TMAMulticastGatherGEMM) norm() (TCGen05TMAMulticastGatherGEMM, error) {
	if k.Name == "" {
		k.Name = "tcgen05_tma_multicast_gather_gemm"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 tma multicast gather gemm: bad kernel name %q", k.Name)
	}
	if k.M == 0 {
		k.M = 256
	}
	if k.N == 0 {
		k.N = 128
	}
	if k.K == 0 {
		k.K = 32
	}
	if k.Target == "" {
		k.Target = "sm_100a"
	}
	switch k.Target {
	case "sm_100a", "sm_103a", "sm_100f":
	default:
		return k, fmt.Errorf("tcgen05 tma multicast gather gemm %s: unsupported target %q (want sm_100a, sm_103a, or sm_100f)", k.Name, k.Target)
	}
	if k.M != 256 || k.N != 128 || k.K <= 0 || k.K%16 != 0 {
		return k, fmt.Errorf("tcgen05 tma multicast gather gemm %s: invalid shape M=%d N=%d K=%d (want M=256, N=128, K=16*m)", k.Name, k.M, k.N, k.K)
	}
	return k, nil
}

// Grid returns the 2-CTA cluster grid and 128-thread (4-warp) block dimensions.
func (k TCGen05TMAMulticastGatherGEMM) Grid() (grid, block [3]int) {
	return [3]int{2, 1, 1}, [3]int{128, 1, 1}
}

// TraceEvents returns the formal sm_100a/sm_103a 2-CTA (cta_group::2) tcgen05
// hardware protocol event trace for verification with cudatmem.VerifyTrace.
func (k TCGen05TMAMulticastGatherGEMM) TraceEvents() []cudatmem.TraceEvent {
	events := []cudatmem.TraceEvent{
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventAlloc, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventClusterSync, CTAGroup: 2},
		{Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventWait, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
	}
	for cta := range 2 {
		for w := range 4 {
			events = append(events, cudatmem.TraceEvent{
				Kind:          cudatmem.EventLd,
				CTA:           cta,
				Warp:          w,
				ActiveThreads: 32,
				CTAGroup:      2,
				Offset:        0,
				Columns:       128,
				RowStart:      uint32(32 * w),
				Rows:          32,
			})
		}
	}
	events = append(events,
		cudatmem.TraceEvent{Kind: cudatmem.EventClusterSync, CTAGroup: 2},
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
	)
	return events
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05TMAMulticastGatherGEMM) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	scatterFlag := 0
	if k.ScatterOutput {
		scatterFlag = 1
	}
	idesc, err := tf32InstDesc(k.M, k.N)
	if err != nil {
		return "", fmt.Errorf("tcgen05 tma multicast gather gemm %s: %w", k.Name, err)
	}
	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString("\n")
	b.WriteString(TCGen05BlackwellTMAPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: target=%s 2-CTA (cta_group::2) TMA .gather4 + .multicast::cluster M=%d, N=%d, K=%d GEMM.\n",
		k.Name, k.Target, k.M, k.N, k.K)
	fmt.Fprintf(&b, `__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) %s(
    const float* __restrict__ a_global,
    const int32_t* __restrict__ gather_rows,
    const float* __restrict__ b,
    float* __restrict__ out) {
    constexpr int K = %d;
    constexpr int N = %d;
    constexpr int SCATTER_OUT = %d;
    constexpr uint32_t IDESC = 0x%08xu;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rank = eg_tcgen05_cluster_rank();

    EG_TCGEN05_SHARED(float, smem_a, [128 * K]);
    EG_TCGEN05_SHARED(float, smem_b, [K * N]);
    EG_TCGEN05_SHARED(float, smem_epilogue, [4 * N]);
    EG_TCGEN05_SHARED(float, tmem, [128 * 512]);
    EG_TCGEN05_SHARED(uint32_t, alloc_slot, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar_a, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar_b, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar_mma, [1]);

    // 1. Both CTAs allocate 128 TMEM columns under cta_group::2.
    if (warp == 0) {
        eg_tcgen05_alloc_cta2(tmem, &alloc_slot[0], 0u, 128u);
    }
    __syncthreads();
    const uint32_t taddr = alloc_slot[0];

    // 2. Construct L2 cache eviction policy descriptors:
    //    - L2::evict_first (fraction 0.25) for streaming gathered token rows
    //    - L2::evict_last  (fraction 1.00) for shared expert weights B
    const uint64_t l2_act_policy = eg_tma_createpolicy_evict_first(0.25f);
    const uint64_t l2_weight_policy = eg_tma_createpolicy_evict_last(1.0f);

    // 3. Each CTA gathers its 128 non-contiguous token rows in 32 batches of 4 rows via TMA .gather4.
    for (int batch = 0; batch < 32; ++batch) {
        const int base = rank * 128 + batch * 4;
        eg_tma_gather4_2d_f32(
            &smem_a[batch * 4 * K], a_global, K,
            gather_rows[base + 0], gather_rows[base + 1],
            gather_rows[base + 2], gather_rows[base + 3],
            &mbar_a[0], l2_act_policy);
    }
    eg_tcgen05_cluster_sync();

    // 4. CTA 0 multicasts the shared expert weight tile B [K, N] into BOTH CTAs'
    //    shared memory simultaneously via TMA .multicast::cluster (cta_mask = 0x3).
    if (rank == 0) {
        eg_tma_multicast_2d_f32(smem_b, b, K, N, 0x3u, &mbar_b[0], l2_weight_policy);
    }
    eg_tcgen05_cluster_sync();

    // 5. CTA 0's elected leader thread issues cooperative tcgen05.mma.cta_group::2 + commit.
    if (rank == 0 && warp == 0 && eg_elect_one_sync()) {
        eg_tcgen05_mma_cta2_multicast_f32(tmem, taddr, smem_a, smem_b, K, IDESC);
        eg_tcgen05_commit_cta2(&mbar_mma[0]);
    }
    eg_tcgen05_cluster_sync();

    // 6. 4-warp 32-row partitioned tcgen05.ld epilogue (with optional TMA .scatter4 output).
    const int local_row = 32 * warp + lane;
    const int global_out_row = rank * 128 + local_row;
    if constexpr (SCATTER_OUT != 0) {
        for (int batch = 0; batch < 32; ++batch) {
            const int r_lo = batch * 4;
            if (local_row >= r_lo && local_row < r_lo + 4) {
                for (int col = 0; col < N; ++col) {
                    smem_epilogue[(local_row - r_lo) * N + col] =
                        eg_tcgen05_ld_row(tmem, warp, lane, taddr + (uint32_t)col);
                }
            }
            __syncthreads();
            const int32_t out_base = (int32_t)(rank * 128 + r_lo);
            eg_tma_scatter4_2d_f32(
                out, smem_epilogue, N,
                out_base + 0, out_base + 1, out_base + 2, out_base + 3,
                l2_act_policy);
            __syncthreads();
        }
        eg_tcgen05_wait_ld();
    } else {
        for (int col = 0; col < N; ++col) {
            const float v = eg_tcgen05_ld_row(tmem, warp, lane, taddr + (uint32_t)col);
            out[global_out_row * N + col] = v;
        }
        eg_tcgen05_wait_ld();
    }

    // 7. Cluster barrier before cooperative cta_group::2 TMEM deallocation.
    eg_tcgen05_cluster_sync();
    if (warp == 0) {
        eg_tcgen05_dealloc_cta2(taddr, 128u);
    }
}
EG_REGISTER_CLUSTER2(%s)
`, k.Name, k.K, k.N, scatterFlag, idesc, k.Name)
	return b.String(), nil
}

// CUDAFile emits the self-contained CUDA C++ translation unit (including
// Prelude) for k, ready for cudahostemu.Run or nvcc.
func (k TCGen05TMAMulticastGatherGEMM) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// Reference computes the host reference output [M, N] for the routed gather4 +
// multicast GEMM. Only the rows of aGlobal indexed by gatherRows are accessed,
// so ungathered rows in aGlobal (even if poisoned with NaN) are never read.
func (k TCGen05TMAMulticastGatherGEMM) Reference(aGlobal []float32, totalGlobalRows int, gatherRows []int32, b []float32) ([]float32, error) {
	k, err := k.norm()
	if err != nil {
		return nil, err
	}
	if totalGlobalRows < 1 {
		return nil, fmt.Errorf("tcgen05 tma multicast gather gemm: totalGlobalRows=%d must be positive", totalGlobalRows)
	}
	if len(aGlobal) != totalGlobalRows*k.K {
		return nil, fmt.Errorf("tcgen05 tma multicast gather gemm: len(aGlobal)=%d, want %d (%d*%d)", len(aGlobal), totalGlobalRows*k.K, totalGlobalRows, k.K)
	}
	if len(gatherRows) != k.M {
		return nil, fmt.Errorf("tcgen05 tma multicast gather gemm: len(gatherRows)=%d, want %d", len(gatherRows), k.M)
	}
	if len(b) != k.K*k.N {
		return nil, fmt.Errorf("tcgen05 tma multicast gather gemm: len(b)=%d, want %d (%d*%d)", len(b), k.K*k.N, k.K, k.N)
	}

	out := make([]float32, k.M*k.N)
	for r := 0; r < k.M; r++ {
		srcRow := int(gatherRows[r])
		if srcRow < 0 || srcRow >= totalGlobalRows {
			return nil, fmt.Errorf("tcgen05 tma multicast gather gemm: gatherRows[%d]=%d out of [0, %d)", r, srcRow, totalGlobalRows)
		}
		aRow := aGlobal[srcRow*k.K : (srcRow+1)*k.K]
		outRow := out[r*k.N : (r+1)*k.N]
		for c := 0; c < k.N; c++ {
			var acc float64
			for kk := 0; kk < k.K; kk++ {
				acc += float64(aRow[kk]) * float64(b[kk*k.N+c])
			}
			outRow[c] = float32(acc)
		}
	}
	return out, nil
}
