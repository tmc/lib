package tcgen05

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatma"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// A LoRAAdapterSlot locates one rank-r adapter in the unified adapter pools:
// rows [AOffset, AOffset+Rank) of the A pool [rows][DimIn] hold A_a and
// rows [BOffset, BOffset+Rank) of the B pool [rows][DimOut] hold B_aᵀ.
type LoRAAdapterSlot struct {
	AdapterID int
	Rank      int     // 8, 16, 32 or 64
	Scale     float32 // alpha / r
	AOffset   int
	BOffset   int
}

// A LoRASegment is a run of tokens that share one adapter in a
// [LoRAGatherPlan]'s order.
type LoRASegment struct {
	Slot       LoRAAdapterSlot
	Start, End int    // positions [Start, End) in Order
	TokenQuads [2]int // range of TokenQuads covering the segment's tokens
	ARowQuads  [2]int // range of ARowQuads covering the adapter's A rows
}

// A LoRAGatherPlan groups a batch's tokens by adapter for segmented LoRA.
type LoRAGatherPlan struct {
	// Order lists token indices grouped by ascending adapter ID, in index
	// order within a group; base-only tokens (adapter -1) come last.
	Order []int32
	// Segments covers the adapter tokens of Order; base-only tokens are in
	// no segment and get no LoRA work.
	Segments []LoRASegment
	// TokenQuads and ARowQuads are the row indices of four-row TMA
	// .gather4 loads of x and of the A pool. A segment's last quad repeats
	// its final row.
	TokenQuads [][4]int32
	ARowQuads  [][4]int32
}

func quads(rows []int32) [][4]int32 {
	var q [][4]int32
	for i := 0; i < len(rows); i += 4 {
		var x [4]int32
		for j := range x {
			x[j] = rows[min(i+j, len(rows)-1)]
		}
		q = append(q, x)
	}
	return q
}

// PlanLoRAGather groups tokens by adapter. tokenAdapterIDs[i] is token i's
// adapter ID, or -1 for the base model only; every other ID must be in
// adapters.
func PlanLoRAGather(tokenAdapterIDs []int, adapters map[int]LoRAAdapterSlot) (LoRAGatherPlan, error) {
	byID := make(map[int][]int32)
	var base []int32
	for i, id := range tokenAdapterIDs {
		if id == -1 {
			base = append(base, int32(i))
			continue
		}
		a, ok := adapters[id]
		if !ok {
			return LoRAGatherPlan{}, fmt.Errorf("plan lora gather: token %d uses unknown adapter %d", i, id)
		}
		if err := a.check(); err != nil {
			return LoRAGatherPlan{}, fmt.Errorf("plan lora gather: %w", err)
		}
		byID[id] = append(byID[id], int32(i))
	}
	ids := make([]int, 0, len(byID))
	for id := range byID {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	var p LoRAGatherPlan
	for _, id := range ids {
		a := adapters[id]
		toks := byID[id]
		seg := LoRASegment{Slot: a, Start: len(p.Order), End: len(p.Order) + len(toks)}
		p.Order = append(p.Order, toks...)
		seg.TokenQuads[0] = len(p.TokenQuads)
		p.TokenQuads = append(p.TokenQuads, quads(toks)...)
		seg.TokenQuads[1] = len(p.TokenQuads)
		rows := make([]int32, a.Rank)
		for r := range rows {
			rows[r] = int32(a.AOffset + r)
		}
		seg.ARowQuads[0] = len(p.ARowQuads)
		p.ARowQuads = append(p.ARowQuads, quads(rows)...)
		seg.ARowQuads[1] = len(p.ARowQuads)
		p.Segments = append(p.Segments, seg)
	}
	p.Order = append(p.Order, base...)
	return p, nil
}

func (a LoRAAdapterSlot) check() error {
	switch a.Rank {
	case 8, 16, 32, 64:
	default:
		return fmt.Errorf("adapter %d: rank %d not 8, 16, 32 or 64", a.AdapterID, a.Rank)
	}
	if a.AOffset < 0 || a.BOffset < 0 {
		return fmt.Errorf("adapter %d: negative pool offset", a.AdapterID)
	}
	return nil
}

// Gather4Specs returns the TMA .gather4 loads of p's token quads followed by
// its A-pool quads, for rows of dimIn elements of elemBytes bytes.
func (p LoRAGatherPlan) Gather4Specs(target string, dimIn, elemBytes int) ([]cudatma.TMAGather4Spec, error) {
	var specs []cudatma.TMAGather4Spec
	for _, q := range slices.Concat(p.TokenQuads, p.ARowQuads) {
		s := cudatma.TMAGather4Spec{Target: target, Mode: cudatma.TMAGather4Load, ColDim: dimIn, ElemBytes: elemBytes, RowIndices: q}
		if err := s.Validate(); err != nil {
			return nil, fmt.Errorf("lora gather4: %w", err)
		}
		specs = append(specs, s)
	}
	return specs, nil
}

// ReferenceFusedLoRA computes in float64, for each token i,
//
//	y_i = W x_i + scale_a (B_a (A_a x_i))     with a = tokenAdapterIDs[i]
//
// and y_i = W x_i when a is -1. x is [tokens][DimIn], w [DimOut][DimIn],
// and the pools are laid out as [LoRAAdapterSlot] describes.
func ReferenceFusedLoRA(x, w [][]float32, tokenAdapterIDs []int, aPool, bPool [][]float32, adapters map[int]LoRAAdapterSlot) ([][]float32, error) {
	if len(tokenAdapterIDs) != len(x) {
		return nil, fmt.Errorf("lora reference: %d adapter ids for %d tokens", len(tokenAdapterIDs), len(x))
	}
	if len(w) == 0 {
		return nil, fmt.Errorf("lora reference: empty w")
	}
	dimIn, dimOut := len(w[0]), len(w)
	y := make([][]float32, len(x))
	u := make([]float64, 64)
	for i, xi := range x {
		if len(xi) != dimIn {
			return nil, fmt.Errorf("lora reference: x[%d] has %d values, want %d", i, len(xi), dimIn)
		}
		y[i] = make([]float32, dimOut)
		id := tokenAdapterIDs[i]
		var a LoRAAdapterSlot
		if id != -1 {
			var ok bool
			if a, ok = adapters[id]; !ok {
				return nil, fmt.Errorf("lora reference: token %d uses unknown adapter %d", i, id)
			}
			if err := a.check(); err != nil {
				return nil, fmt.Errorf("lora reference: %w", err)
			}
			if a.AOffset+a.Rank > len(aPool) || a.BOffset+a.Rank > len(bPool) {
				return nil, fmt.Errorf("lora reference: adapter %d exceeds the pools", id)
			}
			for r := range a.Rank {
				u[r] = 0
				for k, xv := range xi {
					u[r] += float64(aPool[a.AOffset+r][k]) * float64(xv)
				}
			}
		}
		for n := range y[i] {
			var acc float64
			for k, xv := range xi {
				acc += float64(w[n][k]) * float64(xv)
			}
			if id != -1 {
				var l float64
				for r := range a.Rank {
					l += float64(bPool[a.BOffset+r][n]) * u[r]
				}
				acc += float64(a.Scale) * l
			}
			y[i][n] = float32(acc)
		}
	}
	return y, nil
}

// A SegmentedLoRASchedule configures a fused base + multi-adapter LoRA
// projection. Each CTA owns 128 consecutive tokens of a [LoRAGatherPlan]'s
// order and TileN output columns:
//
//	D  = X Wᵀ                         (TMEM columns [0, TileN))
//	U  = X_s A_aᵀ                     (TMEM columns [TileN, TileN+R), per segment s)
//	D += (scale_a U) B_aᵀ             (enable_input_d = 1 onto the base accumulator)
//
// where X_s keeps only the rows of segment s, so other tokens' rows of U
// are zero. Y_base and U never leave the CTA. Tokens are gathered by row
// index in four-row groups by the threads; the TMA .gather4 form of those
// loads is described by [LoRAGatherPlan.Gather4Specs], not issued by the
// kernel. The sm_100a/sm_103a/sm_107a path of the emitted source has not
// been executed; the EG_EMU path runs in the tests.
type SegmentedLoRASchedule struct {
	Name    string
	Target  string // sm_100a, sm_103a, sm_107a or an f variant (default sm_100a)
	DimIn   int    // K, a multiple of BK
	DimOut  int    // N, a multiple of TileN
	TileN   int    // output columns per CTA: a multiple of 16 in [16, 256] (default min(DimOut, 256))
	BK      int    // K step: 8, 16, 32 or 64 (default 32)
	MaxRank int    // largest adapter rank, at most 64
}

func (s SegmentedLoRASchedule) norm() (SegmentedLoRASchedule, error) {
	if s.Name == "" {
		s.Name = "segmented_lora"
	}
	if s.Target == "" {
		s.Target = "sm_100a"
	}
	if s.TileN == 0 {
		s.TileN = min(s.DimOut, 256)
	}
	if s.BK == 0 {
		s.BK = 32
	}
	if !cIdent(s.Name) {
		return s, fmt.Errorf("segmented lora: bad name %q", s.Name)
	}
	if err := checkTMTarget("segmented lora", s.Target); err != nil {
		return s, err
	}
	if s.TileN < 16 || s.TileN > 256 || s.TileN%16 != 0 || s.DimOut < s.TileN || s.DimOut%s.TileN != 0 {
		return s, fmt.Errorf("segmented lora: tile %d must be a multiple of 16 in [16, 256] dividing DimOut %d", s.TileN, s.DimOut)
	}
	switch s.BK {
	case 8, 16, 32, 64:
	default:
		return s, fmt.Errorf("segmented lora: BK %d not 8, 16, 32 or 64", s.BK)
	}
	if s.DimIn < s.BK || s.DimIn%s.BK != 0 {
		return s, fmt.Errorf("segmented lora: DimIn %d must be a positive multiple of BK %d", s.DimIn, s.BK)
	}
	if s.MaxRank < 8 || s.MaxRank > 64 {
		return s, fmt.Errorf("segmented lora: max rank %d not in [8, 64]", s.MaxRank)
	}
	if b := s.sharedBytes(); b > maxDynamicShared {
		return s, fmt.Errorf("segmented lora: %d bytes of shared memory exceed %d", b, maxDynamicShared)
	}
	return s, nil
}

// Validate checks that the base accumulator and the shrink intermediate fit
// disjointly in TMEM and the operand tiles in shared memory.
func (s SegmentedLoRASchedule) Validate() error {
	_, err := s.norm()
	return err
}

// rankCols is the shrink width: MaxRank rounded up to the M=128 MMA N step.
func (s SegmentedLoRASchedule) rankCols() int { return (s.MaxRank + 15) &^ 15 }

func (s SegmentedLoRASchedule) sharedBytes() int {
	r := s.rankCols()
	return 4 * (128*s.BK + s.TileN*s.BK + r*s.BK + 128*r + s.TileN*r)
}

// TMEMColumns returns the TMEM allocation width: the base tile plus the
// shrink intermediate.
func (s SegmentedLoRASchedule) TMEMColumns() uint32 { return allocColumns(s.TileN + s.rankCols()) }

// SharedBytes returns the dynamic shared memory the kernel needs, or 0 if s
// is invalid.
func (s SegmentedLoRASchedule) SharedBytes() int {
	s, err := s.norm()
	if err != nil {
		return 0
	}
	return s.sharedBytes()
}

// Grid returns the launch shape for tokens tokens.
func (s SegmentedLoRASchedule) Grid(tokens int) (grid, block [3]int) {
	s, err := s.norm()
	if err != nil {
		return
	}
	return [3]int{s.DimOut / s.TileN, (tokens + 127) / 128, 1}, [3]int{128, 1, 1}
}

// TraceEvents returns the tcgen05 operation sequence of a CTA whose tokens
// span segments adapter segments.
func (s SegmentedLoRASchedule) TraceEvents(segments int) []cudatmem.TraceEvent {
	s, err := s.norm()
	if err != nil {
		return nil
	}
	tn, r := uint32(s.TileN), uint32(s.rankCols())
	tr := newTMTrace(s.TMEMColumns())
	for range s.DimIn / s.BK {
		tr.mma(0, tn, false, 0)
		tr.commitWait()
	}
	for range segments {
		for range s.DimIn / s.BK {
			tr.mma(tn, r, false, 0)
			tr.commitWait()
		}
		tr.rows(cudatmem.EventLd, tn, r, 128)
		tr.mma(0, tn, false, 0)
		tr.commitWait()
	}
	tr.rows(cudatmem.EventLd, 0, tn, 128)
	return tr.done()
}

// CUDA returns the kernel source without [Prelude]. The kernel signature is
//
//	name(int tokens, int nsegs, const int* order,
//	     const int* seg_start, const int* seg_end, const int* seg_rank,
//	     const int* seg_aoff, const int* seg_boff, const float* seg_scale,
//	     const float* x, const float* w, const float* a_pool,
//	     const float* b_pool, float* y)
//
// with order and the seg_ arrays taken from a [LoRAGatherPlan], x
// [tokens][DimIn], w [DimOut][DimIn], and y [tokens][DimOut] in token
// order.
func (s SegmentedLoRASchedule) CUDA() (string, error) {
	s, err := s.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(tmPrelude)
	fmt.Fprintf(&b, `
// %[1]s: fused base + segmented multi-adapter LoRA accumulated in TMEM
// (K=%[2]d, N=%[3]d, tile 128x%[4]d, BK=%[5]d, max rank %[6]d, target %[7]s).
__global__ void __launch_bounds__(128) %[1]s(
    int tokens, int nsegs, const int* __restrict__ order,
    const int* __restrict__ seg_start, const int* __restrict__ seg_end, const int* __restrict__ seg_rank,
    const int* __restrict__ seg_aoff, const int* __restrict__ seg_boff, const float* __restrict__ seg_scale,
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ a_pool, const float* __restrict__ b_pool, float* __restrict__ y) {
    constexpr int K = %[2]d, N = %[3]d, TN = %[4]d, BK = %[5]d, R = %[8]d;
    constexpr uint32_t COLS = %[9]d, C_B = 0, C_U = TN;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int m0 = blockIdx.y * 128, n0 = blockIdx.x * TN;
    const int rows = tokens - m0 < 128 ? tokens - m0 : 128;
    const int tok = tid < rows ? order[m0 + tid] : -1;
    EG_TM_BUF(tm);
    EG_TM_SHARED(uint32_t, slot, [1]);
    EG_TM_SHARED(uint64_t, mbar, [1]);
    EG_DYNAMIC_SMEM(float, sm);
    float* const xs = sm;            // X rows [128][BK]: A of the base and shrink MMAs
    float* const wb = xs + 128 * BK; // W [TN][BK]: B of the base MMA
    float* const as = wb + TN * BK;  // A_a [R][BK], rows >= rank zero: B of the shrink MMA
    float* const us = as + R * BK;   // scale * U [128][R]: A of the expand MMA
    float* const bs = us + 128 * R;  // B_a [TN][R], columns >= rank zero: B of the expand MMA
    uint32_t phase = 0;
    if (tid == 0) eg_tm_mbar_init(&mbar[0]);
    if (warp == 0) eg_tm_alloc(tm, &slot[0], COLS);
    eg_tm_sync();
    const uint32_t base = slot[0];

    // 1. D = X Wᵀ.
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int kk = 0; kk < BK; ++kk) xs[eg_cm_idx(tid, kk, BK)] = tok >= 0 ? x[(size_t)tok * K + k0 + kk] : 0.0f;
        for (int i = tid; i < TN * BK; i += 128) wb[eg_cm_idx(i / BK, i %% BK, BK)] = w[(size_t)(n0 + i / BK) * K + k0 + i %% BK];
        eg_tm_sync();
        if (tid == 0) {
            eg_tm_mma_ss(tm, base + C_B, xs, wb, TN, BK, k0 > 0);
            eg_tm_commit(&mbar[0]);
        }
        eg_tm_wait(&mbar[0], &phase);
        __syncthreads();
    }

    // 2. Per adapter segment overlapping this tile: U = X_s A_aᵀ, then
    // D += (scale U) B_aᵀ onto the base accumulator.
    for (int s = 0; s < nsegs; ++s) {
        const int lo = seg_start[s] > m0 ? seg_start[s] : m0;
        const int hi = seg_end[s] < m0 + rows ? seg_end[s] : m0 + rows;
        if (lo >= hi) continue;
        const int rank = seg_rank[s], aoff = seg_aoff[s], boff = seg_boff[s];
        const bool mine = m0 + tid >= lo && m0 + tid < hi;
        for (int k0 = 0; k0 < K; k0 += BK) {
            for (int kk = 0; kk < BK; ++kk) xs[eg_cm_idx(tid, kk, BK)] = mine ? x[(size_t)tok * K + k0 + kk] : 0.0f;
            for (int i = tid; i < R * BK; i += 128) {
                const int rr = i / BK, kk = i %% BK;
                as[eg_cm_idx(rr, kk, BK)] = rr < rank ? a_pool[(size_t)(aoff + rr) * K + k0 + kk] : 0.0f;
            }
            eg_tm_sync();
            if (tid == 0) {
                eg_tm_mma_ss(tm, base + C_U, xs, as, R, BK, k0 > 0);
                eg_tm_commit(&mbar[0]);
            }
            eg_tm_wait(&mbar[0], &phase);
            __syncthreads();
        }
        const float scale = seg_scale[s];
        for (int rr = 0; rr < R; ++rr) us[eg_cm_idx(tid, rr, R)] = scale * eg_tm_ld(tm, tid, base + C_U + rr);
        for (int i = tid; i < TN * R; i += 128) {
            const int n = i / R, rr = i %% R;
            bs[eg_cm_idx(n, rr, R)] = rr < rank ? b_pool[(size_t)(boff + rr) * N + n0 + n] : 0.0f;
        }
        eg_tm_sync();
        if (tid == 0) {
            eg_tm_mma_ss(tm, base + C_B, us, bs, TN, R, 1);
            eg_tm_commit(&mbar[0]);
        }
        eg_tm_wait(&mbar[0], &phase);
        __syncthreads();
    }

    // 3. One epilogue store of Y = X Wᵀ + Σ_s scale_s U_s B_sᵀ.
    for (int n = 0; n < TN; ++n) {
        const float v = eg_tm_ld(tm, tid, base + C_B + n);
        if (tok >= 0) y[(size_t)tok * N + n0 + n] = v;
    }
    eg_tm_sync();
    if (warp == 0) eg_tm_dealloc(base, COLS);
}
`, s.Name, s.DimIn, s.DimOut, s.TileN, s.BK, s.MaxRank, s.Target, s.rankCols(), s.TMEMColumns())
	return b.String(), nil
}

// CUDAFile returns [Prelude] followed by [SegmentedLoRASchedule.CUDA].
func (s SegmentedLoRASchedule) CUDAFile() (string, error) {
	src, err := s.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// LoRAKernelArgs returns the order and seg_ arrays of the kernel signature
// for plan.
func LoRAKernelArgs(plan LoRAGatherPlan) (order, start, end, rank, aoff, boff []int32, scale []float32) {
	order = slices.Clone(plan.Order)
	for _, s := range plan.Segments {
		start = append(start, int32(s.Start))
		end = append(end, int32(s.End))
		rank = append(rank, int32(s.Slot.Rank))
		aoff = append(aoff, int32(s.Slot.AOffset))
		boff = append(boff, int32(s.Slot.BOffset))
		scale = append(scale, s.Slot.Scale)
	}
	return order, start, end, rank, aoff, boff, scale
}
