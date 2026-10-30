package tcgen05

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// selectTop2Of4 returns the column indices (0 <= idx0 < idx1 <= 3) of the two
// elements in mag[0..3] with the largest magnitude, breaking ties
// deterministically by lower index (0 < 1 < 2 < 3).
func selectTop2Of4(mag [4]float64, tieMag [4]float64) (idx0, idx1 int) {
	order := [4]int{0, 1, 2, 3}
	better := func(a, b int) bool {
		if mag[a] != mag[b] {
			return mag[a] > mag[b]
		}
		if tieMag[a] != tieMag[b] {
			return tieMag[a] > tieMag[b]
		}
		return a < b
	}
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			if better(order[j], order[i]) {
				order[i], order[j] = order[j], order[i]
			}
		}
	}
	idx0, idx1 = order[0], order[1]
	if idx0 > idx1 {
		idx0, idx1 = idx1, idx0
	}
	return idx0, idx1
}

// Prune2To4F32 prunes a dense [M, K] float32 matrix along the reduction
// dimension K to NVIDIA 2:4 structured sparsity. In each contiguous 4-element
// group a[r, 4g..4g+3], it selects the 2 elements with largest |v| (breaking
// ties deterministically by lower index 0 < 1 < 2 < 3) and orders the kept
// indices as 0 <= idx0 < idx1 <= 3.
//
// It returns:
//   - compressed: [M, K/2] non-zero values (2 per 4-element group),
//   - meta:       [M, K/32] uint32 metadata words (8 groups x 4-bit nibble per word),
//   - prunedDense:[M, K] dense matrix with the pruned elements zeroed.
func Prune2To4F32(a []float32, M, K int) (compressed []float32, meta []uint32, prunedDense []float32, err error) {
	if M < 1 || K < 32 || K%32 != 0 {
		return nil, nil, nil, fmt.Errorf("tcgen05 prune 2:4 f32: invalid shape M=%d K=%d (want M>=1, K>=32 multiple of 32)", M, K)
	}
	if len(a) != M*K {
		return nil, nil, nil, fmt.Errorf("tcgen05 prune 2:4 f32: len(a)=%d, want %d", len(a), M*K)
	}
	kHalf := K / 2
	kMeta := K / 32
	compressed = make([]float32, M*kHalf)
	meta = make([]uint32, M*kMeta)
	prunedDense = make([]float32, M*K)

	numGroups := K / 4
	for r := 0; r < M; r++ {
		rowA := a[r*K : (r+1)*K]
		rowComp := compressed[r*kHalf : (r+1)*kHalf]
		rowMeta := meta[r*kMeta : (r+1)*kMeta]
		rowDense := prunedDense[r*K : (r+1)*K]
		for g := 0; g < numGroups; g++ {
			base := 4 * g
			var mag [4]float64
			for i := 0; i < 4; i++ {
				mag[i] = math.Abs(float64(rowA[base+i]))
			}
			idx0, idx1 := selectTop2Of4(mag, mag)
			v0 := rowA[base+idx0]
			v1 := rowA[base+idx1]
			rowComp[2*g+0] = v0
			rowComp[2*g+1] = v1
			rowDense[base+idx0] = v0
			rowDense[base+idx1] = v1

			nibble := uint32(idx0&0x3) | (uint32(idx1&0x3) << 2)
			rowMeta[g>>3] |= nibble << (4 * (g & 7))
		}
	}
	return compressed, meta, prunedDense, nil
}

// Prune2To4NVFP4 prunes a packed NVFP4 tensor aPacked [M, K/2] with E4M3 block
// scales aScales [M, K/16] along K to 2:4 structured sparsity:
//   - compPacked: [M, K/4] bytes (one byte holding the 2 kept 4-bit E2M1 nibbles per 4-element group),
//   - compScales: [M, K/16] E4M3 block scales,
//   - meta:       [M, K/32] uint32 metadata words (8 groups x 4-bit nibble per word).
func Prune2To4NVFP4(aPacked []uint8, aScales []uint8, M, K int) (compPacked []uint8, compScales []uint8, meta []uint32, err error) {
	if M < 1 || K < 32 || K%32 != 0 {
		return nil, nil, nil, fmt.Errorf("tcgen05 prune 2:4 nvfp4: invalid shape M=%d K=%d (want M>=1, K>=32 multiple of 32)", M, K)
	}
	kBytes := K / 2
	kBlocks := K / 16
	if len(aPacked) != M*kBytes {
		return nil, nil, nil, fmt.Errorf("tcgen05 prune 2:4 nvfp4: len(aPacked)=%d, want %d", len(aPacked), M*kBytes)
	}
	if len(aScales) != M*kBlocks {
		return nil, nil, nil, fmt.Errorf("tcgen05 prune 2:4 nvfp4: len(aScales)=%d, want %d", len(aScales), M*kBlocks)
	}

	kCompBytes := K / 4
	kMeta := K / 32
	compPacked = make([]uint8, M*kCompBytes)
	compScales = make([]uint8, len(aScales))
	copy(compScales, aScales)
	meta = make([]uint32, M*kMeta)

	numGroups := K / 4
	for r := 0; r < M; r++ {
		rowPacked := aPacked[r*kBytes : (r+1)*kBytes]
		rowScales := aScales[r*kBlocks : (r+1)*kBlocks]
		rowComp := compPacked[r*kCompBytes : (r+1)*kCompBytes]
		rowMeta := meta[r*kMeta : (r+1)*kMeta]
		for g := 0; g < numGroups; g++ {
			b0 := rowPacked[2*g+0]
			b1 := rowPacked[2*g+1]
			codes := [4]uint8{
				b0 & 0xf,
				(b0 >> 4) & 0xf,
				b1 & 0xf,
				(b1 >> 4) & 0xf,
			}
			scale := math.Abs(float64(FromE4M3Bits(rowScales[(4*g)/16])))
			var mag, rawMag [4]float64
			for i := 0; i < 4; i++ {
				rawMag[i] = math.Abs(float64(FromE2M1(codes[i])))
				mag[i] = rawMag[i] * scale
			}
			idx0, idx1 := selectTop2Of4(mag, rawMag)
			rowComp[g] = (codes[idx0] & 0xf) | ((codes[idx1] & 0xf) << 4)
			nibble := uint32(idx0&0x3) | (uint32(idx1&0x3) << 2)
			rowMeta[g>>3] |= nibble << (4 * (g & 7))
		}
	}
	return compPacked, compScales, meta, nil
}

// Unpack2To4F32 reconstructs the dense [M, K] float32 matrix from a 2:4
// compressed value matrix compressed [M, K/2] and packed metadata meta [M, K/32].
func Unpack2To4F32(compressed []float32, meta []uint32, M, K int) ([]float32, error) {
	if M < 1 || K < 32 || K%32 != 0 {
		return nil, fmt.Errorf("tcgen05 unpack 2:4 f32: invalid shape M=%d K=%d (want M>=1, K>=32 multiple of 32)", M, K)
	}
	kHalf := K / 2
	kMeta := K / 32
	if len(compressed) != M*kHalf {
		return nil, fmt.Errorf("tcgen05 unpack 2:4 f32: len(compressed)=%d, want %d", len(compressed), M*kHalf)
	}
	if len(meta) != M*kMeta {
		return nil, fmt.Errorf("tcgen05 unpack 2:4 f32: len(meta)=%d, want %d", len(meta), M*kMeta)
	}

	dense := make([]float32, M*K)
	numGroups := K / 4
	for r := 0; r < M; r++ {
		rowComp := compressed[r*kHalf : (r+1)*kHalf]
		rowMeta := meta[r*kMeta : (r+1)*kMeta]
		rowDense := dense[r*K : (r+1)*K]
		for g := 0; g < numGroups; g++ {
			nibble := (rowMeta[g>>3] >> (4 * (g & 7))) & 0xf
			idx0 := int(nibble & 0x3)
			idx1 := int((nibble >> 2) & 0x3)
			if idx0 >= idx1 {
				return nil, fmt.Errorf("tcgen05 unpack 2:4 f32: row %d group %d invalid metadata indices (%d, %d)", r, g, idx0, idx1)
			}
			rowDense[4*g+idx0] = rowComp[2*g+0]
			rowDense[4*g+idx1] = rowComp[2*g+1]
		}
	}
	return dense, nil
}

// TCGen05SparseGEMM emits a Blackwell sm_100a/sm_103a 2:4 structured sparse
// Tensor Memory (TMEM) GEMM kernel using tcgen05.mma.sp.
type TCGen05SparseGEMM struct {
	Name             string
	M, N, K          int    // M=128, N=128, K multiple of 64 in 64..256
	Target           string // "sm_100a" or "sm_103a" (defaults to "sm_100a")
	CTAGroup         int    // 1 or 2 (defaults to 1)
	BlockScaledNVFP4 bool   // when true, emits kind::mxf4nvf4.block_scale (or .scale_vec::4X on sm_103a)
}

func (k TCGen05SparseGEMM) norm() (TCGen05SparseGEMM, error) {
	if k.Name == "" {
		k.Name = "tcgen05_sparse_gemm"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 sparse gemm: bad kernel name %q", k.Name)
	}
	if k.M == 0 {
		k.M = 128
	}
	if k.N == 0 {
		k.N = 128
	}
	if k.K == 0 {
		k.K = 64
	}
	if k.M != 128 || k.N != 128 || k.K < 64 || k.K > 256 || k.K%64 != 0 {
		return k, fmt.Errorf("tcgen05 sparse gemm %s: invalid shape M=%d N=%d K=%d (want M=128, N=128, K in 64..256 multiple of 64)", k.Name, k.M, k.N, k.K)
	}
	if k.Target == "" {
		k.Target = "sm_100a"
	}
	switch k.Target {
	case "sm_100a", "sm_103a", "sm_107a", "sm_100f", "sm_103f", "sm_107f":
	default:
		return k, fmt.Errorf("tcgen05 sparse gemm %s: unsupported target %q (want sm_100a, sm_103a, or sm_107a)", k.Name, k.Target)
	}
	if k.CTAGroup == 0 {
		k.CTAGroup = 1
	}
	if k.CTAGroup != 1 && k.CTAGroup != 2 {
		return k, fmt.Errorf("tcgen05 sparse gemm %s: invalid CTAGroup %d (want 1 or 2)", k.Name, k.CTAGroup)
	}
	return k, nil
}

// InstDescriptor returns the 32-bit tcgen05.mma.sp instruction descriptor
// (idesc) for the kind k emits: .kind::tf32 (TF32 x TF32 -> F32, K=16) or,
// with BlockScaledNVFP4, .kind::mxf4nvf4 (E2M1 x E2M1, UE4M3 scales, K=128)
// with the target's sparsity version (v0 on sm_100/sm_103, v1 on sm_107).
func (k TCGen05SparseGEMM) InstDescriptor() (uint32, error) {
	k, err := k.norm()
	if err != nil {
		return 0, err
	}
	d := InstDesc{
		Kind:   KindTF32,
		Sparse: true,
		DType:  ElemF32,
		AType:  ElemTF32,
		BType:  ElemTF32,
		M:      k.M,
		N:      k.N,
		K:      16,
	}
	if k.BlockScaledNVFP4 {
		d = InstDesc{
			Kind:      KindMXF4NVF4,
			Sparse:    true,
			AType:     ElemE2M1,
			BType:     ElemE2M1,
			M:         k.M,
			N:         k.N,
			K:         128,
			ScaleType: ScaleUE4M3,
		}
		if k.Target == "sm_107a" || k.Target == "sm_107f" {
			d.SparsityVersion = 1
		}
	}
	if err := d.ValidateForTarget(k.Target); err != nil {
		return 0, fmt.Errorf("tcgen05 sparse gemm %s: %w", k.Name, err)
	}
	return d.Encode()
}

// Grid returns the launch grid and 128-thread (4-warp) block dimensions.
func (k TCGen05SparseGEMM) Grid() (grid, block [3]int) {
	if k.CTAGroup == 2 {
		return [3]int{2, 1, 1}, [3]int{128, 1, 1}
	}
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

// TraceEvents returns the formal sm_100a/sm_103a tcgen05 hardware protocol
// event trace for verification with cudatmem.VerifyTrace.
func (k TCGen05SparseGEMM) TraceEvents() []cudatmem.TraceEvent {
	ctaGroup := k.CTAGroup
	if ctaGroup != 2 {
		ctaGroup = 1
	}
	if ctaGroup == 2 {
		events := []cudatmem.TraceEvent{
			{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
			{Kind: cudatmem.EventAlloc, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128},
			{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 128, Columns: 32, Rows: 128},
			{Kind: cudatmem.EventAlloc, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 128, Columns: 32, Rows: 128},
			{Kind: cudatmem.EventClusterSync, CTAGroup: 2},
			{Kind: cudatmem.EventCp, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 2, Offset: 128, Columns: 32, Rows: 128},
			{Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 2, Offset: 0, Columns: 128, Rows: 128, HasScaleInTMEM: k.BlockScaledNVFP4, ScaleOffset: 128},
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
			cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 128, Columns: 32, Rows: 128},
			cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 1, Warp: 0, ActiveThreads: 32, CTAGroup: 2, Offset: 128, Columns: 32, Rows: 128},
		)
		return events
	}

	events := []cudatmem.TraceEvent{
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		{Kind: cudatmem.EventCp, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		{Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128, HasScaleInTMEM: k.BlockScaledNVFP4, ScaleOffset: 128},
		{Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
	}
	for w := range 4 {
		events = append(events, cudatmem.TraceEvent{
			Kind:          cudatmem.EventLd,
			CTA:           0,
			Warp:          w,
			ActiveThreads: 32,
			CTAGroup:      1,
			Offset:        0,
			Columns:       128,
			RowStart:      uint32(32 * w),
			Rows:          32,
		})
	}
	events = append(events,
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
	)
	return events
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05SparseGEMM) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	idesc, err := k.InstDescriptor()
	if err != nil {
		return "", err
	}

	mmaKind := "kind::tf32"
	if k.BlockScaledNVFP4 {
		if k.Target == "sm_103a" || k.Target == "sm_103f" || k.Target == "sm_107a" || k.Target == "sm_107f" {
			mmaKind = "kind::mxf4nvf4.block_scale.scale_vec::4X"
		} else {
			mmaKind = "kind::mxf4nvf4.block_scale.scale_vec::2X"
		}
	}
	ctaGroupStr := fmt.Sprintf("cta_group::%d", k.CTAGroup)

	var mmaStmt string
	if k.BlockScaledNVFP4 {
		mmaStmt = fmt.Sprintf(`        const uint32_t sfa_taddr = e_taddr + 16u;
        const uint32_t sfb_taddr = e_taddr + 24u;
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.sp.%s.%s [%%0], %%1, %%2, [%%3], %%4, [%%5], [%%6], p;\n"
            "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(e_taddr), "r"(idesc), "r"(sfa_taddr), "r"(sfb_taddr) : "memory");`,
			ctaGroupStr, mmaKind)
	} else {
		mmaStmt = fmt.Sprintf(`        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.sp.%s.%s [%%0], %%1, %%2, [%%3], %%4, p;\n"
            "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(e_taddr), "r"(idesc) : "memory");`,
			ctaGroupStr, mmaKind)
	}

	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: Blackwell %s 2:4 structured sparse GEMM (M=%d, N=%d, K=%d, %s, %s).\n",
		k.Name, k.Target, k.M, k.N, k.K, ctaGroupStr, mmaKind)
	fmt.Fprintf(&b, `__global__ void __launch_bounds__(128) %s(
    const float* __restrict__ a_compressed,
    const uint32_t* __restrict__ a_meta,
    const float* __restrict__ b,
    float* __restrict__ out) {
    constexpr int M = %d, N = %d, K = %d;
    constexpr int K_HALF = K / 2, K_META = K / 32, K_GROUPS = K / 4;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    EG_TCGEN05_SHARED(float, smem_a_comp, [M * K_HALF]);
    EG_TCGEN05_SHARED(uint32_t, smem_a_meta, [M * K_META]);
    EG_TCGEN05_SHARED(float, smem_b, [K * N]);
    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot_d, [1]);
    EG_TCGEN05_SHARED(uint32_t, slot_e, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    // 1. Allocate 128 accumulator columns (offset 0) and 32 metadata/scale columns (offset 128).
    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot_d[0], 0u, 128u);
        eg_tcgen05_alloc_cta1(tmem, &slot_e[0], 128u, 32u);
    }
    __syncthreads();
    const uint32_t d_taddr = slot_d[0];
    const uint32_t e_taddr = slot_e[0];

    // 2. Stage 2:4 compressed operand A [M, K/2], metadata [M, K/32], and dense B [K, N] in shared memory.
    for (int i = tid; i < M * K_HALF; i += 128) smem_a_comp[i] = a_compressed[i];
    for (int i = tid; i < M * K_META; i += 128) smem_a_meta[i] = a_meta[i];
    for (int i = tid; i < K * N; i += 128) smem_b[i] = b[i];
    __syncthreads();

    // 3. Single elected thread stages 2:4 metadata into TMEM [e_taddr] and issues tcgen05.mma.sp.
    if (warp == 0 && eg_elect_one_sync()) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
        const uint64_t desc_meta = eg_wgmma_desc(smem_a_meta, 16, 128);
        asm volatile("tcgen05.cp.%s.32x128b.warpx4 [%%0], %%1;"
                     :: "r"(e_taddr), "l"(desc_meta) : "memory");
        const uint64_t desc_a = eg_wgmma_desc(smem_a_comp, (uint32_t)(K_HALF * 4), (uint32_t)(8 * K_HALF * 4));
        const uint64_t desc_b = eg_wgmma_desc(smem_b, (uint32_t)(N * 4), (uint32_t)(8 * N * 4));
        const uint32_t idesc = 0x%08xu;
%s
#else
        (void)e_taddr;
        for (int r = 0; r < M; ++r) {
            for (int n = 0; n < N; ++n) {
                float acc = tmem[r * 512 + d_taddr + n];
                for (int g = 0; g < K_GROUPS; ++g) {
                    const uint32_t meta_word = smem_a_meta[r * K_META + (g >> 3)];
                    const uint32_t nib = (meta_word >> (4 * (g & 7))) & 0xfu;
                    const int idx0 = (int)(nib & 0x3u);
                    const int idx1 = (int)((nib >> 2) & 0x3u);
                    const float v0 = smem_a_comp[r * K_HALF + 2 * g + 0];
                    const float v1 = smem_a_comp[r * K_HALF + 2 * g + 1];
                    const int k0 = 4 * g + idx0;
                    const int k1 = 4 * g + idx1;
                    acc += v0 * smem_b[k0 * N + n] + v1 * smem_b[k1 * N + n];
                }
                tmem[r * 512 + d_taddr + n] = acc;
            }
        }
#endif
        eg_tcgen05_commit_cta1(&mbar[0]);
    }
    __syncthreads();

    // 4. 4-warp 32-row partitioned tcgen05.ld epilogue.
    const int row = 32 * warp + lane;
    for (int col = 0; col < N; ++col) {
        out[row * N + col] = eg_tcgen05_ld_row(tmem, warp, lane, d_taddr + (uint32_t)col);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 5. Release TMEM allocations.
    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(d_taddr, 128u);
        eg_tcgen05_dealloc_cta1(e_taddr, 32u);
    }
}
`, k.Name, k.M, k.N, k.K, ctaGroupStr, idesc, mmaStmt)
	return b.String(), nil
}

// CUDAFile emits the self-contained CUDA C++ translation unit (including
// Prelude) for k, ready for cudahostemu.Run or nvcc.
func (k TCGen05SparseGEMM) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// Reference computes the 2:4 metadata-indexed sparse matrix product
// C[M, N] = A_sparse[M, K] * B[K, N] from compressed values aCompressed [M, K/2],
// packed 2-bit index metadata aMeta [M, K/32], and dense B [K, N].
func (k TCGen05SparseGEMM) Reference(aCompressed []float32, aMeta []uint32, b []float32) ([]float32, error) {
	k, err := k.norm()
	if err != nil {
		return nil, err
	}
	kHalf := k.K / 2
	kMeta := k.K / 32
	if len(aCompressed) != k.M*kHalf {
		return nil, fmt.Errorf("tcgen05 sparse gemm %s reference: len(aCompressed)=%d, want %d", k.Name, len(aCompressed), k.M*kHalf)
	}
	if len(aMeta) != k.M*kMeta {
		return nil, fmt.Errorf("tcgen05 sparse gemm %s reference: len(aMeta)=%d, want %d", k.Name, len(aMeta), k.M*kMeta)
	}
	if len(b) != k.K*k.N {
		return nil, fmt.Errorf("tcgen05 sparse gemm %s reference: len(b)=%d, want %d", k.Name, len(b), k.K*k.N)
	}

	out := make([]float32, k.M*k.N)
	numGroups := k.K / 4
	for r := 0; r < k.M; r++ {
		rowComp := aCompressed[r*kHalf : (r+1)*kHalf]
		rowMeta := aMeta[r*kMeta : (r+1)*kMeta]
		for n := 0; n < k.N; n++ {
			var acc float32
			for g := 0; g < numGroups; g++ {
				nib := (rowMeta[g>>3] >> (4 * (g & 7))) & 0xf
				idx0 := int(nib & 0x3)
				idx1 := int((nib >> 2) & 0x3)
				if idx0 >= idx1 {
					return nil, fmt.Errorf("tcgen05 sparse gemm %s reference: row %d group %d invalid metadata indices (%d, %d)", k.Name, r, g, idx0, idx1)
				}
				v0 := rowComp[2*g+0]
				v1 := rowComp[2*g+1]
				k0 := 4*g + idx0
				k1 := 4*g + idx1
				acc += v0*b[k0*k.N+n] + v1*b[k1*k.N+n]
			}
			out[r*k.N+n] = acc
		}
	}
	return out, nil
}
