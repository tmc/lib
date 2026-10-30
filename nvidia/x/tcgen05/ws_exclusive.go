package tcgen05

import (
	"fmt"
	"strings"
)

// ZeroColumnMaskDesc models the 64-bit zero-column mask descriptor consumed by
// tcgen05.mma and tcgen05.mma.ws (PTX ISA 9.4 section 9.7.18.4.3).
//
// The 64-bit descriptor partitions each 64-column K-tile into 8 sub-masks
// i in 0..7 of 8 columns each:
//   - Bits 0..23:  StartCount[i] (3 bits at i*3), number of initial non-zero columns in group i.
//   - Bits 24..31: NonZeroMask (bit i == 0 means all columns in group i are used;
//     bit i == 1 means SkipSpan[i]+1 columns starting at StartCount[i] are zeroed).
//   - Bits 32..55: SkipSpan[i] (3 bits at 32 + i*3), storing skip_span - 1.
type ZeroColumnMaskDesc struct {
	StartCount  [8]uint8
	SkipSpan    [8]uint8
	NonZeroMask uint8
}

// Encode packs d into the 64-bit PTX zero-column mask descriptor word.
func (d ZeroColumnMaskDesc) Encode() uint64 {
	var v uint64
	for i := 0; i < 8; i++ {
		v |= uint64(d.StartCount[i]&0x7) << uint(i*3)
		v |= uint64(d.SkipSpan[i]&0x7) << uint(32+i*3)
	}
	v |= uint64(d.NonZeroMask) << 24
	return v
}

// DecodeZeroColumnMaskDesc unpacks a 64-bit PTX zero-column mask descriptor word.
func DecodeZeroColumnMaskDesc(v uint64) ZeroColumnMaskDesc {
	var d ZeroColumnMaskDesc
	for i := 0; i < 8; i++ {
		d.StartCount[i] = uint8((v >> uint(i*3)) & 0x7)
		d.SkipSpan[i] = uint8((v >> uint(32+i*3)) & 0x7)
	}
	d.NonZeroMask = uint8((v >> 24) & 0xff)
	return d
}

// IsColumnZeroed reports whether K-dimension column col is zeroed in operand A
// by the zero-column mask descriptor.
func (d ZeroColumnMaskDesc) IsColumnZeroed(col int) bool {
	if col < 0 {
		return false
	}
	group := (col / 8) & 7
	if (d.NonZeroMask & (1 << uint(group))) == 0 {
		return false
	}
	localCol := col & 7
	start := int(d.StartCount[group] & 0x7)
	span := int(d.SkipSpan[group]&0x7) + 1
	return localCol >= start && localCol < start+span
}

// WSCollectorBuffer identifies one of the 4 hardware B-matrix collector reuse
// buffers in tcgen05.mma.ws (PTX ISA 9.4 section 9.7.18.10.10.3–4).
type WSCollectorBuffer string

const (
	WSCollectorB0 WSCollectorBuffer = "b0"
	WSCollectorB1 WSCollectorBuffer = "b1"
	WSCollectorB2 WSCollectorBuffer = "b2"
	WSCollectorB3 WSCollectorBuffer = "b3"
)

func (b WSCollectorBuffer) valid() bool {
	switch b {
	case WSCollectorB0, WSCollectorB1, WSCollectorB2, WSCollectorB3:
		return true
	default:
		return false
	}
}

// WSCollectorAction specifies the lifecycle operation performed on a
// WSCollectorBuffer during tcgen05.mma.ws execution.
type WSCollectorAction string

const (
	WSActionFill    WSCollectorAction = "fill"
	WSActionUse     WSCollectorAction = "use"
	WSActionLastUse WSCollectorAction = "lastuse"
	WSActionDiscard WSCollectorAction = "discard"
)

func (a WSCollectorAction) valid() bool {
	switch a {
	case WSActionFill, WSActionUse, WSActionLastUse, WSActionDiscard:
		return true
	default:
		return false
	}
}

// TCGen05WeightsStationaryGEMM configures a weights-stationary Tensor Memory
// GEMM kernel using tcgen05.mma.ws{.sp} (PTX ISA 9.4 §9.7.18.10.10.3–4),
// optional exclusive TMEM allocation (tcgen05.alloc.exclusive.cta_group::1,
// §9.7.18.7.1: 576 columns on sm_107, 512 on sm_100/sm_103), and a 64-bit ZeroColumnMaskDesc (§9.7.18.4.3).
type TCGen05WeightsStationaryGEMM struct {
	Name                string
	M, N, K             int
	CollectorBuffer     WSCollectorBuffer
	CollectorAction     WSCollectorAction
	Kind                string
	UseExclusive576Cols bool
	ZeroColumnMask      ZeroColumnMaskDesc
	Sparse              bool // not supported: no sparsity metadata is staged in TMEM
	Target              string
}

func (k TCGen05WeightsStationaryGEMM) norm() (TCGen05WeightsStationaryGEMM, error) {
	if k.Name == "" {
		k.Name = "eg_tcgen05_ws_gemm"
	}
	if k.Name != "eg_tcgen05_ws_gemm" && !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 ws gemm: bad kernel name %q", k.Name)
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
	if k.CollectorBuffer == "" {
		k.CollectorBuffer = WSCollectorB0
	}
	if k.CollectorAction == "" {
		k.CollectorAction = WSActionFill
	}
	if k.Kind == "" {
		k.Kind = "tf32"
	}
	if k.Target == "" {
		k.Target = "sm_107a"
	}
	switch k.Target {
	case "sm_100a", "sm_103a", "sm_107a", "sm_107f":
	default:
		return k, fmt.Errorf("tcgen05 ws gemm %s: unsupported target %q (want sm_100a, sm_103a, sm_107a, or sm_107f)", k.Name, k.Target)
	}
	if !k.CollectorBuffer.valid() {
		return k, fmt.Errorf("tcgen05 ws gemm %s: invalid CollectorBuffer %q (want b0..b3)", k.Name, k.CollectorBuffer)
	}
	if !k.CollectorAction.valid() {
		return k, fmt.Errorf("tcgen05 ws gemm %s: invalid CollectorAction %q (want fill, use, lastuse, or discard)", k.Name, k.CollectorAction)
	}
	switch k.Kind {
	case "f16", "tf32", "f8f6f4", "i8":
	case "ti16":
		if k.Target != "sm_107a" && k.Target != "sm_107f" {
			return k, fmt.Errorf("tcgen05 ws gemm %s: .kind::ti16 requires sm_107a or sm_107f, got %q", k.Name, k.Target)
		}
	default:
		return k, fmt.Errorf("tcgen05 ws gemm %s: invalid Kind %q (want f16, tf32, f8f6f4, i8, or ti16)", k.Name, k.Kind)
	}
	if k.M != 128 || k.N <= 0 || k.N > 256 || k.N%32 != 0 || k.K <= 0 || k.K%16 != 0 {
		return k, fmt.Errorf("tcgen05 ws gemm %s: invalid shape M=%d N=%d K=%d (want M=128, N in 32..256 multiple of 32, K multiple of 16)", k.Name, k.M, k.N, k.K)
	}
	if k.Sparse {
		// tcgen05.mma.ws.sp reads 2:4 metadata from [sp-meta-tmem], whose
		// TMEM layout (PTX ISA 9.4 section 9.7.18.10.9) this kernel does not
		// produce, and A would have to be staged compressed.
		return k, fmt.Errorf("tcgen05 ws gemm %s: sparse form unsupported: kernel does not stage 2:4 sparsity metadata in tmem", k.Name)
	}
	return k, nil
}

// Validate checks that k has valid dimensions, collector qualifiers, kind, and target.
func (k TCGen05WeightsStationaryGEMM) Validate() error {
	_, err := k.norm()
	return err
}

// Grid returns the single-CTA grid and 128-thread (4-warp) block dimensions.
func (k TCGen05WeightsStationaryGEMM) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

// InstDescriptor returns the 32-bit tcgen05.mma.ws instruction descriptor
// (idesc) for k.Kind: TF32, BF16, E4M3, S8, or S1Z4M11 inputs with an F32
// (S32 for i8 and ti16) accumulator, M=128, N=k.N, and the kind's dense K.
func (k TCGen05WeightsStationaryGEMM) InstDescriptor() (uint32, error) {
	k, err := k.norm()
	if err != nil {
		return 0, err
	}
	kind, _ := parseKind(k.Kind)
	d := InstDesc{Kind: kind, DType: ElemF32, M: k.M, N: k.N}
	switch kind {
	case KindTF32:
		d.AType, d.K = ElemTF32, 8
	case KindF16:
		d.AType, d.K = ElemBF16, 16
	case KindF8F6F4:
		d.AType, d.K = ElemE4M3, 32
	case KindI8:
		d.AType, d.DType = ElemS8, ElemS32
	case KindTI16:
		d.AType, d.DType = ElemS1Z4M11, ElemS32
	}
	d.BType = d.AType
	if err := d.ValidateForTarget(k.Target); err != nil {
		return 0, fmt.Errorf("tcgen05 ws gemm %s: %w", k.Name, err)
	}
	return d.Encode()
}

func (k TCGen05WeightsStationaryGEMM) mmaMnemonic() string {
	var b strings.Builder
	b.WriteString("tcgen05.mma.ws.cta_group::1.kind::")
	b.WriteString(k.Kind)
	b.WriteString(".collector::")
	b.WriteString(string(k.CollectorBuffer))
	b.WriteString("::")
	b.WriteString(string(k.CollectorAction))
	return b.String()
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05WeightsStationaryGEMM) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	maskWord := k.ZeroColumnMask.Encode()
	mnemonic := k.mmaMnemonic()
	idesc, err := k.InstDescriptor()
	if err != nil {
		return "", err
	}

	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString(`
// eg_tcgen05_is_col_zeroed evaluates PTX ISA 9.4 §9.7.18.4.3 zero-column mask
// descriptor bitfields for K-dimension column col.
EG_DEV bool eg_tcgen05_is_col_zeroed(uint64_t mask_desc, int col) {
    if (col < 0) return false;
    const int group = (col / 8) & 7;
    const uint32_t non_zero_mask = (uint32_t)((mask_desc >> 24) & 0xffu);
    if (((non_zero_mask >> group) & 1u) == 0u) return false;
    const int local_col = col & 7;
    const int start = (int)((mask_desc >> (group * 3)) & 0x7u);
    const int span = (int)((mask_desc >> (32 + group * 3)) & 0x7u) + 1;
    return local_col >= start && local_col < (start + span);
}
`)

	if k.UseExclusive576Cols {
		b.WriteString(`
// EG_TCGEN05_EXCLUSIVE_COLS is the largest .exclusive allocation: 576 columns
// on sm_107 and 512 on sm_100/sm_103 (PTX ISA 9.4 section 9.7.18.7.1, Table 58).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070)
#define EG_TCGEN05_EXCLUSIVE_COLS 576
#else
#define EG_TCGEN05_EXCLUSIVE_COLS 512
#endif

// eg_tcgen05_alloc_exclusive claims the CTA's whole allocation permit with
// tcgen05.alloc.exclusive. tcgen05.mma.ws is single-CTA, so the allocation
// uses cta_group::1; a kernel may not mix CTA groups.
EG_DEV void eg_tcgen05_alloc_exclusive(float* tmem, uint32_t* slot) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070) && !defined(EG_EMU)
    (void)tmem;
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(slot);
    asm volatile("tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(s), "n"(EG_TCGEN05_EXCLUSIVE_COLS) : "memory");
#else
    eg_tcgen05_alloc_cta1(tmem, slot, 0u, 256u);
#endif
}

// eg_tcgen05_dealloc_exclusive releases an allocation made by
// eg_tcgen05_alloc_exclusive; .exclusive allocations must be freed with
// tcgen05.dealloc.exclusive of the same column count.
EG_DEV void eg_tcgen05_dealloc_exclusive(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070) && !defined(EG_EMU)
    asm volatile("tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 :: "r"(taddr), "n"(EG_TCGEN05_EXCLUSIVE_COLS));
#else
    (void)taddr;
    __syncwarp();
#endif
}
`)
	}

	fmt.Fprintf(&b, "// %s: %s weights-stationary GEMM (M=%d, N=%d, K=%d, mma=%s, exclusive576=%v, zmask=0x%016x).\n",
		k.Name, k.Target, k.M, k.N, k.K, mnemonic, k.UseExclusive576Cols, maskWord)
	fmt.Fprintf(&b, `__global__ void __launch_bounds__(128) %s(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out) {
    constexpr int M = %d, N = %d, K = %d;
    constexpr uint64_t ZERO_COL_MASK = 0x%016xULL;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    EG_TCGEN05_SHARED(float, smem_a, [M * K]);
    EG_TCGEN05_SHARED(float, smem_b, [K * N]);
    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot_d, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    if (warp == 0) {
`, k.Name, k.M, k.N, k.K, maskWord)

	if k.UseExclusive576Cols {
		b.WriteString("        eg_tcgen05_alloc_exclusive(tmem, &slot_d[0]);\n")
	} else {
		fmt.Fprintf(&b, "        eg_tcgen05_alloc_cta1(tmem, &slot_d[0], 0u, %du);\n", k.allocCols())
	}

	b.WriteString(`    }
    __syncthreads();
    const uint32_t d_taddr = slot_d[0];

    for (int i = tid; i < M * K; i += 128) smem_a[i] = a[i];
    for (int i = tid; i < K * N; i += 128) smem_b[i] = b[i];
    __syncthreads();

    if (warp == 0 && eg_elect_one_sync()) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070) && !defined(EG_EMU)
        const uint64_t desc_a = eg_wgmma_desc(smem_a, 128, 1024);
        const uint64_t desc_b = eg_wgmma_desc(smem_b, 128, 1024);
        const uint32_t idesc = `)
	fmt.Fprintf(&b, "0x%08xu;\n", idesc)
	if maskWord != 0 {
		fmt.Fprintf(&b, `        const uint64_t zmask = ZERO_COL_MASK;
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "%s [%%0], %%1, %%2, %%3, p, %%4;\n"
            "}"
            :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(idesc), "l"(zmask)
            : "memory");
`, mnemonic)
	} else {
		fmt.Fprintf(&b, `        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "%s [%%0], %%1, %%2, %%3, p;\n"
            "}"
            :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(idesc)
            : "memory");
`, mnemonic)
	}
	b.WriteString(`#else
        for (int r = 0; r < M; ++r) {
            for (int n = 0; n < N; ++n) {
                float acc = tmem[r * 512 + d_taddr + n];
                for (int kk = 0; kk < K; ++kk) {
                    const float av = eg_tcgen05_is_col_zeroed(ZERO_COL_MASK, kk) ? 0.0f : smem_a[r * K + kk];
                    acc += av * smem_b[kk * N + n];
                }
                tmem[r * 512 + d_taddr + n] = acc;
            }
        }
#endif
        eg_tcgen05_commit_cta1(&mbar[0]);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    const int row = 32 * warp + lane;
    for (int col = 0; col < N; ++col) {
        out[row * N + col] = eg_tcgen05_ld_row(tmem, warp, lane, d_taddr + (uint32_t)col);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    if (warp == 0) {
`)
	if k.UseExclusive576Cols {
		b.WriteString("        eg_tcgen05_dealloc_exclusive(d_taddr);\n")
	} else {
		fmt.Fprintf(&b, "        eg_tcgen05_dealloc_cta1(d_taddr, %du);\n", k.allocCols())
	}
	b.WriteString(`    }
}
`)

	return b.String(), nil
}

// CUDAFile emits the self-contained CUDA C++ translation unit (including
// Prelude) for k, ready for cudahostemu.Run or nvcc.
func (k TCGen05WeightsStationaryGEMM) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// Reference computes C[m, n] = sum_k A_masked[m, k] * B[k, n] where
// A_masked[m, k] = 0 when k.ZeroColumnMask.IsColumnZeroed(k) is true.
func (k TCGen05WeightsStationaryGEMM) Reference(a, b []float32) ([]float32, error) {
	k, err := k.norm()
	if err != nil {
		return nil, err
	}
	if len(a) != k.M*k.K {
		return nil, fmt.Errorf("tcgen05 ws gemm %s reference: len(a)=%d, want %d", k.Name, len(a), k.M*k.K)
	}
	if len(b) != k.K*k.N {
		return nil, fmt.Errorf("tcgen05 ws gemm %s reference: len(b)=%d, want %d", k.Name, len(b), k.K*k.N)
	}
	out := make([]float32, k.M*k.N)
	for r := 0; r < k.M; r++ {
		for n := 0; n < k.N; n++ {
			var acc float32
			for kk := 0; kk < k.K; kk++ {
				av := a[r*k.K+kk]
				if k.ZeroColumnMask.IsColumnZeroed(kk) {
					av = 0
				}
				acc += av * b[kk*k.N+n]
			}
			out[r*k.N+n] = acc
		}
	}
	return out, nil
}

// allocCols reports the TMEM columns k allocates without .exclusive: the N
// accumulator columns.
func (k TCGen05WeightsStationaryGEMM) allocCols() int {
	return k.N
}
