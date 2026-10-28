package tcgen05

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// LdRedOp specifies the inline reduction operator (.redOp) for tcgen05.ld.red
// and tcgen05.ld.red.spcompress instructions (PTX ISA 9.4 section 9.7.18.8.3).
type LdRedOp string

const (
	LdRedMax LdRedOp = "max"
	LdRedMin LdRedOp = "min"
)

func (op LdRedOp) valid() bool {
	return op == LdRedMax || op == LdRedMin
}

// TCGen05LdRedConfig configures a Blackwell Ultra (sm_103a/sm_103f) or Vera
// Rubin (sm_107a/sm_107f) Tensor Memory load-with-reduction kernel modeling
// tcgen05.ld.red.sync.aligned.32x32b.x4.<max|min>{.abs}{.NaN}.f32 (PTX ISA 9.4
// section 9.7.18.8.3).
type TCGen05LdRedConfig struct {
	M      int     // rows in TMEM (defaults to 128, must be a positive multiple of 32)
	N      int     // columns in TMEM (defaults to 128, must be a positive multiple of 32)
	RedOp  LdRedOp // inline reduction operator ("max" or "min", defaults to LdRedMax)
	Abs    bool    // when true, reduces |x| via .abs qualifier
	NaN    bool    // when true, propagates NaN via .NaN qualifier
	Target string  // "sm_103a", "sm_103f", "sm_107a", or "sm_107f" (defaults to "sm_107a")
}

// TCGen05LdRed models the PTX ISA 9.4 §9.7.18.8.3 Tensor Memory load with
// inline row reduction instruction:
//
//	tcgen05.ld.red.sync.aligned.32x32b.x4.<max|min>{.abs}{.NaN}.f32 {r0..r3}, redval, [taddr];
//
// supported on sm_103a/sm_103f and sm_107a/sm_107f.
type TCGen05LdRed = TCGen05LdRedConfig

func (k TCGen05LdRedConfig) norm() (TCGen05LdRedConfig, error) {
	if k.M == 0 {
		k.M = 128
	}
	if k.N == 0 {
		k.N = 128
	}
	if k.RedOp == "" {
		k.RedOp = LdRedMax
	}
	if k.Target == "" {
		k.Target = "sm_107a"
	}
	if k.M <= 0 || k.M%32 != 0 || k.M > 128 {
		return k, fmt.Errorf("tcgen05 ld.red: invalid M=%d (want positive multiple of 32 in 32..128)", k.M)
	}
	if k.N <= 0 || k.N%32 != 0 || k.N > 512 {
		return k, fmt.Errorf("tcgen05 ld.red: invalid N=%d (want positive multiple of 32 in 32..512)", k.N)
	}
	if !k.RedOp.valid() {
		return k, fmt.Errorf("tcgen05 ld.red: unsupported RedOp %q (want %q or %q)", k.RedOp, LdRedMax, LdRedMin)
	}
	switch k.Target {
	case "sm_103a", "sm_103f", "sm_107a", "sm_107f":
	default:
		return k, fmt.Errorf("tcgen05 ld.red: unsupported target %q (want sm_103a, sm_103f, sm_107a, or sm_107f)", k.Target)
	}
	return k, nil
}

// Validate checks that k has valid matrix dimensions, reduction operator, and
// PTX target architecture.
func (k TCGen05LdRedConfig) Validate() error {
	_, err := k.norm()
	return err
}

// Grid returns the single-CTA grid and 128-thread (4-warp) block dimensions.
func (k TCGen05LdRedConfig) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

func (k TCGen05LdRedConfig) ptxInstruction() string {
	var b strings.Builder
	b.WriteString("tcgen05.ld.red.sync.aligned.32x32b.x4.")
	b.WriteString(string(k.RedOp))
	if k.Abs {
		b.WriteString(".abs")
	}
	if k.NaN {
		b.WriteString(".NaN")
	}
	b.WriteString(".f32")
	return b.String()
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05LdRedConfig) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	ptxInst := k.ptxInstruction()
	isMax := 0
	if k.RedOp == LdRedMax {
		isMax = 1
	}
	useAbs := 0
	if k.Abs {
		useAbs = 1
	}
	useNaN := 0
	if k.NaN {
		useNaN = 1
	}

	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	fmt.Fprintf(&b, `
// eg_tcgen05_ld_red_f32: %s Tensor Memory load with inline %s reduction (M=%d, N=%d, abs=%v, NaN=%v).
// PTX instruction: %s {r0..r3}, redval, [taddr];
__global__ void __launch_bounds__(128) eg_tcgen05_ld_red_f32(
    const float* __restrict__ in_tmem,
    float* __restrict__ out_data,
    float* __restrict__ out_row_red,
    int M,
    int N) {
    constexpr int MAX_COLS = %d;
    constexpr int IS_MAX = %d;
    constexpr int USE_ABS = %d;
    constexpr int USE_NAN = %d;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row = 32 * warp + lane;

    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot_tmem, [1]);

    // 1. Allocate TMEM columns via tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32.
    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot_tmem[0], 0u, (uint32_t)MAX_COLS);
    }
    __syncthreads();
    const uint32_t base_taddr = slot_tmem[0];

    // 2. Stage in_tmem [M, N] into TMEM via 4-warp 32-row partitioned tcgen05.st.
    if (row < M) {
        for (int col = 0; col < N; ++col) {
            eg_tcgen05_st_row(tmem, warp, lane, base_taddr + (uint32_t)col, in_tmem[row * N + col]);
        }
    }
    eg_tcgen05_wait_st();
    __syncthreads();

    // 3. Load 4-column chunks from TMEM with inline tcgen05.ld.red reduction.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070) && !defined(EG_EMU)
    if (row < M) {
        for (int col = 0; col < N; col += 4) {
            float r0, r1, r2, r3, redval;
            const uint32_t taddr = ((uint32_t)row << 16) | ((base_taddr + (uint32_t)col) & 0xffffu);
            asm volatile(
                "%s {%%0, %%1, %%2, %%3}, %%4, [%%5];\n"
                "tcgen05.wait::ld.sync.aligned;"
                : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3), "=f"(redval)
                : "r"(taddr)
                : "memory");
            out_data[row * N + col + 0] = r0;
            out_data[row * N + col + 1] = r1;
            out_data[row * N + col + 2] = r2;
            out_data[row * N + col + 3] = r3;
        }
    }
#endif

    // 4. Exact host-emulation and device fallback for out_data [M, N] and out_row_red [M].
    if (row < M) {
        float row_acc = 0.0f;
        bool has_acc = false;
        bool seen_nan = false;
        for (int col = 0; col < N; ++col) {
            const float raw = eg_tcgen05_ld_row(tmem, warp, lane, base_taddr + (uint32_t)col);
            out_data[row * N + col] = raw;
            float v = raw;
            if constexpr (USE_ABS != 0) {
                v = fabsf(v);
            }
            if (isnan(v)) {
                if constexpr (USE_NAN != 0) {
                    seen_nan = true;
                }
                continue;
            }
            if (!has_acc) {
                row_acc = v;
                has_acc = true;
            } else if constexpr (IS_MAX != 0) {
                if (v > row_acc) row_acc = v;
            } else {
                if (v < row_acc) row_acc = v;
            }
        }
        if (seen_nan || !has_acc) {
            out_row_red[row] = __uint_as_float(0x7fc00000u);
        } else {
            out_row_red[row] = row_acc;
        }
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 5. Deallocate TMEM columns.
    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(base_taddr, (uint32_t)MAX_COLS);
    }
}
`, k.Target, k.RedOp, k.M, k.N, k.Abs, k.NaN, ptxInst,
		k.N, isMax, useAbs, useNaN, ptxInst)

	return b.String(), nil
}

// CUDAFile emits the self-contained CUDA C++ translation unit (including
// Prelude) for k, ready for cudahostemu.Run or nvcc.
func (k TCGen05LdRedConfig) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

func reduceRowF32(row []float32, op LdRedOp, useAbs, useNaN bool) float32 {
	var acc float32
	hasAcc := false
	seenNaN := false
	for _, raw := range row {
		v := raw
		if useAbs {
			v = float32(math.Abs(float64(v)))
		}
		if math.IsNaN(float64(v)) {
			if useNaN {
				seenNaN = true
			}
			continue
		}
		if !hasAcc {
			acc = v
			hasAcc = true
		} else if op == LdRedMax {
			if v > acc {
				acc = v
			}
		} else {
			if v < acc {
				acc = v
			}
		}
	}
	if seenNaN || !hasAcc {
		return math.Float32frombits(0x7fc00000)
	}
	return acc
}

// Reference computes the exact host reference for k given row-major input
// in [M, N], returning a copy of in (data [M*N]) and the per-row reduction
// rowRed [M].
func (k TCGen05LdRedConfig) Reference(in []float32) (data []float32, rowRed []float32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, err
	}
	if len(in) != k.M*k.N {
		return nil, nil, fmt.Errorf("tcgen05 ld.red reference: len(in)=%d, want %d (M=%d, N=%d)", len(in), k.M*k.N, k.M, k.N)
	}
	data = slices.Clone(in)
	rowRed = make([]float32, k.M)
	for r := 0; r < k.M; r++ {
		row := in[r*k.N : (r+1)*k.N]
		rowRed[r] = reduceRowF32(row, k.RedOp, k.Abs, k.NaN)
	}
	return data, rowRed, nil
}

// TCGen05LdSpCompress models the PTX ISA 9.4 §9.7.18.8.3 Vera Rubin
// (sm_107a/sm_107f) Tensor Memory load with hardware 2:4 structured sparsity
// compression and optional inline reduction:
//
//	tcgen05.ld{.red}.spcompress.sync.aligned.32x32b.x4.<max|min>.sp::2:4{.abs}.f32.b2
//	    {mdata}, {c0, c1}, {redval,} [taddr];
//
// For each group of 4 consecutive FP32 elements along N, it selects the 2
// elements with the largest (LdRedMax) or smallest (LdRedMin) value or
// magnitude (when Abs is true), breaking ties deterministically by smaller
// column index (0 < 1 < 2 < 3), orders the selected indices as
// 0 <= idx0 < idx1 <= 3, writes the 4-bit metadata nibble idx0 | (idx1 << 2)
// into packed uint32 metadata words (8 groups = 32 columns per uint32 word),
// writes the two selected FP32 values to cdata [M, N/2], and computes the
// per-row reduction rowRed [M].
type TCGen05LdSpCompress struct {
	M             int     // rows in TMEM (defaults to 128, must be in 1..128)
	N             int     // columns in TMEM (defaults to 128, must be a positive multiple of 4)
	RedOp         LdRedOp // "max" or "min" selector for 2:4 pruning and reduction (defaults to LdRedMax)
	Abs           bool    // when true, prunes and reduces by magnitude |x| via .abs
	WithReduction bool    // when true, emits tcgen05.ld.red.spcompress with inline redval output
	Target        string  // "sm_107a" or "sm_107f" (defaults to "sm_107a")
}

func (k TCGen05LdSpCompress) norm() (TCGen05LdSpCompress, error) {
	if k.M == 0 {
		k.M = 128
	}
	if k.N == 0 {
		k.N = 128
	}
	if k.RedOp == "" {
		k.RedOp = LdRedMax
	}
	if k.Target == "" {
		k.Target = "sm_107a"
	}
	if k.M < 1 || k.M > 128 {
		return k, fmt.Errorf("tcgen05 ld.spcompress: invalid M=%d (want 1..128)", k.M)
	}
	if k.N < 4 || k.N%4 != 0 || k.N > 512 {
		return k, fmt.Errorf("tcgen05 ld.spcompress: invalid N=%d (want multiple of 4 in 4..512)", k.N)
	}
	if !k.RedOp.valid() {
		return k, fmt.Errorf("tcgen05 ld.spcompress: unsupported RedOp %q (want %q or %q)", k.RedOp, LdRedMax, LdRedMin)
	}
	switch k.Target {
	case "sm_107a", "sm_107f":
	default:
		return k, fmt.Errorf("tcgen05 ld.spcompress: unsupported target %q (want sm_107a or sm_107f)", k.Target)
	}
	return k, nil
}

// Validate checks that k has valid dimensions, reduction operator, and Rubin
// sm_107a/sm_107f target.
func (k TCGen05LdSpCompress) Validate() error {
	_, err := k.norm()
	return err
}

// Grid returns the single-CTA grid and 128-thread (4-warp) block dimensions.
func (k TCGen05LdSpCompress) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

// MetadataWordsPerRow returns the number of packed uint32 metadata words per
// row: ceil(N / 32), where each uint32 word holds up to 8 groups of 4-bit
// (2x2-bit .b2) indices.
func (k TCGen05LdSpCompress) MetadataWordsPerRow() int {
	n := k.N
	if n <= 0 {
		n = 128
	}
	return (n + 31) / 32
}

func (k TCGen05LdSpCompress) ptxInstruction() string {
	var b strings.Builder
	b.WriteString("tcgen05.ld")
	if k.WithReduction {
		b.WriteString(".red")
	}
	b.WriteString(".spcompress.sync.aligned.32x32b.x4.")
	b.WriteString(string(k.RedOp))
	b.WriteString(".sp::2:4")
	if k.Abs {
		b.WriteString(".abs")
	}
	b.WriteString(".f32.b2")
	return b.String()
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05LdSpCompress) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	ptxInst := k.ptxInstruction()
	allocCols := max(32, ((k.N+31)/32)*32)
	isMax := 0
	if k.RedOp == LdRedMax {
		isMax = 1
	}
	useAbs := 0
	if k.Abs {
		useAbs = 1
	}

	var ptxAsmBlock string
	if k.WithReduction {
		ptxAsmBlock = fmt.Sprintf(`            uint32_t mdata_word = 0u;
            float c0 = 0.0f, c1 = 0.0f, redval = 0.0f;
            const uint32_t taddr = ((uint32_t)row << 16) | ((base_taddr + (uint32_t)(4 * g)) & 0xffffu);
            asm volatile(
                "%s {%%0}, {%%1, %%2}, %%3, [%%4];\n"
                "tcgen05.wait::ld.sync.aligned;"
                : "=r"(mdata_word), "=f"(c0), "=f"(c1), "=f"(redval)
                : "r"(taddr)
                : "memory");`, ptxInst)
	} else {
		ptxAsmBlock = fmt.Sprintf(`            uint32_t mdata_word = 0u;
            float c0 = 0.0f, c1 = 0.0f;
            const uint32_t taddr = ((uint32_t)row << 16) | ((base_taddr + (uint32_t)(4 * g)) & 0xffffu);
            asm volatile(
                "%s {%%0}, {%%1, %%2}, [%%3];\n"
                "tcgen05.wait::ld.sync.aligned;"
                : "=r"(mdata_word), "=f"(c0), "=f"(c1)
                : "r"(taddr)
                : "memory");`, ptxInst)
	}

	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	fmt.Fprintf(&b, `
// eg_tcgen05_ld_spcompress_f32: Rubin %s 2:4 structured sparse TMEM load-and-compress (M=%d, N=%d, op=%s, abs=%v, red=%v).
// PTX instruction: %s
__global__ void __launch_bounds__(128) eg_tcgen05_ld_spcompress_f32(
    const float* __restrict__ in_tmem,
    float* __restrict__ out_cdata,
    unsigned int* __restrict__ out_mdata,
    float* __restrict__ out_row_red,
    int M,
    int N) {
    constexpr int ALLOC_COLS = %d;
    constexpr int IS_MAX = %d;
    constexpr int USE_ABS = %d;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row = 32 * warp + lane;

    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot_tmem, [1]);

    // 1. Allocate TMEM columns via tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32.
    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot_tmem[0], 0u, (uint32_t)ALLOC_COLS);
    }
    __syncthreads();
    const uint32_t base_taddr = slot_tmem[0];

    // 2. Stage in_tmem [M, N] into TMEM.
    if (row < M) {
        for (int col = 0; col < N; ++col) {
            eg_tcgen05_st_row(tmem, warp, lane, base_taddr + (uint32_t)col, in_tmem[row * N + col]);
        }
    }
    eg_tcgen05_wait_st();
    __syncthreads();

    const int num_groups = N / 4;
    const int mdata_words = (N + 31) / 32;

    // 3. Issue Rubin sm_107a PTX tcgen05.ld{.red}.spcompress instruction on device.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1070 && !defined(EG_EMU)
    if (row < M) {
        for (int g = 0; g < num_groups; ++g) {
%s
            out_cdata[row * (N / 2) + 2 * g + 0] = c0;
            out_cdata[row * (N / 2) + 2 * g + 1] = c1;
            out_mdata[row * mdata_words + (g >> 3)] = mdata_word;
        }
    }
#endif

    // 4. Exact 2:4 structured pruning, 2-bit metadata packing, and row reduction.
    if (row < M) {
        for (int w = 0; w < mdata_words; ++w) {
            out_mdata[row * mdata_words + w] = 0u;
        }
        float row_acc = 0.0f;
        bool has_acc = false;
        for (int g = 0; g < num_groups; ++g) {
            float vals[4];
            float scores[4];
            for (int i = 0; i < 4; ++i) {
                const float raw = eg_tcgen05_ld_row(tmem, warp, lane, base_taddr + (uint32_t)(4 * g + i));
                vals[i] = raw;
                const float s = (USE_ABS != 0) ? fabsf(raw) : raw;
                scores[i] = s;
                if (!isnan(s)) {
                    if (!has_acc) {
                        row_acc = s;
                        has_acc = true;
                    } else if constexpr (IS_MAX != 0) {
                        if (s > row_acc) row_acc = s;
                    } else {
                        if (s < row_acc) row_acc = s;
                    }
                }
            }
            int order[4] = {0, 1, 2, 3};
            for (int i = 0; i < 4; ++i) {
                for (int j = i + 1; j < 4; ++j) {
                    const int a = order[i];
                    const int b_idx = order[j];
                    bool b_better = false;
                    if (scores[b_idx] != scores[a]) {
                        if constexpr (IS_MAX != 0) {
                            b_better = (scores[b_idx] > scores[a]);
                        } else {
                            b_better = (scores[b_idx] < scores[a]);
                        }
                    } else {
                        b_better = (b_idx < a);
                    }
                    if (b_better) {
                        order[i] = b_idx;
                        order[j] = a;
                    }
                }
            }
            int idx0 = order[0];
            int idx1 = order[1];
            if (idx0 > idx1) {
                const int tmp = idx0;
                idx0 = idx1;
                idx1 = tmp;
            }
            out_cdata[row * (N / 2) + 2 * g + 0] = vals[idx0];
            out_cdata[row * (N / 2) + 2 * g + 1] = vals[idx1];
            const uint32_t nibble = (uint32_t)(idx0 & 0x3) | ((uint32_t)(idx1 & 0x3) << 2);
            out_mdata[row * mdata_words + (g >> 3)] |= (nibble << (4 * (g & 7)));
        }
        if (out_row_red != nullptr) {
            out_row_red[row] = has_acc ? row_acc : __uint_as_float(0x7fc00000u);
        }
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 5. Deallocate TMEM columns.
    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(base_taddr, (uint32_t)ALLOC_COLS);
    }
}
`, k.Target, k.M, k.N, k.RedOp, k.Abs, k.WithReduction, ptxInst,
		allocCols, isMax, useAbs, ptxAsmBlock)

	return b.String(), nil
}

// CUDAFile emits the self-contained CUDA C++ translation unit (including
// Prelude) for k, ready for cudahostemu.Run or nvcc.
func (k TCGen05LdSpCompress) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

func selectSpCompress2Of4(vals [4]float32, op LdRedOp, useAbs bool) (idx0, idx1 int) {
	var scores [4]float32
	for i := 0; i < 4; i++ {
		if useAbs {
			scores[i] = float32(math.Abs(float64(vals[i])))
		} else {
			scores[i] = vals[i]
		}
	}
	order := [4]int{0, 1, 2, 3}
	better := func(a, b int) bool {
		if scores[a] != scores[b] {
			if op == LdRedMax {
				return scores[a] > scores[b]
			}
			return scores[a] < scores[b]
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

// Reference computes the exact host reference for k given row-major input
// in [M, N], returning:
//   - cdata:  [M * (N/2)] compressed non-zero float32 values (2 per 4-element group),
//   - mdata:  [M * ceil(N/32)] packed uint32 metadata words (4 bits = idx0 | (idx1<<2) per 4-element group),
//   - rowRed: [M] per-row reduction across all N columns.
func (k TCGen05LdSpCompress) Reference(in []float32) (cdata []float32, mdata []uint32, rowRed []float32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, nil, err
	}
	if len(in) != k.M*k.N {
		return nil, nil, nil, fmt.Errorf("tcgen05 ld.spcompress reference: len(in)=%d, want %d (M=%d, N=%d)", len(in), k.M*k.N, k.M, k.N)
	}
	nHalf := k.N / 2
	numGroups := k.N / 4
	mdataWords := (k.N + 31) / 32

	cdata = make([]float32, k.M*nHalf)
	mdata = make([]uint32, k.M*mdataWords)
	rowRed = make([]float32, k.M)

	for r := 0; r < k.M; r++ {
		rowIn := in[r*k.N : (r+1)*k.N]
		rowC := cdata[r*nHalf : (r+1)*nHalf]
		rowM := mdata[r*mdataWords : (r+1)*mdataWords]
		for g := 0; g < numGroups; g++ {
			vals := [4]float32{
				rowIn[4*g+0],
				rowIn[4*g+1],
				rowIn[4*g+2],
				rowIn[4*g+3],
			}
			idx0, idx1 := selectSpCompress2Of4(vals, k.RedOp, k.Abs)
			rowC[2*g+0] = vals[idx0]
			rowC[2*g+1] = vals[idx1]
			nibble := uint32(idx0&0x3) | (uint32(idx1&0x3) << 2)
			rowM[g>>3] |= nibble << (4 * (g & 7))
		}
		rowRed[r] = reduceRowF32(rowIn, k.RedOp, k.Abs, false)
	}
	return cdata, mdata, rowRed, nil
}

// DecompressReference reconstructs the dense [M, N] float32 tensor (with
// unselected positions zeroed) from compressed values cdata [M * (N/2)] and
// packed 2-bit-per-selected-element metadata mdata [M * ceil(N/32)].
func DecompressReference(cdata []float32, mdata []uint32, M, N int) ([]float32, error) {
	if M < 1 || N < 4 || N%4 != 0 {
		return nil, fmt.Errorf("tcgen05 decompress 2:4 f32: invalid shape M=%d N=%d (want M>=1, N>=4 multiple of 4)", M, N)
	}
	nHalf := N / 2
	numGroups := N / 4
	mdataWords := (N + 31) / 32
	if len(cdata) != M*nHalf {
		return nil, fmt.Errorf("tcgen05 decompress 2:4 f32: len(cdata)=%d, want %d", len(cdata), M*nHalf)
	}
	if len(mdata) != M*mdataWords {
		return nil, fmt.Errorf("tcgen05 decompress 2:4 f32: len(mdata)=%d, want %d", len(mdata), M*mdataWords)
	}

	dense := make([]float32, M*N)
	for r := 0; r < M; r++ {
		rowC := cdata[r*nHalf : (r+1)*nHalf]
		rowM := mdata[r*mdataWords : (r+1)*mdataWords]
		rowD := dense[r*N : (r+1)*N]
		for g := 0; g < numGroups; g++ {
			nibble := (rowM[g>>3] >> (4 * (g & 7))) & 0xf
			idx0 := int(nibble & 0x3)
			idx1 := int((nibble >> 2) & 0x3)
			if idx0 >= idx1 {
				return nil, fmt.Errorf("tcgen05 decompress 2:4 f32: row %d group %d invalid metadata indices (%d, %d)", r, g, idx0, idx1)
			}
			rowD[4*g+idx0] = rowC[2*g+0]
			rowD[4*g+idx1] = rowC[2*g+1]
		}
	}
	return dense, nil
}

// DecompressReference is a method alias for [DecompressReference] using k's
// configured M and N dimensions when M or N are omitted.
func (k TCGen05LdSpCompress) DecompressReference(cdata []float32, mdata []uint32, M, N int) ([]float32, error) {
	if M == 0 {
		M = k.M
		if M == 0 {
			M = 128
		}
	}
	if N == 0 {
		N = k.N
		if N == 0 {
			N = 128
		}
	}
	return DecompressReference(cdata, mdata, M, N)
}
