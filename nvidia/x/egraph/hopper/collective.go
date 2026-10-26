package hopper

import (
	"fmt"
	"math"
	"strings"
)

// CollectivePrelude provides device helpers for NVLink symmetric memory
// (multimem.ld_reduce and peer pointer addressing) across Hopper/Blackwell
// GPUs (__CUDA_ARCH__ >= 900) and cudahostemu multi-rank host emulation.
const CollectivePrelude = `#ifndef EG_COLLECTIVE_PRELUDE
#define EG_COLLECTIVE_PRELUDE

#if defined(EG_HOST_EMU) || defined(EG_EMU)
EG_DEV int eg_nvlink_rank() { return eg_emu_rank(); }
template <typename T>
EG_DEV T* eg_nvlink_local_ptr(T* base, size_t elems_per_rank) {
    return eg_emu_local_ptr(base, elems_per_rank);
}
template <typename T>
EG_DEV const T* eg_nvlink_peer_ptr(const T* base, int rank, int my_rank, size_t bytes_per_rank) {
    return eg_emu_peer_ptr(base, rank, my_rank, bytes_per_rank);
}
#else
EG_DEV int eg_nvlink_rank() { return 0; }
template <typename T>
EG_DEV T* eg_nvlink_local_ptr(T* base, size_t) { return base; }
template <typename T>
EG_DEV const T* eg_nvlink_peer_ptr(const T* base, int rank, int my_rank, size_t bytes_per_rank) {
    return reinterpret_cast<const T*>(
        reinterpret_cast<const char*>(base) +
        ((ptrdiff_t)rank - (ptrdiff_t)my_rank) * (ptrdiff_t)bytes_per_rank);
}
#endif

EG_DEV float eg_multimem_ld_reduce_add_f32(const float* sym_base, size_t elem_idx) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return eg_emu_multimem_ld_reduce_add_f32(sym_base, elem_idx);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float r;
    asm volatile("multimem.ld_reduce.global.add.f32 %0, [%1];" : "=f"(r) : "l"(sym_base + elem_idx));
    return r;
#else
    return sym_base[elem_idx];
#endif
}

EG_DEV void eg_multimem_ld_reduce_add_bf16x2(const uint16_t* sym_base, size_t elem_idx, float* lo, float* hi) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    const float2 pair = eg_emu_multimem_ld_reduce_add_bf16x2(sym_base, elem_idx);
    *lo = pair.x;
    *hi = pair.y;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // multimem.ld_reduce.global.add.bf16x2 (.v2.bf16)
    uint32_t pair;
    asm volatile("multimem.ld_reduce.global.add.v2.bf16 %0, [%1];" : "=r"(pair) : "l"(sym_base + elem_idx));
    *lo = eg_bf16_to_f32(pair);
    *hi = eg_bf16_hi_to_f32(pair);
#else
    *lo = eg_bf16_to_f32(sym_base[elem_idx]);
    *hi = eg_bf16_to_f32(sym_base[elem_idx + 1]);
#endif
}

EG_DEV float eg_multimem_ld_reduce_add_bf16(const uint16_t* sym_base, size_t elem_idx) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return (float)eg_emu_multimem_ld_reduce_add_bf16(sym_base, elem_idx);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // multimem.ld_reduce.global.add.bf16x2 (.v2.bf16)
    uint32_t pair;
    const size_t base_idx = elem_idx & ~size_t(1);
    asm volatile("multimem.ld_reduce.global.add.v2.bf16 %0, [%1];" : "=r"(pair) : "l"(sym_base + base_idx));
    return (elem_idx & 1) ? eg_bf16_hi_to_f32(pair) : eg_bf16_to_f32(pair);
#else
    return eg_bf16_to_f32(sym_base[elem_idx]);
#endif
}

#endif // EG_COLLECTIVE_PRELUDE
`

// AllReduceResidualRMSNorm fuses NVLink multimem.ld_reduce (across Ranks GPUs)
// with residual addition and RMSNorm into a single kernel launch.
type AllReduceResidualRMSNorm struct {
	Name    string
	Cols    int
	Ranks   int // e.g., 2, 4, 8
	Eps     float32
	DType   DType // BF16 or F32
	Threads int
}

func (k AllReduceResidualRMSNorm) norm() (AllReduceResidualRMSNorm, error) {
	if k.Name == "" {
		k.Name = "allreduce_residual_rmsnorm"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("allreduce residual rmsnorm: bad name %q", k.Name)
	}
	if k.Cols < 2 || k.Cols%2 != 0 {
		return k, fmt.Errorf("allreduce residual rmsnorm %s: cols %d must be a positive multiple of 2", k.Name, k.Cols)
	}
	if k.Ranks < 1 {
		return k, fmt.Errorf("allreduce residual rmsnorm %s: ranks %d must be >= 1", k.Name, k.Ranks)
	}
	if k.Eps == 0 {
		k.Eps = 1e-6
	}
	if k.DType == 0 {
		k.DType = F32
	}
	if k.DType != F32 && k.DType != BF16 {
		return k, fmt.Errorf("allreduce residual rmsnorm %s: dtype %v must be F32 or BF16", k.Name, k.DType)
	}
	if k.Threads == 0 {
		k.Threads = 256
	}
	if k.Threads < 32 || k.Threads > 1024 || k.Threads%32 != 0 {
		return k, fmt.Errorf("allreduce residual rmsnorm %s: threads %d must be a multiple of 32 in 32..1024", k.Name, k.Threads)
	}
	return k, nil
}

func (k AllReduceResidualRMSNorm) params() []string {
	ct := k.DType.cType()
	return []string{
		"int rows",
		"const " + ct + "* __restrict__ sym_in",
		"const " + ct + "* __restrict__ residual",
		"const " + ct + "* __restrict__ weight",
		ct + "* __restrict__ residual_out",
		ct + "* __restrict__ norm_out",
	}
}

// Grid returns the launch grid and block for rows token rows.
func (k AllReduceResidualRMSNorm) Grid(rows int) (grid, block [3]int) {
	threads := k.Threads
	if threads == 0 {
		threads = 256
	}
	return [3]int{rows, 1, 1}, [3]int{threads, 1, 1}
}

// CUDA returns the kernel definition without [Prelude].
func (k AllReduceResidualRMSNorm) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	ct := k.DType.cType()
	per := (k.Cols + k.Threads - 1) / k.Threads
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank NVLink multimem.ld_reduce + residual + RMSNorm over %d columns.\n",
		k.Name, k.Ranks, k.Cols)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int COLS = %d, THREADS = %d, PER = %d, RANKS = %d;\n",
		k.Cols, k.Threads, per, k.Ranks)
	fmt.Fprintf(&b, "    constexpr float EPS = %s;\n", floatLiteral(k.Eps))
	b.WriteString(`    (void)RANKS;
    const int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    __shared__ float eg_red[32];

    const size_t rank_elems = (size_t)rows * COLS;
    const ` + ct + `* local_res = eg_nvlink_local_ptr(residual, rank_elems);
    const ` + ct + `* local_w = eg_nvlink_local_ptr(weight, (size_t)COLS);
    ` + ct + `* local_res_out = eg_nvlink_local_ptr(residual_out, rank_elems);
    ` + ct + `* local_norm_out = eg_nvlink_local_ptr(norm_out, rank_elems);

    float h_reg[PER];
    float sq = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int col = tid + j * THREADS;
        if (col < COLS) {
            const size_t idx = (size_t)row * COLS + col;
`)
	if k.DType == F32 {
		b.WriteString(`            const float ar = eg_multimem_ld_reduce_add_f32(sym_in, idx);
            const float res = local_res[idx];
            const float h = ar + res;
            local_res_out[idx] = h;
            h_reg[j] = h;
            sq += h * h;
`)
	} else {
		b.WriteString(`            const float ar = eg_multimem_ld_reduce_add_bf16(sym_in, idx);
            const float res = eg_bf16_to_f32(local_res[idx]);
            const float h = ar + res;
            local_res_out[idx] = eg_f32_to_bf16(h);
            h_reg[j] = h;
            sq += h * h;
`)
	}
	b.WriteString(`        } else {
            h_reg[j] = 0.0f;
        }
    }
    sq = eg_warp_sum(sq);
    if constexpr (THREADS > 32) {
        sq = eg_block_combine<0>(sq, eg_red);
    }
    const float inv_rms = rsqrtf(sq / (float)COLS + EPS);

#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int col = tid + j * THREADS;
        if (col < COLS) {
            const size_t idx = (size_t)row * COLS + col;
`)
	if k.DType == F32 {
		b.WriteString(`            const float w = local_w[col];
            local_norm_out[idx] = h_reg[j] * inv_rms * w;
`)
	} else {
		b.WriteString(`            const float w = eg_bf16_to_f32(local_w[col]);
            local_norm_out[idx] = eg_f32_to_bf16(h_reg[j] * inv_rms * w);
`)
	}
	b.WriteString(`        }
    }
}
`)
	return b.String(), nil
}

// Launcher returns the host launcher function for k.
func (k AllReduceResidualRMSNorm) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    if (rows > 0) EG_LAUNCH(%s, dim3(rows), dim3(%d), 0, stream, %s);\n}\n",
		k.Name, k.Threads, strings.Join(paramNames(params), ", "))
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude],
// [CollectivePrelude], the kernel, and its launcher.
func (k AllReduceResidualRMSNorm) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + CollectivePrelude + "\n" + src + "\n" + l, nil
}

// Reference computes the float64 reference for AllReduceResidualRMSNorm across
// k.Ranks GPUs. symIn has length Ranks * rows * Cols; residual has length
// rows * Cols (or Ranks * rows * Cols); weight has length Cols.
func (k AllReduceResidualRMSNorm) Reference(rows int, symIn, residual, weight []float64) (residualOut, normOut []float64, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, err
	}
	elemsPerRank := rows * k.Cols
	if len(symIn) != k.Ranks*elemsPerRank {
		return nil, nil, fmt.Errorf("allreduce residual rmsnorm %s reference: symIn len %d != %d", k.Name, len(symIn), k.Ranks*elemsPerRank)
	}
	if len(residual) != elemsPerRank && len(residual) != k.Ranks*elemsPerRank {
		return nil, nil, fmt.Errorf("allreduce residual rmsnorm %s reference: bad residual len %d", k.Name, len(residual))
	}
	if len(weight) != k.Cols {
		return nil, nil, fmt.Errorf("allreduce residual rmsnorm %s reference: weight len %d != %d", k.Name, len(weight), k.Cols)
	}

	totalOut := len(residual)
	residualOut = make([]float64, totalOut)
	normOut = make([]float64, totalOut)
	outRanks := totalOut / elemsPerRank

	for r := 0; r < outRanks; r++ {
		for row := 0; row < rows; row++ {
			hRow := make([]float64, k.Cols)
			var sqSum float64
			for col := 0; col < k.Cols; col++ {
				idx := row*k.Cols + col
				var ar float64
				for g := 0; g < k.Ranks; g++ {
					v := float32(symIn[g*elemsPerRank+idx])
					ar += float64(k.DType.Round(v))
				}
				res := float64(k.DType.Round(float32(residual[r*elemsPerRank+idx])))
				h := ar + res
				hRow[col] = h
				residualOut[r*elemsPerRank+idx] = float64(k.DType.Round(float32(h)))
				sqSum += h * h
			}
			meanSq := float32(sqSum)/float32(k.Cols) + k.Eps
			invRMS := float32(1.0) / float32(math.Sqrt(float64(meanSq)))
			for col := 0; col < k.Cols; col++ {
				idx := row*k.Cols + col
				w := k.DType.Round(float32(weight[col]))
				y := float32(hRow[col]) * invRMS * w
				normOut[r*elemsPerRank+idx] = float64(k.DType.Round(y))
			}
		}
	}
	return residualOut, normOut, nil
}

// NVLinkMoECombine fuses cross-rank NVLink expert output gathering and
// router-weighted Top-K combination into a single kernel.
type NVLinkMoECombine struct {
	Name    string
	Cols    int
	TopK    int
	Ranks   int
	DType   DType
	Threads int
}

func (k NVLinkMoECombine) norm() (NVLinkMoECombine, error) {
	if k.Name == "" {
		k.Name = "nvlink_moe_combine"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("nvlink moe combine: bad name %q", k.Name)
	}
	if k.Cols < 1 {
		return k, fmt.Errorf("nvlink moe combine %s: cols %d must be >= 1", k.Name, k.Cols)
	}
	if k.TopK < 1 || k.TopK > 32 {
		return k, fmt.Errorf("nvlink moe combine %s: topK %d must be in 1..32", k.Name, k.TopK)
	}
	if k.Ranks < 1 {
		return k, fmt.Errorf("nvlink moe combine %s: ranks %d must be >= 1", k.Name, k.Ranks)
	}
	if k.DType == 0 {
		k.DType = F32
	}
	if k.DType != F32 && k.DType != BF16 {
		return k, fmt.Errorf("nvlink moe combine %s: dtype %v must be F32 or BF16", k.Name, k.DType)
	}
	if k.Threads == 0 {
		k.Threads = 256
	}
	if k.Threads < 32 || k.Threads > 1024 || k.Threads%32 != 0 {
		return k, fmt.Errorf("nvlink moe combine %s: threads %d must be a multiple of 32 in 32..1024", k.Name, k.Threads)
	}
	return k, nil
}

func (k NVLinkMoECombine) params() []string {
	ct := k.DType.cType()
	return []string{
		"int tokens",
		"int experts_per_rank",
		"const " + ct + "* __restrict__ sym_expert_out",
		"const int32_t* __restrict__ expert_ids",
		"const float* __restrict__ router_weights",
		ct + "* __restrict__ out",
	}
}

// Grid returns the launch grid and block for tokens token rows.
func (k NVLinkMoECombine) Grid(tokens int) (grid, block [3]int) {
	threads := k.Threads
	if threads == 0 {
		threads = 256
	}
	return [3]int{tokens, 1, 1}, [3]int{threads, 1, 1}
}

// CUDA returns the kernel definition without [Prelude].
func (k NVLinkMoECombine) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	ct := k.DType.cType()
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank NVLink Top-%d MoE combine over %d columns.\n",
		k.Name, k.Ranks, k.TopK, k.Cols)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int COLS = %d, TOPK = %d, THREADS = %d, RANKS = %d;\n",
		k.Cols, k.TopK, k.Threads, k.Ranks)
	b.WriteString(`    (void)RANKS;
    const int m = blockIdx.x, tid = threadIdx.x;
    if (m >= tokens) return;

    const int my_rank = eg_nvlink_rank();
    const size_t bytes_per_rank = (size_t)experts_per_rank * (size_t)tokens * (size_t)COLS * sizeof(` + ct + `);
    const int32_t* local_ids = eg_nvlink_local_ptr(expert_ids, (size_t)tokens * TOPK);
    const float* local_weights = eg_nvlink_local_ptr(router_weights, (size_t)tokens * TOPK);
    ` + ct + `* local_out = eg_nvlink_local_ptr(out, (size_t)tokens * COLS);

    for (int c = tid; c < COLS; c += THREADS) {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const int expert_id = local_ids[(size_t)m * TOPK + k];
            if (expert_id < 0) continue;
            const int expert_rank = expert_id / experts_per_rank;
            const int local_expert = expert_id % experts_per_rank;
            const float w = local_weights[(size_t)m * TOPK + k];
            const ` + ct + `* peer_slice = eg_nvlink_peer_ptr(sym_expert_out, expert_rank, my_rank, bytes_per_rank);
            const size_t elem_off = ((size_t)local_expert * (size_t)tokens + (size_t)m) * (size_t)COLS + (size_t)c;
`)
	if k.DType == F32 {
		b.WriteString(`            const float v = peer_slice[elem_off];
            acc += v * w;
`)
	} else {
		b.WriteString(`            const float v = eg_bf16_to_f32(peer_slice[elem_off]);
            acc += v * w;
`)
	}
	b.WriteString(`        }
`)
	if k.DType == F32 {
		b.WriteString(`        local_out[(size_t)m * COLS + c] = acc;
`)
	} else {
		b.WriteString(`        local_out[(size_t)m * COLS + c] = eg_f32_to_bf16(acc);
`)
	}
	b.WriteString(`    }
}
`)
	return b.String(), nil
}

// Launcher returns the host launcher function for k.
func (k NVLinkMoECombine) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    if (tokens > 0) EG_LAUNCH(%s, dim3(tokens), dim3(%d), 0, stream, %s);\n}\n",
		k.Name, k.Threads, strings.Join(paramNames(params), ", "))
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude],
// [CollectivePrelude], the kernel, and its launcher.
func (k NVLinkMoECombine) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + CollectivePrelude + "\n" + src + "\n" + l, nil
}

// Reference computes the float64 reference for NVLinkMoECombine.
// symExpertOut has length Ranks * expertsPerRank * tokens * Cols.
// expertIDs and routerWeights have length tokens * TopK (or Ranks * tokens * TopK).
func (k NVLinkMoECombine) Reference(tokens, expertsPerRank int, symExpertOut []float64, expertIDs []int32, routerWeights []float64) ([]float64, error) {
	k, err := k.norm()
	if err != nil {
		return nil, err
	}
	sliceElems := expertsPerRank * tokens * k.Cols
	if len(symExpertOut) != k.Ranks*sliceElems {
		return nil, fmt.Errorf("nvlink moe combine %s reference: symExpertOut len %d != %d",
			k.Name, len(symExpertOut), k.Ranks*sliceElems)
	}
	routesPerRank := tokens * k.TopK
	if len(expertIDs) != routesPerRank && len(expertIDs) != k.Ranks*routesPerRank {
		return nil, fmt.Errorf("nvlink moe combine %s reference: bad expertIDs len %d", k.Name, len(expertIDs))
	}
	if len(routerWeights) != len(expertIDs) {
		return nil, fmt.Errorf("nvlink moe combine %s reference: routerWeights len %d != expertIDs len %d",
			k.Name, len(routerWeights), len(expertIDs))
	}

	outRanks := len(expertIDs) / routesPerRank
	out := make([]float64, outRanks*tokens*k.Cols)
	for r := 0; r < outRanks; r++ {
		for m := 0; m < tokens; m++ {
			for c := 0; c < k.Cols; c++ {
				var acc float64
				for rt := 0; rt < k.TopK; rt++ {
					eid := int(expertIDs[r*routesPerRank+m*k.TopK+rt])
					if eid < 0 {
						continue
					}
					ownerRank := eid / expertsPerRank
					localExpert := eid % expertsPerRank
					w := float64(float32(routerWeights[r*routesPerRank+m*k.TopK+rt]))
					off := ownerRank*sliceElems + (localExpert*tokens+m)*k.Cols + c
					v := float64(k.DType.Round(float32(symExpertOut[off])))
					acc += v * w
				}
				out[(r*tokens+m)*k.Cols+c] = float64(k.DType.Round(float32(acc)))
			}
		}
	}
	return out, nil
}
