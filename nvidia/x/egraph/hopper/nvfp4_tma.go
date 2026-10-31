package hopper

import (
	"fmt"
	"math"
	"strings"
)

// NVFP4TMAGEMM emits the retained routed NVFP4 schedule with TMA activation
// loads, a persistent CTA grid and two consumer warpgroups. It is a host port
// of the prefill v27 schedule; CUDA compilation, numerical parity, graph
// replay and performance must be qualified separately on sm_90a.
//
// The caller first gathers BF16 activations into expert-sorted, BM-padded
// storage x[rows,K]. A device tile list contains eids[rows/BM], one expert
// per tile, bvalid[rows/BM] in [1,BM], and npost[1] == rows. Empty experts
// have no tile-list entry. Only each tile's first bvalid rows are semantic
// output rows; the other output rows are unspecified.
//
// Weights use RepackNVFP4Dequant2Word and CompactNVFP4Scales, concatenated
// per expert: words[E,N/128,K/64,256,4] and scales[E,N/128,K/64,64,8]. The
// launcher's global scales are the original expert scales multiplied by 256.
// GLU selects the repacker's GLU row order and the epilogue
// silu(gate)*up, using CUDA's __expf like the retained hand kernel. Otherwise
// the epilogue is identity. No arbitrary epilogue, NSUB>1 or row gather is
// currently supported.
//
// The zero value is invalid. Name, N and K are required; N must be a positive
// multiple of 128 and K a positive multiple of 128. The latter restriction
// preserves the retained two-step k-loop's even K/64 precondition.
// This lowering is not yet connected to e-graph extraction or autotuning.
type NVFP4TMAGEMM struct {
	Name string
	N, K int
	// BM is 64, 128 or 256; zero selects 128.
	BM int
	// Stages defaults to 16, 10 or 6 for BM64, BM128 or BM256.
	// Stages must be at least three: the retained consumer loop fetches
	// the next stage before releasing the previous one. The emitted dynamic
	// shared-memory allocation must fit 227 KiB.
	Stages int
	GLU    bool
	// FastScales requires every block scale to be a positive normal,
	// finite E4M3 value. It enables the retained variable-N schedule.
	// False keeps the general scale conversion and full-BM schedule.
	FastScales bool
	// GroupM is the persistent scheduler's M-band size; zero selects one.
	GroupM int
	// Producers is the number of producer warp leaders, from one to four.
	// Zero selects one. They partition k-steps by the continuous ring index.
	Producers int
}

func (g NVFP4TMAGEMM) norm() (NVFP4TMAGEMM, error) {
	if !cIdent(g.Name) {
		return g, fmt.Errorf("nvfp4 tma gemm: bad name %q", g.Name)
	}
	if g.N <= 0 || g.N%128 != 0 || g.N > math.MaxInt32 {
		return g, fmt.Errorf("nvfp4 tma gemm %s: N must be a positive int32 multiple of 128", g.Name)
	}
	if g.K <= 0 || g.K%128 != 0 || g.K > math.MaxInt32 {
		return g, fmt.Errorf("nvfp4 tma gemm %s: K must be a positive int32 multiple of 128", g.Name)
	}
	if g.N > math.MaxInt/g.K {
		return g, fmt.Errorf("nvfp4 tma gemm %s: weight shape overflows int", g.Name)
	}
	if g.BM == 0 {
		g.BM = 128
	}
	if g.BM != 64 && g.BM != 128 && g.BM != 256 {
		return g, fmt.Errorf("nvfp4 tma gemm %s: BM must be 64, 128 or 256", g.Name)
	}
	if g.Stages == 0 {
		switch g.BM {
		case 64:
			g.Stages = 16
		case 128:
			g.Stages = 10
		case 256:
			g.Stages = 6
		}
	}
	if g.Stages < 3 || g.Stages > 16 || nvfp4TMASharedBytes(g.BM, g.Stages) > 227*1024 {
		return g, fmt.Errorf("nvfp4 tma gemm %s: stages must be 3..16 and fit 227 KiB shared memory", g.Name)
	}
	if g.GroupM == 0 {
		g.GroupM = 1
	}
	if g.GroupM < 1 || g.GroupM > math.MaxInt32/(g.N/128) {
		return g, fmt.Errorf("nvfp4 tma gemm %s: GroupM must be positive and GroupM*N/128 must fit int32", g.Name)
	}
	if g.Producers == 0 {
		g.Producers = 1
	}
	if g.Producers < 1 || g.Producers > 4 {
		return g, fmt.Errorf("nvfp4 tma gemm %s: Producers must be 1..4", g.Name)
	}
	return g, nil
}

func nvfp4TMASharedBytes(bm, stages int) int {
	stage := (bm*64*2 + 256*16 + 64*8 + 1023) / 1024 * 1024
	return stages*stage + 2*stages*8 + 1024
}

// SharedBytes returns the dynamic shared memory required by the emitted
// kernel, including mbarriers and alignment slack, or zero for invalid g.
func (g NVFP4TMAGEMM) SharedBytes() int {
	g, err := g.norm()
	if err != nil {
		return 0
	}
	return nvfp4TMASharedBytes(g.BM, g.Stages)
}

// CUDA returns the standalone CUDA device implementation. Its templates are
// instantiated by Launcher. It requires CUDA headers and an sm_90a target.
// The namespace is derived from Name so separate emitted kernels may coexist.
func (g NVFP4TMAGEMM) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	return strings.ReplaceAll(nvfp4TMADevice, "namespace pm", "namespace "+g.Name+"_detail"), nil
}

// Launcher returns plain CUDA host functions. Name_make_map encodes the
// BF16 input descriptor through a caller-resolved cuTensorMapEncodeTiled
// function pointer. Name_configure must succeed on the active device before
// CUDA graph capture; Name_launch consumes the descriptor and device tile
// list. The launch has no Torch, driver-symbol lookup, host sync or input-value
// read. Configuration is explicit rather than cached across CUDA devices.
//
// The input allocation must be at least rows*K BF16 elements, aligned to 16
// bytes, with the supplied row stride in bytes. The tensor-map box is [64,16]
// for fast BM64/128, [64,32] for fast BM256, and [64,BM] otherwise. It uses
// rank two, 128-byte swizzle, no interleave and 256-byte L2 promotion.
//
// grid is an explicit positive persistent CTA count. The caller must validate
// all buffer extents, tile experts and valid-row counts, and ensure the active
// tile count times N/128 fits int32 and its continuous k-step count fits uint32.
// Integer limits can be checked conservatively from allocation capacities;
// the launcher does not read device values to check them. The allocation and
// descriptor must remain alive through graph replay.
func (g NVFP4TMAGEMM) Launcher() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	boxRows := g.BM
	if g.FastScales {
		boxRows = 16
		if g.BM == 256 {
			boxRows = 32
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, `
#include <cudaTypedefs.h>

inline CUresult %[1]s_make_map(CUtensorMap* map, void* x, uint64_t rows,
                              uint64_t row_stride_bytes, PFN_cuTensorMapEncodeTiled_v12000 encode) {
    if (!map || !x || !encode || rows == 0 || (reinterpret_cast<uintptr_t>(x) & 15) ||
        row_stride_bytes < %[2]dULL * 2 || row_stride_bytes %% 16 != 0)
        return CUDA_ERROR_INVALID_VALUE;
    cuuint64_t dims[2] = {%[2]d, rows};
    cuuint64_t strides[1] = {row_stride_bytes};
    cuuint32_t box[2] = {64, %[3]d};
    cuuint32_t elem[2] = {1, 1};
    return encode(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x, dims, strides, box, elem,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline cudaError_t %[1]s_configure() {
    auto kernel = %[1]s_detail::gemm_kernel<%[4]d, 1, %[5]d, %[6]t, %[7]t, 0>;
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, %[8]d);
}

inline cudaError_t %[1]s_launch(const CUtensorMap& map, const uint4* words,
                              const uint2* scales, const float* global_scales_256,
                              const int32_t* eids, const int32_t* bvalid,
                              const int32_t* npost, __nv_bfloat16* out,
                              int grid, cudaStream_t stream) {
    if (!words || !scales || !global_scales_256 || !eids || !bvalid || !npost || !out || grid <= 0)
        return cudaErrorInvalidValue;
    auto kernel = %[1]s_detail::gemm_kernel<%[4]d, 1, %[5]d, %[6]t, %[7]t, 0>;
    kernel<<<grid, 384, %[8]d, stream>>>(map, words, scales, global_scales_256, eids,
                                       bvalid, npost, out, %[2]d, %[9]d, %[10]d, %[11]d);
    return cudaGetLastError();
}
`, g.Name, g.K, boxRows, g.BM, g.Stages, g.GLU, g.FastScales, g.SharedBytes(), g.N/128, g.GroupM, g.Producers)
	return b.String(), nil
}

// Source returns CUDA followed by Launcher.
func (g NVFP4TMAGEMM) Source() (string, error) {
	cuda, err := g.CUDA()
	if err != nil {
		return "", err
	}
	launch, err := g.Launcher()
	if err != nil {
		return "", err
	}
	return cuda + launch, nil
}

// Reference computes the semantic rows of the padded tile list from raw
// GEMMWeights. Padding is returned as NaN because the CUDA implementation
// leaves some padded rows unwritten. It uses the package's FP32 reference
// SiLU, rather than emulating the device-specific approximation of __expf.
// It is an oracle for numerical checks, not a CUDA execution or timing model.
func (g NVFP4TMAGEMM) Reference(x []uint16, eids, bvalid []int32, weights []GEMMWeights) ([]float32, error) {
	g, err := g.norm()
	if err != nil {
		return nil, err
	}
	if len(eids) > math.MaxInt/g.BM || len(eids) != len(bvalid) || len(x)%g.K != 0 || len(x)/g.K != g.BM*len(eids) {
		return nil, fmt.Errorf("nvfp4 tma gemm %s reference: inconsistent padded tile storage", g.Name)
	}
	for e, w := range weights {
		if len(w.Packed) != g.N*g.K/2 || len(w.Scales) != g.N*g.K/16 {
			return nil, fmt.Errorf("nvfp4 tma gemm %s reference: expert %d weight extent mismatch", g.Name, e)
		}
		if g.FastScales {
			for i, scale := range w.Scales {
				if scale&0x80 != 0 || (scale>>3)&15 == 0 || scale&0x7f == 0x7f {
					return nil, fmt.Errorf("nvfp4 tma gemm %s reference: expert %d scale %d violates FastScales", g.Name, e, i)
				}
			}
		}
		if math.IsNaN(float64(w.GlobalScal)) || math.IsInf(float64(w.GlobalScal*256), 0) {
			return nil, fmt.Errorf("nvfp4 tma gemm %s reference: expert %d global scale times 256 is not finite", g.Name, e)
		}
	}
	nout := g.N
	epi := Identity
	if g.GLU {
		nout /= 2
		epi = MustLambda("(mul (silu $0) $1)")
	}
	if len(x)/g.K > math.MaxInt/nout {
		return nil, fmt.Errorf("nvfp4 tma gemm %s reference: output size overflows int", g.Name)
	}
	out := make([]float32, len(x)/g.K*nout)
	for i := range out {
		out[i] = float32(math.NaN())
	}
	ref := TileGEMM{Name: g.Name + "_reference", N: g.N, K: g.K, Weight: WeightNVFP4, Out: BF16, GLU: g.GLU, Epilogue: epi}
	for tile, e := range eids {
		valid := int(bvalid[tile])
		if e < 0 || int(e) >= len(weights) || valid < 1 || valid > g.BM {
			return nil, fmt.Errorf("nvfp4 tma gemm %s reference: invalid expert or row count at tile %d", g.Name, tile)
		}
		start := tile * g.BM
		y, err := ref.Reference(valid, x[start*g.K:(start+valid)*g.K], nil, nil, []GEMMWeights{weights[e]}, nil)
		if err != nil {
			return nil, err
		}
		copy(out[start*nout:], y)
	}
	return out, nil
}

// nvfp4TMATile maps one persistent work index to activation and weight tiles.
// It is the host model of the source's GroupM permutation, not a GPU scheduler.
func nvfp4TMATile(t, numMB, nt, groupM int) (mb, weightTile int) {
	inGroup := groupM * nt
	first := t / inGroup * groupM
	groupSize := min(numMB-first, groupM)
	return first + (t%inGroup)%groupSize, (t % inGroup) / groupSize
}

func nvfp4TMAVariableN(bm, valid int, fast bool) int {
	if !fast {
		return bm
	}
	step := 16
	if bm == 256 {
		step = 32
	}
	n := (valid + step - 1) / step * step
	if bm == 128 && n > 96 {
		return 128
	}
	if bm == 256 {
		if n <= 160 {
			return 160
		}
		if n <= 192 {
			return 192
		}
		return 256
	}
	return n
}
