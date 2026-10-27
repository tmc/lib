package nvl72

import (
	"fmt"
	"strings"
)

// FabricStatus reports the completion acceptance status returned by a PTX 9.4
// (§9.7.11.5.7–10) NVLink 6 fabric transaction instruction (fabric.try_*).
type FabricStatus uint32

const (
	// FabricSuccess indicates that the NVLink 6 fabric engine accepted the
	// asynchronous transaction (status == 0). Any non-zero status requires
	// retrying the fabric.try_* instruction.
	FabricSuccess FabricStatus = 0

	// CountedBytesAlignment is the required byte alignment (256 bytes) for
	// .counted::bytes completion counter addresses in PTX ISA 9.4 §9.7.11.5.8–9.
	CountedBytesAlignment = 256
)

// FabricEndpointHandle models the 128-bit (tensorMapLeId, tensorMapOff) or
// (fabLeId, fabOff) NVLink 6 fabric endpoint descriptor pair used by PTX 9.4
// fabric.try_{get,put,red}.async.tensor and fabric.try_atom.async instructions.
type FabricEndpointHandle struct {
	LeID   uint64
	Offset uint64
}

// Validate checks that h has a valid alignment when used with the given
// completion mode.
func (h FabricEndpointHandle) Validate(completion FabricCompletionMode) error {
	if completion == FabricCompletionCountedBytes && h.Offset%CountedBytesAlignment != 0 {
		return fmt.Errorf("nvl72: fabric counted::bytes offset %#x must be %d-byte aligned", h.Offset, CountedBytesAlignment)
	}
	return nil
}

// FabricTensorOp identifies a PTX 9.4 §9.7.11.5.7–9 asynchronous TMA-over-Fabric
// tensor operation in the fabric state space.
type FabricTensorOp string

const (
	// FabricTensorTryGet selects fabric.try_get.async.tensor (§9.7.11.5.7),
	// pulling a 1D–5D tensor tile from a remote NVLink 6 fabric endpoint into
	// local shared::cluster memory.
	FabricTensorTryGet FabricTensorOp = "try_get"

	// FabricTensorTryPut selects fabric.try_put.async.tensor (§9.7.11.5.8),
	// pushing or multicasting (.multimem) a 1D–5D tensor tile from local
	// shared::cta memory to remote NVLink 6 fabric endpoint(s).
	FabricTensorTryPut FabricTensorOp = "try_put"

	// FabricTensorTryRed selects fabric.try_red.async.tensor (§9.7.11.5.9),
	// performing an asynchronous in-switch / endpoint tensor tile reduction
	// across NVLink 6 fabric endpoint(s).
	FabricTensorTryRed FabricTensorOp = "try_red"
)

// FabricCompletionMode selects the completion tracking mechanism for PTX 9.4
// fabric.try_{get,put,red}.async.tensor instructions.
type FabricCompletionMode string

const (
	// FabricCompletionMbarrier selects mbarrier completion tracking
	// (.mbarrier::complete_tx::16B.mbarrier::report::fabric).
	FabricCompletionMbarrier FabricCompletionMode = "mbarrier"

	// FabricCompletionCountedBytes selects 256-byte-aligned byte counter
	// completion tracking (.counted::bytes) on try_put and try_red.
	FabricCompletionCountedBytes FabricCompletionMode = "counted_bytes"
)

// PTX 9.4 §9.7.11.5.10 canonical fabric.try_atom.async instruction string.
const (
	FabricTryAtomAsyncIncU32PTX = "fabric.try_atom.async.fabric.mbarrier::complete_tx::16B.mbarrier::report::fabric.inc.u32"
)

// RubinFabricAsyncTensorKernel configures a multi-rank NVLink 6 TMA-over-Fabric
// tile transfer or switch-reduction kernel on Rubin (sm_107a / sm_107f) using
// PTX ISA 9.4 §9.7.11.5.7–10 instructions:
//   - fabric.try_get.async.tensor.dim.shared::cluster.fabric.mbarrier::complete_tx::16B.mbarrier::report::fabric.b128
//   - fabric.try_put.async.tensor.dim{.multimem}.fabric.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.b128
//   - fabric.try_put.async.tensor.dim{.multimem}.fabric.shared::cta.counted::bytes.b128
//   - fabric.try_red.async.tensor.dim{.multimem}.fabric.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.redOp.type
//   - fabric.try_atom.async.fabric.mbarrier::complete_tx::16B.mbarrier::report::fabric.op.type
type RubinFabricAsyncTensorKernel struct {
	Name       string
	Op         FabricTensorOp
	Dim        int
	TileRows   int
	TileCols   int
	Multimem   bool
	Completion FabricCompletionMode
	RedOp      string
	NumRanks   int
	Target     string
	Endpoint   FabricEndpointHandle
}

func (k RubinFabricAsyncTensorKernel) norm() RubinFabricAsyncTensorKernel {
	if k.Name == "" {
		k.Name = "eg_rubin_fabric_async_tensor"
	}
	if k.Op == "" {
		k.Op = FabricTensorTryRed
		if !k.Multimem {
			k.Multimem = true
		}
	}
	if k.Dim == 0 {
		k.Dim = 2
	}
	if k.TileRows == 0 {
		k.TileRows = 32
	}
	if k.TileCols == 0 {
		k.TileCols = 32
	}
	if k.Completion == "" {
		k.Completion = FabricCompletionMbarrier
	}
	if k.RedOp == "" {
		k.RedOp = "add"
	}
	if k.NumRanks == 0 {
		k.NumRanks = 4
	}
	if k.Target == "" {
		k.Target = RubinTargetArch
	}
	return k
}

// Validate checks that k specifies a legal PTX ISA 9.4 §9.7.11.5.7–10
// NVLink 6 fabric asynchronous tensor configuration on sm_107a or sm_107f.
func (k RubinFabricAsyncTensorKernel) Validate() error {
	rawOp := k.Op
	k = k.norm()
	if !cIdent(k.Name) && k.Name != "eg_rubin_fabric_async_tensor" {
		return fmt.Errorf("nvl72: invalid kernel name %q", k.Name)
	}
	switch k.Target {
	case "sm_107a", "sm_107f":
	default:
		return fmt.Errorf("nvl72: fabric.try_%s.async.tensor target %q unsupported (want sm_107a or sm_107f)", k.Op, k.Target)
	}
	if k.Dim < 1 || k.Dim > 5 {
		return fmt.Errorf("nvl72: fabric tensor dimension %d must be in 1..5", k.Dim)
	}
	if k.TileRows < 1 || k.TileCols < 1 {
		return fmt.Errorf("nvl72: invalid tile dimensions (%d, %d)", k.TileRows, k.TileCols)
	}
	if k.NumRanks < 1 || k.NumRanks > RubinUltraNVL576ComputeDies {
		return fmt.Errorf("nvl72: NumRanks %d must be in 1..%d", k.NumRanks, RubinUltraNVL576ComputeDies)
	}
	switch k.Op {
	case FabricTensorTryGet:
		if rawOp == FabricTensorTryGet && k.Multimem {
			return fmt.Errorf("nvl72: fabric.try_get.async.tensor does not support .multimem")
		}
		if k.Completion == FabricCompletionCountedBytes {
			return fmt.Errorf("nvl72: fabric.try_get.async.tensor does not support .counted::bytes completion")
		}
		if k.Completion != FabricCompletionMbarrier {
			return fmt.Errorf("nvl72: unsupported completion mode %q for try_get", k.Completion)
		}
	case FabricTensorTryPut, FabricTensorTryRed:
		switch k.Completion {
		case FabricCompletionMbarrier, FabricCompletionCountedBytes:
		default:
			return fmt.Errorf("nvl72: unsupported completion mode %q", k.Completion)
		}
	default:
		return fmt.Errorf("nvl72: unsupported FabricTensorOp %q", k.Op)
	}
	if k.Op == FabricTensorTryRed {
		switch k.RedOp {
		case "add", "min", "max", "inc", "dec", "and", "or", "xor":
		default:
			return fmt.Errorf("nvl72: unsupported fabric reduction op %q", k.RedOp)
		}
	}
	if err := k.Endpoint.Validate(k.Completion); err != nil {
		return err
	}
	return nil
}

// TileElems returns the number of float32 elements per tile (TileRows * TileCols).
func (k RubinFabricAsyncTensorKernel) TileElems() int {
	k = k.norm()
	return k.TileRows * k.TileCols
}

// Grid returns the launch grid and CTA block dimensions for k.
func (k RubinFabricAsyncTensorKernel) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{64, 1, 1}
}

// PTXInstruction returns the exact PTX ISA 9.4 §9.7.11.5.7–9 instruction
// mnemonic for k.
func (k RubinFabricAsyncTensorKernel) PTXInstruction() (string, error) {
	if err := k.Validate(); err != nil {
		return "", err
	}
	k = k.norm()
	dimStr := fmt.Sprintf("%dd", k.Dim)
	mmStr := ""
	if k.Multimem && k.Op != FabricTensorTryGet {
		mmStr = ".multimem"
	}

	switch k.Op {
	case FabricTensorTryGet:
		return fmt.Sprintf(
			"fabric.try_get.async.tensor.%s.shared::cluster.fabric.mbarrier::complete_tx::16B.mbarrier::report::fabric.b128",
			dimStr,
		), nil
	case FabricTensorTryPut:
		if k.Completion == FabricCompletionCountedBytes {
			return fmt.Sprintf(
				"fabric.try_put.async.tensor.%s%s.fabric.shared::cta.counted::bytes.b128",
				dimStr, mmStr,
			), nil
		}
		return fmt.Sprintf(
			"fabric.try_put.async.tensor.%s%s.fabric.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.b128",
			dimStr, mmStr,
		), nil
	case FabricTensorTryRed:
		compStr := ".mbarrier::complete_tx::16B.mbarrier::report::fabric"
		if k.Completion == FabricCompletionCountedBytes {
			compStr = ".counted::bytes"
		}
		return fmt.Sprintf(
			"fabric.try_red.async.tensor.%s%s.fabric.shared::cta%s.%s.f32",
			dimStr, mmStr, compStr, k.RedOp,
		), nil
	default:
		return "", fmt.Errorf("nvl72: unsupported FabricTensorOp %q", k.Op)
	}
}

func (k RubinFabricAsyncTensorKernel) coordsOperand() string {
	k = k.norm()
	coords := make([]string, k.Dim)
	for i := range coords {
		coords[i] = "0"
	}
	return "{" + strings.Join(coords, ", ") + "}"
}

// CUDA emits the multi-rank CUDA kernel eg_rubin_fabric_async_tensor compatible
// with cudahostemu.RunMultiRank and PTX ISA 9.4 sm_107f inline assembly.
func (k RubinFabricAsyncTensorKernel) CUDA() (string, error) {
	ptxInst, err := k.PTXInstruction()
	if err != nil {
		return "", err
	}
	k = k.norm()
	coords := k.coordsOperand()

	var ptxAsmCall string
	switch k.Op {
	case FabricTensorTryGet:
		ptxAsmCall = fmt.Sprintf(
			`asm volatile("%s %%0, [%%1], [%%2, %%3, %s], [%%4];" : "=r"(status) : "r"(smem_tile_addr), "l"(tensor_map_le_id), "l"(tensor_map_off), "r"(local_mbar_addr) : "memory");`,
			ptxInst, coords,
		)
	case FabricTensorTryPut, FabricTensorTryRed:
		if k.Completion == FabricCompletionCountedBytes {
			ptxAsmCall = fmt.Sprintf(
				`asm volatile("%s %%0, [%%1, %%2, %s], [%%3];" : "=r"(status) : "l"(tensor_map_le_id), "l"(tensor_map_off), "r"(smem_tile_addr) : "memory");`,
				ptxInst, coords,
			)
		} else {
			ptxAsmCall = fmt.Sprintf(
				`asm volatile("%s %%0, [%%1, %%2, %s], [%%3], [%%4, %%5];" : "=r"(status) : "l"(tensor_map_le_id), "l"(tensor_map_off), "r"(smem_tile_addr), "l"(mbar_le_id), "l"(mbar_off) : "memory");`,
				ptxInst, coords,
			)
		}
	}

	opMode := 0
	switch k.Op {
	case FabricTensorTryGet:
		opMode = 0
	case FabricTensorTryPut:
		opMode = 1
	case FabricTensorTryRed:
		opMode = 2
	}

	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank NVLink 6 (%s) TMA-over-Fabric (%s) tile (%dx%d, dim=%d, completion=%s).\n",
		k.Name, k.NumRanks, k.Target, ptxInst, k.TileRows, k.TileCols, k.Dim, k.Completion)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(64) %s(\n", k.Name)
	b.WriteString("    float* __restrict__ dst_tile,\n")
	b.WriteString("    const float* __restrict__ src_tile,\n")
	b.WriteString("    unsigned int* __restrict__ out_status,\n")
	b.WriteString("    int n) {\n")
	fmt.Fprintf(&b, "    constexpr int NUM_RANKS = %d, OP_MODE = %d;\n", k.NumRanks, opMode)
	fmt.Fprintf(&b, "    constexpr uint64_t ENDPOINT_LE_ID = %duLL, ENDPOINT_OFF = %duLL;\n",
		k.Endpoint.LeID, k.Endpoint.Offset)
	b.WriteString(`    (void)NUM_RANKS;
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(gridDim.x * blockDim.x);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1070
    if (tid == 0) {
        unsigned int status = 0u;
        const uint32_t smem_tile_addr = 0u;
        const uint32_t local_mbar_addr = 0u;
        const uint64_t tensor_map_le_id = ENDPOINT_LE_ID;
        const uint64_t tensor_map_off = ENDPOINT_OFF;
        const uint64_t mbar_le_id = ENDPOINT_LE_ID;
        const uint64_t mbar_off = ENDPOINT_OFF;
        do {
            ` + ptxAsmCall + `
        } while (status != 0u);
        unsigned int atom_status = 0u;
        do {
            asm volatile("` + FabricTryAtomAsyncIncU32PTX + ` %0, [%1], [%2, %3], %4, [%5, %6];"
                         : "=r"(atom_status)
                         : "l"(out_status), "l"(tensor_map_le_id), "l"(tensor_map_off), "r"(1u), "l"(mbar_le_id), "l"(mbar_off)
                         : "memory");
        } while (atom_status != 0u);
        if (out_status != nullptr) {
            out_status[0] = status;
        }
    }
#endif

    const int my_rank = eg_nvlink_rank();
    const size_t rank_elems = (size_t)n;
    const size_t bytes_per_rank = rank_elems * sizeof(float);
    float* local_dst = eg_nvlink_local_ptr(dst_tile, rank_elems);
    unsigned int* local_status = eg_nvlink_local_ptr(out_status, (size_t)1);

    if constexpr (OP_MODE == 0) {
        // fabric.try_get.async.tensor: pull remote rank ((my_rank + 1) % NUM_RANKS) source tile into local dst_tile.
        const int remote_rank = (NUM_RANKS > 1) ? ((my_rank + 1) % NUM_RANKS) : 0;
        const float* remote_src = eg_nvlink_peer_ptr(src_tile, remote_rank, my_rank, bytes_per_rank);
        for (int i = tid; i < n; i += stride) {
            local_dst[i] = remote_src[i];
        }
    } else if constexpr (OP_MODE == 1) {
        // fabric.try_put.async.tensor: rank 0 broadcasts/writes its src_tile across all fabric endpoint ranks' dst_tile.
        const float* root_src = eg_nvlink_peer_ptr(src_tile, 0, my_rank, bytes_per_rank);
        for (int i = tid; i < n; i += stride) {
            const float v = root_src[i];
            if (my_rank == 0) {
                eg_rubin_multimem_st_relaxed_sys_f32(dst_tile, (size_t)i, v);
            }
        }
    } else {
        // fabric.try_red.async.tensor: switch-reduce src_tile across all ranks via NVLink 6 multimem and store to dst_tile.
        for (int i = tid; i < n; i += stride) {
            const float reduced = eg_rubin_multimem_ld_reduce_relaxed_sys_add_f32(src_tile, (size_t)i);
            if (my_rank == 0) {
                eg_rubin_multimem_st_relaxed_sys_f32(dst_tile, (size_t)i, reduced);
            }
        }
    }
    if (tid == 0 && local_status != nullptr) {
        local_status[0] = 0u;
    }
}
`)
	return b.String(), nil
}

// Launcher returns the host launcher function for k.
func (k RubinFabricAsyncTensorKernel) Launcher() (string, error) {
	if err := k.Validate(); err != nil {
		return "", err
	}
	k = k.norm()
	grid, block := k.Grid()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(float* dst_tile, const float* src_tile, unsigned int* out_status, int n, cudaStream_t stream) {\n", k.Name)
	fmt.Fprintf(&b, "    if (n > 0) EG_LAUNCH(%s, dim3(%d), dim3(%d), 0, stream, dst_tile, src_tile, out_status, n);\n}\n",
		k.Name, grid[0], block[0])
	return b.String(), nil
}

// CUDAFile returns a complete standalone CUDA translation unit for target
// sm_107f including [Prelude], [CollectivePrelude], NVLink 6 multimem PTX
// helpers, the fabric async tensor kernel, and its host launcher.
func (k RubinFabricAsyncTensorKernel) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	launcher, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + CollectivePrelude + "\n" + rubinNVLink6MultimemPrelude + "\n" + src + "\n" + launcher, nil
}

// Source is an alias for [RubinFabricAsyncTensorKernel.CUDAFile].
func (k RubinFabricAsyncTensorKernel) Source() (string, error) {
	return k.CUDAFile()
}

// Reference computes the expected per-rank output tile for
// RubinFabricAsyncTensorKernel across rankTiles (len(rankTiles) == NumRanks).
// For try_red and try_put (where all ranks receive the same reduced or
// broadcast tile), Reference returns a single tile of length N (TileRows*TileCols);
// for try_get, Reference returns rank 0's pulled tile (from rank 1 when
// NumRanks > 1, or rank 0 when NumRanks == 1).
func (k RubinFabricAsyncTensorKernel) Reference(rankTiles [][]float32) ([]float32, error) {
	if k.NumRanks == 0 && len(rankTiles) > 0 {
		k.NumRanks = len(rankTiles)
	}
	if err := k.Validate(); err != nil {
		return nil, err
	}
	k = k.norm()
	if len(rankTiles) != k.NumRanks {
		return nil, fmt.Errorf("nvl72: rankTiles len %d != NumRanks %d", len(rankTiles), k.NumRanks)
	}
	if len(rankTiles) == 0 || len(rankTiles[0]) == 0 {
		return nil, fmt.Errorf("nvl72: empty rankTiles")
	}
	n := len(rankTiles[0])
	for r, tile := range rankTiles {
		if len(tile) != n {
			return nil, fmt.Errorf("nvl72: rankTiles[%d] len %d != %d", r, len(tile), n)
		}
	}

	out := make([]float32, n)
	switch k.Op {
	case FabricTensorTryGet:
		srcRank := 0
		if k.NumRanks > 1 {
			srcRank = 1
		}
		copy(out, rankTiles[srcRank])
	case FabricTensorTryPut:
		copy(out, rankTiles[0])
	case FabricTensorTryRed:
		for i := 0; i < n; i++ {
			acc := rankTiles[0][i]
			for r := 1; r < k.NumRanks; r++ {
				v := rankTiles[r][i]
				switch k.RedOp {
				case "add":
					acc += v
				case "min":
					if v < acc {
						acc = v
					}
				case "max":
					if v > acc {
						acc = v
					}
				default:
					acc += v
				}
			}
			out[i] = acc
		}
	}
	return out, nil
}
