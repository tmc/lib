package hopper

import "github.com/tmc/lib/nvidia/x/nvl72"

// CollectivePrelude provides device helpers for NVLink symmetric memory
// (multimem.ld_reduce and peer pointer addressing) across Hopper/Blackwell
// GPUs (__CUDA_ARCH__ >= 900) and cudahostemu multi-rank host emulation.
const CollectivePrelude = nvl72.CollectivePrelude

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

func (k AllReduceResidualRMSNorm) toNVL72() nvl72.AllReduceResidualRMSNorm {
	return nvl72.AllReduceResidualRMSNorm{
		Name:    k.Name,
		Cols:    k.Cols,
		Ranks:   k.Ranks,
		Eps:     k.Eps,
		DType:   nvl72.DType(k.DType),
		Threads: k.Threads,
	}
}

// Grid returns the launch grid and block for rows token rows.
func (k AllReduceResidualRMSNorm) Grid(rows int) (grid, block [3]int) {
	return k.toNVL72().Grid(rows)
}

// CUDA returns the kernel definition without [Prelude].
func (k AllReduceResidualRMSNorm) CUDA() (string, error) {
	return k.toNVL72().CUDA()
}

// Launcher returns the host launcher function for k.
func (k AllReduceResidualRMSNorm) Launcher() (string, error) {
	return k.toNVL72().Launcher()
}

// Source returns a complete translation unit with [Prelude],
// [CollectivePrelude], the kernel, and its launcher.
func (k AllReduceResidualRMSNorm) Source() (string, error) {
	return k.toNVL72().Source()
}

// Reference computes the float64 reference for AllReduceResidualRMSNorm across
// k.Ranks GPUs. symIn has length Ranks * rows * Cols; residual has length
// rows * Cols (or Ranks * rows * Cols); weight has length Cols.
func (k AllReduceResidualRMSNorm) Reference(rows int, symIn, residual, weight []float64) (residualOut, normOut []float64, err error) {
	return k.toNVL72().Reference(rows, symIn, residual, weight)
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

func (k NVLinkMoECombine) toNVL72() nvl72.NVLinkMoECombine {
	return nvl72.NVLinkMoECombine{
		Name:    k.Name,
		Cols:    k.Cols,
		TopK:    k.TopK,
		Ranks:   k.Ranks,
		DType:   nvl72.DType(k.DType),
		Threads: k.Threads,
	}
}

// Grid returns the launch grid and block for tokens token rows.
func (k NVLinkMoECombine) Grid(tokens int) (grid, block [3]int) {
	return k.toNVL72().Grid(tokens)
}

// CUDA returns the kernel definition without [Prelude].
func (k NVLinkMoECombine) CUDA() (string, error) {
	return k.toNVL72().CUDA()
}

// Launcher returns the host launcher function for k.
func (k NVLinkMoECombine) Launcher() (string, error) {
	return k.toNVL72().Launcher()
}

// Source returns a complete translation unit with [Prelude],
// [CollectivePrelude], the kernel, and its launcher.
func (k NVLinkMoECombine) Source() (string, error) {
	return k.toNVL72().Source()
}

// Reference computes the float64 reference for NVLinkMoECombine.
// symExpertOut has length Ranks * expertsPerRank * tokens * Cols.
// expertIDs and routerWeights have length tokens * TopK (or Ranks * tokens * TopK).
func (k NVLinkMoECombine) Reference(tokens, expertsPerRank int, symExpertOut []float64, expertIDs []int32, routerWeights []float64) ([]float64, error) {
	return k.toNVL72().Reference(tokens, expertsPerRank, symExpertOut, expertIDs, routerWeights)
}
