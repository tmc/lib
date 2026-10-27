package hopper

import "github.com/tmc/lib/nvidia/x/nvl72"

// GB200 NVL72 rack topology and 3-tier bandwidth constants.
const (
	NVL72MaxRanks              = nvl72.NVL72MaxRanks
	NVL72NumTrays              = nvl72.NVL72NumTrays
	NVL72GPUsPerTray           = nvl72.NVL72GPUsPerTray
	NVL72HBM3eGBps             = nvl72.NVL72HBM3eGBps
	NVL72NVLink5UniGBps        = nvl72.NVL72NVLink5UniGBps
	NVL72GraceC2CGBps          = nvl72.NVL72GraceC2CGBps
	NVL72BlackwellTensorTFLOPS = nvl72.NVL72BlackwellTensorTFLOPS
)

// NVL72FabricHelpers returns the device and host-emulation C++/PTX helpers for
// GB200 NVL72 18-tray / 72-GPU rack topology, O(1) NVSwitch multicast barriers,
// multicast stores, and the 72-rank cooperative host emulator runner.
func NVL72FabricHelpers() string {
	return nvl72.NVL72FabricHelpers()
}

// NVL72WideEPPlan computes the deterministic routing plan across R <= 72 ranks.
type NVL72WideEPPlan = nvl72.NVL72WideEPPlan

// NVL72CostEstimate models the 3-tier memory/fabric traffic and Two-Batch
// Overlap (TBO) latency of GB200 NVL72 Wide Expert Parallelism.
type NVL72CostEstimate = nvl72.NVL72CostEstimate

// EstimateNVL72WideEP models the GB200 NVL72 3-tier roofline and computes the
// latency savings from Two-Batch Overlap (TBO).
func EstimateNVL72WideEP(tokensPerRank, hiddenSize, topK, totalExperts, numRanks int, c2cColdFraction float64) NVL72CostEstimate {
	return nvl72.EstimateNVL72WideEP(tokensPerRank, hiddenSize, topK, totalExperts, numRanks, c2cColdFraction)
}

// NVL72WideEPDispatch pushes each rank's local input tokens over NVLink 5
// directly into dest_rank's symmetric receive buffer.
type NVL72WideEPDispatch struct {
	Name          string
	TokensPerRank int
	HiddenSize    int
	TopK          int
	TotalExperts  int
	Ranks         int
	DType         DType
	QuantizeScale float32
	Threads       int
}

func (k NVL72WideEPDispatch) toNVL72() nvl72.NVL72WideEPDispatch {
	return nvl72.NVL72WideEPDispatch{
		Name:          k.Name,
		TokensPerRank: k.TokensPerRank,
		HiddenSize:    k.HiddenSize,
		TopK:          k.TopK,
		TotalExperts:  k.TotalExperts,
		Ranks:         k.Ranks,
		DType:         nvl72.DType(k.DType),
		QuantizeScale: k.QuantizeScale,
		Threads:       k.Threads,
	}
}

// Grid returns the per-rank cooperative launch grid and block.
func (k NVL72WideEPDispatch) Grid() (grid, block [3]int) {
	return k.toNVL72().Grid()
}

// CUDA returns the kernel definition for NVL72WideEPDispatch.
func (k NVL72WideEPDispatch) CUDA() (string, error) {
	return k.toNVL72().CUDA()
}

// Launcher returns the host launcher function for NVL72WideEPDispatch.
func (k NVL72WideEPDispatch) Launcher() (string, error) {
	return k.toNVL72().Launcher()
}

// Source returns a complete translation unit for NVL72WideEPDispatch.
func (k NVL72WideEPDispatch) Source() (string, error) {
	return k.toNVL72().Source()
}

// Reference computes the Go float64 reference for NVL72WideEPDispatch.
func (k NVL72WideEPDispatch) Reference(
	x []float64,
	topkIDs []int32,
	topkWeights []float32,
	destSlots []int32,
) (symRecvX []float64, srcInfo []int32, recvWeights []float32, err error) {
	return k.toNVL72().Reference(x, topkIDs, topkWeights, destSlots)
}

// NVL72WideEPExpertScale applies per-expert transformation on each rank's
// received token slots with Grace NVLink-C2C cold-expert offload.
type NVL72WideEPExpertScale struct {
	Name          string
	TokensPerRank int
	HiddenSize    int
	TopK          int
	TotalExperts  int
	Ranks         int
	DType         DType
	Threads       int
}

func (k NVL72WideEPExpertScale) toNVL72() nvl72.NVL72WideEPExpertScale {
	return nvl72.NVL72WideEPExpertScale{
		Name:          k.Name,
		TokensPerRank: k.TokensPerRank,
		HiddenSize:    k.HiddenSize,
		TopK:          k.TopK,
		TotalExperts:  k.TotalExperts,
		Ranks:         k.Ranks,
		DType:         nvl72.DType(k.DType),
		Threads:       k.Threads,
	}
}

// CUDA returns the kernel definition for NVL72WideEPExpertScale.
func (k NVL72WideEPExpertScale) CUDA() (string, error) {
	return k.toNVL72().CUDA()
}

// Launcher returns the host launcher function for NVL72WideEPExpertScale.
func (k NVL72WideEPExpertScale) Launcher() (string, error) {
	return k.toNVL72().Launcher()
}

// NVL72WideEPCombine gathers the K routed expert outputs across up to 72 ranks
// over NVLink 5 or Grace NVLink-C2C and applies fused residual RMSNorm.
type NVL72WideEPCombine struct {
	Name          string
	TokensPerRank int
	HiddenSize    int
	TopK          int
	TotalExperts  int
	Ranks         int
	Eps           float32
	DType         DType
	Threads       int
}

func (k NVL72WideEPCombine) toNVL72() nvl72.NVL72WideEPCombine {
	return nvl72.NVL72WideEPCombine{
		Name:          k.Name,
		TokensPerRank: k.TokensPerRank,
		HiddenSize:    k.HiddenSize,
		TopK:          k.TopK,
		TotalExperts:  k.TotalExperts,
		Ranks:         k.Ranks,
		Eps:           k.Eps,
		DType:         nvl72.DType(k.DType),
		Threads:       k.Threads,
	}
}

// Grid returns the per-rank cooperative launch grid and block.
func (k NVL72WideEPCombine) Grid() (grid, block [3]int) {
	return k.toNVL72().Grid()
}

// CUDA returns the kernel definition for NVL72WideEPCombine.
func (k NVL72WideEPCombine) CUDA() (string, error) {
	return k.toNVL72().CUDA()
}

// Launcher returns the host launcher function for NVL72WideEPCombine.
func (k NVL72WideEPCombine) Launcher() (string, error) {
	return k.toNVL72().Launcher()
}

// Source returns a complete translation unit for NVL72WideEPCombine.
func (k NVL72WideEPCombine) Source() (string, error) {
	return k.toNVL72().Source()
}

// Reference computes the Go float64 reference for NVL72WideEPCombine.
func (k NVL72WideEPCombine) Reference(
	expertOut []float64,
	c2cColdExpertOut []float64,
	coldExpertMask []int32,
	topkIDs []int32,
	topkWeights []float32,
	destSlots []int32,
	residual []float64,
	rmsWeight []float64,
) (residualOut, normOut []float64, err error) {
	return k.toNVL72().Reference(expertOut, c2cColdExpertOut, coldExpertMask, topkIDs, topkWeights, destSlots, residual, rmsWeight)
}

// NVL72WideEP bundles the complete 4-stage GB200 NVL72 Wide-EP pipeline.
type NVL72WideEP struct {
	Name          string
	TokensPerRank int
	HiddenSize    int
	TopK          int
	TotalExperts  int
	Ranks         int
	Eps           float32
	DType         DType
	QuantizeScale float32
	Threads       int
}

func (w NVL72WideEP) toNVL72() nvl72.NVL72WideEP {
	return nvl72.NVL72WideEP{
		Name:          w.Name,
		TokensPerRank: w.TokensPerRank,
		HiddenSize:    w.HiddenSize,
		TopK:          w.TopK,
		TotalExperts:  w.TotalExperts,
		Ranks:         w.Ranks,
		Eps:           w.Eps,
		DType:         nvl72.DType(w.DType),
		QuantizeScale: w.QuantizeScale,
		Threads:       w.Threads,
	}
}

// Stages returns the four constituent kernel descriptors for w.
func (w NVL72WideEP) Stages() (NVL72WideEPPlan, NVL72WideEPDispatch, NVL72WideEPExpertScale, NVL72WideEPCombine, error) {
	plan, disp, scale, comb, err := w.toNVL72().Stages()
	if err != nil {
		return NVL72WideEPPlan{}, NVL72WideEPDispatch{}, NVL72WideEPExpertScale{}, NVL72WideEPCombine{}, err
	}
	return plan,
		NVL72WideEPDispatch{
			Name:          disp.Name,
			TokensPerRank: disp.TokensPerRank,
			HiddenSize:    disp.HiddenSize,
			TopK:          disp.TopK,
			TotalExperts:  disp.TotalExperts,
			Ranks:         disp.Ranks,
			DType:         DType(disp.DType),
			QuantizeScale: disp.QuantizeScale,
			Threads:       disp.Threads,
		},
		NVL72WideEPExpertScale{
			Name:          scale.Name,
			TokensPerRank: scale.TokensPerRank,
			HiddenSize:    scale.HiddenSize,
			TopK:          scale.TopK,
			TotalExperts:  scale.TotalExperts,
			Ranks:         scale.Ranks,
			DType:         DType(scale.DType),
			Threads:       scale.Threads,
		},
		NVL72WideEPCombine{
			Name:          comb.Name,
			TokensPerRank: comb.TokensPerRank,
			HiddenSize:    comb.HiddenSize,
			TopK:          comb.TopK,
			TotalExperts:  comb.TotalExperts,
			Ranks:         comb.Ranks,
			Eps:           comb.Eps,
			DType:         DType(comb.DType),
			Threads:       comb.Threads,
		},
		nil
}

// Source returns a single translation unit containing all four pipeline kernels.
func (w NVL72WideEP) Source() (string, error) {
	return w.toNVL72().Source()
}

// Reference executes the full 4-stage float64 reference pipeline.
func (w NVL72WideEP) Reference(
	x []float64,
	topkIDs []int32,
	topkWeights []float32,
	expertScales []float64,
	coldExpertMask []int32,
	residual []float64,
	rmsWeight []float64,
) (
	tokensPerExpert, expertOffsets, destSlots, srcInfo []int32,
	residualOut, normOut []float64,
	err error,
) {
	return w.toNVL72().Reference(x, topkIDs, topkWeights, expertScales, coldExpertMask, residual, rmsWeight)
}
