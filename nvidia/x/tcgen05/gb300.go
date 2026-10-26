package tcgen05

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// GB300Spec describes the hardware capabilities of an NVIDIA Blackwell Ultra
// B300 / GB300 GPU (compute capability 10.3, PTX target sm_103a, family target
// sm_100f) in the Cordelia socketed SXM Puck + ConnectX-8 SuperNIC platform.
type GB300Spec struct {
	Target                 string  // PTX architecture-accelerated target ("sm_103a")
	FamilyTarget           string  // PTX family-conditional target ("sm_100f")
	HBMBytes               int64   // 12-Hi HBM3e capacity per GPU in bytes (288 GiB)
	RackHBMBytes           int64   // 72-GPU NVL72 rack HBM3e capacity in bytes (~20.7 TiB)
	HBMBandwidthGBps       float64 // HBM3e bandwidth per GPU in GB/s (8000 GB/s = 8.0 TB/s)
	DenseFP4TFLOPS         float64 // dense NVFP4/MXFP4 throughput in TFLOPS (15,000 TFLOPS = 15 PFLOPS, 1.5x B200)
	DenseFP8TFLOPS         float64 // dense FP8/BF16 throughput in TFLOPS (5,000 TFLOPS = 5 PFLOPS)
	AttentionSFUMultiplier float64 // MUFU EX2/TANH throughput multiplier vs B200 (2.0x)
	ScaleOutBandwidthGBps  float64 // ConnectX-8 SuperNIC PCIe Gen6 x16 scale-out bandwidth in GB/s (100 GB/s = 800 Gb/s)
	GraceLPDDR5XBytes      int64   // Grace CPU LPDDR5X capacity in bytes (512 GiB per Grace CPU)
}

// DefaultGB300Spec returns the canonical hardware specification for an NVIDIA
// Blackwell Ultra GB300 (sm_103a) GPU.
func DefaultGB300Spec() GB300Spec {
	const gib = 1024 * 1024 * 1024
	return GB300Spec{
		Target:                 "sm_103a",
		FamilyTarget:           "sm_100f",
		HBMBytes:               288 * gib,
		RackHBMBytes:           72 * 288 * gib,
		HBMBandwidthGBps:       8000,
		DenseFP4TFLOPS:         15000,
		DenseFP8TFLOPS:         5000,
		AttentionSFUMultiplier: 2.0,
		ScaleOutBandwidthGBps:  100,
		GraceLPDDR5XBytes:      512 * gib,
	}
}

// TCGen05ScaleVec4XGEMM emits a Blackwell Ultra sm_103a NVFP4 block-scaled GEMM
// using tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
// (K = 128 FP4 elements per MMA instruction with 4 UE4M3 scale factors per
// 64-byte K-step staged into TMEM via tcgen05.cp.cta_group::1.32x128b).
type TCGen05ScaleVec4XGEMM struct {
	Name string
	M    int // defaults to 128 (must be 128)
	N    int // defaults to 128 (must be 128)
	K    int // defaults to 128 (must be 64, 128, 192, or 256)
}

func (k TCGen05ScaleVec4XGEMM) asBlockScaled() (TCGen05BlockScaledGEMM, error) {
	name := k.Name
	if name == "" {
		name = "tcgen05_scale_vec_4x_gemm"
	}
	K := k.K
	if K == 0 {
		K = 128
	}
	bs := TCGen05BlockScaledGEMM{
		Name:       name,
		M:          k.M,
		N:          k.N,
		K:          K,
		ScaleVec4X: true,
	}
	return bs.norm()
}

// Grid returns the single-CTA grid and 128-thread (4-warp) block dimensions.
func (k TCGen05ScaleVec4XGEMM) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

// TraceEvents returns the formal sm_103a tcgen05 hardware protocol event trace
// for verification with cudatmem.VerifyTrace(cudatmem.SM103A, ...).
func (k TCGen05ScaleVec4XGEMM) TraceEvents() []cudatmem.TraceEvent {
	bs := TCGen05BlockScaledGEMM{
		Name:       k.Name,
		M:          k.M,
		N:          k.N,
		K:          k.K,
		ScaleVec4X: true,
	}
	return bs.TraceEvents()
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05ScaleVec4XGEMM) CUDA() (string, error) {
	bs, err := k.asBlockScaled()
	if err != nil {
		return "", fmt.Errorf("tcgen05 scale_vec::4X gemm: %w", err)
	}
	return bs.CUDA()
}

// CUDAFile emits the self-contained CUDA C++ translation unit (including
// Prelude) for k, ready for cudahostemu.Run or nvcc.
func (k TCGen05ScaleVec4XGEMM) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// GB300ResidencyPlan summarizes the HBM3e memory residency and NVLink-C2C cold
// spill fraction for a model deployed across numGPUs Blackwell Ultra GB300 GPUs.
type GB300ResidencyPlan struct {
	Target            string  // PTX target ("sm_103a")
	NumGPUs           int     // number of GPUs in the deployment domain
	HBMBytesPerGPU    int64   // HBM3e capacity per GPU in bytes
	AvailableHBMBytes int64   // total HBM3e capacity across NumGPUs in bytes
	WeightBytes       int64   // total quantized weight + block-scale footprint in bytes
	KVCacheBytes      int64   // total KV cache footprint at (batchSize, seqLen) in bytes
	ActivationBytes   int64   // activation and scratch workspace footprint in bytes
	TotalRequiredBytes int64  // WeightBytes + KVCacheBytes + ActivationBytes
	RemainingHBMBytes int64   // AvailableHBMBytes - TotalRequiredBytes (0 when spilling)
	SpilledC2CBytes   int64   // bytes spilled to Grace LPDDR5X over NVLink-C2C (0 when FitsInHBM)
	C2CColdFraction   float64 // fraction of WeightBytes spilled to Grace LPDDR5X in [0, 1]
	MaxHBMContextLen  int64   // maximum sequence length per request that fits entirely in HBM3e
	FitsInHBM         bool    // true when TotalRequiredBytes <= AvailableHBMBytes
}

// PlanGB300Residency computes the HBM3e residency plan on 288 GiB/GPU Blackwell
// Ultra GB300 (sm_103a) GPUs for a model with modelParams parameters quantized
// to weightBits bits per parameter, kvBytesPerToken bytes of KV state per token,
// and active request shape (batchSize, seqLen) distributed across numGPUs GPUs.
// For 4-bit block-scaled formats (NVFP4/MXFP4), it includes the 1 UE4M3 byte
// per 16-element block scale overhead (0.5 bits/param).
func PlanGB300Residency(modelParams int64, weightBits int, kvBytesPerToken int64, batchSize, seqLen, numGPUs int) GB300ResidencyPlan {
	spec := DefaultGB300Spec()
	return PlanHBMResidencyWithCapacity(spec.Target, spec.HBMBytes, modelParams, weightBits, kvBytesPerToken, batchSize, seqLen, numGPUs)
}

// PlanHBMResidencyWithCapacity computes an HBM3e residency plan for an arbitrary
// per-GPU HBM capacity (e.g., 288 GiB on GB300 sm_103a vs 192 GiB on GB200 sm_100a).
func PlanHBMResidencyWithCapacity(target string, hbmBytesPerGPU int64, modelParams int64, weightBits int, kvBytesPerToken int64, batchSize, seqLen, numGPUs int) GB300ResidencyPlan {
	if numGPUs <= 0 {
		numGPUs = 1
	}
	if weightBits <= 0 {
		weightBits = 4
	}
	if batchSize <= 0 {
		batchSize = 1
	}
	if seqLen < 0 {
		seqLen = 0
	}

	// Quantized weight payload + block scale overhead for 4-bit NVFP4/MXFP4
	// (1 byte UE4M3 scale per 16 parameters = 0.5 bits/param).
	weightBytes := (modelParams*int64(weightBits) + 7) / 8
	if weightBits == 4 {
		weightBytes += (modelParams + 15) / 16
	}

	kvCacheBytes := int64(batchSize) * int64(seqLen) * kvBytesPerToken
	// Reserve 2 GiB per GPU for TMEM/TMA staging, collective buffers, and activation workspace.
	const activationPerGPU = 2 * 1024 * 1024 * 1024
	activationBytes := int64(numGPUs) * activationPerGPU

	availHBM := int64(numGPUs) * hbmBytesPerGPU
	totalReq := weightBytes + kvCacheBytes + activationBytes

	var remainingHBM, spilledBytes int64
	var coldFrac float64
	fits := totalReq <= availHBM
	if fits {
		remainingHBM = availHBM - totalReq
	} else {
		spilledBytes = totalReq - availHBM
		if weightBytes > 0 {
			coldFrac = float64(spilledBytes) / float64(weightBytes)
			if coldFrac > 1.0 {
				coldFrac = 1.0
			}
		}
	}

	var maxCtx int64
	headroomForKV := availHBM - weightBytes - activationBytes
	denom := int64(batchSize) * kvBytesPerToken
	if headroomForKV > 0 && denom > 0 {
		maxCtx = headroomForKV / denom
	}

	return GB300ResidencyPlan{
		Target:             target,
		NumGPUs:            numGPUs,
		HBMBytesPerGPU:     hbmBytesPerGPU,
		AvailableHBMBytes:  availHBM,
		WeightBytes:        weightBytes,
		KVCacheBytes:       kvCacheBytes,
		ActivationBytes:    activationBytes,
		TotalRequiredBytes: totalReq,
		RemainingHBMBytes:  remainingHBM,
		SpilledC2CBytes:    spilledBytes,
		C2CColdFraction:    coldFrac,
		MaxHBMContextLen:   maxCtx,
		FitsInHBM:          fits,
	}
}
