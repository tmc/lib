package nvl72

import (
	"fmt"
	"math"
	"strings"
)

// GB200 vs. GB300 NVL72 4-tier pod generation names and hardware constants.
const (
	// GenGB200 selects the GB200 NVL72 rack generation (192 GB HBM3e 8-Hi per
	// GPU, ConnectX-7 400 Gb/s = 50 GB/s scale-out RDMA per GPU).
	GenGB200 = "GB200"

	// GenGB300 selects the GB300 NVL72 (Blackwell Ultra) rack generation
	// (288 GB HBM3e 12-Hi per GPU, ConnectX-8 800 Gb/s PCIe Gen6 = 100 GB/s
	// scale-out RDMA per GPU).
	GenGB300 = "GB300"

	// Tier 1: Local HBM3e capacity and bandwidth per GPU.
	GB200HBM3eCapGB = 192 // 8-Hi HBM3e capacity in GB per GPU on GB200
	GB300HBM3eCapGB = 288 // 12-Hi HBM3e capacity in GB per GPU on GB300

	// Tier 2: Intra-Rack 72-GPU NVLink 5 bandwidth per GPU.
	NVL72NVLink5BidiGBps = 1800.0 // 1.8 TB/s bidirectional NVLink 5 per GPU

	// Tier 4: Inter-Rack Scale-Out RDMA NIC bandwidth per GPU.
	ConnectX7NICGbps = 400.0 // 400 Gb/s ConnectX-7 per GPU on GB200
	ConnectX7UniGBps = 50.0  // 50 GB/s unidirectional RDMA per GPU on GB200
	ConnectX8NICGbps = 800.0 // 800 Gb/s ConnectX-8 PCIe Gen6 per GPU on GB300
	ConnectX8UniGBps = 100.0 // 100 GB/s unidirectional RDMA per GPU on GB300

	// Per-GPU Tensor Core compute ceilings in TFLOPS (dense BF16/FP8 equivalent).
	GB200TensorTFLOPS = 2250.0
	GB300TensorTFLOPS = 3375.0 // 1.5x Blackwell Ultra dense/FP4 compute scaling
)

// PodSpec describes the 4-tier memory and fabric hierarchy of a GB200 or GB300
// NVL72 rack inside a multi-rack pod:
//
//   - Tier 1: Local HBM3e (8.0 TB/s; 192 GB on GB200 vs. 288 GB 12-Hi on GB300)
//   - Tier 2: Intra-Rack 72-GPU NVLink 5 (900 GB/s unidirectional / 1.8 TB/s bidirectional per GPU)
//   - Tier 3: Intra-Tray Grace LPDDR5X NVLink-C2C (450 GB/s per GPU)
//   - Tier 4: Inter-Rack Scale-Out RDMA (50 GB/s ConnectX-7 400G on GB200 vs. 100 GB/s ConnectX-8 800G PCIe Gen6 on GB300)
type PodSpec struct {
	Generation           string
	GPUsPerRack          int
	TraysPerRack         int
	HBM3eCapacityGB      int
	RackHBM3eCapacityTB  float64
	Tier1HBM3eGBps       float64
	Tier2NVLink5UniGBps  float64
	Tier2NVLink5BidiGBps float64
	Tier3GraceC2CGBps    float64
	Tier4ScaleOutNIC     string
	Tier4ScaleOutGbps    float64
	Tier4ScaleOutGBps    float64
	TensorTFLOPS         float64
}

// LookupPodSpec returns the 4-tier hardware specification for gen ("GB200" or
// "GB300", case-insensitive).
func LookupPodSpec(gen string) (PodSpec, error) {
	switch strings.ToUpper(strings.TrimSpace(gen)) {
	case GenGB200:
		return PodSpec{
			Generation:           GenGB200,
			GPUsPerRack:          NVL72MaxRanks,
			TraysPerRack:         NVL72NumTrays,
			HBM3eCapacityGB:      GB200HBM3eCapGB,
			RackHBM3eCapacityTB:  float64(NVL72MaxRanks*GB200HBM3eCapGB) / 1000.0,
			Tier1HBM3eGBps:       NVL72HBM3eGBps,
			Tier2NVLink5UniGBps:  NVL72NVLink5UniGBps,
			Tier2NVLink5BidiGBps: NVL72NVLink5BidiGBps,
			Tier3GraceC2CGBps:    NVL72GraceC2CGBps,
			Tier4ScaleOutNIC:     "ConnectX-7 400G",
			Tier4ScaleOutGbps:    ConnectX7NICGbps,
			Tier4ScaleOutGBps:    ConnectX7UniGBps,
			TensorTFLOPS:         GB200TensorTFLOPS,
		}, nil
	case GenGB300:
		return PodSpec{
			Generation:           GenGB300,
			GPUsPerRack:          NVL72MaxRanks,
			TraysPerRack:         NVL72NumTrays,
			HBM3eCapacityGB:      GB300HBM3eCapGB,
			RackHBM3eCapacityTB:  float64(NVL72MaxRanks*GB300HBM3eCapGB) / 1000.0,
			Tier1HBM3eGBps:       NVL72HBM3eGBps,
			Tier2NVLink5UniGBps:  NVL72NVLink5UniGBps,
			Tier2NVLink5BidiGBps: NVL72NVLink5BidiGBps,
			Tier3GraceC2CGBps:    NVL72GraceC2CGBps,
			Tier4ScaleOutNIC:     "ConnectX-8 800G PCIe Gen6",
			Tier4ScaleOutGbps:    ConnectX8NICGbps,
			Tier4ScaleOutGBps:    ConnectX8UniGBps,
			TensorTFLOPS:         GB300TensorTFLOPS,
		}, nil
	default:
		return PodSpec{}, fmt.Errorf("nvl72: unsupported pod generation %q (want %q or %q)", gen, GenGB200, GenGB300)
	}
}

// PodEPEstimate models hierarchical Two-Stage All-to-All Wide-EP traffic,
// rack-deduped RDMA routing, 4-tier memory/fabric residency, and Two-Batch
// Overlap (TBO) across numRacks * 72 GPUs (e.g., 2 racks = 144 GPUs, 4 racks =
// 288 GPUs, 8 racks = 576 GPUs).
type PodEPEstimate struct {
	Generation    string
	NumRacks      int
	TotalGPUs     int
	TokensPerGPU  int
	HiddenSize    int
	TopK          int
	TotalExperts  int
	ExpertsPerGPU int
	Spec          PodSpec

	// 4-Tier capacity and per-GPU bandwidths.
	HBM3eCapacityPerGPUGB int
	PodHBM3eCapacityTB    float64
	Tier1HBM3eGBps        float64 // 8000 GB/s (8.0 TB/s)
	Tier2NVLink5UniGBps   float64 // 900 GB/s unidirectional (1.8 TB/s bidi)
	Tier3GraceC2CGBps     float64 // 450 GB/s per GPU
	Tier4RDMAUniGBps      float64 // 50 GB/s (ConnectX-7 400G) vs. 100 GB/s (ConnectX-8 800G)

	// Per-token routing fan-out across racks.
	ExpectedRemoteRoutesPerToken float64 // Flat remote-rack routes per token: K * (numRacks - 1) / numRacks
	ExpectedUniqueRemoteRacks    float64 // Rack-deduped remote racks per token: (R - 1) * (1 - ((R - 1) / R)^K)
	RackDedupSavingsRatio        float64 // Fraction of inter-rack RDMA bytes saved by rack-deduped Stage 1 dispatch

	// Per-GPU traffic in bytes across the 4-tier hierarchy.
	FlatInterRackDispatchBytes         uint64 // Non-hierarchical outbound Tier 4 RDMA dispatch bytes per GPU
	HierarchicalInterRackDispatchBytes uint64 // Stage 1 rack-deduped outbound Tier 4 RDMA dispatch bytes per GPU
	Stage2IntraRackNVLinkDispatchBytes uint64 // Stage 2 intra-rack Tier 2 NVLink 5 fan-out bytes per GPU
	Stage1IntraRackNVLinkCombineBytes  uint64 // Stage 1 intra-rack Tier 2 NVLink 5 partial-sum combine bytes per GPU
	HierarchicalInterRackCombineBytes  uint64 // Stage 2 rack-deduped inbound Tier 4 RDMA combine bytes per GPU
	LocalHBMBytes                      uint64 // Tier 1 HBM3e expert MLP + RMSNorm bytes per GPU

	// Stage latencies in nanoseconds.
	FlatDispatchNs            float64 // Non-deduped Tier 4 RDMA + Tier 2 NVLink dispatch latency
	Stage1InterRackDispatchNs float64 // Hierarchical Stage 1 Tier 4 RDMA dispatch latency (ConnectX-7/8)
	Stage2IntraRackDispatchNs float64 // Hierarchical Stage 2 Tier 2 NVLink 5 fan-out + multicast barrier latency
	HierarchicalDispatchNs    float64 // Stage1InterRackDispatchNs + Stage2IntraRackDispatchNs
	ExpertComputeNs           float64 // Local Tier 1 HBM3e / Tensor Core expert MLP latency
	Stage1IntraRackCombineNs  float64 // Hierarchical Stage 1 Tier 2 NVLink 5 partial reduction latency
	Stage2InterRackCombineNs  float64 // Hierarchical Stage 2 Tier 4 RDMA combine latency (ConnectX-7/8)
	HierarchicalCombineNs     float64 // Stage1IntraRackCombineNs + Stage2InterRackCombineNs + RMSNorm latency
	SingleBatchTotalNs        float64 // HierarchicalDispatchNs + ExpertComputeNs + HierarchicalCombineNs

	// Two-Batch Overlap (TBO) metrics across a 2-microbatch window.
	SequentialTwoBatchNs float64 // 2 * SingleBatchTotalNs without communication/compute overlap
	TBOTwoBatchNs        float64 // 2 microbatches overlapping hierarchical RDMA+NVLink comm with expert compute
	TBOOverlapSavingsNs  float64 // SequentialTwoBatchNs - TBOTwoBatchNs
	TBOSpeedup           float64 // SequentialTwoBatchNs / TBOTwoBatchNs
}

// EstimateMultiRackPodEP models hierarchical Two-Stage All-to-All / Wide-EP
// latency across numRacks * 72 GPUs (e.g., 2 racks = 144 GPUs, 4 racks = 288
// GPUs, 8 racks = 576 GPUs) on GB200 (ConnectX-7 400G = 50 GB/s, 192 GB HBM3e)
// vs. GB300 (ConnectX-8 800G PCIe Gen6 = 100 GB/s, 288 GB 12-Hi HBM3e).
//
// In hierarchical rack-deduped dispatch:
//   - Stage 1 (Tier 4 Inter-Rack RDMA): each token is sent at most once to
//     each remote destination rack over ConnectX-7/8 RDMA, eliminating
//     duplicate cross-rack transfers when multiple Top-K experts for the same
//     token reside in the same destination rack.
//   - Stage 2 (Tier 2 Intra-Rack NVLink 5, 900 GB/s): the ingress rank on each
//     destination rack fans the token out over the 72-GPU NVSwitch fabric to
//     all target experts in that rack.
//   - Combine reverses the hierarchy: intra-rack NVLink 5 partial reduction
//     first collapses multiple expert outputs per rack into one vector per
//     originating token before returning a single rack-deduped vector over
//     Tier 4 RDMA.
func EstimateMultiRackPodEP(gen string, numRacks, tokensPerGPU, hiddenSize, topK, totalExperts int) (PodEPEstimate, error) {
	spec, err := LookupPodSpec(gen)
	if err != nil {
		return PodEPEstimate{}, err
	}
	if numRacks < 1 {
		return PodEPEstimate{}, fmt.Errorf("nvl72: numRacks %d must be >= 1", numRacks)
	}
	if tokensPerGPU < 1 {
		return PodEPEstimate{}, fmt.Errorf("nvl72: tokensPerGPU %d must be >= 1", tokensPerGPU)
	}
	if hiddenSize < 2 || hiddenSize%2 != 0 {
		return PodEPEstimate{}, fmt.Errorf("nvl72: hiddenSize %d must be a positive multiple of 2", hiddenSize)
	}
	if totalExperts < 1 {
		return PodEPEstimate{}, fmt.Errorf("nvl72: totalExperts %d must be >= 1", totalExperts)
	}
	if topK < 1 || topK > totalExperts {
		return PodEPEstimate{}, fmt.Errorf("nvl72: topK %d must be in 1..%d", topK, totalExperts)
	}

	totalGPUs := numRacks * spec.GPUsPerRack
	expertsPerGPU := max((totalExperts+totalGPUs-1)/totalGPUs, 1)

	// BF16 token payload (2B/elem) + 16B routing/provenance metadata for dispatch,
	// and BF16 payload (2B/elem) for combine.
	dispatchBytesPerToken := hiddenSize*2 + 16
	combineBytesPerToken := hiddenSize * 2

	var (
		expectedRemoteRoutesPerToken float64
		expectedUniqueRemoteRacks    float64
		rackDedupSavingsRatio        float64
	)
	if numRacks > 1 {
		rFloat := float64(numRacks)
		expectedRemoteRoutesPerToken = float64(topK) * (rFloat - 1.0) / rFloat
		// Probability a specific remote rack is not selected in K draws = ((R-1)/R)^K.
		probMissRack := math.Pow((rFloat-1.0)/rFloat, float64(topK))
		expectedUniqueRemoteRacks = (rFloat - 1.0) * (1.0 - probMissRack)
		if expectedRemoteRoutesPerToken > 0 {
			rackDedupSavingsRatio = math.Max(0.0, 1.0-expectedUniqueRemoteRacks/expectedRemoteRoutesPerToken)
		}
	}

	flatInterRackDispatchBytes := uint64(math.Round(float64(tokensPerGPU) * expectedRemoteRoutesPerToken * float64(dispatchBytesPerToken)))
	hierInterRackDispatchBytes := uint64(math.Round(float64(tokensPerGPU) * expectedUniqueRemoteRacks * float64(dispatchBytesPerToken)))

	// Stage 2 Intra-Rack NVLink 5 fan-out: every one of the K routed tokens
	// (both locally originated and RDMA-forwarded from remote racks) is
	// distributed across the 72-GPU intra-rack NVSwitch domain.
	routesPerGPU := tokensPerGPU * topK
	stage2IntraRackNVLinkDispatchBytes := uint64(routesPerGPU * (hiddenSize*2 + 8))

	// Hierarchical Combine:
	// Stage 1 Intra-Rack NVLink 5 partial reduction aggregates local rack
	// expert outputs, and Stage 2 Inter-Rack RDMA sends one partially-reduced
	// vector per unique remote rack back to the source rack.
	stage1IntraRackNVLinkCombineBytes := uint64(routesPerGPU * combineBytesPerToken)
	hierInterRackCombineBytes := uint64(math.Round(float64(tokensPerGPU) * expectedUniqueRemoteRacks * float64(combineBytesPerToken)))

	// Local Tier 1 HBM3e traffic: expert MLP activations + local expert weights + residual RMSNorm.
	expertMLPActBytes := uint64(routesPerGPU * hiddenSize * 2 * 6)
	expertWeightBytes := uint64(expertsPerGPU) * uint64(hiddenSize) * uint64(hiddenSize) * 2
	rmsNormBytes := uint64(tokensPerGPU * hiddenSize * 6)
	localHBMBytes := expertMLPActBytes + expertWeightBytes + rmsNormBytes

	// Latencies in nanoseconds (bytes / (GB/s) = ns).
	const (
		nvswitchMulticastBaseNs = 350.0
		rdmaFabricBaseNs        = 1200.0 // Scale-out InfiniBand/RoCE switch hop base latency when numRacks > 1
	)
	var rdmaBaseNs float64
	if numRacks > 1 {
		rdmaBaseNs = rdmaFabricBaseNs
	}

	flatRDMANs := rdmaBaseNs + float64(flatInterRackDispatchBytes)/spec.Tier4ScaleOutGBps
	stage1InterRackDispatchNs := rdmaBaseNs + float64(hierInterRackDispatchBytes)/spec.Tier4ScaleOutGBps
	stage2IntraRackDispatchNs := nvswitchMulticastBaseNs + float64(stage2IntraRackNVLinkDispatchBytes)/spec.Tier2NVLink5UniGBps
	flatDispatchNs := flatRDMANs + stage2IntraRackDispatchNs
	hierDispatchNs := stage1InterRackDispatchNs + stage2IntraRackDispatchNs

	expertFLOPs := float64(2 * 3 * routesPerGPU * hiddenSize * hiddenSize)
	hbmComputeNs := float64(localHBMBytes) / spec.Tier1HBM3eGBps
	tensorComputeNs := expertFLOPs / (spec.TensorTFLOPS * 1000.0)
	expertComputeNs := math.Max(hbmComputeNs, tensorComputeNs)

	stage1IntraRackCombineNs := float64(stage1IntraRackNVLinkCombineBytes) / spec.Tier2NVLink5UniGBps
	stage2InterRackCombineNs := rdmaBaseNs + float64(hierInterRackCombineBytes)/spec.Tier4ScaleOutGBps
	rmsNormNs := float64(rmsNormBytes) / spec.Tier1HBM3eGBps
	hierCombineNs := stage1IntraRackCombineNs + stage2InterRackCombineNs + rmsNormNs

	singleBatchTotalNs := hierDispatchNs + expertComputeNs + hierCombineNs
	seqTwoBatchNs := 2.0 * singleBatchTotalNs

	// Two-Batch Overlap (TBO) schedule across 2 microbatches (MB0, MB1):
	//   Step 1: MB0 Hierarchical Dispatch (RDMA Stage 1 + NVLink Stage 2)
	//   Step 2: overlap(MB0 ExpertCompute, MB1 Hierarchical Dispatch)
	//   Step 3: overlap(MB1 ExpertCompute, MB0 Hierarchical Combine)
	//   Step 4: MB1 Hierarchical Combine
	tboTwoBatchNs := hierDispatchNs +
		math.Max(expertComputeNs, hierDispatchNs) +
		math.Max(expertComputeNs, hierCombineNs) +
		hierCombineNs
	savingsNs := seqTwoBatchNs - tboTwoBatchNs
	speedup := 1.0
	if tboTwoBatchNs > 0 {
		speedup = seqTwoBatchNs / tboTwoBatchNs
	}

	return PodEPEstimate{
		Generation:                         spec.Generation,
		NumRacks:                           numRacks,
		TotalGPUs:                          totalGPUs,
		TokensPerGPU:                       tokensPerGPU,
		HiddenSize:                         hiddenSize,
		TopK:                               topK,
		TotalExperts:                       totalExperts,
		ExpertsPerGPU:                      expertsPerGPU,
		Spec:                               spec,
		HBM3eCapacityPerGPUGB:              spec.HBM3eCapacityGB,
		PodHBM3eCapacityTB:                 float64(numRacks) * spec.RackHBM3eCapacityTB,
		Tier1HBM3eGBps:                     spec.Tier1HBM3eGBps,
		Tier2NVLink5UniGBps:                spec.Tier2NVLink5UniGBps,
		Tier3GraceC2CGBps:                  spec.Tier3GraceC2CGBps,
		Tier4RDMAUniGBps:                   spec.Tier4ScaleOutGBps,
		ExpectedRemoteRoutesPerToken:       expectedRemoteRoutesPerToken,
		ExpectedUniqueRemoteRacks:          expectedUniqueRemoteRacks,
		RackDedupSavingsRatio:              rackDedupSavingsRatio,
		FlatInterRackDispatchBytes:         flatInterRackDispatchBytes,
		HierarchicalInterRackDispatchBytes: hierInterRackDispatchBytes,
		Stage2IntraRackNVLinkDispatchBytes: stage2IntraRackNVLinkDispatchBytes,
		Stage1IntraRackNVLinkCombineBytes:  stage1IntraRackNVLinkCombineBytes,
		HierarchicalInterRackCombineBytes:  hierInterRackCombineBytes,
		LocalHBMBytes:                      localHBMBytes,
		FlatDispatchNs:                     flatDispatchNs,
		Stage1InterRackDispatchNs:          stage1InterRackDispatchNs,
		Stage2IntraRackDispatchNs:          stage2IntraRackDispatchNs,
		HierarchicalDispatchNs:             hierDispatchNs,
		ExpertComputeNs:                    expertComputeNs,
		Stage1IntraRackCombineNs:           stage1IntraRackCombineNs,
		Stage2InterRackCombineNs:           stage2InterRackCombineNs,
		HierarchicalCombineNs:              hierCombineNs,
		SingleBatchTotalNs:                 singleBatchTotalNs,
		SequentialTwoBatchNs:               seqTwoBatchNs,
		TBOTwoBatchNs:                      tboTwoBatchNs,
		TBOOverlapSavingsNs:                savingsNs,
		TBOSpeedup:                         speedup,
	}, nil
}
