package nvl72

import (
	"fmt"
	"math"
	"strings"
)

// RubinRackTier identifies the Vera Rubin or Rubin Ultra NVLink 6 rack-scale
// system generation.
type RubinRackTier string

const (
	// TierVeraRubinNVL144 selects the Vera Rubin NVL144 rack (144 Rubin compute
	// dies across 72 dual-die packages, 36 Vera CPUs, NVLink 6 at 3.6 TB/s
	// bidirectional per die, 13 TB/s HBM4 per die, ConnectX-9 1600 Gb/s per die).
	TierVeraRubinNVL144 RubinRackTier = "VeraRubin-NVL144"

	// TierRubinUltraNVL576 selects the Rubin Ultra NVL576 rack (576 Rubin Ultra
	// compute dies across 144 quad-die packages, 36 Vera CPUs, NVLink 6 at
	// 1.5 PB/s rack switch bisection, 22 TB/s HBM4e per die, ConnectX-9 1600 Gb/s).
	TierRubinUltraNVL576 RubinRackTier = "RubinUltra-NVL576"
)

// Rubin NVL144 and Rubin Ultra NVL576 architectural constants.
const (
	RubinNVL144ComputeDies      = 144
	RubinNVL144PackagesPerRack  = 72
	RubinNVL144DiesPerPackage   = 2
	RubinUltraNVL576ComputeDies = 576
	RubinUltraPackagesPerRack   = 144
	RubinUltraDiesPerPackage    = 4

	VeraCPUsPerRack         = 36
	VeraOlympusCoresPerCPU  = 88
	RubinNVLinkVersion      = 6
	RubinNVLink6UnidirGBps  = 1800.0 // 1.8 TB/s unidirectional per compute die
	RubinNVLink6BidirGBps   = 3600.0 // 3.6 TB/s bidirectional per compute die (2x Blackwell NVLink 5)
	VeraC2CUnidirGBps       = 900.0  // 900 GB/s unidirectional coherent NVLink-C2C (2x Grace-Blackwell)
	VeraC2CBidirGBps        = 1800.0 // 1.8 TB/s bidirectional coherent NVLink-C2C
	ConnectX9GbpsPerDie     = 1600.0 // 1600 Gb/s SuperNIC per compute die (2x ConnectX-8)
	ConnectX9UnidirGBps     = 200.0  // 200 GB/s unidirectional scale-out RDMA per die
	RubinTargetArch         = "sm_107f"
	RubinMaxPodRacks        = 16
	RubinIntraPkgBridgeGBps = 3600.0 // Coherent die-to-die bridge bandwidth inside a dual/quad-die Rubin package
)

// RubinRackSpec describes the compute die, package, Vera CPU, NVLink 6 switch,
// HBM4, and ConnectX-9 scale-out architecture of a Vera Rubin NVL144 or Rubin
// Ultra NVL576 rack.
type RubinRackSpec struct {
	Tier                    RubinRackTier
	ComputeDiesPerRack      int     // 144 for Vera Rubin NVL144; 576 for Rubin Ultra NVL576
	PackagesPerRack         int     // 72 dual-die packages for NVL144; 144 quad-die packages for NVL576
	DiesPerPackage          int     // 2 for NVL144; 4 for NVL576
	VeraCPUsPerRack         int     // 36 Vera CPUs (88 Olympus Arm cores each)
	NVLinkVersion           int     // 6 (NVLink 6)
	NVLinkUnidirGBpsPerDie  float64 // 1800.0 GB/s unidirectional (3600.0 GB/s bidirectional per compute die)
	RackSwitchBisectionTBps float64 // 259.2 TB/s bidirectional for NVL144; 1500.0 TB/s (1.5 PB/s) for NVL576
	VeraC2CBandwidthGBps    float64 // 900.0 GB/s unidirectional (1800.0 GB/s bidirectional Coherent NVLink-C2C)
	HBM4BytesPerDie         int64   // 288 GiB on NVL144; 576 GiB on NVL576
	HBM4BandwidthGBpsPerDie float64 // 13000.0 GB/s (13 TB/s) on NVL144; 22000.0 GB/s (22 TB/s) on NVL576
	ConnectX9GbPerSecPerDie float64 // 1600.0 Gb/s (200.0 GB/s unidirectional per die)
	FP4DensePFLOPSPerDie    float64 // 50.0 PFLOPS dense NVFP4 on Rubin; 100.0 PFLOPS on Rubin Ultra
}

// DefaultVeraRubinNVL144Spec returns the canonical hardware specification for
// a Vera Rubin NVL144 rack (144 compute dies across 72 dual-die packages,
// 36 Vera CPUs, 3.6 TB/s bidirectional NVLink 6 per die, 13 TB/s HBM4 per die,
// and 1600 Gb/s ConnectX-9 per die).
func DefaultVeraRubinNVL144Spec() RubinRackSpec {
	return RubinRackSpec{
		Tier:                    TierVeraRubinNVL144,
		ComputeDiesPerRack:      RubinNVL144ComputeDies,
		PackagesPerRack:         RubinNVL144PackagesPerRack,
		DiesPerPackage:          RubinNVL144DiesPerPackage,
		VeraCPUsPerRack:         VeraCPUsPerRack,
		NVLinkVersion:           RubinNVLinkVersion,
		NVLinkUnidirGBpsPerDie:  RubinNVLink6UnidirGBps,
		RackSwitchBisectionTBps: 259.2,
		VeraC2CBandwidthGBps:    VeraC2CUnidirGBps,
		HBM4BytesPerDie:         288 << 30,
		HBM4BandwidthGBpsPerDie: 13000.0,
		ConnectX9GbPerSecPerDie: ConnectX9GbpsPerDie,
		FP4DensePFLOPSPerDie:    50.0,
	}
}

// DefaultRubinUltraNVL576Spec returns the canonical hardware specification for
// a Rubin Ultra NVL576 rack (576 compute dies across 144 quad-die packages,
// 36 Vera CPUs, 1.5 PB/s NVLink 6 switch bisection, 22 TB/s HBM4e per die, and
// 1600 Gb/s ConnectX-9 per die).
func DefaultRubinUltraNVL576Spec() RubinRackSpec {
	return RubinRackSpec{
		Tier:                    TierRubinUltraNVL576,
		ComputeDiesPerRack:      RubinUltraNVL576ComputeDies,
		PackagesPerRack:         RubinUltraPackagesPerRack,
		DiesPerPackage:          RubinUltraDiesPerPackage,
		VeraCPUsPerRack:         VeraCPUsPerRack,
		NVLinkVersion:           RubinNVLinkVersion,
		NVLinkUnidirGBpsPerDie:  RubinNVLink6UnidirGBps,
		RackSwitchBisectionTBps: 1500.0,
		VeraC2CBandwidthGBps:    VeraC2CUnidirGBps,
		HBM4BytesPerDie:         576 << 30,
		HBM4BandwidthGBpsPerDie: 22000.0,
		ConnectX9GbPerSecPerDie: ConnectX9GbpsPerDie,
		FP4DensePFLOPSPerDie:    100.0,
	}
}

// NVLinkBidirGBpsPerDie returns the bidirectional NVLink 6 bandwidth per
// compute die in GB/s (3600.0 GB/s = 3.6 TB/s).
func (s RubinRackSpec) NVLinkBidirGBpsPerDie() float64 {
	return 2.0 * s.NVLinkUnidirGBpsPerDie
}

// VeraC2CBidirGBps returns the bidirectional coherent NVLink-C2C bandwidth
// between Vera CPUs and Rubin packages in GB/s (1800.0 GB/s = 1.8 TB/s).
func (s RubinRackSpec) VeraC2CBidirGBps() float64 {
	return 2.0 * s.VeraC2CBandwidthGBps
}

// ConnectX9UnidirGBpsPerDie returns the unidirectional scale-out RDMA bandwidth
// per compute die in GB/s (200.0 GB/s for 1600 Gb/s ConnectX-9).
func (s RubinRackSpec) ConnectX9UnidirGBpsPerDie() float64 {
	return s.ConnectX9GbPerSecPerDie / 8.0
}

// Validate checks that s describes a consistent Vera Rubin NVL144 or Rubin
// Ultra NVL576 rack configuration.
func (s RubinRackSpec) Validate() error {
	switch s.Tier {
	case TierVeraRubinNVL144, TierRubinUltraNVL576:
	default:
		return fmt.Errorf("nvl72: unsupported RubinRackTier %q", s.Tier)
	}
	if s.PackagesPerRack < 1 || s.DiesPerPackage < 1 {
		return fmt.Errorf("nvl72: invalid package topology: %d packages * %d dies/package", s.PackagesPerRack, s.DiesPerPackage)
	}
	if s.ComputeDiesPerRack != s.PackagesPerRack*s.DiesPerPackage {
		return fmt.Errorf("nvl72: ComputeDiesPerRack %d != PackagesPerRack (%d) * DiesPerPackage (%d)",
			s.ComputeDiesPerRack, s.PackagesPerRack, s.DiesPerPackage)
	}
	if s.VeraCPUsPerRack < 1 {
		return fmt.Errorf("nvl72: VeraCPUsPerRack %d must be >= 1", s.VeraCPUsPerRack)
	}
	if s.NVLinkVersion < 6 {
		return fmt.Errorf("nvl72: NVLinkVersion %d must be >= 6", s.NVLinkVersion)
	}
	if s.NVLinkUnidirGBpsPerDie <= 0 || s.RackSwitchBisectionTBps <= 0 {
		return fmt.Errorf("nvl72: NVLink bandwidth (%v GB/s) and rack bisection (%v TB/s) must be > 0",
			s.NVLinkUnidirGBpsPerDie, s.RackSwitchBisectionTBps)
	}
	if s.VeraC2CBandwidthGBps <= 0 {
		return fmt.Errorf("nvl72: VeraC2CBandwidthGBps %v must be > 0", s.VeraC2CBandwidthGBps)
	}
	if s.HBM4BytesPerDie <= 0 || s.HBM4BandwidthGBpsPerDie <= 0 {
		return fmt.Errorf("nvl72: HBM4 capacity (%d) and bandwidth (%v GB/s) must be > 0",
			s.HBM4BytesPerDie, s.HBM4BandwidthGBpsPerDie)
	}
	if s.ConnectX9GbPerSecPerDie <= 0 {
		return fmt.Errorf("nvl72: ConnectX9GbPerSecPerDie %v must be > 0", s.ConnectX9GbPerSecPerDie)
	}
	if s.FP4DensePFLOPSPerDie <= 0 {
		return fmt.Errorf("nvl72: FP4DensePFLOPSPerDie %v must be > 0", s.FP4DensePFLOPSPerDie)
	}
	return nil
}

// RubinPodEPConfig configures multi-rack Wide Expert Parallelism (Wide-EP)
// estimation across 1..16 Vera Rubin NVL144 or Rubin Ultra NVL576 racks
// (modeling 144 up to 9216 compute dies).
type RubinPodEPConfig struct {
	Spec                          RubinRackSpec
	NumRacks                      int
	BatchTokensPerDie             int
	HiddenDim                     int
	TopK                          int
	ExpertsPerDie                 int
	UseHierarchicalPackageCombine bool
	UseFP4Dispatch                bool
}

// RubinPodEPEstimate summarizes capacity, dispatch/combine communication
// latencies, and speedup against an equivalent-die-count GB300 NVL72 +
// ConnectX-8 800 Gb/s pod.
type RubinPodEPEstimate struct {
	TotalDies           int
	TotalExperts        int
	TotalHBM4Bytes      int64
	IntraRackDispatchUS float64
	InterRackDispatchUS float64
	CombineLatencyUS    float64
	TotalCommLatencyUS  float64
	SpeedupVsGB300Pod   float64
}

// EstimateRubinPodWideEP estimates Wide-EP dispatch and combine communication
// latencies across cfg.NumRacks racks (1..16 racks, 144..9216 compute dies)
// and computes the communication speedup relative to an equivalent die count
// deployed on GB300 NVL72 racks with ConnectX-8 800 Gb/s scale-out fabrics.
func EstimateRubinPodWideEP(cfg RubinPodEPConfig) (RubinPodEPEstimate, error) {
	spec := cfg.Spec
	if spec.Tier == "" && spec.ComputeDiesPerRack == 0 {
		spec = DefaultVeraRubinNVL144Spec()
	}
	if err := spec.Validate(); err != nil {
		return RubinPodEPEstimate{}, err
	}
	if cfg.NumRacks < 1 || cfg.NumRacks > RubinMaxPodRacks {
		return RubinPodEPEstimate{}, fmt.Errorf("nvl72: NumRacks %d must be in 1..%d", cfg.NumRacks, RubinMaxPodRacks)
	}
	if cfg.BatchTokensPerDie < 1 {
		return RubinPodEPEstimate{}, fmt.Errorf("nvl72: BatchTokensPerDie %d must be >= 1", cfg.BatchTokensPerDie)
	}
	if cfg.HiddenDim < 2 || cfg.HiddenDim%2 != 0 {
		return RubinPodEPEstimate{}, fmt.Errorf("nvl72: HiddenDim %d must be a positive multiple of 2", cfg.HiddenDim)
	}
	if cfg.ExpertsPerDie < 1 {
		return RubinPodEPEstimate{}, fmt.Errorf("nvl72: ExpertsPerDie %d must be >= 1", cfg.ExpertsPerDie)
	}
	totalDies := cfg.NumRacks * spec.ComputeDiesPerRack
	totalExperts := totalDies * cfg.ExpertsPerDie
	if cfg.TopK < 1 || cfg.TopK > totalExperts {
		return RubinPodEPEstimate{}, fmt.Errorf("nvl72: TopK %d must be in 1..%d", cfg.TopK, totalExperts)
	}

	// Dispatch token payload size:
	//   - FP4 dispatch: 0.5 B/elem NVFP4 payload + 1 B per 16 elems block scale + 16 B metadata.
	//   - BF16 dispatch: 2 B/elem BF16 payload + 16 B routing metadata.
	dispatchPayloadBytes := cfg.HiddenDim * 2
	if cfg.UseFP4Dispatch {
		scaleBytes := (cfg.HiddenDim + 15) / 16
		dispatchPayloadBytes = cfg.HiddenDim/2 + scaleBytes
	}
	dispatchBytesPerToken := float64(dispatchPayloadBytes + 16)
	combineBytesPerToken := float64(cfg.HiddenDim * 2) // BF16 expert activations for combine

	tokens := float64(cfg.BatchTokensPerDie)
	topK := float64(cfg.TopK)
	routesPerDie := tokens * topK

	// Package-level deduplication factor: when UseHierarchicalPackageCombine is
	// enabled, tokens routed to multiple experts within the same dual-die or
	// quad-die Rubin package traverse the coherent intra-package die-to-die
	// bridge (3.6 TB/s) and coalesce before entering the NVLink 6 rack switch.
	pkgCoalesceFactor := 1.0
	if cfg.UseHierarchicalPackageCombine && spec.DiesPerPackage > 1 {
		dpp := float64(spec.DiesPerPackage)
		pkgCoalesceFactor = 1.0 - 0.18*(dpp-1.0)/dpp
	}

	const (
		nvlink6SwitchBaseUS   = 0.18 // NVLink 6 259.2 TB/s / 1.5 PB/s switch traversal base latency in us
		cx9RailTreeBaseUS     = 0.65 // ConnectX-9 1600 Gb/s rail-optimized tree hop latency in us
		cx9FlatContentionBase = 1.15 // Flat non-hierarchical scale-out switch contention base latency in us
	)

	// Intra-rack NVLink 6 dispatch latency in microseconds (bytes / (GB/s * 1e3)).
	intraRackDispatchBytes := routesPerDie * dispatchBytesPerToken * pkgCoalesceFactor
	intraRackDispatchUS := nvlink6SwitchBaseUS + intraRackDispatchBytes/(spec.NVLinkUnidirGBpsPerDie*1e3)

	// Inter-rack ConnectX-9 1600 Gb/s (200 GB/s unidir) dispatch latency in microseconds.
	cx9UnidirGBps := spec.ConnectX9UnidirGBpsPerDie()
	var (
		interRackDispatchUS float64
		remoteRoutesPerTok  float64
		uniqueRemoteRacks   float64
	)
	if cfg.NumRacks > 1 {
		rFloat := float64(cfg.NumRacks)
		remoteRoutesPerTok = topK * (rFloat - 1.0) / rFloat
		probMissRack := math.Pow((rFloat-1.0)/rFloat, topK)
		uniqueRemoteRacks = (rFloat - 1.0) * (1.0 - probMissRack)

		if cfg.UseHierarchicalPackageCombine {
			// Rail-optimized ConnectX-9 tree + rack-deduped + package-coalesced dispatch.
			interRackBytes := tokens * uniqueRemoteRacks * dispatchBytesPerToken * pkgCoalesceFactor
			interRackDispatchUS = cx9RailTreeBaseUS + interRackBytes/(cx9UnidirGBps*1e3)
		} else {
			// Flat all-to-all across racks without rack deduplication or package coalescing.
			interRackBytes := tokens * remoteRoutesPerTok * dispatchBytesPerToken
			interRackDispatchUS = cx9FlatContentionBase + interRackBytes/(cx9UnidirGBps*1e3)
		}
	}

	// Combine latency:
	//   - Intra-package die-to-die + intra-rack NVLink 6 partial reduction.
	//   - Inter-rack ConnectX-9 return when NumRacks > 1.
	intraRackCombineBytes := routesPerDie * combineBytesPerToken * pkgCoalesceFactor
	intraRackCombineUS := nvlink6SwitchBaseUS + intraRackCombineBytes/(spec.NVLinkUnidirGBpsPerDie*1e3)

	var interRackCombineUS float64
	if cfg.NumRacks > 1 {
		if cfg.UseHierarchicalPackageCombine {
			interRackCombineBytes := tokens * uniqueRemoteRacks * combineBytesPerToken * pkgCoalesceFactor
			interRackCombineUS = cx9RailTreeBaseUS + interRackCombineBytes/(cx9UnidirGBps*1e3)
		} else {
			interRackCombineBytes := tokens * remoteRoutesPerTok * combineBytesPerToken
			interRackCombineUS = cx9FlatContentionBase + interRackCombineBytes/(cx9UnidirGBps*1e3)
		}
	}
	combineLatencyUS := intraRackCombineUS + interRackCombineUS
	totalCommLatencyUS := intraRackDispatchUS + interRackDispatchUS + combineLatencyUS

	// Compare against an equivalent die/GPU count on a GB300 NVL72 pod
	// (72 GPUs/rack, NVLink 5 at 900 GB/s unidir, ConnectX-8 800 Gb/s at 100 GB/s unidir).
	gb300Racks := max((totalDies+NVL72MaxRanks-1)/NVL72MaxRanks, 1)
	gb300Est, err := EstimateMultiRackPodEP(
		GenGB300,
		gb300Racks,
		cfg.BatchTokensPerDie,
		cfg.HiddenDim,
		cfg.TopK,
		totalExperts,
	)
	if err != nil {
		return RubinPodEPEstimate{}, err
	}
	var gb300CommUS float64
	if cfg.UseHierarchicalPackageCombine {
		gb300CommUS = (gb300Est.HierarchicalDispatchNs + gb300Est.HierarchicalCombineNs) / 1e3
	} else {
		gb300FlatCombineNs := gb300Est.FlatDispatchNs - gb300Est.Stage2IntraRackDispatchNs + gb300Est.Stage1IntraRackCombineNs
		gb300CommUS = (gb300Est.FlatDispatchNs + gb300FlatCombineNs) / 1e3
	}
	speedup := 1.0
	if totalCommLatencyUS > 0 {
		speedup = gb300CommUS / totalCommLatencyUS
	}

	return RubinPodEPEstimate{
		TotalDies:           totalDies,
		TotalExperts:        totalExperts,
		TotalHBM4Bytes:      int64(totalDies) * spec.HBM4BytesPerDie,
		IntraRackDispatchUS: intraRackDispatchUS,
		InterRackDispatchUS: interRackDispatchUS,
		CombineLatencyUS:    combineLatencyUS,
		TotalCommLatencyUS:  totalCommLatencyUS,
		SpeedupVsGB300Pod:   speedup,
	}, nil
}

// RubinDisaggConfig configures Vera C2C and NVLink 6 / ConnectX-9 prefill/decode
// disaggregation planning across Vera Rubin NVL144 and Rubin Ultra NVL576 racks.
type RubinDisaggConfig struct {
	PrefillSpec               RubinRackSpec
	DecodeSpec                RubinRackSpec
	PrefillRacks              int
	DecodeRacks               int
	Model                     PDModelSpec
	SLA                       PDSLA
	PageTokens                int  // tokens per KV cache page (defaults to 16)
	UseLUTDecompressedPrefill bool // uses tcgen05 sm_107f LUT-decompressed / mxf4 prefill compute
	UseVeraC2CStaging         bool // stages KV pages via Vera CPU LPDDR5X (1.8 TB/s C2C)
}

// RubinDisaggPlan reports the prefill/decode disaggregation schedule, KV page
// transfer latency over ConnectX-9 (1600 Gb/s) + NVLink 6 (1800 GB/s unidir) +
// Vera CPU C2C (1.8 TB/s bidir), and TTFT/TPOT SLA satisfaction.
type RubinDisaggPlan struct {
	PrefillTier                   RubinRackTier
	DecodeTier                    RubinRackTier
	PrefillRacks                  int
	DecodeRacks                   int
	PrefillDies                   int
	DecodeDies                    int
	PrefillTargetArch             string
	KVBytesPerToken               int64
	KVPageBytes                   int64
	TotalKVPages                  int
	TotalKVTransferBytes          int64
	KVPageTransferUS              float64
	TotalKVTransferMS             float64
	VeraC2CStagingBandwidthGBps   float64
	ConnectX9TransferBandwidthGBps float64
	NVLink6IntraRackBandwidthGBps float64
	PrefillComputeMS              float64
	EstimatedTTFTMS               float64
	EstimatedTPOTMS               float64
	MeetsTTFT                     bool
	MeetsTPOT                     bool
	MeetsSLA                      bool
	ClusterTokensPerSec           float64
}

// PlanRubinDisaggregation models disaggregated Prefill racks (Vera Rubin NVL144
// using tcgen05 sm_107f LUT-decompressed / mxf4 compute) streaming KV pages
// over ConnectX-9 (1600 Gb/s = 200 GB/s unidir per die), intra-rack NVLink 6
// (1800 GB/s unidir), and Vera CPU LPDDR5X (900 GB/s unidir / 1.8 TB/s bidir
// coherent C2C) to Decode racks (NVL144 or NVL576 with 13-22 TB/s HBM4).
func PlanRubinDisaggregation(cfg RubinDisaggConfig) (RubinDisaggPlan, error) {
	prefillSpec := cfg.PrefillSpec
	if prefillSpec.Tier == "" && prefillSpec.ComputeDiesPerRack == 0 {
		prefillSpec = DefaultVeraRubinNVL144Spec()
	}
	if err := prefillSpec.Validate(); err != nil {
		return RubinDisaggPlan{}, fmt.Errorf("prefill spec: %w", err)
	}
	decodeSpec := cfg.DecodeSpec
	if decodeSpec.Tier == "" && decodeSpec.ComputeDiesPerRack == 0 {
		decodeSpec = DefaultVeraRubinNVL144Spec()
	}
	if err := decodeSpec.Validate(); err != nil {
		return RubinDisaggPlan{}, fmt.Errorf("decode spec: %w", err)
	}
	prefillRacks := cfg.PrefillRacks
	if prefillRacks < 1 {
		prefillRacks = 1
	}
	decodeRacks := cfg.DecodeRacks
	if decodeRacks < 1 {
		decodeRacks = 1
	}
	if cfg.SLA.PromptTokens < 1 {
		return RubinDisaggPlan{}, fmt.Errorf("nvl72: PromptTokens %d must be >= 1", cfg.SLA.PromptTokens)
	}
	if cfg.SLA.DecodeBatchPerGPU < 1 {
		return RubinDisaggPlan{}, fmt.Errorf("nvl72: DecodeBatchPerGPU %d must be >= 1", cfg.SLA.DecodeBatchPerGPU)
	}
	if cfg.SLA.MaxTTFTMs <= 0 || cfg.SLA.MaxTPOTMs <= 0 {
		return RubinDisaggPlan{}, fmt.Errorf("nvl72: MaxTTFTMs (%v) and MaxTPOTMs (%v) must be > 0",
			cfg.SLA.MaxTTFTMs, cfg.SLA.MaxTPOTMs)
	}
	if cfg.Model.NumLayers < 1 {
		return RubinDisaggPlan{}, fmt.Errorf("nvl72: NumLayers %d must be >= 1", cfg.Model.NumLayers)
	}
	if cfg.Model.HiddenSize < 2 || cfg.Model.HiddenSize%2 != 0 {
		return RubinDisaggPlan{}, fmt.Errorf("nvl72: HiddenSize %d must be a positive multiple of 2", cfg.Model.HiddenSize)
	}

	kvBytesPerTokenPerLayer, err := cfg.Model.kvBytesPerTokenPerLayer()
	if err != nil {
		return RubinDisaggPlan{}, err
	}
	pageTokens := cfg.PageTokens
	if pageTokens < 1 {
		pageTokens = 16
	}
	kvBytesPerToken := kvBytesPerTokenPerLayer * int64(cfg.Model.NumLayers)
	kvPageBytes := int64(pageTokens) * kvBytesPerTokenPerLayer
	pagesPerLayer := (cfg.SLA.PromptTokens + pageTokens - 1) / pageTokens
	totalKVPages := pagesPerLayer * cfg.Model.NumLayers
	totalKVTransferBytes := int64(cfg.SLA.PromptTokens) * kvBytesPerToken

	prefillDies := prefillRacks * prefillSpec.ComputeDiesPerRack
	decodeDies := decodeRacks * decodeSpec.ComputeDiesPerRack

	// Effective multi-rail KV streaming bandwidth combines ConnectX-9 (200 GB/s/die)
	// across the participating prefill shard (e.g. 8-way TP/CP pipeline group)
	// pipelined with intra-rack NVLink 6 (1800 GB/s/die) and Vera C2C (1800 GB/s bidir).
	cx9PerDieGBps := prefillSpec.ConnectX9UnidirGBpsPerDie()
	nvlink6GBps := prefillSpec.NVLinkUnidirGBpsPerDie
	veraC2CBidirGBps := prefillSpec.VeraC2CBidirGBps()

	streamGroupDies := float64(min(prefillSpec.ComputeDiesPerRack, 8))
	effectiveStreamGBps := math.Min(streamGroupDies*cx9PerDieGBps, nvlink6GBps)
	if cfg.UseVeraC2CStaging {
		// Pipelined Vera CPU LPDDR5X C2C staging overlaps RDMA egress with prefill compute.
		effectiveStreamGBps = math.Min(effectiveStreamGBps*1.15, veraC2CBidirGBps)
	}

	kvPageTransferUS := float64(kvPageBytes) / (effectiveStreamGBps * 1e3)
	totalKVTransferMS := float64(totalKVTransferBytes) / (effectiveStreamGBps * 1e6)

	topK := cfg.Model.TopK
	if topK < 1 {
		topK = 8
	}
	totalExperts := cfg.Model.TotalExperts
	if totalExperts < topK {
		totalExperts = max(topK, 144)
	}
	numQHeads := cfg.Model.NumQHeads
	if numQHeads < 1 {
		numQHeads = 64
	}
	headDim := cfg.Model.HeadDim
	if headDim < 1 {
		headDim = 128
	}
	activeParams := cfg.Model.ActiveParamsBillions * 1e9
	if activeParams <= 0 {
		activeParams = float64(cfg.Model.NumLayers) * 12.0 * float64(cfg.Model.HiddenSize) * float64(cfg.Model.HiddenSize)
	}
	totalParams := cfg.Model.TotalParamsBillions * 1e9
	if totalParams < activeParams {
		totalParams = activeParams * float64(max(totalExperts/topK, 1))
	}

	// Prefill compute on a single Vera Rubin NVL144 rack instance (144 dies,
	// 50 PFLOPS dense mxf4 / LUT-decompressed per die on sm_107f).
	lutBoost := 1.0
	if cfg.UseLUTDecompressedPrefill {
		lutBoost = 1.25 // tcgen05 sm_107f in-register LUT decompression + mxf4 speedup
	}
	instanceDies := float64(prefillSpec.ComputeDiesPerRack)
	tensorRate := instanceDies * prefillSpec.FP4DensePFLOPSPerDie * 1e15 * 0.62 * lutBoost
	sfuRate := instanceDies * 1.4e15 // 4x GB200 SFU attention throughput per rack

	linearFLOPs := 2.0 * float64(cfg.SLA.PromptTokens) * activeParams
	attnFLOPs := 2.0 * float64(cfg.Model.NumLayers) * float64(cfg.SLA.PromptTokens) * float64(cfg.SLA.PromptTokens) * float64(numQHeads*headDim)
	sfuOps := float64(cfg.Model.NumLayers) * float64(cfg.SLA.PromptTokens) * float64(cfg.SLA.PromptTokens) * float64(numQHeads) * 16.0

	prefillEP, err := EstimateRubinPodWideEP(RubinPodEPConfig{
		Spec:                          prefillSpec,
		NumRacks:                      1,
		BatchTokensPerDie:             max(cfg.SLA.PromptTokens/prefillSpec.ComputeDiesPerRack, 1),
		HiddenDim:                     cfg.Model.HiddenSize,
		TopK:                          topK,
		ExpertsPerDie:                 max((totalExperts+prefillSpec.ComputeDiesPerRack-1)/prefillSpec.ComputeDiesPerRack, 1),
		UseHierarchicalPackageCombine: true,
		UseFP4Dispatch:                true,
	})
	if err != nil {
		return RubinDisaggPlan{}, err
	}
	prefillCommMS := float64(cfg.Model.NumLayers) * prefillEP.TotalCommLatencyUS / 1e3 * 0.45 // overlapped via TBO
	prefillComputeMS := ((linearFLOPs+attnFLOPs)/tensorRate+sfuOps/sfuRate)*1e3 + prefillCommMS
	estimatedTTFTMS := prefillComputeMS + totalKVTransferMS

	// Decode TPOT on decodeRacks of DecodeSpec (HBM4 13 TB/s on NVL144 or 22 TB/s on NVL576).
	decodeEP, err := EstimateRubinPodWideEP(RubinPodEPConfig{
		Spec:                          decodeSpec,
		NumRacks:                      min(decodeRacks, RubinMaxPodRacks),
		BatchTokensPerDie:             cfg.SLA.DecodeBatchPerGPU,
		HiddenDim:                     cfg.Model.HiddenSize,
		TopK:                          topK,
		ExpertsPerDie:                 max((totalExperts+decodeDies-1)/decodeDies, 1),
		UseHierarchicalPackageCombine: true,
		UseFP4Dispatch:                true,
	})
	if err != nil {
		return RubinDisaggPlan{}, err
	}
	epStepMS := float64(cfg.Model.NumLayers) * decodeEP.TotalCommLatencyUS / 1e3 * 0.45

	weightBytesPerDie := (totalParams * 0.5) / float64(decodeDies)
	kvScanBytesPerDie := float64(cfg.SLA.DecodeBatchPerGPU) * float64(totalKVTransferBytes) / float64(decodeSpec.ComputeDiesPerRack)
	hbm4ScanMS := (weightBytesPerDie + kvScanBytesPerDie) / (decodeSpec.HBM4BandwidthGBpsPerDie * 1e6)
	decodeSFUMS := (float64(cfg.Model.NumLayers) * float64(cfg.SLA.DecodeBatchPerGPU) * float64(cfg.SLA.PromptTokens) * float64(numQHeads) * 16.0) /
		(float64(decodeDies) * 1.4e15) * 1e3

	estimatedTPOTMS := epStepMS + hbm4ScanMS + decodeSFUMS

	meetsTTFT := estimatedTTFTMS <= cfg.SLA.MaxTTFTMs
	meetsTPOT := estimatedTPOTMS <= cfg.SLA.MaxTPOTMs
	meetsSLA := meetsTTFT && meetsTPOT

	prefillTokPerSec := float64(prefillRacks) * float64(cfg.SLA.PromptTokens) * (1000.0 / estimatedTTFTMS)
	decodeTokPerSec := float64(decodeDies*cfg.SLA.DecodeBatchPerGPU) * (1000.0 / estimatedTPOTMS)
	clusterTokPerSec := (2.0 * prefillTokPerSec * decodeTokPerSec) / (prefillTokPerSec + decodeTokPerSec)

	return RubinDisaggPlan{
		PrefillTier:                    prefillSpec.Tier,
		DecodeTier:                     decodeSpec.Tier,
		PrefillRacks:                   prefillRacks,
		DecodeRacks:                    decodeRacks,
		PrefillDies:                    prefillDies,
		DecodeDies:                     decodeDies,
		PrefillTargetArch:              RubinTargetArch,
		KVBytesPerToken:                kvBytesPerToken,
		KVPageBytes:                    kvPageBytes,
		TotalKVPages:                   totalKVPages,
		TotalKVTransferBytes:           totalKVTransferBytes,
		KVPageTransferUS:               kvPageTransferUS,
		TotalKVTransferMS:              totalKVTransferMS,
		VeraC2CStagingBandwidthGBps:    veraC2CBidirGBps,
		ConnectX9TransferBandwidthGBps: cx9PerDieGBps,
		NVLink6IntraRackBandwidthGBps:  nvlink6GBps,
		PrefillComputeMS:               prefillComputeMS,
		EstimatedTTFTMS:                estimatedTTFTMS,
		EstimatedTPOTMS:                estimatedTPOTMS,
		MeetsTTFT:                      meetsTTFT,
		MeetsTPOT:                      meetsTPOT,
		MeetsSLA:                       meetsSLA,
		ClusterTokensPerSec:            clusterTokPerSec,
	}, nil
}

// rubinNVLink6MultimemPrelude provides NVLink 6 symmetric-memory PTX wrappers
// for sm_107f (multimem.ld_reduce.relaxed.sys.global.add.f32 and
// multimem.st.relaxed.sys.global.f32) along with cudahostemu multi-rank hooks.
const rubinNVLink6MultimemPrelude = `#ifndef EG_RUBIN_NVLINK6_MULTIMEM_PRELUDE
#define EG_RUBIN_NVLINK6_MULTIMEM_PRELUDE

EG_DEV float eg_rubin_multimem_ld_reduce_relaxed_sys_add_f32(const float* sym_base, size_t elem_idx) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return eg_emu_multimem_ld_reduce_add_f32(sym_base, elem_idx);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float r;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(r) : "l"(sym_base + elem_idx));
    return r;
#else
    return sym_base[elem_idx];
#endif
}

EG_DEV void eg_rubin_multimem_st_relaxed_sys_f32(float* sym_base, size_t elem_idx, float val) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    eg_emu_multimem_st_f32(sym_base, elem_idx, val);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;" :: "l"(sym_base + elem_idx), "f"(val) : "memory");
#else
    sym_base[elem_idx] = val;
#endif
}

#endif // EG_RUBIN_NVLINK6_MULTIMEM_PRELUDE
`

// RubinNVL144AllReduceResidualRMSNorm wraps a 144-rank NVLink 6 symmetric
// memory collective kernel (Target "sm_107f", supporting NumRanks in 1..576,
// defaulting to 144) that fuses multimem.ld_reduce.relaxed.sys.global.add.f32
// across all ranks with residual addition, multimem.st.relaxed.sys.global.f32
// broadcast of the post-residual sum, and RMSNorm.
type RubinNVL144AllReduceResidualRMSNorm struct {
	Name      string
	Target    string
	NumRanks  int
	HiddenDim int
	Cols      int
	Eps       float32
	Threads   int
}

func (k RubinNVL144AllReduceResidualRMSNorm) norm() (RubinNVL144AllReduceResidualRMSNorm, error) {
	if k.Name == "" {
		k.Name = "rubin_nvl144_allreduce_residual_rmsnorm"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("rubin nvl144 allreduce residual rmsnorm: bad kernel name %q", k.Name)
	}
	if k.Target == "" {
		k.Target = RubinTargetArch
	}
	if k.Target != RubinTargetArch {
		return k, fmt.Errorf("rubin nvl144 allreduce residual rmsnorm %s: target %q must be %q", k.Name, k.Target, RubinTargetArch)
	}
	if k.NumRanks == 0 {
		k.NumRanks = RubinNVL144ComputeDies
	}
	if k.NumRanks < 1 || k.NumRanks > RubinUltraNVL576ComputeDies {
		return k, fmt.Errorf("rubin nvl144 allreduce residual rmsnorm %s: NumRanks %d must be in 1..%d",
			k.Name, k.NumRanks, RubinUltraNVL576ComputeDies)
	}
	cols := k.HiddenDim
	if cols == 0 {
		cols = k.Cols
	}
	if cols == 0 {
		cols = 128
	}
	if cols < 2 || cols%2 != 0 {
		return k, fmt.Errorf("rubin nvl144 allreduce residual rmsnorm %s: HiddenDim %d must be a positive multiple of 2", k.Name, cols)
	}
	k.HiddenDim = cols
	k.Cols = cols
	if k.Eps == 0 {
		k.Eps = 1e-6
	}
	if k.Threads == 0 {
		k.Threads = 256
	}
	if k.Threads < 32 || k.Threads > 1024 || k.Threads%32 != 0 {
		return k, fmt.Errorf("rubin nvl144 allreduce residual rmsnorm %s: Threads %d must be a multiple of 32 in 32..1024",
			k.Name, k.Threads)
	}
	return k, nil
}

func (k RubinNVL144AllReduceResidualRMSNorm) params() []string {
	return []string{
		"int rows",
		"const float* __restrict__ sym_partials",
		"const float* __restrict__ residual",
		"const float* __restrict__ weight",
		"float* __restrict__ sum_out",
		"float* __restrict__ norm_out",
	}
}

// Grid returns the launch grid and CTA block dimensions for rows token rows.
func (k RubinNVL144AllReduceResidualRMSNorm) Grid(rows int) (grid, block [3]int) {
	threads := k.Threads
	if threads == 0 {
		threads = 256
	}
	if rows < 1 {
		rows = 1
	}
	return [3]int{rows, 1, 1}, [3]int{threads, 1, 1}
}

// CUDA returns the device kernel source for RubinNVL144AllReduceResidualRMSNorm
// without file preludes.
func (k RubinNVL144AllReduceResidualRMSNorm) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	per := (k.HiddenDim + k.Threads - 1) / k.Threads
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-rank NVLink 6 (%s) multimem.ld_reduce.relaxed.sys.global.add + multimem.st.relaxed.sys.global + RMSNorm over %d cols.\n",
		k.Name, k.NumRanks, k.Target, k.HiddenDim)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int COLS = %d, THREADS = %d, PER = %d, NUM_RANKS = %d;\n",
		k.HiddenDim, k.Threads, per, k.NumRanks)
	fmt.Fprintf(&b, "    constexpr float EPS = %s;\n", floatLiteral(k.Eps))
	b.WriteString(`    (void)NUM_RANKS;
    const int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    __shared__ float eg_red[32];

    const size_t rank_elems = (size_t)rows * COLS;
    const float* local_res = eg_nvlink_local_ptr(residual, rank_elems);
    const float* local_w = eg_nvlink_local_ptr(weight, (size_t)COLS);

    float h_reg[PER];
    float sq = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int col = tid + j * THREADS;
        if (col < COLS) {
            const size_t idx = (size_t)row * COLS + col;
            const float ar = eg_rubin_multimem_ld_reduce_relaxed_sys_add_f32(sym_partials, idx);
            const float res = local_res[idx];
            const float h = ar + res;
            eg_rubin_multimem_st_relaxed_sys_f32(sum_out, idx, h);
            h_reg[j] = h;
            sq += h * h;
        } else {
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
            const float w = local_w[col];
            const float y = h_reg[j] * inv_rms * w;
            eg_rubin_multimem_st_relaxed_sys_f32(norm_out, idx, y);
        }
    }
}
`)
	return b.String(), nil
}

// Launcher returns the host launcher function for RubinNVL144AllReduceResidualRMSNorm.
func (k RubinNVL144AllReduceResidualRMSNorm) Launcher() (string, error) {
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

// CUDAFile returns a complete standalone CUDA translation unit for target
// sm_107f including [Prelude], [CollectivePrelude], NVLink 6 multimem PTX
// helpers, the kernel definition, and its host launcher.
func (k RubinNVL144AllReduceResidualRMSNorm) CUDAFile() (string, error) {
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

// Source is an alias for [RubinNVL144AllReduceResidualRMSNorm.CUDAFile].
func (k RubinNVL144AllReduceResidualRMSNorm) Source() (string, error) {
	return k.CUDAFile()
}

// Reference computes the exact host reference for
// RubinNVL144AllReduceResidualRMSNorm across len(partials) ranks (e.g. 144
// ranks on Vera Rubin NVL144). Each partials[r] slice and residual have length
// rows * HiddenDim, and weight has length HiddenDim.
func (k RubinNVL144AllReduceResidualRMSNorm) Reference(partials [][]float32, residual, weight []float32) (sumOut, normOut []float32, err error) {
	if k.HiddenDim == 0 && k.Cols == 0 && len(weight) >= 2 && len(weight)%2 == 0 {
		k.HiddenDim = len(weight)
	}
	if k.NumRanks == 0 && len(partials) > 0 {
		k.NumRanks = len(partials)
	}
	k, err = k.norm()
	if err != nil {
		return nil, nil, err
	}
	if len(partials) != k.NumRanks {
		return nil, nil, fmt.Errorf("rubin nvl144 allreduce residual rmsnorm %s reference: partials len %d != NumRanks %d",
			k.Name, len(partials), k.NumRanks)
	}
	if len(weight) != k.HiddenDim {
		return nil, nil, fmt.Errorf("rubin nvl144 allreduce residual rmsnorm %s reference: weight len %d != HiddenDim %d",
			k.Name, len(weight), k.HiddenDim)
	}
	if len(residual) == 0 || len(residual)%k.HiddenDim != 0 {
		return nil, nil, fmt.Errorf("rubin nvl144 allreduce residual rmsnorm %s reference: residual len %d must be a positive multiple of %d",
			k.Name, len(residual), k.HiddenDim)
	}
	totalElems := len(residual)
	for r, p := range partials {
		if len(p) != totalElems {
			return nil, nil, fmt.Errorf("rubin nvl144 allreduce residual rmsnorm %s reference: partials[%d] len %d != %d",
				k.Name, r, len(p), totalElems)
		}
	}

	rows := totalElems / k.HiddenDim
	sumOut = make([]float32, totalElems)
	normOut = make([]float32, totalElems)

	for row := 0; row < rows; row++ {
		var sq float32
		rowOff := row * k.HiddenDim
		for col := 0; col < k.HiddenDim; col++ {
			idx := rowOff + col
			var ar float32
			for r := 0; r < k.NumRanks; r++ {
				ar += partials[r][idx]
			}
			h := ar + residual[idx]
			sumOut[idx] = h
			sq += h * h
		}
		meanSq := sq/float32(k.HiddenDim) + k.Eps
		invRMS := float32(1.0 / math.Sqrt(float64(meanSq)))
		for col := 0; col < k.HiddenDim; col++ {
			idx := rowOff + col
			normOut[idx] = sumOut[idx] * invRMS * weight[col]
		}
	}
	return sumOut, normOut, nil
}
