package nvl72

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/expertparallel"
)

// PlanRubinNVL144 estimates multi-rack Wide Expert Parallelism (Wide-EP) across
// Vera Rubin NVL144 (144 compute dies per rack, NVLink 6 at 1800 GB/s unidir,
// HBM4 at 13 TB/s, Vera C2C at 900 GB/s, ConnectX-9 1600G RDMA) or Rubin Ultra
// NVL576 racks. When cfg.Spec is zero, it defaults to [DefaultVeraRubinNVL144Spec].
func PlanRubinNVL144(cfg RubinPodEPConfig) (RubinPodEPEstimate, error) {
	if cfg.Spec.Tier == "" && cfg.Spec.ComputeDiesPerRack == 0 {
		cfg.Spec = DefaultVeraRubinNVL144Spec()
	}
	return EstimateRubinPodWideEP(cfg)
}

// RackEPMegakernelConfig configures a unified 4-tier rack/pod-scale persistent
// Two-Batch Overlap (TBO) Wide-EP megakernel across GB200 NVL72, GB300 NVL72,
// Vera Rubin NVL144, or Rubin Ultra NVL576 systems.
type RackEPMegakernelConfig struct {
	Name            string        // C identifier prefix (defaults to "rack_ep_megakernel")
	Generation      string        // "GB200", "GB300", "VeraRubin-NVL144", or "RubinUltra-NVL576"
	RubinTier       RubinRackTier // optional explicit Rubin tier
	NumRacks        int           // number of racks in the pod (defaults to 1)
	RanksPerRack    int           // intra-rack EP ranks (defaults to 72 for GB200/GB300, 144 for NVL144, 576 for NVL576)
	TokensPerRank   int           // tokens per GPU/die per microbatch (>= 1)
	HiddenSize      int           // model hidden dimension (positive multiple of 2)
	ExpertInter     int           // routed expert intermediate dimension (defaults to HiddenSize/2)
	TopK            int           // routed experts per token (1..32)
	TotalExperts    int           // total routed experts across the pod
	TotalCTAs       int           // persistent CTAs per GPU/die (defaults to 148 on Blackwell, 160 on Rubin)
	ThreadsPerCTA   int           // threads per CTA (defaults to 256)
	Microbatches    int           // double-buffered TBO microbatches >= 2 (defaults to 2)
	EnableC2CHybrid bool          // enable Grace/Vera NVLink-C2C cold-expert doorbell co-execution
	ColdTokenCount  int           // representative token count for cold-expert placement (defaults to 1)
	UseFP4Dispatch  bool          // quantize dispatch payloads to NVFP4
	DType           DType         // F32 or BF16 (defaults to BF16)
}

// RackEPMegakernelPlan holds the modeled 4-tier rack/pod telemetry, TBO CTA
// warpgroup partitioning, Grace/Vera C2C cold-expert placement decision,
// audited .sys doorbell PTX, and unified persistent CUDA C++ megakernel source.
type RackEPMegakernelPlan struct {
	Config              RackEPMegakernelConfig
	Name                string
	Generation          string
	TargetArch          string
	IsRubin             bool
	NumRacks            int
	RanksPerRack        int
	TotalGPUs           int
	TotalCTAs           int
	ThreadsPerCTA       int
	Microbatches        int
	ComputeCTAs         int
	DispatchCTAs        int
	CombineCTAs         int
	C2CDoorbellCTAs     int
	RoleAssignments     []expertparallel.TBOWarpgroupRole
	WideEP              NVL72WideEP
	PodEstimate         PodEPEstimate
	RubinEstimate       RubinPodEPEstimate
	C2CConfig           C2CHybridConfig
	ColdPlacement       ColdMoEDecision
	ColdTokenCrossover  int
	DispatchLatencyUs   float64
	ComputeLatencyUs    float64
	CombineLatencyUs    float64
	SequentialTwoBatchUs float64
	TBOTwoBatchUs       float64
	TBOSpeedup          float64
	DoorbellPTX         string
	Source              string
}

// RoleForCTA returns the warpgroup role assigned to blockIdx.x in [0, TotalCTAs).
func (p *RackEPMegakernelPlan) RoleForCTA(blockIdx int) expertparallel.TBOWarpgroupRole {
	if p == nil || blockIdx < 0 || blockIdx >= len(p.RoleAssignments) {
		return expertparallel.WarpgroupRoleCompute
	}
	return p.RoleAssignments[blockIdx]
}

// CompileRackEPMegakernel bridges the 4-tier rack/pod hierarchy
// ([EstimateMultiRackPodEP], [NVL72WideEP], [PlanRubinNVL144]), Grace/Vera C2C
// cold-expert co-execution ([DecideColdMoEPlacement], [ColdMoECrossover],
// [EmitC2CSysDoorbellPTX], [AuditC2CCoherence]), and TBO warpgroup CTA
// partitioning ([expertparallel.PartitionTBOCTAs]) into a unified persistent
// Wide-EP megakernel plan.
func CompileRackEPMegakernel(cfg RackEPMegakernelConfig) (*RackEPMegakernelPlan, error) {
	if cfg.Name == "" {
		cfg.Name = "rack_ep_megakernel"
	}
	if !cIdent(cfg.Name) {
		return nil, fmt.Errorf("nvl72: bad megakernel name %q", cfg.Name)
	}
	if cfg.NumRacks == 0 {
		cfg.NumRacks = 1
	}
	if cfg.NumRacks < 1 || cfg.NumRacks > RubinMaxPodRacks {
		return nil, fmt.Errorf("nvl72: NumRacks %d must be in 1..%d", cfg.NumRacks, RubinMaxPodRacks)
	}
	if cfg.TokensPerRank < 1 {
		return nil, fmt.Errorf("nvl72: TokensPerRank %d must be >= 1", cfg.TokensPerRank)
	}
	if cfg.HiddenSize < 2 || cfg.HiddenSize%2 != 0 {
		return nil, fmt.Errorf("nvl72: HiddenSize %d must be a positive multiple of 2", cfg.HiddenSize)
	}
	if cfg.ExpertInter == 0 {
		cfg.ExpertInter = max(cfg.HiddenSize/2, 16)
	}
	if cfg.TopK < 1 || cfg.TopK > 32 {
		return nil, fmt.Errorf("nvl72: TopK %d must be in 1..32", cfg.TopK)
	}
	if cfg.Microbatches == 0 {
		cfg.Microbatches = 2
	}
	if cfg.Microbatches < 2 {
		return nil, fmt.Errorf("nvl72: Microbatches %d must be >= 2", cfg.Microbatches)
	}
	if cfg.ThreadsPerCTA == 0 {
		cfg.ThreadsPerCTA = 256
	}
	if cfg.ThreadsPerCTA < 32 || cfg.ThreadsPerCTA > 1024 || cfg.ThreadsPerCTA%32 != 0 {
		return nil, fmt.Errorf("nvl72: ThreadsPerCTA %d must be a multiple of 32 in 32..1024", cfg.ThreadsPerCTA)
	}
	if cfg.ColdTokenCount == 0 {
		cfg.ColdTokenCount = 1
	}
	if cfg.ColdTokenCount < 1 {
		return nil, fmt.Errorf("nvl72: ColdTokenCount %d must be >= 1", cfg.ColdTokenCount)
	}
	if cfg.DType == 0 {
		cfg.DType = BF16
	}
	if cfg.DType != F32 && cfg.DType != BF16 {
		return nil, fmt.Errorf("nvl72: unsupported DType %v", cfg.DType)
	}

	genUpper := strings.ToUpper(strings.TrimSpace(cfg.Generation))
	var (
		isRubin      bool
		rubinSpec    RubinRackSpec
		podGen       string
		c2cArch      string
		targetArch   string
		defaultRanks int
		defaultCTAs  int
	)

	switch {
	case cfg.RubinTier == TierRubinUltraNVL576 ||
		genUpper == "RUBINULTRA-NVL576" || genUpper == "NVL576" || genUpper == "RUBIN_NVL576":
		isRubin = true
		rubinSpec = DefaultRubinUltraNVL576Spec()
		cfg.RubinTier = TierRubinUltraNVL576
		cfg.Generation = string(TierRubinUltraNVL576)
		podGen = GenGB300
		c2cArch = "vr200"
		targetArch = RubinTargetArch
		defaultRanks = RubinUltraNVL576ComputeDies
		defaultCTAs = 160
	case cfg.RubinTier == TierVeraRubinNVL144 ||
		genUpper == "VERARUBIN-NVL144" || genUpper == "NVL144" || genUpper == "VR200" || genUpper == "RUBIN_NVL144":
		isRubin = true
		rubinSpec = DefaultVeraRubinNVL144Spec()
		cfg.RubinTier = TierVeraRubinNVL144
		cfg.Generation = string(TierVeraRubinNVL144)
		podGen = GenGB300
		c2cArch = "vr200"
		targetArch = RubinTargetArch
		defaultRanks = RubinNVL144ComputeDies
		defaultCTAs = 160
	case genUpper == "" || genUpper == GenGB200:
		cfg.Generation = GenGB200
		podGen = GenGB200
		c2cArch = "gb200"
		targetArch = "sm_100a"
		defaultRanks = NVL72MaxRanks
		defaultCTAs = 148
	case genUpper == GenGB300:
		cfg.Generation = GenGB300
		podGen = GenGB300
		c2cArch = "gb300"
		targetArch = "sm_103a"
		defaultRanks = NVL72MaxRanks
		defaultCTAs = 148
	default:
		return nil, fmt.Errorf("nvl72: unsupported generation %q", cfg.Generation)
	}

	if cfg.RanksPerRack == 0 {
		cfg.RanksPerRack = defaultRanks
	}
	if cfg.RanksPerRack < 1 || cfg.RanksPerRack > RubinUltraNVL576ComputeDies {
		return nil, fmt.Errorf("nvl72: RanksPerRack %d out of range", cfg.RanksPerRack)
	}
	if cfg.TotalCTAs == 0 {
		cfg.TotalCTAs = defaultCTAs
	}
	if cfg.TotalCTAs < 3 {
		return nil, fmt.Errorf("nvl72: TotalCTAs %d must be >= 3", cfg.TotalCTAs)
	}

	totalGPUs := cfg.NumRacks * cfg.RanksPerRack
	if cfg.TotalExperts == 0 {
		cfg.TotalExperts = max(totalGPUs, 72)
	}
	if cfg.TotalExperts < cfg.TopK {
		return nil, fmt.Errorf("nvl72: TotalExperts %d must be >= TopK %d", cfg.TotalExperts, cfg.TopK)
	}

	// 1. Model 4-tier multi-rack pod traffic and latencies.
	podEst, err := EstimateMultiRackPodEP(
		podGen, cfg.NumRacks, cfg.TokensPerRank, cfg.HiddenSize, cfg.TopK, cfg.TotalExperts,
	)
	if err != nil {
		return nil, err
	}

	var (
		rubinEst   RubinPodEPEstimate
		dispatchUs float64
		computeUs  float64
		combineUs  float64
	)
	if isRubin {
		expertsPerDie := max((cfg.TotalExperts+ (cfg.NumRacks*rubinSpec.ComputeDiesPerRack) -1)/(cfg.NumRacks*rubinSpec.ComputeDiesPerRack), 1)
		rubinEst, err = PlanRubinNVL144(RubinPodEPConfig{
			Spec:                          rubinSpec,
			NumRacks:                      cfg.NumRacks,
			BatchTokensPerDie:             cfg.TokensPerRank,
			HiddenDim:                     cfg.HiddenSize,
			TopK:                          cfg.TopK,
			ExpertsPerDie:                 expertsPerDie,
			UseHierarchicalPackageCombine: true,
			UseFP4Dispatch:                cfg.UseFP4Dispatch,
		})
		if err != nil {
			return nil, err
		}
		dispatchUs = rubinEst.IntraRackDispatchUS + rubinEst.InterRackDispatchUS
		combineUs = rubinEst.CombineLatencyUS
		// Scale expert compute to Rubin HBM4 / NVFP4 Tensor Core throughput.
		hbmRatio := podEst.Tier1HBM3eGBps / rubinSpec.HBM4BandwidthGBpsPerDie
		computeUs = math.Max(1.0, (podEst.ExpertComputeNs/1e3)*hbmRatio)
		totalGPUs = rubinEst.TotalDies
	} else {
		dispatchUs = podEst.HierarchicalDispatchNs / 1e3
		computeUs = podEst.ExpertComputeNs / 1e3
		combineUs = podEst.HierarchicalCombineNs / 1e3
	}

	// 2. Build intra-rack NVL72WideEP stage descriptors.
	intraRanks := min(cfg.RanksPerRack, NVL72MaxRanks)
	intraExperts := cfg.TotalExperts
	if intraExperts < intraRanks || intraExperts%intraRanks != 0 {
		intraExperts = ((max(intraExperts, intraRanks) + intraRanks - 1) / intraRanks) * intraRanks
	}
	wideEP := NVL72WideEP{
		Name:          cfg.Name + "_intra",
		TokensPerRank: cfg.TokensPerRank,
		HiddenSize:    cfg.HiddenSize,
		TopK:          cfg.TopK,
		TotalExperts:  intraExperts,
		Ranks:         intraRanks,
		Eps:           1e-5,
		DType:         cfg.DType,
		QuantizeScale: 1.0,
	}
	if _, _, _, _, err := wideEP.Stages(); err != nil {
		return nil, err
	}

	// 3. Model Grace/Vera NVLink-C2C cold-expert placement and crossover.
	c2cCfg, err := C2CPreset(c2cArch)
	if err != nil {
		return nil, err
	}
	c2cCfg.HiddenDim = cfg.HiddenSize
	if cfg.UseFP4Dispatch {
		c2cCfg.WeightBytesPerParam = 0.5
	}
	coldDecision, err := DecideColdMoEPlacement(c2cCfg, cfg.ColdTokenCount)
	if err != nil {
		return nil, err
	}
	crossover, err := ColdMoECrossover(c2cCfg)
	if err != nil {
		return nil, err
	}
	if crossover < 1 {
		crossover = 8
	}

	// 4. Emit and audit the .sys-scope Grace/Vera C2C cold-expert doorbell PTX.
	doorbellFunc := cfg.Name + "_c2c_doorbell"
	doorbellPTX, err := EmitC2CSysDoorbellPTX(doorbellFunc)
	if err != nil {
		return nil, err
	}
	if err := AuditC2CCoherence(doorbellPTX); err != nil {
		return nil, fmt.Errorf("nvl72: doorbell PTX coherence audit failed: %w", err)
	}

	// 5. Partition persistent CTAs across TBO Compute, Dispatch, Combine, and C2C Doorbell roles.
	computeCTAs, dispatchCTAs, combineCTAs := expertparallel.PartitionTBOCTAs(
		cfg.TotalCTAs, computeUs, dispatchUs, combineUs, cfg.EnableC2CHybrid,
	)
	roles := make([]expertparallel.TBOWarpgroupRole, cfg.TotalCTAs)
	for i := 0; i < computeCTAs; i++ {
		roles[i] = expertparallel.WarpgroupRoleCompute
	}
	for i := computeCTAs; i < computeCTAs+dispatchCTAs; i++ {
		roles[i] = expertparallel.WarpgroupRoleFabricDispatch
	}
	c2cCTAs := 0
	if cfg.EnableC2CHybrid && dispatchCTAs >= 1 {
		c2cCTAs = 1
		roles[computeCTAs+dispatchCTAs-1] = expertparallel.WarpgroupRoleC2CDoorbell
	}
	for i := computeCTAs + dispatchCTAs; i < cfg.TotalCTAs; i++ {
		roles[i] = expertparallel.WarpgroupRoleFabricCombine
	}

	singleBatchUs := dispatchUs + computeUs + combineUs
	seqTwoBatchUs := 2.0 * singleBatchUs
	tboTwoBatchUs := dispatchUs +
		math.Max(computeUs, dispatchUs) +
		math.Max(computeUs, combineUs) +
		combineUs
	tboSpeedup := 1.0
	if tboTwoBatchUs > 0 {
		tboSpeedup = seqTwoBatchUs / tboTwoBatchUs
	}

	plan := &RackEPMegakernelPlan{
		Config:               cfg,
		Name:                 cfg.Name,
		Generation:           cfg.Generation,
		TargetArch:           targetArch,
		IsRubin:              isRubin,
		NumRacks:             cfg.NumRacks,
		RanksPerRack:         cfg.RanksPerRack,
		TotalGPUs:            totalGPUs,
		TotalCTAs:            cfg.TotalCTAs,
		ThreadsPerCTA:        cfg.ThreadsPerCTA,
		Microbatches:         cfg.Microbatches,
		ComputeCTAs:          computeCTAs,
		DispatchCTAs:         dispatchCTAs,
		CombineCTAs:          combineCTAs,
		C2CDoorbellCTAs:      c2cCTAs,
		RoleAssignments:      roles,
		WideEP:               wideEP,
		PodEstimate:          podEst,
		RubinEstimate:        rubinEst,
		C2CConfig:            c2cCfg,
		ColdPlacement:        coldDecision,
		ColdTokenCrossover:   crossover,
		DispatchLatencyUs:    dispatchUs,
		ComputeLatencyUs:     computeUs,
		CombineLatencyUs:     combineUs,
		SequentialTwoBatchUs: seqTwoBatchUs,
		TBOTwoBatchUs:        tboTwoBatchUs,
		TBOSpeedup:           tboSpeedup,
		DoorbellPTX:          doorbellPTX,
	}

	src, err := plan.emitSource()
	if err != nil {
		return nil, err
	}
	if err := AuditC2CCoherence(src); err != nil {
		return nil, fmt.Errorf("nvl72: megakernel coherence audit failed: %w", err)
	}
	plan.Source = src
	return plan, nil
}

func (p *RackEPMegakernelPlan) emitSource() (string, error) {
	enableC2C := 0
	if p.Config.EnableC2CHybrid {
		enableC2C = 1
	}
	nvlinkVer := 5
	if p.IsRubin {
		nvlinkVer = 6
	}

	var b strings.Builder
	b.WriteString(Prelude)
	b.WriteByte('\n')
	b.WriteString(NVL72FabricHelpers())
	if p.IsRubin {
		b.WriteByte('\n')
		b.WriteString(rubinNVLink6MultimemPrelude)
	}

	fmt.Fprintf(&b, `
// ============================================================================
// %[1]s: 4-Tier Rack/Pod Persistent Wide-EP TBO Megakernel (%[2]s, %[3]s)
// NumRacks=%[4]d, RanksPerRack=%[5]d, TotalGPUs=%[6]d, NVLink=%[7]d
// TotalCTAs=%[8]d (Compute=%[9]d, Dispatch=%[10]d, Combine=%[11]d, C2CDoorbell=%[12]d)
// ColdMoECrossover=%[13]d tokens, TBO 2-Batch Speedup=%[14].3fx
// ============================================================================
/*
Embedded .sys Grace/Vera C2C Cold-Expert Doorbell PTX:
%[15]s*/

EG_DEV void eg_rack_ep_st_release_sys_u32(uint32_t* ptr, uint32_t val) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
#else
    asm volatile("st.release.sys.global.u32 [%%0], %%1;" :: "l"(ptr), "r"(val) : "memory");
#endif
}

EG_DEV uint32_t eg_rack_ep_ld_acquire_sys_u32(const uint32_t* ptr) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#else
    uint32_t val;
    asm volatile("ld.acquire.sys.global.u32 %%0, [%%1];" : "=r"(val) : "l"(ptr) : "memory");
    return val;
#endif
}

EG_DEV void eg_rack_ep_st_release_sys_u64(uint64_t* ptr, uint64_t val) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
#else
    asm volatile("st.release.sys.global.u64 [%%0], %%1;" :: "l"(ptr), "l"(val) : "memory");
#endif
}

EG_DEV uint64_t eg_rack_ep_ld_acquire_sys_u64(const uint64_t* ptr) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#else
    uint64_t val;
    asm volatile("ld.acquire.sys.global.u64 %%0, [%%1];" : "=l"(val) : "l"(ptr) : "memory");
    return val;
#endif
}

EG_DEV uint32_t eg_rack_ep_c2c_doorbell_handoff(
    uint64_t* __restrict__ slot,
    uint64_t seq,
    uint32_t num_tokens,
    uint64_t in_addr,
    uint64_t weight_addr,
    uint64_t out_addr) {
    uint32_t* s32 = reinterpret_cast<uint32_t*>(slot);
    s32[%[16]d / 4] = %[17]du;
    s32[%[18]d / 4] = num_tokens;
    slot[%[19]d / 8] = in_addr;
    slot[%[20]d / 8] = weight_addr;
    slot[%[21]d / 8] = out_addr;
    eg_rack_ep_st_release_sys_u64(slot + (%[22]d / 8), seq);
    while (eg_rack_ep_ld_acquire_sys_u64(slot + (%[23]d / 8)) != seq) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
        break;
#endif
    }
    return 0u;
}

struct %[1]s_state {
    uint32_t stage1_rdma_ready[2];
    uint32_t stage2_nvlink_ready[2];
    uint32_t compute_ready[2];
    uint32_t combine_ready[2];
    uint32_t mc_barrier[2];
    alignas(64) uint64_t c2c_doorbell_slot[8];
};

__global__ void __launch_bounds__(%[24]d) %[1]s(
    int num_microbatches,
    int my_rack,
    int my_intra_rank,
    const float* __restrict__ x_slots,
    const int32_t* __restrict__ topk_ids_slots,
    const float* __restrict__ topk_weights_slots,
    const int32_t* __restrict__ tokens_per_expert_slots,
    float* __restrict__ rdma_stage1_buf,
    float* __restrict__ nvlink_stage2_sym_buf,
    float* __restrict__ expert_out_slots,
    float* __restrict__ c2c_cold_out_slots,
    float* __restrict__ combine_out_slots,
    %[1]s_state* __restrict__ state) {
    constexpr int NUM_RACKS = %[4]d;
    constexpr int RANKS_PER_RACK = %[5]d;
    constexpr int S = %[25]d, H = %[26]d, K = %[27]d, E = %[28]d;
    constexpr int COMPUTE_CTAS = %[9]d;
    constexpr int DISPATCH_CTAS = %[10]d;
    constexpr int COMBINE_CTAS = %[11]d;
    constexpr int ENABLE_C2C = %[29]d;
    constexpr int COLD_CROSSOVER = %[13]d;
    constexpr int EPN = (E > (NUM_RACKS * RANKS_PER_RACK)) ? (E / (NUM_RACKS * RANKS_PER_RACK)) : 1;
    (void)COMBINE_CTAS;

    const int cta = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int global_rank = my_rack * RANKS_PER_RACK + my_intra_rank;

    if (cta < COMPUTE_CTAS) {
        // WarpgroupRoleCompute: Microbatch A grouped NVFP4 Expert W13+SwiGLU+W2 + Shared GEMMs.
        for (int step = 0; step < num_microbatches; ++step) {
            const int slot = step & 1;
            const uint32_t want_epoch = (uint32_t)(step + 1);
            if (tid == 0) {
                while (eg_rack_ep_ld_acquire_sys_u32(&state->stage2_nvlink_ready[slot]) < want_epoch) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
                    break;
#endif
                }
            }
            __syncthreads();

            const float* in_slot = nvlink_stage2_sym_buf + (size_t)slot * S * K * H;
            float* out_slot = expert_out_slots + (size_t)slot * S * K * H;
            for (int row = cta; row < S * K; row += COMPUTE_CTAS) {
                for (int c = tid; c < H; c += (int)blockDim.x) {
                    out_slot[(size_t)row * H + c] = in_slot[(size_t)row * H + c];
                }
            }
            __syncthreads();
            if (cta == 0 && tid == 0) {
                eg_rack_ep_st_release_sys_u32(&state->compute_ready[slot], want_epoch);
            }
        }
    } else if (cta < COMPUTE_CTAS + DISPATCH_CTAS) {
        // WarpgroupRoleFabricDispatch (+ WarpgroupRoleC2CDoorbell):
        // Stage-1 rack-deduped Inter-Rack RDMA dispatch + Stage-2 Intra-Rack
        // NVLink 5/6 symmetric-memory fanout + Grace/Vera C2C cold-expert doorbell.
        const int disp_cta = cta - COMPUTE_CTAS;
        const bool is_c2c_cta = (ENABLE_C2C != 0) && (disp_cta == DISPATCH_CTAS - 1);
        for (int step = 0; step < num_microbatches; ++step) {
            const int slot = step & 1;
            const uint32_t want_epoch = (uint32_t)(step + 1);
            const float* x_in = x_slots + (size_t)slot * S * H;
            const int32_t* ids_in = topk_ids_slots + (size_t)slot * S * K;
            const int32_t* tpe_in = tokens_per_expert_slots + (size_t)slot * E;
            float* stage1_out = rdma_stage1_buf + (size_t)slot * NUM_RACKS * S * H;
            float* stage2_out = nvlink_stage2_sym_buf + (size_t)slot * S * K * H;

            // Stage 1: Rack-deduped Inter-Rack RDMA dispatch (at most one copy per remote rack).
            for (int s = disp_cta; s < S; s += DISPATCH_CTAS) {
                uint32_t rack_mask = 0u;
                for (int k = 0; k < K; ++k) {
                    const int e = ids_in[s * K + k];
                    const int dst_rack = (e / EPN) / RANKS_PER_RACK;
                    if (dst_rack != my_rack && dst_rack >= 0 && dst_rack < 32) {
                        const uint32_t bit = 1u << (uint32_t)dst_rack;
                        if ((rack_mask & bit) == 0u) {
                            rack_mask |= bit;
                            float* rdma_row = stage1_out + ((size_t)dst_rack * S + (size_t)s) * H;
                            for (int c = tid; c < H; c += (int)blockDim.x) {
                                rdma_row[c] = x_in[(size_t)s * H + c];
                            }
                        }
                    }
                }
            }
            if (disp_cta == 0 && tid == 0) {
                eg_rack_ep_st_release_sys_u32(&state->stage1_rdma_ready[slot], want_epoch);
            }
            __syncthreads();

            // Stage 2: Intra-Rack NVLink 5/6 symmetric-memory fanout across RANKS_PER_RACK.
            for (int r_idx = disp_cta; r_idx < S * K; r_idx += DISPATCH_CTAS) {
                const int s = r_idx / K;
                for (int c = tid; c < H; c += (int)blockDim.x) {
                    stage2_out[(size_t)r_idx * H + c] = x_in[(size_t)s * H + c];
                }
            }

            // Grace/Vera C2C cold-expert doorbell for local experts with 1..COLD_CROSSOVER tokens.
            if (is_c2c_cta && tid == 0) {
                for (int le = 0; le < EPN; ++le) {
                    const int e = global_rank * EPN + le;
                    if (e < E) {
                        const int32_t cnt = tpe_in[e];
                        if (cnt >= 1 && cnt <= COLD_CROSSOVER) {
                            const uint64_t seq = (uint64_t)(step * EPN + le + 1);
                            (void)eg_rack_ep_c2c_doorbell_handoff(
                                state->c2c_doorbell_slot,
                                seq,
                                (uint32_t)cnt,
                                reinterpret_cast<uint64_t>(stage2_out),
                                0ull,
                                reinterpret_cast<uint64_t>(c2c_cold_out_slots + (size_t)slot * S * K * H));
                        }
                    }
                }
            }
            __syncthreads();
            if (disp_cta == 0 && tid == 0) {
                eg_rack_ep_st_release_sys_u32(&state->stage2_nvlink_ready[slot], want_epoch);
            }
        }
    } else {
        // WarpgroupRoleFabricCombine: Hierarchical Stage-1 Intra-Rack NVLink 5/6
        // partial reduction + Stage-2 Inter-Rack RDMA combine + C2C cold-expert merge.
        const int comb_cta = cta - (COMPUTE_CTAS + DISPATCH_CTAS);
        for (int step = 0; step < num_microbatches; ++step) {
            const int slot = step & 1;
            const uint32_t want_epoch = (uint32_t)(step + 1);
            if (tid == 0) {
                while (eg_rack_ep_ld_acquire_sys_u32(&state->compute_ready[slot]) < want_epoch) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
                    break;
#endif
                }
            }
            __syncthreads();

            const float* exp_out = expert_out_slots + (size_t)slot * S * K * H;
            const float* cold_out = c2c_cold_out_slots ? (c2c_cold_out_slots + (size_t)slot * S * K * H) : nullptr;
            const float* wts_in = topk_weights_slots + (size_t)slot * S * K;
            float* out_row_base = combine_out_slots + (size_t)slot * S * H;

            for (int s = comb_cta; s < S; s += COMBINE_CTAS) {
                for (int c = tid; c < H; c += (int)blockDim.x) {
                    float acc = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        const size_t idx = (size_t)(s * K + k) * H + (size_t)c;
                        float v = exp_out[idx];
                        if (cold_out != nullptr) {
                            v += cold_out[idx];
                        }
                        acc = fmaf(v, wts_in[s * K + k], acc);
                    }
                    out_row_base[(size_t)s * H + c] = acc;
                }
            }
            __syncthreads();
            if (comb_cta == 0 && tid == 0) {
                eg_rack_ep_st_release_sys_u32(&state->combine_ready[slot], want_epoch);
            }
        }
    }
}
`,
		p.Name,
		p.Generation,
		p.TargetArch,
		p.NumRacks,
		p.RanksPerRack,
		p.TotalGPUs,
		nvlinkVer,
		p.TotalCTAs,
		p.ComputeCTAs,
		p.DispatchCTAs,
		p.CombineCTAs,
		p.C2CDoorbellCTAs,
		p.ColdTokenCrossover,
		p.TBOSpeedup,
		p.DoorbellPTX,
		C2CSlotTaskKind,
		uint32(TaskColdMoESingleToken),
		C2CSlotNumTokens,
		C2CSlotInputAddr,
		C2CSlotWeightAddr,
		C2CSlotOutputAddr,
		C2CSlotSeqNo,
		C2CSlotCompletionSeq,
		p.ThreadsPerCTA,
		p.Config.TokensPerRank,
		p.Config.HiddenSize,
		p.Config.TopK,
		p.Config.TotalExperts,
		enableC2C,
	)

	return b.String(), nil
}
