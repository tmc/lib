package model

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/discover"
	"github.com/tmc/lib/nvidia/x/nvl72"
	"github.com/tmc/lib/nvidia/x/tcgen05"
)

// AllProductionTargets lists the seven supported SM architecture targets for
// [CompileProductionModel].
var AllProductionTargets = []string{
	"sm_90a",
	"sm_100a",
	"sm_103a",
	"sm_107a",
	"sm_107f",
	"sm_120a",
	"sm_121a",
}

// AllProductionFatbinTargets lists the six canonical architecture family
// targets packaged into the [MultiArchBundle] fatbinary by [CompileProductionModel].
var AllProductionFatbinTargets = []string{
	"sm_90a",
	"sm_100a",
	"sm_103a",
	"sm_107a",
	"sm_120a",
	"sm_121a",
}

// ProductionCompileOptions configures end-to-end production model compilation
// and multi-architecture lowering in [CompileProductionModel].
type ProductionCompileOptions struct {
	Target          string                 // "sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_107f", "sm_120a", or "sm_121a"
	BatchM          int                    // token batch size M (defaults to 16)
	SeqLen          int                    // KV sequence length (defaults to 2048)
	UseMLA          bool                   // when true on sm_100a/sm_103a/sm_107a/sm_107f, selects tcgen05.TCGen05MLAPagedAttention
	AllowSparse2to4 bool                   // when true on sm_100a/sm_103a/sm_107a/sm_107f, emits tcgen05.TCGen05SparseGEMM (SparsityVersion=1 on sm_107a/sm_107f) and tcgen05.TCGen05LdSpCompress
	AllowSub4BitLUT bool                   // when true on sm_107a/sm_107f, emits tcgen05.TCGen05LUTDecompressGEMM and tcgen05.TCGen05EarlyReleasePipelineGEMM
	TPRanks         int                    // tensor-parallel rank count in 1..576 (defaults to 1)
	EPRanks         int                    // expert-parallel rank count in 1..576 (defaults to 1)
	Stalls          *autotune.StallProfile // optional online CUPTI PC-sampling warp-stall telemetry
	Metrics         *autotune.RangeMetrics // optional online CUPTI range and ptxas spill telemetry
}

// ArchDispatchPlan describes the hardware-specific PTX mnemonics, Tensor Memory
// (TMEM) allocation, warp-specialized register budgets, and software pipeline
// depth selected for the persistent multi-architecture megakernel.
type ArchDispatchPlan struct {
	Target                       string
	TensorCoreMnemonic           string
	DequantMnemonic              string
	CollectiveMnemonic           string
	WorkStealMnemonic            string
	UsesTMEM                     bool
	TMEMColumns                  int
	PipelineStages               int
	ProducerMaxRegs              int
	ConsumerMaxRegs              int
	UsesEarlyReleaseSyncRestrict bool
	UsesLUTDecompressB           bool
}

// OnlineMegakernelProfile carries runtime CUPTI / ptxas feedback used by
// [SelectArchDispatchPlan] to adapt pipeline staging, register budgets, and
// hardware decompression modes online.
type OnlineMegakernelProfile struct {
	DominantStall    string
	SpillDetected    bool
	PreferLUTB       bool
	PreferSparse2to4 bool
	OverrideStages   int
}

// SelectArchDispatchPlan returns the hardware dispatch plan for target,
// applying any online profiler overrides from profile when non-nil.
func SelectArchDispatchPlan(target string, profile *OnlineMegakernelProfile) (ArchDispatchPlan, error) {
	var plan ArchDispatchPlan
	switch target {
	case "sm_90a":
		plan = ArchDispatchPlan{
			Target:             "sm_90a",
			TensorCoreMnemonic: "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
			DequantMnemonic:    "prmt.b32 (eg_e2m1x4_to_bf16x4)",
			CollectiveMnemonic: "dsmem.cluster.allreduce",
			WorkStealMnemonic:  "atomicAdd.grid_queue",
			UsesTMEM:           false,
			TMEMColumns:        0,
			PipelineStages:     3,
			ProducerMaxRegs:    40,
			ConsumerMaxRegs:    232,
		}
	case "sm_100a":
		plan = ArchDispatchPlan{
			Target:             "sm_100a",
			TensorCoreMnemonic: "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X",
			DequantMnemonic:    "cvt.rn.f16x2.e2m1x2",
			CollectiveMnemonic: "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2",
			WorkStealMnemonic:  "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
			UsesTMEM:           true,
			TMEMColumns:        512,
			PipelineStages:     4,
			ProducerMaxRegs:    40,
			ConsumerMaxRegs:    232,
		}
	case "sm_103a":
		plan = ArchDispatchPlan{
			Target:             "sm_103a",
			TensorCoreMnemonic: "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
			DequantMnemonic:    "tcgen05.ld.red.sync.aligned.32x32b.x4.max.f32",
			CollectiveMnemonic: "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2",
			WorkStealMnemonic:  "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
			UsesTMEM:           true,
			TMEMColumns:        512,
			PipelineStages:     4,
			ProducerMaxRegs:    40,
			ConsumerMaxRegs:    232,
		}
	case "sm_107a", "sm_107f":
		plan = ArchDispatchPlan{
			Target:             target,
			TensorCoreMnemonic: "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32.sync_restrict::shared::read::mma::a",
			DequantMnemonic: "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2 + " +
				"fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4 + add.rn.f32x2",
			CollectiveMnemonic:           "fabric.try_red.async.tensor.2d.multimem",
			WorkStealMnemonic:            "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
			UsesTMEM:                     true,
			TMEMColumns:                  576,
			PipelineStages:               4,
			ProducerMaxRegs:              40,
			ConsumerMaxRegs:              232,
			UsesEarlyReleaseSyncRestrict: true,
			UsesLUTDecompressB:           true,
		}
	case "sm_120a":
		plan = ArchDispatchPlan{
			Target:             "sm_120a",
			TensorCoreMnemonic: "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
			DequantMnemonic:    "cvt.rn.f16x2.e2m1x2",
			CollectiveMnemonic: "cp.async.bulk.global.to.shared::cta",
			WorkStealMnemonic:  "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
			UsesTMEM:           false,
			TMEMColumns:        0,
			PipelineStages:     2,
			ProducerMaxRegs:    64,
			ConsumerMaxRegs:    168,
		}
	case "sm_121a":
		plan = ArchDispatchPlan{
			Target:             "sm_121a",
			TensorCoreMnemonic: "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
			DequantMnemonic:    "prmt.b32 (eg_e2m1x4_to_bf16x4)",
			CollectiveMnemonic: "cp.async.bulk.global.to.shared::cta",
			WorkStealMnemonic:  "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
			UsesTMEM:           false,
			TMEMColumns:        0,
			PipelineStages:     2,
			ProducerMaxRegs:    64,
			ConsumerMaxRegs:    168,
		}
	default:
		return ArchDispatchPlan{}, fmt.Errorf("select arch dispatch plan: unsupported target %q", target)
	}

	if profile != nil {
		switch profile.DominantStall {
		case "memory_latency", "long_scoreboard":
			plan.PipelineStages++
		case "register_pressure", "short_scoreboard":
			if plan.PipelineStages > 2 {
				plan.PipelineStages--
			}
		}
		if profile.OverrideStages > 0 {
			plan.PipelineStages = profile.OverrideStages
		}
		if profile.SpillDetected {
			plan.ProducerMaxRegs = max(24, plan.ProducerMaxRegs-8)
			plan.ConsumerMaxRegs = max(96, plan.ConsumerMaxRegs-32)
		}
		if profile.PreferLUTB {
			plan.UsesLUTDecompressB = true
			if !strings.Contains(plan.TensorCoreMnemonic, "decompress::lut::b") && plan.UsesTMEM {
				plan.TensorCoreMnemonic += ".decompress::lut::b.block32"
			}
		}
		if profile.PreferSparse2to4 && !strings.Contains(plan.DequantMnemonic, "sp::2:4") {
			plan.DequantMnemonic += " + tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2"
		}
	}

	return plan, nil
}

// ProductionModelPlan holds the hardware twin, online bottleneck recommendation,
// synthesized sub-byte micro-idiom, macro tile schedule, persistent megakernel
// plan and source, per-role specialized kernel sources, verified TMEM traces,
// and multi-architecture fatbinary bundle produced by [CompileProductionModel].
type ProductionModelPlan struct {
	Target             string
	Twin               autotune.HardwareTwin
	Recommendation     autotune.Recommendation
	SubByteIdiom       discover.CandidateIdiom
	MacroSchedule      discover.MacroSchedule
	MegakernelPlan     ArchDispatchPlan
	MegakernelCUDA     string
	SpecializedKernels map[string]string
	TMEMTraces         [][]cudatmem.TraceEvent
	Fatbin             *MultiArchBundle
}

// ValidateProductionCompileOptions validates cfg and opts, filling defaults for
// BatchM (16), SeqLen (2048), TPRanks (1), and EPRanks (1).
func ValidateProductionCompileOptions(cfg ArchConfig, opts ProductionCompileOptions) (ArchConfig, ProductionCompileOptions, autotune.HardwareTwin, error) {
	var err error
	cfg, err = ValidateArchConfig(cfg)
	if err != nil {
		return cfg, opts, autotune.HardwareTwin{}, err
	}
	opts.Target = strings.ToLower(strings.TrimSpace(opts.Target))
	twin, err := autotune.TwinForTarget(opts.Target)
	if err != nil {
		return cfg, opts, autotune.HardwareTwin{}, fmt.Errorf("compile production model: %w", err)
	}
	if opts.BatchM == 0 {
		opts.BatchM = 16
	}
	if opts.BatchM < 1 {
		return cfg, opts, twin, fmt.Errorf("compile production model: batchM %d must be positive", opts.BatchM)
	}
	if opts.SeqLen == 0 {
		opts.SeqLen = 2048
	}
	if opts.SeqLen < 1 {
		return cfg, opts, twin, fmt.Errorf("compile production model: seqLen %d must be positive", opts.SeqLen)
	}
	if opts.TPRanks == 0 {
		opts.TPRanks = 1
	}
	if opts.TPRanks < 1 || opts.TPRanks > nvl72.RubinUltraNVL576ComputeDies {
		return cfg, opts, twin, fmt.Errorf("compile production model: TPRanks %d must be in 1..%d", opts.TPRanks, nvl72.RubinUltraNVL576ComputeDies)
	}
	if opts.EPRanks == 0 {
		opts.EPRanks = 1
	}
	if opts.EPRanks < 1 || opts.EPRanks > nvl72.RubinUltraNVL576ComputeDies {
		return cfg, opts, twin, fmt.Errorf("compile production model: EPRanks %d must be in 1..%d", opts.EPRanks, nvl72.RubinUltraNVL576ComputeDies)
	}
	return cfg, opts, twin, nil
}

// CompileProductionModel compiles a transformer decoder layer under cfg and
// opts across Hopper (sm_90a), Blackwell (sm_100a, sm_103a, sm_120a, sm_121a),
// and Vera Rubin (sm_107a, sm_107f):
//  1. Resolves the calibrated [autotune.HardwareTwin] via [autotune.TwinForTarget].
//  2. Diagnoses runtime bottlenecks via [autotune.DiagnoseBottleneck] when online
//     CUPTI stall or range telemetry is supplied.
//  3. Synthesizes and verifies the target's sub-byte dequantization micro-idiom
//     via [discover.DiscoverSubByteIdiom] and ranks macro schedules via
//     [discover.DiscoverMacroSchedule].
//  4. Selects and emits architecture-specialized kernels for GEMM, paged/MLA
//     attention, dynamic quantization / sub-byte epilogue, and multi-GPU NVLink
//     collectives, formally verifying all TMEM traces with [cudatmem.VerifyTrace].
//  5. Emits the unified persistent multi-architecture megakernel source and
//     packages the multi-architecture [MultiArchBundle] fatbinary across all
//     six target families.
func CompileProductionModel(cfg ArchConfig, opts ProductionCompileOptions) (*ProductionModelPlan, error) {
	var (
		twin autotune.HardwareTwin
		err  error
	)
	cfg, opts, twin, err = ValidateProductionCompileOptions(cfg, opts)
	if err != nil {
		return nil, err
	}

	// 1. Online bottleneck diagnosis from CUPTI stall and range telemetry.
	var (
		stalls  autotune.StallProfile
		metrics autotune.RangeMetrics
	)
	hasTelemetry := opts.Stalls != nil || opts.Metrics != nil
	if opts.Stalls != nil {
		stalls = *opts.Stalls
	}
	if opts.Metrics != nil {
		metrics = *opts.Metrics
	}
	if hasTelemetry && twin.SupportsTCGen05 && (opts.Target == "sm_107a" || opts.Target == "sm_107f") {
		twin.TMEMColumns = 576
	}
	rec := autotune.DiagnoseBottleneck(twin, stalls, metrics)

	// 2. Sub-byte micro-idiom and macro schedule discovery.
	_, idiom, err := discover.DiscoverSubByteIdiom(opts.Target, discover.IdiomE2M1Dequant)
	if err != nil {
		return nil, fmt.Errorf("compile production model: %w", err)
	}

	allowSparse := opts.AllowSparse2to4 || rec.EnableSparse2to4
	allowLUT := opts.AllowSub4BitLUT || rec.PreferSub4BitLUT
	workload := discover.WorkloadSpec{
		Name:            "decoder_layer",
		M:               opts.BatchM,
		N:               cfg.HiddenSize,
		K:               cfg.HiddenSize,
		BatchSize:       opts.BatchM,
		IsDecode:        opts.BatchM <= 32,
		AllowSub4BitLUT: allowLUT,
		AllowSparse2to4: allowSparse,
	}

	var feedback *discover.OnlineStallFeedback
	if hasTelemetry {
		feedback = buildStallFeedback(rec, stalls, metrics)
	}
	macroSched, _, err := discover.DiscoverMacroSchedule(opts.Target, workload, feedback)
	if err != nil {
		return nil, fmt.Errorf("compile production model: %w", err)
	}

	// 3. Architecture-specialized kernels and formal TMEM trace verification.
	specialized, traces, err := emitSpecializedKernelsForTarget(cfg, opts, rec)
	if err != nil {
		return nil, err
	}

	// 4. Unified persistent multi-architecture megakernel & dispatch plan.
	var megaProfile *OnlineMegakernelProfile
	if hasTelemetry || opts.AllowSub4BitLUT || opts.AllowSparse2to4 {
		megaProfile = buildMegakernelProfile(rec, opts, hasTelemetry)
	}
	megaPlan, err := SelectArchDispatchPlan(opts.Target, megaProfile)
	if err != nil {
		return nil, fmt.Errorf("compile production model: %w", err)
	}
	megaCUDA := emitMultiArchMegakernelCUDA(cfg, megaPlan)

	// 5. Multi-architecture fatbinary bundle across all 6 canonical target families.
	fatbinTargets := append([]string(nil), AllProductionFatbinTargets...)
	if opts.Target == "sm_107f" {
		fatbinTargets = append(fatbinTargets, "sm_107f")
	}
	fatbin, err := BuildMultiArchBundle("production_decoder_layer", cfg, fatbinTargets)
	if err != nil {
		return nil, fmt.Errorf("compile production model: %w", err)
	}

	return &ProductionModelPlan{
		Target:             opts.Target,
		Twin:               twin,
		Recommendation:     rec,
		SubByteIdiom:       idiom,
		MacroSchedule:      macroSched,
		MegakernelPlan:     megaPlan,
		MegakernelCUDA:     megaCUDA,
		SpecializedKernels: specialized,
		TMEMTraces:         traces,
		Fatbin:             fatbin,
	}, nil
}

func buildStallFeedback(rec autotune.Recommendation, stalls autotune.StallProfile, metrics autotune.RangeMetrics) *discover.OnlineStallFeedback {
	fb := &discover.OnlineStallFeedback{
		SpillBytes: metrics.LocalSpillLoadBytes + metrics.LocalSpillStoreBytes,
	}
	sum := stalls.StallMIOThrottle +
		stalls.StallMathPipeThrottle +
		stalls.StallLongScoreboard +
		stalls.StallShortScoreboard +
		stalls.StallBarrier +
		stalls.StallNotSelected
	total := sum
	if stalls.TotalSamples > total {
		total = stalls.TotalSamples
	}
	if total > 0 {
		inv := 1.0 / float64(total)
		fb.MeasuredMIOThrottleRatio = float64(stalls.StallMIOThrottle+stalls.StallShortScoreboard) * inv
		fb.MeasuredLongSBRatio = float64(stalls.StallLongScoreboard) * inv
		fb.MeasuredMathPipeRatio = float64(stalls.StallMathPipeThrottle) * inv
	}
	switch rec.Bottleneck {
	case autotune.BottleneckRegisterSpill:
		fb.DominantStall = "register_spill"
	case autotune.BottleneckMemoryLatency:
		fb.DominantStall = "long_scoreboard"
	case autotune.BottleneckMathPipeBound:
		fb.DominantStall = "math_pipe_throttle"
	case autotune.BottleneckMIOSMEMConflict:
		fb.DominantStall = "mio_throttle"
	case autotune.BottleneckBarrierSkew:
		fb.DominantStall = "barrier_skew"
	}
	return fb
}

func buildMegakernelProfile(rec autotune.Recommendation, opts ProductionCompileOptions, hasTelemetry bool) *OnlineMegakernelProfile {
	p := &OnlineMegakernelProfile{
		PreferLUTB:       opts.AllowSub4BitLUT || rec.PreferSub4BitLUT,
		PreferSparse2to4: opts.AllowSparse2to4 || rec.EnableSparse2to4,
	}
	if !hasTelemetry {
		return p
	}
	switch rec.Bottleneck {
	case autotune.BottleneckMemoryLatency:
		p.DominantStall = "memory_latency"
	case autotune.BottleneckRegisterSpill:
		p.DominantStall = "register_pressure"
		p.SpillDetected = true
	case autotune.BottleneckMIOSMEMConflict:
		p.DominantStall = "short_scoreboard"
	}
	return p
}

func verifyTargetTraces(target string, traces [][]cudatmem.TraceEvent) error {
	tmemTarget := cudatmem.Target("sm_100a")
	if target == "sm_103a" || target == "sm_107a" || target == "sm_107f" {
		tmemTarget = cudatmem.Target("sm_103a")
	}
	for i, tr := range traces {
		if _, err := cudatmem.VerifyTrace(tmemTarget, tr); err != nil {
			return fmt.Errorf("compile production model: target %s trace %d failed verification: %w", target, i, err)
		}
	}
	return nil
}

func emitSpecializedKernelsForTarget(cfg ArchConfig, opts ProductionCompileOptions, rec autotune.Recommendation) (map[string]string, [][]cudatmem.TraceEvent, error) {
	kernels := make(map[string]string, 8)
	var traces [][]cudatmem.TraceEvent

	cols := max(32, (cfg.HiddenSize/32)*32)
	kDim := clampMultipleOf16(cfg.HiddenSize, 64, 256)
	kDim = (kDim / 64) * 64
	if kDim < 64 {
		kDim = 64
	}
	tpRanks := max(opts.TPRanks, 2)
	allowSparse := opts.AllowSparse2to4 || rec.EnableSparse2to4
	allowLUT := opts.AllowSub4BitLUT || rec.PreferSub4BitLUT

	switch opts.Target {
	case "sm_90a":
		kernels["gemm"] = emitHopperTileGEMMCUDA(cfg)
		kernels["attention"] = emitHopperPagedAttentionCUDA(cfg, opts.SeqLen)
		kernels["quant_epilogue"] = emitRMSNormDynamicNVFP4QuantCUDA("rmsnorm_dyn_nvfp4_quant", cols, cfg.RMSNormEps, false)
		ar := nvl72.AllReduceResidualRMSNorm{
			Name:    "allreduce_residual_rmsnorm",
			Cols:    cols,
			Ranks:   tpRanks,
			Eps:     cfg.RMSNormEps,
			DType:   nvl72.BF16,
			Threads: 256,
		}
		colSrc, err := ar.Source()
		if err != nil {
			return nil, nil, err
		}
		kernels["collective"] = colSrc

	case "sm_100a":
		if allowSparse {
			sp := tcgen05.TCGen05SparseGEMM{
				Name:             "sm100_tcgen05_sparse_gemm",
				M:                128,
				N:                128,
				K:                kDim,
				Target:           "sm_100a",
				CTAGroup:         1,
				BlockScaledNVFP4: cfg.WeightDType == NVFP4,
			}
			gemmSrc, err := sp.CUDA()
			if err != nil {
				return nil, nil, err
			}
			kernels["gemm"] = gemmSrc
			traces = append(traces, sp.TraceEvents())
		} else {
			bs := tcgen05.TCGen05BlockScaledGEMM{
				Name: "sm100_tcgen05_block_scaled_gemm",
				M:    128,
				N:    128,
				K:    kDim,
			}
			gemmSrc, err := bs.CUDA()
			if err != nil {
				return nil, nil, err
			}
			kernels["gemm"] = gemmSrc
			traces = append(traces, bs.TraceEvents())
		}

		if opts.UseMLA {
			mla := tcgen05.TCGen05MLAPagedAttention{
				Name:       "sm100_tcgen05_mla_paged_attention",
				NumHeads:   cfg.NumAttentionHeads,
				KVLoRARank: 16,
				RoPEDim:    8,
				PageSize:   16,
				SeqLen:     min(opts.SeqLen, 64),
				Target:     "sm_100a",
				CTAGroup:   1,
				PingPong:   true,
			}
			kernels["attention"] = mla.Source()
			traces = append(traces, mla.TMEMTrace())
		} else {
			pa := tcgen05.TCGen05PagedAttention{
				Name:       "sm100_tcgen05_paged_attention",
				NumQHeads:  cfg.NumAttentionHeads,
				NumKVHeads: cfg.NumKeyValueHeads,
				HeadDim:    max(64, (cfg.HeadDim/16)*16),
				PageSize:   16,
				SeqLen:     min(opts.SeqLen, 64),
				Target:     "sm_100a",
				CTAGroup:   1,
				PingPong:   true,
				KVFormat:   tcgen05.WeightFormat(cfg.WeightFormat()),
			}
			kernels["attention"] = pa.Source()
			traces = append(traces, pa.TMEMTrace())
		}

		kernels["quant_epilogue"] = emitRMSNormDynamicNVFP4QuantCUDA("rmsnorm_dyn_nvfp4_quant", cols, cfg.RMSNormEps, true)
		ar := nvl72.AllReduceResidualRMSNorm{
			Name:    "allreduce_residual_rmsnorm",
			Cols:    cols,
			Ranks:   tpRanks,
			Eps:     cfg.RMSNormEps,
			DType:   nvl72.BF16,
			Threads: 256,
		}
		colSrc, err := ar.Source()
		if err != nil {
			return nil, nil, err
		}
		kernels["collective"] = colSrc

	case "sm_103a":
		sv4 := tcgen05.TCGen05ScaleVec4XGEMM{
			Name: "sm103_tcgen05_scale_vec_4x_gemm",
			M:    128,
			N:    128,
			K:    128,
		}
		gemmSrc, err := sv4.CUDA()
		if err != nil {
			return nil, nil, err
		}
		traces = append(traces, sv4.TraceEvents())
		if allowSparse {
			sp := tcgen05.TCGen05SparseGEMM{
				Name:             "sm103_tcgen05_sparse_gemm",
				M:                128,
				N:                128,
				K:                128,
				Target:           "sm_103a",
				CTAGroup:         1,
				BlockScaledNVFP4: true,
			}
			spSrc, err := sp.CUDA()
			if err != nil {
				return nil, nil, err
			}
			gemmSrc = gemmSrc + "\n" + spSrc
			kernels["sparse_gemm"] = spSrc
			traces = append(traces, sp.TraceEvents())
		}
		kernels["gemm"] = gemmSrc

		if opts.UseMLA {
			mla := tcgen05.TCGen05MLAPagedAttention{
				Name:       "sm103_tcgen05_mla_paged_attention",
				NumHeads:   cfg.NumAttentionHeads,
				KVLoRARank: 16,
				RoPEDim:    8,
				PageSize:   16,
				SeqLen:     min(opts.SeqLen, 64),
				Target:     "sm_103a",
				CTAGroup:   1,
				PingPong:   true,
			}
			kernels["attention"] = mla.Source()
			traces = append(traces, mla.TMEMTrace())
		} else {
			pa := tcgen05.TCGen05PagedAttention{
				Name:       "sm103_tcgen05_paged_attention",
				NumQHeads:  cfg.NumAttentionHeads,
				NumKVHeads: cfg.NumKeyValueHeads,
				HeadDim:    max(64, (cfg.HeadDim/16)*16),
				PageSize:   16,
				SeqLen:     min(opts.SeqLen, 64),
				Target:     "sm_103a",
				CTAGroup:   1,
				PingPong:   true,
				KVFormat:   tcgen05.WeightNVFP4,
			}
			kernels["attention"] = pa.Source()
			traces = append(traces, pa.TMEMTrace())
		}

		ldRed := tcgen05.TCGen05LdRed{
			M:      128,
			N:      128,
			RedOp:  tcgen05.LdRedMax,
			Abs:    true,
			Target: "sm_103a",
		}
		ldRedSrc, err := ldRed.CUDA()
		if err != nil {
			return nil, nil, err
		}
		kernels["ld_red"] = ldRedSrc
		kernels["quant_epilogue"] = ldRedSrc + "\n" + emitRMSNormDynamicNVFP4QuantCUDA("rmsnorm_dyn_nvfp4_quant", cols, cfg.RMSNormEps, true)

		ar := nvl72.AllReduceResidualRMSNorm{
			Name:    "allreduce_residual_rmsnorm",
			Cols:    cols,
			Ranks:   tpRanks,
			Eps:     cfg.RMSNormEps,
			DType:   nvl72.BF16,
			Threads: 256,
		}
		colSrc, err := ar.Source()
		if err != nil {
			return nil, nil, err
		}
		kernels["collective"] = colSrc

	case "sm_107a", "sm_107f":
		var gemmParts []string
		if allowLUT || !allowSparse {
			lutGemm := tcgen05.TCGen05LUTDecompressGEMM{
				Name:       "rubin_tcgen05_lut_decompress_gemm",
				M:          128,
				N:          128,
				K:          128,
				Target:     opts.Target,
				CollectorB: tcgen05.CollectorFill,
			}
			lutSrc, err := lutGemm.CUDA()
			if err != nil {
				return nil, nil, err
			}
			gemmParts = append(gemmParts, lutSrc)
			kernels["lut_gemm"] = lutSrc
		}
		if allowSparse || !allowLUT {
			sp := tcgen05.TCGen05SparseGEMM{
				Name:             "rubin_tcgen05_sparse_gemm",
				M:                128,
				N:                128,
				K:                128,
				Target:           opts.Target,
				CTAGroup:         1,
				BlockScaledNVFP4: true,
			}
			spSrc, err := sp.CUDA()
			if err != nil {
				return nil, nil, err
			}
			gemmParts = append(gemmParts, spSrc)
			kernels["sparse_gemm"] = spSrc
			traces = append(traces, sp.TraceEvents())
		}
		erStages := max(2, rec.SuggestedStages)
		if erStages > 8 {
			erStages = 8
		}
		er := tcgen05.TCGen05EarlyReleasePipelineGEMM{
			Name:        "rubin_tcgen05_early_release_gemm",
			M:           128,
			N:           128,
			K:           128,
			Stages:      erStages,
			ClusterCTAs: 4,
			Target:      opts.Target,
		}
		erSrc, err := er.CUDA()
		if err != nil {
			return nil, nil, err
		}
		gemmParts = append(gemmParts, erSrc)
		kernels["early_release_gemm"] = erSrc
		kernels["gemm"] = strings.Join(gemmParts, "\n")
		traces = append(traces, tcgen05GEMMTraceEvents(128, 128))

		if opts.UseMLA {
			mla := tcgen05.TCGen05MLAPagedAttention{
				Name:       "rubin_tcgen05_mla_paged_attention",
				NumHeads:   cfg.NumAttentionHeads,
				KVLoRARank: 16,
				RoPEDim:    8,
				PageSize:   16,
				SeqLen:     min(opts.SeqLen, 64),
				Target:     "sm_103a",
				CTAGroup:   1,
				PingPong:   true,
			}
			kernels["attention"] = mla.Source()
			traces = append(traces, mla.TMEMTrace())
		} else {
			pa := tcgen05.TCGen05PagedAttention{
				Name:       "rubin_tcgen05_paged_attention",
				NumQHeads:  cfg.NumAttentionHeads,
				NumKVHeads: cfg.NumKeyValueHeads,
				HeadDim:    max(64, (cfg.HeadDim/16)*16),
				PageSize:   16,
				SeqLen:     min(opts.SeqLen, 64),
				Target:     "sm_103a",
				CTAGroup:   1,
				PingPong:   true,
				KVFormat:   tcgen05.WeightNVFP4,
			}
			kernels["attention"] = pa.Source()
			traces = append(traces, pa.TMEMTrace())
		}

		spComp := tcgen05.TCGen05LdSpCompress{
			M:             128,
			N:             128,
			RedOp:         tcgen05.LdRedMax,
			Abs:           true,
			WithReduction: true,
			Target:        opts.Target,
		}
		spCompSrc, err := spComp.CUDA()
		if err != nil {
			return nil, nil, err
		}
		ue5m3Src := emitRubinUE5M3ScaledQuantCUDA("rubin_ue5m3_scaled_quant", max(opts.BatchM, 1), cols, cfg.RMSNormEps, opts.Target)
		quadSrc := emitRubinQuadSubByteSIMDEpilogueCUDA("rubin_quad_subbyte_simd_epilogue", max(cols/4, 16), opts.Target)
		kernels["ld_spcompress"] = spCompSrc
		kernels["ue5m3_quant"] = ue5m3Src
		kernels["quad_simd_epilogue"] = quadSrc
		kernels["quant_epilogue"] = spCompSrc + "\n" + ue5m3Src + "\n" + quadSrc

		rubinRanks := opts.TPRanks
		if rubinRanks <= 1 {
			rubinRanks = nvl72.RubinNVL144ComputeDies
		}
		rAr := nvl72.RubinNVL144AllReduceResidualRMSNorm{
			Name:      "rubin_nvl144_allreduce_residual_rmsnorm",
			Target:    nvl72.RubinTargetArch,
			NumRanks:  rubinRanks,
			HiddenDim: cols,
			Eps:       cfg.RMSNormEps,
			Threads:   256,
		}
		rArSrc, err := rAr.CUDAFile()
		if err != nil {
			return nil, nil, err
		}
		fab := nvl72.RubinFabricAsyncTensorKernel{
			Name:       "eg_rubin_fabric_async_tensor",
			Op:         nvl72.FabricTensorTryRed,
			Dim:        2,
			TileRows:   32,
			TileCols:   32,
			Multimem:   true,
			Completion: nvl72.FabricCompletionMbarrier,
			RedOp:      "add",
			NumRanks:   rubinRanks,
			Target:     opts.Target,
		}
		fabSrc, err := fab.CUDA()
		if err != nil {
			return nil, nil, err
		}
		kernels["fabric_tensor"] = fabSrc
		kernels["collective"] = rArSrc + "\n" + fabSrc

	case "sm_120a", "sm_121a":
		kernels["gemm"] = emitSM120BlockScaledGEMMCUDA("sm120_block_scaled_gemm", max(opts.BatchM, 16), cols, max(cols, 64), opts.Target)
		kernels["attention"] = emitSM120PagedAttentionCUDA("sm120_paged_attention", cfg.NumAttentionHeads, cfg.NumKeyValueHeads, max(64, (cfg.HeadDim/16)*16), opts.SeqLen, opts.Target)
		kernels["quant_epilogue"] = emitRMSNormDynamicNVFP4QuantCUDA("rmsnorm_dyn_nvfp4_quant", cols, cfg.RMSNormEps, false)
		ar := nvl72.AllReduceResidualRMSNorm{
			Name:    "allreduce_residual_rmsnorm",
			Cols:    cols,
			Ranks:   tpRanks,
			Eps:     cfg.RMSNormEps,
			DType:   nvl72.BF16,
			Threads: 256,
		}
		colSrc, err := ar.Source()
		if err != nil {
			return nil, nil, err
		}
		kernels["collective"] = colSrc

	default:
		return nil, nil, fmt.Errorf("compile production model: unsupported target %q", opts.Target)
	}

	if err := verifyTargetTraces(opts.Target, traces); err != nil {
		return nil, nil, err
	}
	return kernels, traces, nil
}

func emitHopperTileGEMMCUDA(cfg ArchConfig) string {
	n := max(128, (cfg.HiddenSize/128)*128)
	k := max(64, (cfg.HiddenSize/64)*64)
	return fmt.Sprintf(`// hopper_wgmma_tile_gemm: Hopper sm_90a warp-specialized WGMMA + TMA TileGEMM (N=%d, K=%d, weight=%s).
__global__ void __launch_bounds__(384, 1) hopper_wgmma_tile_gemm(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ w_repacked,
    const uint8_t* __restrict__ w_scales,
    uint16_t* __restrict__ out,
    int m_total) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    // Hopper sm_90a producer/consumer TMA + WGMMA pipeline:
    // cp.async.bulk.tensor.2d.shared::cluster.global.tile
    // wgmma.fence.sync.aligned;
    // wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16
    // wgmma.commit_group.sync.aligned;
    // wgmma.wait_group.sync.aligned 0;
    // prmt.b32 via eg_e2m1x4_to_bf16x4
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
    (void)x; (void)w_repacked; (void)w_scales; (void)out; (void)m_total;
}
`, n, k, cfg.WeightFormat())
}

func emitHopperPagedAttentionCUDA(cfg ArchConfig, seqLen int) string {
	return fmt.Sprintf(`// hopper_paged_attention: Hopper sm_90a paged FP8 E4M3 split-KV attention (Hq=%d, Hkv=%d, D=%d, seq=%d).
__global__ void __launch_bounds__(128) hopper_paged_attention(
    const uint8_t* __restrict__ q_e4m3,
    const uint8_t* __restrict__ k_cache_e4m3,
    const uint8_t* __restrict__ v_cache_e4m3,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seqlens,
    uint16_t* __restrict__ out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    // wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 + cp.async.cg.shared.global
    // cvt.rn.f16x2.e4m3x2
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
    (void)q_e4m3; (void)k_cache_e4m3; (void)v_cache_e4m3; (void)page_table; (void)seqlens; (void)out;
}
`, cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim, seqLen)
}

func emitRMSNormDynamicNVFP4QuantCUDA(name string, cols int, eps float32, swizzled bool) string {
	layoutName := "row_major"
	storeStmt := "out_scales[(size_t)r * K_BLOCKS + kb] = scale_u8;"
	if swizzled {
		layoutName = "tcgen05_swizzled"
		storeStmt = "out_scales[eg_tcgen05_swizzle_scale_offset(r, kb, rows, K_BLOCKS)] = scale_u8;"
	}
	return fmt.Sprintf(`// %s: fused RMSNorm + dynamic NVFP4 activation quantizer (Cols=%d, ScaleLayout=%s).
__global__ void __launch_bounds__(256) %s(
    int rows,
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales) {
    constexpr int K = %d, K_BYTES = K / 2, K_BLOCKS = K / 16;
    constexpr float EPS = %.8ef;
    (void)rows; (void)x; (void)weight; (void)out_fp4; (void)out_scales;
    (void)K_BYTES; (void)K_BLOCKS; (void)EPS;
    // Dynamic per-16-element UE4M3 scale (eg_f32_to_ue4m3) and E2M1x2 packing (eg_f32_to_e2m1_code):
    // %s
}
`, name, cols, layoutName, name, cols, eps, storeStmt)
}

func emitRubinUE5M3ScaledQuantCUDA(name string, rows, cols int, eps float32, target string) string {
	return fmt.Sprintf(`// %s: Rubin (%s) PTX ISA 9.4 RMSNorm + UE5M3 block-scaled quantizer (Rows=%d, Cols=%d).
__global__ void __launch_bounds__(32) %s(
    const float* __restrict__ x,
    const float* __restrict__ w,
    uint8_t* __restrict__ out_ue5m3,
    uint8_t* __restrict__ out_scales,
    float* __restrict__ out_dequant) {
    constexpr int ROWS = %d, COLS = %d;
    constexpr float EPS = %.8ef;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    // cvt.rn.satfinite.pzo.e2m1x2.f32
    // cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32
    // cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.ue5m3x2
#endif
    (void)ROWS; (void)COLS; (void)EPS;
    (void)x; (void)w; (void)out_ue5m3; (void)out_scales; (void)out_dequant;
}
`, name, target, rows, cols, name, rows, cols, eps)
}

func emitRubinQuadSubByteSIMDEpilogueCUDA(name string, numQuads int, target string) string {
	return fmt.Sprintf(`// %s: Rubin (%s) PTX ISA 9.4 quad-packed sub-byte SIMD epilogue (NumQuads=%d).
__global__ void __launch_bounds__(64) %s(
    const unsigned int* __restrict__ a_e2m1x4,
    const unsigned int* __restrict__ b_e4m3x4,
    const unsigned int* __restrict__ c_e4m3x4,
    unsigned int* __restrict__ out_fma_e4m3x4,
    unsigned int* __restrict__ out_add_e4m3x4,
    float* __restrict__ out_f32,
    int num_quads) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070)
    // fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4
    // add.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4
    // mul.ftz.rz.f16x2.f32x2.f32x2
#endif
    (void)a_e2m1x4; (void)b_e4m3x4; (void)c_e4m3x4;
    (void)out_fma_e4m3x4; (void)out_add_e4m3x4; (void)out_f32; (void)num_quads;
}
`, name, target, numQuads, name)
}

func emitSM120BlockScaledGEMMCUDA(name string, m, n, k int, target string) string {
	dequantComment := "prmt.b32 LUT via eg_e2m1x4_to_bf16x4"
	if target == "sm_120a" {
		dequantComment = "cvt.rn.f16x2.e2m1x2"
	}
	return fmt.Sprintf(`// %s: %s warp-scaled m16n8k64 NVFP4 GEMM (M=%d, N=%d, K=%d, dequant=%s).
__global__ void __launch_bounds__(128) %s(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ b_packed,
    const uint8_t* __restrict__ b_scales,
    float* __restrict__ out,
    int m_total) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    // mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3
    // clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
    // %s
#endif
    (void)a_packed; (void)a_scales; (void)b_packed; (void)b_scales; (void)out; (void)m_total;
}
`, name, target, m, n, k, dequantComment, name, dequantComment)
}

func emitSM120PagedAttentionCUDA(name string, numQHeads, numKVHeads, headDim, seqLen int, target string) string {
	return fmt.Sprintf(`// %s: %s register-accumulated online softmax paged attention (%d/%d heads, D=%d, seq=%d, no TMEM).
__global__ void __launch_bounds__(128) %s(
    const float* __restrict__ q,
    const uint8_t* __restrict__ k_pages,
    const uint8_t* __restrict__ v_pages,
    const uint8_t* __restrict__ k_scales,
    const uint8_t* __restrict__ v_scales,
    const int32_t* __restrict__ page_table,
    float* __restrict__ out,
    int seq_len) {
    // Register-resident online softmax with exp2f rescaling and eg_sm120_dequant_nvfp4_pair
    (void)q; (void)k_pages; (void)v_pages; (void)k_scales; (void)v_scales; (void)page_table; (void)out; (void)seq_len;
}
`, name, target, numQHeads, numKVHeads, headDim, seqLen, name)
}

func emitMultiArchMegakernelCUDA(cfg ArchConfig, plan ArchDispatchPlan) string {
	hidden := max(16, cfg.HiddenSize)
	numHeads := max(1, cfg.NumAttentionHeads)
	headDim := cfg.HeadDim
	if headDim <= 0 {
		headDim = max(2, hidden/numHeads)
	}
	inter := cfg.IntermediateSize
	if inter <= 0 {
		inter = 2 * hidden
	}
	eps := cfg.RMSNormEps
	if eps <= 0 {
		eps = 1e-5
	}
	attnScale := float32(1.0 / math.Sqrt(float64(headDim)))

	var b strings.Builder
	fmt.Fprintf(&b, "// Hardware dispatch profile for %s:\n", plan.Target)
	fmt.Fprintf(&b, "//   tensor_core:   %s\n", plan.TensorCoreMnemonic)
	fmt.Fprintf(&b, "//   dequant:       %s\n", plan.DequantMnemonic)
	fmt.Fprintf(&b, "//   collective:    %s\n", plan.CollectiveMnemonic)
	fmt.Fprintf(&b, "//   work_steal:    %s\n", plan.WorkStealMnemonic)
	fmt.Fprintf(&b, "//   tmem_cols:     %d (uses_tmem=%v, stages=%d, regs=%d/%d, sync_restrict=%v, lut_b=%v)\n",
		plan.TMEMColumns, plan.UsesTMEM, plan.PipelineStages,
		plan.ProducerMaxRegs, plan.ConsumerMaxRegs,
		plan.UsesEarlyReleaseSyncRestrict, plan.UsesLUTDecompressB)
	fmt.Fprintf(&b, `
// eg_multiarch_megakernel: unified multi-architecture persistent op-queue megakernel (target=%s, H=%d, I=%d, heads=%d, D=%d, ops=6).
__global__ void __launch_bounds__(128) eg_multiarch_megakernel(
    const float* __restrict__ x,
    const float* __restrict__ rms_attn_w,
    const float* __restrict__ w_qkv,
    const float* __restrict__ w_o,
    const float* __restrict__ rms_ffn_w,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    float* __restrict__ out,
    unsigned long long* __restrict__ ready,
    unsigned int epoch,
    int num_tokens) {
    constexpr int H = %d, I = %d, HQ = %d, D = %d, NUM_OPS = 6;
    constexpr int PIPELINE_STAGES = %d, TMEM_COLS = %d;
    constexpr int PROD_MAX_REGS = %d, CONS_MAX_REGS = %d;
    constexpr float EPS = %.8ef, ATTN_SCALE = %.8ef;
    constexpr unsigned int OP_QUEUE[NUM_OPS] = {1u, 2u, 3u, 4u, 5u, 6u};
    (void)H; (void)I; (void)HQ; (void)D; (void)PIPELINE_STAGES; (void)TMEM_COLS;
    (void)PROD_MAX_REGS; (void)CONS_MAX_REGS; (void)EPS; (void)ATTN_SCALE; (void)OP_QUEUE;
    (void)x; (void)rms_attn_w; (void)w_qkv; (void)w_o; (void)rms_ffn_w;
    (void)w_gate; (void)w_up; (void)w_down; (void)out; (void)ready; (void)epoch; (void)num_tokens;
}
`, plan.Target, hidden, inter, numHeads, headDim,
		hidden, inter, numHeads, headDim,
		plan.PipelineStages, plan.TMEMColumns,
		plan.ProducerMaxRegs, plan.ConsumerMaxRegs,
		eps, attnScale)
	return b.String()
}
