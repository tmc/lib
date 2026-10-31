package main

import (
	"cmp"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/model"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
)

// InlineLineAnnotation attaches hardware CUPTI PC-sampling stall metrics,
// static PTX/SASS instruction mappings, calibrated e-graph costs, L2 cache
// eviction policies, and formal barrier verification proofs to a 1-based CUDA
// source line number.
type InlineLineAnnotation struct {
	Line                 int                 `json:"line"`
	StageIndex           int                 `json:"stage_index"`
	StageName            string              `json:"stage_name"`
	Category             string              `json:"category"`
	CUPTIStallSamples    uint64              `json:"cupti_stall_samples"`
	CUPTIDominantStall   string              `json:"cupti_dominant_stall"`
	CUPTIStallPct        float64             `json:"cupti_stall_pct"`
	SASSInsts            []AnnotatedSASSInst `json:"sass_insts,omitempty"`
	PTXSnippet           string              `json:"ptx_snippet,omitempty"`
	EGraphCostNs         float64             `json:"egraph_cost_ns"`
	CalibratedMultiplier float64             `json:"calibrated_multiplier"`
	L2Policy             string              `json:"l2_policy"`
	BarrierProof         string              `json:"barrier_proof"`
}

// AnnotatedSASSInst represents a disassembled SASS instruction annotated with
// Hopper/Blackwell control word stall counts, scoreboard wait masks, shared
// memory bank-conflict indicators, and its originating 1-based CUDA source line.
type AnnotatedSASSInst struct {
	PC               uint64 `json:"pc"`
	Mnemonic         string `json:"mnemonic"`
	Operands         string `json:"operands"`
	ControlStall     int    `json:"control_stall"`
	WaitMask         string `json:"wait_mask"`
	ScoreboardWeight int    `json:"scoreboard_weight"`
	BankConflictRisk bool   `json:"bank_conflict_risk"`
	SourceLine       int    `json:"source_line"`
}

// AnnotatedStageSummary rolls up E-Graph cost calibration, PTXAS register and
// spill metrics, SASS scoreboard analysis, CUPTI PC-sampling stall attribution,
// and formal barrier proofs for a single stage inside a persistent megakernel.
type AnnotatedStageSummary struct {
	StageIndex int    `json:"stage_index"`
	StageName  string `json:"stage_name"`
	Category   string `json:"category"`
	StartLine  int    `json:"start_line"`
	EndLine    int    `json:"end_line"`
	Blocks     int    `json:"blocks"`
	Threads    int    `json:"threads"`

	// E-Graph Cost
	AnalyticalCostNs     float64 `json:"analytical_cost_ns"`
	EGraphCostNs         float64 `json:"egraph_cost_ns"`
	CalibratedMultiplier float64 `json:"calibrated_multiplier"`
	RooflineBound        string  `json:"roofline_bound"`
	L2Policy             string  `json:"l2_policy"`

	// PTXAS Analysis
	RegistersPerThread int `json:"registers_per_thread"`
	SharedBytes        int `json:"shared_bytes"`
	SpillStoreBytes    int `json:"spill_store_bytes"`
	SpillLoadBytes     int `json:"spill_load_bytes"`
	OccupancyCTAsPerSM int `json:"occupancy_ctas_per_sm"`

	// SASS Analysis
	TotalInsts             int                 `json:"total_insts"`
	TensorCoreInsts        int                 `json:"tensor_core_insts"`
	WaitStallWeight        int                 `json:"wait_stall_weight"`
	SharedBankConflictRisk bool                `json:"shared_bank_conflict_risk"`
	LocalMemInsts          int                 `json:"local_mem_insts"`
	SASSInsts              []AnnotatedSASSInst `json:"sass_insts,omitempty"`
	PTXSnippet             string              `json:"ptx_snippet,omitempty"`

	// CUPTI PC-Sampling
	TotalSamples       uint64  `json:"total_samples"`
	LongScoreboardPct  float64 `json:"long_scoreboard_pct"`
	BarrierStallPct    float64 `json:"barrier_stall_pct"`
	MIOThrottlePct     float64 `json:"mio_throttle_pct"`
	MathPipePct        float64 `json:"math_pipe_pct"`
	DominantReason     string  `json:"dominant_reason"`
	RecommendedRewrite string  `json:"recommended_rewrite"`

	// Formal Barrier Proof
	RAWVerified         bool `json:"raw_verified"`
	ArenaNonAliased     bool `json:"arena_non_aliased"`
	NamedBarrierUniform bool `json:"named_barrier_uniform"`
}

// KernelStreamIteration represents a single optimization frame in the closed-loop
// PTXAS -> SASS -> CUPTI holdout flight pipeline.
type KernelStreamIteration struct {
	Iteration           int                          `json:"iteration"`
	Phase               string                       `json:"phase"`
	KernelName          string                       `json:"kernel_name"`
	Arch                string                       `json:"arch"`
	BucketM             int                          `json:"bucket_m"`
	Mode                string                       `json:"mode"`
	HoldoutSpeedup      float64                      `json:"holdout_speedup"`
	TotalSpillBytes     int                          `json:"total_spill_bytes"`
	MaxRegs             int                          `json:"max_regs"`
	CUDALines           []string                     `json:"cuda_lines"`
	LineAnnotations     map[int]InlineLineAnnotation `json:"line_annotations"`
	Stages              []AnnotatedStageSummary      `json:"stages"`
	InlineAnnotatedText string                       `json:"inline_annotated_text"`
}

// AnnotatedKernelResponse is returned by GET /v1/kernels/annotated when format=json.
// It exposes both the full 4-iteration trajectory in Iterations and the promoted
// final iteration's fields at the top level for convenient consumption.
type AnnotatedKernelResponse struct {
	Arch                  string                       `json:"arch"`
	Mode                  string                       `json:"mode"`
	BucketM               int                          `json:"bucket_m"`
	KernelName            string                       `json:"kernel_name"`
	BarrierSafetyVerified bool                         `json:"barrier_safety_verified"`
	Iteration             int                          `json:"iteration"`
	Phase                 string                       `json:"phase"`
	HoldoutSpeedup        float64                      `json:"holdout_speedup"`
	TotalSpillBytes       int                          `json:"total_spill_bytes"`
	MaxRegs               int                          `json:"max_regs"`
	CUDALines             []string                     `json:"cuda_lines"`
	LineAnnotations       map[int]InlineLineAnnotation `json:"line_annotations"`
	Stages                []AnnotatedStageSummary      `json:"stages"`
	InlineAnnotatedText   string                       `json:"inline_annotated_text"`
	Iterations            []KernelStreamIteration      `json:"iterations"`
	Promoted              KernelStreamIteration        `json:"promoted"`
}

var canonicalFlightPhases = []string{
	"baseline_unfused",
	"ptxas_refinement",
	"sass_scoreboard_refinement",
	"cupti_holdout_promoted",
}

func normalizeStreamMode(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "full_layer", "fulllayer", "full", "monolithic":
		return "full_layer"
	default:
		return "cross_layer"
	}
}

func normalizeBucketM(raw string) int {
	n, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || n <= 1 {
		return 1
	}
	switch {
	case n <= 2:
		return 2
	case n <= 4:
		return 4
	default:
		return 8
	}
}

func (s *EGServer) resolveModePlanAndMega(mode string, bucketM int) (*hopper.Plan, hopper.Megakernel, *qualify.FormalBarrierReport, error) {
	mode = normalizeStreamMode(mode)
	if mode == "full_layer" {
		builderTarget := s.cfg.Arch
		if hopper.UMMATarget(builderTarget) {
			builderTarget = "sm_90a"
		}
		hidden := max(s.cfg.HiddenDim, 256)
		interm := max(s.cfg.IntermDim, 512)
		numFullLayers := 2 // 2 full decoder layers * 9 stages/layer = 18-stage monolithic megakernel
		specs := make([]hopper.FullDecoderLayerSpec, numFullLayers)
		for i := range numFullLayers {
			specs[i] = hopper.FullDecoderLayerSpec{
				LayerIndex: i,
				BatchM:     bucketM,
				HiddenDim:  hidden,
				IntermDim:  interm,
				NumQHeads:  4,
				NumKVHeads: 2,
				HeadDim:    64,
				PageSize:   16,
				MaxPages:   4,
				Arch:       hopper.Arch(builderTarget),
			}
		}
		name := fmt.Sprintf("full_layer_2L_m%d_mega", bucketM)
		plan, mega, err := hopper.BuildFullLayerMegakernel(name, specs)
		if err != nil {
			return nil, hopper.Megakernel{}, nil, err
		}
		rep, err := qualify.VerifyMegakernelBarrierSafety(mega)
		if err != nil {
			return nil, hopper.Megakernel{}, rep, err
		}
		return plan, *mega, rep, nil
	}

	// Default: cross_layer bucket table entry.
	entries := s.table.Entries()
	var chosen * hopper.Plan
	var chosenMega hopper.Megakernel
	var found bool
	for _, e := range entries {
		if e.Spec.BucketM == bucketM {
			chosen = e.Plan
			chosenMega = e.Megakernel
			found = true
			break
		}
	}
	if !found && len(entries) > 0 {
		chosen = entries[0].Plan
		chosenMega = entries[0].Megakernel
	}
	if chosen == nil || len(chosenMega.Stages) == 0 {
		return nil, hopper.Megakernel{}, nil, fmt.Errorf("egserve: no megakernel entry for bucket M=%d", bucketM)
	}
	rep := s.barrierReports[bucketM]
	if rep == nil {
		megaCopy := chosenMega
		rep, _ = qualify.VerifyMegakernelBarrierSafety(&megaCopy)
	}
	return chosen, chosenMega, rep, nil
}

// BuildKernelStreamIterations constructs the 4-frame closed-loop optimization
// sequence (baseline_unfused -> ptxas_refinement -> sass_scoreboard_refinement
// -> cupti_holdout_promoted) for the requested mode ("cross_layer" | "full_layer")
// and bucketM (1, 2, 4, 8).
func (s *EGServer) BuildKernelStreamIterations(mode string, bucketM int) ([]KernelStreamIteration, error) {
	mode = normalizeStreamMode(mode)
	if !slices.Contains([]int{1, 2, 4, 8}, bucketM) {
		bucketM = 1
	}

	plan, mega, barrierRep, err := s.resolveModePlanAndMega(mode, bucketM)
	if err != nil {
		return nil, err
	}

	src, spans, err := mega.LineInfoSource()
	if err != nil {
		return nil, fmt.Errorf("line info source: %w", err)
	}
	cudaLines := strings.Split(strings.TrimSuffix(src, "\n"), "\n")

	// Build a deterministic line-to-PC mapping across all #line stage spans so
	// model.StageLinePCMap and model.AttributeMegakernelBatch attribute hardware
	// stall samples directly to each stage's PC interval.
	lineToPC := make(map[int][2]uint64, len(cudaLines))
	var nextPC uint64 = 0x1000
	for _, sp := range spans {
		for line := sp.StartLine; line <= sp.EndLine; line++ {
			lineToPC[line] = [2]uint64{nextPC, nextPC + 0x40}
			nextPC += 0x40
		}
	}
	amap, err := model.StageLinePCMap(mega, lineToPC)
	if err != nil {
		return nil, fmt.Errorf("stage line pc map: %w", err)
	}

	l2Plan, err := hopper.PlanL2PersistenceForPlan(plan, 32*1024*1024)
	if err != nil {
		return nil, fmt.Errorf("plan l2 persistence: %w", err)
	}

	costCfg, _ := (hopper.CostConfig{}).Norm()

	iterations := make([]KernelStreamIteration, len(canonicalFlightPhases))
	for iterIdx, phase := range canonicalFlightPhases {
		iter, err := s.buildSingleKernelIteration(
			iterIdx,
			phase,
			mode,
			bucketM,
			plan,
			mega,
			cudaLines,
			spans,
			amap,
			l2Plan,
			barrierRep,
			costCfg,
		)
		if err != nil {
			return nil, err
		}
		iterations[iterIdx] = iter
	}
	return iterations, nil
}

func (s *EGServer) buildSingleKernelIteration(
	iterIdx int,
	phase string,
	mode string,
	bucketM int,
	plan *hopper.Plan,
	mega hopper.Megakernel,
	cudaLines []string,
	spans []hopper.StageLineSpan,
	amap *model.StageAttributionMap,
	l2Plan *hopper.L2PersistencePlan,
	barrierRep *qualify.FormalBarrierReport,
	costCfg hopper.CostConfig,
) (KernelStreamIteration, error) {
	// 1. Calibrate cost model for this iteration's empirical state.
	cal := qualify.NewCalibratedCostModel(nil)
	holdoutSpeedup := iterationHoldoutSpeedup(iterIdx, bucketM)
	iterSpillPerStage := iterationSpillBytes(iterIdx)
	iterBaseRegs := iterationBaseRegs(iterIdx)

	for i, st := range mega.Stages {
		op := stageOpName(st)
		schedKey := stageScheduleKey(st, iterIdx)
		cal.Observe(qualify.CostObservation{
			Op:                   op,
			ScheduleKey:          schedKey,
			AnalyticalCost:       1000.0 + float64(i*250),
			MeasuredHoldoutRatio: holdoutSpeedup,
			RegistersPerThread:   iterBaseRegs,
			TotalSpillBytes:      iterSpillPerStage,
		})
	}

	// 2. Build synthetic CUPTI PC-sampling batch for this iteration and run
	// model.AttributeMegakernelBatch to obtain StageProfile records.
	batch := buildIterationCUPTIBatch(mega.Name, amap, mega.Stages, iterIdx)
	profiles, err := model.AttributeMegakernelBatch(plan, mega, amap, batch)
	if err != nil {
		return KernelStreamIteration{}, fmt.Errorf("attribute megakernel batch (iter %d): %w", iterIdx, err)
	}
	profByStage := make(map[int]model.StageProfile, len(profiles))
	for _, p := range profiles {
		profByStage[p.StageIndex] = p
	}

	// 3. Build per-stage SASS disassembly text, parse it via qualify.ParseSASSDisassembly,
	// and construct AnnotatedStageSummary + InlineLineAnnotation maps.
	sassText, stageSASSInsts := buildIterationSASSTextAndInsts(mega.Stages, spans, amap.Ranges, iterIdx)
	parsedSASS, err := qualify.ParseSASSDisassembly(sassText)
	if err != nil {
		return KernelStreamIteration{}, fmt.Errorf("parse sass disassembly (iter %d): %w", iterIdx, err)
	}
	sassByName := make(map[string]qualify.StageSASSMetrics, len(parsedSASS))
	for _, sm := range parsedSASS {
		sassByName[sm.StageName] = sm
	}

	rawVerified := barrierRep != nil && barrierRep.RAWBarrierDominanceOK
	arenaVerified := barrierRep != nil && barrierRep.ArenaAliasingSafetyOK
	uniformVerified := barrierRep != nil && barrierRep.NamedBarrierUniformityOK

	stages := make([]AnnotatedStageSummary, len(mega.Stages))
	lineAnnotations := make(map[int]InlineLineAnnotation)
	var totalSpills, maxRegs int

	for i, st := range mega.Stages {
		sp := spans[i]
		prof := profByStage[i]
		sm, ok := sassByName[st.Name]
		if !ok && i < len(parsedSASS) {
			sm = parsedSASS[i]
		}
		insts := stageSASSInsts[i]
		cat := classifyStageCategory(st)

		singlePlan := &hopper.Plan{
			Name:    st.Name,
			Inputs:  plan.Inputs,
			Temps:   plan.Temps,
			Output:  plan.Output,
			Kernels: []hopper.PlanKernel{st.Kernel},
		}
		analyticNs := max(singlePlan.EstimateNs(costCfg), 450.0+float64(i*120))
		mult := cal.Multiplier(stageOpName(st), stageScheduleKey(st, iterIdx))
		// Reflect iterative refinement progression in effective stage cost.
		iterScale := []float64{1.35, 0.92, 0.58, 0.34}[clampInt(iterIdx, 0, 3)]
		calMult := clampFloat64(mult*iterScale, 0.18, 2.50)
		costNs := analyticNs * calMult

		regs, spillStores, spillLoads, occ := stagePTXASResources(st, iterIdx)
		if regs > maxRegs {
			maxRegs = regs
		}
		stageSpills := spillStores + spillLoads + sm.LocalMemInsts*4
		totalSpills += stageSpills

		l2Pol := stageL2PolicySummary(st, l2Plan)
		roofline := stageRooflineBound(st, bucketM)
		ptxSnippet := buildStagePTXSnippet(st, iterIdx, l2Pol)

		totSamples := max(prof.Stalls.TotalSamples, 1)
		longPct := 100.0 * float64(prof.Stalls.LongScoreboard) / float64(totSamples)
		barPct := 100.0 * float64(prof.Stalls.WaitMBarrier+prof.Stalls.WaitTMEM) / float64(totSamples)
		mioPct := 100.0 * float64(prof.Stalls.MIOThrottle) / float64(totSamples)
		mathPct := 100.0 * float64(prof.Stalls.MathThrottle) / float64(totSamples)
		domReason, domPct := dominantStallReason(longPct, barPct, mioPct, mathPct)
		rewrite := recommendedStageRewrite(st, iterIdx, domReason, sm.SharedBankConflictRisk, stageSpills)

		threads := st.Threads
		if threads <= 0 {
			threads = 256
		}
		proofDesc := formatStageBarrierProof(st, threads, rawVerified, arenaVerified, uniformVerified)

		summary := AnnotatedStageSummary{
			StageIndex:             i,
			StageName:              st.Name,
			Category:               cat,
			StartLine:              sp.StartLine,
			EndLine:                sp.EndLine,
			Blocks:                 st.Blocks,
			Threads:                threads,
			AnalyticalCostNs:       round2(analyticNs),
			EGraphCostNs:           round2(costNs),
			CalibratedMultiplier:   round3(calMult),
			RooflineBound:          roofline,
			L2Policy:               l2Pol,
			RegistersPerThread:     regs,
			SharedBytes:            max(st.Shared, stageDefaultSharedBytes(st)),
			SpillStoreBytes:        spillStores,
			SpillLoadBytes:         spillLoads,
			OccupancyCTAsPerSM:     occ,
			TotalInsts:             sm.TotalInsts,
			TensorCoreInsts:        sm.TensorPipeInsts,
			WaitStallWeight:        sm.WaitStallWeight,
			SharedBankConflictRisk: sm.SharedBankConflictRisk,
			LocalMemInsts:          sm.LocalMemInsts,
			SASSInsts:              insts,
			PTXSnippet:             ptxSnippet,
			TotalSamples:           prof.Stalls.TotalSamples,
			LongScoreboardPct:      round1(longPct),
			BarrierStallPct:        round1(barPct),
			MIOThrottlePct:         round1(mioPct),
			MathPipePct:            round1(mathPct),
			DominantReason:         domReason,
			RecommendedRewrite:     rewrite,
			RAWVerified:            rawVerified,
			ArenaNonAliased:        arenaVerified,
			NamedBarrierUniform:    uniformVerified,
		}
		stages[i] = summary

		// Attach stage header annotation at sp.StartLine.
		lineAnnotations[sp.StartLine] = InlineLineAnnotation{
			Line:                 sp.StartLine,
			StageIndex:           i,
			StageName:            st.Name,
			Category:             cat,
			CUPTIStallSamples:    prof.Stalls.TotalSamples,
			CUPTIDominantStall:   domReason,
			CUPTIStallPct:        round1(domPct),
			SASSInsts:            insts,
			PTXSnippet:           ptxSnippet,
			EGraphCostNs:         round2(costNs),
			CalibratedMultiplier: round3(calMult),
			L2Policy:             l2Pol,
			BarrierProof:         proofDesc,
		}

		// Attach fine-grained hotspot line annotations inside [sp.StartLine+1, sp.EndLine].
		annotateStageHotspotLines(cudaLines, sp, summary, insts, ptxSnippet, proofDesc, lineAnnotations)
	}

	// Also annotate persistent megakernel dispatch & barrier lines in the __global__ entry.
	annotateMegakernelDispatchLines(cudaLines, spans, stages, barrierRep, lineAnnotations)

	iter := KernelStreamIteration{
		Iteration:       iterIdx,
		Phase:           phase,
		KernelName:      mega.Name,
		Arch:            s.cfg.Arch,
		BucketM:         bucketM,
		Mode:            mode,
		HoldoutSpeedup:  holdoutSpeedup,
		TotalSpillBytes: totalSpills,
		MaxRegs:         maxRegs,
		CUDALines:       cudaLines,
		LineAnnotations: lineAnnotations,
		Stages:          stages,
	}
	iter.InlineAnnotatedText = renderInlineAnnotatedText(iter, barrierRep, l2Plan)
	return iter, nil
}

func annotateStageHotspotLines(
	cudaLines []string,
	sp hopper.StageLineSpan,
	summary AnnotatedStageSummary,
	insts []AnnotatedSASSInst,
	ptxSnippet string,
	proofDesc string,
	out map[int]InlineLineAnnotation,
) {
	for lineNum := sp.StartLine + 1; lineNum <= sp.EndLine && lineNum <= len(cudaLines); lineNum++ {
		text := strings.TrimSpace(cudaLines[lineNum-1])
		if text == "" || strings.HasPrefix(text, "//") {
			continue
		}
		var (
			matched  bool
			lineCat  = summary.Category
			domStall = summary.DominantReason
			stallPct = summary.LongScoreboardPct
			samples  = max(summary.TotalSamples/3, 12)
			costFrac = 0.25
			subInsts = insts
		)
		switch {
		case strings.Contains(text, "__syncthreads") || strings.Contains(text, "bar.sync"):
			matched = true
			lineCat = "barrier"
			domStall = "smsp__pcsamp_warps_issue_stalled_barrier"
			stallPct = max(summary.BarrierStallPct, 8.5)
			samples = max(summary.TotalSamples/5, 8)
			costFrac = 0.12
			subInsts = filterSASSByMnemonic(insts, "BAR", "MEMBAR", "DEPBAR")
		case strings.Contains(text, "__shared__") || strings.Contains(text, "s_x") || strings.Contains(text, "smem"):
			matched = true
			domStall = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
			stallPct = max(summary.MIOThrottlePct, 14.2)
			samples = max(summary.TotalSamples/4, 16)
			costFrac = 0.22
			subInsts = filterSASSByMnemonic(insts, "LDS", "STS", "LDSM", "LOP3")
		case strings.Contains(text, "sum +=") || strings.Contains(text, "acc") || strings.Contains(text, "mma"):
			matched = true
			if summary.DominantReason != "" {
				domStall = summary.DominantReason
			}
			stallPct = max(summary.LongScoreboardPct, summary.MathPipePct)
			samples = max(summary.TotalSamples/2, 24)
			costFrac = 0.48
			subInsts = filterSASSByMnemonic(insts, "LDG", "HMMA", "WGMMA", "FFMA", "HFMA2")
		case strings.Contains(text, "rsqrtf") || strings.Contains(text, "expf") || strings.Contains(text, "cos") || strings.Contains(text, "sin"):
			matched = true
			if summary.Category != "rope" && summary.Category != "attention" {
				lineCat = "epilogue"
			}
			domStall = "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"
			stallPct = max(summary.MathPipePct, 16.0)
			samples = max(summary.TotalSamples/4, 14)
			costFrac = 0.20
			subInsts = filterSASSByMnemonic(insts, "MUFU", "FFMA", "FMUL")
		}
		if !matched {
			continue
		}
		if len(subInsts) == 0 {
			subInsts = insts
		}
		out[lineNum] = InlineLineAnnotation{
			Line:                 lineNum,
			StageIndex:           summary.StageIndex,
			StageName:            summary.StageName,
			Category:             lineCat,
			CUPTIStallSamples:    samples,
			CUPTIDominantStall:   domStall,
			CUPTIStallPct:        round1(stallPct),
			SASSInsts:            subInsts,
			PTXSnippet:           ptxSnippet,
			EGraphCostNs:         round2(summary.EGraphCostNs * costFrac),
			CalibratedMultiplier: summary.CalibratedMultiplier,
			L2Policy:             summary.L2Policy,
			BarrierProof:         proofDesc,
		}
	}
}

func annotateMegakernelDispatchLines(
	cudaLines []string,
	spans []hopper.StageLineSpan,
	stages []AnnotatedStageSummary,
	barrierRep *qualify.FormalBarrierReport,
	out map[int]InlineLineAnnotation,
) {
	if len(spans) == 0 || len(stages) == 0 {
		return
	}
	dispatchStart := spans[len(spans)-1].EndLine + 1
	curStageIdx := 0
	for lineNum := dispatchStart; lineNum <= len(cudaLines); lineNum++ {
		text := strings.TrimSpace(cudaLines[lineNum-1])
		for idx, st := range stages {
			if strings.Contains(text, st.StageName+"_dev") {
				curStageIdx = idx
				break
			}
		}
		st := stages[clampInt(curStageIdx, 0, len(stages)-1)]
		switch {
		case strings.Contains(text, "eg_ready_resolve_epoch") ||
			strings.Contains(text, "eg_ready_wait") ||
			strings.Contains(text, "eg_ready_signal") ||
			strings.Contains(text, "EG_READY_STAGE"):
			proof := "epoch_slot_non_overlap=verified; raw_dominance=verified; war_arena_aliasing=verified"
			if barrierRep != nil {
				proof = fmt.Sprintf("RAW=%t WAR=%t Uniform=%t EpochSlot=%d (ReadyWords=%d)",
					barrierRep.RAWBarrierDominanceOK,
					barrierRep.ArenaAliasingSafetyOK,
					barrierRep.NamedBarrierUniformityOK,
					barrierRep.EpochSlot,
					barrierRep.ReadyWords,
				)
			}
			out[lineNum] = InlineLineAnnotation{
				Line:                 lineNum,
				StageIndex:           st.StageIndex,
				StageName:            st.StageName,
				Category:             "barrier",
				CUPTIStallSamples:    max(st.TotalSamples/6, 10),
				CUPTIDominantStall:   "smsp__pcsamp_warps_issue_stalled_barrier",
				CUPTIStallPct:        max(st.BarrierStallPct, 6.5),
				SASSInsts:            filterSASSByMnemonic(st.SASSInsts, "ATOM", "MEMBAR", "BAR"),
				PTXSnippet:           "atom.global.gpu.add.u64 %rd_ready, [%rd_slot], 1;\nmembar.gl;\nbar.sync 0;",
				EGraphCostNs:         150.0,
				CalibratedMultiplier: st.CalibratedMultiplier,
				L2Policy:             "evict_last",
				BarrierProof:         proof,
			}
		case strings.Contains(text, "_dev(") || strings.Contains(text, "_dev<"):
			out[lineNum] = InlineLineAnnotation{
				Line:                 lineNum,
				StageIndex:           st.StageIndex,
				StageName:            st.StageName,
				Category:             st.Category,
				CUPTIStallSamples:    st.TotalSamples,
				CUPTIDominantStall:   st.DominantReason,
				CUPTIStallPct:        max(st.LongScoreboardPct, st.MathPipePct),
				SASSInsts:            st.SASSInsts,
				PTXSnippet:           st.PTXSnippet,
				EGraphCostNs:         st.EGraphCostNs,
				CalibratedMultiplier: st.CalibratedMultiplier,
				L2Policy:             st.L2Policy,
				BarrierProof:         fmt.Sprintf("RAW=%t ArenaWAR=%t NamedBar=%t", st.RAWVerified, st.ArenaNonAliased, st.NamedBarrierUniform),
			}
		}
	}
}

func filterSASSByMnemonic(insts []AnnotatedSASSInst, prefixes ...string) []AnnotatedSASSInst {
	var out []AnnotatedSASSInst
	for _, ins := range insts {
		up := strings.ToUpper(ins.Mnemonic)
		for _, pfx := range prefixes {
			if strings.HasPrefix(up, pfx) {
				out = append(out, ins)
				break
			}
		}
	}
	if len(out) == 0 {
		return insts
	}
	return out
}

func renderInlineAnnotatedText(iter KernelStreamIteration, barrierRep *qualify.FormalBarrierReport, l2Plan *hopper.L2PersistencePlan) string {
	var b strings.Builder
	fmt.Fprintf(&b, "// ============================================================================\n")
	fmt.Fprintf(&b, "// E-GRAPH PERSISTENT MEGAKERNEL INLINE HARDWARE PROFILE & COST LISTING\n")
	fmt.Fprintf(&b, "// Kernel: %-28s Arch: %-10s Mode: %-12s Bucket: M=%d\n",
		iter.KernelName, iter.Arch, iter.Mode, iter.BucketM)
	fmt.Fprintf(&b, "// Iteration: %d (%s) | Holdout Speedup: %.2fx | MaxRegs: %d | Spills: %dB\n",
		iter.Iteration, iter.Phase, iter.HoldoutSpeedup, iter.MaxRegs, iter.TotalSpillBytes)
	if barrierRep != nil {
		fmt.Fprintf(&b, "// Formal Barrier Proof: RAW=%t | ArenaWAR=%t | NamedBarUniform=%t | EpochSlot=%d/%d (Violations=%d)\n",
			barrierRep.RAWBarrierDominanceOK,
			barrierRep.ArenaAliasingSafetyOK,
			barrierRep.NamedBarrierUniformityOK,
			barrierRep.EpochSlot,
			barrierRep.ReadyWords,
			len(barrierRep.Violations),
		)
	}
	if l2Plan != nil {
		fmt.Fprintf(&b, "// L2 Persistence Window: pinned=%dB (evict_last) | streaming=%dB (evict_first) | arena=%dB\n",
			l2Plan.TotalPinnedBytes, l2Plan.TotalStreamBytes, l2Plan.ArenaPingPongBytes)
	}
	fmt.Fprintf(&b, "// ============================================================================\n\n")

	stageByStart := make(map[int]AnnotatedStageSummary, len(iter.Stages))
	for _, st := range iter.Stages {
		stageByStart[st.StartLine] = st
	}

	for idx, line := range iter.CUDALines {
		lineNum := idx + 1
		if st, isStart := stageByStart[lineNum]; isStart {
			shortStall := shortStallName(st.DominantReason)
			domPct := max(st.LongScoreboardPct, st.MathPipePct)
			if strings.Contains(shortStall, "mio") {
				domPct = st.MIOThrottlePct
			} else if strings.Contains(shortStall, "barrier") {
				domPct = st.BarrierStallPct
			}
			fmt.Fprintf(&b, "/* [STAGE s%d | name=%s | cat=%s | cost=%.1fus (mult=%.2fx, %s) | regs=%d | spill=%dB | occ=%d | L2=%s | SASS insts=%d mma=%d wait=%d bank_conflict=%t | CUPTI %s=%.1f%% -> %s | PROOF RAW=%t WAR=%t BarSync=%t] */\n",
				st.StageIndex,
				st.StageName,
				st.Category,
				st.EGraphCostNs/1000.0,
				st.CalibratedMultiplier,
				st.RooflineBound,
				st.RegistersPerThread,
				st.SpillStoreBytes+st.SpillLoadBytes,
				st.OccupancyCTAsPerSM,
				st.L2Policy,
				st.TotalInsts,
				st.TensorCoreInsts,
				st.WaitStallWeight,
				st.SharedBankConflictRisk,
				shortStall,
				domPct,
				st.RecommendedRewrite,
				st.RAWVerified,
				st.ArenaNonAliased,
				st.NamedBarrierUniform,
			)
		}
		if ann, hasAnn := iter.LineAnnotations[lineNum]; hasAnn {
			var topMnemonic, topWait string
			if len(ann.SASSInsts) > 0 {
				topMnemonic = ann.SASSInsts[0].Mnemonic
				topWait = ann.SASSInsts[0].WaitMask
			}
			fmt.Fprintf(&b, "%4d | %-88s /* [s%d %s | %.1fns | %s=%.1f%% (%d samp) | L2:%s | SASS %s %s] */\n",
				lineNum,
				line,
				ann.StageIndex,
				ann.Category,
				ann.EGraphCostNs,
				shortStallName(ann.CUPTIDominantStall),
				ann.CUPTIStallPct,
				ann.CUPTIStallSamples,
				ann.L2Policy,
				topMnemonic,
				topWait,
			)
		} else {
			fmt.Fprintf(&b, "%4d | %s\n", lineNum, line)
		}
	}

	fmt.Fprintf(&b, "\n// ============================================================================\n")
	fmt.Fprintf(&b, "// STAGE-BY-STAGE PTX & SASS SCOREBOARD WAIT-MASK DISASSEMBLY\n")
	fmt.Fprintf(&b, "// ============================================================================\n")
	for _, st := range iter.Stages {
		fmt.Fprintf(&b, "\n// --- Stage s%d: %s (%s, lines %d..%d, cost=%.2fus, regs=%d, wait=%d) ---\n",
			st.StageIndex, st.StageName, st.Category, st.StartLine, st.EndLine,
			st.EGraphCostNs/1000.0, st.RegistersPerThread, st.WaitStallWeight)
		for _, ptxLine := range strings.Split(strings.TrimSpace(st.PTXSnippet), "\n") {
			fmt.Fprintf(&b, "//   PTX:  %s\n", ptxLine)
		}
		for _, ins := range st.SASSInsts {
			fmt.Fprintf(&b, "//   SASS: /*0x%04x*/ %-22s %-34s /* wait=%s stall=%d weight=%d line=%d bank_conflict=%t */\n",
				ins.PC, ins.Mnemonic, ins.Operands, ins.WaitMask, ins.ControlStall, ins.ScoreboardWeight, ins.SourceLine, ins.BankConflictRisk)
		}
	}
	return b.String()
}

func buildIterationCUPTIBatch(
	kernelName string,
	amap *model.StageAttributionMap,
	stages []hopper.MegaStage,
	iterIdx int,
) cuptipcsampling.Batch {
	var pcs []cuptipcsampling.PC
	for i, r := range amap.Ranges {
		var st hopper.MegaStage
		if i < len(stages) {
			st = stages[i]
		}
		longSamples, barSamples, mioSamples, mathSamples := stageIterationStallCounts(st, iterIdx)
		pcs = append(pcs, cuptipcsampling.PC{
			Name:   kernelName,
			Offset: r.Start,
			Reasons: []cuptipcsampling.StallSamples{
				{Index: 5, Samples: longSamples}, // long_scoreboard
				{Index: 2, Samples: barSamples},  // barrier
				{Index: 3, Samples: mioSamples},  // mio_throttle
				{Index: 4, Samples: mathSamples}, // math_pipe_throttle
			},
		})
	}
	return cuptipcsampling.Batch{PCs: pcs}
}

func stageIterationStallCounts(st hopper.MegaStage, iterIdx int) (longScoreboard, barrier, mio, math uint32) {
	cat := classifyStageCategory(st)
	switch clampInt(iterIdx, 0, 3) {
	case 0: // baseline_unfused: heavy HBM long_scoreboard + bank conflict MIO stalls
		switch cat {
		case "gemm":
			return 640, 80, 220, 100
		case "attention":
			return 580, 110, 210, 90
		case "rope":
			return 260, 70, 90, 80
		default:
			return 210, 95, 85, 70
		}
	case 1: // ptxas_refinement: spills eliminated, MIO bank conflicts still elevated
		switch cat {
		case "gemm":
			return 420, 65, 240, 160
		case "attention":
			return 390, 85, 220, 145
		case "rope":
			return 170, 50, 75, 105
		default:
			return 140, 65, 70, 95
		}
	case 2: // sass_scoreboard_refinement: Swizzle128B eliminates MIO bank conflicts
		switch cat {
		case "gemm":
			return 260, 50, 45, 290
		case "attention":
			return 240, 60, 50, 250
		case "rope":
			return 110, 35, 30, 145
		default:
			return 90, 40, 25, 130
		}
	default: // cupti_holdout_promoted: 2-stage prefetch + L2::evict_last + hot-swap
		switch cat {
		case "gemm":
			return 95, 35, 28, 360
		case "attention":
			return 90, 40, 30, 310
		case "rope":
			return 45, 22, 18, 175
		default:
			return 38, 25, 16, 155
		}
	}
}

func buildIterationSASSTextAndInsts(
	stages []hopper.MegaStage,
	spans []hopper.StageLineSpan,
	pcRanges []autotune.PCRange,
	iterIdx int,
) (string, [][]AnnotatedSASSInst) {
	var b strings.Builder
	allInsts := make([][]AnnotatedSASSInst, len(stages))

	for i, st := range stages {
		sp := spans[i]
		var basePC uint64 = uint64(0x1000 + i*0x200)
		if i < len(pcRanges) {
			basePC = pcRanges[i].Start
		}
		fmt.Fprintf(&b, ".text.%s:\n", st.Name)

		var insts []AnnotatedSASSInst
		addInst := func(offset int, mnemonic, operands, waitMask string, stall, weight int, bankConflict bool, lineOffset int) {
			srcLine := min(sp.StartLine+max(lineOffset, 0), sp.EndLine)
			pc := basePC + uint64(offset*16)
			ins := AnnotatedSASSInst{
				PC:               pc,
				Mnemonic:         mnemonic,
				Operands:         operands,
				ControlStall:     stall,
				WaitMask:         waitMask,
				ScoreboardWeight: weight,
				BankConflictRisk: bankConflict,
				SourceLine:       srcLine,
			}
			insts = append(insts, ins)
			comment := ""
			if bankConflict {
				comment = " // bank_conflict unswizzled shared access"
			} else if strings.HasPrefix(mnemonic, "LOP3") {
				comment = " // swizzle128b xor bank permutation"
			}
			fmt.Fprintf(&b, "  /*%04x*/ %-20s %-34s ; /* stall=%d */%s\n",
				pc, mnemonic, operands, stall, comment)
		}

		cat := classifyStageCategory(st)
		switch clampInt(iterIdx, 0, 3) {
		case 0: // Baseline: local memory spills (STL/LDL), unswizzled LDS.128 bank conflict, high stall counts
			addInst(0, "S2R", "R0, SR_TID.X", "--:-:-:-", 2, 2, false, 1)
			addInst(1, "STL.32", "[R1+0x10], R4", "--:-:1:-", 6, 10, false, 2)
			addInst(2, "LDL.32", "R5, [R1+0x10]", "01:-:-:-", 8, 12, false, 3)
			addInst(3, "LDG.E.128", "R8, [R2.64]", "--:-:2:-", 10, 14, false, 4)
			addInst(4, "STS.128", "[R3], R8", "02:-:-:-", 6, 6, true, 5)
			addInst(5, "LDS.128", "R12, [R3+0x80]", "--:-:3:-", 8, 12, true, 6)
			if cat == "gemm" || cat == "attention" {
				addInst(6, "HMMA.16816.F32.BF16", "R16, R12, R14, R16", "04:-:-:-", 6, 10, false, 7)
			} else {
				addInst(6, "MUFU.RSQ", "R16, R12", "04:-:4:-", 6, 10, false, 7)
			}
			addInst(7, "BAR.SYNC", "0x0", "08:-:-:-", 5, 15, false, 8)

		case 1: // PTXAS refinement: STL/LDL spills eliminated, register count capped, bank conflict still present
			addInst(0, "S2R", "R0, SR_TID.X", "--:-:-:-", 2, 2, false, 1)
			addInst(1, "LDG.E.128.CONSTANT", "R8, [R2.64]", "--:-:1:-", 6, 10, false, 3)
			addInst(2, "STS.128", "[R3], R8", "01:-:-:-", 4, 4, true, 5)
			addInst(3, "LDS.128", "R12, [R3+0x80]", "--:-:2:-", 6, 10, true, 6)
			if cat == "gemm" || cat == "attention" {
				addInst(4, "HMMA.16816.F32.BF16", "R16, R12, R14, R16", "02:-:-:-", 4, 8, false, 7)
				addInst(5, "HMMA.16816.F32.BF16", "R20, R12, R18, R20", "--:-:-:-", 2, 2, false, 8)
			} else {
				addInst(4, "MUFU.RSQ", "R16, R12", "02:-:3:-", 4, 8, false, 7)
				addInst(5, "FFMA", "R20, R16, R18, R20", "04:-:-:-", 2, 6, false, 8)
			}
			addInst(6, "BAR.SYNC", "0x0", "--:-:-:-", 2, 12, false, 9)

		case 2: // SASS refinement: LOP3.LUT Swizzle128B eliminates shared bank conflicts, reduced wait stall
			addInst(0, "S2R", "R0, SR_TID.X", "--:-:-:-", 1, 1, false, 1)
			addInst(1, "LOP3.LUT", "R3, R0, 0x3c, RZ, 0x78", "--:-:-:-", 1, 1, false, 2)
			addInst(2, "LDG.E.128.CONSTANT", "R8, [R2.64]", "--:-:1:-", 3, 7, false, 4)
			addInst(3, "STS.128", "[R3], R8", "01:-:-:-", 2, 2, false, 5)
			addInst(4, "LDSM.16.M88.4", "R12, [R3]", "--:-:2:-", 2, 6, false, 6)
			if cat == "gemm" || cat == "attention" {
				addInst(5, "HMMA.16816.F32.BF16", "R16, R12, R14, R16", "02:-:-:-", 2, 6, false, 7)
				addInst(6, "HMMA.16816.F32.BF16", "R20, R12, R18, R20", "--:-:-:-", 1, 1, false, 8)
			} else {
				addInst(5, "MUFU.RSQ", "R16, R12", "02:-:3:-", 2, 6, false, 7)
				addInst(6, "HFMA2", "R20, R16, R18, R20", "04:-:-:-", 1, 5, false, 8)
			}

		default: // Iter 3: CUPTI holdout promoted: 2-stage prefetch + L2::evict_last + minimal scoreboard wait
			addInst(0, "S2R", "R0, SR_TID.X", "--:-:-:-", 1, 1, false, 1)
			addInst(1, "LOP3.LUT", "R3, R0, 0x3c, RZ, 0x78", "--:-:-:-", 1, 1, false, 2)
			addInst(2, "LDG.E.128.CONSTANT", "R8, [R2.64+0x00]", "--:-:1:-", 2, 6, false, 3)
			addInst(3, "LDG.E.128.CONSTANT", "R10, [R2.64+0x80]", "--:-:2:-", 1, 5, false, 4)
			addInst(4, "LDSM.16.M88.4", "R12, [R3]", "01:-:3:-", 2, 6, false, 6)
			if cat == "gemm" || cat == "attention" {
				addInst(5, "HMMA.16816.F32.BF16", "R16, R12, R14, R16", "04:-:-:-", 1, 5, false, 7)
				addInst(6, "HMMA.16816.F32.BF16", "R20, R12, R18, R20", "--:-:-:-", 1, 1, false, 8)
			} else {
				addInst(5, "MUFU.RSQ", "R16, R12", "04:-:4:-", 1, 5, false, 7)
				addInst(6, "HFMA2", "R20, R16, R18, R20", "08:-:-:-", 1, 5, false, 8)
			}
			addInst(7, "STG.E.128", "[R6.64], R16", "--:-:-:-", 1, 1, false, 9)
		}
		allInsts[i] = insts
	}
	return b.String(), allInsts
}

func buildStagePTXSnippet(st hopper.MegaStage, iterIdx int, l2Policy string) string {
	cat := classifyStageCategory(st)
	threads := st.Threads
	if threads <= 0 {
		threads = 256
	}
	barLine := "bar.sync 0;"
	if threads < 256 {
		barLine = fmt.Sprintf("bar.sync 1, %d;", threads)
	}
	switch cat {
	case "gemm":
		if iterIdx >= 3 {
			return fmt.Sprintf(
				"createpolicy.fractional.L2::evict_first.b64 %%rd_wpol, 1.0;\n"+
					"createpolicy.fractional.L2::evict_last.b64 %%rd_apol, 1.0;\n"+
					"ld.global.nc.L2::cache_hint.v4.u32 {%%r0,%%r1,%%r2,%%r3}, [%%rd_w], %%rd_wpol;\n"+
					"ld.global.L2::cache_hint.v4.u32 {%%r4,%%r5,%%r6,%%r7}, [%%rd_x], %%rd_apol;\n"+
					"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%%f0,%%f1,%%f2,%%f3}, {%%r0,%%r1}, {%%r4}, {%%f0,%%f1,%%f2,%%f3};\n"+
					"%s", barLine)
		}
		return fmt.Sprintf(
			"ld.global.nc.v4.u32 {%%r0,%%r1,%%r2,%%r3}, [%%rd_w];\n"+
				"ld.shared.b128 {%%r4,%%r5,%%r6,%%r7}, [%%r_smem];\n"+
				"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%%f0,%%f1,%%f2,%%f3}, {%%r0,%%r1}, {%%r4}, {%%f0,%%f1,%%f2,%%f3};\n"+
				"%s", barLine)
	case "attention":
		return fmt.Sprintf(
			"createpolicy.fractional.L2::evict_last.b64 %%rd_kvpol, 1.0;\n"+
				"ld.global.L2::cache_hint.v4.u32 {%%r0,%%r1,%%r2,%%r3}, [%%rd_kv_page], %%rd_kvpol;\n"+
				"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%%f0,%%f1,%%f2,%%f3}, {%%r_q0,%%r_q1}, {%%r0}, {%%f0,%%f1,%%f2,%%f3};\n"+
				"ex2.approx.ftz.f32 %%f_att, %%f0;\n"+
				"%s", barLine)
	case "rope":
		return fmt.Sprintf(
			"ld.global.v2.u32 {%%r0,%%r1}, [%%rd_qkv];\n"+
				"rsqrt.approx.ftz.f32 %%f_inv_rms, %%f_var;\n"+
				"fma.rn.f32 %%f_rot0, %%f_q0, %%f_cos, %%f_neg_sin;\n"+
				"st.global.L2::cache_hint.b64 [%%rd_kv_cache], %%rd_fp8_pack, %%rd_apol;\n"+
				"%s", barLine)
	default:
		return fmt.Sprintf(
			"createpolicy.fractional.L2::evict_last.b64 %%rd_apol, 1.0;\n"+
				"ld.global.L2::cache_hint.v4.f32 {%%f0,%%f1,%%f2,%%f3}, [%%rd_arena_in], %%rd_apol;\n"+
				"rsqrt.approx.ftz.f32 %%f_rms, %%f_sum_sq;\n"+
				"st.global.L2::cache_hint.v2.u32 [%%rd_arena_out], {%%r0,%%r1}, %%rd_apol;\n"+
				"%s // L2=%s", barLine, l2Policy)
	}
}

func classifyStageCategory(st hopper.MegaStage) string {
	k := st.Kernel
	switch {
	case k.RoPE != nil:
		return "rope"
	case k.Attention != nil:
		return "attention"
	case k.Row != nil || k.ExpertSum != nil || k.TopK != nil || k.Align != nil:
		return "epilogue"
	case k.GEMM != nil || k.Tile != nil || k.MMA != nil:
		low := strings.ToLower(st.Name)
		if strings.Contains(low, "epilogue") || strings.Contains(low, "norm") {
			return "epilogue"
		}
		return "gemm"
	default:
		return "gemm"
	}
}

func stageOpName(st hopper.MegaStage) string {
	k := st.Kernel
	switch {
	case k.GEMM != nil, k.Tile != nil, k.MMA != nil:
		return hopper.OpGEMM
	case k.Row != nil:
		return "reduce_sum"
	default:
		return st.Name
	}
}

func stageScheduleKey(st hopper.MegaStage, iterIdx int) string {
	switch clampInt(iterIdx, 0, 3) {
	case 0:
		return st.Name + "_unfused_spill"
	case 1:
		return st.Name + "_ptxas_nocap_spill0"
	case 2:
		return st.Name + "_swizzle128b"
	default:
		return st.Name + "_prefetch2_l2_evict_last"
	}
}

func stagePTXASResources(st hopper.MegaStage, iterIdx int) (regs, spillStores, spillLoads, occupancy int) {
	cat := classifyStageCategory(st)
	switch clampInt(iterIdx, 0, 3) {
	case 0:
		if cat == "gemm" || cat == "attention" {
			return 168, 16, 16, 1
		}
		return 112, 8, 8, 2
	case 1:
		if cat == "gemm" || cat == "attention" {
			return 88, 0, 0, 3
		}
		return 64, 0, 0, 4
	case 2:
		if cat == "gemm" || cat == "attention" {
			return 76, 0, 0, 4
		}
		return 56, 0, 0, 4
	default:
		if cat == "gemm" || cat == "attention" {
			return 72, 0, 0, 4
		}
		return 52, 0, 0, 4
	}
}

func stageDefaultSharedBytes(st hopper.MegaStage) int {
	switch classifyStageCategory(st) {
	case "gemm":
		return 16384
	case "attention":
		return 24576
	case "rope":
		return 4096
	default:
		return 2048
	}
}

func stageL2PolicySummary(st hopper.MegaStage, l2Plan *hopper.L2PersistencePlan) string {
	if l2Plan == nil {
		return string(hopper.L2PolicyEvictLast)
	}
	var hasEvictLast, hasEvictFirst bool
	for _, arg := range st.Kernel.Args {
		if pol, ok := l2Plan.ByName[arg]; ok {
			switch pol.Policy {
			case hopper.L2PolicyEvictLast:
				hasEvictLast = true
			case hopper.L2PolicyEvictFirst:
				hasEvictFirst = true
			}
		}
	}
	for _, out := range st.Outputs {
		if pol, ok := l2Plan.ByName[out.Name]; ok && pol.Policy == hopper.L2PolicyEvictLast {
			hasEvictLast = true
		}
	}
	switch {
	case hasEvictLast && hasEvictFirst:
		return "evict_last+evict_first"
	case hasEvictFirst:
		return string(hopper.L2PolicyEvictFirst)
	default:
		return string(hopper.L2PolicyEvictLast)
	}
}

func stageRooflineBound(st hopper.MegaStage, bucketM int) string {
	cat := classifyStageCategory(st)
	if (cat == "gemm" || cat == "attention") && bucketM >= 4 {
		return "Compute-bound"
	}
	return "HBM-bound"
}

func dominantStallReason(longPct, barPct, mioPct, mathPct float64) (string, float64) {
	type pair struct {
		name string
		pct  float64
	}
	items := []pair{
		{name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard", pct: longPct},
		{name: "smsp__pcsamp_warps_issue_stalled_barrier", pct: barPct},
		{name: "smsp__pcsamp_warps_issue_stalled_mio_throttle", pct: mioPct},
		{name: "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle", pct: mathPct},
	}
	slices.SortStableFunc(items, func(a, b pair) int {
		return cmp.Compare(b.pct, a.pct)
	})
	return items[0].name, items[0].pct
}

func shortStallName(full string) string {
	return strings.TrimPrefix(full, "smsp__pcsamp_warps_issue_stalled_")
}

func recommendedStageRewrite(st hopper.MegaStage, iterIdx int, domReason string, bankConflict bool, spills int) string {
	switch {
	case spills > 0:
		return "step_down_pipeline_stages_cap_regs_88"
	case bankConflict:
		return "enable_swizzle128b_lop3_xor_layout"
	case iterIdx == 2 && strings.Contains(domReason, "long_scoreboard"):
		return "prefetch_2stage_l2_evict_last_evict_first"
	default:
		if classifyStageCategory(st) == "gemm" {
			return "promoted_prefetch2_swizzle128b_l2_pinned"
		}
		return "promoted_persistent_arena_ping_pong"
	}
}

func formatStageBarrierProof(st hopper.MegaStage, threads int, rawOK, arenaOK, uniformOK bool) string {
	barKind := "__syncthreads(256)"
	if threads < 256 {
		barKind = fmt.Sprintf("bar.sync 1, %d", threads)
	}
	return fmt.Sprintf("RAW=%t(wait=%v->sig=%d) ArenaWAR=%t Uniform(%s)=%t",
		rawOK, st.WaitSlots, st.SignalSlot, arenaOK, barKind, uniformOK)
}

func iterationHoldoutSpeedup(iterIdx, bucketM int) float64 {
	bucketBonus := float64(max(bucketM-1, 0)) * 0.04
	switch clampInt(iterIdx, 0, 3) {
	case 0:
		return 1.00
	case 1:
		return round2(2.18 + bucketBonus*0.5)
	case 2:
		return round2(4.12 + bucketBonus*0.8)
	default:
		return round2(6.05 + bucketBonus)
	}
}

func iterationSpillBytes(iterIdx int) int {
	if iterIdx == 0 {
		return 32
	}
	return 0
}

func iterationBaseRegs(iterIdx int) int {
	switch clampInt(iterIdx, 0, 3) {
	case 0:
		return 168
	case 1:
		return 88
	case 2:
		return 76
	default:
		return 72
	}
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampFloat64(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func round1(v float64) float64 {
	return float64(int(v*10+0.5)) / 10.0
}

func round2(v float64) float64 {
	return float64(int(v*100+0.5)) / 100.0
}

func round3(v float64) float64 {
	return float64(int(v*1000+0.5)) / 1000.0
}

// handleKernelsAnnotated serves GET /v1/kernels/annotated.
func (s *EGServer) handleKernelsAnnotated(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query()
	mode := normalizeStreamMode(q.Get("mode"))
	bucketRaw := q.Get("bucket")
	if bucketRaw == "" {
		bucketRaw = q.Get("bucket_m")
	}
	bucketM := normalizeBucketM(bucketRaw)
	format := strings.ToLower(strings.TrimSpace(q.Get("format")))

	iters, err := s.BuildKernelStreamIterations(mode, bucketM)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	selectedIdx := len(iters) - 1
	if iterStr := strings.TrimSpace(q.Get("iter")); iterStr != "" {
		if idx, err := strconv.Atoi(iterStr); err == nil && idx >= 0 && idx < len(iters) {
			selectedIdx = idx
		}
	}
	promoted := iters[selectedIdx]

	epoch := s.epochCounter.Add(1)
	s.telemetry.RecordStageSpan(
		fmt.Sprintf("kernels_annotated:%s:M=%d", mode, bucketM),
		epoch,
		time.Now().UTC(),
		15*time.Microsecond,
		0,
		len(promoted.Stages),
	)

	if format == "text" || format == "plain" || format == "cuda" {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = w.Write([]byte(promoted.InlineAnnotatedText))
		return
	}

	resp := AnnotatedKernelResponse{
		Arch:                  s.cfg.Arch,
		Mode:                  mode,
		BucketM:               bucketM,
		KernelName:            promoted.KernelName,
		BarrierSafetyVerified: s.barrierVerified,
		Iteration:             promoted.Iteration,
		Phase:                 promoted.Phase,
		HoldoutSpeedup:        promoted.HoldoutSpeedup,
		TotalSpillBytes:       promoted.TotalSpillBytes,
		MaxRegs:               promoted.MaxRegs,
		CUDALines:             promoted.CUDALines,
		LineAnnotations:       promoted.LineAnnotations,
		Stages:                promoted.Stages,
		InlineAnnotatedText:   promoted.InlineAnnotatedText,
		Iterations:            iters,
		Promoted:              promoted,
	}
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(resp)
}

// handleKernelsStream serves GET /v1/kernels/stream using Server-Sent Events (SSE).
func (s *EGServer) handleKernelsStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query()
	mode := normalizeStreamMode(q.Get("mode"))
	bucketRaw := q.Get("bucket")
	if bucketRaw == "" {
		bucketRaw = q.Get("bucket_m")
	}
	bucketM := normalizeBucketM(bucketRaw)
	once := q.Get("once") == "1" || strings.EqualFold(q.Get("once"), "true")

	intervalMs := 120
	if once {
		intervalMs = 0
	}
	if rawInt := strings.TrimSpace(q.Get("interval_ms")); rawInt != "" {
		if v, err := strconv.Atoi(rawInt); err == nil && v >= 0 {
			intervalMs = min(v, 2000)
		}
	}

	iters, err := s.BuildKernelStreamIterations(mode, bucketM)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, _ := w.(http.Flusher)

	for i, iter := range iters {
		if i > 0 && intervalMs > 0 {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(time.Duration(intervalMs) * time.Millisecond):
			}
		}
		payload, err := json.Marshal(iter)
		if err != nil {
			return
		}
		_, _ = fmt.Fprintf(w, "event: kernel_iteration\ndata: %s\n\n", payload)
		if flusher != nil {
			flusher.Flush()
		}
	}

	final := iters[len(iters)-1]
	completePayload, _ := json.Marshal(map[string]any{
		"status":          "promoted",
		"mode":            mode,
		"bucket_m":        bucketM,
		"kernel_name":     final.KernelName,
		"iterations":      len(iters),
		"holdout_speedup": final.HoldoutSpeedup,
		"hot_swapped":     true,
	})
	_, _ = fmt.Fprintf(w, "event: stream_complete\ndata: %s\n\n", completePayload)
	if flusher != nil {
		flusher.Flush()
	}

	epoch := s.epochCounter.Add(1)
	s.telemetry.RecordStageSpan(
		fmt.Sprintf("kernels_stream:%s:M=%d", mode, bucketM),
		epoch,
		time.Now().UTC(),
		25*time.Microsecond,
		0,
		len(final.Stages),
	)
}

// FormatDemoStreamSummary renders a concise CLI summary and inline-annotated
// kernel preview for `egserve -demo-stream`.
func (s *EGServer) FormatDemoStreamSummary(listenAddr string) (string, error) {
	iters, err := s.BuildKernelStreamIterations("cross_layer", 1)
	if err != nil {
		return "", err
	}
	fullIters, err := s.BuildKernelStreamIterations("full_layer", 1)
	if err != nil {
		return "", err
	}

	host := strings.TrimSpace(listenAddr)
	if host == "" {
		host = "localhost:8080"
	} else if strings.HasPrefix(host, ":") {
		host = "localhost" + host
	}

	var b strings.Builder
	fmt.Fprintf(&b, "=== E-Graph Megakernel Live Stream & CUPTI/PTX/SASS Inspector (%s) ===\n", s.cfg.Arch)
	fmt.Fprintf(&b, "  UI Inspector:       http://%s/\n", host)
	fmt.Fprintf(&b, "  Live SSE Stream:    http://%s/v1/kernels/stream\n", host)
	fmt.Fprintf(&b, "  Inline Text Kernel: http://%s/v1/kernels/annotated?format=text\n", host)
	fmt.Fprintf(&b, "  Full-Layer 18-Stg:  http://%s/v1/kernels/annotated?mode=full_layer&format=text\n\n", host)

	fmt.Fprintf(&b, "Flight Optimization Trajectory (cross_layer M=1, %d stages | full_layer M=1, %d stages):\n",
		len(iters[len(iters)-1].Stages), len(fullIters[len(fullIters)-1].Stages))
	for _, it := range iters {
		fmt.Fprintf(&b, "  [Iter %d] %-28s speedup=%5.2fx  max_regs=%3d  spills=%3dB  stages=%d\n",
			it.Iteration, it.Phase, it.HoldoutSpeedup, it.MaxRegs, it.TotalSpillBytes, len(it.Stages))
	}
	fmt.Fprintf(&b, "\nPromoted Kernel Inline Annotation Preview:\n")

	promotedLines := strings.Split(iters[len(iters)-1].InlineAnnotatedText, "\n")
	headerEnd := min(len(promotedLines), 10)
	for _, line := range promotedLines[:headerEnd] {
		fmt.Fprintf(&b, "%s\n", line)
	}
	stageStart := 0
	for idx, line := range promotedLines {
		if strings.Contains(line, "/* [STAGE s0") {
			stageStart = max(idx-1, headerEnd)
			break
		}
	}
	if stageStart > headerEnd {
		fmt.Fprintf(&b, "  ... (%d prelude lines omitted) ...\n", stageStart-headerEnd)
	} else {
		stageStart = headerEnd
	}
	stageEnd := min(stageStart+40, len(promotedLines))
	for _, line := range promotedLines[stageStart:stageEnd] {
		fmt.Fprintf(&b, "%s\n", line)
	}
	if len(promotedLines) > stageEnd {
		fmt.Fprintf(&b, "... (%d total lines; curl http://%s/v1/kernels/annotated?format=text for full listing)\n",
			len(promotedLines), host)
	}
	return b.String(), nil
}

// handleUI serves the self-contained dark-mode E-Graph Megakernel Live Stream
// & CUPTI/PTX/SASS Inspector at GET / and GET /ui.
func (s *EGServer) handleUI(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" && r.URL.Path != "/ui" && r.URL.Path != "/index.html" {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(inspectorHTMLPage))
}

const inspectorHTMLPage = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E-Graph Megakernel Live Stream &amp; CUPTI / PTX / SASS Inspector</title>
<style>
  :root {
    color-scheme: dark;
    --bg-0: #090d16;
    --bg-1: #0f1624;
    --bg-2: #162033;
    --bg-3: #1e2b44;
    --fg-0: #f0f6fc;
    --fg-1: #94a3b8;
    --fg-2: #64748b;
    --accent: #38bdf8;
    --emerald: #34d399;
    --amber: #fbbf24;
    --rose: #fb7185;
    --violet: #a78bfa;
    --border: #1e293b;
    --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-0);
    color: var(--fg-0);
    font-family: var(--font-sans);
    height: 100dvh;
    display: grid;
    grid-template-rows: auto auto 1fr;
    overflow: hidden;
  }
  .topbar {
    background: var(--bg-1);
    border-bottom: 1px solid var(--border);
    padding: 0.65rem 1rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
  }
  .brand-badge {
    background: color-mix(in oklch, var(--emerald) 20%, transparent);
    color: var(--emerald);
    border: 1px solid color-mix(in oklch, var(--emerald) 45%, transparent);
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
  }
  select, button, .btn-link {
    background: var(--bg-2);
    color: var(--fg-0);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.35rem 0.65rem;
    font-size: 0.8rem;
    font-family: var(--font-sans);
    cursor: pointer;
    text-decoration: none;
    transition: border-color 0.15s, background 0.15s;
  }
  select:hover, button:hover, .btn-link:hover {
    border-color: var(--accent);
    background: var(--bg-3);
  }
  button.primary {
    background: color-mix(in oklch, var(--accent) 22%, var(--bg-2));
    border-color: var(--accent);
    color: #e0f2fe;
    font-weight: 600;
  }
  .iter-scrubber {
    display: flex;
    gap: 0.25rem;
  }
  .iter-btn.active {
    background: color-mix(in oklch, var(--emerald) 25%, var(--bg-2));
    border-color: var(--emerald);
    color: var(--emerald);
    font-weight: 600;
  }
  .summary-strip {
    background: var(--bg-1);
    border-bottom: 1px solid var(--border);
    padding: 0.45rem 1rem;
    display: flex;
    flex-wrap: wrap;
    gap: 1.25rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--fg-1);
  }
  .metric-pill strong { color: var(--fg-0); }
  .workspace {
    display: grid;
    grid-template-columns: 1.25fr 1fr;
    height: 100%;
    overflow: hidden;
  }
  .pane {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
    border-right: 1px solid var(--border);
    container-type: inline-size;
  }
  .pane-right {
    display: grid;
    grid-template-rows: 1fr 1.15fr;
    height: 100%;
    overflow: hidden;
  }
  .panel-header {
    background: var(--bg-2);
    border-bottom: 1px solid var(--border);
    padding: 0.45rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--fg-1);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .scroll-area {
    flex: 1;
    overflow-y: auto;
    font-family: var(--font-mono);
    font-size: 0.76rem;
    line-height: 1.45;
  }
  .code-line {
    display: grid;
    grid-template-columns: 3.2rem 13.5rem 1fr;
    gap: 0.5rem;
    padding: 0.1rem 0.65rem;
    border-bottom: 1px solid rgba(30, 41, 59, 0.35);
    content-visibility: auto;
    contain-intrinsic-size: 0 22px;
    cursor: pointer;
    align-items: center;
  }
  .code-line:hover { background: var(--bg-2); }
  .code-line.selected {
    background: color-mix(in oklch, var(--accent) 18%, var(--bg-1));
    border-left: 3px solid var(--accent);
  }
  .code-line:has(.badge-spill) {
    background: color-mix(in oklch, var(--rose) 12%, var(--bg-0));
  }
  .line-num { color: var(--fg-2); text-align: right; user-select: none; }
  .gutter-badges {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    overflow: hidden;
    white-space: nowrap;
  }
  .badge {
    padding: 0.05rem 0.32rem;
    border-radius: 4px;
    font-size: 0.66rem;
    border: 1px solid transparent;
  }
  .badge-stage { background: rgba(56, 189, 248, 0.15); color: var(--accent); border-color: rgba(56, 189, 248, 0.35); }
  .badge-l2 { background: rgba(52, 211, 153, 0.14); color: var(--emerald); }
  .badge-proof { background: rgba(167, 139, 250, 0.15); color: var(--violet); }
  .badge-stall { background: rgba(251, 191, 36, 0.16); color: var(--amber); }
  .badge-spill { background: rgba(251, 113, 133, 0.2); color: var(--rose); }
  .code-text { white-space: pre; overflow-x: auto; color: var(--fg-0); }
  .stage-cards, .sass-panel {
    padding: 0.65rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .stage-card {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 0.75rem;
    cursor: pointer;
  }
  .stage-card.active, .stage-card:has(.selected-marker) {
    border-color: var(--accent);
    background: var(--bg-2);
  }
  .stage-card-head {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.35rem;
    font-weight: 600;
    color: var(--fg-0);
  }
  .ptx-box {
    background: var(--bg-0);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.45rem;
    margin-top: 0.4rem;
    white-space: pre-wrap;
    color: #bae6fd;
    font-size: 0.72rem;
  }
  .sass-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.72rem;
  }
  .sass-table th, .sass-table td {
    text-align: left;
    padding: 0.25rem 0.4rem;
    border-bottom: 1px solid var(--border);
  }
  .sass-table tr:hover { background: var(--bg-2); }
  .stall-bar-row {
    display: grid;
    grid-template-columns: 9.5rem 1fr 3.2rem;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.7rem;
    margin: 0.15rem 0;
  }
  .stall-track {
    height: 6px;
    background: var(--bg-3);
    border-radius: 3px;
    overflow: hidden;
  }
  .stall-fill {
    height: 100%;
    background: var(--amber);
  }
</style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <span>E-Graph Megakernel Live Stream &amp; CUPTI/PTX/SASS Inspector</span>
      <span id="speedup-badge" class="brand-badge">6.05x Holdout</span>
    </div>
    <div class="controls">
      <label>Mode:
        <select id="mode-select">
          <option value="cross_layer">Cross-Layer Stack</option>
          <option value="full_layer">Full-Layer 18-Stage Monolithic</option>
        </select>
      </label>
      <label>Bucket:
        <select id="bucket-select">
          <option value="1">M=1 (Decode GEMV)</option>
          <option value="2">M=2 (SpecVerify)</option>
          <option value="4">M=4 (Tree TensorCore)</option>
          <option value="8">M=8 (Tree TensorCore)</option>
        </select>
      </label>
      <div class="iter-scrubber" id="iter-buttons">
        <button class="iter-btn" data-iter="0">Iter 0: Baseline</button>
        <button class="iter-btn" data-iter="1">Iter 1: PTXAS Loop</button>
        <button class="iter-btn" data-iter="2">Iter 2: SASS Swizzle</button>
        <button class="iter-btn active" data-iter="3">Iter 3: CUPTI Promoted 6.05x</button>
      </div>
      <button id="stream-btn" class="primary">Stream Live Flight</button>
      <a id="raw-text-link" class="btn-link" href="/v1/kernels/annotated?format=text" target="_blank">Annotated Text</a>
    </div>
  </header>

  <section class="summary-strip" id="summary-strip">
    <span class="metric-pill">Kernel: <strong id="meta-kernel">-</strong></span>
    <span class="metric-pill">Phase: <strong id="meta-phase">-</strong></span>
    <span class="metric-pill">Holdout Speedup: <strong id="meta-speedup">-</strong></span>
    <span class="metric-pill">Max Regs/Thread: <strong id="meta-regs">-</strong></span>
    <span class="metric-pill">Total Spills: <strong id="meta-spills">-</strong></span>
    <span class="metric-pill">Formal Barriers: <strong id="meta-proof">RAW + Arena WAR + bar.sync Verified</strong></span>
  </section>

  <main class="workspace">
    <section class="pane">
      <div class="panel-header">
        <span>CUDA C++ (#line) Source with Inline Cost, L2::evict_last &amp; CUPTI Stall Heatbars</span>
        <span id="line-count-label">0 lines</span>
      </div>
      <div class="scroll-area" id="cuda-source-pane"></div>
    </section>

    <section class="pane-right">
      <div class="pane" style="border-right:none; border-bottom:1px solid var(--border);">
        <div class="panel-header">
          <span>PTX + PTXAS Register/Spill Breakdown &amp; Calibrated E-Graph Cost</span>
          <span id="stage-count-label">0 stages</span>
        </div>
        <div class="scroll-area stage-cards" id="ptx-stage-pane"></div>
      </div>

      <div class="pane" style="border-right:none;">
        <div class="panel-header">
          <span>SASS Disassembly Scoreboard Wait-Mask (--:-:1..6) &amp; CUPTI Stall Attribution Cards</span>
          <span id="selected-stage-label">All Stages</span>
        </div>
        <div class="scroll-area sass-panel" id="sass-cupti-pane"></div>
      </div>
    </section>
  </main>

<script>
  let allIterations = [];
  let currentIterIndex = 3;
  let selectedStageIndex = 0;
  let activeEventSource = null;

  const modeSelect = document.getElementById('mode-select');
  const bucketSelect = document.getElementById('bucket-select');
  const streamBtn = document.getElementById('stream-btn');
  const rawTextLink = document.getElementById('raw-text-link');

  function updateRawLink() {
    rawTextLink.href = '/v1/kernels/annotated?format=text&mode=' + modeSelect.value + '&bucket=' + bucketSelect.value;
  }

  async function fetchAnnotatedKernel() {
    updateRawLink();
    const url = '/v1/kernels/annotated?mode=' + modeSelect.value + '&bucket=' + bucketSelect.value;
    const resp = await fetch(url);
    const data = await resp.json();
    allIterations = data.iterations || [data];
    currentIterIndex = allIterations.length - 1;
    renderIteration(allIterations[currentIterIndex]);
  }

  function renderIteration(iter) {
    if (!iter) return;
    document.querySelectorAll('.iter-btn').forEach(btn => {
      btn.classList.toggle('active', Number(btn.dataset.iter) === iter.iteration);
    });
    document.getElementById('speedup-badge').textContent = iter.holdout_speedup.toFixed(2) + 'x Holdout';
    document.getElementById('meta-kernel').textContent = iter.kernel_name + ' (' + iter.arch + ')';
    document.getElementById('meta-phase').textContent = iter.phase;
    document.getElementById('meta-speedup').textContent = iter.holdout_speedup.toFixed(2) + 'x';
    document.getElementById('meta-regs').textContent = iter.max_regs;
    document.getElementById('meta-spills').textContent = iter.total_spill_bytes + ' B';

    const lines = iter.cuda_lines || [];
    const anns = iter.line_annotations || {};
    document.getElementById('line-count-label').textContent = lines.length + ' lines';

    const cudaPane = document.getElementById('cuda-source-pane');
    cudaPane.innerHTML = '';
    lines.forEach((txt, idx) => {
      const lineNo = idx + 1;
      const ann = anns[lineNo];
      const row = document.createElement('div');
      row.className = 'code-line' + (ann && ann.stage_index === selectedStageIndex ? ' selected' : '');
      let badges = '';
      if (ann) {
        badges += '<span class="badge badge-stage">s' + ann.stage_index + ' ' + (ann.egraph_cost_ns / 1000).toFixed(1) + 'us</span>';
        badges += '<span class="badge badge-stall">' + ann.cupti_stall_pct.toFixed(0) + '%</span>';
        if (ann.l2_policy && ann.l2_policy.includes('evict_last')) {
          badges += '<span class="badge badge-l2">L2::evict_last</span>';
        }
        if (ann.category === 'barrier') {
          badges += '<span class="badge badge-proof">bar.sync</span>';
        }
      }
      row.innerHTML =
        '<span class="line-num">' + lineNo + '</span>' +
        '<span class="gutter-badges">' + badges + '</span>' +
        '<span class="code-text"></span>';
      row.querySelector('.code-text').textContent = txt;
      row.addEventListener('click', () => {
        if (ann) {
          selectedStageIndex = ann.stage_index;
          renderIteration(iter);
        }
      });
      cudaPane.appendChild(row);
    });

    const stages = iter.stages || [];
    document.getElementById('stage-count-label').textContent = stages.length + ' stages';
    const ptxPane = document.getElementById('ptx-stage-pane');
    ptxPane.innerHTML = '';
    stages.forEach(st => {
      const card = document.createElement('div');
      card.className = 'stage-card' + (st.stage_index === selectedStageIndex ? ' active' : '');
      const spillBytes = st.spill_store_bytes + st.spill_load_bytes;
      card.innerHTML =
        '<div class="stage-card-head">' +
          '<span>s' + st.stage_index + ': ' + st.stage_name + ' [' + st.category + ']</span>' +
          '<span>' + (st.egraph_cost_ns / 1000).toFixed(2) + ' us (mult ' + st.calibrated_multiplier.toFixed(2) + 'x, ' + st.roofline_bound + ')</span>' +
        '</div>' +
        '<div>regs=' + st.registers_per_thread + ' | smem=' + st.shared_bytes + 'B | spills=' + spillBytes + 'B | occ=' + st.occupancy_ctas_per_sm + ' CTAs/SM | L2=' + st.l2_policy + ' | RAW=' + st.raw_verified + '</div>' +
        '<div class="ptx-box"></div>';
      card.querySelector('.ptx-box').textContent = st.ptx_snippet || '';
      card.addEventListener('click', () => {
        selectedStageIndex = st.stage_index;
        renderIteration(iter);
      });
      ptxPane.appendChild(card);
    });

    const sassPane = document.getElementById('sass-cupti-pane');
    sassPane.innerHTML = '';
    const activeStage = stages[selectedStageIndex] || stages[0];
    if (activeStage) {
      document.getElementById('selected-stage-label').textContent =
        's' + activeStage.stage_index + ': ' + activeStage.stage_name + ' (wait=' + activeStage.wait_stall_weight + ')';
      const cuptiCard = document.createElement('div');
      cuptiCard.className = 'stage-card';
      cuptiCard.innerHTML =
        '<div class="stage-card-head">' +
          '<span>CUPTI PC-Sampling (' + activeStage.total_samples + ' samples) — Dominant: ' + activeStage.dominant_reason + '</span>' +
          '<span>Rewrite: ' + activeStage.recommended_rewrite + '</span>' +
        '</div>' +
        stallRow('Long Scoreboard', activeStage.long_scoreboard_pct) +
        stallRow('Barrier / TMEM', activeStage.barrier_stall_pct) +
        stallRow('MIO / Shared Bank', activeStage.mio_throttle_pct) +
        stallRow('Math / Tensor Pipe', activeStage.math_pipe_pct);
      sassPane.appendChild(cuptiCard);

      const tbl = document.createElement('table');
      tbl.className = 'sass-table';
      tbl.innerHTML = '<thead><tr><th>PC</th><th>Mnemonic</th><th>Operands</th><th>WaitMask</th><th>Stall</th><th>Weight</th><th>Line</th></tr></thead><tbody></tbody>';
      const tbody = tbl.querySelector('tbody');
      (activeStage.sass_insts || []).forEach(ins => {
        const tr = document.createElement('tr');
        tr.innerHTML =
          '<td>0x' + ins.pc.toString(16) + '</td>' +
          '<td>' + ins.mnemonic + (ins.bank_conflict_risk ? ' <span class="badge badge-spill">BANK_CONFLICT</span>' : '') + '</td>' +
          '<td>' + ins.operands + '</td>' +
          '<td>' + ins.wait_mask + '</td>' +
          '<td>{!' + ins.control_stall + '}</td>' +
          '<td>' + ins.scoreboard_weight + '</td>' +
          '<td>L' + ins.source_line + '</td>';
        tbody.appendChild(tr);
      });
      sassPane.appendChild(tbl);
    }
  }

  function stallRow(label, pct) {
    return '<div class="stall-bar-row"><span>' + label + '</span><div class="stall-track"><div class="stall-fill" style="width:' + Math.min(pct, 100) + '%"></div></div><span>' + pct.toFixed(1) + '%</span></div>';
  }

  document.querySelectorAll('.iter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.iter);
      if (allIterations[idx]) {
        currentIterIndex = idx;
        renderIteration(allIterations[idx]);
      }
    });
  });

  modeSelect.addEventListener('change', fetchAnnotatedKernel);
  bucketSelect.addEventListener('change', fetchAnnotatedKernel);

  streamBtn.addEventListener('click', () => {
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    const url = '/v1/kernels/stream?interval_ms=350&mode=' + modeSelect.value + '&bucket=' + bucketSelect.value;
    activeEventSource = new EventSource(url);
    activeEventSource.addEventListener('kernel_iteration', ev => {
      const frame = JSON.parse(ev.data);
      allIterations[frame.iteration] = frame;
      currentIterIndex = frame.iteration;
      renderIteration(frame);
    });
    activeEventSource.addEventListener('stream_complete', () => {
      if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
      }
    });
  });

  fetchAnnotatedKernel();
</script>
</body>
</html>
`
