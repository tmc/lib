package main

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

func testFlightModelConfig() *ModelConfig {
	return &ModelConfig{
		NumHiddenLayers:   2,
		HiddenSize:        4096,
		IntermediateSize:  14336,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8,
		RmsNormEps:        1e-5,
		VocabSize:         128256,
		MaxSeqLen:         2048,
	}
}

func testReasonInventory() cuptipcsampling.ReasonInventory {
	return cuptipcsampling.ReasonInventory{
		Reasons: []cuptipcsampling.StallReason{
			{Index: 1, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard"},
			{Index: 2, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued"},
			{Index: 3, Name: "smsp__pcsamp_warps_issue_stalled_mio_throttle"},
			{Index: 4, Name: "smsp__pcsamp_warps_issue_stalled_barrier"},
			{Index: 5, Name: "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"},
			{Index: 6, Name: "smsp__pcsamp_warps_issue_stalled_wait_tmem"},
		},
	}
}

func TestEGraphFlightStageAttributionAndRanking(t *testing.T) {
	cfg := testFlightModelConfig()
	_, post, err := buildLayerPlans(cfg, false, true, "llama", 1, hopper.BF16, "sm_121a", true)
	if err != nil {
		t.Fatalf("buildLayerPlans decode post: %v", err)
	}
	loop, err := newEGraphFlightLoop(post, "", nil, nil)
	if err != nil {
		t.Fatalf("newEGraphFlightLoop: %v", err)
	}
	if len(loop.stages) != 3 {
		t.Fatalf("got %d megakernel stages, want 3", len(loop.stages))
	}
	for i, st := range loop.stages {
		if st.StageIndex != i || st.StageName == "" || st.Region != st.StageName {
			t.Fatalf("stage %d attribution = %+v", i, st)
		}
		wantDirective := fmt.Sprintf("#line 1 %q", st.File)
		if !strings.Contains(loop.source, wantDirective) {
			t.Fatalf("megakernel source missing %s", wantDirective)
		}
		if st.PCRange.End <= st.PCRange.Start || st.PCRange.Label != st.StageName {
			t.Fatalf("stage %d invalid PCRange %+v", i, st.PCRange)
		}
	}

	// Also verify explicit PCFilePoint coalescing via StageFileLabels.
	points := []autotune.PCFilePoint{
		{Offset: 0x100, Size: 32, File: loop.stages[0].File},
		{Offset: 0x120, Size: 32, File: loop.stages[0].File},
		{Offset: 0x200, Size: 64, File: loop.stages[1].File},
		{Offset: 0x300, Size: 48, File: loop.stages[2].File},
	}
	explicitStages, explicitRanges, err := StageAttributionMap(loop.mega, points)
	if err != nil {
		t.Fatalf("StageAttributionMap with PCFilePoints: %v", err)
	}
	if len(explicitRanges) != 3 || explicitStages[0].PCRange != (autotune.PCRange{Start: 0x100, End: 0x140, Label: loop.stages[0].StageName}) {
		t.Fatalf("unexpected coalesced ranges: stages=%+v ranges=%+v", explicitStages, explicitRanges)
	}

	// Attribute a synthetic CUPTI PC-sampling batch across the megakernel stages.
	inv := testReasonInventory()
	pc0 := loop.stages[0].PCRange.Start
	pc1 := loop.stages[1].PCRange.Start
	pc2 := loop.stages[2].PCRange.Start
	batches := []cuptipcsampling.Batch{{
		PCs: []cuptipcsampling.PC{
			{
				Offset: pc0,
				Name:   loop.name,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 1, Samples: 20},
					{Index: 2, Samples: 999}, // _not_issued must be ignored
				},
			},
			{
				Offset: pc1,
				Name:   loop.name,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 1, Samples: 140}, // LongScoreboard on stage 1 (gate_up)
					{Index: 5, Samples: 20},
				},
			},
			{
				Offset: pc2,
				Name:   loop.name,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 3, Samples: 50},
				},
			},
			{
				Offset: 0x8, // outside stage ranges -> autotune.Unattributed
				Name:   loop.name,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 4, Samples: 7},
				},
			},
		},
	}}

	total, byRegion, err := HistogramsByRegionFromBatch(batches, inv, loop.name, loop.ranges)
	if err != nil {
		t.Fatalf("HistogramsByRegionFromBatch: %v", err)
	}
	if total.TotalSamples != 20+160+50+7 {
		t.Fatalf("total.TotalSamples = %d, want %d", total.TotalSamples, 20+160+50+7)
	}
	if got := byRegion[autotune.Unattributed].WaitMBarrier; got != 7 {
		t.Fatalf("unattributed WaitMBarrier = %d, want 7", got)
	}

	ranks, err := loop.AttributeAndRankStages(batches, inv)
	if err != nil {
		t.Fatalf("AttributeAndRankStages: %v", err)
	}
	if len(ranks) != 3 {
		t.Fatalf("len(ranks) = %d, want 3", len(ranks))
	}
	if ranks[0].StageIndex != 1 || ranks[0].StallSample.TotalSamples != 160 || ranks[0].StallSample.LongScoreboard != 140 {
		t.Fatalf("hottest stage = %+v, want stage 1 with 160 total / 140 LongScoreboard", ranks[0])
	}
	if ranks[1].StageIndex != 2 || ranks[1].StallSample.TotalSamples != 50 {
		t.Fatalf("second stage = %+v, want stage 2 with 50 total", ranks[1])
	}
	if ranks[2].StageIndex != 0 || ranks[2].StallSample.TotalSamples != 20 {
		t.Fatalf("third stage = %+v, want stage 0 with 20 total", ranks[2])
	}
}

func TestEGraphFlightGuidedCandidateGeneration(t *testing.T) {
	cfg := testFlightModelConfig()
	// Build a prefill post plan (m=128) which contains 3 TileGEMMs and 2 RowKernels.
	_, pfPost, err := buildLayerPlans(cfg, false, true, "llama_pf", 128, hopper.BF16, "sm_121a", false)
	if err != nil {
		t.Fatalf("buildLayerPlans prefill post: %v", err)
	}
	loop, err := newEGraphFlightLoop(pfPost, "", nil, nil)
	if err != nil {
		t.Fatalf("newEGraphFlightLoop prefill post: %v", err)
	}

	// Find the first TileGEMM stage in the prefill post megakernel.
	tileStageIdx := -1
	for i, st := range loop.mega.Stages {
		if st.Kernel.Tile != nil {
			tileStageIdx = i
			break
		}
	}
	if tileStageIdx < 0 {
		t.Fatal("no TileGEMM stage found in prefill post megakernel")
	}
	targetStage := loop.stages[tileStageIdx]

	// Inject high MIOThrottle stall samples on the target TileGEMM stage.
	inv := testReasonInventory()
	batches := []cuptipcsampling.Batch{{
		PCs: []cuptipcsampling.PC{{
			Offset: targetStage.PCRange.Start,
			Name:   loop.name,
			Reasons: []cuptipcsampling.StallSamples{
				{Index: 3, Samples: 180}, // MIOThrottle (75% share >= 0.15)
				{Index: 5, Samples: 60},  // MathThrottle
			},
		}},
	}}
	ranks, err := loop.AttributeAndRankStages(batches, inv)
	if err != nil {
		t.Fatalf("AttributeAndRankStages: %v", err)
	}
	if ranks[0].StageIndex != tileStageIdx {
		t.Fatalf("hottest stage index = %d, want %d", ranks[0].StageIndex, tileStageIdx)
	}

	// Provide a causal observation (register spills from ptxas) alongside the stall samples.
	causalObs := qualify.ObservationFromCountersAndPTXAS(
		ptxasinfo.Function{
			Name:        targetStage.StageName,
			Registers:   196,
			SpillStores: 16,
			SpillLoads:  16,
		},
		qualify.CounterMetrics{DurationNs: 50000},
		targetStage.StageIndex+1,
		132,
	)
	if causalObs.SharedBankConflict != 0 {
		t.Fatalf("causal observation polluted with SharedBankConflict = %v", causalObs.SharedBankConflict)
	}

	hottest, proposals, err := loop.ProposeHottestStageCandidates(ranks, map[string]qualify.KernelObservation{
		targetStage.StageName: causalObs,
	})
	if err != nil {
		t.Fatalf("ProposeHottestStageCandidates: %v", err)
	}
	if hottest.StageName != targetStage.StageName {
		t.Fatalf("hottest stage = %q, want %q", hottest.StageName, targetStage.StageName)
	}
	if len(proposals) < 2 {
		t.Fatalf("got %d candidate proposals, want at least 2 (1 causal + >=1 stall hypothesis): %+v", len(proposals), proposals)
	}
	if !proposals[0].Causal {
		t.Fatalf("first proposal should be Causal=true, got %+v", proposals[0])
	}
	foundStallBankConflict := false
	for _, p := range proposals[1:] {
		if p.Causal {
			t.Fatalf("subsequent stall-hypothesis proposal marked Causal=true: %+v", p)
		}
		if p.Hypothesis == qualify.BottleneckBankConflict && p.Step.Tile != nil && p.Step.Tile.Swizzle128B {
			foundStallBankConflict = true
		}
	}
	if !foundStallBankConflict {
		t.Fatalf("expected non-causal BottleneckBankConflict proposal enabling Swizzle128B, got %+v", proposals)
	}

	//Also verify decode SkinnyGEMM stall-driven candidate generation (LongScoreboard -> Prefetch)
	// and lowering the rewritten stage back into a persistent megakernel.
	_, decPost, err := buildLayerPlans(cfg, false, true, "llama", 1, hopper.BF16, "sm_121a", true)
	if err != nil {
		t.Fatalf("buildLayerPlans decode post: %v", err)
	}
	decLoop, err := newEGraphFlightLoop(decPost, "", nil, nil)
	if err != nil {
		t.Fatalf("newEGraphFlightLoop decode post: %v", err)
	}
	decBatches := []cuptipcsampling.Batch{{
		PCs: []cuptipcsampling.PC{{
			Offset: decLoop.stages[1].PCRange.Start,
			Name:   decLoop.name,
			Reasons: []cuptipcsampling.StallSamples{
				{Index: 1, Samples: 200}, // LongScoreboard -> BottleneckMemoryBoundKV -> Prefetch
			},
		}},
	}}
	decRanks, err := decLoop.AttributeAndRankStages(decBatches, inv)
	if err != nil {
		t.Fatalf("decLoop.AttributeAndRankStages: %v", err)
	}
	decHottest, decProposals, err := decLoop.ProposeHottestStageCandidates(decRanks, nil)
	if err != nil {
		t.Fatalf("decLoop.ProposeHottestStageCandidates: %v", err)
	}
	if len(decProposals) == 0 || decProposals[0].Causal || decProposals[0].Step.GEMM == nil || decProposals[0].Step.GEMM.Prefetch <= 0 {
		t.Fatalf("expected stall-driven SkinnyGEMM Prefetch proposal, got %+v", decProposals)
	}
	_, candMega, candSrc, err := decLoop.CandidateMegakernel(decHottest.StageIndex, decProposals[0].Step)
	if err != nil {
		t.Fatalf("CandidateMegakernel: %v", err)
	}
	if len(candMega.Stages) != len(decLoop.mega.Stages) || candSrc == "" || candSrc == decLoop.source {
		t.Fatalf("expected modified megakernel source for rewritten stage")
	}
}

func TestEGraphFlightTwoStageHoldoutGateAndABBABridge(t *testing.T) {
	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("llama-infer:sm_121a:llama_post_mega")),
		Admission: sha256.Sum256([]byte("flight-gate-admission-v1")),
	}
	refSrc := "// reference megakernel v1"
	candSrc := "// candidate megakernel v2"
	gateCfg := HoldoutGateConfig{
		Identity:                id,
		ReferenceSource:         refSrc,
		CandidateSource:         candSrc,
		ExpectedReferenceDigest: sha256.Sum256([]byte(refSrc)),
		ExpectedCandidateDigest: sha256.Sum256([]byte(candSrc)),
		RequireAANull:           true,
		PairTimeUnit:            time.Microsecond,
	}
	verify := func(p qualify.MeasuredPair) error {
		if p.Identity != id {
			return errors.New("unexpected identity")
		}
		return nil
	}

	screenPair := qualify.MeasuredPair{
		Identity:  id,
		Reference: "v1",
		Candidate: "v2",
		Blocks: []pairedtiming.Block{
			{A1: 100, B1: 90, B2: 90, A2: 100},
			{A1: 100, B1: 89, B2: 91, A2: 100},
			{A1: 100, B1: 90, B2: 90, A2: 100},
		},
	}
	aaNullBlocks := []pairedtiming.Block{
		{A1: 100, B1: 100, B2: 100, A2: 100},
		{A1: 100, B1: 101, B2: 99, A2: 100},
		{A1: 100, B1: 100, B2: 100, A2: 100},
	}
	tokens := []int{128000, 791, 6864, 315, 9822, 13}

	// 1. 3-block screen alone (p=0.125 without >=6 holdout blocks) MUST NOT promote.
	screenOnlyEv := qualify.HoldoutCandidateEvidence{
		Screen:          screenPair,
		HoldoutBlocks:   screenPair.Blocks, // only 3 blocks (< 6 required)
		AANullBlocks:    aaNullBlocks,
		ReferenceTokens: tokens,
		CandidateTokens: tokens,
	}
	rScreenOnly, err := SelectMeasuredWithHoldout(gateCfg, screenOnlyEv, verify)
	if err != nil {
		t.Fatalf("SelectMeasuredWithHoldout (3-block holdout): %v", err)
	}
	if rScreenOnly.Promotion.Status != "REJECTED_HOLDOUT" {
		t.Fatalf("3-block holdout Status = %q, want REJECTED_HOLDOUT", rScreenOnly.Promotion.Status)
	}

	// 2. Full 6-block held-out ABBA + A/A null + exact token-ID match promotes and bridges to EvaluateABBAPairs.
	holdoutBlocks := []pairedtiming.Block{
		{A1: 100, B1: 90, B2: 91, A2: 100},
		{A1: 100, B1: 89, B2: 90, A2: 100},
		{A1: 100, B1: 90, B2: 90, A2: 100},
		{A1: 101, B1: 90, B2: 91, A2: 100},
		{A1: 100, B1: 89, B2: 90, A2: 101},
		{A1: 100, B1: 90, B2: 90, A2: 100},
	}
	fullEv := qualify.HoldoutCandidateEvidence{
		Screen:          screenPair,
		HoldoutBlocks:   holdoutBlocks,
		AANullBlocks:    aaNullBlocks,
		ReferenceTokens: tokens,
		CandidateTokens: tokens,
	}
	rPromoted, err := SelectMeasuredWithHoldout(gateCfg, fullEv, verify)
	if err != nil {
		t.Fatalf("SelectMeasuredWithHoldout (6-block holdout): %v", err)
	}
	if rPromoted.Promotion.Status != "EXACT_HOLDOUT_PROMOTED" || rPromoted.Promotion.CandidateState != "PROMOTED_HOLDOUT_VERIFIED" {
		t.Fatalf("expected promotion, got %+v", rPromoted.Promotion)
	}
	if rPromoted.ABBASummary == nil || !rPromoted.ABBASummary.Pass || rPromoted.ABBASummary.Pairs != 12 {
		t.Fatalf("expected passing 12-pair ABBASummary from 6 holdout blocks, got %+v", rPromoted.ABBASummary)
	}
}

func TestEGraphFlightTokenIDDivergenceRejection(t *testing.T) {
	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("llama-infer:sm_121a:token-gate")),
		Admission: sha256.Sum256([]byte("flight-gate-admission-v1")),
	}
	gateCfg := HoldoutGateConfig{Identity: id}
	verify := func(qualify.MeasuredPair) error { return nil }

	screenPair := qualify.MeasuredPair{
		Identity:  id,
		Reference: "v1",
		Candidate: "v2",
		Blocks: []pairedtiming.Block{
			{A1: 100, B1: 88, B2: 88, A2: 100},
			{A1: 100, B1: 88, B2: 88, A2: 100},
			{A1: 100, B1: 88, B2: 88, A2: 100},
		},
	}
	holdoutBlocks := []pairedtiming.Block{
		{A1: 100, B1: 88, B2: 88, A2: 100},
		{A1: 100, B1: 88, B2: 88, A2: 100},
		{A1: 100, B1: 88, B2: 88, A2: 100},
		{A1: 100, B1: 88, B2: 88, A2: 100},
		{A1: 100, B1: 88, B2: 88, A2: 100},
		{A1: 100, B1: 88, B2: 88, A2: 100},
	}

	// Even when LogitMaxAbs is tiny (1e-4) and all timing blocks win 6/6,
	// a single greedy token-ID flip blocks promotion.
	ev := qualify.HoldoutCandidateEvidence{
		Screen:          screenPair,
		HoldoutBlocks:   holdoutBlocks,
		ReferenceTokens: []int{101, 202, 303, 404, 505},
		CandidateTokens: []int{101, 202, 999, 404, 505},
		LogitMaxAbs:     1e-4,
	}
	if err := VerifyTokenSequence(ev.ReferenceTokens, ev.CandidateTokens, ev.LogitMaxAbs); err == nil {
		t.Fatal("VerifyTokenSequence accepted diverging token sequence")
	}
	receipt, err := SelectMeasuredWithHoldout(gateCfg, ev, verify)
	if err != nil {
		t.Fatalf("SelectMeasuredWithHoldout returned unexpected error: %v", err)
	}
	if receipt.Promotion.Status != "REJECTED_TOKEN_MISMATCH" || receipt.Promotion.TokensMatched {
		t.Fatalf("expected REJECTED_TOKEN_MISMATCH, got %+v", receipt.Promotion)
	}
	if receipt.Promotion.Selected == nil || receipt.Promotion.Selected.Name != "v1" {
		t.Fatalf("expected Selected=v1 on token mismatch, got %+v", receipt.Promotion.Selected)
	}
}

type flightTestKernel struct {
	raw    driver.Function
	closed int
}

func (k *flightTestKernel) Raw() driver.Function                   { return k.raw }
func (k *flightTestKernel) Launch(cudalaunch.Config, ...any) error { return nil }
func (k *flightTestKernel) Wait(context.Context, *cuda.Stream) error {
	return nil
}
func (k *flightTestKernel) Close() error {
	k.closed++
	return nil
}

type flightTestLoader struct {
	nextFn driver.Function
	loads  int
}

func (l *flightTestLoader) Compile(_ context.Context, src cudahotswap.Source) (*cudahotswap.Compilation, error) {
	if src.Entry == "" || src.Code == "" {
		return nil, errors.New("empty source or entry")
	}
	ptx := fmt.Sprintf(".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry %s(\n\t.param .u64 p0\n)\n{\n\tret;\n}\n", src.Entry)
	sum := sha256.Sum256([]byte(src.Code))
	return &cudahotswap.Compilation{
		PTX:   ptx,
		Key:   fmt.Sprintf("key-%x", sum[:6]),
		Entry: src.Entry,
	}, nil
}

func (l *flightTestLoader) Load(context.Context, *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	l.loads++
	l.nextFn++
	return &flightTestKernel{raw: l.nextFn}, nil
}

type noopCollector struct{}

func (noopCollector) SampleKernel(context.Context, autotune.Variant, int) (autotune.ProfileObservation, autotune.StallHistogram, error) {
	return autotune.ProfileObservation{MedianUS: 10}, autotune.StallHistogram{}, nil
}

type funcCloser func() error

func (f funcCloser) Close() error { return f() }

func TestEGraphFlightHotSwapAndMultiStepRollback(t *testing.T) {
	lockParent := t.TempDir()
	if err := os.MkdirAll(lockParent, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GPU_LOCKDIR", filepath.Join(lockParent, "gpu.lock"))

	cfg := testFlightModelConfig()
	_, post, err := buildLayerPlans(cfg, false, true, "llama", 1, hopper.BF16, "sm_121a", true)
	if err != nil {
		t.Fatalf("buildLayerPlans: %v", err)
	}
	loader := &flightTestLoader{}
	svc, err := cudahotswap.NewService(loader)
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	ctx := context.Background()
	t.Cleanup(func() {
		if err := svc.Close(ctx); err != nil {
			t.Errorf("svc.Close: %v", err)
		}
	})

	loop, err := newEGraphFlightLoop(post, "", svc, nil)
	if err != nil {
		t.Fatalf("newEGraphFlightLoop: %v", err)
	}
	if loop.launchCfg.Block != driver.Dims(256, 1, 1) {
		t.Fatalf("launchCfg.Block = %+v, want (256, 1, 1)", loop.launchCfg.Block)
	}

	// Register baseline (v1).
	v1State, err := loop.RegisterBaseline(ctx, loop.source)
	if err != nil {
		t.Fatalf("RegisterBaseline: %v", err)
	}
	if v1State.Version != 1 {
		t.Fatalf("v1State.Version = %d, want 1", v1State.Version)
	}

	// Configure isolated profiling epochs and verify an open epoch blocks PromoteCandidate.
	epochsOpened, epochsClosed := 0, 0
	loop.SetEpochConfig(EpochConfig{
		NewEpochCollector: func() (autotune.LiveCollector, io.Closer, error) {
			epochsOpened++
			return noopCollector{}, funcCloser(func() error {
				epochsClosed++
				return nil
			}), nil
		},
	})

	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("llama-infer:sm_121a:hotswap")),
		Admission: sha256.Sum256([]byte("flight-hotswap-admission")),
	}
	gateCfg := HoldoutGateConfig{Identity: id}
	verify := func(qualify.MeasuredPair) error { return nil }
	goodEv := qualify.HoldoutCandidateEvidence{
		Screen: qualify.MeasuredPair{
			Identity:  id,
			Reference: "v1",
			Candidate: "v2",
			Blocks: []pairedtiming.Block{
				{A1: 100, B1: 88, B2: 88, A2: 100},
				{A1: 100, B1: 88, B2: 88, A2: 100},
				{A1: 100, B1: 88, B2: 88, A2: 100},
			},
		},
		HoldoutBlocks: []pairedtiming.Block{
			{A1: 100, B1: 88, B2: 88, A2: 100},
			{A1: 100, B1: 88, B2: 88, A2: 100},
			{A1: 100, B1: 88, B2: 88, A2: 100},
			{A1: 100, B1: 88, B2: 88, A2: 100},
			{A1: 100, B1: 88, B2: 88, A2: 100},
			{A1: 100, B1: 88, B2: 88, A2: 100},
		},
		ReferenceTokens: []int{1, 2, 3, 4},
		CandidateTokens: []int{1, 2, 3, 4},
	}

	if err := loop.RunIsolatedEpoch(ctx, func(ctx context.Context, _ autotune.LiveCollector) error {
		if _, _, swapErr := loop.PromoteCandidate(ctx, "v2-during-epoch", loop.source+"\n// v2", gateCfg, goodEv, verify, nil); swapErr == nil {
			t.Fatal("PromoteCandidate succeeded while profiling epoch was active")
		}
		return nil
	}); err != nil {
		t.Fatalf("RunIsolatedEpoch: %v", err)
	}
	if epochsOpened != 1 || epochsClosed != 1 || loop.activeEpochs != 0 {
		t.Fatalf("epoch lifecycle mismatch: opened=%d closed=%d active=%d", epochsOpened, epochsClosed, loop.activeEpochs)
	}

	// Promote v2 after epoch closes.
	resV2, _, err := loop.PromoteCandidate(ctx, "v2", loop.source+"\n// v2", gateCfg, goodEv, verify, nil)
	if err != nil {
		t.Fatalf("PromoteCandidate v2: %v", err)
	}
	v2Version := resV2.Selected.Version
	if v2Version <= v1State.Version || loop.CurrentState().Version != v2Version {
		t.Fatalf("after v2 promotion: got version %d (v1=%d)", v2Version, v1State.Version)
	}

	// Promote v3 on top of v2.
	resV3, _, err := loop.PromoteCandidate(ctx, "v3", loop.source+"\n// v3", gateCfg, goodEv, verify, nil)
	if err != nil {
		t.Fatalf("PromoteCandidate v3: %v", err)
	}
	v3Version := resV3.Selected.Version
	if v3Version <= v2Version || loop.CurrentState().Version != v3Version {
		t.Fatalf("after v3 promotion: got version %d (v2=%d)", v3Version, v2Version)
	}

	// Verify version history is [v1, v2, v3].
	hist := loop.VersionHistory()
	if len(hist) != 3 || hist[0] != v1State.Version || hist[1] != v2Version || hist[2] != v3Version {
		t.Fatalf("VersionHistory = %v, want [%d %d %d]", hist, v1State.Version, v2Version, v3Version)
	}

	// Multi-step rollback 1: v3 -> RestorePrevious MUST restore v2 (NOT v1).
	restoredV2, err := loop.RestorePrevious(ctx, "rollback-to-v2")
	if err != nil {
		t.Fatalf("RestorePrevious (v3 -> v2): %v", err)
	}
	if restoredV2.Version != v2Version || loop.CurrentState().Version != v2Version {
		t.Fatalf("after first RestorePrevious: got version %d, want v2 (%d), not v1 (%d)", restoredV2.Version, v2Version, v1State.Version)
	}

	// Also verify that if a new candidate v4 fails its post-swap Checker while on v2,
	// cudahotswap.Service rolls back to v2 (not v1).
	checkFailErr := errors.New("post-swap latency regression")
	resV4, _, err := loop.PromoteCandidate(ctx, "v4-bad", loop.source+"\n// v4", gateCfg, goodEv, verify, func(context.Context, cudahotswap.State) error {
		return checkFailErr
	})
	if !errors.Is(err, checkFailErr) {
		t.Fatalf("PromoteCandidate v4 err = %v, want %v", err, checkFailErr)
	}
	if resV4 == nil || !resV4.RolledBack || resV4.Selected.Version != v2Version || loop.CurrentState().Version != v2Version {
		t.Fatalf("after v4 Checker failure: res=%+v current=%+v, want rollback to v2 (%d)", resV4, loop.CurrentState(), v2Version)
	}

	// Multi-step rollback 2: v2 -> RestorePrevious restores v1.
	restoredV1, err := loop.RestorePrevious(ctx, "rollback-to-v1")
	if err != nil {
		t.Fatalf("RestorePrevious (v2 -> v1): %v", err)
	}
	if restoredV1.Version != v1State.Version || loop.CurrentState().Version != v1State.Version {
		t.Fatalf("after second RestorePrevious: got version %d, want v1 (%d)", restoredV1.Version, v1State.Version)
	}
}
