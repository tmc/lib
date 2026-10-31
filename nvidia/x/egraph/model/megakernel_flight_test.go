package model

import (
	"context"
	"crypto/sha256"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

func testMegakernelPlan(t *testing.T) (*hopper.Plan, hopper.Megakernel) {
	t.Helper()
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.BF16, 2, 256)
	w := b.Input("w", hopper.BF16, 256, 256)
	qkv := b.MatMul(x, w)
	q := b.Reshape(b.Slice(qkv, 0, 128), 2, 2, 64)
	ss := b.ReduceSum(b.Map(hopper.MustLambda("(mul $0 $0)"), q))
	norm := b.Map(hopper.MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), q, ss)
	gate := b.Reshape(b.Slice(qkv, 128, 128), 2, 2, 64)
	root := b.Map(hopper.MustLambda("(mul $0 (sigmoid $1))"), norm, gate)

	expr, err := hopper.Optimize(b.G, root)
	if err != nil {
		t.Fatalf("hopper.Optimize: %v", err)
	}
	basePlan, err := hopper.Compile(expr, "decode_block", hopper.BF16)
	if err != nil {
		t.Fatalf("hopper.Compile: %v", err)
	}
	if len(basePlan.Kernels) != 2 || basePlan.Kernels[0].GEMM == nil || basePlan.Kernels[1].Row == nil {
		t.Fatalf("unexpected compiled kernels: %+v", basePlan.Kernels)
	}
	basePlan.Kernels[0].GEMM.Name = "k_proj"
	basePlan.Kernels[1].Row.Name = "rms_norm"

	qProj := &hopper.TileGEMM{
		Name:        "q_proj",
		N:           128,
		K:           128,
		BM:          128,
		BN:          128,
		Stages:      3,
		WGMMA:       true,
		Swizzle128B: false,
		Weight:      hopper.WeightBF16,
		Out:         hopper.BF16,
	}

	plan := &hopper.Plan{
		Name: "decode_block",
		Inputs: append(append([]hopper.Tensor(nil), basePlan.Inputs...),
			hopper.Tensor{Name: "x_tile", DType: hopper.BF16, Shape: []int{16, 128}},
			hopper.Tensor{Name: "wq", DType: hopper.BF16, Shape: []int{128, 128}},
		),
		Temps: append(append([]hopper.Tensor(nil), basePlan.Temps...),
			hopper.Tensor{Name: basePlan.Output.Name, DType: basePlan.Output.DType, Shape: basePlan.Output.Shape},
		),
		Output: hopper.Tensor{Name: "q_out", DType: hopper.BF16, Shape: []int{16, 128}},
		Kernels: []hopper.PlanKernel{
			basePlan.Kernels[1],
			{Tile: qProj, M: 16, Args: []string{"x_tile", "wq", "q_out"}},
			basePlan.Kernels[0],
		},
	}
	// Ensure stage order is [0: k_proj (GEMM), 1: q_proj (Tile), 2: rms_norm (Row)]
	// so all tensor dependencies are satisfied in order.
	plan.Kernels = []hopper.PlanKernel{
		basePlan.Kernels[0],
		{Tile: qProj, M: 16, Args: []string{"x_tile", "wq", "q_out"}},
		basePlan.Kernels[1],
	}
	plan.Temps = append(append([]hopper.Tensor(nil), basePlan.Temps...),
		hopper.Tensor{Name: "q_out", DType: hopper.BF16, Shape: []int{16, 128}},
	)
	plan.Output = basePlan.Output

	mega, err := plan.Megakernel("decode_block_mega")
	if err != nil {
		t.Fatalf("plan.Megakernel: %v", err)
	}
	return plan, mega
}

func TestMegakernelFlightStageLinePCMapAndAttribution(t *testing.T) {
	plan, mega := testMegakernelPlan(t)
	_, spans, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("mega.LineInfoSource: %v", err)
	}
	if len(spans) != 3 {
		t.Fatalf("len(spans) = %d, want 3", len(spans))
	}

	lineToPC := map[int][2]uint64{
		spans[0].StartLine: {0x1000, 0x1080},
		spans[0].EndLine:   {0x1080, 0x1100},
		spans[1].StartLine: {0x2000, 0x2200},
		spans[1].EndLine:   {0x2200, 0x2400},
		spans[2].StartLine: {0x3000, 0x3100},
	}
	amap, err := StageLinePCMap(mega, lineToPC)
	if err != nil {
		t.Fatalf("StageLinePCMap: %v", err)
	}
	if amap.Kernel != mega.Name {
		t.Fatalf("amap.Kernel = %q, want %q", amap.Kernel, mega.Name)
	}
	if len(amap.Ranges) != 3 {
		t.Fatalf("len(amap.Ranges) = %d, want 3", len(amap.Ranges))
	}

	// Verify overlapping PC ranges across stages are rejected.
	overlapMap := map[int][2]uint64{
		spans[0].StartLine: {0x1000, 0x2100},
		spans[1].StartLine: {0x2000, 0x2400},
	}
	if _, err := StageLinePCMap(mega, overlapMap); err == nil {
		t.Fatal("expected StageLinePCMap to reject overlapping stage PC ranges")
	}

	// Construct a synthetic PC-sampling batch where stage 1 (q_proj) is hottest
	// with WaitTMEM + MIOThrottle stalls, stage 0 (k_proj) is second with
	// LongScoreboard stalls, and stage 2 (rms_norm) is coldest.
	batch := cuptipcsampling.Batch{
		PCs: []cuptipcsampling.PC{
			{
				Name:    mega.Name,
				Offset:  0x1040,
				Reasons: []cuptipcsampling.StallSamples{{Index: 5, Samples: 60}},
			},
			{
				Name:   mega.Name,
				Offset: 0x2100,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 1, Samples: 120},
					{Index: 3, Samples: 80},
				},
			},
			{
				Name:    mega.Name,
				Offset:  0x3050,
				Reasons: []cuptipcsampling.StallSamples{{Index: 2, Samples: 10}},
			},
		},
	}

	profiles, err := AttributeMegakernelBatch(plan, mega, amap, batch)
	if err != nil {
		t.Fatalf("AttributeMegakernelBatch: %v", err)
	}
	if len(profiles) != 3 {
		t.Fatalf("len(profiles) = %d, want 3", len(profiles))
	}

	// Verify descending hotness sort order: q_proj (200) -> k_proj (60) -> rms_norm (10).
	if profiles[0].StageName != "q_proj" || profiles[0].StageIndex != 1 || profiles[0].Stalls.TotalSamples != 200 {
		t.Fatalf("profiles[0] = %+v, want stage 1 q_proj with 200 samples", profiles[0])
	}
	if profiles[0].Stalls.WaitTMEM != 120 || profiles[0].Stalls.MIOThrottle != 80 {
		t.Fatalf("profiles[0].Stalls = %+v, want WaitTMEM=120 MIOThrottle=80", profiles[0].Stalls)
	}
	if len(profiles[0].Hypotheses) != 2 {
		t.Fatalf("len(profiles[0].Hypotheses) = %d, want 2 (WGMMASerial and BankConflict)", len(profiles[0].Hypotheses))
	}
	if profiles[1].StageName != "k_proj" || profiles[1].StageIndex != 0 || profiles[1].Stalls.TotalSamples != 60 || profiles[1].Stalls.LongScoreboard != 60 {
		t.Fatalf("profiles[1] = %+v, want stage 0 k_proj with 60 LongScoreboard samples", profiles[1])
	}
	if profiles[2].StageName != "rms_norm" || profiles[2].StageIndex != 2 || profiles[2].Stalls.TotalSamples != 10 {
		t.Fatalf("profiles[2] = %+v, want stage 2 rms_norm with 10 samples", profiles[2])
	}
}

func TestMegakernelFlightProposeStageRewrites(t *testing.T) {
	plan, mega := testMegakernelPlan(t)
	_, spans, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("mega.LineInfoSource: %v", err)
	}
	lineToPC := map[int][2]uint64{
		spans[0].StartLine: {0x1000, 0x1100},
		spans[1].StartLine: {0x2000, 0x2400},
		spans[2].StartLine: {0x3000, 0x3200},
	}
	amap, err := StageLinePCMap(mega, lineToPC)
	if err != nil {
		t.Fatalf("StageLinePCMap: %v", err)
	}

	batch := cuptipcsampling.Batch{
		PCs: []cuptipcsampling.PC{
			{
				Name:   mega.Name,
				Offset: 0x2100,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 1, Samples: 120},
					{Index: 3, Samples: 80},
				},
			},
			{
				Name:    mega.Name,
				Offset:  0x1050,
				Reasons: []cuptipcsampling.StallSamples{{Index: 5, Samples: 50}},
			},
		},
	}
	profiles, err := AttributeMegakernelBatch(plan, mega, amap, batch)
	if err != nil {
		t.Fatalf("AttributeMegakernelBatch: %v", err)
	}

	nodeCostBefore := (hopper.Cost{}).NodeCost(egraph.ENode{Op: hopper.OpGEMM}, nil)
	estBefore := plan.EstimateNs(hopper.CostConfig{Device: "NVIDIA H100"})

	// Provide empty causal counters for q_proj so proposals come purely from
	// stall hypotheses (Causal == false), and a direct causal counter for k_proj
	// (Causal == true) to verify causal vs. stall-hypothesis separation.
	causal := map[string]qualify.KernelObservation{
		"q_proj": {Name: "q_proj", DeviceSMs: 132},
		"k_proj": {Name: "k_proj", DeviceSMs: 132, MemoryBoundKV: true},
	}
	qCausalBefore := causal["q_proj"]

	candidates, err := ProposeMegakernelStageRewrites(plan, profiles, causal)
	if err != nil {
		t.Fatalf("ProposeMegakernelStageRewrites: %v", err)
	}
	if len(candidates) == 0 {
		t.Fatal("expected at least one megakernel stage rewrite candidate")
	}

	// Verify e-graph NodeCost, baseline plan estimate, and causal observation map were not mutated.
	nodeCostAfter := (hopper.Cost{}).NodeCost(egraph.ENode{Op: hopper.OpGEMM}, nil)
	estAfter := plan.EstimateNs(hopper.CostConfig{Device: "NVIDIA H100"})
	if nodeCostBefore != nodeCostAfter || estBefore != estAfter {
		t.Fatalf("NodeCost or plan estimate mutated: nodeCost (%v -> %v), est (%v -> %v)", nodeCostBefore, nodeCostAfter, estBefore, estAfter)
	}
	if causal["q_proj"] != qCausalBefore {
		t.Fatalf("q_proj causal observation mutated: got %+v, want %+v", causal["q_proj"], qCausalBefore)
	}
	if plan.Kernels[1].Tile.Swizzle128B || plan.Kernels[1].Tile.BN != 128 {
		t.Fatal("baseline plan step was mutated in place")
	}

	// The first candidates must target the hottest stage (q_proj, stage 1) and
	// be marked Causal: false (stall-hypothesis driven, not conflated with causal counters).
	first := candidates[0]
	if first.StageName != "q_proj" || first.StageIndex != 1 {
		t.Fatalf("first candidate stage = (%d, %q), want (1, \"q_proj\")", first.StageIndex, first.StageName)
	}
	if first.Proposal.Causal {
		t.Fatalf("expected stall-derived q_proj candidate to have Causal=false, got true")
	}

	foundCausalKProj := false
	for _, cand := range candidates {
		if cand.Source == "" || !strings.Contains(cand.Source, cand.Megakernel.Name) {
			t.Fatalf("candidate %s lowered to empty or invalid megakernel source", cand.Name)
		}
		if cand.Geometry.Block.X == 0 || cand.Geometry.Grid.X == 0 {
			t.Fatalf("candidate %s has invalid 3D geometry: %+v", cand.Name, cand.Geometry)
		}
		if cand.StageName == "k_proj" && cand.Proposal.Causal {
			foundCausalKProj = true
		}
	}
	if !foundCausalKProj {
		t.Fatal("expected k_proj candidate from direct MemoryBoundKV observation to have Causal=true")
	}
}

type flightTestKernel struct {
	id int
}

func (k *flightTestKernel) Launch(cudalaunch.Config, ...any) error         { return nil }
func (k *flightTestKernel) Wait(context.Context, *cuda.Stream) error       { return nil }
func (k *flightTestKernel) Raw() driver.Function                           { return driver.Function(k.id) }
func (k *flightTestKernel) Close() error                                   { return nil }

type flightTestLoader struct {
	nextID int
}

func (l *flightTestLoader) Compile(_ context.Context, src cudahotswap.Source) (*cudahotswap.Compilation, error) {
	return &cudahotswap.Compilation{
		Entry: src.Entry,
		PTX:   ".version 8.5\n.target sm_90a\n.address_size 64\n.visible .entry " + src.Entry + "(.param .u64 p0)\n{\n\tret;\n}\n",
	}, nil
}

func (l *flightTestLoader) Load(_ context.Context, _ *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	l.nextID++
	return &flightTestKernel{id: l.nextID}, nil
}

func TestMegakernelFlightEvaluateAndHotSwap(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "gpu-lock"))
	ctx := context.Background()

	plan, mega := testMegakernelPlan(t)
	_, spans, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("mega.LineInfoSource: %v", err)
	}
	amap, err := StageLinePCMap(mega, map[int][2]uint64{
		spans[0].StartLine: {0x1000, 0x1100},
		spans[1].StartLine: {0x2000, 0x2400},
		spans[2].StartLine: {0x3000, 0x3200},
	})
	if err != nil {
		t.Fatalf("StageLinePCMap: %v", err)
	}
	batch := cuptipcsampling.Batch{
		PCs: []cuptipcsampling.PC{
			{
				Name:   mega.Name,
				Offset: 0x2100,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 1, Samples: 150},
					{Index: 3, Samples: 100},
				},
			},
			{
				Name:    mega.Name,
				Offset:  0x1050,
				Reasons: []cuptipcsampling.StallSamples{{Index: 5, Samples: 80}},
			},
		},
	}
	profiles, err := AttributeMegakernelBatch(plan, mega, amap, batch)
	if err != nil {
		t.Fatalf("AttributeMegakernelBatch: %v", err)
	}
	candidates, err := ProposeMegakernelStageRewrites(plan, profiles, nil)
	if err != nil || len(candidates) < 2 {
		t.Fatalf("ProposeMegakernelStageRewrites: len=%d err=%v (need >= 2 candidates)", len(candidates), err)
	}

	loader := &flightTestLoader{}
	svc, err := cudahotswap.NewService(loader)
	if err != nil {
		t.Fatalf("cudahotswap.NewService: %v", err)
	}
	defer func() {
		if err := svc.Close(ctx); err != nil {
			t.Errorf("svc.Close: %v", err)
		}
	}()

	baseSrc, err := mega.Source()
	if err != nil {
		t.Fatalf("mega.Source: %v", err)
	}
	v1State, err := svc.Register(ctx, mega.Name, cudahotswap.Source{
		Code:  baseSrc,
		Entry: mega.Name,
	}, "baseline_v1")
	if err != nil {
		t.Fatalf("svc.Register: %v", err)
	}
	if v1State.Version != 1 {
		t.Fatalf("v1State.Version = %d, want 1", v1State.Version)
	}
	if _, err := svc.SetCheck(mega.Name, cudahotswap.CheckConfig{
		Checker: func(context.Context, cudahotswap.State) error { return nil },
	}, "flight_check"); err != nil {
		t.Fatalf("svc.SetCheck: %v", err)
	}

	grid, block := mega.Grid()
	tracker, err := NewMegakernelEpochTracker(EpochConfig{
		Daemon: autotune.DaemonConfig{MaxTrials: 2, Iters: 20, ImproveFloor: 0.015},
	}, svc, mega.Name, grid[0])
	if err != nil {
		t.Fatalf("NewMegakernelEpochTracker: %v", err)
	}
	baseGeom := cudahotswap.LaunchGeometry{
		Grid:      candidates[0].Geometry.Grid,
		Block:     candidates[0].Geometry.Block,
		SharedMem: uint32(mega.SharedBytes()),
	}
	_ = block

	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("ctx:decode_block_mega:sm90a")),
		Admission: sha256.Sum256([]byte("adm:trusted_flight_gate")),
	}
	tokens := []int{128000, 791, 6864, 315, 9688, 374, 264, 1176}
	screenBlocks := []pairedtiming.Block{
		{A1: 100, B1: 88, B2: 88, A2: 100},
		{A1: 101, B1: 89, B2: 89, A2: 101},
		{A1: 99, B1: 87, B2: 87, A2: 99},
	}
	holdoutBlocks := []pairedtiming.Block{
		{A1: 100, B1: 88, B2: 88, A2: 100},
		{A1: 102, B1: 90, B2: 90, A2: 102},
		{A1: 99, B1: 87, B2: 87, A2: 99},
		{A1: 101, B1: 89, B2: 89, A2: 101},
		{A1: 100, B1: 88, B2: 88, A2: 100},
		{A1: 98, B1: 86, B2: 86, A2: 98},
	}
	aaNullBlocks := []pairedtiming.Block{
		{A1: 100, B1: 100.2, B2: 100.1, A2: 100},
		{A1: 99.8, B1: 99.9, B2: 99.9, A2: 99.8},
		{A1: 100.1, B1: 100.0, B2: 100.0, A2: 100.1},
	}

	// 1. Promote candidate 0 -> version 2.
	resV2, err := EvaluateAndHotSwapStageRewrite(ctx, StageHotSwapRequest{
		Service:       svc,
		KernelName:    mega.Name,
		Candidate:     candidates[0],
		PriorGeometry: &baseGeom,
		Identity:      id,
		Evidence: qualify.HoldoutCandidateEvidence{
			Screen: qualify.MeasuredPair{
				Identity:  id,
				Reference: mega.Name,
				Candidate: candidates[0].Name,
				Blocks:    screenBlocks,
			},
			HoldoutBlocks:   holdoutBlocks,
			AANullBlocks:    aaNullBlocks,
			ReferenceTokens: tokens,
			CandidateTokens: tokens,
			LogitMaxAbs:     1e-4,
		},
		Verify: VerifyDigest(id),
		Epoch:  tracker,
	})
	if err != nil {
		t.Fatalf("EvaluateAndHotSwapStageRewrite v2: %v", err)
	}
	if !resV2.Promoted || resV2.SelectedState.Version != 2 || resV2.PriorState.Version != 1 {
		t.Fatalf("v2 promotion unexpected state: promoted=%v selected=%+v prior=%+v", resV2.Promoted, resV2.SelectedState, resV2.PriorState)
	}
	if tracker.Epoch != 1 || tracker.ExpectedReadyCounter() != uint64(grid[0]) {
		t.Fatalf("tracker epoch=%d ready=%d, want epoch=1 ready=%d", tracker.Epoch, tracker.ExpectedReadyCounter(), grid[0])
	}

	// 2. Promote candidate 1 -> version 3.
	resV3, err := EvaluateAndHotSwapStageRewrite(ctx, StageHotSwapRequest{
		Service:       svc,
		KernelName:    mega.Name,
		Candidate:     candidates[1],
		PriorGeometry: &candidates[0].Geometry,
		Identity:      id,
		Evidence: qualify.HoldoutCandidateEvidence{
			Screen: qualify.MeasuredPair{
				Identity:  id,
				Reference: candidates[0].Name,
				Candidate: candidates[1].Name,
				Blocks:    screenBlocks,
			},
			HoldoutBlocks:   holdoutBlocks,
			AANullBlocks:    aaNullBlocks,
			ReferenceTokens: tokens,
			CandidateTokens: tokens,
			LogitMaxAbs:     1e-4,
		},
		Verify: VerifyDigest(id),
		Epoch:  tracker,
	})
	if err != nil {
		t.Fatalf("EvaluateAndHotSwapStageRewrite v3: %v", err)
	}
	if !resV3.Promoted || resV3.SelectedState.Version != 3 || resV3.PriorState.Version != 2 {
		t.Fatalf("v3 promotion unexpected state: promoted=%v selected=%+v prior=%+v", resV3.Promoted, resV3.SelectedState, resV3.PriorState)
	}
	if tracker.Epoch != 2 || tracker.ExpectedReadyCounter() != uint64(2*grid[0]) {
		t.Fatalf("tracker epoch=%d ready=%d, want epoch=2 ready=%d", tracker.Epoch, tracker.ExpectedReadyCounter(), 2*grid[0])
	}

	// 3. Reject a candidate with a single greedy token-ID mismatch even when
	// timing is fast and LogitMaxAbs is tiny.
	badTokens := append([]int(nil), tokens...)
	badTokens[4] = 9689
	resBad, err := EvaluateAndHotSwapStageRewrite(ctx, StageHotSwapRequest{
		Service:       svc,
		KernelName:    mega.Name,
		Candidate:     candidates[0],
		PriorGeometry: &candidates[1].Geometry,
		Identity:      id,
		Evidence: qualify.HoldoutCandidateEvidence{
			Screen: qualify.MeasuredPair{
				Identity:  id,
				Reference: candidates[1].Name,
				Candidate: candidates[0].Name + "_bad_tok",
				Blocks:    screenBlocks,
			},
			HoldoutBlocks:   holdoutBlocks,
			AANullBlocks:    aaNullBlocks,
			ReferenceTokens: tokens,
			CandidateTokens: badTokens,
			LogitMaxAbs:     1e-7,
		},
		Verify: VerifyDigest(id),
		Epoch:  tracker,
	})
	if err != nil {
		t.Fatalf("EvaluateAndHotSwapStageRewrite token mismatch: %v", err)
	}
	if resBad.Promoted || resBad.Promotion.Status != "REJECTED_TOKEN_MISMATCH" {
		t.Fatalf("expected token mismatch rejection, got promoted=%v status=%q", resBad.Promoted, resBad.Promotion.Status)
	}
	curState, err := svc.Status(mega.Name)
	if err != nil || curState.Version != 3 {
		t.Fatalf("service version after rejected candidate = %d (err=%v), want 3", curState.Version, err)
	}

	// 4. Roll back from version 3 to the immediate prior version (version 2, NOT version 1)
	// via RestorePrevious().
	restoredV2, err := resV3.RestorePrevious(ctx)
	if err != nil {
		t.Fatalf("resV3.RestorePrevious: %v", err)
	}
	if restoredV2.Version != 2 {
		t.Fatalf("restored version = %d, want immediate prior version 2", restoredV2.Version)
	}
	statusAfterRollback, err := svc.Status(mega.Name)
	if err != nil || statusAfterRollback.Version != 2 {
		t.Fatalf("svc.Status after RestorePrevious = %+v (err=%v), want Version=2", statusAfterRollback, err)
	}

	// A second RestorePrevious on the tracker rolls back from version 2 to version 1.
	restoredV1, err := tracker.RestorePrevious(ctx)
	if err != nil {
		t.Fatalf("tracker.RestorePrevious to v1: %v", err)
	}
	if restoredV1.Version != 1 {
		t.Fatalf("second restored version = %d, want baseline version 1", restoredV1.Version)
	}
}
