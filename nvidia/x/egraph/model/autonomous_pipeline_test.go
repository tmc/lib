package model

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

var globalKernelRe = regexp.MustCompile(`__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?([A-Za-z0-9_]+)\s*\(`)

// autonomousTestPTXASCompiler simulates ptxas -v compilation. On the first
// compile of an unrefined plan (before RefinePlanByPTXASLoop steps down warps,
// unroll, or WGMMA), it reports register spills so the closed-loop refiner
// cures the spill and converges on iteration 2.
type autonomousTestPTXASCompiler struct {
	callsByPlan map[string]int
}

func (c *autonomousTestPTXASCompiler) CompilePTXAS(_ context.Context, cudaSource, arch string) (ptxasinfo.Report, error) {
	if c.callsByPlan == nil {
		c.callsByPlan = make(map[string]int)
	}
	if arch == "" {
		arch = "sm_90a"
	}
	matches := globalKernelRe.FindAllStringSubmatch(cudaSource, -1)
	var key string
	if len(matches) > 0 {
		key = matches[0][1]
	} else {
		key = "default"
	}
	c.callsByPlan[key]++
	callNum := c.callsByPlan[key]

	var fns []*ptxasinfo.Function
	for i, m := range matches {
		fnName := m[1]
		fn := &ptxasinfo.Function{
			Name:      fnName,
			Arch:      arch,
			Entry:     true,
			Registers: 96,
			SharedMem: 4096,
		}
		if i == 0 && callNum == 1 {
			fn.Registers = 168
			fn.SpillStores = 24
			fn.SpillLoads = 24
		}
		fns = append(fns, fn)
	}
	if len(fns) == 0 {
		fns = []*ptxasinfo.Function{{
			Name:      "kernel_step",
			Arch:      arch,
			Entry:     true,
			Registers: 64,
		}}
	}
	return ptxasinfo.Report{
		Targets: []*ptxasinfo.Target{{
			Arch:      arch,
			Functions: fns,
		}},
	}, nil
}

func buildAutonomousTestGraph(t *testing.T) (*egraph.EGraph, egraph.Id, *hopper.Plan, *hopper.Plan) {
	t.Helper()
	b := hopper.NewBuilder()
	xn := b.Input("xn", hopper.BF16, 1, 256)
	wO := b.Input("w_o", hopper.BF16, 256, 256)
	x := b.Input("x", hopper.F32, 1, 256)
	hid := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(xn, wO), x)
	ffnNorm := b.Cast(hopper.BF16, b.RMSNorm(hid, b.Input("ffn_norm", hopper.F32, 256), 256, 1e-5))
	act := b.SwiGLU(b.MatMul(ffnNorm, b.Input("w_gu", hopper.BF16, 512, 256)))
	down := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", hopper.BF16, 256, 256)), hid)
	root := b.Cast(hopper.BF16, b.RMSNorm(down, b.Input("next_norm", hopper.F32, 256), 256, 1e-5))

	exprPost, err := hopper.Optimize(b.G.Clone(), root)
	if err != nil {
		t.Fatalf("hopper.Optimize post: %v", err)
	}
	postPlan, err := hopper.CompileFor(exprPost, "l0_post", hopper.BF16, "sm_90a")
	if err != nil {
		t.Fatalf("hopper.CompileFor post: %v", err)
	}

	bNext := hopper.NewBuilder()
	xnNext := bNext.Input("xn", hopper.BF16, 1, 256)
	wQKV := bNext.Input("w_qkv", hopper.BF16, 384, 256)
	bQKV := bNext.Input("b_qkv", hopper.F32, 384)
	qkv := bNext.Map(hopper.MustLambda("(add $0 $1)"), bNext.MatMul(xnNext, wQKV), bQKV)
	exprNext, err := hopper.Optimize(bNext.G, qkv)
	if err != nil {
		t.Fatalf("hopper.Optimize nextAttnIn: %v", err)
	}
	nextAttnInPlan, err := hopper.CompileFor(exprNext, "l1_attn_in", hopper.F32, "sm_90a")
	if err != nil {
		t.Fatalf("hopper.CompileFor nextAttnIn: %v", err)
	}

	return b.G, root, postPlan, nextAttnInPlan
}

func testWinningHoldoutEvidence(id qualify.MeasuredIdentity) qualify.HoldoutCandidateEvidence {
	tokens := []int{128000, 791, 6864, 315, 9688, 374, 264, 1176}
	return qualify.HoldoutCandidateEvidence{
		Screen: qualify.MeasuredPair{
			Identity: id,
			Blocks: []pairedtiming.Block{
				{A1: 100, B1: 86, B2: 86, A2: 100},
				{A1: 101, B1: 87, B2: 87, A2: 101},
				{A1: 99, B1: 85, B2: 85, A2: 99},
			},
		},
		HoldoutBlocks: []pairedtiming.Block{
			{A1: 100, B1: 86, B2: 86, A2: 100},
			{A1: 102, B1: 88, B2: 88, A2: 102},
			{A1: 99, B1: 85, B2: 85, A2: 99},
			{A1: 101, B1: 87, B2: 87, A2: 101},
			{A1: 100, B1: 86, B2: 86, A2: 100},
			{A1: 98, B1: 84, B2: 84, A2: 98},
		},
		AANullBlocks: []pairedtiming.Block{
			{A1: 100, B1: 100.1, B2: 100.0, A2: 100},
			{A1: 99.9, B1: 100.0, B2: 99.9, A2: 99.9},
			{A1: 100.1, B1: 100.0, B2: 100.1, A2: 100.1},
		},
		ReferenceTokens: tokens,
		CandidateTokens: tokens,
		LogitMaxAbs:     1e-5,
	}
}

func TestAutonomousFlightPipelineColdAndWarmStart(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "gpu-lock"))
	ctx := context.Background()

	g, root, postPlan, nextAttnInPlan := buildAutonomousTestGraph(t)

	glue := hopper.CrossLayerGlue{
		Name:               "llama_cross_l0_l1",
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	}
	_, crossMega, err := hopper.BuildCrossLayerMegakernel(postPlan, nextAttnInPlan, glue)
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel: %v", err)
	}
	_, spans, err := crossMega.LineInfoSource()
	if err != nil || len(spans) == 0 {
		t.Fatalf("crossMega.LineInfoSource: len=%d err=%v", len(spans), err)
	}
	lineToPC := make(map[int][2]uint64, len(spans))
	for i, sp := range spans {
		base := uint64(i+1) * 0x1000
		lineToPC[sp.StartLine] = [2]uint64{base, base + 0x200}
	}
	pcBatch := cuptipcsampling.Batch{
		PCs: []cuptipcsampling.PC{
			{
				Name:   crossMega.Name,
				Offset: 0x1040,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 5, Samples: 180}, // LongScoreboard on stage 0
				},
			},
			{
				Name:   crossMega.Name,
				Offset: 0x2040,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 3, Samples: 90}, // MIOThrottle on stage 1
				},
			},
		},
	}

	cacheDir := filepath.Join(t.TempDir(), "pgo-cache")
	pgoCache := autotune.NewVerifiedPGOCache(cacheDir)
	cacheKey := autotune.PGOCacheKey{
		Arch:          "sm_90a",
		DriverVersion: "570.86.10",
		CUDAVersion:   "12.8",
		WeightDigest:  "sha256:llama3_8b_bf16",
		PlanKey:       "llama_cross_l0_l1",
		ShapeBucket:   4,
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

	bucketCtrl, err := specdecode.NewAdaptiveBucketController(specdecode.AdaptiveBucketConfig{
		HysteresisSteps:       1,
		InitialAcceptanceRate: 0.50,
		InitialBucketM:        1,
	})
	if err != nil {
		t.Fatalf("NewAdaptiveBucketController: %v", err)
	}

	ptxasCompiler := &autonomousTestPTXASCompiler{}
	cfg := AutonomousPipelineConfig{
		Cache:            pgoCache,
		CacheKey:         cacheKey,
		Device:           "NVIDIA H100",
		Arch:             "sm_90a",
		DeviceSMs:        132,
		BeamWidth:        4,
		PTXASCompiler:    ptxasCompiler,
		PTXASGate:        qualify.StaticGate{NoSpills: true, MaxRegisters: 255, MinCTAsPerSM: 1},
		MaxPTXASIters:    4,
		HotSwapService:   svc,
		BucketController: bucketCtrl,
	}

	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("ctx:autonomous:sm_90a")),
		Admission: sha256.Sum256([]byte("adm:autonomous:trusted")),
	}
	ev := testWinningHoldoutEvidence(id)
	winningCubin := []byte("ELF-CUBIN-AUTONOMOUS-SM90A-VERIFIED-BINARY")

	// 1. Cold-start execution: Cache Miss -> Beam -> PTXAS spill cure -> PC Attribution ->
	// Holdout tournament -> HotSwapService push -> BucketController update -> VerifiedPGOCache.Store.
	coldRes, err := RunAutonomousFlightPipeline(ctx, g, root, "llama_decode", hopper.BF16, cfg, AutonomousEvalRequest{
		CrossLayerPost:       postPlan,
		CrossLayerNextAttnIn: nextAttnInPlan,
		CrossLayerGlue:       glue,
		PCBatch:              pcBatch,
		LineToPC:             lineToPC,
		Identity:             id,
		DefaultEvidence:      &ev,
		WinningCubin:         winningCubin,
		ActiveBatch:          1,
		DraftedTokens:        4,
		AcceptedTokens:       2, // 0.50 acceptance -> BucketM=4 (RegimeMTPTreeVerify)
	})
	if err != nil {
		t.Fatalf("RunAutonomousFlightPipeline (cold start): %v", err)
	}
	if coldRes.CacheHit {
		t.Fatal("expected CacheHit=false on cold start")
	}
	if len(coldRes.BeamCandidates) < 2 {
		t.Fatalf("len(BeamCandidates) = %d, want >= 2", len(coldRes.BeamCandidates))
	}
	if coldRes.CrossLayerPlan == nil || coldRes.CrossLayerMegakernel == nil {
		t.Fatal("expected CrossLayerPlan and CrossLayerMegakernel to be populated")
	}
	if len(coldRes.PTXASRefinements) == 0 {
		t.Fatal("expected PTXASRefinements to be populated")
	}
	sawSpillCure := false
	for _, ref := range coldRes.PTXASRefinements {
		if len(ref.Iterations) >= 2 && ref.Iterations[0].TotalSpillBytes > 0 && ref.Converged {
			last := ref.Iterations[len(ref.Iterations)-1]
			if last.TotalSpillBytes == 0 && last.Clean {
				sawSpillCure = true
			}
		}
	}
	if !sawSpillCure {
		t.Fatalf("expected RefinePlanByPTXASLoop to cure initial register spills; refinements=%+v", coldRes.PTXASRefinements)
	}
	if len(coldRes.StageProfiles) != len(crossMega.Stages) {
		t.Fatalf("len(StageProfiles) = %d, want %d", len(coldRes.StageProfiles), len(crossMega.Stages))
	}
	if coldRes.StageProfiles[0].Stalls.TotalSamples != 180 {
		t.Fatalf("hottest StageProfile TotalSamples = %d, want 180", coldRes.StageProfiles[0].Stalls.TotalSamples)
	}
	if coldRes.Winner == nil || !coldRes.Winner.Promoted || !coldRes.Winner.TokensMatched {
		t.Fatalf("expected holdout tournament winner to be promoted with matched tokens, got %+v", coldRes.Winner)
	}
	if coldRes.SelectedState.Version != 2 || coldRes.PriorState.Version != 1 || coldRes.PromotedVersion != "v2" {
		t.Fatalf("cold hot-swap state: prior=%+v selected=%+v promotedVersion=%q, want v1->v2",
			coldRes.PriorState, coldRes.SelectedState, coldRes.PromotedVersion)
	}
	if !coldRes.PersistedToCache || coldRes.CachedProof == nil {
		t.Fatalf("expected PersistedToCache=true and non-nil CachedProof, got %+v", coldRes)
	}
	if bucketCtrl.CurrentBucket().BucketM != 4 {
		t.Fatalf("BucketController CurrentBucket.BucketM = %d, want 4", bucketCtrl.CurrentBucket().BucketM)
	}

	// 2. Warm-start execution on the same PGOCacheKey: must hit VerifiedPGOCache.LoadVerified
	// in <1 ms (CacheHit == true) and install directly into a fresh HotSwapService.
	warmLoader := &flightTestLoader{}
	warmSvc, err := cudahotswap.NewService(warmLoader)
	if err != nil {
		t.Fatalf("cudahotswap.NewService (warm): %v", err)
	}
	defer func() {
		if err := warmSvc.Close(ctx); err != nil {
			t.Errorf("warmSvc.Close: %v", err)
		}
	}()

	warmCfg := cfg
	warmCfg.HotSwapService = warmSvc
	// Set PTXASCompiler to nil to verify warm start returns before Step 2.
	warmCfg.PTXASCompiler = nil

	t0 := time.Now()
	warmRes, err := RunAutonomousFlightPipeline(ctx, nil, 0, "llama_decode", hopper.BF16, warmCfg, AutonomousEvalRequest{})
	warmElapsed := time.Since(t0)
	if err != nil {
		t.Fatalf("RunAutonomousFlightPipeline (warm start): %v", err)
	}
	if !warmRes.CacheHit {
		t.Fatal("expected CacheHit=true on warm start")
	}
	if warmElapsed >= 25*time.Millisecond {
		t.Logf("note: warm-start took %v on loaded filesystem", warmElapsed)
	}
	if warmRes.CachedProof == nil || warmRes.CachedProof.CubinSHA256 != sha256.Sum256(winningCubin) {
		t.Fatalf("warm CachedProof digest mismatch: %+v", warmRes.CachedProof)
	}
	if !bytes.Equal(warmRes.CachedCubin, winningCubin) {
		t.Fatalf("warm CachedCubin mismatch: got %d bytes, want %d", len(warmRes.CachedCubin), len(winningCubin))
	}
	if warmRes.SelectedState.Version != 1 || warmRes.PromotedVersion != "v1" {
		t.Fatalf("warm HotSwapService state = %+v (PromotedVersion=%q), want Version=1",
			warmRes.SelectedState, warmRes.PromotedVersion)
	}
	st, err := warmSvc.Status("llama_decode")
	if err != nil || st.Version != 1 {
		t.Fatalf("warmSvc.Status = %+v (err=%v), want Version=1", st, err)
	}
}

func TestAutonomousFlightPipelineCrossLayerAndRollback(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "gpu-lock"))
	ctx := context.Background()

	g, root, postPlan, nextAttnInPlan := buildAutonomousTestGraph(t)
	glue := hopper.CrossLayerGlue{
		Name:               "llama_cross_l0_l1",
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
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

	cfg := AutonomousPipelineConfig{
		Device:         "NVIDIA H100",
		Arch:           "sm_90a",
		DeviceSMs:      132,
		BeamWidth:      3,
		HotSwapService: svc,
	}

	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("ctx:crosslayer:sm_90a")),
		Admission: sha256.Sum256([]byte("adm:crosslayer:trusted")),
	}
	crossEv := testWinningHoldoutEvidence(id)

	// 1. Promote the cross-layer candidate (v1 baseline -> v2 cross-layer megakernel).
	resV2, err := RunAutonomousFlightPipeline(ctx, g, root, "llama_cross_kernel", hopper.BF16, cfg, AutonomousEvalRequest{
		CrossLayerPost:       postPlan,
		CrossLayerNextAttnIn: nextAttnInPlan,
		CrossLayerGlue:       glue,
		Identity:             id,
		Evidence: map[string]qualify.HoldoutCandidateEvidence{
			"llama_cross_l0_l1": crossEv,
		},
	})
	if err != nil {
		t.Fatalf("RunAutonomousFlightPipeline v2: %v", err)
	}
	if resV2.CrossLayerPlan == nil || resV2.CrossLayerMegakernel == nil {
		t.Fatal("expected CrossLayerPlan and CrossLayerMegakernel in result")
	}
	if resV2.Winner == nil || !resV2.Winner.Promoted || !strings.Contains(resV2.Winner.Winner.Handle(), "llama_cross_l0_l1") {
		t.Fatalf("expected cross-layer candidate to win tournament, got %+v", resV2.Winner)
	}
	if resV2.SelectedState.Version != 2 || resV2.PriorState.Version != 1 {
		t.Fatalf("v2 promotion state: prior=%+v selected=%+v, want v1->v2", resV2.PriorState, resV2.SelectedState)
	}

	// 2. Attempt a subsequent promotion on top of v2 whose PostSwapCheck fails;
	// verify cudahotswap rolls back to immediate prior version v2 (not v1).
	checkErr := errors.New("post-promotion latency guard failed")
	resFailed, err := RunAutonomousFlightPipeline(ctx, g, root, "llama_cross_kernel", hopper.BF16, cfg, AutonomousEvalRequest{
		Identity:        id,
		DefaultEvidence: &crossEv,
		PostSwapCheck: func(context.Context, cudahotswap.State) error {
			return checkErr
		},
	})
	if !errors.Is(err, checkErr) {
		t.Fatalf("expected PostSwapCheck error %v, got %v", checkErr, err)
	}
	if resFailed == nil || resFailed.SelectedState.Version != 2 {
		t.Fatalf("after failed PostSwapCheck, SelectedState = %+v, want rollback to v2", resFailed)
	}
	curState, err := svc.Status("llama_cross_kernel")
	if err != nil || curState.Version != 2 {
		t.Fatalf("svc.Status after failed PostSwapCheck = %+v (err=%v), want Version=2", curState, err)
	}

	// 3. Explicit RestorePrevious on resV2 rolls back from v2 to v1.
	restoredV1, err := resV2.RestorePrevious(ctx)
	if err != nil {
		t.Fatalf("resV2.RestorePrevious: %v", err)
	}
	if restoredV1.Version != 1 {
		t.Fatalf("restoredV1.Version = %d, want 1", restoredV1.Version)
	}
	curState, err = svc.Status("llama_cross_kernel")
	if err != nil || curState.Version != 1 {
		t.Fatalf("svc.Status after RestorePrevious = %+v (err=%v), want Version=1", curState, err)
	}
}

func TestRunAutonomousStackPipeline(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "gpu-lock"))
	ctx := context.Background()

	_, _, postPlan, nextAttnInPlan := buildAutonomousTestGraph(t)
	const numLayers = 4
	layers := make([]hopper.DecoderLayerPlans, numLayers)
	for i := 0; i < numLayers; i++ {
		attnIn := cloneHopperPlan(nextAttnInPlan)
		attnIn.Name = "l" + string(rune('0'+i)) + "_attn_in"
		post := cloneHopperPlan(postPlan)
		post.Name = "l" + string(rune('0'+i)) + "_post"
		layers[i] = hopper.DecoderLayerPlans{
			LayerIndex: i,
			AttnIn:     attnIn,
			Post:       post,
		}
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

	cfg := AutonomousPipelineConfig{
		Device:         "NVIDIA H100",
		Arch:           "sm_90a",
		DeviceSMs:      132,
		BeamWidth:      4,
		PTXASCompiler:  &autonomousTestPTXASCompiler{},
		PTXASGate:      qualify.StaticGate{NoSpills: true, MaxRegisters: 255, MinCTAsPerSM: 1},
		MaxPTXASIters:  4,
		HotSwapService: svc,
		KernelName:     "llama_stack_cross",
	}
	stackCfg := hopper.CrossLayerStackConfig{
		NamePrefix:         "llama_cross",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
		PackArena:          true,
		Mega:               true,
	}

	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("ctx:stack:sm_90a")),
		Admission: sha256.Sum256([]byte("adm:stack:trusted")),
	}
	ev := testWinningHoldoutEvidence(id)

	stackRes, err := RunAutonomousStackPipeline(ctx, cfg, stackCfg, layers, AutonomousEvalRequest{
		Identity:        id,
		DefaultEvidence: &ev,
	})
	if err != nil {
		t.Fatalf("RunAutonomousStackPipeline: %v", err)
	}
	if stackRes == nil || stackRes.Stack == nil || stackRes.Flight == nil {
		t.Fatalf("unexpected nil in AutonomousStackResult: %+v", stackRes)
	}
	if stackRes.Stack.NumLayers != numLayers {
		t.Fatalf("Stack.NumLayers = %d, want %d", stackRes.Stack.NumLayers, numLayers)
	}
	if got, want := len(stackRes.Stack.CrossBoundaries), numLayers-1; got != want {
		t.Fatalf("len(Stack.CrossBoundaries) = %d, want %d", got, want)
	}
	if got, want := len(stackRes.Stack.CrossMegakernels), numLayers-1; got != want {
		t.Fatalf("len(Stack.CrossMegakernels) = %d, want %d", got, want)
	}
	if stackRes.Stack.ArenaBytes == 0 {
		t.Fatal("expected non-zero Stack.ArenaBytes")
	}
	if got, want := stackRes.Stack.UnfusedLaunchCount(), 2*numLayers; got != want {
		t.Fatalf("UnfusedLaunchCount = %d, want %d", got, want)
	}
	if got, want := stackRes.Stack.FusedLaunchCount(), numLayers+1; got != want {
		t.Fatalf("FusedLaunchCount = %d, want %d", got, want)
	}
	if stackRes.Flight.CrossLayerPlan == nil || stackRes.Flight.CrossLayerMegakernel == nil {
		t.Fatal("expected Flight.CrossLayerPlan and Flight.CrossLayerMegakernel to be populated from layers[0]->layers[1]")
	}
	if stackRes.Flight.Winner == nil || !stackRes.Flight.Winner.Promoted {
		t.Fatalf("expected Flight.Winner to be promoted, got %+v", stackRes.Flight.Winner)
	}
	if stackRes.Flight.SelectedState.Version != 2 {
		t.Fatalf("Flight.SelectedState.Version = %d, want 2", stackRes.Flight.SelectedState.Version)
	}
}
