package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"flag"
	"path/filepath"
	"regexp"
	"slices"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/model"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

var autonomousGlobalKernelRe = regexp.MustCompile(`__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?([A-Za-z0-9_]+)\s*\(`)

type autonomousStubPTXASCompiler struct {
	calls map[string]int
}

func (c *autonomousStubPTXASCompiler) CompilePTXAS(_ context.Context, cudaSource, arch string) (ptxasinfo.Report, error) {
	if c.calls == nil {
		c.calls = make(map[string]int)
	}
	if arch == "" {
		arch = "sm_121a"
	}
	matches := autonomousGlobalKernelRe.FindAllStringSubmatch(cudaSource, -1)
	key := "default"
	if len(matches) > 0 {
		key = matches[0][1]
	}
	c.calls[key]++
	callNum := c.calls[key]

	var fns []*ptxasinfo.Function
	for i, m := range matches {
		fn := &ptxasinfo.Function{
			Name:      m[1],
			Arch:      arch,
			Entry:     true,
			Registers: 96,
			SharedMem: 4096,
		}
		if i == 0 && callNum == 1 {
			fn.Registers = 172
			fn.SpillStores = 16
			fn.SpillLoads = 16
		}
		fns = append(fns, fn)
	}
	return ptxasinfo.Report{
		Targets: []*ptxasinfo.Target{{
			Arch:      arch,
			Functions: fns,
		}},
	}, nil
}

func testAutonomousHoldoutEvidence(id qualify.MeasuredIdentity) qualify.HoldoutCandidateEvidence {
	tokens := []int{128000, 791, 6864, 315, 9822, 13, 42, 99}
	return qualify.HoldoutCandidateEvidence{
		Screen: qualify.MeasuredPair{
			Identity: id,
			Blocks: []pairedtiming.Block{
				{A1: 100, B1: 87, B2: 87, A2: 100},
				{A1: 101, B1: 88, B2: 88, A2: 101},
				{A1: 99, B1: 86, B2: 86, A2: 99},
			},
		},
		HoldoutBlocks: []pairedtiming.Block{
			{A1: 100, B1: 87, B2: 87, A2: 100},
			{A1: 101, B1: 88, B2: 88, A2: 101},
			{A1: 99, B1: 86, B2: 86, A2: 99},
			{A1: 100, B1: 87, B2: 87, A2: 100},
			{A1: 102, B1: 89, B2: 89, A2: 102},
			{A1: 98, B1: 85, B2: 85, A2: 98},
		},
		AANullBlocks: []pairedtiming.Block{
			{A1: 100, B1: 100.1, B2: 100.0, A2: 100},
			{A1: 99.9, B1: 100.0, B2: 99.9, A2: 99.9},
			{A1: 100.0, B1: 100.0, B2: 100.1, A2: 100.0},
		},
		ReferenceTokens: tokens,
		CandidateTokens: tokens,
		LogitMaxAbs:     1e-5,
	}
}

func TestEGraphAutonomousColdAndWarmStart(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "gpu.lock"))
	ctx := context.Background()

	modelCfg := ModelConfig{
		ModelType:         "llama",
		HiddenSize:        256,
		IntermediateSize:  256,
		NumHiddenLayers:   2,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  1,
		VocabSize:         256,
		RmsNormEps:        1e-5,
		RopeTheta:         10000,
		MaxSeqLen:         16,
	}

	loader := &flightTestLoader{}
	svc, err := cudahotswap.NewService(loader)
	if err != nil {
		t.Fatalf("cudahotswap.NewService: %v", err)
	}
	t.Cleanup(func() {
		if err := svc.Close(ctx); err != nil {
			t.Errorf("svc.Close: %v", err)
		}
	})

	cacheDir := filepath.Join(t.TempDir(), "pgo-cache")
	pgoCache := autotune.NewVerifiedPGOCache(cacheDir)
	compiler := &autonomousStubPTXASCompiler{}

	eng, err := NewAutonomousDecodeEngine(AutonomousDecodeConfig{
		Model:          &modelCfg,
		QKVBias:        true,
		LayerL:         0,
		LayerNext:      1,
		BatchM:         1,
		Target:         "sm_121a",
		Device:         "NVIDIA GB10",
		DeviceSMs:      48,
		Cache:          pgoCache,
		BeamWidth:      4,
		PTXASCompiler:  compiler,
		PTXASGate:      qualify.StaticGate{NoSpills: true, MaxRegisters: 255, MinCTAsPerSM: 1},
		MaxPTXASIters:  4,
		HotSwapService: svc,
		BucketConfig: &specdecode.AdaptiveBucketConfig{
			HysteresisSteps:       1,
			InitialAcceptanceRate: 0.50,
			InitialBucketM:        1,
		},
	})
	if err != nil {
		t.Fatalf("NewAutonomousDecodeEngine: %v", err)
	}
	if len(eng.CrossLayerMegakernel.Stages) != 6 {
		t.Fatalf("len(CrossLayerMegakernel.Stages) = %d, want 6", len(eng.CrossLayerMegakernel.Stages))
	}

	_, spans, err := eng.CrossLayerMegakernel.LineInfoSource()
	if err != nil {
		t.Fatalf("LineInfoSource: %v", err)
	}
	lineToPC := make(map[int][2]uint64, len(spans))
	for i, sp := range spans {
		base := uint64(i+1) * 0x1000
		lineToPC[sp.StartLine] = [2]uint64{base, base + 0x200}
	}
	pcBatch := cuptipcsampling.Batch{
		PCs: []cuptipcsampling.PC{
			{
				Name:   eng.CrossLayerMegakernel.Name,
				Offset: 0x1040,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 5, Samples: 150},
				},
			},
		},
	}

	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("llama-infer:autonomous:sm_121a")),
		Admission: sha256.Sum256([]byte("llama-infer:autonomous:admission")),
	}
	ev := testAutonomousHoldoutEvidence(id)
	winningCubin := []byte("ELF-CUBIN-LLAMA-CROSS-L0-L1-SM121A")

	// 1. Cold start: builds beam + cross-layer megakernel, cures spills via ptxas loop,
	// attributes PCs, promotes holdout winner, updates HotSwapService & BucketController,
	// and stores verified proof + cubin in VerifiedPGOCache.
	coldRes, err := eng.Run(ctx, model.AutonomousEvalRequest{
		PCBatch:         pcBatch,
		LineToPC:        lineToPC,
		Identity:        id,
		DefaultEvidence: &ev,
		WinningCubin:    winningCubin,
		ActiveBatch:     1,
		DraftedTokens:   4,
		AcceptedTokens:  2,
	})
	if err != nil {
		t.Fatalf("eng.Run (cold start): %v", err)
	}
	if coldRes.CacheHit {
		t.Fatal("expected CacheHit=false on cold start")
	}
	if coldRes.CrossLayerPlan == nil || coldRes.CrossLayerMegakernel == nil {
		t.Fatal("expected CrossLayerPlan and CrossLayerMegakernel on cold start")
	}
	if len(coldRes.PTXASRefinements) == 0 {
		t.Fatal("expected PTXASRefinements on cold start")
	}
	if len(coldRes.StageProfiles) != 6 {
		t.Fatalf("len(StageProfiles) = %d, want 6", len(coldRes.StageProfiles))
	}
	if coldRes.Winner == nil || !coldRes.Winner.Promoted || !coldRes.Winner.TokensMatched {
		t.Fatalf("expected promoted winner with matched tokens, got %+v", coldRes.Winner)
	}
	if coldRes.SelectedState.Version != 2 || coldRes.PriorState.Version != 1 {
		t.Fatalf("cold hot-swap versions: prior=%d selected=%d, want 1->2",
			coldRes.PriorState.Version, coldRes.SelectedState.Version)
	}
	if !coldRes.PersistedToCache || coldRes.CachedProof == nil {
		t.Fatalf("expected PersistedToCache=true, got %+v", coldRes)
	}
	if eng.BucketController.CurrentBucket().BucketM != 4 {
		t.Fatalf("BucketController CurrentBucket.BucketM = %d, want 4", eng.BucketController.CurrentBucket().BucketM)
	}

	// 2. Warm start on the same cache key (BucketM=1): reset bucket controller to M=1
	// or create a fresh engine at BatchM=1 pointing to the same VerifiedPGOCache.
	warmLoader := &flightTestLoader{}
	warmSvc, err := cudahotswap.NewService(warmLoader)
	if err != nil {
		t.Fatalf("cudahotswap.NewService (warm): %v", err)
	}
	t.Cleanup(func() {
		if err := warmSvc.Close(ctx); err != nil {
			t.Errorf("warmSvc.Close: %v", err)
		}
	})

	warmEng, err := NewAutonomousDecodeEngine(AutonomousDecodeConfig{
		Model:          &modelCfg,
		QKVBias:        true,
		LayerL:         0,
		LayerNext:      1,
		BatchM:         1,
		Target:         "sm_121a",
		Device:         "NVIDIA GB10",
		DeviceSMs:      48,
		Cache:          pgoCache,
		HotSwapService: warmSvc,
	})
	if err != nil {
		t.Fatalf("NewAutonomousDecodeEngine (warm): %v", err)
	}

	t0 := time.Now()
	warmRes, err := warmEng.Run(ctx, model.AutonomousEvalRequest{})
	warmElapsed := time.Since(t0)
	if err != nil {
		t.Fatalf("warmEng.Run: %v", err)
	}
	if !warmRes.CacheHit {
		t.Fatal("expected CacheHit=true on warm start")
	}
	if warmElapsed >= 25*time.Millisecond {
		t.Logf("note: warm start took %v", warmElapsed)
	}
	if !bytes.Equal(warmRes.CachedCubin, winningCubin) {
		t.Fatalf("warm CachedCubin mismatch")
	}
	if warmRes.SelectedState.Version != 1 || warmRes.PromotedVersion != "v1" {
		t.Fatalf("warm HotSwapService state = %+v, want Version=1", warmRes.SelectedState)
	}
}

func TestEGraphAutonomousRollbackAndBucketAdaptation(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "gpu.lock"))
	ctx := context.Background()

	modelCfg := ModelConfig{
		ModelType:         "llama",
		HiddenSize:        256,
		IntermediateSize:  256,
		NumHiddenLayers:   2,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  1,
		VocabSize:         256,
		RmsNormEps:        1e-5,
		RopeTheta:         10000,
		MaxSeqLen:         16,
	}

	loader := &flightTestLoader{}
	svc, err := cudahotswap.NewService(loader)
	if err != nil {
		t.Fatalf("cudahotswap.NewService: %v", err)
	}
	t.Cleanup(func() {
		if err := svc.Close(ctx); err != nil {
			t.Errorf("svc.Close: %v", err)
		}
	})

	eng, err := NewAutonomousDecodeEngine(AutonomousDecodeConfig{
		Model:          &modelCfg,
		QKVBias:        true,
		LayerL:         0,
		LayerNext:      1,
		BatchM:         1,
		Target:         "sm_121a",
		Device:         "NVIDIA GB10",
		DeviceSMs:      48,
		HotSwapService: svc,
		BucketConfig: &specdecode.AdaptiveBucketConfig{
			HysteresisSteps:       1,
			InitialAcceptanceRate: 0.20,
			InitialBucketM:        1,
		},
	})
	if err != nil {
		t.Fatalf("NewAutonomousDecodeEngine: %v", err)
	}

	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("llama-infer:rollback:sm_121a")),
		Admission: sha256.Sum256([]byte("llama-infer:rollback:admission")),
	}
	ev := testAutonomousHoldoutEvidence(id)

	// Promote v2 and step bucket controller to high acceptance (M=8).
	resV2, err := eng.Run(ctx, model.AutonomousEvalRequest{
		Identity:        id,
		DefaultEvidence: &ev,
		ActiveBatch:     1,
		DraftedTokens:   8,
		AcceptedTokens:  8,
	})
	if err != nil {
		t.Fatalf("eng.Run v2: %v", err)
	}
	if resV2.SelectedState.Version != 2 {
		t.Fatalf("resV2.SelectedState.Version = %d, want 2", resV2.SelectedState.Version)
	}

	// Verify a failing PostSwapCheck rolls HotSwapService back to v2.
	checkErr := errors.New("post-promotion verification failed")
	_, err = eng.Run(ctx, model.AutonomousEvalRequest{
		Identity:        id,
		DefaultEvidence: &ev,
		PostSwapCheck: func(context.Context, cudahotswap.State) error {
			return checkErr
		},
	})
	if !errors.Is(err, checkErr) {
		t.Fatalf("expected PostSwapCheck error %v, got %v", checkErr, err)
	}
	st, err := svc.Status(eng.KernelName)
	if err != nil || st.Version != 2 {
		t.Fatalf("svc.Status after failed check = %+v (err=%v), want Version=2", st, err)
	}

	// RestorePrevious rolls back from v2 to v1.
	restored, err := eng.RestorePrevious(ctx, resV2)
	if err != nil {
		t.Fatalf("eng.RestorePrevious: %v", err)
	}
	if restored.Version != 1 {
		t.Fatalf("restored.Version = %d, want 1", restored.Version)
	}
}

func TestEGraphAutonomousCLIFlagsAndPlanWiring(t *testing.T) {
	for _, name := range []string{
		"egraph-cross-layer",
		"egraph-ptxas-loop",
		"egraph-pgo-cache-dir",
		"egraph-autonomous",
	} {
		if flag.Lookup(name) == nil {
			t.Errorf("missing CLI flag -%s", name)
		}
	}

	cfg := llama3_8B
	cfg.NumHiddenLayers = 2
	prof := mustProfile(t, "GB10")
	compiler := &autonomousStubPTXASCompiler{}
	cacheDir := filepath.Join(t.TempDir(), "pgo")
	emitDir := t.TempDir()

	if err := emitEgraph(emitDir, "GB10", &cfg, egraphOptions{
		Profile:       prof,
		Mega:          true,
		Attn:          true,
		AttnSplits:    2,
		Autonomous:    true,
		PTXASLoop:     true,
		PTXASCompiler: compiler,
		PGOCacheDir:   cacheDir,
	}); err != nil {
		t.Fatalf("emitEgraph Autonomous+PTXASLoop: %v", err)
	}
	if len(compiler.calls) == 0 {
		t.Fatal("expected PTXASCompiler to be invoked during buildEgraphPlans with PTXASLoop=true")
	}

	plans, err := buildEgraphPlans(&cfg, false, egraphOptions{
		Profile:       prof,
		Autonomous:    true,
		PTXASLoop:     true,
		PTXASCompiler: compiler,
		PGOCacheDir:   cacheDir,
	})
	if err != nil {
		t.Fatalf("buildEgraphPlans Autonomous: %v", err)
	}
	if plans.cross == nil {
		t.Fatal("expected plans.cross != nil when Autonomous=true")
	}
	gen, _ := plans.ops()
	if !slices.Contains(gen, "cross_layer_post_qkv") {
		t.Fatalf("plans.ops() = %v, missing cross_layer_post_qkv", gen)
	}
}

func TestEGraphAutonomousMultiBucketPreWarmingAndSelection(t *testing.T) {
	modelCfg := ModelConfig{
		ModelType:         "llama",
		HiddenSize:        256,
		IntermediateSize:  256,
		NumHiddenLayers:   2,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  1,
		VocabSize:         256,
		RmsNormEps:        1e-5,
		RopeTheta:         10000,
		MaxSeqLen:         16,
	}

	buckets := []int{1, 2, 4, 8}
	eng, err := NewAutonomousDecodeEngine(AutonomousDecodeConfig{
		Model:     &modelCfg,
		QKVBias:   true,
		LayerL:    0,
		LayerNext: 1,
		BatchM:    1,
		Buckets:   buckets,
		Target:    "sm_121a",
		Device:    "NVIDIA GB10",
		DeviceSMs: 48,
	})
	if err != nil {
		t.Fatalf("NewAutonomousDecodeEngine: %v", err)
	}
	for _, m := range buckets {
		plan, okP := eng.BucketPlans[m]
		mega, okM := eng.BucketMegakernels[m]
		if !okP || !okM || plan == nil {
			t.Fatalf("missing pre-warmed bucket M=%d in BucketPlans/BucketMegakernels", m)
		}
		if len(mega.Stages) != 6 {
			t.Fatalf("bucket M=%d len(mega.Stages) = %d, want 6", m, len(mega.Stages))
		}
		if gotM := plan.Output.Shape[0]; gotM != m {
			t.Fatalf("bucket M=%d plan.Output.Shape = %v, want M=%d", m, plan.Output.Shape, m)
		}
	}

	steps := []struct {
		name           string
		draftTokens    int
		acceptanceRate float64
		wantBucketM    int
		wantSwapped    bool
	}{
		{
			name:           "single-token decode M=1",
			draftTokens:    0,
			acceptanceRate: 0.0,
			wantBucketM:    1,
			wantSwapped:    false,
		},
		{
			name:           "speculative tree verify draft=3 -> M=4",
			draftTokens:    3,
			acceptanceRate: 0.60,
			wantBucketM:    4,
			wantSwapped:    true,
		},
		{
			name:           "speculative tree verify draft=7 -> M=8",
			draftTokens:    7,
			acceptanceRate: 0.85,
			wantBucketM:    8,
			wantSwapped:    true,
		},
		{
			name:           "low acceptance collapse -> M=1",
			draftTokens:    7,
			acceptanceRate: 0.10,
			wantBucketM:    1,
			wantSwapped:    true,
		},
		{
			name:           "return to single-token decode M=1",
			draftTokens:    0,
			acceptanceRate: 0.0,
			wantBucketM:    1,
			wantSwapped:    false,
		},
	}

	for _, tc := range steps {
		t.Run(tc.name, func(t *testing.T) {
			bucketM, plan, mega, trans, err := eng.SelectBucketMegakernel(tc.draftTokens, tc.acceptanceRate)
			if err != nil {
				t.Fatalf("SelectBucketMegakernel(draft=%d, rate=%.2f): %v", tc.draftTokens, tc.acceptanceRate, err)
			}
			if bucketM != tc.wantBucketM {
				t.Fatalf("bucketM = %d, want %d", bucketM, tc.wantBucketM)
			}
			if trans.Swapped != tc.wantSwapped {
				t.Fatalf("trans.Swapped = %v, want %v", trans.Swapped, tc.wantSwapped)
			}
			if plan == nil || plan.Output.Shape[0] != tc.wantBucketM {
				t.Fatalf("plan output shape = %v, want first dim %d", plan.Output.Shape, tc.wantBucketM)
			}
			if len(mega.Stages) != 6 {
				t.Fatalf("len(mega.Stages) = %d, want 6", len(mega.Stages))
			}
		})
	}
}

func TestDecodeGraphWithEGraphMegaAndCrossLayer(t *testing.T) {
	const (
		attnFn   = driver.Function(200)
		sampleFn = driver.Function(201)
	)
	gpuW, cfg, sp, kvs, gm := fakeStepFixture(t, 2, weightBF16)
	cfg.HiddenSize = 256
	cfg.IntermediateSize = 256
	cfg.NumAttentionHeads = 4
	cfg.NumKeyValueHeads = 1
	cfg.ExplicitHeadDim = 64
	cfg.VocabSize = 256
	layerKV := func(i int) (cudakv.LayerCache, error) { return kvs[i], nil }

	prof := mustProfile(t, "GB10")
	plans, err := buildEgraphPlans(cfg, false, egraphOptions{
		Profile:    prof,
		Mega:       true,
		Attn:       true,
		AttnSplits: 2,
		CrossLayer: true,
	})
	if err != nil {
		t.Fatalf("buildEgraphPlans: %v", err)
	}
	if plans.cross == nil {
		t.Fatal("expected non-nil plans.cross with CrossLayer=true")
	}

	qkvSlice, ok := fakeDeviceSliceLen[float32](0x700000, plans.attnIn.Output.Shape[1])
	if !ok {
		t.Fatal("fakeDeviceSliceLen[float32] failed")
	}
	eg := &egraphDecoder{
		plans:       plans,
		megas:       make(map[*hopper.Plan]hopper.Megakernel),
		fns:         make(map[string]driver.Function),
		pdl:         make(map[driver.Function]bool),
		ready:       make(map[*hopper.Plan]cudamem.Slice[uint64]),
		deviceEpoch: true,
		tempPtr:     make(map[string]driver.DevicePtr),
		posKind:     make(map[driver.Function]string),
		posArgs:     make(map[driver.Function][]egraphPosArg),
		qkv:         qkvSlice,
	}

	nextFn := driver.Function(300)
	readyPtr := driver.DevicePtr(0x800000)
	megaFns := make(map[driver.Function]string)
	for _, p := range eg.decodePlans() {
		m, err := p.Megakernel("")
		if err != nil {
			t.Fatalf("Megakernel(%s): %v", p.Name, err)
		}
		eg.megas[p] = m
		words := max(m.ReadyWords(), len(m.Stages)+1)
		rs, ok := fakeDeviceSliceLen[uint64](readyPtr, words)
		if !ok {
			t.Fatal("fakeDeviceSliceLen[uint64] failed")
		}
		eg.ready[p] = rs
		readyPtr += driver.DevicePtr(words * 8)
		eg.fns[m.Name] = nextFn
		megaFns[nextFn] = m.Name
		nextFn++
	}
	eg.fns[plans.rope.Name] = nextFn
	nextFn++
	eg.fns[plans.sdpa.Name] = nextFn
	nextFn++
	eg.fns[plans.sdpa.Name+"_combine"] = nextFn
	nextFn++

	tempBase := driver.DevicePtr(0x900000)
	for _, p := range eg.planList() {
		for _, tmp := range p.Temps {
			if _, ok := eg.tempPtr[tmp.Name]; !ok {
				eg.tempPtr[tmp.Name] = tempBase
				tempBase += 0x1000
			}
		}
	}

	b := egraphBuffers{
		x:        sp.X.Ptr(),
		x2:       0xA00000,
		xn:       0xA10000,
		qkv:      eg.qkv.Ptr(),
		attnOut:  sp.AttnOut.Ptr(),
		norm:     gpuW.Norm.Ptr(),
		lmHead:   gpuW.LMHead.Ptr(),
		logits:   sp.Logits.Ptr(),
		qbf:      0xA20000,
		cs:       0xA30000,
		ropePage: 0xA40000,
		seqLen1:  0xA50000,
		pages:    0xA60000,
		ws:       0xA70000,
	}
	for i := range gpuW.Layers {
		lw := &gpuW.Layers[i]
		b.layers = append(b.layers, egraphLayerBuffers{
			attnNorm: lw.InputNorm.Ptr(),
			wqkv:     0xB00000 + driver.DevicePtr(i*0x10000),
			wo:       lw.WO.Ptr(),
			ffnNorm:  lw.PostNorm.Ptr(),
			wgu:      0xB80000 + driver.DevicePtr(i*0x10000),
			wdown:    lw.WDown.Ptr(),
			k:        0xC00000 + driver.DevicePtr(i*0x10000),
			v:        0xC80000 + driver.DevicePtr(i*0x10000),
		})
	}
	if err := eg.bindDecode(cfg, b); err != nil {
		t.Fatalf("bindDecode: %v", err)
	}
	gm.eg = eg

	h := newFakeGraphHarness()
	dg, err := h.build(gpuW, cfg, layerKV, sp, gm, attnFn, sampleFn)
	if err != nil {
		t.Fatalf("build DecodeGraph with -egraph-mega -egraph-cross-layer: %v", err)
	}

	megaNodeCount := 0
	sawCrossMega := false
	for i, n := range h.added {
		name, ok := megaFns[n.desc.Function]
		if !ok {
			continue
		}
		megaNodeCount++
		if name == eg.megas[plans.cross].Name {
			sawCrossMega = true
		}
		if len(n.args) < 2 {
			t.Fatalf("megakernel node %d (%s) has %d args, want >= 2", i, name, len(n.args))
		}
		gotEpoch, ok := n.args[len(n.args)-1].(uint32)
		if !ok || gotEpoch != 0 {
			t.Fatalf("megakernel node %d (%s) epoch arg = %v (%T), want uint32(0) for device-resolved epoch",
				i, name, n.args[len(n.args)-1], n.args[len(n.args)-1])
		}
	}
	// For 2 layers with CrossLayer=true and Mega=true:
	// norm (1) + attnIn[0] (1) + cross[0] (1) + post[1] (1) + head (1) = 5 megakernel nodes.
	if megaNodeCount != 5 {
		t.Fatalf("megaNodeCount = %d, want 5", megaNodeCount)
	}
	if !sawCrossMega {
		t.Fatal("DecodeGraph did not capture cross-layer megakernel node")
	}

	for _, pos := range []int{0, 1, 7, cfg.MaxSeqLen - 1} {
		if err := dg.SetPosition(pos); err != nil {
			t.Fatalf("SetPosition(%d): %v", pos, err)
		}
	}
}
