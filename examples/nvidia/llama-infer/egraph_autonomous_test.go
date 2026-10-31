package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"path/filepath"
	"regexp"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
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
