package model

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
	"github.com/tmc/lib/nvidia/x/nvl72"
)

func makePDTestPage(base int32, pageSize int) []int32 {
	p := make([]int32, pageSize)
	for i := range pageSize {
		p[i] = base + int32(i)
	}
	return p
}

func TestDisaggregatedPDEngineMultiTurnRadixAndC2CPromotion(t *testing.T) {
	const pageSize = 16
	eng, err := NewDisaggregatedPDEngine(PDEngineConfig{
		Arch: ArchConfig{
			HiddenSize:        512,
			IntermediateSize:  1024,
			NumHiddenLayers:   4,
			NumAttentionHeads: 8,
			NumKeyValueHeads:  2,
			HeadDim:           64,
		},
		Generation:      nvl72.GenGB300,
		TotalRacks:      2,
		CacheKind:       nvl72.CacheMLA,
		KVLoRARank:      512,
		RoPEDim:         64,
		PageSize:        pageSize,
		HBMPageCapacity: 4,
		C2CPageCapacity: 4,
		SLA: nvl72.PDSLA{
			PromptTokens:      1024,
			DecodeBatchPerGPU: 64,
			MaxTTFTMs:         200.0,
			MaxTPOTMs:         20.0,
		},
	})
	if err != nil {
		t.Fatalf("NewDisaggregatedPDEngine: %v", err)
	}

	sys0 := makePDTestPage(100, pageSize)
	sys1 := makePDTestPage(200, pageSize)
	turn1 := makePDTestPage(300, pageSize)
	turn2 := makePDTestPage(400, pageSize)

	// Turn 1: cold prefix [sys0, sys1, turn1] allocates 3 fresh Tier-1 HBM3e pages.
	res1, err := eng.AdmitAndPrefill(PDRequest{
		ReqID:               "conv-turn-1",
		TokenPages:          [][]int32{sys0, sys1, turn1},
		ReleaseAfterPrefill: true,
		ActiveDecodes:       []string{"dec-0"},
	})
	if err != nil {
		t.Fatalf("AdmitAndPrefill turn 1: %v", err)
	}
	if res1.Receipt.NewlyAllocatedPages != 3 || res1.Receipt.HBMHitPages != 0 || res1.Receipt.C2CPromotedPages != 0 {
		t.Fatalf("turn 1 receipt = %+v, want NewlyAllocatedPages=3", res1.Receipt)
	}
	hbm1, c2c1 := eng.RadixStats()
	if hbm1 != 3 || c2c1 != 0 {
		t.Fatalf("RadixStats after turn 1 = (%d, %d), want (3, 0)", hbm1, c2c1)
	}

	// Turn 2: reuses [sys0, sys1] in Tier-1 HBM3e (2 HBM hits) and appends [turn2] (1 new HBM page).
	res2, err := eng.AdmitAndPrefill(PDRequest{
		ReqID:               "conv-turn-2",
		TokenPages:          [][]int32{sys0, sys1, turn2},
		ReleaseAfterPrefill: true,
		ActiveDecodes:       []string{"dec-0", "dec-1"},
	})
	if err != nil {
		t.Fatalf("AdmitAndPrefill turn 2: %v", err)
	}
	if res2.Receipt.HBMHitPages != 2 || res2.Receipt.NewlyAllocatedPages != 1 || res2.Receipt.PrefillTokensSaved != 2*pageSize {
		t.Fatalf("turn 2 receipt = %+v, want HBMHitPages=2, NewlyAllocatedPages=1, PrefillTokensSaved=%d",
			res2.Receipt, 2*pageSize)
	}
	if res2.EstimatedPrefillTTFTMs >= res1.EstimatedPrefillTTFTMs {
		t.Fatalf("expected turn 2 prefix-hit TTFT (%.4f ms) < turn 1 cold TTFT (%.4f ms)",
			res2.EstimatedPrefillTTFTMs, res1.EstimatedPrefillTTFTMs)
	}

	// Evict all 4 HBM pages to Tier-2 Grace LPDDR5X C2C by admitting a 4-page unrelated burst.
	burst := [][]int32{
		makePDTestPage(1000, pageSize),
		makePDTestPage(1100, pageSize),
		makePDTestPage(1200, pageSize),
		makePDTestPage(1300, pageSize),
	}
	resBurst, err := eng.AdmitAndPrefill(PDRequest{
		ReqID:               "burst-evict",
		TokenPages:          burst,
		ReleaseAfterPrefill: true,
	})
	if err != nil {
		t.Fatalf("AdmitAndPrefill burst: %v", err)
	}
	if resBurst.Receipt.EvictedToC2CPages != 4 {
		t.Fatalf("burst EvictedToC2CPages = %d, want 4", resBurst.Receipt.EvictedToC2CPages)
	}
	hbmBurst, c2cBurst := eng.RadixStats()
	if hbmBurst != 4 || c2cBurst != 4 {
		t.Fatalf("RadixStats after burst = (%d, %d), want (4, 4)", hbmBurst, c2cBurst)
	}

	// Turn 3: revisits [sys0, sys1] + new [turn3], promoting [sys0, sys1] from Tier-2 Grace C2C back to Tier-1 HBM3e.
	turn3 := makePDTestPage(500, pageSize)
	res3, err := eng.AdmitAndPrefill(PDRequest{
		ReqID:               "conv-turn-3",
		TokenPages:          [][]int32{sys0, sys1, turn3},
		ReleaseAfterPrefill: true,
	})
	if err != nil {
		t.Fatalf("AdmitAndPrefill turn 3: %v", err)
	}
	if res3.Receipt.C2CPromotedPages != 2 || res3.Receipt.NewlyAllocatedPages != 1 {
		t.Fatalf("turn 3 receipt = %+v, want C2CPromotedPages=2, NewlyAllocatedPages=1", res3.Receipt)
	}
	if res3.Receipt.C2CPromoteBytes <= 0 || res3.Receipt.C2CPromoteLatencyUs <= 0 {
		t.Fatalf("expected positive C2C promotion bytes and latency, got (%d B, %g us)",
			res3.Receipt.C2CPromoteBytes, res3.Receipt.C2CPromoteLatencyUs)
	}
	if !res3.TransferKernel.UseMLALatent || !res3.TransferKernel.UseGraceC2CStaging {
		t.Fatalf("TransferKernel = %+v, want UseMLALatent=true and UseGraceC2CStaging=true", res3.TransferKernel)
	}
	for _, want := range []string{
		"eg_fabric_peer_ptr",
		"st.global.v4.u32",
		"st.release.sys.global.u32",
		"ld.acquire.sys.global.u32",
	} {
		if !strings.Contains(res3.TransferKernelSource, want) {
			t.Fatalf("TransferKernelSource missing %q", want)
		}
	}
}

func TestDisaggregatedPDEngine2DContextParallelAndRubin(t *testing.T) {
	eng, err := NewDisaggregatedPDEngine(PDEngineConfig{
		Arch: ArchConfig{
			HiddenSize:        512,
			IntermediateSize:  1024,
			NumHiddenLayers:   4,
			NumAttentionHeads: 8,
			NumKeyValueHeads:  2,
			HeadDim:           64,
		},
		Generation:       string(nvl72.TierVeraRubinNVL144),
		PrefillRacks:     2,
		DecodeRacks:      1,
		UseRubinDisagg:   true,
		UseGraceC2CStage: true,
		CacheKind:        nvl72.CacheMLA,
		CPMode:           nvl72.CPHybrid2D,
		CPUlyssesDeg:     4,
		CPRingDeg:        4,
		CPTarget:         "sm_107a",
	})
	if err != nil {
		t.Fatalf("NewDisaggregatedPDEngine (Rubin): %v", err)
	}
	if eng.RubinPlan() == nil {
		t.Fatal("expected non-nil RubinPlan for VeraRubin-NVL144")
	}

	tokens := make([]int32, 256)
	for i := range tokens {
		tokens[i] = int32(10 + i)
	}

	for _, mode := range []struct {
		name    string
		cpMode  nvl72.ContextParallelMode
		ulysses int
		ring    int
	}{
		{"StripedRing", nvl72.CPStripedRing, 1, 4},
		{"UlyssesAll2All", nvl72.CPUlyssesAll2All, 4, 1},
		{"Hybrid2D", nvl72.CPHybrid2D, 4, 4},
	} {
		t.Run(mode.name, func(t *testing.T) {
			res, err := eng.AdmitAndPrefill(PDRequest{
				ReqID:               "req-" + mode.name,
				PromptTokens:        tokens,
				ReleaseAfterPrefill: true,
				CPMode:              mode.cpMode,
				HasCPMode:           true,
				CPUlyssesDeg:        mode.ulysses,
				CPRingDeg:           mode.ring,
			})
			if err != nil {
				t.Fatalf("AdmitAndPrefill(%s): %v", mode.name, err)
			}
			if res.CPSchedule.Mode != mode.cpMode {
				t.Fatalf("CPSchedule.Mode = %v, want %v", res.CPSchedule.Mode, mode.cpMode)
			}
			if len(res.CPPlan) != mode.ulysses*mode.ring {
				t.Fatalf("len(CPPlan) = %d, want %d", len(res.CPPlan), mode.ulysses*mode.ring)
			}
			if res.CPImbalanceRatio != 1.0 {
				t.Fatalf("CPImbalanceRatio = %.4f, want 1.0", res.CPImbalanceRatio)
			}
		})
	}
}

func TestDisaggregatedPDEngineDecodeBucketsAndSLARebalance(t *testing.T) {
	ctx := context.Background()
	eng, err := NewDisaggregatedPDEngine(PDEngineConfig{
		Arch: ArchConfig{
			HiddenSize:        512,
			IntermediateSize:  1024,
			NumHiddenLayers:   4,
			NumAttentionHeads: 8,
			NumKeyValueHeads:  2,
			HeadDim:           64,
		},
		Generation: nvl72.GenGB300,
		TotalRacks: 4,
		CacheKind:  nvl72.CacheMLA,
		SLA: nvl72.PDSLA{
			PromptTokens:      4096,
			DecodeBatchPerGPU: 64,
			MaxTTFTMs:         200.0,
			MaxTPOTMs:         20.0,
		},
		DecodeTableConfig: specdecode.MultiBucketTableConfig{
			Arch: "sm_100a",
		},
	})
	if err != nil {
		t.Fatalf("NewDisaggregatedPDEngine: %v", err)
	}

	gotBuckets := eng.DecodeBuckets()
	wantBuckets := []int{1, 2, 4, 8}
	if !slices.Equal(gotBuckets, wantBuckets) {
		t.Fatalf("DecodeBuckets = %v, want %v", gotBuckets, wantBuckets)
	}

	steps := []struct {
		draftTokens    int
		acceptanceRate float64
		wantM          int
	}{
		{0, 0.20, 1},
		{1, 0.50, 2},
		{3, 0.70, 4},
		{7, 0.95, 8},
		{0, 0.0, 1},
	}
	for i, st := range steps {
		gotM, kName, err := eng.StepDecodePool(ctx, st.draftTokens, st.acceptanceRate)
		if err != nil {
			t.Fatalf("step %d StepDecodePool: %v", i, err)
		}
		if gotM != st.wantM {
			t.Fatalf("step %d bucketM = %d, want %d", i, gotM, st.wantM)
		}
		if kName == "" {
			t.Fatalf("step %d empty kernelName", i)
		}
	}

	// Verify SLA-driven dynamic prefill/decode pool rebalancing.
	initPlan := eng.Plan()

	// 1. Within SLA: no rebalance needed.
	samePlan, changed, err := eng.RebalanceToSLA(initPlan.EstimatedTTFTMs*0.8, initPlan.EstimatedTPOTMs*0.8)
	if err != nil {
		t.Fatalf("RebalanceToSLA (within SLA): %v", err)
	}
	if changed || samePlan.PrefillGPUs != initPlan.PrefillGPUs {
		t.Fatalf("expected changed=false within SLA, got changed=%v plan=%+v", changed, samePlan)
	}

	// 2. TTFT SLA spike (e.g., 450 ms > 200 ms MaxTTFTMs): shifts GPUs/racks to prefill pool.
	ttftPlan, changedTTFT, err := eng.RebalanceToSLA(450.0, 10.0)
	if err != nil {
		t.Fatalf("RebalanceToSLA (TTFT spike): %v", err)
	}
	if !changedTTFT || ttftPlan.PrefillGPUs <= initPlan.PrefillGPUs {
		t.Fatalf("expected PrefillGPUs to increase on TTFT spike: before=%d after=%d changed=%v",
			initPlan.PrefillGPUs, ttftPlan.PrefillGPUs, changedTTFT)
	}

	// 3. TPOT SLA spike (e.g., 45 ms > 20 ms MaxTPOTMs): shifts GPUs/racks back to decode pool.
	tpotPlan, changedTPOT, err := eng.RebalanceToSLA(80.0, 45.0)
	if err != nil {
		t.Fatalf("RebalanceToSLA (TPOT spike): %v", err)
	}
	if !changedTPOT || tpotPlan.DecodeGPUs <= ttftPlan.DecodeGPUs {
		t.Fatalf("expected DecodeGPUs to increase on TPOT spike: before=%d after=%d changed=%v",
			ttftPlan.DecodeGPUs, tpotPlan.DecodeGPUs, changedTPOT)
	}
}
