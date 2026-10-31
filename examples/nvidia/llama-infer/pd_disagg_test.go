package main

import (
	"context"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvl72"
)

func makeLlamaTurnTokens(base int32, n int) []int32 {
	out := make([]int32, n)
	for i := range n {
		out[i] = base + int32(i)
	}
	return out
}

func TestLlamaPDDisaggMultiTurnPrefixReuseAndSLARebalance(t *testing.T) {
	ctx := context.Background()
	cfg := DefaultModelConfig()
	cfg.HiddenSize = 256
	cfg.IntermediateSize = 512
	cfg.NumHiddenLayers = 4
	cfg.NumAttentionHeads = 8
	cfg.NumKeyValueHeads = 2

	runner, err := NewLlamaPDDisaggRunner(LlamaPDDisaggConfig{
		Model:           &cfg,
		Generation:      nvl72.GenGB300,
		TotalRacks:      2,
		CacheKind:       nvl72.CacheMLA,
		PageSize:        16,
		HBMPageCapacity: 4,
		C2CPageCapacity: 4,
		CPMode:          nvl72.CPHybrid2D,
		CPUlyssesDeg:    2,
		CPRingDeg:       2,
		Target:          "sm_100a",
		SLA: nvl72.PDSLA{
			PromptTokens:      2048,
			DecodeBatchPerGPU: 64,
			MaxTTFTMs:         200.0,
			MaxTPOTMs:         20.0,
		},
	})
	if err != nil {
		t.Fatalf("NewLlamaPDDisaggRunner: %v", err)
	}

	// Turn 1: 32 tokens (2 pages) in session "chat-A" -> 2 new HBM pages, decode M=1.
	turn1, err := runner.ProcessConversationTurn(ctx, LlamaConversationTurn{
		SessionID:      "chat-A",
		TurnID:         "chat-A-t1",
		TurnTokens:     makeLlamaTurnTokens(100, 32),
		DraftTokens:    0,
		AcceptanceRate: 0.20,
		ActiveDecodes:  []string{"active-dec-1"},
	})
	if err != nil {
		t.Fatalf("ProcessConversationTurn turn 1: %v", err)
	}
	if turn1.PrefillResult.Receipt.NewlyAllocatedPages != 2 || turn1.PrefillResult.Receipt.HBMHitPages != 0 {
		t.Fatalf("turn 1 receipt = %+v, want NewlyAllocatedPages=2, HBMHitPages=0", turn1.PrefillResult.Receipt)
	}
	if turn1.SelectedBucketM != 1 || turn1.DecodeKernelName == "" {
		t.Fatalf("turn 1 bucket = (%d, %q), want (1, non-empty)", turn1.SelectedBucketM, turn1.DecodeKernelName)
	}

	// Turn 2: appends 16 tokens (1 page) to session "chat-A" -> 2 HBM prefix hits + 1 new HBM page, decode M=4.
	turn2, err := runner.ProcessConversationTurn(ctx, LlamaConversationTurn{
		SessionID:      "chat-A",
		TurnID:         "chat-A-t2",
		TurnTokens:     makeLlamaTurnTokens(200, 16),
		DraftTokens:    3,
		AcceptanceRate: 0.70,
		ActiveDecodes:  []string{"active-dec-1", "active-dec-2"},
	})
	if err != nil {
		t.Fatalf("ProcessConversationTurn turn 2: %v", err)
	}
	if turn2.TotalHistoryToks != 48 {
		t.Fatalf("turn 2 TotalHistoryToks = %d, want 48", turn2.TotalHistoryToks)
	}
	if turn2.PrefillResult.Receipt.HBMHitPages != 2 || turn2.PrefillResult.Receipt.NewlyAllocatedPages != 1 {
		t.Fatalf("turn 2 receipt = %+v, want HBMHitPages=2, NewlyAllocatedPages=1", turn2.PrefillResult.Receipt)
	}
	if turn2.PrefillResult.Receipt.PrefillTokensSaved != 32 {
		t.Fatalf("turn 2 PrefillTokensSaved = %d, want 32", turn2.PrefillResult.Receipt.PrefillTokensSaved)
	}
	if turn2.SelectedBucketM != 4 {
		t.Fatalf("turn 2 SelectedBucketM = %d, want 4", turn2.SelectedBucketM)
	}
	if !strings.Contains(turn2.PrefillResult.TransferKernelSource, "st.release.sys.global.u32") {
		t.Fatal("turn 2 TransferKernelSource missing st.release.sys.global.u32")
	}

	// Evict "chat-A" pages from Tier-1 HBM3e to Tier-2 Grace C2C via unrelated session "chat-B" (4 pages = 64 tokens).
	_, err = runner.ProcessConversationTurn(ctx, LlamaConversationTurn{
		SessionID:      "chat-B",
		TurnID:         "chat-B-t1",
		TurnTokens:     makeLlamaTurnTokens(1000, 64),
		DraftTokens:    7,
		AcceptanceRate: 0.95,
	})
	if err != nil {
		t.Fatalf("ProcessConversationTurn chat-B: %v", err)
	}

	// Turn 3 on "chat-A": appends 16 tokens (total 64 tokens = 4 pages: 3 promoted from Grace C2C + 1 new HBM page).
	turn3, err := runner.ProcessConversationTurn(ctx, LlamaConversationTurn{
		SessionID:      "chat-A",
		TurnID:         "chat-A-t3",
		TurnTokens:     makeLlamaTurnTokens(300, 16),
		DraftTokens:    7,
		AcceptanceRate: 0.95,
	})
	if err != nil {
		t.Fatalf("ProcessConversationTurn turn 3: %v", err)
	}
	if turn3.PrefillResult.Receipt.C2CPromotedPages != 3 || turn3.PrefillResult.Receipt.NewlyAllocatedPages != 1 {
		t.Fatalf("turn 3 receipt = %+v, want C2CPromotedPages=3, NewlyAllocatedPages=1", turn3.PrefillResult.Receipt)
	}
	if turn3.SelectedBucketM != 8 {
		t.Fatalf("turn 3 SelectedBucketM = %d, want 8", turn3.SelectedBucketM)
	}

	// Verify SLA-driven pool rebalancing through RebalancePools.
	initPlan := runner.Engine().Plan()
	rebalanced, changed, err := runner.RebalancePools(500.0, 10.0)
	if err != nil {
		t.Fatalf("RebalancePools: %v", err)
	}
	if !changed || rebalanced.PrefillGPUs <= initPlan.PrefillGPUs {
		t.Fatalf("expected PrefillGPUs to increase after TTFT violation: before=%d after=%d changed=%v",
			initPlan.PrefillGPUs, rebalanced.PrefillGPUs, changed)
	}
}
