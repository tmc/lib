package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/egraph/model"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
	"github.com/tmc/lib/nvidia/x/nvl72"
)

// LlamaPDDisaggConfig configures a [LlamaPDDisaggRunner] binding a LLaMA
// [ModelConfig] to [model.DisaggregatedPDEngine].
type LlamaPDDisaggConfig struct {
	Model           *ModelConfig
	Generation      string
	TotalRacks      int
	PrefillRacks    int
	DecodeRacks     int
	SLA             nvl72.PDSLA
	CacheKind       nvl72.AttentionCacheKind
	PageSize        int
	HBMPageCapacity int
	C2CPageCapacity int
	CPMode          nvl72.ContextParallelMode
	CPUlyssesDeg    int
	CPRingDeg       int
	CPTarget        string
	Target          string
	UseRubinDisagg  bool
}

// LlamaConversationTurn describes one turn in a multi-turn conversation
// submitted to [LlamaPDDisaggRunner.ProcessConversationTurn].
type LlamaConversationTurn struct {
	SessionID      string
	TurnID         string
	TurnTokens     []int32
	DraftTokens    int
	AcceptanceRate float64
	ActiveDecodes  []string
}

// LlamaTurnResult reports the disaggregated prefill and decode execution
// metrics for a single conversation turn processed by
// [LlamaPDDisaggRunner.ProcessConversationTurn].
type LlamaTurnResult struct {
	SessionID        string
	ReqID            string
	TotalHistoryToks int
	PrefillResult    *model.PDRequestTickResult
	SelectedBucketM  int
	DecodeKernelName string
	HBMPagesUsed     int
	C2CPagesUsed     int
}

// LlamaPDDisaggRunner coordinates multi-turn LLaMA inference across
// disaggregated prefill and decode pools using [model.DisaggregatedPDEngine],
// preserving session token histories so subsequent turns hit the two-tier
// HBM3e + Grace/Vera C2C LPDDR5X Radix prefix cache.
type LlamaPDDisaggRunner struct {
	mu       sync.Mutex
	cfg      LlamaPDDisaggConfig
	engine   *model.DisaggregatedPDEngine
	sessions map[string][]int32
	turnSeq  int
}

func archConfigFromModelConfig(mc *ModelConfig) model.ArchConfig {
	if mc == nil {
		def := DefaultModelConfig()
		mc = &def
	}
	return model.ArchConfig{
		HiddenSize:        max(mc.HiddenSize, 64),
		IntermediateSize:  max(mc.IntermediateSize, 128),
		NumHiddenLayers:   max(mc.NumHiddenLayers, 1),
		NumAttentionHeads: max(mc.NumAttentionHeads, 1),
		NumKeyValueHeads:  max(mc.NumKeyValueHeads, 1),
		HeadDim:           max(mc.HeadDim(), 16),
		RMSNormEps:        mc.RmsNormEps,
	}
}

// NewLlamaPDDisaggRunner constructs a [LlamaPDDisaggRunner] backed by
// [model.NewDisaggregatedPDEngine].
func NewLlamaPDDisaggRunner(cfg LlamaPDDisaggConfig) (*LlamaPDDisaggRunner, error) {
	if cfg.Model == nil {
		def := DefaultModelConfig()
		cfg.Model = &def
	}
	arch := archConfigFromModelConfig(cfg.Model)
	target := strings.TrimSpace(cfg.Target)
	if target == "" {
		target = "sm_100a"
	}
	cfg.Target = target

	pageSize := cfg.PageSize
	if pageSize <= 0 {
		pageSize = 16
	}
	cfg.PageSize = pageSize

	eng, err := model.NewDisaggregatedPDEngine(model.PDEngineConfig{
		Arch:            arch,
		Generation:      cfg.Generation,
		TotalRacks:      cfg.TotalRacks,
		PrefillRacks:    cfg.PrefillRacks,
		DecodeRacks:     cfg.DecodeRacks,
		SLA:             cfg.SLA,
		CacheKind:       cfg.CacheKind,
		PageSize:        pageSize,
		HBMPageCapacity: cfg.HBMPageCapacity,
		C2CPageCapacity: cfg.C2CPageCapacity,
		CPMode:          cfg.CPMode,
		CPUlyssesDeg:    cfg.CPUlyssesDeg,
		CPRingDeg:       cfg.CPRingDeg,
		CPTarget:        cfg.CPTarget,
		UseRubinDisagg:  cfg.UseRubinDisagg,
		DecodeTableConfig: specdecode.MultiBucketTableConfig{
			Arch: target,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("llama pd disagg runner: %w", err)
	}

	return &LlamaPDDisaggRunner{
		cfg:      cfg,
		engine:   eng,
		sessions: make(map[string][]int32),
	}, nil
}

// Engine returns the underlying [*model.DisaggregatedPDEngine].
func (r *LlamaPDDisaggRunner) Engine() *model.DisaggregatedPDEngine {
	if r == nil {
		return nil
	}
	return r.engine
}

// ProcessConversationTurn appends turn.TurnTokens (aligned to full page
// boundaries for prior turns) to the session's conversation history, runs
// [model.DisaggregatedPDEngine.AdmitAndPrefill] so cached conversation prefixes
// hit Tier-1 HBM3e or promote from Tier-2 Grace C2C LPDDR5X, and steps the
// multi-bucket cross-layer decode pool via
// [model.DisaggregatedPDEngine.StepDecodePool].
func (r *LlamaPDDisaggRunner) ProcessConversationTurn(ctx context.Context, turn LlamaConversationTurn) (*LlamaTurnResult, error) {
	if r == nil || r.engine == nil {
		return nil, errors.New("llama pd disagg runner: nil runner")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if len(turn.TurnTokens) == 0 {
		return nil, errors.New("llama pd disagg runner: empty TurnTokens")
	}

	r.mu.Lock()
	r.turnSeq++
	seq := r.turnSeq
	sessionID := strings.TrimSpace(turn.SessionID)
	if sessionID == "" {
		sessionID = "default-session"
	}
	reqID := strings.TrimSpace(turn.TurnID)
	if reqID == "" {
		reqID = fmt.Sprintf("%s-turn-%d", sessionID, seq)
	}

	pageSize := max(r.cfg.PageSize, 1)
	alignedTurn := slices.Clone(turn.TurnTokens)
	if rem := len(alignedTurn) % pageSize; rem != 0 {
		pad := pageSize - rem
		for range pad {
			alignedTurn = append(alignedTurn, 0)
		}
	}
	history := append(slices.Clone(r.sessions[sessionID]), alignedTurn...)
	r.sessions[sessionID] = history
	r.mu.Unlock()

	prefillRes, err := r.engine.AdmitAndPrefill(model.PDRequest{
		ReqID:               reqID,
		PromptTokens:        history,
		ReleaseAfterPrefill: true,
		ActiveDecodes:       turn.ActiveDecodes,
	})
	if err != nil {
		return nil, err
	}

	bucketM, kernelName, err := r.engine.StepDecodePool(ctx, turn.DraftTokens, turn.AcceptanceRate)
	if err != nil {
		return nil, err
	}
	hbmPages, c2cPages := r.engine.RadixStats()

	return &LlamaTurnResult{
		SessionID:        sessionID,
		ReqID:            reqID,
		TotalHistoryToks: len(history),
		PrefillResult:    prefillRes,
		SelectedBucketM:  bucketM,
		DecodeKernelName: kernelName,
		HBMPagesUsed:     hbmPages,
		C2CPagesUsed:     c2cPages,
	}, nil
}

// RebalancePools delegates observed TTFT and TPOT SLA rebalancing to
// [model.DisaggregatedPDEngine.RebalanceToSLA].
func (r *LlamaPDDisaggRunner) RebalancePools(observedTTFTMs, observedTPOTMs float64) (nvl72.PDDisaggPlan, bool, error) {
	if r == nil || r.engine == nil {
		return nvl72.PDDisaggPlan{}, false, errors.New("llama pd disagg runner: nil runner")
	}
	return r.engine.RebalanceToSLA(observedTTFTMs, observedTPOTMs)
}
