// Command egserve runs a standalone OpenAI-compatible HTTP inference server
// backed by e-graph cross-layer persistent megakernels across buckets
// M in {1, 2, 4, 8}, Radix KV prefix-cached continuous batching, two-tier
// HBM3e + Grace/Vera C2C LPDDR5X disaggregated prefill/decode scheduling,
// formal barrier verification, and Perfetto / Chrome Trace flight telemetry.
//
// Usage:
//
//	egserve [-addr :8080] [-arch sm_121a] [-hidden-dim 256] [-interm-dim 512] [-layers 4] [-pgo-cache-dir dir] [-self-test]
package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/model"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
	"github.com/tmc/lib/nvidia/x/nvl72"
)

// ServerConfig configures an [EGServer] instance.
type ServerConfig struct {
	Addr            string
	Arch            string
	HiddenDim       int
	IntermDim       int
	Layers          int
	PGOCacheDir     string
	PageSize        int
	TotalKVPages    int
	HBMPageCapacity int
	C2CPageCapacity int
	BatchWindow     time.Duration
	LiveNVCC        bool
	SSHHost         string
	NVCCPath        string
	NVDISASMPath    string
}

func (c ServerConfig) withDefaults() ServerConfig {
	if strings.TrimSpace(c.Addr) == "" {
		c.Addr = ":8080"
	}
	if strings.TrimSpace(c.Arch) == "" {
		c.Arch = "sm_121a"
	}
	if c.HiddenDim <= 0 {
		c.HiddenDim = 256
	}
	if c.IntermDim <= 0 {
		c.IntermDim = 512
	}
	if c.Layers <= 0 {
		c.Layers = 4
	}
	if c.PageSize <= 0 {
		c.PageSize = 4
	}
	if c.TotalKVPages <= 0 {
		c.TotalKVPages = 256
	}
	if c.HBMPageCapacity <= 0 {
		c.HBMPageCapacity = 64
	}
	if c.C2CPageCapacity <= 0 {
		c.C2CPageCapacity = 128
	}
	return c
}

// ChatMessage represents a single message in an OpenAI chat completion request.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// DraftTokensSpec accepts either an integer count of draft tokens (e.g. 3) or
// an explicit slice of draft token IDs (e.g. [101, 102, 103]) in JSON requests.
type DraftTokensSpec struct {
	Count  int
	Tokens []int
}

// UnmarshalJSON decodes either a JSON integer or an array of integers.
func (d *DraftTokensSpec) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		*d = DraftTokensSpec{}
		return nil
	}
	if trimmed[0] == '[' {
		var toks []int
		if err := json.Unmarshal(trimmed, &toks); err != nil {
			return err
		}
		d.Tokens = toks
		d.Count = len(toks)
		return nil
	}
	var n int
	if err := json.Unmarshal(trimmed, &n); err != nil {
		return err
	}
	if n < 0 {
		n = 0
	}
	d.Count = n
	if n > 0 {
		d.Tokens = make([]int, n)
		for i := range n {
			d.Tokens[i] = 900 + i
		}
	}
	return nil
}

// MarshalJSON encodes DraftTokensSpec as an integer count.
func (d DraftTokensSpec) MarshalJSON() ([]byte, error) {
	if len(d.Tokens) > 0 && d.Count == 0 {
		return json.Marshal(len(d.Tokens))
	}
	return json.Marshal(d.Count)
}

// CompletionRequest represents an OpenAI-compatible /v1/completions or
// /v1/chat/completions payload with optional speculative decoding controls.
type CompletionRequest struct {
	Model       string          `json:"model"`
	Prompt      any             `json:"prompt,omitempty"`
	Messages    []ChatMessage   `json:"messages,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
	DraftTokens DraftTokensSpec `json:"draft_tokens,omitempty"`
}

// EGraphFlightMetadata reports per-request e-graph megakernel, Radix KV prefix
// cache, and disaggregated P/D telemetry attached to completion responses.
type EGraphFlightMetadata struct {
	ActiveBucketM          int     `json:"active_bucket_m"`
	ActiveTokens           int     `json:"active_tokens"`
	PaddedRows             int     `json:"padded_rows"`
	ActiveRowMask          uint64  `json:"active_row_mask"`
	RadixPrefixHitTokens   int     `json:"radix_prefix_hit_tokens"`
	C2CPromotedPages       int     `json:"c2c_promoted_pages"`
	MegakernelKernelName   string  `json:"megakernel_kernel_name"`
	BarrierSafetyVerified  bool    `json:"barrier_safety_verified"`
	EstimatedPrefillTTFTMs float64 `json:"estimated_prefill_ttft_ms"`
	PrefillGPUs            int     `json:"prefill_gpus"`
	DecodeGPUs             int     `json:"decode_gpus"`
}

// CompletionChoice is a single completion entry in an OpenAI response.
type CompletionChoice struct {
	Index        int          `json:"index"`
	Text         string       `json:"text,omitempty"`
	Message      *ChatMessage `json:"message,omitempty"`
	FinishReason string       `json:"finish_reason"`
}

// CompletionUsage reports token counts in an OpenAI completion response.
type CompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CompletionResponse is an OpenAI-compatible response enriched with
// [EGraphFlightMetadata].
type CompletionResponse struct {
	ID           string               `json:"id"`
	Object       string               `json:"object"`
	Created      int64                `json:"created"`
	Model        string               `json:"model"`
	Choices      []CompletionChoice   `json:"choices"`
	Usage        CompletionUsage      `json:"usage"`
	EGraphFlight EGraphFlightMetadata `json:"egraph_flight"`
}

// BucketStatusEntry summarizes one pre-warmed bucket in GET /v1/pgo/status.
type BucketStatusEntry struct {
	BucketM               int    `json:"bucket_m"`
	Regime                string `json:"regime"`
	KernelName            string `json:"kernel_name"`
	NumStages             int    `json:"num_stages"`
	CacheHit              bool   `json:"cache_hit"`
	BarrierSafetyVerified bool   `json:"barrier_safety_verified"`
}

// RadixKVCacheStats reports cumulative Radix KV prefix cache counters.
type RadixKVCacheStats struct {
	HitCount   int `json:"hit_count"`
	MissCount  int `json:"miss_count"`
	EvictCount int `json:"evict_count"`
	UsedPages  int `json:"used_pages"`
	FreePages  int `json:"free_pages"`
	HBMPages   int `json:"hbm_pages"`
	C2CPages   int `json:"c2c_pages"`
}

// PDPoolStatus reports the disaggregated prefill/decode GPU allocation and SLA status.
type PDPoolStatus struct {
	Generation  string  `json:"generation"`
	PrefillGPUs int     `json:"prefill_gpus"`
	DecodeGPUs  int     `json:"decode_gpus"`
	PDRatio     float64 `json:"pd_ratio"`
	MeetsSLA    bool    `json:"meets_sla"`
}

// PGOStatusResponse is returned by GET /v1/pgo/status.
type PGOStatusResponse struct {
	Arch                  string                          `json:"arch"`
	BarrierSafetyVerified bool                            `json:"barrier_safety_verified"`
	BarrierReports        map[int]*qualify.FormalBarrierReport `json:"barrier_reports"`
	Buckets               []BucketStatusEntry             `json:"buckets"`
	BucketHotSwaps        int                             `json:"bucket_hot_swaps"`
	HitCount              int                             `json:"hit_count"`
	MissCount             int                             `json:"miss_count"`
	EvictCount            int                             `json:"evict_count"`
	RadixKVCache          RadixKVCacheStats               `json:"radix_kv_cache"`
	PrefillGPUs           int                             `json:"prefill_gpus"`
	DecodeGPUs            int                             `json:"decode_gpus"`
	PDPool                PDPoolStatus                    `json:"pd_pool"`
}

// EGServer is an OpenAI-compatible HTTP inference server backed by
// [specdecode.MultiBucketMegakernelTable], [model.ContinuousBatchScheduler],
// [model.DisaggregatedPDEngine], and [autotune.FlightTelemetryRecorder].
type EGServer struct {
	cfg             ServerConfig
	mux             *http.ServeMux
	pgoCache        *autotune.VerifiedPGOCache
	table           *specdecode.MultiBucketMegakernelTable
	tree            *model.RadixKVTree
	scheduler       *model.ContinuousBatchScheduler
	pdEngine        *model.DisaggregatedPDEngine
	telemetry       *autotune.FlightTelemetryRecorder
	barrierReports  map[int]*qualify.FormalBarrierReport
	barrierVerified bool

	mu             sync.Mutex
	liveMu         sync.RWMutex
	liveSimpleIters []KernelStreamIteration
	liveProvenance string
	reqCounter     atomic.Uint64
	epochCounter   atomic.Uint64
	lastBucketM    int
	radixHitCount  int
	radixMissCount int
	radixEvictCnt  int
	c2cPromotedCnt int
}

// NewEGServer initializes and formally verifies a new [EGServer].
func NewEGServer(cfg ServerConfig) (*EGServer, error) {
	cfg = cfg.withDefaults()

	var pgoCache *autotune.VerifiedPGOCache
	if strings.TrimSpace(cfg.PGOCacheDir) != "" {
		pgoCache = autotune.NewVerifiedPGOCache(strings.TrimSpace(cfg.PGOCacheDir))
	}

	telemetry := autotune.NewFlightTelemetryRecorder()

	srv := &EGServer{
		cfg:            cfg,
		mux:            http.NewServeMux(),
		pgoCache:       pgoCache,
		telemetry:      telemetry,
		barrierReports: make(map[int]*qualify.FormalBarrierReport, 4),
		lastBucketM:    1,
	}

	builderTarget := cfg.Arch
	if hopper.UMMATarget(builderTarget) {
		builderTarget = "sm_90a"
	}
	builder := specdecode.DefaultCrossLayerBucketBuilder(cfg.HiddenDim, cfg.IntermDim, cfg.HiddenDim, builderTarget)

	table, err := specdecode.NewMultiBucketMegakernelTable(specdecode.MultiBucketTableConfig{
		Arch:          cfg.Arch,
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.4",
		WeightDigest:  fmt.Sprintf("egserve_h%d_i%d_l%d", cfg.HiddenDim, cfg.IntermDim, cfg.Layers),
		PlanKeyPrefix: "egserve_cross_layer",
		Cache:         pgoCache,
		Builder:       builder,
		OnTransition: func(ctx context.Context, entry *specdecode.BucketMegakernelEntry, tr specdecode.BucketTransition) error {
			telemetry.RecordBucketTransition(time.Now().UTC(), tr.From().BucketM, tr.To().BucketM, 0.85)
			return nil
		},
	})
	if err != nil {
		return nil, fmt.Errorf("egserve init bucket table: %w", err)
	}
	srv.table = table

	// Formally verify every bucket megakernel at startup.
	entries := table.Entries()
	if len(entries) == 0 {
		return nil, errors.New("egserve: multi-bucket table has no entries")
	}
	allVerified := true
	for _, entry := range entries {
		megaCopy := entry.Megakernel
		rep, vErr := qualify.VerifyMegakernelBarrierSafety(&megaCopy)
		if rep != nil {
			srv.barrierReports[entry.Spec.BucketM] = rep
		}
		if vErr != nil || rep == nil || !rep.Passed() {
			allVerified = false
			return nil, fmt.Errorf("egserve startup barrier verification failed for M=%d: %w", entry.Spec.BucketM, vErr)
		}
		// Record startup verification telemetry event per bucket.
		epoch := srv.epochCounter.Add(1)
		grid, _ := entry.Megakernel.Grid()
		telemetry.RecordStageSpan(
			fmt.Sprintf("startup_verify:%s", entry.Megakernel.Name),
			epoch,
			time.Now().UTC(),
			50*time.Microsecond,
			0,
			grid[0],
		)
	}
	srv.barrierVerified = allVerified

	tree, err := model.NewRadixKVTree(cfg.TotalKVPages, cfg.PageSize)
	if err != nil {
		return nil, fmt.Errorf("egserve init radix kv tree: %w", err)
	}
	srv.tree = tree

	sched, err := model.NewContinuousBatchScheduler(tree, table, 8)
	if err != nil {
		return nil, fmt.Errorf("egserve init continuous batch scheduler: %w", err)
	}
	srv.scheduler = sched

	headDim := 64
	heads := max(cfg.HiddenDim/headDim, 4)
	kvHeads := max(heads/2, 1)
	if heads%kvHeads != 0 {
		kvHeads = 1
	}

	pdEngine, err := model.NewDisaggregatedPDEngine(model.PDEngineConfig{
		Arch: model.ArchConfig{
			HiddenSize:        cfg.HiddenDim,
			IntermediateSize:  cfg.IntermDim,
			NumHiddenLayers:   cfg.Layers,
			NumAttentionHeads: heads,
			NumKeyValueHeads:  kvHeads,
			HeadDim:           headDim,
		},
		Generation:       nvl72.GenGB300,
		TotalRacks:       2,
		CacheKind:        nvl72.CacheMLA,
		KVLoRARank:       512,
		RoPEDim:          64,
		PageSize:         cfg.PageSize,
		HBMPageCapacity:  cfg.HBMPageCapacity,
		C2CPageCapacity:  cfg.C2CPageCapacity,
		UseGraceC2CStage: true,
		DecodeTable:      table,
		SLA: nvl72.PDSLA{
			PromptTokens:      1024,
			DecodeBatchPerGPU: 64,
			MaxTTFTMs:         200.0,
			MaxTPOTMs:         20.0,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("egserve init disaggregated pd engine: %w", err)
	}
	srv.pdEngine = pdEngine

	srv.mux.HandleFunc("/", srv.handleUI)
	srv.mux.HandleFunc("/ui", srv.handleUI)
	srv.mux.HandleFunc("/v1/kernels/annotated", srv.handleKernelsAnnotated)
	srv.mux.HandleFunc("/v1/kernels/compile", srv.handleKernelsCompile)
	srv.mux.HandleFunc("/v1/kernels/stream", srv.handleKernelsStream)
	srv.mux.HandleFunc("/v1/completions", srv.handleCompletions)
	srv.mux.HandleFunc("/v1/chat/completions", srv.handleChatCompletions)
	srv.mux.HandleFunc("/v1/flight/trace.json", srv.handleFlightTrace)
	srv.mux.HandleFunc("/v1/pgo/status", srv.handlePGOStatus)

	if cfg.LiveNVCC || strings.TrimSpace(cfg.SSHHost) != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		baseIters := srv.buildSimpleKernelStreamIterations(1)
		if liveIters, prov, err := CompileSimpleKernelTrajectoryLive(ctx, baseIters, LiveNVCCConfig{
			Arch:     cfg.Arch,
			NVCC:     cfg.NVCCPath,
			NVDISASM: cfg.NVDISASMPath,
			SSHHost:  cfg.SSHHost,
		}); err == nil {
			srv.liveSimpleIters = liveIters
			srv.liveProvenance = prov
		}
		cancel()
	}

	return srv, nil
}

// ServeHTTP dispatches HTTP requests to the registered OpenAI and telemetry endpoints.
func (s *EGServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *EGServer) handleCompletions(w http.ResponseWriter, r *http.Request) {
	s.serveCompletionEndpoint(w, r, false)
}

func (s *EGServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.serveCompletionEndpoint(w, r, true)
}

func (s *EGServer) serveCompletionEndpoint(w http.ResponseWriter, r *http.Request, isChat bool) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid json request: %v", err), http.StatusBadRequest)
		return
	}

	resp, err := s.ProcessCompletion(r.Context(), req, isChat)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(resp)
}

// ProcessCompletion executes one OpenAI completion or chat completion request
// against the DisaggregatedPDEngine and ContinuousBatchScheduler, recording
// flight telemetry and returning the enriched response.
func (s *EGServer) ProcessCompletion(ctx context.Context, req CompletionRequest, isChat bool) (*CompletionResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	promptInts, promptInt32s := s.tokenizeRequest(req)
	if len(promptInts) == 0 {
		return nil, errors.New("egserve: empty prompt or messages")
	}

	reqNum := s.reqCounter.Add(1)
	reqID := fmt.Sprintf("cmpl-%d", reqNum)
	modelName := strings.TrimSpace(req.Model)
	if modelName == "" {
		modelName = "egraph-megakernel-" + s.cfg.Arch
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 16
	}

	// Build sequence phase and draft tokens.
	phase := model.PhaseDecode
	draftTokens := req.DraftTokens.Tokens
	acceptRate := 0.85
	if len(draftTokens) > 0 {
		phase = model.PhaseSpeculativeVerify
		acceptRate = 0.90
	}

	seq := &model.InFlightSequence{
		ID:             reqID,
		Phase:          phase,
		PromptTokens:   promptInts,
		DecodeToken:    promptInts[len(promptInts)-1] + 1,
		DraftTokens:    draftTokens,
		AcceptanceRate: acceptRate,
	}

	// Admit sequence into ContinuousBatchScheduler (and its RadixKVTree).
	usedBefore := s.tree.UsedPages()
	if err := s.scheduler.Admit(seq); err != nil {
		return nil, fmt.Errorf("egserve admit scheduler: %w", err)
	}

	// Optional brief coalescing window so concurrent requests can share a batch step.
	if s.cfg.BatchWindow > 0 {
		time.Sleep(s.cfg.BatchWindow)
	}

	s.mu.Lock()
	stepStart := time.Now().UTC()

	// Admit into DisaggregatedPDEngine for two-tier HBM3e + C2C prefix matching
	// and context-parallel prefill + KV transfer planning.
	pdRes, err := s.pdEngine.AdmitAndPrefill(model.PDRequest{
		ReqID:               reqID,
		PromptTokens:        promptInt32s,
		ReleaseAfterPrefill: true,
	})
	if err != nil {
		s.mu.Unlock()
		_ = s.scheduler.Complete(reqID)
		return nil, fmt.Errorf("egserve admit pd engine: %w", err)
	}

	// Schedule continuous batching step across active sequences.
	step, err := s.scheduler.ScheduleStep(ctx)
	if err != nil {
		s.mu.Unlock()
		_ = s.scheduler.Complete(reqID)
		return nil, fmt.Errorf("egserve schedule step: %w", err)
	}

	// Also step the DisaggregatedPDEngine decode pool for draft token telemetry.
	_, pdKernelName, _ := s.pdEngine.StepDecodePool(ctx, max(step.ActiveTokens-1, len(draftTokens)), acceptRate)

	// Rebalance P/D pools if estimated TTFT exceeds SLA.
	if pdRes.EstimatedPrefillTTFTMs > 0 {
		_, _, _ = s.pdEngine.RebalanceToSLA(max(pdRes.EstimatedPrefillTTFTMs, 1.0), 10.0)
	}

	// Complete sequence so its RadixKVTree node becomes a reusable prefix in cache.
	_ = s.scheduler.Complete(reqID)
	usedAfter := s.tree.UsedPages()

	// Update Radix KV cache hit/miss/evict counters.
	radixHitPages := max(seq.CachedPages, pdRes.Receipt.HBMHitPages+pdRes.Receipt.C2CPromotedPages)
	radixHitTokens := max(seq.CachedPages*s.tree.PageSize(), pdRes.Receipt.PrefillTokensSaved)
	if radixHitPages > 0 {
		s.radixHitCount += radixHitPages
	}
	if pdRes.Receipt.NewlyAllocatedPages > 0 {
		s.radixMissCount += pdRes.Receipt.NewlyAllocatedPages
	} else if radixHitPages == 0 {
		s.radixMissCount++
	}
	if pdRes.Receipt.EvictedToC2CPages > 0 {
		s.radixEvictCnt += pdRes.Receipt.EvictedToC2CPages
	} else if usedAfter < usedBefore {
		s.radixEvictCnt += usedBefore - usedAfter
	}
	s.c2cPromotedCnt += pdRes.Receipt.C2CPromotedPages

	// Record bucket transition if the scheduled bucket changed and wasn't already recorded.
	prevBucketM := s.lastBucketM
	if prevBucketM == 0 {
		prevBucketM = 1
	}
	if step.BucketM != prevBucketM {
		s.lastBucketM = step.BucketM
		if !step.Transition.Swapped {
			s.telemetry.RecordBucketTransition(time.Now().UTC(), prevBucketM, step.BucketM, acceptRate)
		}
	}

	kernelName := pdKernelName
	if step.BucketEntry != nil && step.BucketEntry.Megakernel.Name != "" {
		kernelName = step.BucketEntry.Megakernel.Name
	}

	epoch := s.epochCounter.Add(1)
	blocks := 16
	if step.BucketEntry != nil {
		grid, _ := step.BucketEntry.Megakernel.Grid()
		if grid[0] > 0 {
			blocks = grid[0]
		}
	}
	stepDur := max(time.Since(stepStart), 25*time.Microsecond)
	s.telemetry.RecordStageSpan(
		fmt.Sprintf("batch_step:%s:M=%d", kernelName, step.BucketM),
		epoch,
		stepStart,
		stepDur,
		0,
		blocks,
	)
	plan := s.pdEngine.Plan()
	s.mu.Unlock()

	generatedText := fmt.Sprintf("egraph-megakernel(%s, M=%d, hit_tokens=%d)", kernelName, step.BucketM, radixHitTokens)
	choice := CompletionChoice{
		Index:        0,
		FinishReason: "stop",
	}
	if isChat {
		choice.Message = &ChatMessage{
			Role:    "assistant",
			Content: generatedText,
		}
	} else {
		choice.Text = generatedText
	}

	completionTokens := 1 + len(draftTokens)
	if maxTokens > 0 && completionTokens > maxTokens {
		completionTokens = maxTokens
	}

	return &CompletionResponse{
		ID:      reqID,
		Object:   objectType(isChat),
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []CompletionChoice{choice},
		Usage: CompletionUsage{
			PromptTokens:     len(promptInts),
			CompletionTokens: completionTokens,
			TotalTokens:      len(promptInts) + completionTokens,
		},
		EGraphFlight: EGraphFlightMetadata{
			ActiveBucketM:          step.BucketM,
			ActiveTokens:           step.ActiveTokens,
			PaddedRows:             step.PaddedRows,
			ActiveRowMask:          step.ActiveRowMask,
			RadixPrefixHitTokens:   radixHitTokens,
			C2CPromotedPages:       pdRes.Receipt.C2CPromotedPages,
			MegakernelKernelName:   kernelName,
			BarrierSafetyVerified:  s.barrierVerified,
			EstimatedPrefillTTFTMs: pdRes.EstimatedPrefillTTFTMs,
			PrefillGPUs:            plan.PrefillGPUs,
			DecodeGPUs:             plan.DecodeGPUs,
		},
	}, nil
}

func objectType(isChat bool) string {
	if isChat {
		return "chat.completion"
	}
	return "text_completion"
}

// tokenizeRequest deterministically converts either ChatMessages or Prompt into
// page-aligned token slices so shared conversation prefixes map cleanly to
// physical KV page boundaries in both RadixKVTree and DisaggregatedPDEngine.
func (s *EGServer) tokenizeRequest(req CompletionRequest) ([]int, []int32) {
	pageSize := max(s.cfg.PageSize, 1)
	var toks []int

	if len(req.Messages) > 0 {
		for _, msg := range req.Messages {
			msgToks := tokenizeTextSegment(msg.Role+":"+msg.Content, pageSize)
			toks = append(toks, msgToks...)
		}
	} else if req.Prompt != nil {
		switch v := req.Prompt.(type) {
		case string:
			if strings.TrimSpace(v) != "" {
				toks = tokenizeTextSegment(v, pageSize)
			}
		case []any:
			for _, item := range v {
				switch elem := item.(type) {
				case float64:
					toks = append(toks, int(elem))
				case string:
					toks = append(toks, tokenizeTextSegment(elem, pageSize)...)
				}
			}
		}
	}

	if len(toks) == 0 {
		return nil, nil
	}
	toks32 := make([]int32, len(toks))
	for i, t := range toks {
		toks32[i] = int32(t)
	}
	return toks, toks32
}

// tokenizeTextSegment converts text into a deterministic slice of token IDs whose
// length is a positive multiple of pageSize. Words separated by whitespace each
// produce one token, and the segment is padded deterministically to pageSize
// alignment so each message forms an exact multiple of KV cache pages.
func tokenizeTextSegment(text string, pageSize int) []int {
	if pageSize <= 0 {
		pageSize = 4
	}
	fields := strings.Fields(text)
	if len(fields) == 0 {
		fields = []string{text}
	}
	var out []int
	for i, word := range fields {
		h := fnv.New32a()
		_, _ = h.Write([]byte(word))
		var idxBuf [4]byte
		binary.LittleEndian.PutUint32(idxBuf[:], uint32(i))
		_, _ = h.Write(idxBuf[:])
		tok := int(h.Sum32()&0x7fffff) + 10
		out = append(out, tok)
	}
	rem := len(out) % pageSize
	if rem != 0 || len(out) == 0 {
		pad := pageSize - rem
		baseHash := fnv.New32a()
		_, _ = baseHash.Write([]byte(text))
		seed := int(baseHash.Sum32()&0x7fffff) + 100
		for i := range pad {
			out = append(out, seed+i+1)
		}
	}
	return out
}

func (s *EGServer) handleFlightTrace(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := s.telemetry.ExportChromeTraceJSON(w); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *EGServer) handlePGOStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	status := s.Status()
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(status)
}

// Status returns the current [PGOStatusResponse] snapshot and records a PGO
// status inspection event in the flight telemetry recorder.
func (s *EGServer) Status() PGOStatusResponse {
	s.mu.Lock()
	defer s.mu.Unlock()

	epoch := s.epochCounter.Add(1)
	s.telemetry.RecordStageSpan(
		"pgo_status_snapshot",
		epoch,
		time.Now().UTC(),
		10*time.Microsecond,
		0,
		1,
	)

	entries := s.table.Entries()
	buckets := make([]BucketStatusEntry, 0, len(entries))
	for _, e := range entries {
		rep := s.barrierReports[e.Spec.BucketM]
		verified := rep != nil && rep.Passed()
		buckets = append(buckets, BucketStatusEntry{
			BucketM:               e.Spec.BucketM,
			Regime:                string(e.Spec.Regime),
			KernelName:            e.Megakernel.Name,
			NumStages:             len(e.Megakernel.Stages),
			CacheHit:              e.CacheHit,
			BarrierSafetyVerified: verified,
		})
	}

	hbmPages, c2cPages := s.pdEngine.RadixStats()
	hits := max(s.radixHitCount, s.tree.CacheHits())
	radixStats := RadixKVCacheStats{
		HitCount:   hits,
		MissCount:  s.radixMissCount,
		EvictCount: s.radixEvictCnt,
		UsedPages:  s.tree.UsedPages(),
		FreePages:  s.tree.FreePages(),
		HBMPages:   hbmPages,
		C2CPages:   c2cPages,
	}

	plan := s.pdEngine.Plan()
	pdRatio := 0.0
	if plan.DecodeGPUs > 0 {
		pdRatio = float64(plan.PrefillGPUs) / float64(plan.DecodeGPUs)
	}
	pdPool := PDPoolStatus{
		Generation:  plan.Generation,
		PrefillGPUs: plan.PrefillGPUs,
		DecodeGPUs:  plan.DecodeGPUs,
		PDRatio:     pdRatio,
		MeetsSLA:    plan.MeetsSLA,
	}

	return PGOStatusResponse{
		Arch:                  s.cfg.Arch,
		BarrierSafetyVerified: s.barrierVerified,
		BarrierReports:        s.barrierReports,
		Buckets:               buckets,
		BucketHotSwaps:        s.table.HotSwaps(),
		HitCount:              radixStats.HitCount,
		MissCount:             radixStats.MissCount,
		EvictCount:            radixStats.EvictCount,
		RadixKVCache:          radixStats,
		PrefillGPUs:           plan.PrefillGPUs,
		DecodeGPUs:            plan.DecodeGPUs,
		PDPool:                pdPool,
	}
}

// SelfTestSummary is the deterministic JSON report emitted by `egserve -self-test`.
type SelfTestSummary struct {
	Arch                  string               `json:"arch"`
	BarrierSafetyVerified bool                 `json:"barrier_safety_verified"`
	Turn1PrefixHitTokens  int                  `json:"turn1_prefix_hit_tokens"`
	Turn2PrefixHitTokens  int                  `json:"turn2_prefix_hit_tokens"`
	ObservedBuckets       []int                `json:"observed_buckets"`
	TraceEventCount       int                  `json:"trace_event_count"`
	PGOStatus             PGOStatusResponse    `json:"pgo_status"`
	Responses             []CompletionResponse `json:"responses"`
}

// RunSelfTest executes a deterministic multi-turn chat and multi-bucket
// speculative batching cycle against s and returns a [SelfTestSummary].
func (s *EGServer) RunSelfTest(ctx context.Context) (*SelfTestSummary, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	sysMsg := ChatMessage{
		Role:    "system",
		Content: "You are the NVIDIA e-graph persistent megakernel inference assistant.",
	}

	// Turn 1: initial system + user prompt (cold cache).
	turn1Req := CompletionRequest{
		Model: "egraph-llama",
		Messages: []ChatMessage{
			sysMsg,
			{Role: "user", Content: "Explain cross-layer megakernel barrier dominance."},
		},
		MaxTokens:   16,
		Temperature: 0.0,
		DraftTokens: DraftTokensSpec{Count: 0},
	}
	resp1, err := s.ProcessCompletion(ctx, turn1Req, true)
	if err != nil {
		return nil, fmt.Errorf("self-test turn 1: %w", err)
	}

	// Turn 2: shares system prompt + turn 1 history, exercising Radix KV prefix hits and M=2 bucket.
	turn2Req := CompletionRequest{
		Model: "egraph-llama",
		Messages: []ChatMessage{
			sysMsg,
			{Role: "user", Content: "Explain cross-layer megakernel barrier dominance."},
			{Role: "assistant", Content: resp1.Choices[0].Message.Content},
			{Role: "user", Content: "How does Grace C2C LPDDR5X tiering reduce TTFT on turn 2?"},
		},
		MaxTokens:   16,
		Temperature: 0.0,
		DraftTokens: DraftTokensSpec{Count: 1, Tokens: []int{901}},
	}
	resp2, err := s.ProcessCompletion(ctx, turn2Req, true)
	if err != nil {
		return nil, fmt.Errorf("self-test turn 2: %w", err)
	}

	// Turn 3: speculative tree verify with 3 draft tokens -> M=4 bucket.
	turn3Req := CompletionRequest{
		Model: "egraph-llama",
		Messages: []ChatMessage{
			sysMsg,
			{Role: "user", Content: "Verify M=4 speculative tree step."},
		},
		MaxTokens:   16,
		Temperature: 0.0,
		DraftTokens: DraftTokensSpec{Count: 3, Tokens: []int{901, 902, 903}},
	}
	resp3, err := s.ProcessCompletion(ctx, turn3Req, true)
	if err != nil {
		return nil, fmt.Errorf("self-test turn 3: %w", err)
	}

	// Turn 4: speculative tree verify with 7 draft tokens -> M=8 bucket.
	turn4Req := CompletionRequest{
		Model: "egraph-llama",
		Messages: []ChatMessage{
			sysMsg,
			{Role: "user", Content: "Verify M=8 speculative tree step."},
		},
		MaxTokens:   16,
		Temperature: 0.0,
		DraftTokens: DraftTokensSpec{Count: 7, Tokens: []int{901, 902, 903, 904, 905, 906, 907}},
	}
	resp4, err := s.ProcessCompletion(ctx, turn4Req, true)
	if err != nil {
		return nil, fmt.Errorf("self-test turn 4: %w", err)
	}

	status := s.Status()
	events := s.telemetry.Events()

	return &SelfTestSummary{
		Arch:                  s.cfg.Arch,
		BarrierSafetyVerified: s.barrierVerified,
		Turn1PrefixHitTokens:  resp1.EGraphFlight.RadixPrefixHitTokens,
		Turn2PrefixHitTokens:  resp2.EGraphFlight.RadixPrefixHitTokens,
		ObservedBuckets: []int{
			resp1.EGraphFlight.ActiveBucketM,
			resp2.EGraphFlight.ActiveBucketM,
			resp3.EGraphFlight.ActiveBucketM,
			resp4.EGraphFlight.ActiveBucketM,
		},
		TraceEventCount: len(events),
		PGOStatus:       status,
		Responses:       []CompletionResponse{*resp1, *resp2, *resp3, *resp4},
	}, nil
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("egserve", flag.ContinueOnError)
	fs.SetOutput(stderr)

	addr := fs.String("addr", ":8080", "HTTP listen address")
	arch := fs.String("arch", "sm_121a", "target GPU architecture (e.g. sm_121a, sm_100a, sm_90a)")
	hiddenDim := fs.Int("hidden-dim", 256, "transformer hidden dimension")
	intermDim := fs.Int("interm-dim", 512, "MLP intermediate dimension")
	layers := fs.Int("layers", 4, "number of transformer layers")
	pgoCacheDir := fs.String("pgo-cache-dir", "", "optional directory for VerifiedPGOCache")
	liveNVCC := fs.Bool("live-nvcc", false, "compile and disassemble simple trajectory live with local nvcc/nvdisasm")
	sshDGX := fs.String("ssh-dgx", "", "optional SSH host (e.g. dgx) for live nvcc/ptxas/nvdisasm compilation")
	selfTest := fs.Bool("self-test", false, "run deterministic multi-turn request cycle and print JSON summary")
	demoStream := fs.Bool("demo-stream", false, "print inline-annotated megakernel stream preview and inspector URLs")

	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: egserve [-addr :8080] [-arch sm_121a] [-hidden-dim 256] [-interm-dim 512] [-layers 4] [-pgo-cache-dir dir] [-live-nvcc] [-ssh-dgx host] [-self-test] [-demo-stream]\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fs.Usage()
		return 2
	}
	if *hiddenDim <= 0 || *intermDim <= 0 || *layers <= 0 {
		fmt.Fprintf(stderr, "egserve: dimensions and layers must be > 0\n")
		return 2
	}

	rawAddr := strings.TrimSpace(*addr)
	srv, err := NewEGServer(ServerConfig{
		Addr:        rawAddr,
		Arch:        *arch,
		HiddenDim:   *hiddenDim,
		IntermDim:   *intermDim,
		Layers:      *layers,
		PGOCacheDir: *pgoCacheDir,
		LiveNVCC:    *liveNVCC,
		SSHHost:     *sshDGX,
	})
	if err != nil {
		fmt.Fprintf(stderr, "egserve: %v\n", err)
		return 1
	}

	if *demoStream {
		preview, err := srv.FormatDemoStreamSummary(rawAddr)
		if err != nil {
			fmt.Fprintf(stderr, "egserve demo-stream: %v\n", err)
			return 1
		}
		_, _ = io.WriteString(stdout, preview)
		if rawAddr == "" && !*selfTest {
			return 0
		}
	}

	if *selfTest {
		summary, err := srv.RunSelfTest(context.Background())
		if err != nil {
			fmt.Fprintf(stderr, "egserve self-test: %v\n", err)
			return 1
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(summary); err != nil {
			fmt.Fprintf(stderr, "egserve encode self-test summary: %v\n", err)
			return 1
		}
		return 0
	}

	httpSrv := &http.Server{
		Addr:    srv.cfg.Addr,
		Handler: srv,
	}
	if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fmt.Fprintf(stderr, "egserve listen: %v\n", err)
		return 1
	}
	return 0
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}
