package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
	"github.com/tmc/lib/nvidia/x/nvl72"
)

// PDEngineConfig configures a [DisaggregatedPDEngine] uniting NVL72 / Vera
// Rubin prefill/decode disaggregation planning, two-tier HBM3e + Grace/Vera
// LPDDR5X NVLink-C2C radix prefix caching, 2D context-parallel prefill
// scheduling, zero-copy .sys-scope KV/MLA latent page streaming, and
// multi-bucket cross-layer speculative decode pool execution.
type PDEngineConfig struct {
	Arch              ArchConfig
	Generation        string // "GB200", "GB300", "VeraRubin-NVL144", or "RubinUltra-NVL576"
	TotalRacks        int
	PrefillRacks      int
	DecodeRacks       int
	SLA               nvl72.PDSLA
	CacheKind         nvl72.AttentionCacheKind // CacheGQA or CacheMLA
	KVLoRARank        int                      // MLA compressed latent rank in FP8 bytes (default 512)
	RoPEDim           int                      // MLA decoupled RoPE dimension in BF16 elements (default 64)
	PageSize          int                      // tokens per KV/MLA page (default 16)
	HBMPageCapacity   int                      // Tier-1 HBM3e page capacity
	C2CPageCapacity   int                      // Tier-2 Grace/Vera LPDDR5X C2C page capacity
	CPMode            nvl72.ContextParallelMode
	CPUlyssesDeg      int
	CPRingDeg         int
	CPTarget          string // "sm_100a", "sm_103a", or "sm_107a"
	TransferRanks     int
	UseGraceC2CStage  bool
	UseRubinDisagg    bool
	DecodeTableConfig specdecode.MultiBucketTableConfig
	DecodeTable       *specdecode.MultiBucketMegakernelTable
	MaxTokenBudget    int // token budget per chunked prefill step (default 4096)
}

// PDRequest describes a prefill/decode request submitted to
// [DisaggregatedPDEngine.AdmitAndPrefill].
type PDRequest struct {
	ReqID               string
	PromptTokens        []int32
	TokenPages          [][]int32
	ReleaseAfterPrefill bool
	ActiveDecodes       []string
	CPMode              nvl72.ContextParallelMode
	HasCPMode           bool
	CPUlyssesDeg        int
	CPRingDeg           int
}

// PDRequestTickResult reports the prefix-cache allocation receipt, 2D context
// parallel prefill schedule, chunked prefill batch steps, and zero-copy
// layer-pipelined KV/MLA page transfer kernel emitted by
// [DisaggregatedPDEngine.AdmitAndPrefill].
type PDRequestTickResult struct {
	ReqID                  string
	Receipt                nvl72.AllocationReceipt
	CPSchedule             nvl72.ContextParallelSchedule
	CPPlan                 [][]nvl72.RingStepPlan
	CPImbalanceRatio       float64
	ChunkedSteps           []nvl72.ChunkedBatchStep
	TransferKernel         nvl72.NVL72KVPageTransfer
	TransferKernelSource   string
	LayerTransferPerPageUs float64
	EstimatedPrefillTTFTMs float64
	PDPlan                 nvl72.PDDisaggPlan
	RubinPlan              *nvl72.RubinDisaggPlan
}

// DisaggregatedPDEngine coordinates disaggregated prefill and decode pools,
// two-tier Radix KV/MLA prefix caching (Tier-1 HBM3e + Tier-2 Grace/Vera C2C
// LPDDR5X), context-parallel prefill scheduling, zero-copy .sys-scope NVLink /
// RDMA page streaming, and multi-bucket cross-layer decode megakernel
// selection.
//
// The zero value is not usable; construct via [NewDisaggregatedPDEngine].
type DisaggregatedPDEngine struct {
	mu          sync.Mutex
	cfg         PDEngineConfig
	modelSpec   nvl72.PDModelSpec
	sla         nvl72.PDSLA
	pdPlan      nvl72.PDDisaggPlan
	rubinPlan   *nvl72.RubinDisaggPlan
	radix       *nvl72.RadixPageAllocator
	decodeTable *specdecode.MultiBucketMegakernelTable
	pageSize    int
	pageBytes   int
}

func defaultPDModelSpecFromArch(arch ArchConfig, cacheKind nvl72.AttentionCacheKind, kvLoRA, ropeDim int) nvl72.PDModelSpec {
	layers := arch.NumHiddenLayers
	if layers < 1 {
		layers = 32
	}
	hidden := arch.HiddenSize
	if hidden < 2 {
		hidden = 4096
	}
	if hidden%2 != 0 {
		hidden++
	}
	qHeads := arch.NumAttentionHeads
	if qHeads < 1 {
		qHeads = 32
	}
	kvHeads := arch.NumKeyValueHeads
	if kvHeads < 1 {
		kvHeads = qHeads
	}
	headDim := arch.HeadDim
	if headDim < 1 {
		headDim = max(hidden/qHeads, 64)
	}
	if cacheKind == "" {
		cacheKind = nvl72.CacheGQA
	}
	if kvLoRA <= 0 {
		kvLoRA = 512
	}
	if ropeDim < 0 {
		ropeDim = 64
	}
	if cacheKind == nvl72.CacheMLA && ropeDim == 0 {
		ropeDim = 64
	}
	topK := arch.NumExpertsPerTok
	if topK < 1 {
		topK = 8
	}
	experts := arch.NumExperts
	if experts < topK {
		experts = max(topK, 64)
	}
	return nvl72.PDModelSpec{
		NumLayers:    layers,
		HiddenSize:   hidden,
		NumQHeads:    qHeads,
		NumKVHeads:   kvHeads,
		HeadDim:      headDim,
		CacheKind:    cacheKind,
		KVLoRARank:   kvLoRA,
		RoPEDim:      ropeDim,
		TotalExperts: experts,
		TopK:         topK,
	}
}

func isRubinGeneration(gen string, useRubin bool) bool {
	if useRubin {
		return true
	}
	g := strings.ToLower(strings.TrimSpace(gen))
	return strings.Contains(g, "rubin") || strings.Contains(g, "nvl144") || strings.Contains(g, "nvl576")
}

// NewDisaggregatedPDEngine constructs and initializes a [DisaggregatedPDEngine]
// from cfg.
func NewDisaggregatedPDEngine(cfg PDEngineConfig) (*DisaggregatedPDEngine, error) {
	if cfg.Arch.HiddenSize == 0 {
		cfg.Arch = ArchConfig{
			HiddenSize:        4096,
			IntermediateSize:  14336,
			NumHiddenLayers:   32,
			NumAttentionHeads: 32,
			NumKeyValueHeads:  8,
			HeadDim:           128,
		}
	}
	normArch, err := ValidateArchConfig(cfg.Arch)
	if err != nil {
		return nil, fmt.Errorf("disaggregated pd engine: %w", err)
	}
	cfg.Arch = normArch

	gen := strings.TrimSpace(cfg.Generation)
	if gen == "" {
		gen = nvl72.GenGB300
	}
	cfg.Generation = gen

	totalRacks := cfg.TotalRacks
	if totalRacks < 1 {
		if cfg.PrefillRacks > 0 || cfg.DecodeRacks > 0 {
			totalRacks = max(cfg.PrefillRacks, 1) + max(cfg.DecodeRacks, 1)
		} else {
			totalRacks = 2
		}
	}
	cfg.TotalRacks = totalRacks

	sla := cfg.SLA
	if sla.PromptTokens < 1 {
		sla.PromptTokens = 4096
	}
	if sla.DecodeBatchPerGPU < 1 {
		sla.DecodeBatchPerGPU = 64
	}
	if sla.MaxTTFTMs <= 0 {
		sla.MaxTTFTMs = 250.0
	}
	if sla.MaxTPOTMs <= 0 {
		sla.MaxTPOTMs = 25.0
	}
	cfg.SLA = sla

	pageSize := cfg.PageSize
	if pageSize <= 0 {
		pageSize = 16
	}
	cfg.PageSize = pageSize

	modelSpec := defaultPDModelSpecFromArch(cfg.Arch, cfg.CacheKind, cfg.KVLoRARank, cfg.RoPEDim)

	var kvBytesPerTokPerLayer int64
	if modelSpec.CacheKind == nvl72.CacheMLA {
		kvBytesPerTokPerLayer = int64(modelSpec.KVLoRARank + modelSpec.RoPEDim*2)
	} else {
		kvBytesPerTokPerLayer = int64(2 * modelSpec.NumKVHeads * modelSpec.HeadDim * 2)
	}
	pageBytes := int(int64(pageSize) * kvBytesPerTokPerLayer)
	if pageBytes < 16 {
		pageBytes = 256
	}

	hbmCap := cfg.HBMPageCapacity
	if hbmCap <= 0 {
		hbmCap = 256
	}
	c2cCap := cfg.C2CPageCapacity
	if c2cCap < 0 {
		return nil, fmt.Errorf("disaggregated pd engine: C2CPageCapacity %d must be >= 0", c2cCap)
	}
	if c2cCap == 0 && cfg.HBMPageCapacity == 0 {
		c2cCap = 512
	}

	radix, err := nvl72.NewRadixPageAllocator(pageSize, pageBytes, hbmCap, c2cCap)
	if err != nil {
		return nil, fmt.Errorf("disaggregated pd engine radix allocator: %w", err)
	}

	var (
		pdPlan    nvl72.PDDisaggPlan
		rubinPlan *nvl72.RubinDisaggPlan
	)
	if isRubinGeneration(gen, cfg.UseRubinDisagg) {
		pRacks := cfg.PrefillRacks
		dRacks := cfg.DecodeRacks
		if pRacks < 1 && dRacks < 1 {
			pRacks = max(totalRacks/2, 1)
			dRacks = max(totalRacks-pRacks, 1)
		} else {
			if pRacks < 1 {
				pRacks = 1
			}
			if dRacks < 1 {
				dRacks = 1
			}
		}
		decodeSpec := nvl72.DefaultVeraRubinNVL144Spec()
		if strings.Contains(strings.ToLower(gen), "576") || strings.Contains(strings.ToLower(gen), "ultra") {
			decodeSpec = nvl72.DefaultRubinUltraNVL576Spec()
		}
		rp, rErr := nvl72.PlanRubinDisaggregation(nvl72.RubinDisaggConfig{
			PrefillSpec:               nvl72.DefaultVeraRubinNVL144Spec(),
			DecodeSpec:                decodeSpec,
			PrefillRacks:              pRacks,
			DecodeRacks:               dRacks,
			Model:                     modelSpec,
			SLA:                       sla,
			PageTokens:                pageSize,
			UseLUTDecompressedPrefill: true,
			UseVeraC2CStaging:         cfg.UseGraceC2CStage,
		})
		if rErr != nil {
			return nil, fmt.Errorf("disaggregated pd engine rubin plan: %w", rErr)
		}
		rubinPlan = &rp
		pdPlan = rubinToPDDisaggPlan(rp)
	} else {
		pdPlan, err = nvl72.PlanPDDisaggregation(gen, totalRacks, sla, modelSpec)
		if err != nil {
			return nil, fmt.Errorf("disaggregated pd engine plan: %w", err)
		}
	}

	decodeTable := cfg.DecodeTable
	if decodeTable == nil {
		tcfg := cfg.DecodeTableConfig
		if strings.TrimSpace(tcfg.Arch) == "" {
			tcfg.Arch = "sm_100a"
		}
		decodeTable, err = specdecode.NewMultiBucketMegakernelTable(tcfg)
		if err != nil {
			return nil, fmt.Errorf("disaggregated pd engine decode table: %w", err)
		}
	}

	if cfg.MaxTokenBudget <= 0 {
		cfg.MaxTokenBudget = max(sla.PromptTokens, 4096)
	}

	return &DisaggregatedPDEngine{
		cfg:         cfg,
		modelSpec:   modelSpec,
		sla:         sla,
		pdPlan:      pdPlan,
		rubinPlan:   rubinPlan,
		radix:       radix,
		decodeTable: decodeTable,
		pageSize:    pageSize,
		pageBytes:   pageBytes,
	}, nil
}

func rubinToPDDisaggPlan(rp nvl72.RubinDisaggPlan) nvl72.PDDisaggPlan {
	transport := "nvlink6-intra-rack"
	if rp.PrefillRacks+rp.DecodeRacks > 1 {
		transport = "connectx9-1600g-rdma"
	}
	return nvl72.PDDisaggPlan{
		Generation:           strings.ToLower(string(rp.PrefillTier)),
		TotalRacks:           rp.PrefillRacks + rp.DecodeRacks,
		PrefillRacks:         rp.PrefillRacks,
		DecodeRacks:          rp.DecodeRacks,
		PrefillGPUs:          rp.PrefillDies,
		DecodeGPUs:           rp.DecodeDies,
		KVBytesPerToken:      rp.KVBytesPerToken,
		TotalKVTransferBytes: rp.TotalKVTransferBytes,
		PrefillComputeMs:     rp.PrefillComputeMS,
		KVTransferMs:         rp.TotalKVTransferMS,
		EstimatedTTFTMs:      rp.EstimatedTTFTMS,
		EstimatedTPOTMs:      rp.EstimatedTPOTMS,
		TransferTransport:    transport,
		MeetsSLA:             rp.MeetsSLA,
		TokensPerSecCluster:  rp.ClusterTokensPerSec,
	}
}

// Plan returns the engine's current [nvl72.PDDisaggPlan].
func (e *DisaggregatedPDEngine) Plan() nvl72.PDDisaggPlan {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.pdPlan
}

// RubinPlan returns the engine's current [nvl72.RubinDisaggPlan] when configured
// for Vera Rubin NVL144 / Rubin Ultra NVL576 disaggregation, or nil otherwise.
func (e *DisaggregatedPDEngine) RubinPlan() *nvl72.RubinDisaggPlan {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.rubinPlan == nil {
		return nil
	}
	cp := *e.rubinPlan
	return &cp
}

// ReleaseRequest releases the radix prefix-tree page pins held by reqID so
// zero-refcount HBM3e pages can be demoted to Tier-2 Grace/Vera C2C LPDDR5X on
// future allocations.
func (e *DisaggregatedPDEngine) ReleaseRequest(reqID string) error {
	if e == nil || e.radix == nil {
		return errors.New("disaggregated pd engine: nil engine")
	}
	return e.radix.ReleaseRequest(reqID)
}

// RadixStats returns the number of KV/MLA cache pages currently resident in
// Tier-1 HBM3e and Tier-2 Grace/Vera C2C LPDDR5X memory.
func (e *DisaggregatedPDEngine) RadixStats() (hbmPages, c2cPages int) {
	if e == nil || e.radix == nil {
		return 0, 0
	}
	return e.radix.HBMUsedPages(), e.radix.C2CUsedPages()
}

func partitionTokensIntoPages(tokens []int32, pageSize int) [][]int32 {
	if pageSize <= 0 {
		pageSize = 16
	}
	if len(tokens) == 0 {
		return nil
	}
	numPages := (len(tokens) + pageSize - 1) / pageSize
	pages := make([][]int32, 0, numPages)
	for i := 0; i < len(tokens); i += pageSize {
		end := min(i+pageSize, len(tokens))
		page := make([]int32, pageSize)
		copy(page, tokens[i:end])
		// Pad partial trailing page deterministically so page length equals pageSize.
		for j := end - i; j < pageSize; j++ {
			page[j] = -1
		}
		pages = append(pages, page)
	}
	return pages
}

// AdmitAndPrefill matches or allocates prefix pages in the two-tier Radix KV
// allocator (reusing Tier-1 HBM3e pages and promoting Tier-2 Grace C2C pages),
// plans the 2D context-parallel prefill schedule for uncached tokens, schedules
// chunked prefill steps alongside active decodes, and emits the zero-copy
// .sys-scope [nvl72.NVL72KVPageTransfer] kernel for streaming newly computed
// KV/MLA pages to the decode pool.
func (e *DisaggregatedPDEngine) AdmitAndPrefill(req PDRequest) (*PDRequestTickResult, error) {
	if e == nil || e.radix == nil {
		return nil, errors.New("disaggregated pd engine: nil engine")
	}
	reqID := strings.TrimSpace(req.ReqID)
	if reqID == "" {
		return nil, errors.New("disaggregated pd engine: empty ReqID")
	}

	pages := req.TokenPages
	if len(pages) == 0 {
		pages = partitionTokensIntoPages(req.PromptTokens, e.pageSize)
	}
	if len(pages) == 0 {
		return nil, fmt.Errorf("disaggregated pd engine: request %q has no prompt tokens or token pages", reqID)
	}

	receipt, err := e.radix.MatchOrAllocatePrefix(reqID, pages)
	if err != nil {
		return nil, fmt.Errorf("disaggregated pd engine admit %q: %w", reqID, err)
	}
	if req.ReleaseAfterPrefill {
		if rErr := e.radix.ReleaseRequest(reqID); rErr != nil {
			return nil, fmt.Errorf("disaggregated pd engine release %q: %w", reqID, rErr)
		}
	}

	e.mu.Lock()
	cfg := e.cfg
	modelSpec := e.modelSpec
	pdPlan := e.pdPlan
	var rubinCopy *nvl72.RubinDisaggPlan
	if e.rubinPlan != nil {
		rc := *e.rubinPlan
		rubinCopy = &rc
	}
	e.mu.Unlock()

	cpMode := cfg.CPMode
	if req.HasCPMode {
		cpMode = req.CPMode
	}
	ulysses := cfg.CPUlyssesDeg
	if req.CPUlyssesDeg > 0 {
		ulysses = req.CPUlyssesDeg
	}
	ring := cfg.CPRingDeg
	if req.CPRingDeg > 0 {
		ring = req.CPRingDeg
	}

	cpSched := buildCompatibleCPSchedule(cpMode, ulysses, ring, cfg.CPTarget, modelSpec, max(receipt.PrefillTokensToCompute, len(pages)*e.pageSize))
	cpPlan, err := nvl72.PlanContextParallel(cpSched)
	if err != nil {
		return nil, fmt.Errorf("disaggregated pd engine context parallel plan: %w", err)
	}
	cpImbalance := nvl72.ImbalanceRatio(cpPlan)

	totalPromptTokens := len(pages) * e.pageSize
	if len(req.PromptTokens) > totalPromptTokens {
		totalPromptTokens = len(req.PromptTokens)
	}
	cachedTokens := min(receipt.PrefillTokensSaved, totalPromptTokens)
	chunkReqs := make([]nvl72.ChunkedBatchRequest, 0, len(req.ActiveDecodes)+1)
	for i, decID := range req.ActiveDecodes {
		if strings.TrimSpace(decID) == "" {
			decID = fmt.Sprintf("decode-%d", i)
		}
		chunkReqs = append(chunkReqs, nvl72.ChunkedBatchRequest{
			ReqID:    decID,
			IsDecode: true,
		})
	}
	chunkReqs = append(chunkReqs, nvl72.ChunkedBatchRequest{
		ReqID:              reqID,
		TotalPromptTokens:  totalPromptTokens,
		CachedPrefixTokens: cachedTokens,
		IsDecode:           false,
	})
	budget := max(cfg.MaxTokenBudget, len(req.ActiveDecodes)+64)
	chunkSteps, err := nvl72.PlanChunkedPrefillBatch(chunkReqs, budget)
	if err != nil {
		return nil, fmt.Errorf("disaggregated pd engine chunked prefill: %w", err)
	}

	transferPages := max(receipt.NewlyAllocatedPages, 1)
	transferRanks := cfg.TransferRanks
	if transferRanks <= 0 {
		transferRanks = min(max(pdPlan.PrefillGPUs+pdPlan.DecodeGPUs, 8), nvl72.NVL72MaxRanks)
	}
	xferKernel := nvl72.NVL72KVPageTransfer{
		Name:               "nvl72_kv_page_transfer",
		NumPages:           transferPages,
		PageBytes:          e.pageBytes,
		UseMLALatent:       modelSpec.CacheKind == nvl72.CacheMLA,
		UseGraceC2CStaging: cfg.UseGraceC2CStage || receipt.C2CPromotedPages > 0 || receipt.EvictedToC2CPages > 0,
		NumRanks:           transferRanks,
	}
	xferSrc := xferKernel.Source()

	totalTokens := max(totalPromptTokens, 1)
	computeFrac := float64(receipt.PrefillTokensToCompute) / float64(totalTokens)
	c2cPromoteMs := receipt.C2CPromoteLatencyUs / 1e3
	// Layer-pipelined KV/MLA transfer overlaps layer L page streaming with layer L+1 prefill compute.
	pipelinedTransferMs := pdPlan.KVTransferMs * computeFrac / float64(max(modelSpec.NumLayers, 1))
	estTTFTMs := pdPlan.PrefillComputeMs*computeFrac + pipelinedTransferMs + c2cPromoteMs
	layerPerPageUs := (pdPlan.KVTransferMs * 1e3) / float64(max(len(pages)*max(modelSpec.NumLayers, 1), 1))

	return &PDRequestTickResult{
		ReqID:                  reqID,
		Receipt:                receipt,
		CPSchedule:             cpSched,
		CPPlan:                 cpPlan,
		CPImbalanceRatio:       cpImbalance,
		ChunkedSteps:           chunkSteps,
		TransferKernel:         xferKernel,
		TransferKernelSource:   xferSrc,
		LayerTransferPerPageUs: layerPerPageUs,
		EstimatedPrefillTTFTMs: estTTFTMs,
		PDPlan:                 pdPlan,
		RubinPlan:              rubinCopy,
	}, nil
}

func buildCompatibleCPSchedule(
	mode nvl72.ContextParallelMode,
	ulysses, ring int,
	target string,
	modelSpec nvl72.PDModelSpec,
	seqTokens int,
) nvl72.ContextParallelSchedule {
	switch target {
	case "sm_100a", "sm_103a", "sm_107a":
	default:
		target = "sm_100a"
	}
	qHeads := max(modelSpec.NumQHeads, 4)
	kvHeads := max(modelSpec.NumKVHeads, 1)
	if qHeads%kvHeads != 0 {
		qHeads = kvHeads * max(qHeads/kvHeads, 1)
	}
	headDim := max(modelSpec.HeadDim, 64)

	switch mode {
	case nvl72.CPStripedRing, nvl72.CPContiguousRing:
		ulysses = 1
		if ring < 1 {
			ring = 4
		}
	case nvl72.CPUlyssesAll2All:
		ring = 1
		if ulysses < 1 {
			ulysses = min(4, qHeads)
		}
		for ulysses > 1 && qHeads%ulysses != 0 {
			ulysses--
		}
	case nvl72.CPHybrid2D:
		if ulysses < 1 {
			ulysses = min(4, qHeads)
		}
		for ulysses > 1 && qHeads%ulysses != 0 {
			ulysses--
		}
		if ring < 1 {
			ring = 2
		}
	default:
		mode = nvl72.CPStripedRing
		ulysses = 1
		if ring < 1 {
			ring = 4
		}
	}

	chunks := ring
	if mode == nvl72.CPStripedRing || mode == nvl72.CPHybrid2D {
		chunks = 2 * ring
	}
	tile := 16
	minSeq := chunks * tile
	seqLen := max(seqTokens, minSeq)
	if rem := seqLen % minSeq; rem != 0 {
		seqLen += minSeq - rem
	}

	return nvl72.ContextParallelSchedule{
		Target:     target,
		Mode:       mode,
		SeqLen:     seqLen,
		TileM:      tile,
		TileN:      tile,
		HeadDim:    headDim,
		NumQHeads:  qHeads,
		NumKVHeads: kvHeads,
		UlyssesDeg: ulysses,
		RingDeg:    ring,
	}
}

// StepDecodePool observes a speculative decode step with draftTokens draft
// tokens and acceptanceRate in [0, 1] on the engine's
// [specdecode.MultiBucketMegakernelTable] ($M \in \{1, 2, 4, 8\}$) and returns
// the selected bucket M and cross-layer persistent megakernel name.
func (e *DisaggregatedPDEngine) StepDecodePool(ctx context.Context, draftTokens int, acceptanceRate float64) (bucketM int, kernelName string, err error) {
	if e == nil || e.decodeTable == nil {
		return 0, "", errors.New("disaggregated pd engine: nil decode table")
	}
	entry, _, err := e.decodeTable.ObserveAndSelect(ctx, draftTokens, acceptanceRate)
	if err != nil {
		return 0, "", err
	}
	name := entry.Megakernel.Name
	if name == "" && entry.Plan != nil {
		name = entry.Plan.Name
	}
	return entry.Spec.BucketM, name, nil
}

// RebalanceToSLA evaluates the observed TTFT and TPOT latencies against the
// engine's configured [nvl72.PDSLA] targets and, when either SLA is violated
// (or pool pressure shifts), dynamically adjusts the prefill/decode GPU or rack
// split to restore SLA compliance while maximizing cluster token throughput.
func (e *DisaggregatedPDEngine) RebalanceToSLA(observedTTFTMs, observedTPOTMs float64) (nvl72.PDDisaggPlan, bool, error) {
	if e == nil {
		return nvl72.PDDisaggPlan{}, false, errors.New("disaggregated pd engine: nil engine")
	}
	if observedTTFTMs <= 0 || observedTPOTMs <= 0 || math.IsNaN(observedTTFTMs) || math.IsNaN(observedTPOTMs) {
		return nvl72.PDDisaggPlan{}, false, fmt.Errorf("disaggregated pd engine rebalance: observedTTFTMs (%v) and observedTPOTMs (%v) must be > 0",
			observedTTFTMs, observedTPOTMs)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	cur := e.pdPlan
	sla := e.sla
	ttftExceeded := observedTTFTMs > sla.MaxTTFTMs
	tpotExceeded := observedTPOTMs > sla.MaxTPOTMs
	if !ttftExceeded && !tpotExceeded {
		return cur, false, nil
	}

	// Compute tighter effective SLA targets scaled by the observed violation ratio
	// so PlanPDDisaggregation / PlanRubinDisaggregation shifts resources toward
	// the bottlenecked pool.
	adjSLA := sla
	if ttftExceeded && !tpotExceeded {
		ratio := observedTTFTMs / sla.MaxTTFTMs
		adjSLA.MaxTTFTMs = max(sla.MaxTTFTMs/ratio, 0.1)
	} else if tpotExceeded && !ttftExceeded {
		ratio := observedTPOTMs / sla.MaxTPOTMs
		adjSLA.MaxTPOTMs = max(sla.MaxTPOTMs/ratio, 0.1)
		adjSLA.DecodeBatchPerGPU = max(int(math.Ceil(float64(sla.DecodeBatchPerGPU)*min(ratio, 2.0))), sla.DecodeBatchPerGPU+16)
	} else {
		ttftRatio := observedTTFTMs / sla.MaxTTFTMs
		tpotRatio := observedTPOTMs / sla.MaxTPOTMs
		adjSLA.MaxTTFTMs = max(sla.MaxTTFTMs/ttftRatio, 0.1)
		adjSLA.MaxTPOTMs = max(sla.MaxTPOTMs/tpotRatio, 0.1)
	}

	if e.rubinPlan != nil {
		totalRacks := max(cur.TotalRacks, 2)
		pRacks := cur.PrefillRacks
		dRacks := cur.DecodeRacks
		if ttftExceeded && (!tpotExceeded || observedTTFTMs/sla.MaxTTFTMs >= observedTPOTMs/sla.MaxTPOTMs) {
			if dRacks > 1 {
				pRacks++
				dRacks--
			} else {
				pRacks++
				totalRacks = pRacks + dRacks
			}
		} else {
			if pRacks > 1 {
				dRacks++
				pRacks--
			} else {
				dRacks++
				totalRacks = pRacks + dRacks
			}
		}
		decodeSpec := nvl72.DefaultVeraRubinNVL144Spec()
		if e.rubinPlan.DecodeTier == nvl72.TierRubinUltraNVL576 {
			decodeSpec = nvl72.DefaultRubinUltraNVL576Spec()
		}
		rp, err := nvl72.PlanRubinDisaggregation(nvl72.RubinDisaggConfig{
			PrefillSpec:               nvl72.DefaultVeraRubinNVL144Spec(),
			DecodeSpec:                decodeSpec,
			PrefillRacks:              pRacks,
			DecodeRacks:               dRacks,
			Model:                     e.modelSpec,
			SLA:                       sla,
			PageTokens:                e.pageSize,
			UseLUTDecompressedPrefill: true,
			UseVeraC2CStaging:         e.cfg.UseGraceC2CStage,
		})
		if err != nil {
			return cur, false, err
		}
		e.rubinPlan = &rp
		e.pdPlan = rubinToPDDisaggPlan(rp)
		e.cfg.TotalRacks = totalRacks
		changed := e.pdPlan.PrefillGPUs != cur.PrefillGPUs || e.pdPlan.DecodeGPUs != cur.DecodeGPUs
		return e.pdPlan, changed, nil
	}

	rebalanced, err := nvl72.PlanPDDisaggregation(e.cfg.Generation, e.cfg.TotalRacks, adjSLA, e.modelSpec)
	if err != nil {
		return cur, false, err
	}

	// If the planner returned the same split despite a clear single-side bottleneck,
	// shift one compute unit (or one compute tray = 4 GPUs when only 1 rack remains on
	// the donor side) toward the violated pool so the rebalance makes forward progress.
	if rebalanced.PrefillGPUs == cur.PrefillGPUs && rebalanced.DecodeGPUs == cur.DecodeGPUs {
		spec, sErr := nvl72.LookupPodSpec(e.cfg.Generation)
		if sErr == nil {
			trayGPUs := nvl72.NVL72GPUsPerTray
			rackGPUs := spec.GPUsPerRack
			if ttftExceeded && (!tpotExceeded || observedTTFTMs/sla.MaxTTFTMs >= observedTPOTMs/sla.MaxTPOTMs) {
				if cur.DecodeGPUs > rackGPUs && cur.DecodeRacks > 1 {
					rebalanced.PrefillGPUs = cur.PrefillGPUs + rackGPUs
					rebalanced.DecodeGPUs = cur.DecodeGPUs - rackGPUs
					rebalanced.PrefillRacks = cur.PrefillRacks + 1
					rebalanced.DecodeRacks = cur.DecodeRacks - 1
				} else if cur.DecodeGPUs > trayGPUs {
					rebalanced.PrefillGPUs = cur.PrefillGPUs + trayGPUs
					rebalanced.DecodeGPUs = cur.DecodeGPUs - trayGPUs
				}
			} else if tpotExceeded {
				if cur.PrefillGPUs > rackGPUs && cur.PrefillRacks > 1 {
					rebalanced.DecodeGPUs = cur.DecodeGPUs + rackGPUs
					rebalanced.PrefillGPUs = cur.PrefillGPUs - rackGPUs
					rebalanced.DecodeRacks = cur.DecodeRacks + 1
					rebalanced.PrefillRacks = cur.PrefillRacks - 1
				} else if cur.PrefillGPUs > trayGPUs {
					rebalanced.DecodeGPUs = cur.DecodeGPUs + trayGPUs
					rebalanced.PrefillGPUs = cur.PrefillGPUs - trayGPUs
				}
			}
		}
	}

	changed := rebalanced.PrefillGPUs != cur.PrefillGPUs || rebalanced.DecodeGPUs != cur.DecodeGPUs
	e.pdPlan = rebalanced
	return rebalanced, changed, nil
}

// DecodeBuckets returns the sorted list of pre-warmed bucket M sizes in the
// engine's decode megakernel table.
func (e *DisaggregatedPDEngine) DecodeBuckets() []int {
	if e == nil || e.decodeTable == nil {
		return nil
	}
	entries := e.decodeTable.Entries()
	out := make([]int, 0, len(entries))
	for _, ent := range entries {
		out = append(out, ent.Spec.BucketM)
	}
	slices.Sort(out)
	return out
}
