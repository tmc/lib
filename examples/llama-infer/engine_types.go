package main

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// SequenceStatus tracks the lifecycle phase of an ongoing generation sequence.
type SequenceStatus int

const (
	// SeqPendingPrefill indicates the sequence is waiting for prompt ingestion.
	SeqPendingPrefill SequenceStatus = iota
	// SeqDecoding indicates the sequence is actively generating autoregressive tokens.
	SeqDecoding
	// SeqFinished indicates the sequence has completed generation.
	SeqFinished
	// SeqCanceled indicates the sequence was canceled by the client.
	SeqCanceled
	// SeqFailed indicates the sequence encountered an unrecoverable error during execution.
	SeqFailed
)

func (s SequenceStatus) String() string {
	switch s {
	case SeqPendingPrefill:
		return "PendingPrefill"
	case SeqDecoding:
		return "Decoding"
	case SeqFinished:
		return "Finished"
	case SeqCanceled:
		return "Canceled"
	case SeqFailed:
		return "Failed"
	default:
		return fmt.Sprintf("Status(%d)", int(s))
	}
}

// Sequence represents an isolated generation stream containing its own KV cache state,
// sequence position, and token history.
type Sequence struct {
	ID              string
	Prompt          string
	PromptTokens    []int32
	GeneratedTokens []int32
	CurrentToken    int32
	Pos             int
	MaxTokens       int
	Status          SequenceStatus
	Error           error
	OnToken         func(tokenID int32, tokenText string) bool
	DoneCh          chan struct{}
	CancelCh        chan struct{}
	mu              sync.Mutex

	// Isolated KV cache state
	KVCache *SequenceKVCache

	// Metrics
	PrefillDuration time.Duration
	DecodeDuration  time.Duration
	CreatedTime     time.Time
}

// NewSequence initializes an isolated sequence stream.
func NewSequence(id string, prompt string, promptTokens []int32, maxTokens int, onToken func(tokenID int32, tokenText string) bool) *Sequence {
	if maxTokens <= 0 {
		maxTokens = 64
	}
	return &Sequence{
		ID:           id,
		Prompt:       prompt,
		PromptTokens: promptTokens,
		MaxTokens:    maxTokens,
		Status:       SeqPendingPrefill,
		OnToken:      onToken,
		DoneCh:       make(chan struct{}),
		CancelCh:     make(chan struct{}),
		CreatedTime:  time.Now(),
	}
}

// Cancel requests early termination of the sequence.
func (s *Sequence) Cancel() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.Status == SeqFinished || s.Status == SeqCanceled || s.Status == SeqFailed {
		return
	}
	s.Status = SeqCanceled
	select {
	case <-s.CancelCh:
	default:
		close(s.CancelCh)
	}
	select {
	case <-s.DoneCh:
	default:
		close(s.DoneCh)
	}
}

func (s *Sequence) appendTokenLocked(tokenID int32, tokenText string) bool {
	if s.Status != SeqDecoding {
		return false
	}

	s.GeneratedTokens = append(s.GeneratedTokens, tokenID)
	s.CurrentToken = tokenID

	if len(s.GeneratedTokens) >= s.MaxTokens {
		s.Status = SeqFinished
		select {
		case <-s.DoneCh:
		default:
			close(s.DoneCh)
		}
		return false
	}
	return true
}

// AppendToken records a newly generated token and updates sequence status.
func (s *Sequence) AppendToken(tokenID int32, tokenText string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.appendTokenLocked(tokenID, tokenText)
}

// Fail marks the sequence as failed with the given error.
func (s *Sequence) Fail(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.Error = err
	s.Status = SeqFailed
	select {
	case <-s.DoneCh:
	default:
		close(s.DoneCh)
	}
}

// Scheduler coordinates continuous batching and sequence execution steps.
type Scheduler struct {
	engine   *ModelEngine
	cfg      Config
	mu       sync.Mutex
	active   []*Sequence
	maxBatch int
}

// NewScheduler creates a scheduler for the given model engine and configuration.
func NewScheduler(eng *ModelEngine, cfg Config, maxBatch int) *Scheduler {
	if maxBatch <= 0 {
		maxBatch = 32
	}
	return &Scheduler{
		engine:   eng,
		cfg:      cfg,
		maxBatch: maxBatch,
		active:   make([]*Sequence, 0, maxBatch),
	}
}

// Submit enqueues a new sequence into the scheduler.
func (s *Scheduler) Submit(seq *Sequence) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.active) >= s.maxBatch {
		return fmt.Errorf("scheduler: batch capacity (%d) reached", s.maxBatch)
	}
	s.active = append(s.active, seq)
	return nil
}

// ActiveCount returns the number of active sequences in the scheduler.
func (s *Scheduler) ActiveCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.active)
}

type sequenceStepCallback struct {
	seq       *Sequence
	tokenID   int32
	tokenText string
}

// Step advances each active sequence by a single bounded step (prefill or decode token).
// Callbacks are invoked outside the scheduler lock to avoid contention.
func (s *Scheduler) Step(ctx context.Context) (int, error) {
	s.mu.Lock()
	if len(s.active) == 0 {
		s.mu.Unlock()
		return 0, nil
	}

	// Snapshot active sequences
	activeSeqs := make([]*Sequence, len(s.active))
	copy(activeSeqs, s.active)
	s.mu.Unlock()

	var callbacks []sequenceStepCallback
	var decodeSeqs []*Sequence
	var batchTokens []int32
	var batchPositions []int
	var batchKVs []*SequenceKVCache

	for _, seq := range activeSeqs {
		select {
		case <-ctx.Done():
			seq.Cancel()
			continue
		case <-seq.CancelCh:
			continue
		default:
		}

		seq.mu.Lock()
		status := seq.Status
		seq.mu.Unlock()

		if status == SeqPendingPrefill {
			// Ingest prompt
			if s.engine != nil {
				seq.mu.Lock()
				if seq.KVCache == nil {
					kv, err := s.engine.NewSequenceKV(s.cfg)
					if err != nil {
						seq.mu.Unlock()
						seq.Fail(err)
						continue
					}
					seq.KVCache = kv
				}
				pTokens := seq.PromptTokens
				skv := seq.KVCache
				seq.mu.Unlock()

				start := time.Now()
				nextTok, err := GPUPrefillPassKV(s.engine, s.cfg, skv, pTokens)
				if err != nil {
					seq.Fail(err)
					continue
				}
				seq.mu.Lock()
				seq.PrefillDuration = time.Since(start)
				seq.CurrentToken = nextTok
				seq.Pos = len(pTokens)
				seq.Status = SeqDecoding
				seq.GeneratedTokens = append(seq.GeneratedTokens, nextTok)
				seq.mu.Unlock()

				callbacks = append(callbacks, sequenceStepCallback{
					seq:       seq,
					tokenID:   nextTok,
					tokenText: fmt.Sprintf("%d", nextTok),
				})
			} else {
				seq.mu.Lock()
				seq.Status = SeqFinished
				select {
				case <-seq.DoneCh:
				default:
					close(seq.DoneCh)
				}
				seq.mu.Unlock()
			}
		} else if status == SeqDecoding {
			if s.engine != nil {
				seq.mu.Lock()
				curTok := seq.CurrentToken
				pos := seq.Pos
				skv := seq.KVCache
				seq.mu.Unlock()

				decodeSeqs = append(decodeSeqs, seq)
				batchTokens = append(batchTokens, curTok)
				batchPositions = append(batchPositions, pos)
				batchKVs = append(batchKVs, skv)
			} else {
				seq.mu.Lock()
				nextTok := seq.CurrentToken + 1
				seq.Pos++
				cont := seq.appendTokenLocked(nextTok, fmt.Sprintf("%d", nextTok))
				if s.cfg.IsEOS(nextTok) || !cont {
					seq.Status = SeqFinished
					select {
					case <-seq.DoneCh:
					default:
						close(seq.DoneCh)
					}
				}
				seq.mu.Unlock()

				callbacks = append(callbacks, sequenceStepCallback{
					seq:       seq,
					tokenID:   nextTok,
					tokenText: fmt.Sprintf("%d", nextTok),
				})
			}
		}
	}

	// Concurrent multi-sequence batched decode across all active decoding sequences
	if len(decodeSeqs) > 0 {
		start := time.Now()
		outTokens, err := GPUBatchDecodeStep(s.engine, s.cfg, batchKVs, batchTokens, batchPositions)
		dur := time.Since(start)
		if err != nil {
			for _, seq := range decodeSeqs {
				seq.Fail(err)
			}
		} else {
			perSeqDur := dur / time.Duration(len(decodeSeqs))
			for i, seq := range decodeSeqs {
				nextTok := outTokens[i]
				seq.mu.Lock()
				seq.DecodeDuration += perSeqDur
				seq.Pos++
				cont := seq.appendTokenLocked(nextTok, fmt.Sprintf("%d", nextTok))
				if s.cfg.IsEOS(nextTok) || !cont {
					seq.Status = SeqFinished
					select {
					case <-seq.DoneCh:
					default:
						close(seq.DoneCh)
					}
				}
				seq.mu.Unlock()

				callbacks = append(callbacks, sequenceStepCallback{
					seq:       seq,
					tokenID:   nextTok,
					tokenText: fmt.Sprintf("%d", nextTok),
				})
			}
		}
	}

	// Dispatch callbacks outside scheduler lock
	for _, cb := range callbacks {
		if cb.seq.OnToken != nil {
			cont := cb.seq.OnToken(cb.tokenID, cb.tokenText)
			if !cont {
				cb.seq.Cancel()
			}
		}
	}

	// Prune finished/canceled sequences
	s.mu.Lock()
	defer s.mu.Unlock()

	remaining := s.active[:0]
	for _, seq := range s.active {
		seq.mu.Lock()
		st := seq.Status
		if st != SeqPendingPrefill && st != SeqDecoding {
			if seq.KVCache != nil {
				seq.KVCache.Free()
				seq.KVCache = nil
			}
		} else {
			remaining = append(remaining, seq)
		}
		seq.mu.Unlock()
	}
	s.active = remaining

	return len(callbacks), nil
}

// ExecutionPlan is an immutable compiled blueprint specifying tensor layouts,
// memory buffer allocations, dependency sequences, and selected backend kernel operations.
type ExecutionPlan struct {
	DType      string             `json:"dtype"`
	FuseEGraph bool               `json:"fuse_egraph"`
	BatchShape [3]int             `json:"batch_shape"`
	HeadDim    int                `json:"head_dim"`
	NumHeads   int                `json:"num_heads"`
	NumKVHeads int                `json:"num_kv_heads"`
	NumLayers  int                `json:"num_layers"`
	VocabSize  int                `json:"vocab_size"`
	Ops        []OpDispatch       `json:"ops"`
	Buffers    []BufferAllocation `json:"buffers"`
}

// OpDispatch represents a single planned kernel execution step.
type OpDispatch struct {
	Name        string   `json:"name"`
	Category    string   `json:"category"`
	LayerIndex  int      `json:"layer_index,omitempty"`
	Fused       bool     `json:"fused"`
	Inputs      []string `json:"inputs"`
	Outputs     []string `json:"outputs"`
	Threadgroup [3]int   `json:"threadgroup,omitempty"`
	Grid        [3]int   `json:"grid,omitempty"`
}

// BufferAllocation details a residency requirement for intermediate activations or KV states.
type BufferAllocation struct {
	Name      string `json:"name"`
	SizeBytes int64  `json:"size_bytes"`
	DType     string `json:"dtype"`
	Resident  bool   `json:"resident"`
}

// BuildExecutionPlan deterministically constructs the compiled execution plan for a model configuration.
func BuildExecutionPlan(cfg Config) *ExecutionPlan {
	dtype := "f32"
	if cfg.Precision == "f16" {
		dtype = "f16"
	}
	plan := &ExecutionPlan{
		DType:      dtype,
		FuseEGraph: cfg.FuseEGraph,
		BatchShape: [3]int{1, cfg.MaxSeqLen, cfg.Dim},
		HeadDim:    cfg.HeadDim,
		NumHeads:   cfg.NumHeads,
		NumKVHeads: cfg.NumKVHeads,
		NumLayers:  cfg.NumLayers,
		VocabSize:  cfg.VocabSize,
	}

	elemBytes := int64(4) // float32 activations

	// Plan resident activation buffers
	plan.Buffers = []BufferAllocation{
		{Name: "x", SizeBytes: int64(cfg.Dim) * elemBytes, DType: plan.DType, Resident: true},
		{Name: "norm_x", SizeBytes: int64(cfg.Dim) * elemBytes, DType: plan.DType, Resident: true},
		{Name: "q", SizeBytes: int64(cfg.NumHeads*cfg.HeadDim) * elemBytes, DType: plan.DType, Resident: true},
		{Name: "k", SizeBytes: int64(cfg.NumKVHeads*cfg.HeadDim) * elemBytes, DType: plan.DType, Resident: true},
		{Name: "v", SizeBytes: int64(cfg.NumKVHeads*cfg.HeadDim) * elemBytes, DType: plan.DType, Resident: true},
		{Name: "attn_out", SizeBytes: int64(cfg.Dim) * elemBytes, DType: plan.DType, Resident: true},
		{Name: "ffn_gate", SizeBytes: int64(cfg.HiddenDim) * elemBytes, DType: plan.DType, Resident: true},
		{Name: "ffn_up", SizeBytes: int64(cfg.HiddenDim) * elemBytes, DType: plan.DType, Resident: true},
		{Name: "ffn_act", SizeBytes: int64(cfg.HiddenDim) * elemBytes, DType: plan.DType, Resident: true},
		{Name: "logits", SizeBytes: int64(cfg.VocabSize) * elemBytes, DType: plan.DType, Resident: true},
		{Name: "token_out", SizeBytes: 4, DType: "int32", Resident: true},
	}

	// Plan ops
	plan.Ops = append(plan.Ops, OpDispatch{
		Name:     "EmbeddingLookup",
		Category: "embedding",
		Inputs:   []string{"embed_table", "token_id"},
		Outputs:  []string{"x"},
	})

	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.FuseEGraph {
			plan.Ops = append(plan.Ops, OpDispatch{
				Name:       "FusedRMSNormQKV",
				Category:   "norm_gemv_fusion",
				LayerIndex: l,
				Fused:      true,
				Inputs:     []string{"x", "in_norm_w", "wq", "wk", "wv"},
				Outputs:    []string{"q", "k", "v"},
			})
		} else {
			plan.Ops = append(plan.Ops,
				OpDispatch{Name: "ComputeRMSNorm", Category: "norm", LayerIndex: l, Inputs: []string{"x", "in_norm_w"}, Outputs: []string{"norm_x"}},
				OpDispatch{Name: "FastGEMV_Q", Category: "gemv", LayerIndex: l, Inputs: []string{"norm_x", "wq"}, Outputs: []string{"q"}},
				OpDispatch{Name: "FastGEMV_K", Category: "gemv", LayerIndex: l, Inputs: []string{"norm_x", "wk"}, Outputs: []string{"k"}},
				OpDispatch{Name: "FastGEMV_V", Category: "gemv", LayerIndex: l, Inputs: []string{"norm_x", "wv"}, Outputs: []string{"v"}},
			)
		}

		plan.Ops = append(plan.Ops,
			OpDispatch{Name: "ApplyRoPEStoreKV", Category: "rope", LayerIndex: l, Inputs: []string{"q", "k", "v"}, Outputs: []string{"k_cache", "v_cache"}},
			OpDispatch{Name: "CausalFlashAttentionDecode", Category: "attention", LayerIndex: l, Inputs: []string{"q", "k_cache", "v_cache"}, Outputs: []string{"attn_out"}},
			OpDispatch{Name: "FastGEMVResidual_O", Category: "gemv", LayerIndex: l, Inputs: []string{"attn_out", "wo"}, Outputs: []string{"x"}},
			OpDispatch{Name: "ComputeRMSNorm_Post", Category: "norm", LayerIndex: l, Inputs: []string{"x", "post_norm_w"}, Outputs: []string{"norm_x"}},
		)

		if cfg.FuseEGraph {
			plan.Ops = append(plan.Ops,
				OpDispatch{Name: "FastGEMV_Gate", Category: "gemv", LayerIndex: l, Inputs: []string{"norm_x", "wgate"}, Outputs: []string{"ffn_gate"}},
				OpDispatch{Name: "FastGEMV_Up", Category: "gemv", LayerIndex: l, Inputs: []string{"norm_x", "wup"}, Outputs: []string{"ffn_up"}},
				OpDispatch{Name: "FusedSwiGLUDown", Category: "swiglu_gemv_fusion", LayerIndex: l, Fused: true, Inputs: []string{"ffn_gate", "ffn_up", "wdown"}, Outputs: []string{"x"}},
			)
		} else {
			plan.Ops = append(plan.Ops,
				OpDispatch{Name: "FastGEMV_Gate", Category: "gemv", LayerIndex: l, Inputs: []string{"norm_x", "wgate"}, Outputs: []string{"ffn_gate"}},
				OpDispatch{Name: "FastGEMV_Up", Category: "gemv", LayerIndex: l, Inputs: []string{"norm_x", "wup"}, Outputs: []string{"ffn_up"}},
				OpDispatch{Name: "FastSiLUMul", Category: "ffn", LayerIndex: l, Inputs: []string{"ffn_gate", "ffn_up"}, Outputs: []string{"ffn_act"}},
				OpDispatch{Name: "FastGEMVResidual_Down", Category: "gemv", LayerIndex: l, Inputs: []string{"ffn_act", "wdown"}, Outputs: []string{"x"}},
			)
		}
	}

	plan.Ops = append(plan.Ops,
		OpDispatch{Name: "ComputeRMSNorm_Final", Category: "norm", Inputs: []string{"x", "final_norm_w"}, Outputs: []string{"norm_x"}},
		OpDispatch{Name: "FastGEMV_LMHead", Category: "gemv", Inputs: []string{"norm_x", "lm_head_w"}, Outputs: []string{"logits"}},
		OpDispatch{Name: "ArgmaxSampling", Category: "sampling", Inputs: []string{"logits"}, Outputs: []string{"token_out"}},
	)

	return plan
}

// ValidatePlan verifies that an ExecutionPlan conforms to valid operations,
// layer indices, dtype specifications, and dimension boundaries.
func ValidatePlan(plan *ExecutionPlan, cfg Config) error {
	if plan == nil {
		return errors.New("plan is nil")
	}
	if len(plan.Ops) == 0 {
		return errors.New("plan contains no operations")
	}
	if plan.DType != "f32" && plan.DType != "fp32" && plan.DType != "f16" && plan.DType != "fp16" {
		return fmt.Errorf("unsupported plan dtype: %q", plan.DType)
	}

	// 1. Strict dimension requirements
	if plan.HeadDim <= 0 || (cfg.HeadDim > 0 && plan.HeadDim != cfg.HeadDim) {
		return fmt.Errorf("plan HeadDim invalid: plan=%d, cfg=%d", plan.HeadDim, cfg.HeadDim)
	}
	if plan.NumHeads <= 0 || (cfg.NumHeads > 0 && plan.NumHeads != cfg.NumHeads) {
		return fmt.Errorf("plan NumHeads invalid: plan=%d, cfg=%d", plan.NumHeads, cfg.NumHeads)
	}
	if plan.NumKVHeads <= 0 || (cfg.NumKVHeads > 0 && plan.NumKVHeads != cfg.NumKVHeads) {
		return fmt.Errorf("plan NumKVHeads invalid: plan=%d, cfg=%d", plan.NumKVHeads, cfg.NumKVHeads)
	}
	if plan.NumLayers <= 0 || (cfg.NumLayers > 0 && plan.NumLayers != cfg.NumLayers) {
		return fmt.Errorf("plan NumLayers invalid: plan=%d, cfg=%d", plan.NumLayers, cfg.NumLayers)
	}
	if plan.VocabSize <= 0 || (cfg.VocabSize > 0 && plan.VocabSize != cfg.VocabSize) {
		return fmt.Errorf("plan VocabSize invalid: plan=%d, cfg=%d", plan.VocabSize, cfg.VocabSize)
	}

	validOps := map[string]bool{
		"EmbeddingLookup":            true,
		"ComputeRMSNorm":             true,
		"FastGEMV_Q":                 true,
		"FastGEMV_K":                 true,
		"FastGEMV_V":                 true,
		"FusedRMSNormQKV":            true,
		"ApplyRoPEStoreKV":           true,
		"CausalFlashAttentionDecode": true,
		"FastGEMVResidual_O":         true,
		"ComputeRMSNorm_Post":        true,
		"FastGEMV_Gate":              true,
		"FastGEMV_Up":                true,
		"FusedSwiGLUDown":            true,
		"FastSiLUMul":                true,
		"FastGEMVResidual_Down":      true,
		"ComputeRMSNorm_Final":       true,
		"FastGEMV_LMHead":            true,
		"ArgmaxSampling":             true,
	}

	// 2. Op vocabulary and layer index bound validation
	for i, op := range plan.Ops {
		if !validOps[op.Name] {
			return fmt.Errorf("plan op %d: unknown operation %q", i, op.Name)
		}
		if op.Name != "EmbeddingLookup" && op.Name != "ComputeRMSNorm_Final" && op.Name != "FastGEMV_LMHead" && op.Name != "ArgmaxSampling" {
			if op.LayerIndex < 0 || op.LayerIndex >= plan.NumLayers {
				return fmt.Errorf("plan op %d (%s): layer index %d out of bounds [0, %d)", i, op.Name, op.LayerIndex, plan.NumLayers)
			}
		}
	}

	// 3. Exact topology grammar and per-layer sequence validation
	if len(plan.Ops) < 4 {
		return errors.New("plan topology error: insufficient operations")
	}
	if plan.Ops[0].Name != "EmbeddingLookup" {
		return fmt.Errorf("plan topology error: first op must be EmbeddingLookup, got %q", plan.Ops[0].Name)
	}

	opNameAt := func(i int) string {
		if i >= 0 && i < len(plan.Ops) {
			return plan.Ops[i].Name
		}
		return "<EOF>"
	}

	idx := 1
	for l := 0; l < plan.NumLayers; l++ {
		if idx >= len(plan.Ops) {
			return fmt.Errorf("plan topology error: missing operations for layer %d", l)
		}

		// Parse Attention Norm + QKV
		if plan.Ops[idx].Name == "FusedRMSNormQKV" {
			if plan.Ops[idx].LayerIndex != l {
				return fmt.Errorf("plan topology error: layer %d expected FusedRMSNormQKV with layer %d, got %d", l, l, plan.Ops[idx].LayerIndex)
			}
			idx++
		} else if idx+3 < len(plan.Ops) &&
			plan.Ops[idx].Name == "ComputeRMSNorm" &&
			plan.Ops[idx+1].Name == "FastGEMV_Q" &&
			plan.Ops[idx+2].Name == "FastGEMV_K" &&
			plan.Ops[idx+3].Name == "FastGEMV_V" {
			for k := 0; k < 4; k++ {
				if plan.Ops[idx+k].LayerIndex != l {
					return fmt.Errorf("plan topology error: layer %d expected %s with layer %d, got %d", l, plan.Ops[idx+k].Name, l, plan.Ops[idx+k].LayerIndex)
				}
			}
			idx += 4
		} else {
			return fmt.Errorf("plan topology error: layer %d invalid attention QKV block at op %d (%s)", l, idx, opNameAt(idx))
		}

		// Parse Attention Core (ApplyRoPEStoreKV -> CausalFlashAttentionDecode -> FastGEMVResidual_O -> ComputeRMSNorm_Post)
		if idx+3 >= len(plan.Ops) ||
			plan.Ops[idx].Name != "ApplyRoPEStoreKV" ||
			plan.Ops[idx+1].Name != "CausalFlashAttentionDecode" ||
			plan.Ops[idx+2].Name != "FastGEMVResidual_O" ||
			plan.Ops[idx+3].Name != "ComputeRMSNorm_Post" {
			return fmt.Errorf("plan topology error: layer %d invalid attention core block at op %d (%s)", l, idx, opNameAt(idx))
		}
		for k := 0; k < 4; k++ {
			if plan.Ops[idx+k].LayerIndex != l {
				return fmt.Errorf("plan topology error: layer %d expected %s with layer %d, got %d", l, plan.Ops[idx+k].Name, l, plan.Ops[idx+k].LayerIndex)
			}
		}
		idx += 4

		// Parse FFN block
		if idx >= len(plan.Ops) {
			return fmt.Errorf("plan topology error: layer %d missing FFN block at op %d", l, idx)
		}
		if idx+2 < len(plan.Ops) &&
			plan.Ops[idx].Name == "FastGEMV_Gate" &&
			plan.Ops[idx+1].Name == "FastGEMV_Up" &&
			plan.Ops[idx+2].Name == "FusedSwiGLUDown" {
			for k := 0; k < 3; k++ {
				if plan.Ops[idx+k].LayerIndex != l {
					return fmt.Errorf("plan topology error: layer %d expected %s with layer %d, got %d", l, plan.Ops[idx+k].Name, l, plan.Ops[idx+k].LayerIndex)
				}
			}
			idx += 3
		} else if idx+3 < len(plan.Ops) &&
			plan.Ops[idx].Name == "FastGEMV_Gate" &&
			plan.Ops[idx+1].Name == "FastGEMV_Up" &&
			plan.Ops[idx+2].Name == "FastSiLUMul" &&
			plan.Ops[idx+3].Name == "FastGEMVResidual_Down" {
			for k := 0; k < 4; k++ {
				if plan.Ops[idx+k].LayerIndex != l {
					return fmt.Errorf("plan topology error: layer %d expected %s with layer %d, got %d", l, plan.Ops[idx+k].Name, l, plan.Ops[idx+k].LayerIndex)
				}
			}
			idx += 4
		} else {
			return fmt.Errorf("plan topology error: layer %d invalid FFN block at op %d (%s)", l, idx, opNameAt(idx))
		}
	}

	// Trailing epilogue: ComputeRMSNorm_Final -> FastGEMV_LMHead -> ArgmaxSampling
	if idx+3 != len(plan.Ops) ||
		plan.Ops[idx].Name != "ComputeRMSNorm_Final" ||
		plan.Ops[idx+1].Name != "FastGEMV_LMHead" ||
		plan.Ops[idx+2].Name != "ArgmaxSampling" {
		return fmt.Errorf("plan topology error: expected epilogue ComputeRMSNorm_Final -> FastGEMV_LMHead -> ArgmaxSampling, got remaining %d ops", len(plan.Ops)-idx)
	}

	return nil
}
