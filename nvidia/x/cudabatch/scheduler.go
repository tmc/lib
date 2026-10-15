package cudabatch

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

// SchedulerConfig configures batch scheduler limits, chunking, and fairness.
type SchedulerConfig struct {
	MaxBatchSize      int // Maximum active concurrent rows (e.g. 64)
	MaxBudgetTokens   int // Combined batch token budget per step (e.g. 2048)
	ChunkSize         int // Maximum prefill chunk tokens per request per step (e.g. 512)
	MaxWaitingTimeout time.Duration
	FairnessPolicy    FairnessPolicy
}

// FairnessPolicy controls request arbitration among waiting and prefilling sequences.
type FairnessPolicy int

const (
	// PolicyFCFS serves requests in order of arrival time.
	PolicyFCFS FairnessPolicy = iota
	// PolicyFairShare balances tokens served across active requests, preventing starvation of long prefills.
	PolicyFairShare
	// PolicyPriorityDecode prioritizes decode latency while guaranteeing non-zero prefill progress.
	PolicyPriorityDecode
)

// ScheduledIteration contains the selected rows and their action in the current step.
type ScheduledIteration struct {
	StepID        int64
	PrefillChunks []cudakv.PrefillChunk
	DecodeSeqIDs  []string
	TotalTokens   int
	AllocatedRows []BatchRowBinding
}

// BatchScheduler coordinates request admission, continuous batching iteration scheduling,
// prefill chunking, and deterministic state transitions.
type BatchScheduler struct {
	mu           sync.Mutex
	cfg          SchedulerConfig
	kvTracker    *MemoryBudgetTracker
	requests     map[string]*Request
	waitingQueue []*Request
	seqIDCounter int64
	stepCounter  int64

	// Completion and retained resources
	pendingReleases map[string][]func() // Request ID -> retained release callbacks
}

// NewBatchScheduler constructs a continuous batch scheduler with config and optional KV tracker.
func NewBatchScheduler(cfg SchedulerConfig, kvTracker *MemoryBudgetTracker) *BatchScheduler {
	if cfg.MaxBatchSize <= 0 {
		cfg.MaxBatchSize = 32
	}
	if cfg.MaxBudgetTokens <= 0 {
		cfg.MaxBudgetTokens = 2048
	}
	if cfg.ChunkSize <= 0 {
		cfg.ChunkSize = 512
	}
	return &BatchScheduler{
		cfg:             cfg,
		kvTracker:       kvTracker,
		requests:        make(map[string]*Request),
		waitingQueue:    make([]*Request, 0),
		pendingReleases: make(map[string][]func()),
	}
}

// Submit admits a new request into the scheduler.
func (s *BatchScheduler) Submit(req *Request) error {
	if req == nil || req.ID == "" {
		return ErrInvalidRequest
	}
	if len(req.PromptTokens) == 0 {
		return fmt.Errorf("%w: empty prompt tokens", ErrInvalidRequest)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.requests[req.ID]; exists {
		return fmt.Errorf("%w: %s", ErrRequestAlreadyExists, req.ID)
	}

	req.State = StateWaiting
	if req.CreatedAt.IsZero() {
		req.CreatedAt = time.Now()
	}
	s.requests[req.ID] = req
	s.waitingQueue = append(s.waitingQueue, req)
	return nil
}

// Cancel requests cancellation of an active or queued request.
// If the request is currently in an active GPU/host step, its KV blocks and output
// buffers are retained until CompleteStep is called.
func (s *BatchScheduler) Cancel(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	req, ok := s.requests[id]
	if !ok {
		return ErrRequestNotFound
	}
	if req.IsFinished() {
		return nil
	}

	req.cancelRequested = true

	// If not actively executing an iteration, immediately mark canceled and free resources
	if !req.activeExecution {
		req.State = StateCanceled
		req.completedAt = time.Now()
		s.cleanupRequestResourcesLocked(req)
	}
	return nil
}

// Schedule selects requests for the next forward pass iteration according to token budgets,
// memory availability, and fairness policies.
func (s *BatchScheduler) Schedule() (*ScheduledIteration, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check before admission or KV growth: Submit currently retains requests,
	// so a caller may have erased the token needed by an active decode.
	for _, req := range s.requests {
		if req.State == StateDecoding && len(req.GeneratedTokens) == 0 && len(req.PromptTokens) == 0 {
			return nil, fmt.Errorf("%w: decode request %q has no retained token", ErrInvalidRequest, req.ID)
		}
	}

	s.stepCounter++
	iter := &ScheduledIteration{
		StepID:        s.stepCounter,
		PrefillChunks: make([]cudakv.PrefillChunk, 0),
		DecodeSeqIDs:  make([]string, 0),
		AllocatedRows: make([]BatchRowBinding, 0),
	}

	tokenBudget := s.cfg.MaxBudgetTokens
	maxRows := s.cfg.MaxBatchSize

	// 1. Identify active decoding requests
	var activeDecodes []*Request
	for _, req := range s.requests {
		if req.State == StateDecoding && !req.IsFinished() {
			activeDecodes = append(activeDecodes, req)
		}
	}

	// 2. Identify active prefilling requests
	var activePrefills []*Request
	for _, req := range s.requests {
		if req.State == StatePrefilling && !req.IsFinished() {
			activePrefills = append(activePrefills, req)
		}
	}

	// 3. Check waiting queue for admission
	// Under PolicyFairShare, guarantee at least 1 row slot for prefill admission if waiting queue is non-empty
	admissionSlots := maxRows - len(activeDecodes) - len(activePrefills)
	if admissionSlots <= 0 && s.cfg.FairnessPolicy == PolicyFairShare && len(s.waitingQueue) > 0 {
		admissionSlots = 1
	}
	s.admitWaitingLocked(admissionSlots)

	// Re-gather prefilling requests after admission
	activePrefills = activePrefills[:0]
	for _, req := range s.requests {
		if req.State == StatePrefilling && !req.IsFinished() {
			activePrefills = append(activePrefills, req)
		}
	}

	// Apply fairness ordering
	s.applyFairnessOrdering(activePrefills)

	// Step A: Decode requests (1 token per decode request)
	// Under PolicyFairShare, reserve at least ChunkSize tokens for prefill progress if prefill requests exist
	reservedForPrefill := 0
	if s.cfg.FairnessPolicy == PolicyFairShare && len(activePrefills) > 0 {
		reservedForPrefill = s.cfg.ChunkSize
		if reservedForPrefill > tokenBudget/2 {
			reservedForPrefill = tokenBudget / 2
		}
		if reservedForPrefill <= 0 && tokenBudget > 0 {
			reservedForPrefill = 1
		}
	}

	maxDecodeRows := maxRows
	if reservedForPrefill > 0 && len(activePrefills) > 0 && maxDecodeRows > 1 {
		maxDecodeRows--
	}

	for _, req := range activeDecodes {
		if len(iter.AllocatedRows) >= maxDecodeRows || (tokenBudget-reservedForPrefill) <= 0 {
			break
		}
		decodeTokens := req.GeneratedTokens
		if len(decodeTokens) == 0 {
			decodeTokens = req.PromptTokens
		}
		// Decode requires 1 token room in KV cache
		if s.kvTracker != nil {
			// Ensure token can be appended
			seqHash := hashSeqID(req.ID)
			if err := s.kvTracker.AppendToken(seqHash); err != nil {
				// Memory exhausted during decode
				req.err = err
				continue
			}
			req.BlockTable = s.kvTracker.kvMgr.Table(seqHash)
		}

		iter.DecodeSeqIDs = append(iter.DecodeSeqIDs, req.ID)
		iter.AllocatedRows = append(iter.AllocatedRows, BatchRowBinding{
			RowIndex:        len(iter.AllocatedRows),
			SeqID:           req.ID,
			IsPrefill:       false,
			Tokens:          []int32{decodeTokens[len(decodeTokens)-1]},
			StartPos:        req.ContextTokens(),
			SequenceLength:  req.ContextTokens() + 1,
			BlockTable:      append([]int(nil), req.BlockTable...),
			ContinuationVis: req.Visibility,
		})
		tokenBudget--
		iter.TotalTokens++
		req.activeExecution = true
	}

	// Step B: Prefill chunking
	for _, req := range activePrefills {
		if len(iter.AllocatedRows) >= maxRows || tokenBudget <= 0 {
			break
		}

		remaining := len(req.PromptTokens) - req.PrefilledTokens
		chunkSize := s.cfg.ChunkSize
		if remaining < chunkSize {
			chunkSize = remaining
		}
		if chunkSize > tokenBudget {
			chunkSize = tokenBudget
		}
		if chunkSize <= 0 {
			continue
		}

		chunkTokens := req.PromptTokens[req.PrefilledTokens : req.PrefilledTokens+chunkSize]
		isLast := (req.PrefilledTokens + chunkSize) >= len(req.PromptTokens)

		iter.PrefillChunks = append(iter.PrefillChunks, cudakv.PrefillChunk{
			SeqID:    req.ID,
			Tokens:   chunkTokens,
			StartPos: req.PrefilledTokens,
			IsLast:   isLast,
		})

		iter.AllocatedRows = append(iter.AllocatedRows, BatchRowBinding{
			RowIndex:        len(iter.AllocatedRows),
			SeqID:           req.ID,
			IsPrefill:       true,
			Tokens:          chunkTokens,
			StartPos:        req.PrefilledTokens,
			SequenceLength:  req.PrefilledTokens + chunkSize,
			BlockTable:      append([]int(nil), req.BlockTable...),
			ContinuationVis: req.Visibility,
		})

		tokenBudget -= chunkSize
		iter.TotalTokens += chunkSize
		req.activeExecution = true
	}

	return iter, nil
}

// CompleteStep updates request states after iteration execution completes.
// It applies newly generated tokens, transitions completed prefills to decodes,
// processes cancellations, and releases retained KV memory only after invocation is finished.
func (s *BatchScheduler) CompleteStep(iter *ScheduledIteration, generatedTokens map[string]int32, rowErrors map[string]error) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if iter == nil {
		return nil
	}

	// Process each row in the completed iteration
	for _, row := range iter.AllocatedRows {
		req, exists := s.requests[row.SeqID]
		if !exists {
			continue
		}
		req.activeExecution = false

		if rowErr, hasErr := rowErrors[row.SeqID]; hasErr && rowErr != nil {
			req.err = rowErr
			req.State = StateCompleted
			req.completedAt = time.Now()
			s.cleanupRequestResourcesLocked(req)
			continue
		}

		// Handle cancellation that arrived while activeExecution was true
		if req.cancelRequested {
			req.State = StateCanceled
			req.completedAt = time.Now()
			s.cleanupRequestResourcesLocked(req)
			continue
		}

		if row.IsPrefill {
			// Advance prefill progress
			req.PrefilledTokens += len(row.Tokens)
			if req.PrefilledTokens >= len(req.PromptTokens) {
				req.State = StateDecoding
			}
		} else {
			// Decode step
			if tok, ok := generatedTokens[row.SeqID]; ok {
				req.GeneratedTokens = append(req.GeneratedTokens, tok)
				if req.MaxTokens > 0 && len(req.GeneratedTokens) >= req.MaxTokens {
					req.State = StateCompleted
					req.completedAt = time.Now()
					s.cleanupRequestResourcesLocked(req)
				}
			}
		}
	}

	return nil
}

func (s *BatchScheduler) admitWaitingLocked(availableSlots int) {
	if availableSlots <= 0 || len(s.waitingQueue) == 0 {
		return
	}

	var remainingWaiting []*Request
	for _, req := range s.waitingQueue {
		if availableSlots <= 0 {
			remainingWaiting = append(remainingWaiting, req)
			continue
		}
		if req.cancelRequested {
			req.State = StateCanceled
			req.completedAt = time.Now()
			continue
		}

		// Check memory budget
		seqHash := hashSeqID(req.ID)
		if s.kvTracker != nil {
			if !s.kvTracker.CanAdmit(len(req.PromptTokens), nil) {
				// Cannot admit yet due to memory budget; keep in waiting queue
				remainingWaiting = append(remainingWaiting, req)
				continue
			}
			tbl, err := s.kvTracker.AllocSequence(seqHash, nil, len(req.PromptTokens))
			if err != nil {
				// Memory exhausted on allocation
				remainingWaiting = append(remainingWaiting, req)
				continue
			}
			req.BlockTable = tbl
		}

		req.State = StatePrefilling
		availableSlots--
	}
	s.waitingQueue = remainingWaiting
}

func (s *BatchScheduler) applyFairnessOrdering(prefills []*Request) {
	if len(prefills) <= 1 {
		return
	}
	switch s.cfg.FairnessPolicy {
	case PolicyFairShare:
		// Sort by prefilled tokens ascending so starved/new prefill requests get chunks
		sort.SliceStable(prefills, func(i, j int) bool {
			return prefills[i].PrefilledTokens < prefills[j].PrefilledTokens
		})
	case PolicyFCFS:
		sort.SliceStable(prefills, func(i, j int) bool {
			return prefills[i].CreatedAt.Before(prefills[j].CreatedAt)
		})
	}
}

func (s *BatchScheduler) cleanupRequestResourcesLocked(req *Request) {
	seqHash := hashSeqID(req.ID)
	if s.kvTracker != nil {
		s.kvTracker.FreeSequence(seqHash)
	}
	// Execute and drain any pending resource release callbacks
	if callbacks, ok := s.pendingReleases[req.ID]; ok {
		for _, cb := range callbacks {
			if cb != nil {
				cb()
			}
		}
		delete(s.pendingReleases, req.ID)
	}
}

// RegisterResourceRelease registers a cleanup callback (e.g. output buffer release or KV handle)
// to be executed only when the request completely finishes and execution has completed.
func (s *BatchScheduler) RegisterResourceRelease(seqID string, cb func()) {
	s.mu.Lock()
	defer s.mu.Unlock()

	req, ok := s.requests[seqID]
	if ok && req.IsFinished() && !req.activeExecution {
		// Already finished and inactive, call immediately
		if cb != nil {
			cb()
		}
		return
	}
	s.pendingReleases[seqID] = append(s.pendingReleases[seqID], cb)
}

// Request returns the current status snapshot of a request.
func (s *BatchScheduler) Request(id string) (*Request, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	req, ok := s.requests[id]
	if !ok {
		return nil, false
	}
	// Return shallow copy
	cp := *req
	return &cp, true
}

func hashSeqID(id string) int64 {
	var h int64 = 1469598103934665603
	for i := 0; i < len(id); i++ {
		h ^= int64(id[i])
		h *= 1099511628211
	}
	if h < 0 {
		h = -h
	}
	if h == 0 {
		h = 1
	}
	return h
}
