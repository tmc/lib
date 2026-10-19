package cudabatch

import (
	"context"
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
	// MaxWaitingRequests and MaxFinishedRequests bound queued work and terminal
	// status history. Zero selects 1024 for each. Negative limits are invalid.
	// Retiring history removes only finished requests, never an outstanding step.
	MaxWaitingRequests  int
	MaxFinishedRequests int
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
	inFlight     *ScheduledIteration
	finished     []*Request
	configErr    error

	// Completion and retained resources
	pendingReleases map[string][]func() // Request ID -> retained release callbacks
}

// NewBatchScheduler constructs a continuous batch scheduler with config and optional KV tracker.
func NewBatchScheduler(cfg SchedulerConfig, kvTracker *MemoryBudgetTracker) *BatchScheduler {
	var configErr error
	if cfg.MaxWaitingRequests < 0 || cfg.MaxFinishedRequests < 0 || cfg.MaxWaitingTimeout < 0 || cfg.FairnessPolicy < PolicyFCFS || cfg.FairnessPolicy > PolicyPriorityDecode {
		configErr = fmt.Errorf("%w: invalid scheduler limits or fairness policy", ErrInvalidRequest)
	}
	if cfg.MaxWaitingRequests == 0 {
		cfg.MaxWaitingRequests = 1024
	}
	if cfg.MaxFinishedRequests == 0 {
		cfg.MaxFinishedRequests = 1024
	}
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
		configErr:       configErr,
		kvTracker:       kvTracker,
		requests:        make(map[string]*Request),
		waitingQueue:    make([]*Request, 0),
		pendingReleases: make(map[string][]func()),
	}
}

// Submit copies a fresh request and its prompt. MaxTokens must be positive.
// Caller mutations after return cannot change scheduler state. CreatedAt is
// descriptive; arbitration and waiting timeout use scheduler admission order
// and time. A full waiting queue returns ErrCapacityExhausted before admission.
func (s *BatchScheduler) Submit(req *Request) error {
	if req == nil || req.ID == "" {
		return ErrInvalidRequest
	}
	if len(req.PromptTokens) == 0 {
		return fmt.Errorf("%w: empty prompt tokens", ErrInvalidRequest)
	}
	if req.MaxTokens <= 0 || req.MaxTokens > int(^uint(0)>>1)-len(req.PromptTokens) || req.Visibility < VisibilityDefault || req.Visibility > VisibilityThinkingVisible || req.State < StateWaiting || req.State > StateCanceled {
		return fmt.Errorf("%w: invalid token limit, visibility or state", ErrInvalidRequest)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.configErr != nil {
		return s.configErr
	}
	s.expireWaitingLocked(time.Now())

	if _, exists := s.requests[req.ID]; exists {
		return fmt.Errorf("%w: %s", ErrRequestAlreadyExists, req.ID)
	}

	if len(s.waitingQueue) >= s.cfg.MaxWaitingRequests || s.seqIDCounter == int64(^uint64(0)>>1) {
		return ErrCapacityExhausted
	}
	s.seqIDCounter++
	owned := &Request{ID: req.ID, PromptTokens: append([]int32(nil), req.PromptTokens...), MaxTokens: req.MaxTokens, CreatedAt: req.CreatedAt, Visibility: req.Visibility, SessionID: req.SessionID, OnFinish: req.OnFinish, State: StateWaiting, admissionOrdinal: s.seqIDCounter, admittedAt: time.Now()}
	if owned.CreatedAt.IsZero() {
		owned.CreatedAt = owned.admittedAt
	}
	s.requests[owned.ID] = owned
	s.waitingQueue = append(s.waitingQueue, owned)
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
	req.err = context.Canceled

	// If not actively executing an iteration, immediately mark canceled and free resources
	if !req.activeExecution {
		req.State = StateCanceled
		req.completedAt = time.Now()
		s.cleanupRequestResourcesLocked(req)
		s.retainTerminalLocked(req)
		s.removeWaitingLocked(req)
	}
	return nil
}

// Schedule selects requests for the next forward pass iteration according to token budgets,
// memory availability, and fairness policies. It returns ErrExecutionInFlight
// until the previous nonempty iteration has been completed.
func (s *BatchScheduler) Schedule() (*ScheduledIteration, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.configErr != nil {
		return nil, s.configErr
	}
	s.expireWaitingLocked(time.Now())

	if s.inFlight != nil {
		return nil, ErrExecutionInFlight
	}

	// Check before admission or KV growth; invalid state must not invent input.
	for _, req := range s.requests {
		if req.State == StateDecoding && len(req.GeneratedTokens) == 0 {
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
	if admissionSlots == 0 && s.cfg.FairnessPolicy == PolicyFairShare && len(s.waitingQueue) > 0 {
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
	sort.Slice(activeDecodes, func(i, j int) bool { return activeDecodes[i].admissionOrdinal < activeDecodes[j].admissionOrdinal })
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

		chunkTokens := append([]int32(nil), req.PromptTokens[req.PrefilledTokens:req.PrefilledTokens+chunkSize]...)
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
			IsLastPrefill:   isLast,
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

	if len(iter.AllocatedRows) > 0 {
		// Keep completion progress independent of caller-owned row slices.
		rows := append([]BatchRowBinding(nil), iter.AllocatedRows...)
		for i := range rows {
			rows[i].Tokens = append([]int32(nil), rows[i].Tokens...)
		}
		s.inFlight = &ScheduledIteration{StepID: iter.StepID, AllocatedRows: rows}
	}
	return iter, nil
}

// CompleteStep updates request states after iteration execution completes.
// It applies newly generated tokens, transitions completed prefills to decodes,
// processes cancellations, and releases retained KV memory only after invocation is finished.
// The caller must wait for execution to finish before calling CompleteStep. Final
// prefill and decode rows need a token or row error; intermediate prefills need
// no token. Invalid results leave the step outstanding for corrected completion.
func (s *BatchScheduler) CompleteStep(iter *ScheduledIteration, generatedTokens map[string]int32, rowErrors map[string]error) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if iter == nil {
		return nil
	}

	if s.inFlight == nil || iter.StepID != s.inFlight.StepID {
		return fmt.Errorf("%w: no outstanding step %d", ErrInvalidRequest, iter.StepID)
	}
	rows := s.inFlight.AllocatedRows
	byID := make(map[string]BatchRowBinding, len(rows))
	for _, row := range rows {
		byID[row.SeqID] = row
	}
	// Validate the entire result before committing any row. A missing or
	// malformed result can be corrected without advancing request history.
	for id := range generatedTokens {
		row, ok := byID[id]
		if !ok {
			return fmt.Errorf("%w: unexpected token row %q", ErrInvalidRequest, id)
		}
		if !s.requests[id].cancelRequested && !rowNeedsSample(row) {
			return fmt.Errorf("%w: intermediate prefill token for %q", ErrInvalidRequest, id)
		}
	}
	for id, err := range rowErrors {
		if _, ok := byID[id]; !ok || err == nil {
			return fmt.Errorf("%w: invalid error row %q", ErrInvalidRequest, id)
		}
	}
	for _, row := range rows {
		req := s.requests[row.SeqID]
		if req.cancelRequested || rowErrors[row.SeqID] != nil {
			continue
		}
		if _, ok := generatedTokens[row.SeqID]; rowNeedsSample(row) && !ok {
			return fmt.Errorf("%w: missing output row %q", ErrInvalidRequest, row.SeqID)
		}
	}

	for _, row := range rows {
		req := s.requests[row.SeqID]
		req.activeExecution = false
		// Execution errors take precedence over cancellation, as before.
		if rowErr := rowErrors[row.SeqID]; rowErr != nil {
			req.err = rowErr
			req.State = StateCompleted
			req.completedAt = time.Now()
			s.cleanupRequestResourcesLocked(req)
			s.retainTerminalLocked(req)
			continue
		}
		if req.cancelRequested {
			req.State = StateCanceled
			req.completedAt = time.Now()
			s.cleanupRequestResourcesLocked(req)
			s.retainTerminalLocked(req)
			continue
		}
		if row.IsPrefill {
			req.PrefilledTokens += len(row.Tokens)
			if row.IsLastPrefill {
				req.State = StateDecoding
			}
		}
		if rowNeedsSample(row) {
			req.GeneratedTokens = append(req.GeneratedTokens, generatedTokens[row.SeqID])
			if req.MaxTokens > 0 && len(req.GeneratedTokens) >= req.MaxTokens {
				req.State = StateCompleted
				req.completedAt = time.Now()
				s.cleanupRequestResourcesLocked(req)
				s.retainTerminalLocked(req)
			}
		}
	}
	s.inFlight = nil
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
			if prefills[i].PrefilledTokens != prefills[j].PrefilledTokens {
				return prefills[i].PrefilledTokens < prefills[j].PrefilledTokens
			}
			return prefills[i].admissionOrdinal < prefills[j].admissionOrdinal
		})
	case PolicyFCFS, PolicyPriorityDecode:
		sort.SliceStable(prefills, func(i, j int) bool {
			return prefills[i].admissionOrdinal < prefills[j].admissionOrdinal
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
// to be executed only when the current request completely finishes and execution
// has completed. Unknown or retired IDs release immediately; callbacks cannot
// attach to a future admission reusing the same ID.
func (s *BatchScheduler) RegisterResourceRelease(seqID string, cb func()) {
	s.mu.Lock()
	defer s.mu.Unlock()

	req, ok := s.requests[seqID]
	if !ok || req.IsFinished() && !req.activeExecution {
		// Already finished and inactive, call immediately
		if cb != nil {
			cb()
		}
		return
	}
	s.pendingReleases[seqID] = append(s.pendingReleases[seqID], cb)
}

// Request returns a detached status snapshot. Finished snapshots are retained
// up to MaxFinishedRequests; retired IDs may be submitted as new admissions.
func (s *BatchScheduler) Request(id string) (*Request, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	req, ok := s.requests[id]
	if !ok {
		return nil, false
	}
	return copyRequest(req), true
}

func (s *BatchScheduler) removeWaitingLocked(req *Request) {
	for i, waiting := range s.waitingQueue {
		if waiting == req {
			copy(s.waitingQueue[i:], s.waitingQueue[i+1:])
			s.waitingQueue[len(s.waitingQueue)-1] = nil
			s.waitingQueue = s.waitingQueue[:len(s.waitingQueue)-1]
			return
		}
	}
}

func (s *BatchScheduler) retainTerminalLocked(req *Request) {
	if !req.IsFinished() || req.activeExecution || req.retainedTerminal {
		return
	}
	req.retainedTerminal = true
	if req.OnFinish != nil {
		callback := req.OnFinish
		req.OnFinish = nil
		callback(copyRequest(req))
	}
	s.finished = append(s.finished, req)
	if len(s.finished) > s.cfg.MaxFinishedRequests {
		old := s.finished[0]
		delete(s.requests, old.ID)
		copy(s.finished, s.finished[1:])
		s.finished[len(s.finished)-1] = nil
		s.finished = s.finished[:len(s.finished)-1]
	}
}

// Expiration applies only to waiting work, never prefills, decodes or in-flight
// storage. Zero MaxWaitingTimeout disables expiration. Submit and Schedule
// observe expiry without starting a background timer.
func (s *BatchScheduler) expireWaitingLocked(now time.Time) {
	if s.cfg.MaxWaitingTimeout <= 0 {
		return
	}
	remaining := s.waitingQueue[:0]
	for _, req := range s.waitingQueue {
		if now.Sub(req.admittedAt) >= s.cfg.MaxWaitingTimeout {
			req.State, req.err, req.completedAt = StateCanceled, ErrWaitingTimeout, now
			s.cleanupRequestResourcesLocked(req)
			s.retainTerminalLocked(req)
		} else {
			remaining = append(remaining, req)
		}
	}
	clear(s.waitingQueue[len(remaining):])
	s.waitingQueue = remaining
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
