package cudasrv

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabatch"
)

// BatchEngineAdapter wraps cudabatch.BatchScheduler and an ExecutionEngine
// into cudasrv.Engine, enabling live continuous batching for HTTP requests.
type BatchEngineAdapter struct {
	scheduler   *cudabatch.BatchScheduler
	execEngine  cudabatch.ExecutionEngine
	tokenizer   Tokenizer
	bucket      cudabatch.ShapeBucket
	blockSize   int
	stepTicker  time.Duration
	contTracker *cudabatch.ContinuationTracker

	mu        sync.Mutex
	listeners map[string]chan string
	doneChans map[string]chan error
}

// AdapterOption configures BatchEngineAdapter.
type AdapterOption func(*BatchEngineAdapter)

// WithStepTicker sets the background iteration polling interval.
func WithStepTicker(d time.Duration) AdapterOption {
	return func(a *BatchEngineAdapter) {
		a.stepTicker = d
	}
}

// WithShapeBucket sets the shape bucket used for binding.
func WithShapeBucket(b cudabatch.ShapeBucket) AdapterOption {
	return func(a *BatchEngineAdapter) {
		a.bucket = b
	}
}

// WithContinuationTracker sets the multi-turn session continuation tracker.
func WithContinuationTracker(ct *cudabatch.ContinuationTracker) AdapterOption {
	return func(a *BatchEngineAdapter) {
		a.contTracker = ct
	}
}

// NewBatchEngineAdapter creates an adapter bridging cudabatch and cudasrv.
func NewBatchEngineAdapter(
	scheduler *cudabatch.BatchScheduler,
	execEngine cudabatch.ExecutionEngine,
	tokenizer Tokenizer,
	opts ...AdapterOption,
) *BatchEngineAdapter {
	a := &BatchEngineAdapter{
		scheduler:   scheduler,
		execEngine:  execEngine,
		tokenizer:   tokenizer,
		bucket:      cudabatch.ShapeBucket{BatchSize: 16, MaxSeqLen: 2048, MaxTokens: 2048},
		blockSize:   16,
		stepTicker:  1 * time.Millisecond,
		contTracker: cudabatch.NewContinuationTracker(),
		listeners:   make(map[string]chan string),
		doneChans:   make(map[string]chan error),
	}
	for _, opt := range opts {
		if opt != nil {
			opt(a)
		}
	}
	return a
}

// ContinuationTracker returns the active multi-turn continuation tracker.
func (a *BatchEngineAdapter) ContinuationTracker() *cudabatch.ContinuationTracker {
	return a.contTracker
}

// Generate implements the cudasrv.Engine interface.
func (a *BatchEngineAdapter) Generate(ctx context.Context, req ChatCompletionRequest, tokenChan chan<- string) error {
	return a.generate(ctx, req, tokenChan, nil)
}

// GenerateWithUsage counts the rendered prompt's tokenizer IDs and the
// scheduler's generated token IDs. Stream chunk lengths do not affect usage.
func (a *BatchEngineAdapter) GenerateWithUsage(ctx context.Context, req ChatCompletionRequest, tokenChan chan<- string) (UsageStats, error) {
	var usage UsageStats
	err := a.generate(ctx, req, tokenChan, &usage)
	return usage, err
}

func (a *BatchEngineAdapter) generate(ctx context.Context, req ChatCompletionRequest, tokenChan chan<- string, usage *UsageStats) error {
	if ctx == nil || tokenChan == nil {
		return fmt.Errorf("cudasrv: adapter requires context and token channel")
	}
	if a.scheduler == nil || a.execEngine == nil {
		return fmt.Errorf("cudasrv: adapter missing scheduler or execution engine")
	}
	if a.tokenizer == nil {
		return fmt.Errorf("cudasrv: adapter missing tokenizer")
	}

	// 1. Build prompt string from messages
	fullPrompt := ""
	for _, msg := range req.Messages {
		fullPrompt += msg.Role + ": " + msg.Content + "\n"
	}
	fullPrompt += "assistant: "

	tokens, err := a.tokenizer.Encode(fullPrompt)
	if err != nil {
		return fmt.Errorf("cudasrv: tokenize prompt: %w", err)
	}
	if len(tokens) == 0 {
		return fmt.Errorf("cudasrv: tokenizer returned an empty prompt")
	}

	if usage != nil {
		usage.PromptTokens = len(tokens)
	}

	reqID := fmt.Sprintf("req-%d", time.Now().UnixNano())
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 16
	}

	vis := cudabatch.VisibilityDefault
	if req.ResponsesVisible {
		vis = cudabatch.VisibilityResponsesVisible
	} else if req.ThinkingVisible {
		vis = cudabatch.VisibilityThinkingVisible
	}

	batchReq := &cudabatch.Request{
		ID:           reqID,
		PromptTokens: tokens,
		MaxTokens:    maxTokens,
		Visibility:   vis,
		SessionID:    req.SessionID,
	}

	// Check multi-turn continuation
	if req.SessionID != "" && a.contTracker != nil {
		if sess, ok := a.contTracker.GetSession(req.SessionID); ok && len(sess.Tokens) > 0 {
			// Suffix tokenization on boundary
			_, delta, err := cudabatch.SpliceContinuation(a.tokenizer, sess.Tokens, sess.FullText, fullPrompt, 4)
			if err == nil && len(delta) > 0 {
				// Reusing prefix tokens from previous turn
				batchReq.PromptTokens = delta
				batchReq.PromptBlockCount = len(sess.BlockTable)
			}
		}
	}

	// Set up listener for generated tokens
	tokListener := make(chan string, 32)
	doneChan := make(chan error, 1)

	a.mu.Lock()
	a.listeners[reqID] = tokListener
	a.doneChans[reqID] = doneChan
	a.mu.Unlock()

	defer func() {
		a.mu.Lock()
		delete(a.listeners, reqID)
		delete(a.doneChans, reqID)
		a.mu.Unlock()
	}()

	if err := a.scheduler.Submit(batchReq); err != nil {
		return fmt.Errorf("cudasrv: submit to batch scheduler: %w", err)
	}

	forward := func(token string) error {
		// Empty tokenizer pieces carry no text, but remain counted by the
		// scheduler's token-ID usage.
		if token == "" {
			return nil
		}
		select {
		case tokenChan <- token:
			return nil
		case <-ctx.Done():
			_ = a.scheduler.Cancel(reqID)
			return ctx.Err()
		}
	}
	// Forward tokens until finished or context canceled.
	for {
		select {
		case <-ctx.Done():
			// Request cancellation: do NOT immediately release KV blocks or output buffer;
			// scheduler.Cancel defers release until the active GPU/host step completes.
			_ = a.scheduler.Cancel(reqID)
			return ctx.Err()
		case tok, ok := <-tokListener:
			if !ok {
				tokListener = nil
				continue
			}
			if err := forward(tok); err != nil {
				return err
			}
		case err := <-doneChan:
			if err != nil {
				return err
			}
			// StepIteration queues every piece before completion. Selecting
			// completion first must not drop the pieces already in the queue.
		drain:
			for {
				select {
				case token, ok := <-tokListener:
					if !ok {
						break drain
					}
					if err := forward(token); err != nil {
						return err
					}
				default:
					break drain
				}
			}
			if usage != nil {
				request, ok := a.scheduler.Request(reqID)
				if !ok {
					return fmt.Errorf("cudasrv: completed request missing token usage")
				}
				usage.CompletionTokens = len(request.GeneratedTokens)
				if usage.PromptTokens > int(^uint(0)>>1)-usage.CompletionTokens {
					return fmt.Errorf("cudasrv: token usage overflows")
				}
				usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
			}
			// Save continuation in session if requested
			if req.SessionID != "" && a.contTracker != nil {
				if r, ok := a.scheduler.Request(reqID); ok {
					a.contTracker.StoreSession(req.SessionID, r.TotalTokensSlice(), r.BlockTable, vis, fullPrompt)
				}
			}
			return err
		}
	}
}

// StepIteration advances the continuous batch scheduler by one forward iteration.
// Can be run in a loop or driven deterministically by tests.
func (a *BatchEngineAdapter) StepIteration(ctx context.Context) error {
	if ctx == nil || a.scheduler == nil || a.execEngine == nil || a.tokenizer == nil {
		return fmt.Errorf("cudasrv: adapter requires context, scheduler, engine, and tokenizer")
	}
	iter, err := a.scheduler.Schedule()
	if err != nil {
		return fmt.Errorf("cudasrv: scheduler error: %w", err)
	}
	if iter == nil || len(iter.AllocatedRows) == 0 {
		return nil
	}

	// Bind rows and validate shapes
	metadata, err := cudabatch.ValidateAndBindRows(iter.AllocatedRows, a.bucket, a.blockSize)
	if err != nil {
		return fmt.Errorf("cudasrv: validate row bindings: %w", err)
	}

	execStep := &cudabatch.ExecutionStep{
		StepID:    iter.StepID,
		Iteration: iter,
		Metadata:  metadata,
	}

	genTokens, rowErrors, execErr := a.execEngine.ExecuteForward(ctx, execStep)
	if execErr != nil {
		if rowErrors == nil {
			rowErrors = make(map[string]error)
		}
		for _, row := range iter.AllocatedRows {
			rowErrors[row.SeqID] = execErr
		}
		genTokens = nil
	}

	// Update scheduler
	if err := a.scheduler.CompleteStep(iter, genTokens, rowErrors); err != nil {
		return fmt.Errorf("cudasrv: complete scheduler step: %w", err)
	}

	// Dispatch tokens to listeners
	a.mu.Lock()
	var canceled []string
	defer func() {
		a.mu.Unlock()
		for _, id := range canceled {
			_ = a.scheduler.Cancel(id)
		}
	}()

	for seqID, tokID := range genTokens {
		if rowErrors[seqID] != nil {
			continue
		}
		if listener, ok := a.listeners[seqID]; ok {
			text, decodeErr := a.tokenizer.Decode([]int32{tokID})
			if decodeErr != nil {
				canceled = append(canceled, seqID)
				if done := a.doneChans[seqID]; done != nil {
					select {
					case done <- fmt.Errorf("cudasrv: decode token: %w", decodeErr):
					default:
					}
				}
				continue
			}
			select {
			case listener <- text:
			default:
				// A bounded queue may refuse a slow consumer; it must not
				// silently discard text and later report success.
				canceled = append(canceled, seqID)
				if done := a.doneChans[seqID]; done != nil {
					select {
					case done <- fmt.Errorf("cudasrv: output consumer too slow"):
					default:
					}
				}
			}
		}
	}

	// Check completed or errored requests
	for _, row := range iter.AllocatedRows {
		if r, ok := a.scheduler.Request(row.SeqID); ok && r.IsFinished() {
			if done, ok := a.doneChans[row.SeqID]; ok {
				err := rowErrors[row.SeqID]
				if r.State == cudabatch.StateCanceled {
					err = context.Canceled
				}
				select {
				case done <- err:
				default:
				}
			}
		}
	}
	if execErr != nil {
		return fmt.Errorf("cudasrv: execute forward: %w", execErr)
	}
	return nil
}
