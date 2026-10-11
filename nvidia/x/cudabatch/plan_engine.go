package cudabatch

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// PlanBindingFunc binds actual model inputs, outputs and KV storage. release must
// retain all that storage until called, including after partial submission. The
// engine calls release once after completion and sampling, or on binding failure.
// A non-nil release returned with an error is still called. The model ABI and
// physical KV allocation mapping are supplied by the caller.
type PlanBindingFunc func(context.Context, *ExecutionStep) (*cudaplan.Bindings, func(), error)

// PlanSampleFunc reads completed outputs. Each decode row must have one token or
// row error. Prefill rows may have errors but must not have generated tokens.
// Sampling must not release the storage retained by PlanBindingFunc.
type PlanSampleFunc func(context.Context, *ExecutionStep) (map[string]int32, map[string]error, error)

type planCompletion interface {
	Wait(context.Context) error
	Done() bool
}

// PlanEngine executes an ordinary cudaplan plan. It permits one outstanding step
// and starts no background drain. The engine owns the plan; the caller owns its
// stream and device. Its zero value is unusable. No model or KV ABI is inferred.
type PlanEngine struct {
	mu            sync.Mutex
	plan          *cudaplan.ExecutablePlan
	bind          PlanBindingFunc
	sample        PlanSampleFunc
	invoke        func(*cudaplan.Bindings) (planCompletion, error)
	current, last *planForward
	closed        bool
}

type planForward struct {
	gate       chan struct{}
	step       *ExecutionStep
	completion planCompletion
	release    func()
	released   bool
	done       atomic.Bool
	terminal   error
}

// PendingForwardError transfers completion recovery to the caller. Do not call
// CompleteStep while Done is false. A completion cleanup error can coexist with
// Done being true; Drain must finish cleanup before executing another step.
type PendingForwardError struct {
	engine *PlanEngine
	state  *planForward
	cause  error
}

func (p *PendingForwardError) Error() string { return "cudabatch: pending forward: " + p.cause.Error() }

// Unwrap returns the original execution or wait error, even after recovery.
func (p *PendingForwardError) Unwrap() error { return p.cause }

// Wait retries completion with a recovery context independent of the request.
// It reports the recovery result; the original error remains in Unwrap.
func (p *PendingForwardError) Wait(ctx context.Context) error { return p.engine.drain(ctx, p.state) }

// Done reports native completion and finished storage release callbacks.
func (p *PendingForwardError) Done() bool { return p.state.done.Load() }

// NewPlanEngine takes ownership of plan. bind and sample must describe its actual
// tensor and output ABI. Execution requires a live stream for the plan's
// device. A nil stream permits construction for inspection; execution refuses it.
func NewPlanEngine(plan *cudaplan.ExecutablePlan, stream *cuda.Stream, bind PlanBindingFunc, sample PlanSampleFunc) (*PlanEngine, error) {
	if plan == nil || bind == nil || sample == nil {
		return nil, errors.New("cudabatch: nil plan, binder or sampler")
	}
	e := &PlanEngine{plan: plan, bind: bind, sample: sample}
	e.invoke = func(b *cudaplan.Bindings) (planCompletion, error) {
		c, err := plan.Execute(stream, b)
		if c == nil {
			return nil, err
		}
		return c, err
	}
	return e, nil
}

// CompilePlanEngine compiles and owns a plan. A non-nil engine returned with an
// error retains failed compilation cleanup: retry Close before releasing the
// device. The stream remains owned by the caller. Options specify actual kernels
// and the device contract; this adapter does not compile a model automatically.
func CompilePlanEngine(source *cudair.Snapshot, stream *cuda.Stream, bind PlanBindingFunc, sample PlanSampleFunc, opts ...cudaplan.PlanOption) (*PlanEngine, error) {
	if bind == nil || sample == nil {
		return nil, errors.New("cudabatch: nil binder or sampler")
	}
	plan, err := cudaplan.Compile(source, opts...)
	if plan == nil {
		return nil, err
	}
	e, constructErr := NewPlanEngine(plan, stream, bind, sample)
	return e, errors.Join(err, constructErr)
}

// ExecuteForward validates, binds, submits, waits and samples the plan. Pending
// completion is returned as PendingForwardError with storage still retained.
func (e *PlanEngine) ExecuteForward(ctx context.Context, step *ExecutionStep) (map[string]int32, map[string]error, error) {
	if ctx == nil {
		return nil, nil, errors.New("cudabatch: nil context")
	}
	owned, err := copyPlanStep(step)
	if err != nil {
		return nil, nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, nil, err
	}
	e.mu.Lock()
	if e.closed || e.invoke == nil {
		e.mu.Unlock()
		return nil, nil, errors.New("cudabatch: closed or uninitialized plan engine")
	}
	if e.current != nil {
		e.mu.Unlock()
		return nil, nil, errors.New("cudabatch: plan completion requires drain")
	}
	if e.last != nil && step.StepID <= e.last.step.StepID {
		e.mu.Unlock()
		return nil, nil, errors.New("cudabatch: step id must increase")
	}
	state := &planForward{gate: make(chan struct{}, 1), step: owned}
	e.current, e.last = state, state
	e.mu.Unlock()
	defer func() { state.gate <- struct{}{} }()
	waitCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	if step.CancelWait != nil {
		go func() {
			select {
			case <-step.CancelWait:
				cancel()
			case <-waitCtx.Done():
			}
		}()
	}
	bindings, release, err := e.bind(waitCtx, owned)
	state.release = release
	if err == nil && bindings == nil {
		err = errors.New("cudabatch: binder returned nil bindings")
	}
	if err != nil {
		state.terminal = err
		e.release(state)
		e.clear(state)
		return nil, nil, err
	}
	if err = waitCtx.Err(); err != nil {
		state.terminal = err
		e.release(state)
		e.clear(state)
		return nil, nil, err
	}
	state.completion, err = e.invoke(bindings)
	state.terminal = err
	if state.completion == nil {
		if err == nil {
			err = errors.New("cudabatch: invocation returned no completion")
			state.terminal = err
		}
		e.release(state)
		e.clear(state)
		return nil, nil, err
	}
	waitErr := state.completion.Wait(waitCtx)
	if waitErr != nil || !state.completion.Done() {
		if waitErr == nil {
			waitErr = errors.New("cudabatch: wait returned before completion")
		}
		state.terminal = errors.Join(state.terminal, waitErr)
		if state.completion.Done() {
			e.release(state)
		}
		return nil, nil, &PendingForwardError{e, state, state.terminal}
	}
	if err != nil {
		e.release(state)
		e.clear(state)
		return nil, nil, err
	}
	tokens, rowErrors, err := e.sample(waitCtx, owned)
	if err == nil {
		err = validatePlanResults(owned.Metadata, tokens, rowErrors)
	}
	state.terminal = err
	e.release(state)
	e.clear(state)
	return tokens, rowErrors, err
}
func (e *PlanEngine) release(s *planForward) {
	if s.released {
		return
	}
	if s.release != nil {
		s.release()
	}
	if s.step.ReleaseHook != nil {
		s.step.ReleaseHook()
	}
	s.released = true
	s.done.Store(true)
}
func (e *PlanEngine) clear(s *planForward) {
	e.mu.Lock()
	if e.current == s {
		e.current = nil
	}
	e.mu.Unlock()
}
func (e *PlanEngine) drain(ctx context.Context, s *planForward) error {
	if ctx == nil {
		return errors.New("cudabatch: nil recovery context")
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-s.gate:
	}
	defer func() { s.gate <- struct{}{} }()
	if s.completion == nil {
		return nil
	}
	err := s.completion.Wait(ctx)
	if s.completion.Done() {
		e.release(s)
		if err == nil {
			e.clear(s)
		}
	} else if err == nil {
		err = errors.New("cudabatch: wait returned before completion")
	}
	return err
}

// Drain retries outstanding completion and cleanup. It reports recovery errors;
// original execution errors remain in PendingForwardError and WaitStep.
func (e *PlanEngine) Drain(ctx context.Context) error {
	e.mu.Lock()
	s := e.current
	e.mu.Unlock()
	if s == nil {
		return nil
	}
	return e.drain(ctx, s)
}

// WaitStep recovers only the most recent attempted step and preserves its terminal
// error. Unknown, future and replaced IDs are refused.
func (e *PlanEngine) WaitStep(ctx context.Context, id int64) error {
	e.mu.Lock()
	s := e.last
	e.mu.Unlock()
	if s == nil || s.step.StepID != id {
		return errors.New("cudabatch: unknown execution step")
	}
	return e.waitStep(ctx, s)
}

// StepDone reports ended storage ownership for the exact last attempted step.
// Unknown, future and replaced IDs return false.
func (e *PlanEngine) StepDone(id int64) bool {
	e.mu.Lock()
	s := e.last
	e.mu.Unlock()
	return s != nil && s.step.StepID == id && s.done.Load()
}

// Close closes the owned plan. Pending completion must first be drained with an
// explicit recovery context. Failed plan cleanup can be retried.
func (e *PlanEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.current != nil {
		return errors.New("cudabatch: cannot close pending plan")
	}
	if e.closed {
		return nil
	}
	if e.plan != nil {
		if err := e.plan.Close(); err != nil {
			return err
		}
	}
	e.closed = true
	return nil
}

func copyPlanStep(step *ExecutionStep) (*ExecutionStep, error) {
	if step == nil || step.Metadata == nil {
		return nil, errors.New("cudabatch: nil execution step or metadata")
	}
	m := step.Metadata
	if step.Iteration != nil && (step.Iteration.StepID != step.StepID || !reflect.DeepEqual(step.Iteration.AllocatedRows, m.RowBindings) || step.Iteration.TotalTokens != m.TotalTokens) {
		return nil, errors.New("cudabatch: iteration and metadata disagree")
	}
	if m.BatchShape.Rank() != 2 || m.BatchShape.Dim(0) <= 0 || m.BatchShape.Dim(1) <= 0 || m.BlockSize <= 0 || len(m.RowBindings) == 0 || len(m.RowBindings) > m.BatchShape.Dim(0) {
		return nil, errors.New("cudabatch: invalid batch dimensions")
	}
	rows := append([]BatchRowBinding(nil), m.RowBindings...)
	seen := make(map[string]bool)
	for i := range rows {
		r := &rows[i]
		if r.RowIndex != i || r.SeqID == "" || seen[r.SeqID] || r.StartPos < 0 || r.SequenceLength <= r.StartPos || r.SequenceLength > m.BatchShape.Dim(1) {
			return nil, errors.New("cudabatch: invalid row identity or bounds")
		}
		for _, block := range r.BlockTable {
			if block < 0 || uint64(block) > uint64(1<<31-1) {
				return nil, errors.New("cudabatch: invalid physical block index")
			}
		}
		seen[r.SeqID] = true
		n := 1
		if r.IsPrefill {
			n = len(r.Tokens)
		}
		if n <= 0 || n != r.SequenceLength-r.StartPos {
			return nil, errors.New("cudabatch: invalid row token extent")
		}
		r.Tokens = append([]int32(nil), r.Tokens...)
		r.BlockTable = append([]int(nil), r.BlockTable...)
	}
	b, s := m.BatchShape.Dim(0), m.BatchShape.Dim(1)
	if b > len(m.AttentionMask)/s || b*s != len(m.AttentionMask) || len(m.SequenceLens) != b || m.MaxTokens <= 0 {
		return nil, errors.New("cudabatch: invalid metadata storage")
	}
	blocks := (s-1)/m.BlockSize + 1
	if b > len(m.BlockTableFlat)/blocks || b*blocks != len(m.BlockTableFlat) {
		return nil, errors.New("cudabatch: invalid block table storage")
	}
	expected, err := ValidateAndBindRows(rows, ShapeBucket{b, s, m.MaxTokens}, m.BlockSize)
	if err != nil {
		return nil, err
	}
	if expected.TotalTokens > m.MaxTokens || expected.TotalTokens != m.TotalTokens || !reflect.DeepEqual(expected.SequenceLens, m.SequenceLens) || !reflect.DeepEqual(expected.AttentionMask, m.AttentionMask) || !reflect.DeepEqual(expected.BlockTableFlat, m.BlockTableFlat) {
		return nil, errors.New("cudabatch: inconsistent batch metadata")
	}
	out := *step
	out.Metadata = expected
	if step.Iteration != nil {
		iteration := *step.Iteration
		iteration.AllocatedRows = append([]BatchRowBinding(nil), expected.RowBindings...)
		for i := range iteration.AllocatedRows {
			row := &iteration.AllocatedRows[i]
			row.Tokens = append([]int32(nil), row.Tokens...)
			row.BlockTable = append([]int(nil), row.BlockTable...)
		}
		iteration.DecodeSeqIDs = append([]string(nil), iteration.DecodeSeqIDs...)
		iteration.PrefillChunks = append([]cudakv.PrefillChunk(nil), iteration.PrefillChunks...)
		for i := range iteration.PrefillChunks {
			iteration.PrefillChunks[i].Tokens = append([]int32(nil), iteration.PrefillChunks[i].Tokens...)
		}
		out.Iteration = &iteration
	}
	return &out, nil
}
func validatePlanResults(m *BatchMetadata, tokens map[string]int32, rowErrors map[string]error) error {
	rows := make(map[string]BatchRowBinding)
	for _, r := range m.RowBindings {
		rows[r.SeqID] = r
	}
	for id := range tokens {
		r, ok := rows[id]
		if !ok || r.IsPrefill {
			return fmt.Errorf("cudabatch: unexpected token row %q", id)
		}
		if rowErrors[id] != nil {
			return fmt.Errorf("cudabatch: token and error for row %q", id)
		}
	}
	for id, err := range rowErrors {
		if _, ok := rows[id]; !ok || err == nil {
			return fmt.Errorf("cudabatch: invalid error row %q", id)
		}
	}
	for _, r := range m.RowBindings {
		if !r.IsPrefill {
			if _, ok := tokens[r.SeqID]; !ok && rowErrors[r.SeqID] == nil {
				return fmt.Errorf("cudabatch: missing output row %q", r.SeqID)
			}
		}
	}
	return nil
}

func (e *PlanEngine) waitStep(ctx context.Context, s *planForward) error {
	err := e.drain(ctx, s)
	if ctx == nil {
		return err
	}
	select {
	case <-ctx.Done():
		return errors.Join(err, ctx.Err())
	case <-s.gate:
	}
	terminal := s.terminal
	s.gate <- struct{}{}
	return errors.Join(terminal, err)
}
