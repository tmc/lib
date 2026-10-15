package cudabatch_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

// historyEngine models a position-sensitive recurrent forward pass. Submission
// stages inputs; completion commits the model state; sampling reads its logits.
// It uses only the public execution interface, not scheduler internals.
type historyEngine struct {
	history   map[string][]int32
	state     map[string]int
	events    []string
	submitted chan struct{}
	complete  <-chan struct{}
	failure   error
	rowError  map[string]error
}

func newHistoryEngine() *historyEngine {
	return &historyEngine{history: make(map[string][]int32), state: make(map[string]int)}
}

func (e *historyEngine) ExecuteForward(ctx context.Context, step *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
	e.events = append(e.events, "forward")
	staged := make(map[string][]int32)
	states := make(map[string]int)
	for _, row := range step.Metadata.RowBindings {
		old := e.history[row.SeqID]
		if row.StartPos != len(old) || row.SequenceLength != len(old)+len(row.Tokens) {
			return nil, nil, fmt.Errorf("history position for %s: start=%d end=%d consumed=%d", row.SeqID, row.StartPos, row.SequenceLength, len(old))
		}
		staged[row.SeqID] = append(append([]int32(nil), old...), row.Tokens...)
		state := e.state[row.SeqID]
		for i, token := range row.Tokens {
			state = (state*7 + (int(token)+1)*(row.StartPos+i+3)) % 17
		}
		states[row.SeqID] = state
	}
	if e.submitted != nil {
		e.submitted <- struct{}{}
	}
	if e.complete != nil {
		select {
		case <-e.complete:
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if e.failure != nil {
		return nil, nil, e.failure
	}
	e.events = append(e.events, "complete")
	out := make(map[string]int32)
	for _, row := range step.Metadata.RowBindings {
		if e.rowError[row.SeqID] != nil {
			continue
		}
		e.history[row.SeqID], e.state[row.SeqID] = staged[row.SeqID], states[row.SeqID]
		if !row.IsPrefill || row.IsLastPrefill {
			// A small explicit logits vector makes sampling a real argmax.
			var logits [17]int
			for token := range logits {
				distance := token - states[row.SeqID]
				logits[token] = -distance * distance
			}
			best := 0
			for token := 1; token < len(logits); token++ {
				if logits[token] > logits[best] {
					best = token
				}
			}
			out[row.SeqID] = int32(best)
			e.events = append(e.events, "sample:"+row.SeqID)
		}
	}
	return out, e.rowError, nil
}

// sequentialGreedy recomputes the entire unbatched prefix for each output. It
// knows nothing about chunking, row flags, scheduler states, or engine caches.
func sequentialGreedy(prompt []int32, count int) []int32 {
	prefix := append([]int32(nil), prompt...)
	var generated []int32
	for range count {
		state := 0
		for position, token := range prefix {
			state = (7*state + (position+3)*(int(token)+1)) % 17
		}
		best, bestScore := 0, -1000
		for candidate := 0; candidate < 17; candidate++ {
			score := -(candidate - state) * (candidate - state)
			if score > bestScore {
				best, bestScore = candidate, score
			}
		}
		generated = append(generated, int32(best))
		prefix = append(prefix, int32(best))
	}
	return generated
}

func contractStep(t *testing.T, scheduler *cudabatch.BatchScheduler) *cudabatch.ExecutionStep {
	t.Helper()
	iteration, err := scheduler.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := cudabatch.ValidateAndBindRows(iteration.AllocatedRows, cudabatch.ShapeBucket{BatchSize: 4, MaxSeqLen: 64, MaxTokens: 16}, 2)
	if err != nil {
		t.Fatal(err)
	}
	return &cudabatch.ExecutionStep{StepID: iteration.StepID, Iteration: iteration, Metadata: metadata}
}

func TestSchedulerSequentialGreedy(t *testing.T) {
	for _, chunk := range []int{1, 2, 3, 4} {
		t.Run(fmt.Sprint(chunk), func(t *testing.T) {
			allocator := cudakv.NewBlockAllocator(2, 128)
			scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 3, MaxBudgetTokens: 6, ChunkSize: chunk, FairnessPolicy: cudabatch.PolicyFairShare}, cudabatch.NewMemoryBudgetTracker(allocator))
			requests := []*cudabatch.Request{
				{ID: "short", PromptTokens: []int32{0}, MaxTokens: 4},
				{ID: "long", PromptTokens: []int32{2, 5, 1, 9, 0, 3, 8}, MaxTokens: 3},
				{ID: "late", PromptTokens: []int32{16, 4, 7}, MaxTokens: 1},
			}
			for _, request := range requests[:2] {
				if err := scheduler.Submit(request); err != nil {
					t.Fatal(err)
				}
			}
			engine := newHistoryEngine()
			mixed, finished := false, false
			for iteration := 0; iteration < 64; iteration++ {
				if iteration == 1 {
					if err := scheduler.Submit(requests[2]); err != nil {
						t.Fatal(err)
					}
				}
				step := contractStep(t, scheduler)
				prefill, decode := false, false
				for _, row := range step.Metadata.RowBindings {
					request, _ := scheduler.Request(row.SeqID)
					if row.StartPos != request.ContextTokens() {
						t.Fatalf("scheduled unconsumed position: %+v request=%+v", row, request)
					}
					if row.IsPrefill {
						prefill = true
						if row.IsLastPrefill != (row.SequenceLength == len(request.PromptTokens)) {
							t.Fatalf("incorrect final-prefill marker: %+v", row)
						}
					} else {
						decode = true
						if row.Tokens[0] != request.GeneratedTokens[len(request.GeneratedTokens)-1] {
							t.Fatalf("decode did not consume newest emitted token: %+v", row)
						}
					}
				}
				mixed = mixed || prefill && decode
				tokens, rowErrors, err := engine.ExecuteForward(context.Background(), step)
				if err != nil {
					t.Fatal(err)
				}
				if err := scheduler.CompleteStep(step.Iteration, tokens, rowErrors); err != nil {
					t.Fatal(err)
				}
				finished = true
				for _, original := range requests {
					request, ok := scheduler.Request(original.ID)
					finished = finished && ok && request.IsFinished()
					if ok && request.ContextTokens() != len(engine.history[original.ID]) {
						t.Fatalf("committed consumed history differs: %+v history=%v", request, engine.history[original.ID])
					}
				}
				if finished {
					break
				}
			}
			if !finished || !mixed {
				t.Fatalf("workload did not complete with mixed prefill/decode: finished=%v mixed=%v", finished, mixed)
			}
			for _, original := range requests {
				request, _ := scheduler.Request(original.ID)
				want := sequentialGreedy(original.PromptTokens, original.MaxTokens)
				if !reflect.DeepEqual(request.GeneratedTokens, want) {
					t.Fatalf("%s greedy outputs=%v want=%v", original.ID, request.GeneratedTokens, want)
				}
				consumed := append(append([]int32(nil), original.PromptTokens...), want[:len(want)-1]...)
				if !reflect.DeepEqual(engine.history[original.ID], consumed) {
					t.Fatalf("%s consumed inputs=%v want=%v", original.ID, engine.history[original.ID], consumed)
				}
			}
			completed, samples := false, 0
			for _, event := range engine.events {
				switch event {
				case "forward":
					completed = false
				case "complete":
					completed = true
				default:
					if !completed {
						t.Fatalf("sampling preceded execution completion: %v", engine.events)
					}
					samples++
				}
			}
			if samples != 8 {
				t.Fatalf("sample count=%d want=8", samples)
			}
			if allocator.NumFreeBlocks() != 128 {
				t.Fatal("completed workload retained KV reservations")
			}
			empty, err := scheduler.Schedule()
			if err != nil || len(empty.AllocatedRows) != 0 {
				t.Fatalf("completed workload rescheduled: %+v, %v", empty, err)
			}
		})
	}
}

func TestSchedulerTokenCompletionContract(t *testing.T) {
	for _, phase := range []string{"intermediate", "final", "decode"} {
		t.Run(phase, func(t *testing.T) {
			allocator := cudakv.NewBlockAllocator(2, 8)
			scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 1, MaxBudgetTokens: 2, ChunkSize: 2}, cudabatch.NewMemoryBudgetTracker(allocator))
			prompt := []int32{16} // Its genuine greedy first token is zero.
			if phase == "intermediate" {
				prompt = []int32{1, 2, 3}
			}
			if err := scheduler.Submit(&cudabatch.Request{ID: "a", PromptTokens: prompt, MaxTokens: 3}); err != nil {
				t.Fatal(err)
			}
			engine := newHistoryEngine()
			if phase == "decode" {
				step := contractStep(t, scheduler)
				tokens, rowErrors, err := engine.ExecuteForward(context.Background(), step)
				if err != nil || scheduler.CompleteStep(step.Iteration, tokens, rowErrors) != nil {
					t.Fatalf("prefill: %v", err)
				}
			}
			step := contractStep(t, scheduler)
			before, _ := scheduler.Request("a")
			free := allocator.NumFreeBlocks()
			if _, err := scheduler.Schedule(); !errors.Is(err, cudabatch.ErrExecutionInFlight) || allocator.NumFreeBlocks() != free {
				t.Fatalf("outstanding iteration scheduled twice: %v", err)
			}
			tokens, rowErrors, err := engine.ExecuteForward(context.Background(), step)
			if err != nil {
				t.Fatal(err)
			}
			invalid := []map[string]int32{nil, {"unknown": 1}}
			if phase == "intermediate" {
				invalid[0] = map[string]int32{"a": 1}
			}
			for _, result := range invalid {
				if err := scheduler.CompleteStep(step.Iteration, result, nil); !errors.Is(err, cudabatch.ErrInvalidRequest) {
					t.Fatalf("invalid result accepted: %v", err)
				}
				after, _ := scheduler.Request("a")
				if after.ContextTokens() != before.ContextTokens() || after.PrefilledTokens != before.PrefilledTokens || !reflect.DeepEqual(after.GeneratedTokens, before.GeneratedTokens) || allocator.NumFreeBlocks() != free {
					t.Fatalf("invalid result advanced request: before=%+v after=%+v", before, after)
				}
			}
			// Correct the result of the same completed forward, without replaying
			// input or issuing another capacity reservation.
			if err := scheduler.CompleteStep(step.Iteration, tokens, rowErrors); err != nil {
				t.Fatal(err)
			}
			after, _ := scheduler.Request("a")
			if after.ContextTokens() != step.Metadata.RowBindings[0].SequenceLength {
				t.Fatalf("corrected completion did not consume exact input: %+v", after)
			}
			if phase == "final" && !reflect.DeepEqual(after.GeneratedTokens, []int32{0}) {
				t.Fatalf("real zero sample was treated as absent: %v", after.GeneratedTokens)
			}
			if err := scheduler.CompleteStep(step.Iteration, tokens, rowErrors); !errors.Is(err, cudabatch.ErrInvalidRequest) {
				t.Fatalf("completed step replayed: %v", err)
			}
		})
	}
}

func TestSchedulerSubmittedTokenSettlement(t *testing.T) {
	for _, phase := range []string{"intermediate", "final", "decode"} {
		for _, outcome := range []string{"cancel", "submission-error", "row-error"} {
			t.Run(phase+"/"+outcome, func(t *testing.T) {
				allocator := cudakv.NewBlockAllocator(2, 8)
				scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 1, MaxBudgetTokens: 2, ChunkSize: 2}, cudabatch.NewMemoryBudgetTracker(allocator))
				prompt := []int32{2}
				if phase == "intermediate" {
					prompt = []int32{2, 3, 4}
				}
				if err := scheduler.Submit(&cudabatch.Request{ID: "a", PromptTokens: prompt, MaxTokens: 3}); err != nil {
					t.Fatal(err)
				}
				engine := newHistoryEngine()
				if phase == "decode" {
					step := contractStep(t, scheduler)
					tokens, rowErrors, err := engine.ExecuteForward(context.Background(), step)
					if err != nil || scheduler.CompleteStep(step.Iteration, tokens, rowErrors) != nil {
						t.Fatalf("prefill: %v", err)
					}
				}
				step := contractStep(t, scheduler)
				before, _ := scheduler.Request("a")
				before.GeneratedTokens = append([]int32(nil), before.GeneratedTokens...)
				releases := 0
				scheduler.RegisterResourceRelease("a", func() { releases++ })
				gate := make(chan struct{})
				engine.complete, engine.submitted = gate, make(chan struct{}, 1)
				failure := errors.New("forward refused")
				if outcome == "submission-error" {
					engine.failure = failure
				} else if outcome == "row-error" {
					engine.rowError = map[string]error{"a": failure}
				}
				type forwardResult struct {
					tokens map[string]int32
					errors map[string]error
					err    error
				}
				done := make(chan forwardResult, 1)
				go func() {
					tokens, rowErrors, err := engine.ExecuteForward(context.Background(), step)
					done <- forwardResult{tokens, rowErrors, err}
				}()
				<-engine.submitted
				free := allocator.NumFreeBlocks()
				if outcome == "cancel" {
					if err := scheduler.Cancel("a"); err != nil {
						t.Fatal(err)
					}
				}
				pending, _ := scheduler.Request("a")
				if pending.ContextTokens() != before.ContextTokens() || !reflect.DeepEqual(pending.GeneratedTokens, before.GeneratedTokens) || pending.IsFinished() || releases != 0 || allocator.NumFreeBlocks() != free {
					t.Fatalf("submitted work advanced/released before completion: %+v", pending)
				}
				if _, err := scheduler.Schedule(); !errors.Is(err, cudabatch.ErrExecutionInFlight) || allocator.NumFreeBlocks() != free {
					t.Fatalf("incomplete execution rescheduled: %v", err)
				}
				close(gate)
				result := <-done
				if result.err != nil {
					result.errors = map[string]error{"a": result.err}
					result.tokens = nil
				}
				if err := scheduler.CompleteStep(step.Iteration, result.tokens, result.errors); err != nil {
					t.Fatal(err)
				}
				after, _ := scheduler.Request("a")
				wantState := cudabatch.StateCompleted
				if outcome == "cancel" {
					wantState = cudabatch.StateCanceled
				}
				if after.State != wantState || after.ContextTokens() != before.ContextTokens() || after.PrefilledTokens != before.PrefilledTokens || !reflect.DeepEqual(after.GeneratedTokens, before.GeneratedTokens) || releases != 1 || allocator.NumFreeBlocks() != 8 {
					t.Fatalf("failed/canceled execution changed history or leaked: before=%+v after=%+v releases=%d free=%d", before, after, releases, allocator.NumFreeBlocks())
				}
			})
		}
	}
}

func TestSchedulerNoInventedDecodeToken(t *testing.T) {
	allocator := cudakv.NewBlockAllocator(2, 8)
	scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{}, cudabatch.NewMemoryBudgetTracker(allocator))
	if err := scheduler.Submit(&cudabatch.Request{ID: "empty"}); !errors.Is(err, cudabatch.ErrInvalidRequest) {
		t.Fatalf("empty prompt accepted: %v", err)
	}
	request := &cudabatch.Request{ID: "a", PromptTokens: []int32{4}, MaxTokens: 3}
	if err := scheduler.Submit(request); err != nil {
		t.Fatal(err)
	}
	step := contractStep(t, scheduler)
	if err := scheduler.CompleteStep(step.Iteration, map[string]int32{"a": 9}, nil); err != nil {
		t.Fatal(err)
	}
	// Retained input is caller-owned. Even with a nonempty prompt, losing the
	// emitted token must refuse instead of replaying the prompt or inventing 0.
	request.GeneratedTokens = nil
	free := allocator.NumFreeBlocks()
	if iteration, err := scheduler.Schedule(); iteration != nil || !errors.Is(err, cudabatch.ErrInvalidRequest) || allocator.NumFreeBlocks() != free {
		t.Fatalf("missing generated input was replaced: %+v, %v", iteration, err)
	}
}

func TestSchedulerCanceledEngineContext(t *testing.T) {
	allocator := cudakv.NewBlockAllocator(2, 8)
	scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{}, cudabatch.NewMemoryBudgetTracker(allocator))
	if err := scheduler.Submit(&cudabatch.Request{ID: "a", PromptTokens: []int32{4}, MaxTokens: 3}); err != nil {
		t.Fatal(err)
	}
	step := contractStep(t, scheduler)
	engine := cudabatch.NewHostDeterministicEngine(0, []int32{9})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tokens, _, err := engine.ExecuteForward(ctx, step)
	if !errors.Is(err, context.Canceled) || len(tokens) != 0 {
		t.Fatalf("canceled engine supplied a sample: tokens=%v err=%v", tokens, err)
	}
	if err := scheduler.CompleteStep(step.Iteration, nil, map[string]error{"a": err}); err != nil {
		t.Fatal(err)
	}
	request, _ := scheduler.Request("a")
	if request.ContextTokens() != 0 || request.PrefilledTokens != 0 || len(request.GeneratedTokens) != 0 || allocator.NumFreeBlocks() != 8 {
		t.Fatalf("canceled execution advanced logical history or retained capacity: %+v", request)
	}
}
