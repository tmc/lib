package cudaserver_test

import (
	"context"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaserver"
	"testing"
)

type pendingExecution struct {
	complete bool
	waitErr  error
}

func (p *pendingExecution) Error() string              { return "device wait failed" }
func (p *pendingExecution) Wait(context.Context) error { return p.waitErr }
func (p *pendingExecution) Done() bool                 { return p.complete }

func TestAdapterRetainsPendingExecution(t *testing.T) {
	scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 4, MaxBudgetTokens: 64, ChunkSize: 16}, cudabatch.NewMemoryBudgetTracker(cudakv.NewBlockAllocator(16, 20)))
	if err := scheduler.Submit(&cudabatch.Request{ID: "pending", PromptTokens: []int32{1}, MaxTokens: 1}); err != nil {
		t.Fatal(err)
	}
	releases, calls := 0, 0
	scheduler.RegisterResourceRelease("pending", func() { releases++ })
	pending := &pendingExecution{waitErr: context.Canceled}
	engine := executionFunc(func(context.Context, *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
		calls++
		return nil, nil, pending
	})
	adapter := cudaserver.NewBatchEngineAdapter(scheduler, engine, &simpleWordTokenizer{})
	if err := adapter.StepIteration(context.Background()); err == nil {
		t.Fatal("pending execution succeeded")
	}
	if releases != 0 {
		t.Fatal("released before completion")
	}
	if err := adapter.StepIteration(context.Background()); !errors.Is(err, context.Canceled) {
		t.Fatalf("retry: %v", err)
	}
	if calls != 1 || releases != 0 {
		t.Fatalf("rescheduled/released: %d/%d", calls, releases)
	}
	pending.complete, pending.waitErr = true, errors.New("event cleanup failed")
	if err := adapter.Drain(context.Background()); err == nil {
		t.Fatal("cleanup failure hidden")
	}
	request, _ := scheduler.Request("pending")
	if releases != 1 || !request.IsFinished() {
		t.Fatal("completed failed step not settled")
	}
	pending.waitErr = nil
	if err := adapter.Drain(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := adapter.Drain(context.Background()); err != nil {
		t.Fatal(err)
	}
	if releases != 1 || calls != 1 {
		t.Fatal("duplicate completion")
	}
}

func TestAdapterBindingFailureSettlesStep(t *testing.T) {
	scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 4, MaxBudgetTokens: 64, ChunkSize: 16}, cudabatch.NewMemoryBudgetTracker(cudakv.NewBlockAllocator(16, 20)))
	if err := scheduler.Submit(&cudabatch.Request{ID: "invalid", PromptTokens: []int32{1, 2}, MaxTokens: 1}); err != nil {
		t.Fatal(err)
	}
	releases := 0
	scheduler.RegisterResourceRelease("invalid", func() { releases++ })
	adapter := cudaserver.NewBatchEngineAdapter(scheduler, cudabatch.NewHostDeterministicEngine(0, nil), &simpleWordTokenizer{}, cudaserver.WithShapeBucket(cudabatch.ShapeBucket{BatchSize: 1, MaxSeqLen: 1, MaxTokens: 1}))
	if err := adapter.StepIteration(context.Background()); err == nil {
		t.Fatal("invalid binding admitted")
	}
	request, _ := scheduler.Request("invalid")
	if !request.IsFinished() || releases != 1 {
		t.Fatalf("retained failed binding: %v/%d", request.IsFinished(), releases)
	}
}

func ExampleBatchEngineAdapter_Drain() {
	adapter := cudaserver.NewBatchEngineAdapter(nil, nil, nil)
	fmt.Println(adapter.Drain(context.Background()))
	// Output: <nil>
}
