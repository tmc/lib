package cudaserver

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

type retirementTokenizer struct{}

func (retirementTokenizer) Encode(string) ([]int32, error)        { return []int32{1}, nil }
func (retirementTokenizer) Decode(tokens []int32) (string, error) { return fmt.Sprint(tokens[0]), nil }

func waitAdapterRequests(t *testing.T, a *BatchEngineAdapter, n int) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		a.mu.Lock()
		ready := len(a.completions) == n
		for id := range a.completions {
			if _, ok := a.scheduler.Request(id); !ok {
				ready = false
			}
		}
		a.mu.Unlock()
		if ready {
			return
		}
		select {
		case <-deadline:
			t.Fatal("requests not submitted")
		case <-time.After(time.Millisecond):
		}
	}
}

func TestAdapterTerminalHistoryRetirement(t *testing.T) {
	s := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 2, MaxFinishedRequests: 1}, cudabatch.NewMemoryBudgetTracker(cudakv.NewBlockAllocator(16, 20)))
	a := NewBatchEngineAdapter(s, cudabatch.NewHostDeterministicEngine(0, []int32{9}), retirementTokenizer{})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	type result struct {
		u UsageStats
		e error
	}
	done := make(chan result, 2)
	outputs := []chan string{make(chan string, 2), make(chan string, 2)}
	for _, out := range outputs {
		go func(out chan string) {
			u, e := a.GenerateWithUsage(ctx, ChatCompletionRequest{MaxTokens: 1}, out)
			done <- result{u, e}
		}(out)
	}
	waitAdapterRequests(t, a, 2)
	if err := a.StepIteration(ctx); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		r := <-done
		if r.e != nil || r.u.CompletionTokens != 1 {
			t.Fatalf("retirement lost terminal result: %+v", r)
		}
	}
	for _, out := range outputs {
		select {
		case token := <-out:
			if token != "9" {
				t.Fatal(token)
			}
		default:
			t.Fatal("completion preceded output")
		}
	}
}

func TestAdapterWaitingExpiryDuringOutstandingStep(t *testing.T) {
	s := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 1, MaxWaitingTimeout: 20 * time.Millisecond}, nil)
	if err := s.Submit(&cudabatch.Request{ID: "active", PromptTokens: []int32{1}, MaxTokens: 1}); err != nil {
		t.Fatal(err)
	}
	step, err := s.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	a := NewBatchEngineAdapter(s, cudabatch.NewHostDeterministicEngine(0, nil), retirementTokenizer{})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- a.Generate(ctx, ChatCompletionRequest{MaxTokens: 1}, make(chan string, 1)) }()
	waitAdapterRequests(t, a, 1)
	timer := time.NewTimer(25 * time.Millisecond)
	defer timer.Stop()
	<-timer.C
	if err := a.StepIteration(ctx); !errors.Is(err, cudabatch.ErrExecutionInFlight) {
		t.Fatal(err)
	}
	if err := <-done; !errors.Is(err, cudabatch.ErrWaitingTimeout) {
		t.Fatalf("lost expiry: %v", err)
	}
	if err := s.CompleteStep(step, map[string]int32{"active": 7}, nil); err != nil {
		t.Fatal(err)
	}
}
