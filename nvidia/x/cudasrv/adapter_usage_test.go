package cudasrv_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudasrv"
)

type emptyTokenizer struct{}

func (emptyTokenizer) Encode(string) ([]int32, error) { return nil, nil }
func (emptyTokenizer) Decode([]int32) (string, error) { return "", nil }

func TestAdapterEmptyTokenizationRefused(t *testing.T) {
	adapter := cudasrv.NewBatchEngineAdapter(cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{}, nil), cudabatch.NewHostDeterministicEngine(0, nil), emptyTokenizer{})
	_, err := adapter.GenerateWithUsage(context.Background(), cudasrv.ChatCompletionRequest{}, make(chan string))
	if err == nil || !strings.Contains(err.Error(), "empty prompt") {
		t.Fatalf("empty tokenization error=%v", err)
	}
}

func TestAdapterTokenizerUsage(t *testing.T) {
	scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 4, MaxBudgetTokens: 64, ChunkSize: 16}, cudabatch.NewMemoryBudgetTracker(cudakv.NewBlockAllocator(16, 20)))
	adapter := cudasrv.NewBatchEngineAdapter(scheduler, cudabatch.NewHostDeterministicEngine(0, []int32{101, 102}), &simpleWordTokenizer{})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tokens := make(chan string, 16)
	done := make(chan struct {
		usage cudasrv.UsageStats
		err   error
	}, 1)
	go func() {
		usage, err := adapter.GenerateWithUsage(ctx, cudasrv.ChatCompletionRequest{Messages: []cudasrv.ChatMessage{{Role: "user", Content: "hello world"}}, MaxTokens: 2}, tokens)
		done <- struct {
			usage cudasrv.UsageStats
			err   error
		}{usage, err}
	}()
	for {
		select {
		case result := <-done:
			if result.err != nil {
				t.Fatal(result.err)
			}
			want := cudasrv.UsageStats{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6}
			if result.usage != want {
				t.Fatalf("usage=%+v want=%+v", result.usage, want)
			}
			return
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		default:
			if err := adapter.StepIteration(ctx); err != nil {
				t.Fatal(err)
			}
			time.Sleep(time.Millisecond)
		}
	}
}

func TestAdapterCanceledBlockedOutput(t *testing.T) {
	scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 4, MaxBudgetTokens: 64, ChunkSize: 16}, cudabatch.NewMemoryBudgetTracker(cudakv.NewBlockAllocator(16, 20)))
	adapter := cudasrv.NewBatchEngineAdapter(scheduler, cudabatch.NewHostDeterministicEngine(0, []int32{101, 102}), &simpleWordTokenizer{})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- adapter.Generate(ctx, cudasrv.ChatCompletionRequest{Messages: []cudasrv.ChatMessage{{Role: "user", Content: "hello"}}, MaxTokens: 16}, make(chan string))
	}()
	for range 10 {
		if err := adapter.StepIteration(ctx); err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Fatalf("cancellation error=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("blocked token forwarding ignored cancellation")
	}
}
