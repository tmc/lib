package cudaserver_test

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaserver"
)

type emptyTokenizer struct{}

func (emptyTokenizer) Encode(string) ([]int32, error) { return nil, nil }
func (emptyTokenizer) Decode([]int32) (string, error) { return "", nil }

func TestAdapterEmptyTokenizationRefused(t *testing.T) {
	adapter := cudaserver.NewBatchEngineAdapter(cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{}, nil), cudabatch.NewHostDeterministicEngine(0, nil), emptyTokenizer{})
	_, err := adapter.GenerateWithUsage(context.Background(), cudaserver.ChatCompletionRequest{}, make(chan string))
	if err == nil || !strings.Contains(err.Error(), "empty prompt") {
		t.Fatalf("empty tokenization error=%v", err)
	}
}

func TestAdapterTokenizerUsage(t *testing.T) {
	scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 4, MaxBudgetTokens: 64, ChunkSize: 16}, cudabatch.NewMemoryBudgetTracker(cudakv.NewBlockAllocator(16, 20)))
	adapter := cudaserver.NewBatchEngineAdapter(scheduler, cudabatch.NewHostDeterministicEngine(0, []int32{101, 102}), &simpleWordTokenizer{})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tokens := make(chan string, 16)
	done := make(chan struct {
		usage cudaserver.UsageStats
		err   error
	}, 1)
	go func() {
		usage, err := adapter.GenerateWithUsage(ctx, cudaserver.ChatCompletionRequest{Messages: []cudaserver.ChatMessage{{Role: "user", Content: "hello world"}}, MaxTokens: 2}, tokens)
		done <- struct {
			usage cudaserver.UsageStats
			err   error
		}{usage, err}
	}()
	for {
		select {
		case result := <-done:
			if result.err != nil {
				t.Fatal(result.err)
			}
			want := cudaserver.UsageStats{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6}
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
	adapter := cudaserver.NewBatchEngineAdapter(scheduler, cudabatch.NewHostDeterministicEngine(0, []int32{101, 102}), &simpleWordTokenizer{})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- adapter.Generate(ctx, cudaserver.ChatCompletionRequest{Messages: []cudaserver.ChatMessage{{Role: "user", Content: "hello"}}, MaxTokens: 16}, make(chan string))
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

type countedEngine struct {
	engine cudabatch.ExecutionEngine
	count  int
}

func (e *countedEngine) ExecuteForward(ctx context.Context, step *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
	tokens, failures, err := e.engine.ExecuteForward(ctx, step)
	e.count += len(tokens)
	return tokens, failures, err
}

func TestAdapterDrainsQueuedCompletion(t *testing.T) {
	for range 20 {
		scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 4, MaxBudgetTokens: 64, ChunkSize: 16}, cudabatch.NewMemoryBudgetTracker(cudakv.NewBlockAllocator(16, 20)))
		engine := &countedEngine{engine: cudabatch.NewHostDeterministicEngine(0, []int32{101, 102, 103, 104, 105, 106, 107, 108})}
		adapter := cudaserver.NewBatchEngineAdapter(scheduler, engine, &simpleWordTokenizer{})
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		tokens := make(chan string)
		done := make(chan error, 1)
		go func() {
			err := adapter.Generate(ctx, cudaserver.ChatCompletionRequest{Messages: []cudaserver.ChatMessage{{Role: "user", Content: "hello"}}, MaxTokens: 8}, tokens)
			close(tokens)
			done <- err
		}()
		for engine.count < 8 {
			if ctx.Err() != nil {
				cancel()
				t.Fatal(ctx.Err())
			}
			if err := adapter.StepIteration(ctx); err != nil {
				cancel()
				t.Fatal(err)
			}
			runtime.Gosched()
		}
		// All output and completion are queued before the consumer starts.
		var output strings.Builder
		for token := range tokens {
			output.WriteString(token)
		}
		if err := <-done; err != nil {
			cancel()
			t.Fatal(err)
		}
		cancel()
		const want = "word101word102word103word104word105word106word107word108"
		if output.String() != want {
			t.Fatalf("queued output=%q want=%q", output.String(), want)
		}
	}
}

func TestAdapterQueueOverflowRefusesSuccess(t *testing.T) {
	scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 4, MaxBudgetTokens: 64, ChunkSize: 16}, cudabatch.NewMemoryBudgetTracker(cudakv.NewBlockAllocator(16, 20)))
	engine := &countedEngine{engine: cudabatch.NewHostDeterministicEngine(0, []int32{101})}
	adapter := cudaserver.NewBatchEngineAdapter(scheduler, engine, &simpleWordTokenizer{})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tokens := make(chan string)
	done := make(chan error, 1)
	go func() {
		err := adapter.Generate(ctx, cudaserver.ChatCompletionRequest{Messages: []cudaserver.ChatMessage{{Role: "user", Content: "hello"}}, MaxTokens: 64}, tokens)
		close(tokens)
		done <- err
	}()
	for engine.count < 33 {
		if ctx.Err() != nil {
			t.Fatal(ctx.Err())
		}
		if err := adapter.StepIteration(ctx); err != nil {
			t.Fatal(err)
		}
		runtime.Gosched()
	}
	for range 4 {
		if err := adapter.StepIteration(ctx); err != nil {
			t.Fatal(err)
		}
	}
	for range tokens {
	}
	if err := <-done; err == nil || !strings.Contains(err.Error(), "consumer too slow") {
		t.Fatalf("overflow produced successful/truncated output: %v", err)
	}
}

type executionFunc func(context.Context, *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error)

func (f executionFunc) ExecuteForward(ctx context.Context, step *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
	return f(ctx, step)
}

type failedDecoder struct{ simpleWordTokenizer }

func (failedDecoder) Decode([]int32) (string, error) { return "", errors.New("decoder failed") }

func TestAdapterGenerationErrors(t *testing.T) {
	for _, tt := range []struct {
		name      string
		engine    cudabatch.ExecutionEngine
		tokenizer cudaserver.Tokenizer
		want      string
	}{
		{"global", executionFunc(func(context.Context, *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
			return nil, nil, errors.New("engine failed")
		}), &simpleWordTokenizer{}, "engine failed"},
		{"row", executionFunc(func(_ context.Context, s *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
			id := s.Iteration.AllocatedRows[0].SeqID
			return map[string]int32{id: 999}, map[string]error{id: errors.New("row failed")}, nil
		}), &simpleWordTokenizer{}, "row failed"},
		{"decoder", cudabatch.NewHostDeterministicEngine(0, []int32{101}), &failedDecoder{}, "decoder failed"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 4, MaxBudgetTokens: 64, ChunkSize: 16}, cudabatch.NewMemoryBudgetTracker(cudakv.NewBlockAllocator(16, 20)))
			adapter := cudaserver.NewBatchEngineAdapter(scheduler, tt.engine, tt.tokenizer)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			tokens := make(chan string, 16)
			done := make(chan error, 1)
			go func() {
				done <- adapter.Generate(ctx, cudaserver.ChatCompletionRequest{Messages: []cudaserver.ChatMessage{{Role: "user", Content: "hello"}}, MaxTokens: 1}, tokens)
			}()
			for {
				select {
				case err := <-done:
					if err == nil || !strings.Contains(err.Error(), tt.want) {
						t.Fatalf("generation did not propagate %q: %v", tt.want, err)
					}
					if len(tokens) != 0 {
						t.Fatal("failed row/decoder emitted invented output")
					}
					return
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				default:
					if err := adapter.StepIteration(ctx); err != nil && !strings.Contains(err.Error(), tt.want) {
						t.Fatal(err)
					}
					runtime.Gosched()
				}
			}
		})
	}
}
