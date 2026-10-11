package cudabatch

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

type controlledPlanCompletion struct {
	mu    sync.Mutex
	done  bool
	err   error
	waits int
}

func (c *controlledPlanCompletion) Wait(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.waits++
	if err := ctx.Err(); err != nil {
		return err
	}
	return c.err
}
func (c *controlledPlanCompletion) Done() bool { c.mu.Lock(); defer c.mu.Unlock(); return c.done }
func (c *controlledPlanCompletion) set(done bool, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.done, c.err = done, err
}
func planTestStep(t *testing.T, id int64) *ExecutionStep {
	t.Helper()
	m, err := ValidateAndBindRows([]BatchRowBinding{{RowIndex: 0, SeqID: "a", StartPos: 0, SequenceLength: 1, BlockTable: []int{0}}}, ShapeBucket{1, 1, 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	return &ExecutionStep{StepID: id, Metadata: m}
}
func planTestEngine(c planCompletion, submitErr error, events *[]string) *PlanEngine {
	return &PlanEngine{
		bind: func(context.Context, *ExecutionStep) (*cudaplan.Bindings, func(), error) {
			*events = append(*events, "bind")
			return cudaplan.NewBindings(), func() { *events = append(*events, "release") }, nil
		},
		invoke: func(*cudaplan.Bindings) (planCompletion, error) {
			*events = append(*events, "submit")
			return c, submitErr
		},
		sample: func(context.Context, *ExecutionStep) (map[string]int32, map[string]error, error) {
			*events = append(*events, "sample")
			return map[string]int32{"a": 42}, nil, nil
		},
	}
}
func TestPlanEngineForward(t *testing.T) {
	var events []string
	c := &controlledPlanCompletion{done: true}
	e := planTestEngine(c, nil, &events)
	step := planTestStep(t, 1)
	step.ReleaseHook = func() { events = append(events, "hook") }
	tokens, _, err := e.ExecuteForward(context.Background(), step)
	if err != nil || tokens["a"] != 42 {
		t.Fatalf("tokens=%v err=%v", tokens, err)
	}
	if want := []string{"bind", "submit", "sample", "release", "hook"}; !reflect.DeepEqual(events, want) {
		t.Fatalf("events=%v", events)
	}
	if !e.StepDone(1) || e.StepDone(2) {
		t.Fatal("incorrect step witness")
	}
	if err := e.WaitStep(context.Background(), 2); err == nil {
		t.Fatal("unknown step accepted")
	}
	if err := e.WaitStep(context.Background(), 1); err != nil {
		t.Fatal(err)
	}
	if len(events) != 5 {
		t.Fatal("release repeated")
	}
	if _, _, err := e.ExecuteForward(context.Background(), step); err == nil {
		t.Fatal("duplicate accepted")
	}
}
func TestPlanEnginePendingRecovery(t *testing.T) {
	for _, cleanupDone := range []bool{false, true} {
		t.Run(map[bool]string{false: "inflight", true: "cleanup"}[cleanupDone], func(t *testing.T) {
			failure := errors.New("native wait failed")
			var events []string
			c := &controlledPlanCompletion{done: cleanupDone, err: failure}
			e := planTestEngine(c, nil, &events)
			_, _, err := e.ExecuteForward(context.Background(), planTestStep(t, 1))
			var pending *PendingForwardError
			if !errors.As(err, &pending) || !errors.Is(err, failure) {
				t.Fatalf("err=%v", err)
			}
			if pending.Done() != cleanupDone {
				t.Fatal("incorrect Done")
			}
			if _, _, err := e.ExecuteForward(context.Background(), planTestStep(t, 2)); err == nil {
				t.Fatal("pending execution accepted")
			}
			if err := e.Close(); err == nil {
				t.Fatal("pending close accepted")
			}
			canceled, cancel := context.WithCancel(context.Background())
			cancel()
			if err := pending.Wait(canceled); !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled recovery=%v", err)
			}
			c.set(true, nil)
			if err := pending.Wait(context.Background()); err != nil {
				t.Fatalf("lost terminal error: %v", err)
			}
			if !pending.Done() || !e.StepDone(1) {
				t.Fatal("recovered ownership not observed")
			}
			if err := e.WaitStep(context.Background(), 1); !errors.Is(err, failure) {
				t.Fatal("terminal error hidden")
			}
			releases := 0
			for _, s := range events {
				if s == "release" {
					releases++
				}
				if s == "sample" {
					t.Fatal("sampled failed completion")
				}
			}
			if releases != 1 {
				t.Fatalf("release count=%d", releases)
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
func TestPlanEnginePartialSubmission(t *testing.T) {
	failure := errors.New("partial submission")
	var events []string
	c := &controlledPlanCompletion{done: false, err: failure}
	e := planTestEngine(c, failure, &events)
	_, _, err := e.ExecuteForward(context.Background(), planTestStep(t, 1))
	var pending *PendingForwardError
	if !errors.As(err, &pending) || pending.Done() {
		t.Fatalf("err=%v", err)
	}
	if len(events) != 2 {
		t.Fatalf("premature release: %v", events)
	}
	c.set(true, nil)
	if err := e.Drain(context.Background()); err != nil {
		t.Fatal("lost submission error")
	}
	if !e.StepDone(1) {
		t.Fatal("not done")
	}
}
func TestPlanEngineFalseCompletion(t *testing.T) {
	var events []string
	c := &controlledPlanCompletion{}
	e := planTestEngine(c, nil, &events)
	_, _, err := e.ExecuteForward(context.Background(), planTestStep(t, 1))
	var p *PendingForwardError
	if !errors.As(err, &p) || p.Done() {
		t.Fatalf("err=%v", err)
	}
	if len(events) != 2 {
		t.Fatalf("released without completion: %v", events)
	}
}
func TestPlanEngineValidation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*ExecutionStep)
	}{
		{"row identity", func(s *ExecutionStep) { s.Metadata.RowBindings[0].RowIndex = 1 }},
		{"extent", func(s *ExecutionStep) { s.Metadata.RowBindings[0].StartPos = 1 }},
		{"mask", func(s *ExecutionStep) { s.Metadata.AttentionMask[0] = 0 }},
		{"block table", func(s *ExecutionStep) { s.Metadata.BlockTableFlat[0] = 99 }},
		{"total tokens", func(s *ExecutionStep) { s.Metadata.TotalTokens = 2 }},
		{"budget", func(s *ExecutionStep) { s.Metadata.MaxTokens = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var events []string
			e := planTestEngine(&controlledPlanCompletion{done: true}, nil, &events)
			s := planTestStep(t, 1)
			tc.change(s)
			if _, _, err := e.ExecuteForward(context.Background(), s); err == nil {
				t.Fatal("invalid metadata admitted")
			}
			if len(events) != 0 {
				t.Fatalf("validation submitted: %v", events)
			}
			if e.StepDone(1) {
				t.Fatal("unattempted witness")
			}
		})
	}
}
func TestPlanEngineBindingAndSamplingFailures(t *testing.T) {
	failure := errors.New("binding failed")
	var events []string
	e := planTestEngine(&controlledPlanCompletion{done: true}, nil, &events)
	e.bind = func(context.Context, *ExecutionStep) (*cudaplan.Bindings, func(), error) {
		return nil, func() { events = append(events, "release") }, failure
	}
	if _, _, err := e.ExecuteForward(context.Background(), planTestStep(t, 1)); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	if !e.StepDone(1) || !reflect.DeepEqual(events, []string{"release"}) {
		t.Fatalf("events=%v", events)
	}
	for _, tokens := range []map[string]int32{nil, {"unknown": 1}} {
		var events []string
		e := planTestEngine(&controlledPlanCompletion{done: true}, nil, &events)
		e.sample = func(context.Context, *ExecutionStep) (map[string]int32, map[string]error, error) {
			return tokens, nil, nil
		}
		if _, _, err := e.ExecuteForward(context.Background(), planTestStep(t, 1)); err == nil {
			t.Fatal("invalid outputs accepted")
		}
		if !e.StepDone(1) {
			t.Fatal("completed output error retains storage")
		}
	}
}
func TestPlanEngineOwnsMetadata(t *testing.T) {
	var events []string
	e := planTestEngine(&controlledPlanCompletion{done: true}, nil, &events)
	step := planTestStep(t, 1)
	step.Iteration = &ScheduledIteration{StepID: 1, AllocatedRows: step.Metadata.RowBindings, TotalTokens: 1}
	e.bind = func(_ context.Context, s *ExecutionStep) (*cudaplan.Bindings, func(), error) {
		s.StepID = 99
		s.Metadata.RowBindings[0].BlockTable[0] = 7
		s.Iteration.AllocatedRows[0].BlockTable[0] = 8
		if step.Metadata.RowBindings[0].BlockTable[0] != 0 {
			t.Fatal("binder mutated caller")
		}
		return cudaplan.NewBindings(), nil, nil
	}
	if _, _, err := e.ExecuteForward(context.Background(), step); err != nil {
		t.Fatal(err)
	}
	if !e.StepDone(1) || e.StepDone(99) {
		t.Fatal("binder changed completion identity")
	}
}
func TestCompilePlanEngineHostRefusal(t *testing.T) {
	bind := func(context.Context, *ExecutionStep) (*cudaplan.Bindings, func(), error) {
		return cudaplan.NewBindings(), nil, nil
	}
	sample := func(context.Context, *ExecutionStep) (map[string]int32, map[string]error, error) {
		return map[string]int32{"a": 1}, nil, nil
	}
	if _, err := CompilePlanEngine(nil, nil, bind, sample); err == nil {
		t.Fatal("nil source accepted")
	}
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1, 1}, cudair.Float32)
	y := g.AddMatMul(x, x)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	e, err := CompilePlanEngine(snapshot, nil, bind, sample)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if _, _, err := e.ExecuteForward(context.Background(), planTestStep(t, 1)); err == nil {
		t.Fatal("host compilation pretended native execution")
	}
	if !e.StepDone(1) {
		t.Fatal("refused submission not settled")
	}
}

func TestPlanEngineSchedulerChunks(t *testing.T) {
	allocator := cudakv.NewBlockAllocator(2, 16)
	scheduler := NewBatchScheduler(SchedulerConfig{MaxBatchSize: 1, MaxBudgetTokens: 2, ChunkSize: 2}, NewMemoryBudgetTracker(allocator))
	if err := scheduler.Submit(&Request{ID: "a", PromptTokens: []int32{10, 11, 12, 13, 14}, MaxTokens: 1}); err != nil {
		t.Fatal(err)
	}
	var events []string
	engine := planTestEngine(&controlledPlanCompletion{done: true}, nil, &events)
	var extents [][2]int
	engine.bind = func(_ context.Context, step *ExecutionStep) (*cudaplan.Bindings, func(), error) {
		row := step.Metadata.RowBindings[0]
		extents = append(extents, [2]int{row.StartPos, row.SequenceLength})
		if row.IsPrefill && !reflect.DeepEqual(row.Tokens, []int32{10, 11, 12, 13, 14}[row.StartPos:row.SequenceLength]) {
			t.Fatal("incorrect chunk input")
		}
		return cudaplan.NewBindings(), func() { events = append(events, "release") }, nil
	}
	engine.sample = func(_ context.Context, step *ExecutionStep) (map[string]int32, map[string]error, error) {
		if step.Metadata.RowBindings[0].IsPrefill {
			return nil, nil, nil
		}
		return map[string]int32{"a": 42}, nil, nil
	}
	for range 4 {
		iteration, err := scheduler.Schedule()
		if err != nil {
			t.Fatal(err)
		}
		metadata, err := ValidateAndBindRows(iteration.AllocatedRows, ShapeBucket{1, 8, 2}, 2)
		if err != nil {
			t.Fatal(err)
		}
		step := &ExecutionStep{StepID: iteration.StepID, Iteration: iteration, Metadata: metadata}
		tokens, rowErrors, err := engine.ExecuteForward(context.Background(), step)
		if err != nil {
			t.Fatal(err)
		}
		if !engine.StepDone(step.StepID) {
			t.Fatal("scheduler settlement preceded completion")
		}
		if err := scheduler.CompleteStep(iteration, tokens, rowErrors); err != nil {
			t.Fatal(err)
		}
	}
	if want := [][2]int{{0, 2}, {2, 4}, {4, 5}, {5, 6}}; !reflect.DeepEqual(extents, want) {
		t.Fatalf("extents=%v", extents)
	}
	request, _ := scheduler.Request("a")
	if request.State != StateCompleted || !reflect.DeepEqual(request.GeneratedTokens, []int32{42}) {
		t.Fatalf("request=%+v", request)
	}
	if allocator.NumFreeBlocks() != 16 {
		t.Fatal("scheduler KV references retained after completed request")
	}
}
func TestPlanEngineCancelWait(t *testing.T) {
	var events []string
	canceled := make(chan struct{})
	close(canceled)
	engine := planTestEngine(&controlledPlanCompletion{done: true}, nil, &events)
	engine.bind = func(ctx context.Context, _ *ExecutionStep) (*cudaplan.Bindings, func(), error) {
		<-ctx.Done()
		return cudaplan.NewBindings(), func() { events = append(events, "release") }, nil
	}
	step := planTestStep(t, 1)
	step.CancelWait = canceled
	if _, _, err := engine.ExecuteForward(context.Background(), step); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"release"}) || !engine.StepDone(1) {
		t.Fatalf("canceled bind dispatched: %v", events)
	}
}

func TestPlanEngineReuseCallerStep(t *testing.T) {
	var events []string
	engine := planTestEngine(&controlledPlanCompletion{done: true}, nil, &events)
	step := planTestStep(t, 1)
	for i := int64(1); i <= 100; i++ {
		step.StepID = i
		step.CancelWait = make(chan struct{})
		if _, _, err := engine.ExecuteForward(context.Background(), step); err != nil {
			t.Fatal(err)
		}
		// The bounded cancellation watcher may still be exiting. Reusing caller
		// storage must not change the watcher's captured channel or completion ID.
		step.CancelWait = nil
		step.StepID = -1
		if !engine.StepDone(i) {
			t.Fatal("caller changed completion identity")
		}
	}
}
