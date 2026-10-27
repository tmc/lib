package cudadiagdemo

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// fakeNative is a host device for nativeExecutor. Each opened stream has its
// own buffers unless shared is set. The hooks run on the request worker.
type fakeNative struct {
	mu          sync.Mutex
	shared      []float32
	opened      int
	launches    int
	inflight    int
	maxInflight int
	closes      map[string]int

	launch      func(index int) error
	synchronize func(ctx context.Context, index int) error
	close       func(name string, index int) error
}

type fakeNativeStream struct {
	f       *fakeNative
	index   int
	in, out []float32
}

func (f *fakeNative) open(own *nativeCleanup) (nativeStream, error) {
	f.mu.Lock()
	f.opened++
	f.inflight++
	f.maxInflight = max(f.maxInflight, f.inflight)
	f.mu.Unlock()
	s := &fakeNativeStream{f: f, index: -1}
	for _, name := range []string{"input", "output", "stream"} {
		own.add(name, func() error {
			if f.close != nil {
				if err := f.close(name, s.index); err != nil {
					return err
				}
			}
			f.mu.Lock()
			defer f.mu.Unlock()
			if f.closes == nil {
				f.closes = make(map[string]int)
			}
			f.closes[fmt.Sprintf("%s/%d", name, s.index)]++
			if name == "stream" {
				f.inflight--
			}
			return nil
		})
	}
	return s, nil
}

// upload identifies the request by its first input element, the plan index.
func (s *fakeNativeStream) upload(input []float32) error {
	s.in = append([]float32(nil), input...)
	s.out = make([]float32, len(input))
	s.index = int(input[0])
	return nil
}

func (s *fakeNativeStream) launch() error {
	f := s.f
	if f.launch != nil {
		if err := f.launch(s.index); err != nil {
			return err
		}
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.launches++
	out := s.out
	if f.shared != nil {
		out = f.shared
	}
	for i, v := range s.in {
		out[i] = 2*v + 1
	}
	return nil
}

func (s *fakeNativeStream) synchronize(ctx context.Context) error {
	if s.f.synchronize != nil {
		return s.f.synchronize(ctx, s.index)
	}
	return nil
}

func (s *fakeNativeStream) download(output []float32) error {
	s.f.mu.Lock()
	defer s.f.mu.Unlock()
	if s.f.shared != nil {
		copy(output, s.f.shared)
	} else {
		copy(output, s.out)
	}
	return nil
}

func (f *fakeNative) count(key string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.closes[key]
}

func (f *fakeNative) executor() *nativeExecutor {
	return &nativeExecutor{open: f.open, elements: 8, epoch: time.Now(), evidence: digestBytes([]byte(nativeSource))}
}

// fakeCorrelation is one worker's external correlation stack. Each worker
// has its own annotator, so no stack is shared between threads.
type fakeCorrelation struct{ stack []uint64 }

func (h *fakeCorrelation) Push(_ cupti.ExternalCorrelationKind, id uint64) error {
	h.stack = append(h.stack, id)
	return nil
}

func (h *fakeCorrelation) Pop(cupti.ExternalCorrelationKind) (uint64, error) {
	if len(h.stack) == 0 {
		return 0, errors.New("empty correlation stack")
	}
	id := h.stack[len(h.stack)-1]
	h.stack = h.stack[:len(h.stack)-1]
	return id, nil
}

// barrier releases its first n arrivals together and later arrivals at once.
type barrier struct {
	mu    sync.Mutex
	n     int
	count int
	done  chan struct{}
}

func newBarrier(n int) *barrier { return &barrier{n: n, done: make(chan struct{})} }

func (b *barrier) wait(ctx context.Context) error {
	b.mu.Lock()
	b.count++
	if b.count == b.n {
		close(b.done)
	}
	b.mu.Unlock()
	select {
	case <-b.done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("barrier of %d: %w", b.n, ctx.Err())
	}
}

func testContext(t *testing.T) context.Context {
	// The deadline only stops a broken scheduler from hanging the test;
	// overlap is proven by the barrier, never by elapsed time.
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	t.Cleanup(cancel)
	return ctx
}

func TestNativeWorkersBounds(t *testing.T) {
	tests := []struct {
		name        string
		concurrency int
		elements    int
		rounds      int
		want        int
	}{
		{"default", 0, 1 << 20, 64, 4},
		{"default clamped to plan", 0, 8, 1, 3},
		{"serial", 1, 1 << 20, 64, 1},
		{"max", 8, 1 << 19, 64, 8},
		{"max clamped to plan", 8, 1 << 20, 1, 3},
		{"live bytes at bound", 4, 1 << 20, 2, 4},
		{"negative", -1, 8, 1, 0},
		{"over max", 9, 8, 64, 0},
		{"live bytes over bound", 5, 1 << 20, 2, 0},
		{"max live bytes over bound", 8, 1 << 20, 64, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := nativeWorkers(NativeConfig{Concurrency: tt.concurrency, Elements: tt.elements, Rounds: tt.rounds})
			if got != tt.want || (err == nil) != (tt.want > 0) {
				t.Fatalf("nativeWorkers = %d, %v; want %d", got, err, tt.want)
			}
		})
	}
}

func TestNativeRequestPlan(t *testing.T) {
	type entry struct {
		round    int
		arm      string
		launches int
	}
	var got []entry
	for i, r := range planNativeRequests(2, 16) {
		if r.index != i {
			t.Fatalf("request %d has index %d", i, r.index)
		}
		got = append(got, entry{r.round, r.arm, r.launches})
	}
	want := []entry{{1, "baseline", 1}, {1, "candidate", 1}, {1, "device", 16}, {2, "candidate", 1}, {2, "baseline", 1}, {2, "device", 16}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("plan = %v, want %v", got, want)
	}
}

func TestNativeWorkersOverlap(t *testing.T) {
	for _, workers := range []int{1, 2, 3} {
		t.Run(fmt.Sprint("workers=", workers), func(t *testing.T) {
			ctx := testContext(t)
			// The first workers attempts can only pass the barrier together.
			b := newBarrier(workers)
			f := &fakeNative{synchronize: func(context.Context, int) error { return b.wait(ctx) }}
			plan := planNativeRequests(2, 4)
			retain := &nativeCleanup{}
			attempts, err := f.executor().run(ctx, workers, plan, retain)
			if err != nil {
				t.Fatal(err)
			}
			if f.maxInflight != workers {
				t.Fatalf("max in-flight attempts = %d, want %d", f.maxInflight, workers)
			}
			if len(attempts) != len(plan) || f.opened != len(plan) || len(retain.actions) != 0 {
				t.Fatalf("attempts=%d opened=%d retained=%d, want %d attempts and no retained owners", len(attempts), f.opened, len(retain.actions), len(plan))
			}
			for i, a := range attempts {
				r := plan[i]
				if a.trial.Round != r.round || a.trial.Arm != r.arm || a.trial.RequestedLaunches != r.launches || a.trial.KernelLaunches != r.launches || len(a.trial.Calls) != r.launches+3 || !a.trial.Correct || !a.log.Complete || a.trial.Correlation != "uninstrumented" {
					t.Fatalf("attempt %d = %+v, want plan %+v", i, a.trial, r)
				}
				if a.trial.Worker < 0 || a.trial.Worker >= workers {
					t.Fatalf("attempt %d ran on worker %d of %d", i, a.trial.Worker, workers)
				}
				for _, name := range []string{"input", "output", "stream"} {
					if n := f.count(fmt.Sprintf("%s/%d", name, i)); n != 1 {
						t.Fatalf("%s of attempt %d closed %d times", name, i, n)
					}
				}
				if workers == 1 && i > 0 && a.log.StartedNS < attempts[i-1].log.FinishedNS {
					t.Fatalf("serial attempt %d started at %d before attempt %d finished at %d", i, a.log.StartedNS, i-1, attempts[i-1].log.FinishedNS)
				}
			}
		})
	}
}

func TestNativeWorkersSameNameDistinctIdentity(t *testing.T) {
	ctx := testContext(t)
	const workers = 3
	b := newBarrier(workers)
	f := &fakeNative{synchronize: func(context.Context, int) error { return b.wait(ctx) }}
	e := f.executor()
	process := cudaworkload.ProcessGenID{PID: uint32(os.Getpid()), Generation: 1}
	for range workers {
		e.annotators = append(e.annotators, cudaworkload.NewAnnotator(process, "gpudiagdemo-native", cudaworkload.WithRunID("native-test"), cudaworkload.WithCorrelationHook(&fakeCorrelation{})))
	}
	plan := planNativeRequests(2, 2)
	attempts, err := e.run(ctx, workers, plan, &nativeCleanup{})
	if err != nil {
		t.Fatal(err)
	}
	type key struct{ request, attempt string }
	keys := map[key]bool{}
	ops := map[cudaworkload.OpID]bool{}
	outputs := map[string]bool{}
	result := &NativeResult{}
	for i, a := range attempts {
		g := a.log.GroundTruth
		if a.annotation == nil || a.trial.Correlation != "complete" || g.WorkloadName != "gpudiagdemo-native" || g.Workload.WorkloadName != "gpudiagdemo-native" || g.ProcessGen != process || g.Workload.Process != process {
			t.Fatalf("attempt %d identity = %+v", i, g)
		}
		if g.OpID == 0 || g.OpID != a.annotation.OpID || g.TID != a.annotation.TID || ops[g.OpID] {
			t.Fatalf("attempt %d operation %d (annotation %+v) is missing or shared", i, g.OpID, a.annotation)
		}
		ops[g.OpID] = true
		k := key{a.trial.RequestID, a.trial.Attempt}
		if k.request != g.RequestID || k.attempt != a.log.Attempt || keys[k] {
			t.Fatalf("attempt %d request identity %v is inconsistent or shared", i, k)
		}
		keys[k] = true
		want := nativeInput(e.elements, i)
		for j := range want {
			want[j] = 2*want[j] + 1
		}
		if a.trial.InputSHA256 != digestFloats(nativeInput(e.elements, i)) || a.trial.OutputSHA256 != digestFloats(want) || outputs[a.trial.OutputSHA256] {
			t.Fatalf("attempt %d output %s is not its own", i, a.trial.OutputSHA256)
		}
		outputs[a.trial.OutputSHA256] = true
		result.Requests = append(result.Requests, a.log)
		result.Trials = append(result.Trials, a.trial)
	}
	composition, err := ComposeNativeEvidence(ctx, result, NativeCompositionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(composition.Requests) != len(plan) {
		t.Fatalf("composition retained %d of %d requests", len(composition.Requests), len(plan))
	}
	for _, row := range composition.Requests {
		if row.Diagnosis.Admitted || row.Refusal == "" {
			t.Fatalf("native request admitted: %+v", row)
		}
	}
}

func TestNativeWorkersSharedBufferFailsOracle(t *testing.T) {
	ctx := testContext(t)
	// Negative control: every stream writes one buffer, as if two requests
	// shared device memory. All launches finish before any download.
	b := newBarrier(3)
	f := &fakeNative{shared: make([]float32, 8), synchronize: func(context.Context, int) error { return b.wait(ctx) }}
	attempts, err := f.executor().run(ctx, 3, planNativeRequests(1, 2), &nativeCleanup{})
	incorrect := 0
	for _, a := range attempts {
		if a.incorrect {
			incorrect++
			if a.trial.Correct || a.log.Complete || a.trial.Error == "" {
				t.Fatalf("incorrect attempt retained as %+v", a.trial)
			}
		}
	}
	if err == nil || incorrect < 2 {
		t.Fatalf("shared output passed the oracle: %d incorrect, err %v", incorrect, err)
	}
}

// failSecondAttempt returns a fake whose attempt 1 fails at launch once
// attempt 0 has launched and is waiting in synchronize, which returns only
// when the failure cancels it.
func failSecondAttempt(errLaunch error) *fakeNative {
	waiting := make(chan struct{})
	return &fakeNative{
		launch: func(index int) error {
			if index == 1 {
				<-waiting
				return errLaunch
			}
			return nil
		},
		synchronize: func(ctx context.Context, index int) error {
			if index == 0 {
				close(waiting)
			}
			<-ctx.Done()
			return ctx.Err()
		},
	}
}

func TestNativeWorkersCancellation(t *testing.T) {
	errLaunch := errors.New("launch failed")
	t.Run("partial failure", func(t *testing.T) {
		ctx := testContext(t)
		f := failSecondAttempt(errLaunch)
		plan := planNativeRequests(2, 2)
		attempts, err := f.executor().run(ctx, 2, plan, &nativeCleanup{})
		if !errors.Is(err, errLaunch) || !errors.Is(err, context.Canceled) || !errors.Is(err, errNativeRequestFailed) {
			t.Fatalf("error = %v, want launch failure, sibling cancellation and undispatched cause", err)
		}
		for _, want := range []string{"native request request-1/baseline: context canceled", "native request request-1/candidate: launch failed", "4 of 6 not dispatched"} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("error %q lacks %q", err, want)
			}
		}
		if len(attempts) != 2 || attempts[0].trial.Arm != "baseline" || attempts[1].trial.Arm != "candidate" {
			t.Fatalf("retained attempts = %+v", attempts)
		}
		for i, a := range attempts {
			if a.trial.Correct || a.log.Complete || a.trial.Error == "" {
				t.Fatalf("failed attempt %d retained as %+v", i, a.trial)
			}
			if n := f.count(fmt.Sprintf("stream/%d", i)); n != 1 {
				t.Fatalf("stream of attempt %d closed %d times", i, n)
			}
		}
		if f.opened != 2 || f.launches != 1 {
			t.Fatalf("opened=%d launches=%d, want 2 and 1", f.opened, f.launches)
		}
	})
	t.Run("parent canceled before dispatch", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		f := &fakeNative{}
		attempts, err := f.executor().run(ctx, 2, planNativeRequests(1, 2), &nativeCleanup{})
		if len(attempts) != 0 || f.opened != 0 || !errors.Is(err, context.Canceled) || errors.Is(err, errNativeRequestFailed) {
			t.Fatalf("attempts=%d opened=%d err=%v", len(attempts), f.opened, err)
		}
	})
	t.Run("parent canceled in flight", func(t *testing.T) {
		ctx, cancel := context.WithCancel(testContext(t))
		defer cancel()
		b := newBarrier(3)
		f := &fakeNative{synchronize: func(ctx context.Context, _ int) error {
			if err := b.wait(context.Background()); err != nil {
				return err
			}
			<-ctx.Done()
			return ctx.Err()
		}}
		go func() {
			<-b.done
			cancel()
		}()
		attempts, err := f.executor().run(ctx, 3, planNativeRequests(2, 2), &nativeCleanup{})
		if len(attempts) != 3 || !errors.Is(err, context.Canceled) || !strings.Contains(err.Error(), "3 of 6 not dispatched") {
			t.Fatalf("attempts=%d err=%v", len(attempts), err)
		}
		for i, a := range attempts {
			if a.trial.Correct || !strings.Contains(a.trial.Error, "context canceled") || f.count(fmt.Sprintf("stream/%d", i)) != 1 {
				t.Fatalf("canceled attempt %d retained as %+v", i, a.trial)
			}
		}
	})
}

func TestNativeWorkersDelayedCleanup(t *testing.T) {
	ctx := testContext(t)
	errLaunch := errors.New("launch failed")
	entered, release := make(chan struct{}), make(chan struct{})
	var released bool
	f := failSecondAttempt(errLaunch)
	// Attempt 0's stream close blocks after its cancellation.
	f.close = func(name string, index int) error {
		if name == "stream" && index == 0 {
			close(entered)
			<-release
			released = true
		}
		return nil
	}
	go func() {
		<-entered
		close(release)
	}()
	attempts, err := f.executor().run(ctx, 2, planNativeRequests(1, 2), &nativeCleanup{})
	if !released || f.count("stream/0") != 1 || f.count("input/0") != 1 {
		t.Fatalf("run returned before delayed cleanup finished: released=%t closes=%v", released, f.closes)
	}
	if !errors.Is(err, errLaunch) || len(attempts) != 2 || f.inflight != 0 {
		t.Fatalf("attempts=%d inflight=%d err=%v", len(attempts), f.inflight, err)
	}
}

func TestNativeWorkersFailedCleanupRetry(t *testing.T) {
	ctx := testContext(t)
	errClose := errors.New("stream close failed")
	failures := 1
	f := &fakeNative{close: func(name string, index int) error {
		if name == "stream" && index == 1 && failures > 0 {
			failures--
			return errClose
		}
		return nil
	}}
	var order []string
	retain := &nativeCleanup{}
	retain.add("session", func() error { order = append(order, "session"); return nil })
	retain.add("synchronize", func() error { order = append(order, "synchronize"); return nil })
	plan := planNativeRequests(1, 2)
	attempts, err := f.executor().run(ctx, 1, plan, retain)
	if !errors.Is(err, errClose) || !strings.Contains(err.Error(), "native request request-1/candidate: cleanup: native cleanup stream") || !errors.Is(err, errNativeRequestFailed) {
		t.Fatalf("error = %v", err)
	}
	if len(attempts) != 2 || !attempts[1].trial.Correct || attempts[1].trial.Error == "" {
		t.Fatalf("attempts = %+v", attempts)
	}
	want := map[string]int{"input/0": 1, "output/0": 1, "stream/0": 1}
	if !reflect.DeepEqual(f.closes, want) || len(retain.actions) != 5 {
		t.Fatalf("closes=%v retained=%d, want %v and the failed attempt's 3 owners retained", f.closes, len(retain.actions), want)
	}
	result := &NativeResult{cleanup: retain}
	if err := result.Close(); err != nil || result.CleanupPending {
		t.Fatalf("Close = %v pending=%t", err, result.CleanupPending)
	}
	want = map[string]int{"input/0": 1, "output/0": 1, "stream/0": 1, "input/1": 1, "output/1": 1, "stream/1": 1}
	if !reflect.DeepEqual(f.closes, want) || !reflect.DeepEqual(order, []string{"synchronize", "session"}) {
		t.Fatalf("after retry closes=%v order=%v, want %v", f.closes, order, want)
	}
	if f.opened != 2 || f.launches != 2 {
		t.Fatalf("retry resubmitted work: opened=%d launches=%d", f.opened, f.launches)
	}
	if err := result.Close(); err != nil || !reflect.DeepEqual(f.closes, want) || len(order) != 2 {
		t.Fatalf("completed cleanup repeated: %v closes=%v", err, f.closes)
	}
}

// failingCorrelation refuses every push or every pop.
type failingCorrelation struct {
	fakeCorrelation
	push, pop error
}

func (h *failingCorrelation) Push(kind cupti.ExternalCorrelationKind, id uint64) error {
	if h.push != nil {
		return h.push
	}
	return h.fakeCorrelation.Push(kind, id)
}

func (h *failingCorrelation) Pop(kind cupti.ExternalCorrelationKind) (uint64, error) {
	if h.pop != nil {
		return 0, h.pop
	}
	return h.fakeCorrelation.Pop(kind)
}

func TestNativeWorkersCorrelationFailure(t *testing.T) {
	errPush, errPop := errors.New("push refused"), errors.New("pop refused")
	tests := []struct {
		name        string
		hook        *failingCorrelation
		want        error
		correlation string
		launches    int
	}{
		{"pop after launch", &failingCorrelation{pop: errPop}, errPop, "partial", 1},
		{"push before entry", &failingCorrelation{push: errPush}, errPush, "not_entered", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := testContext(t)
			f := &fakeNative{}
			e := f.executor()
			process := cudaworkload.ProcessGenID{PID: uint32(os.Getpid()), Generation: 1}
			e.annotators = []*cudaworkload.Annotator{cudaworkload.NewAnnotator(process, "gpudiagdemo-native", cudaworkload.WithRunID("native-test"), cudaworkload.WithCorrelationHook(tt.hook))}
			attempts, err := e.run(ctx, 1, planNativeRequests(1, 2), &nativeCleanup{})
			if !errors.Is(err, tt.want) || !errors.Is(err, errNativeRequestFailed) || !strings.Contains(err.Error(), "native request request-1/baseline: ") || !strings.Contains(err.Error(), "2 of 3 not dispatched") {
				t.Fatalf("error = %v", err)
			}
			if len(attempts) != 1 {
				t.Fatalf("retained %d attempts, want the failed one", len(attempts))
			}
			a := attempts[0]
			g := a.log.GroundTruth
			if a.trial.Correlation != tt.correlation || a.trial.KernelLaunches != tt.launches || f.launches != tt.launches || a.trial.Correct || a.log.Complete || !strings.Contains(a.trial.Error, tt.want.Error()) {
				t.Fatalf("failed attempt retained as %+v", a.trial)
			}
			if g.OpID == 0 || g.ProcessGen != process || g.Workload.Process != process || g.Workload.WorkloadName != "gpudiagdemo-native" || g.Workload.RunID != "native-test" || g.RequestID != "request-1" || a.log.Attempt != "baseline" {
				t.Fatalf("failed attempt lost its application identity: %+v", g)
			}
			if tt.correlation == "partial" {
				ann := a.annotation
				if ann == nil || ann.OpID != g.OpID || ann.Process != process || ann.TID != g.TID || ann.NativeCorrelationID != 0 || ann.Provenance != cudaworkload.ProvenanceCUPTI || ann.Symbol != "cuLaunchKernel" {
					t.Fatalf("partial annotation = %+v, ground truth %+v", ann, g)
				}
			} else if a.annotation != nil || g.TID != 0 {
				t.Fatalf("unentered operation claims annotation %+v or thread %d", a.annotation, g.TID)
			}
			if n := f.count("stream/0"); n != 1 {
				t.Fatalf("stream closed %d times", n)
			}
		})
	}
}

func TestNativeFinishCleanup(t *testing.T) {
	all := []string{"synchronize", "activity session", "module", "device", "lease"}
	tests := []struct {
		name    string
		skip    bool
		pending bool
		order   []string
	}{
		{"ordinary", false, false, all},
		{"activity rollback failed", true, true, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var order []string
			result := &NativeResult{cleanup: &nativeCleanup{}}
			for _, name := range []string{"lease", "device", "module", "activity session", "synchronize"} {
				result.cleanup.add(name, func() error { order = append(order, name); return nil })
			}
			primary := errors.New("primary")
			if err := finishNativeCleanup(result, primary, tt.skip); !errors.Is(err, primary) || result.CleanupPending != tt.pending || !reflect.DeepEqual(order, tt.order) {
				t.Fatalf("finish = %v pending=%t order=%v", err, result.CleanupPending, order)
			}
		})
	}
}

func TestNativeFinishCleanupHoldsFailedRequestDrain(t *testing.T) {
	ctx := testContext(t)
	errClose := errors.New("stream close failed")
	streamCalls, failures := 0, 1
	f := &fakeNative{close: func(name string, index int) error {
		if name == "stream" && index == 1 {
			streamCalls++
			if failures > 0 {
				failures--
				return errClose
			}
		}
		return nil
	}}
	var order []string
	result := &NativeResult{cleanup: &nativeCleanup{}}
	for _, name := range []string{"lease", "device", "module", "activity session", "synchronize"} {
		result.cleanup.add(name, func() error { order = append(order, name); return nil })
	}
	_, runErr := f.executor().run(ctx, 1, planNativeRequests(1, 2), result.cleanup)
	err := finishNativeCleanup(result, runErr, false)
	if !errors.Is(err, errClose) || streamCalls != 1 || len(order) != 0 || !result.CleanupPending || len(result.cleanup.actions) != 8 {
		t.Fatalf("automatic cleanup retried the failed drain: err=%v stream calls=%d order=%v pending=%t actions=%d", err, streamCalls, order, result.CleanupPending, len(result.cleanup.actions))
	}
	if err := result.Close(); err != nil || result.CleanupPending {
		t.Fatalf("Close = %v pending=%t", err, result.CleanupPending)
	}
	if want := []string{"synchronize", "activity session", "module", "device", "lease"}; streamCalls != 2 || !reflect.DeepEqual(order, want) {
		t.Fatalf("explicit retry: stream calls=%d order=%v, want 2 and %v", streamCalls, order, want)
	}
	if f.opened != 2 || f.launches != 2 {
		t.Fatalf("retry resubmitted work: opened=%d launches=%d", f.opened, f.launches)
	}
}

// TestRunNativeFinishesCleanupOnce ties the automatic cleanup that the tests
// above exercise to RunNative, which cannot run without a GPU: its only
// cleanup finalization is one deferred finishNativeCleanup call.
func TestRunNativeFinishesCleanupOnce(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "native.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	calls := func(n ast.Node, name string) int {
		count := 0
		ast.Inspect(n, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				if id, ok := call.Fun.(*ast.Ident); ok && id.Name == name {
					count++
				}
			}
			return true
		})
		return count
	}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "RunNative" {
			continue
		}
		deferred := 0
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if d, ok := n.(*ast.DeferStmt); ok {
				deferred += calls(d, "finishNativeCleanup")
			}
			return true
		})
		if deferred != 1 || calls(fn.Body, "finishNativeCleanup") != 1 || calls(fn.Body, "joinNativeCleanup") != 0 {
			t.Fatalf("RunNative: %d deferred and %d total finishNativeCleanup calls, %d joinNativeCleanup calls; want 1, 1, 0", deferred, calls(fn.Body, "finishNativeCleanup"), calls(fn.Body, "joinNativeCleanup"))
		}
		return
	}
	t.Fatal("RunNative not found in native.go")
}
