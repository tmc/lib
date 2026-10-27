package cudadiagdemo

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// fakeNative is a host device for nativeExecutor. Each opened stream has its
// own buffers unless shared is set. Streams are identified by the plan index
// of the request that opened them, never by their data. The hooks run on the
// request worker.
type fakeNative struct {
	mu          sync.Mutex
	shared      []float32
	opened      int
	launches    int
	inflight    int
	maxInflight int
	closes      map[string]int

	upload      func(index int) error
	launch      func(index int) error
	synchronize func(ctx context.Context, index int) error
	close       func(name string, index int) error
}

type fakeNativeStream struct {
	f       *fakeNative
	index   int
	in, out []float32
}

func (f *fakeNative) open(req nativeRequest, own *nativeCleanup) (nativeStream, error) {
	f.mu.Lock()
	f.opened++
	f.inflight++
	f.maxInflight = max(f.maxInflight, f.inflight)
	f.mu.Unlock()
	s := &fakeNativeStream{f: f, index: req.index}
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

func (s *fakeNativeStream) upload(input []float32) error {
	if s.f.upload != nil {
		if err := s.f.upload(s.index); err != nil {
			return err
		}
	}
	s.in = append([]float32(nil), input...)
	s.out = make([]float32, len(input))
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
				if tm := a.trial.Timing; tm == nil || tm.Recipe != nativeRecipe || tm.Ready == nil || tm.Ready.CUPTI != nil || tm.OpID != 0 || tm.Workload != nil || tm.TID != 0 || tm.TimestampAPI != "unavailable" || tm.TimestampMethod != "" {
					t.Fatalf("attempt %d timing = %+v, want an unbound ready observation without samples", i, tm)
				}
				for _, c := range a.trial.Calls {
					if c.Before != nil || c.After != nil {
						t.Fatalf("attempt %d call %+v has samples without a timestamp source", i, c)
					}
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
	inputs := map[int]string{}
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
		// Matched arms share values, never buffers or identity.
		in := nativeInput(e.elements, a.trial.Round-1)
		if a.trial.InputSHA256 != digestFloats(in) || a.trial.OutputSHA256 != digestFloats(nativeReference(in)) {
			t.Fatalf("attempt %d digests %s -> %s are not its round's", i, a.trial.InputSHA256, a.trial.OutputSHA256)
		}
		if d, ok := inputs[a.trial.Round]; ok && d != a.trial.InputSHA256 {
			t.Fatalf("attempt %d input %s differs from its round's %s", i, a.trial.InputSHA256, d)
		}
		inputs[a.trial.Round] = a.trial.InputSHA256
		for _, name := range []string{"input", "output", "stream"} {
			if n := f.count(fmt.Sprintf("%s/%d", name, i)); n != 1 {
				t.Fatalf("%s of attempt %d closed %d times", name, i, n)
			}
		}
		// The ready observation is bound to the attempt's own operation and
		// worker thread, not to its name or plan index.
		if tm := a.trial.Timing; tm == nil || tm.Ready == nil || tm.OpID != g.OpID || tm.Workload == nil || tm.Workload.Process != process || tm.Workload.RunID != "native-test" || tm.TID != g.TID {
			t.Fatalf("attempt %d timing %+v is not bound to operation %d thread %d", i, tm, g.OpID, g.TID)
		}
		result.Requests = append(result.Requests, a.log)
		result.Trials = append(result.Trials, a.trial)
	}
	if len(inputs) != 2 || inputs[1] == inputs[2] || f.opened != len(plan) {
		t.Fatalf("round inputs %v from %d opened streams, want 2 distinct rounds of %d streams", inputs, f.opened, len(plan))
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
	// shared device memory. All launches finish before any download. The
	// concurrent attempts come from three rounds, so they have different
	// correct outputs; matched arms of one round could not fail.
	plan := planNativeRequests(3, 2)
	requests := []nativeRequest{plan[0], plan[3], plan[6]}
	for i, r := range requests {
		if r.round != i+1 {
			t.Fatalf("request %+v is not from round %d", r, i+1)
		}
	}
	b := newBarrier(len(requests))
	f := &fakeNative{shared: make([]float32, 8), synchronize: func(context.Context, int) error { return b.wait(ctx) }}
	attempts, err := f.executor().run(ctx, len(requests), requests, &nativeCleanup{})
	if len(attempts) != len(requests) || f.maxInflight != len(requests) {
		t.Fatalf("attempts=%d max in-flight=%d, want %d concurrent", len(attempts), f.maxInflight, len(requests))
	}
	inputs := map[string]bool{}
	for _, a := range attempts {
		inputs[a.trial.InputSHA256] = true
	}
	if len(inputs) != len(requests) {
		t.Fatalf("%d distinct inputs among %d rounds", len(inputs), len(requests))
	}
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
		// Each attempt retains every wrapper it called, the failed one once,
		// and none after it.
		for i, want := range [][]string{
			{"cuMemcpyHtoDAsync_v2", "cuLaunchKernel", "cuStreamSynchronize: context canceled"},
			{"cuMemcpyHtoDAsync_v2", "cuLaunchKernel: launch failed"},
		} {
			if got := callNames(attempts[i].trial.Calls); !reflect.DeepEqual(got, want) || attempts[i].trial.KernelLaunches != 1-i || attempts[i].trial.Timing.Ready == nil {
				t.Fatalf("attempt %d calls %v launches %d, want %v", i, got, attempts[i].trial.KernelLaunches, want)
			}
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
			// Ready precedes correlation entry, so it is retained with the
			// application identity either way; entry alone sets the thread.
			wantCalls := []string{"cuMemcpyHtoDAsync_v2"}
			if tt.launches > 0 {
				wantCalls = append(wantCalls, "cuLaunchKernel")
			}
			if tm := a.trial.Timing; tm == nil || tm.Ready == nil || tm.OpID != g.OpID || tm.Workload == nil || tm.Workload.Process != process || !reflect.DeepEqual(callNames(a.trial.Calls), wantCalls) {
				t.Fatalf("timing %+v calls %v, want ready for operation %d and calls %v", tm, callNames(a.trial.Calls), g.OpID, wantCalls)
			}
			checkObservedTID(t, a.trial.Timing.TID)
			if n := f.count("stream/0"); n != 1 {
				t.Fatalf("stream closed %d times", n)
			}
		})
	}
}

// callNames returns the name of each call, with its error if it failed.
func callNames(calls []NativeCall) []string {
	var names []string
	for _, c := range calls {
		name := c.Name
		if c.Error != "" {
			name += ": " + c.Error
		}
		names = append(names, name)
	}
	return names
}

// checkObservedTID checks tid against the existing thread ID provider,
// which observes threads only on Linux; elsewhere zero means unknown.
func checkObservedTID(t *testing.T, tid uint64) {
	t.Helper()
	if (tid != 0) != (runtime.GOOS == "linux") {
		t.Errorf("observed TID %d on %s", tid, runtime.GOOS)
	}
}

// checkSample checks that s is an observed raw sample taken after after and
// before before, with its CUPTI read between its host bounds.
func checkSample(t *testing.T, what string, s *NativeSample, after, before int64) {
	t.Helper()
	if s == nil || s.CUPTI != "observed" || s.Error != "" || s.HostBeforeNS <= after || s.HostAfterNS >= before || int64(s.CUPTINS) <= s.HostBeforeNS || int64(s.CUPTINS) >= s.HostAfterNS {
		t.Errorf("%s: sample %+v, want an observed read within (%d, %d)", what, s, after, before)
	}
}

// TestNativeWorkersLaunchReadyOrder checks the launch-ready-v1 chronology of
// the real executor on one logical clock shared by its observations, its raw
// timestamps and the fake device: each upload is enqueued, then ready is
// observed, then the baseline alone waits, then every planned launch is
// called.
func TestNativeWorkersLaunchReadyOrder(t *testing.T) {
	ctx := testContext(t)
	var (
		clock    atomic.Int64
		mu       sync.Mutex
		uploads  = map[int]int64{}
		launches = map[int][]int64{}
		waits    []int64
	)
	f := &fakeNative{
		upload: func(index int) error {
			mu.Lock()
			defer mu.Unlock()
			uploads[index] = clock.Add(1)
			return nil
		},
		launch: func(index int) error {
			mu.Lock()
			defer mu.Unlock()
			launches[index] = append(launches[index], clock.Add(1))
			return nil
		},
	}
	e := f.executor()
	e.now = func() int64 { return clock.Add(1) }
	e.timestamp = func() (uint64, error) { return uint64(clock.Add(1)), nil }
	e.timestampAPI = "logical clock"
	e.wait = func(context.Context, time.Duration) error {
		mu.Lock()
		defer mu.Unlock()
		waits = append(waits, clock.Add(1))
		return nil
	}
	plan := planNativeRequests(2, 4)
	attempts, err := e.run(ctx, 1, plan, &nativeCleanup{})
	if err != nil || len(attempts) != len(plan) {
		t.Fatalf("attempts=%d err=%v", len(attempts), err)
	}
	waited := 0
	for i, a := range attempts {
		r, tm, calls := plan[i], a.trial.Timing, a.trial.Calls
		want := []string{"cuMemcpyHtoDAsync_v2"}
		for range r.launches {
			want = append(want, "cuLaunchKernel")
		}
		want = append(want, "cuStreamSynchronize", "cuMemcpyDtoH_v2")
		if got := callNames(calls); !reflect.DeepEqual(got, want) || a.trial.KernelLaunches != r.launches || len(launches[i]) != r.launches || !a.trial.Correct {
			t.Fatalf("attempt %d calls %v launched %d (%d by the device), want %v", i, got, a.trial.KernelLaunches, len(launches[i]), want)
		}
		if tm == nil || tm.Recipe != nativeRecipe || tm.TimestampAPI != "logical clock" || tm.TimestampMethod != "unknown" || tm.Ready == nil {
			t.Fatalf("attempt %d timing = %+v", i, tm)
		}
		name := a.trial.RequestID + "/" + a.trial.Attempt
		upload, ready, first := calls[0], tm.Ready.HostNS, calls[1].StartNS
		if at := uploads[i]; at <= upload.StartNS || at >= upload.EndNS {
			t.Errorf("%s: upload at %d outside its wrapper [%d, %d]", name, at, upload.StartNS, upload.EndNS)
		}
		if ready <= upload.EndNS {
			t.Errorf("%s: ready at %d, not after upload ended at %d", name, ready, upload.EndNS)
		}
		if ready >= first {
			t.Errorf("%s: ready at %d, not before first launch at %d", name, ready, first)
		}
		n := 0
		for _, at := range waits {
			if upload.EndNS < at && ready < at && at < first {
				n++
			}
		}
		wantWaits := 0
		if r.arm == "baseline" {
			wantWaits = 1
		}
		if n != wantWaits {
			t.Errorf("%s: %d waits after upload and ready, before first launch, want %d", name, n, wantWaits)
		}
		if r.arm != "baseline" && a.trial.DelayNS != 0 {
			t.Errorf("%s: delay %d inherited by a non-baseline arm", name, a.trial.DelayNS)
		}
		waited += n
		for j, at := range launches[i] {
			if c := calls[1+j]; at <= c.StartNS || at >= c.EndNS {
				t.Errorf("%s: launch %d at %d outside its wrapper [%d, %d]", name, j, at, c.StartNS, c.EndNS)
			}
		}
		checkSample(t, name+" ready", tm.Ready.CUPTI, ready, first)
		for j, c := range calls {
			checkSample(t, fmt.Sprintf("%s call %d before", name, j), c.Before, 0, c.StartNS)
			checkSample(t, fmt.Sprintf("%s call %d after", name, j), c.After, c.EndNS, math.MaxInt64)
		}
	}
	if len(waits) != 2 || waited != len(waits) {
		t.Errorf("%d baseline waits, %d between an upload and a first launch; want 2", len(waits), waited)
	}
}

func TestNativeWorkersReadyRequiresUpload(t *testing.T) {
	errOpen, errUpload, errPush := errors.New("open failed"), errors.New("upload failed"), errors.New("push refused")
	process := cudaworkload.ProcessGenID{PID: uint32(os.Getpid()), Generation: 1}
	annotators := func(hook cudaworkload.CorrelationHook) []*cudaworkload.Annotator {
		return []*cudaworkload.Annotator{cudaworkload.NewAnnotator(process, "gpudiagdemo-native", cudaworkload.WithRunID("native-test"), cudaworkload.WithCorrelationHook(hook))}
	}
	t.Run("open failure", func(t *testing.T) {
		f := &fakeNative{}
		e := f.executor()
		e.open = func(nativeRequest, *nativeCleanup) (nativeStream, error) { return nil, errOpen }
		attempts, err := e.run(testContext(t), 1, planNativeRequests(1, 2), &nativeCleanup{})
		if !errors.Is(err, errOpen) || len(attempts) != 1 {
			t.Fatalf("attempts=%d err=%v", len(attempts), err)
		}
		a := attempts[0]
		if tm := a.trial.Timing; tm == nil || tm.Recipe != nativeRecipe || tm.Ready != nil || len(a.trial.Calls) != 0 || a.trial.Correlation != "" || a.log.StartedNS > a.log.FinishedNS {
			t.Fatalf("failed open retained as %+v timing %+v", a.trial, tm)
		}
	})
	t.Run("upload failure", func(t *testing.T) {
		waited := false
		f := &fakeNative{upload: func(int) error { return errUpload }}
		e := f.executor()
		e.annotators = annotators(&fakeCorrelation{})
		e.wait = func(context.Context, time.Duration) error { waited = true; return nil }
		attempts, err := e.run(testContext(t), 1, planNativeRequests(1, 2), &nativeCleanup{})
		if !errors.Is(err, errUpload) || len(attempts) != 1 {
			t.Fatalf("attempts=%d err=%v", len(attempts), err)
		}
		a := attempts[0]
		if tm := a.trial.Timing; tm == nil || tm.Ready != nil || tm.OpID != 0 || a.log.GroundTruth.OpID != 0 || waited || f.launches != 0 || !reflect.DeepEqual(callNames(a.trial.Calls), []string{"cuMemcpyHtoDAsync_v2: upload failed"}) {
			t.Fatalf("failed upload retained as %+v timing %+v, waited=%t", a.trial, tm, waited)
		}
		if n := f.count("stream/0"); n != 1 {
			t.Fatalf("stream closed %d times", n)
		}
	})
	t.Run("canceled wait", func(t *testing.T) {
		ctx, cancel := context.WithCancel(testContext(t))
		defer cancel()
		f := &fakeNative{}
		e := f.executor()
		// A push would fail, so an entered operation would report errPush.
		e.annotators = annotators(&failingCorrelation{push: errPush})
		e.wait = func(ctx context.Context, _ time.Duration) error {
			cancel()
			<-ctx.Done()
			return ctx.Err()
		}
		attempts, err := e.run(ctx, 1, planNativeRequests(1, 2), &nativeCleanup{})
		if !errors.Is(err, context.Canceled) || errors.Is(err, errPush) || len(attempts) != 1 {
			t.Fatalf("attempts=%d err=%v", len(attempts), err)
		}
		a := attempts[0]
		g, tm := a.log.GroundTruth, a.trial.Timing
		if tm == nil || tm.Ready == nil || tm.OpID == 0 || tm.OpID != g.OpID || g.ProcessGen != process || tm.Workload == nil || tm.Workload.Process != process {
			t.Fatalf("canceled wait lost ready or application identity: timing %+v ground truth %+v", tm, g)
		}
		if a.trial.Correlation != "" || a.annotation != nil || g.TID != 0 || a.trial.KernelLaunches != 0 || f.launches != 0 || a.trial.DelayNS != 0 || !reflect.DeepEqual(callNames(a.trial.Calls), []string{"cuMemcpyHtoDAsync_v2"}) {
			t.Fatalf("canceled wait claims entry or launches: %+v annotation %+v", a.trial, a.annotation)
		}
		checkObservedTID(t, tm.TID)
	})
}

func TestNativeWorkersTimestampSamples(t *testing.T) {
	errStamp, errLaunch := errors.New("timestamp failed"), errors.New("launch failed")
	// attempt runs the first planned request, a baseline, with a timestamp
	// source whose nth read returns read(n). Reads are numbered from 1:
	// CUPTIBefore, then two per wrapper and one at ready, then CUPTIAfter.
	attempt := func(t *testing.T, f *fakeNative, read func(n int) (uint64, error)) (nativeAttempt, error) {
		t.Helper()
		n := 0
		e := f.executor()
		e.timestamp = func() (uint64, error) {
			n++
			return read(n)
		}
		e.timestampAPI = "test"
		attempts, err := e.run(testContext(t), 1, planNativeRequests(1, 2)[:1], &nativeCleanup{})
		if len(attempts) != 1 {
			t.Fatalf("attempts=%d err=%v", len(attempts), err)
		}
		return attempts[0], err
	}
	t.Run("launch error survives failed samples", func(t *testing.T) {
		f := &fakeNative{launch: func(int) error { return errLaunch }}
		// Reads 2 and 3 bracket the upload, 4 is ready, 5 and 6 bracket the launch.
		a, err := attempt(t, f, func(n int) (uint64, error) {
			switch n {
			case 1:
				return 7, nil
			case 3:
				return 0, nil
			}
			return 0, errStamp
		})
		if !errors.Is(err, errLaunch) || errors.Is(err, errStamp) || a.trial.CUPTIBefore != 7 || a.trial.KernelLaunches != 0 || f.launches != 0 {
			t.Fatalf("trial %+v err %v", a.trial, err)
		}
		calls := a.trial.Calls
		if got := callNames(calls); !reflect.DeepEqual(got, []string{"cuMemcpyHtoDAsync_v2", "cuLaunchKernel: launch failed"}) {
			t.Fatalf("calls = %v", got)
		}
		for _, s := range []*NativeSample{calls[0].Before, a.trial.Timing.Ready.CUPTI, calls[1].Before, calls[1].After} {
			if s == nil || s.CUPTI != "error" || s.Error != errStamp.Error() || s.CUPTINS != 0 {
				t.Fatalf("failed read recorded as %+v", s)
			}
		}
		if s := calls[0].After; s == nil || s.CUPTI != "zero" || s.CUPTINS != 0 || s.Error != "" {
			t.Fatalf("zero read recorded as %+v", s)
		}
	})
	t.Run("failed samples keep correct work", func(t *testing.T) {
		f := &fakeNative{}
		a, err := attempt(t, f, func(n int) (uint64, error) {
			if n == 1 || n == 11 {
				return uint64(n), nil
			}
			return 0, errStamp
		})
		if err != nil || !a.trial.Correct || a.trial.Error != "" || a.trial.CUPTIBefore != 1 || a.trial.CUPTIAfter != 11 || len(a.trial.Calls) != 4 || a.trial.Timing.Ready.CUPTI.CUPTI != "error" {
			t.Fatalf("trial %+v err %v", a.trial, err)
		}
		for _, c := range a.trial.Calls {
			if c.Error != "" || c.Before.CUPTI != "error" || c.After.CUPTI != "error" {
				t.Fatalf("call %+v", c)
			}
		}
	})
	t.Run("CUPTIBefore failure", func(t *testing.T) {
		f := &fakeNative{}
		a, err := attempt(t, f, func(int) (uint64, error) { return 0, errStamp })
		if !errors.Is(err, errStamp) || len(a.trial.Calls) != 0 || a.trial.Timing.Ready != nil || f.opened != 1 || f.count("stream/0") != 1 {
			t.Fatalf("trial %+v err %v", a.trial, err)
		}
	})
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
