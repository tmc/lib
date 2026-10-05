package cudahotswap

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

type pushRepairLease struct {
	calls int
	fail  bool
	err   error
}

func (l *pushRepairLease) Close() error {
	l.calls++
	if l.fail {
		return l.err
	}
	return nil
}

func TestPushRollbackPartialAcquireOwner(t *testing.T) {
	for _, operation := range []string{"push", "select", "revert"} {
		for _, retry := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/retry_%t", operation, retry), func(t *testing.T) {
				l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
				s, leases := testService(t, l)
				registerService(t, s)
				mismatch := errors.New("application mismatch")
				acquireErr := errors.New("rollback acquisition failed")
				closeErr := errors.New("partial lease release failed")
				partial := &pushRepairLease{fail: retry, err: closeErr}
				acquire := s.acquire
				s.acquire = func(ctx context.Context, name string) (io.Closer, error) {
					if operation != "push" || len(*leases) >= 2 {
						return partial, acquireErr
					}
					return acquire(ctx, name)
				}
				var err error
				if operation == "push" {
					l.kernel = &serviceKernel{raw: 2}
					if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { return mismatch }}, "check"); err != nil {
						t.Fatal(err)
					}
					var result *PushResult
					result, err = s.Push(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}, "partial rollback")
					if result == nil || !result.Published || result.RolledBack || result.Selected.Version != 2 || !errors.Is(err, mismatch) {
						t.Fatalf("failed rollback lost candidate or original failure: %+v, %v", result, err)
					}
				} else {
					var state State
					if operation == "select" {
						state, err = s.Select(context.Background(), "kernel", 1, "select")
					} else {
						state, err = s.Revert(context.Background(), "kernel", "revert")
					}
					if state.Generation != 0 {
						t.Fatalf("failed acquisition claimed publication: %+v", state)
					}
				}
				if !errors.Is(err, acquireErr) || errors.Is(err, closeErr) != retry || partial.calls != 1 {
					t.Fatalf("partial lease not closed with joined errors: calls=%d, err=%v", partial.calls, err)
				}
				wantRetained := 0
				if retry {
					wantRetained = 1
				}
				if len(s.cleanup) != wantRetained {
					t.Fatalf("retained owners=%d, want %d", len(s.cleanup), wantRetained)
				}
				s.acquire = acquire
				partial.fail = false
				if err := s.Close(context.Background()); err != nil {
					t.Fatal(err)
				}
				if partial.calls != 1+wantRetained || len(s.cleanup) != 0 {
					t.Fatal("Close did not retry exactly the retained lease")
				}
			})
		}
	}
}

func TestPushCheckerUnwindCleanup(t *testing.T) {
	for _, mode := range []string{"panic", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			original := &serviceKernel{raw: 1}
			l := &serviceLoader{kernel: original}
			s, leases := testService(t, l)
			registerService(t, s)
			candidate := &serviceKernel{raw: 2}
			l.kernel = candidate
			sentinel := errors.New("checker panic")
			if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error {
				if mode == "goexit" {
					runtime.Goexit()
				}
				panic(sentinel)
			}}, "check"); err != nil {
				t.Fatal(err)
			}
			done := make(chan any, 1)
			go func() {
				defer func() { done <- recover() }()
				_, _ = s.Push(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}, "unwind")
			}()
			select {
			case recovered := <-done:
				if mode == "panic" && recovered != sentinel || mode == "goexit" && recovered != nil {
					t.Fatalf("changed callback unwind: %v", recovered)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("callback did not unwind")
			}
			if len(s.checking) != 0 || s.table.entries["kernel"].versions[1].pins.Load() != 0 {
				t.Fatal("callback unwind stranded checking marker or prior generation pin")
			}
			if state, _ := s.Status("kernel"); state.Version != 2 || (*leases)[1].calls != 1 {
				t.Fatal("callback unwind changed publication or retained request lease")
			}
			if err := s.Close(context.Background()); err != nil || original.calls != 1 || candidate.calls != 1 {
				t.Fatalf("Close after callback unwind: %v", err)
			}
		})
	}
}

func TestPushApplicationPublicationRollbackAndReentry(t *testing.T) {
	for _, mode := range []string{"pass", "rollback", "keep", "poison"} {
		t.Run(mode, func(t *testing.T) {
			original := &serviceKernel{raw: 1}
			loader := &serviceLoader{kernel: original}
			s, leases := testService(t, loader)
			registerService(t, s)
			candidate := &serviceKernel{raw: 2}
			loader.kernel = candidate
			sub, err := s.Subscribe(context.Background(), 0)
			if err != nil {
				t.Fatal(err)
			}
			defer sub.Close()
			config := CheckConfig{KeepOnFailure: mode == "keep", Checker: func(ctx context.Context, state State) error {
				if state.Version != 2 || state.ConfigRevision != 1 || (*leases)[1].calls != 1 {
					t.Fatal("checker did not observe committed candidate after lease release")
				}
				if status, _ := s.Status("kernel"); status != state {
					t.Fatal("checker state differs from published table")
				}
				pin, err := s.Pin("kernel")
				if err != nil || pin.Function() != 2 {
					t.Fatalf("reentrant pin: %v", err)
				}
				if err := pin.Release(ctx); err != nil {
					t.Fatal(err)
				}
				work, err := s.Launch(ctx, "kernel", cudalaunch.Config{})
				if err != nil || work.kernel != candidate || work.Wait(ctx) != nil {
					t.Fatalf("reentrant launch: %v", err)
				}
				if err := s.Close(ctx); err == nil || s.closed || original.calls != 0 || candidate.calls != 0 {
					t.Fatal("Close invalidated resources needed by callback")
				}
				if _, err := s.Select(ctx, "kernel", 1, "foreign select"); err == nil {
					t.Fatal("callback invalidated exact rollback target")
				}
				if _, err := s.SetCompileFlags("kernel", nil, "foreign config"); err == nil {
					t.Fatal("config changed under callback")
				}
				if mode == "poison" {
					return errors.Join(context.Canceled, driver.ErrIllegalAddress)
				}
				if mode == "rollback" || mode == "keep" {
					return errors.New("logits mismatch")
				}
				return nil
			}}
			if _, err := s.SetCheck("kernel", config, "configure"); err != nil {
				t.Fatal(err)
			}
			result, err := s.Push(context.Background(), "kernel", Source{Code: "new source", Entry: "entry"}, "candidate")
			if result == nil || !result.Published || (err == nil) != (mode == "pass") || result.State.Version != 2 || result.Report.ConfigRevision != 1 {
				t.Fatalf("push result: %+v, %v", result, err)
			}
			want := uint64(2)
			if mode == "rollback" {
				want = 1
				if !result.RolledBack || result.Selected.Generation != 3 || candidate.calls != 0 {
					t.Fatal("rollback did not restore prior code under new generation")
				}
			}
			if status, _ := s.Status("kernel"); status.Version != want || result.Selected != status {
				t.Fatal("final selection differs from result")
			}
			if mode == "poison" {
				if result.Report.Outcome != OutcomeDeviceFault || !result.Report.ContextPoisoned || result.RolledBack || s.poison == nil {
					t.Fatal("joined cancellation masked terminal CUDA fault")
				}
				if _, err := s.Pin("kernel"); err == nil {
					t.Fatal("poisoned service still admits requests")
				}
			}
			event := <-sub.Events()
			if event.Report.Generation != 2 || event.Report.ConfigRevision != 1 || event.Version != 2 {
				t.Fatal("event is stamped with rollback or wrong config revision")
			}
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestPushPublicReplayConsumer(t *testing.T) {
	for _, mode := range []string{"correct", "wrong", "noop", "nonfinite", "stable-write"} {
		t.Run(mode, func(t *testing.T) {
			s, baseline, backend, config, leases := serviceFixture(t)
			f, err := s.RecordFixture(context.Background(), "kernel", config)
			if err != nil {
				t.Fatal(err)
			}
			called := false
			if _, err := s.SetCheck("kernel", CheckConfig{Fixture: f, Checker: func(context.Context, State) error { called = true; return nil }}, "check"); err != nil {
				t.Fatal(err)
			}
			candidate := &serviceFixtureKernel{fixtureTestKernel: fixtureKernel("candidate", false)}
			good := candidate.run
			candidate.run = func(c cudalaunch.Config, args []any) error {
				if mode == "noop" {
					return nil
				}
				if err := good(c, args); err != nil {
					return err
				}
				if mode == "wrong" {
					copy(args[1].([]byte), fixtureBytes(100, 200))
				}
				if mode == "nonfinite" {
					copy(args[1].([]byte), fixtureBytes(float32(math.NaN()), 10))
				}
				if mode == "stable-write" {
					copy(args[2].([]byte), fixtureBytes(1, 2))
				}
				return nil
			}
			s.loader.(*serviceLoader).kernel = candidate
			before := len(*leases)
			result, err := s.Push(context.Background(), "kernel", Source{Code: "candidate " + mode, Entry: "entry"}, mode)
			if result == nil || (err == nil) != (mode == "correct") || called != (mode == "correct") || result.Published != (mode == "correct") {
				t.Fatalf("public replay %s: %+v, %v", mode, result, err)
			}
			if len(*leases) != before+1 || (*leases)[before].calls != 1 || backend.lease != nil {
				t.Fatal("replay took a second exclusive lease or retained idle lease")
			}
			if result.Report.Replay == nil || result.Report.Replay.Native || result.Report.Timing != nil || result.Report.Speedup != nil {
				t.Fatal("fake host replay promoted into native timing")
			}
			if mode != "stable-write" && (len(result.Report.Replay.Outputs) != 1 || len(result.Report.Replay.Trials) != 10) {
				t.Fatal("report omitted actual engine comparisons/arms")
			}
			if mode == "nonfinite" && result.Report.Replay.Outputs[0].Candidate.NonFinite != 1 {
				t.Fatal("nonfinite comparison lost")
			}
			if mode == "correct" {
				*result.Report.Replay.Outputs[0].Noise.MaxAbs = 100
				if *f.replay.noise[1].MaxAbs != 0 {
					t.Fatal("returned report aliases retained fixture noise")
				}
			}
			if mode != "correct" && candidate.calls != 1 {
				t.Fatal("refused candidate was not closed after completion")
			}
			if !reflect.DeepEqual(backend.data[2], fixtureBytes(5, 7)) || baseline.calls != 0 {
				t.Fatal("stable recovery or baseline pin lifetime failed")
			}
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestPushRefusesBeforeLease(t *testing.T) {
	for _, mode := range []string{"unconfigured", "compile", "abi", "stale", "nil-context"} {
		t.Run(mode, func(t *testing.T) {
			l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
			s, leases := testService(t, l)
			registerService(t, s)
			if mode != "unconfigured" {
				if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { return nil }}, "check"); err != nil {
					t.Fatal(err)
				}
			}
			l.compile = func(context.Context, Source) (*Compilation, error) {
				if len(*leases) != 1 {
					t.Fatal("compilation took a GPU lease")
				}
				if mode == "compile" {
					return nil, errors.New("compiler diagnostic")
				}
				if mode == "stale" {
					_, err := s.SetCompileFlags("kernel", []string{"--std=c++20"}, "change")
					if err != nil {
						t.Fatal(err)
					}
				}
				params := ""
				if mode == "abi" {
					params = ".param .u32 n"
				}
				return &Compilation{Entry: "entry", PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry(" + params + ") { ret; }"}, nil
			}
			ctx := context.Background()
			if mode == "nil-context" {
				ctx = nil
			}
			result, err := s.Push(ctx, "kernel", Source{Code: "candidate", Entry: "entry"}, "test")
			if err == nil || len(*leases) != 1 || l.loads != 1 {
				t.Fatalf("pre-lease refusal failed: %v", err)
			}
			if mode == "compile" && result.Report.Outcome != OutcomeCompileError {
				t.Fatal("compiler error classified as device/harness error")
			}
			if mode == "abi" && result.Report.Outcome != OutcomeABIMismatch {
				t.Fatal("real PTX ABI mismatch lost")
			}
			if mode == "stale" && (result.Report.ConfigRevision != 1 || result.Selected.ConfigRevision != 2) {
				t.Fatal("stale-compilation report replaced request stamp with new settings")
			}
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestPushPartialLoadAndCleanupLeaseRetry(t *testing.T) {
	l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, leases := testService(t, l)
	registerService(t, s)
	_, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { t.Fatal("partial load invoked checker"); return nil }}, "check")
	if err != nil {
		t.Fatal(err)
	}
	candidate := &serviceKernel{raw: 2, serviceCloser: serviceCloser{fail: true}}
	extra := &serviceCleanupError{&serviceCloser{fail: true}}
	l.kernel, l.loadErr = candidate, extra
	result, err := s.Push(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}, "partial")
	if !errors.Is(err, extra) || result.Published || len(s.cleanup) != 1 || (*leases)[1].calls != 0 {
		t.Fatal("partial owner or actual lease discarded")
	}
	candidate.fail = false
	if err := s.Close(context.Background()); err == nil || (*leases)[1].calls != 0 {
		t.Fatal("lease released before extra owner")
	}
	extra.fail = false
	if err := s.Close(context.Background()); err != nil || (*leases)[1].calls != 1 || candidate.calls != 2 {
		t.Fatalf("partial retry: %v", err)
	}
}

type canceledPushLoader struct {
	*serviceLoader
	cancel context.CancelFunc
}

func (l *canceledPushLoader) Load(ctx context.Context, c *Compilation) (Kernel, error) {
	k, err := l.serviceLoader.Load(ctx, c)
	if l.cancel != nil {
		l.cancel()
		return k, errors.Join(err, context.Canceled)
	}
	return k, err
}

func TestPushCancellationWithPartialLoadedOwner(t *testing.T) {
	l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, leases := testService(t, l)
	registerService(t, s)
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { t.Fatal("canceled load reached app"); return nil }}, "check"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	k := &serviceKernel{raw: 2, serviceCloser: serviceCloser{fail: true}}
	l.kernel = k
	s.loader = &canceledPushLoader{l, cancel}
	r, err := s.Push(ctx, "kernel", Source{Code: "partial load", Entry: "entry"}, "cancel")
	if !errors.Is(err, context.Canceled) || r.Published || k.calls != 1 || (*leases)[1].calls != 0 || len(s.cleanup) != 1 || s.poison != nil {
		t.Fatal("partial canceled load abandoned failed owner/lease")
	}
	k.fail = false
	if err := s.Close(context.Background()); err != nil || k.calls != 2 || (*leases)[1].calls != 1 {
		t.Fatalf("partial load retry: %v", err)
	}
}

func TestPushPreloadStaleRefusalPreservesRequestAndSelectedStamps(t *testing.T) {
	l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, leases := testService(t, l)
	registerService(t, s)
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { return nil }}, "check"); err != nil {
		t.Fatal(err)
	}
	p, err := s.prepareCandidate(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"})
	if err != nil {
		t.Fatal(err)
	}
	latest, err := s.SetCompileFlags("kernel", nil, "stale after compile")
	if err != nil {
		t.Fatal(err)
	}
	r := &PushResult{State: p.state, Selected: p.state}
	checking, _, err := s.pushGPU(context.Background(), p, "candidate", r)
	if err == nil || checking != nil || r.Published || len(*leases) != 1 || l.loads != 1 || r.State.ConfigRevision != 1 || r.Selected != latest {
		t.Fatal("preload stale refusal lost request stamp/current selection or touched GPU")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestPushCanceledReplayRetainsActualLeaseAndCandidate(t *testing.T) {
	s, baseline, backend, config, leases := serviceFixture(t)
	f, err := s.RecordFixture(context.Background(), "kernel", config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.SetCheck("kernel", CheckConfig{Fixture: f}, "check"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	candidate := &serviceFixtureKernel{fixtureTestKernel: fixtureKernel("candidate", false)}
	candidate.wait = func(context.Context) error { cancel(); return context.Canceled }
	s.loader.(*serviceLoader).kernel = candidate
	result, err := s.Push(ctx, "kernel", Source{Code: "candidate", Entry: "entry"}, "cancel")
	if !errors.Is(err, context.Canceled) || result.Published || candidate.calls != 0 || !backend.pending || (*leases)[2].calls != 0 || len(s.cleanup) != 1 || s.poison != nil {
		t.Fatal("cancellation substituted for completion or lost candidate/lease")
	}
	if err := s.Close(context.Background()); err != nil || candidate.calls != 1 || baseline.calls != 1 || (*leases)[2].calls != 1 {
		t.Fatalf("retained drain retry: %v", err)
	}
}

func TestPushConcurrentCloseAndMutationRefusal(t *testing.T) {
	l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, _ := testService(t, l)
	registerService(t, s)
	l.kernel = &serviceKernel{raw: 2}
	started, release, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	_, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { close(started); <-release; return nil }}, "check")
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		_, err := s.Push(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}, "candidate")
		done <- err
	}()
	<-started
	var group sync.WaitGroup
	for _, operation := range []func() error{
		func() error { return s.Close(context.Background()) },
		func() error {
			_, e := s.Push(context.Background(), "kernel", Source{Code: "other", Entry: "entry"}, "other")
			return e
		},
		func() error { _, e := s.Revert(context.Background(), "kernel", "other"); return e },
		func() error { _, e := s.SetGeometry("kernel", nil, "other"); return e },
	} {
		group.Add(1)
		go func(op func() error) {
			defer group.Done()
			if op() == nil {
				t.Error("concurrent operation invalidated check")
			}
		}(operation)
	}
	group.Wait()
	close(release)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("app hook retained service gates")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestPushLeaseFailureAndCommittedCleanupError(t *testing.T) {
	for _, mode := range []string{"acquire", "release"} {
		t.Run(mode, func(t *testing.T) {
			l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
			s, _ := testService(t, l)
			registerService(t, s)
			candidate := &serviceKernel{raw: 2}
			l.kernel = candidate
			called := false
			if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { called = true; return nil }}, "check"); err != nil {
				t.Fatal(err)
			}
			lease := &serviceCloser{fail: true}
			s.acquire = func(context.Context, string) (io.Closer, error) {
				if mode == "acquire" {
					return lease, errors.New("partial lease acquisition")
				}
				return lease, nil
			}
			result, err := s.Push(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}, "candidate")
			if err == nil || called || result.Published != (mode == "release") || len(s.cleanup) != 1 || result.Selected.Version != map[string]uint64{"acquire": 1, "release": 2}[mode] {
				t.Fatal("partial lease or committed-state cleanup error lost")
			}
			lease.fail = false
			s.acquire = func(context.Context, string) (io.Closer, error) { return new(serviceCloser), nil }
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

type pushExclusiveLease struct {
	held *bool
	fail bool
}

func (l *pushExclusiveLease) Close() error {
	if l.fail {
		return errors.New("exclusive lease release failed")
	}
	*l.held = false
	return nil
}

func TestPushFailedReleaseRetriesBeforeFixtureAcquire(t *testing.T) {
	s, _, backend, config, _ := serviceFixture(t)
	f, err := s.RecordFixture(context.Background(), "kernel", config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.SetCheck("kernel", CheckConfig{Fixture: f}, "replay"); err != nil {
		t.Fatal(err)
	}
	k := &serviceFixtureKernel{fixtureTestKernel: fixtureKernel("candidate", false)}
	s.loader.(*serviceLoader).kernel = k
	held := false
	var actual *pushExclusiveLease
	s.acquire = func(context.Context, string) (io.Closer, error) {
		if held {
			return nil, errors.New("second exclusive lease before retained release")
		}
		held = true
		actual = &pushExclusiveLease{held: &held}
		return actual, nil
	}
	k.wait = func(ctx context.Context) error { actual.fail = true; return ctx.Err() }
	r, err := s.Push(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}, "candidate")
	if err == nil || !r.Published || !held || len(s.cleanup) != 1 || backend.lease != nil || f.replay == nil {
		t.Fatal("completed replay release failure discarded parent or fixture")
	}
	actual.fail = false
	if err := s.Close(context.Background()); err != nil || held || f.replay != nil {
		t.Fatalf("Close reacquired before retrying retained parent: %v", err)
	}
}

func TestPushReplayReportDetachedBoundedAndClock(t *testing.T) {
	r := replayResult{Native: true, Timing: "cuda_event_interval", Pass: true, SignP: 0.5, CandidateWins: 2}
	for i := 0; i < 4; i++ {
		arm := "baseline"
		d := 20 * time.Nanosecond
		if i%2 == 1 {
			arm, d = "candidate", 10*time.Nanosecond
		}
		receipt := cudatrial.Receipt{Name: arm, Round: i / 2, Complete: true}
		if err := receipt.RecordTiming(r.Timing, d); err != nil {
			t.Fatal(err)
		}
		r.Trials = append(r.Trials, receipt)
	}
	m, _ := MeasureReplay([]float32{1, 2}, []float32{1, 2})
	r.Outputs = []replayComparison{{Arg: 1, Noise: m, Candidate: m, Pass: true}}
	report := &Report{}
	setPushReplay(report, r)
	if report.Timing == nil || report.Timing.MedianNanosPerLaunch != 10 || report.Speedup.Ratio != 2 || report.Replay.SignP == nil {
		t.Fatal("actual labelled event samples not summarized")
	}
	clone := cloneEventReport(report)
	*report.Replay.Outputs[0].Candidate.MaxAbs = 50
	*report.Replay.Trials[0].DeviceNanos = 99
	if *clone.Replay.Outputs[0].Candidate.MaxAbs != 0 || *clone.Replay.Trials[0].DeviceNanos != 20 {
		t.Fatal("event aliases replay report")
	}
	clone.Replay.Trials[0].Digest = strings.Repeat("x", 1<<20)
	if reportEventFits(State{}, clone, 64<<10) {
		t.Fatal("new replay fields bypass retained report budget")
	}
	r.Native = false
	report = &Report{}
	setPushReplay(report, r)
	if report.Timing != nil || report.Replay.SignP != nil || report.Replay.Trials[0].DeviceNanos != nil {
		t.Fatal("fake timing became native performance")
	}
}

func TestPushFixturePolicyOverridesAndGeometryRefusal(t *testing.T) {
	s, _, _, config, leases := serviceFixture(t)
	f, err := s.RecordFixture(context.Background(), "kernel", config)
	if err != nil {
		t.Fatal(err)
	}
	stored := f.replay.policy
	policy := ReplayPolicy{Rounds: 3, MaxAbs: 2, RMS: 2, MaxULP: math.MaxUint32}
	if _, err := s.SetCheck("kernel", CheckConfig{Fixture: f, Policy: &policy}, "relaxed"); err != nil {
		t.Fatal(err)
	}
	// The configured snapshot must not alias the caller's policy.
	policy.Rounds, policy.MaxAbs = 1, 0
	k := &serviceFixtureKernel{fixtureTestKernel: fixtureKernel("candidate", false)}
	good := k.run
	k.run = func(c cudalaunch.Config, args []any) error {
		if err := good(c, args); err != nil {
			return err
		}
		copy(args[1].([]byte), fixtureBytes(8, 11))
		return nil
	}
	s.loader.(*serviceLoader).kernel = k
	r, err := s.Push(context.Background(), "kernel", Source{Code: "within explicit policy", Entry: "entry"}, "relaxed")
	if err != nil || !r.Published || !r.Report.Replay.Passed || len(r.Report.Replay.Trials) != 6 || f.replay.policy != stored {
		t.Fatalf("policy snapshot not applied/restored: %+v, %v", r, err)
	}
	g := LaunchGeometry{Grid: config.Launch.Grid, Block: config.Launch.Block, SharedMem: config.Launch.SharedMem}
	g.Grid.X = 2
	if _, err := s.SetGeometry("kernel", &g, "changed launch"); err != nil {
		t.Fatal(err)
	}
	before := len(*leases)
	if r, err := s.Push(context.Background(), "kernel", Source{Code: "another", Entry: "entry"}, "geometry"); err == nil || r.Published || len(*leases) != before {
		t.Fatal("recorded replay checked a different published launch geometry")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestPushNewGenerationRemainsRecordableAndLaunchable(t *testing.T) {
	s, _, _, config, _ := serviceFixture(t)
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { return nil }}, "app"); err != nil {
		t.Fatal(err)
	}
	k := &serviceFixtureKernel{fixtureTestKernel: fixtureKernel("candidate", false)}
	s.loader.(*serviceLoader).kernel = k
	r, err := s.Push(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}, "candidate")
	if err != nil {
		t.Fatal(err)
	}
	s.newReplay = func(string, replaySpec, replayPolicy, []replayDeviceBinding) (replayBackend, error) {
		return &serviceFixtureBackend{fixtureTestBackend: fixtureBackend(), service: s}, nil
	}
	f, err := s.RecordFixture(context.Background(), "kernel", config)
	if err != nil || f.State().Version != r.State.Version || f.replay.baseline.Kernel != k {
		t.Fatalf("published owner no longer satisfies actual Kernel: %v", err)
	}
	if pin, err := s.Pin("kernel"); err != nil || pin.Function() != k.Raw() || pin.Release(context.Background()) != nil {
		t.Fatal("published candidate lost public pin contract")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestPushPublicGraphPatchAppRollbackAndFailedRestoration(t *testing.T) {
	for _, failRestore := range []bool{false, true} {
		t.Run(map[bool]string{false: "app rollback", true: "partial patch retention"}[failRestore], func(t *testing.T) {
			l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
			s, leases := testService(t, l)
			registerService(t, s)
			var installed driver.Function
			fail := false
			s.graphs.ops.patch = func(_ *cudagraph.Exec, _ *cudagraph.Node, fn driver.Function, _ driver.Dim3, _ []any) error {
				installed = fn
				if fail {
					return errors.New("partial graph patch or restoration")
				}
				return nil
			}
			if _, err := s.RegisterGraph(context.Background(), "kernel", new(cudagraph.Exec), new(cudagraph.Node), new(cuda.Stream), driver.Dims(1, 1, 1)); err != nil {
				t.Fatal(err)
			}
			called := false
			if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error {
				called = true
				if installed != 2 {
					t.Fatal("published candidate graph still uses original")
				}
				return errors.New("application differs")
			}}, "app"); err != nil {
				t.Fatal(err)
			}
			k := &serviceKernel{raw: 2}
			l.kernel = k
			fail = failRestore
			r, err := s.Push(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}, "candidate")
			if err == nil || r.Published == failRestore || called == failRestore || installed != 1 {
				t.Fatalf("graph check: %+v, %v, installed %d", r, err, installed)
			}
			if failRestore && (k.calls != 0 || s.graphs.patch == nil || (*leases)[2].calls != 0) {
				t.Fatal("candidate unloaded or lease released before graph restoration")
			}
			fail = false
			if err := s.Close(context.Background()); err != nil || k.calls != 1 {
				t.Fatalf("graph cleanup retry: %v", err)
			}
		})
	}
}

func TestPushCallbackWithCloseWriterQueued(t *testing.T) {
	l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, _ := testService(t, l)
	registerService(t, s)
	l.kernel = &serviceKernel{raw: 2}
	closeDone := make(chan error, 1)
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(ctx context.Context, _ State) error {
		// Hold the mutation gate briefly so Close obtains the requests writer
		// gate first. Then release it before calling any public consumer API.
		s.mutations.RLock()
		go func() { closeDone <- s.Close(ctx) }()
		deadline := time.Now().Add(time.Second)
		for s.requests.TryRLock() {
			s.requests.RUnlock()
			if time.Now().After(deadline) {
				s.mutations.RUnlock()
				return errors.New("Close writer did not queue")
			}
			time.Sleep(time.Millisecond)
		}
		s.mutations.RUnlock()
		work, err := s.Launch(ctx, "kernel", cudalaunch.Config{})
		if err != nil {
			return err
		}
		if err := work.Wait(ctx); err != nil {
			return err
		}
		// Push uses requests.RLock itself; it must refuse promptly rather
		// than deadlock against a queued Close writer as a recursive reader.
		if _, err := s.Push(ctx, "kernel", Source{Code: "nested", Entry: "entry"}, "nested"); err == nil {
			return errors.New("nested mutation admitted")
		}
		return nil
	}}, "app"); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := s.Push(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}, "candidate")
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("callback deadlocked with queued Close writer")
	}
	if err := <-closeDone; err == nil {
		t.Fatal("Close changed resources during callback")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func FuzzPushReplayReportClone(f *testing.F) {
	f.Add([]byte("arm digest"), uint8(3))
	f.Add([]byte{}, uint8(0))
	f.Fuzz(func(t *testing.T, raw []byte, count uint8) {
		if len(raw) > 4096 {
			t.Skip()
		}
		n := int(count)
		value := float64(1)
		ns := int64(10)
		r := &Report{Replay: &ReplayReport{Clock: string(raw), SignP: &value}}
		for i := 0; i < n; i++ {
			r.Replay.Trials = append(r.Replay.Trials, ReplayTrialReport{Arm: string(raw), Digest: string(raw), DeviceNanos: &ns})
			r.Replay.Outputs = append(r.Replay.Outputs, ReplayOutputReport{Arg: i, Candidate: ReplayMetrics{MaxAbs: &value, RMS: &value}})
		}
		if !reportEventFits(State{}, r, 64<<10) {
			return
		}
		clone := cloneEventReport(r)
		value, ns = 5, 50
		if *clone.Replay.SignP != 1 || len(clone.Replay.Trials) != n || cap(clone.Replay.Trials) != n || cap(clone.Replay.Outputs) != n {
			t.Fatal("replay report retained mutable owner or excess slice backing")
		}
		for i := 0; i < n; i++ {
			if *clone.Replay.Trials[i].DeviceNanos != 10 || *clone.Replay.Outputs[i].Candidate.MaxAbs != 1 || *clone.Replay.Outputs[i].Candidate.RMS != 1 {
				t.Fatal("replay scalar pointer aliases publisher")
			}
		}
	})
}
