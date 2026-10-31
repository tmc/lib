package cudadiagdemo

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func identityExecutor(f *fakeNative, workers int) *nativeExecutor {
	e := f.executor()
	e.retain = true
	e.wait = func(context.Context, time.Duration) error { return nil }
	for range workers {
		e.annotators = append(e.annotators, cudaworkload.NewAnnotator(cudaworkload.ProcessGenID{PID: 42, Generation: 7}, "same-display-name", cudaworkload.WithRunID("identity-control"), cudaworkload.WithCorrelationHook(&fakeCorrelation{})))
	}
	return e
}
func identityBatch(a nativeAttempt, contextID, streamID uint32) cudaworkload.ActivityBatch {
	return cudaworkload.ActivityBatch{Session: "controlled-host", Run: "identity-control", ClockEpoch: "controlled-device-epoch", EvidenceRef: "constructed-not-native", Process: cudaworkload.ProcessGenID{PID: 42, Generation: 7}, ExternalKind: cupti.ExternalCorrelationKindCustom0, Records: []cuptiactivity.Record{
		{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 9, ContextID: contextID, StreamID: streamID, Start: 10, End: 20, Complete: true},
		{Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 9, ExternalID: uint64(a.annotation.OpID), ExternalKind: cupti.ExternalCorrelationKindCustom0},
	}}
}
func TestNativeQueriedIdentityPropagation(t *testing.T) {
	ctx := testContext(t)
	barrier := newBarrier(2)
	f := &fakeNative{synchronize: func(context.Context, int) error { return barrier.wait(ctx) }}
	e := identityExecutor(f, 2)
	var mu sync.Mutex
	queries := map[int]int{}
	e.identity = func(stream nativeStream) (cudaworkload.ContextStreamID, error) {
		s := stream.(*fakeNativeStream)
		mu.Lock()
		queries[s.index]++
		mu.Unlock()
		return cudaworkload.ContextStreamID{Context: 44, Stream: uint64(100 + s.index)}, nil
	}
	attempts, err := e.run(ctx, 2, planNativeRequests(1, 2), &nativeCleanup{})
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 3 {
		t.Fatal("missing attempts", len(attempts))
	}
	ops := map[cudaworkload.OpID]bool{}
	// This oracle computes the arithmetic bytes independently of nativeReference,
	// the fake's launch method and the joiner's ID matching implementation.
	expected := make([]byte, e.elements*4)
	for i := 0; i < e.elements; i++ {
		binary.LittleEndian.PutUint32(expected[4*i:], math.Float32bits(float32(2*i+1)))
	}
	for i, a := range attempts {
		want := cudaworkload.ContextStreamID{Context: 44, Stream: uint64(100 + i)}
		if a.trial.ContextStream == nil || *a.trial.ContextStream != want || a.annotation == nil || a.annotation.Context != want {
			t.Fatalf("queried identity lost for attempt %d: trial=%v annotation=%+v", i, a.trial.ContextStream, a.annotation)
		}
		if a.trial.OperationName != "cuLaunchKernel" || a.annotation.Workload.WorkloadName != "same-display-name" {
			t.Fatal("operation display-name collision control absent", a.trial.OperationName)
		}
		if ops[a.annotation.OpID] || a.annotation.OpID == 0 {
			t.Fatal("operation IDs collided")
		}
		ops[a.annotation.OpID] = true
		if queries[i] != 1 {
			t.Fatal("identity query count", i, queries[i])
		}
		if !a.trial.Correct || a.trial.OutputSHA256 != digestBytes(expected) {
			t.Fatal("independent numerical output differs", i, a.trial)
		}
		batch := identityBatch(a, 44, uint32(100+i))
		joins, err := cudaworkload.JoinActivities(batch, []cudaworkload.HostSubmission{*a.annotation})
		if err != nil || len(joins) != 1 || joins[0].Join.Outcome != cudaworkload.OutcomeExact {
			t.Fatal("queried public join not exact", joins, err)
		}
		// Expected activity identities come from the independent request-index
		// oracle, never from annotation fields under test or a raw activity echo.
		batch.Records[0].StreamID++
		joins, err = cudaworkload.JoinActivities(batch, []cudaworkload.HostSubmission{*a.annotation})
		if err != nil || joins[0].Join.Outcome == cudaworkload.OutcomeExact {
			t.Fatal("swapped stream produced exact join", joins, err)
		}
		batch = identityBatch(a, 44, uint32(100+i))
		batch.Records = batch.Records[:1]
		joins, err = cudaworkload.JoinActivities(batch, []cudaworkload.HostSubmission{*a.annotation})
		if err != nil || joins[0].Join.Outcome != cudaworkload.OutcomeMissing {
			t.Fatal("absent external ID became exact", joins, err)
		}
	}
}

func TestNativeIdentityQueryScope(t *testing.T) {
	for _, zero := range []bool{false, true} {
		t.Run(fmt.Sprint(zero), func(t *testing.T) {
			var calls []string
			withHandle := func(fn func(driver.Stream) error) error {
				calls = append(calls, "handle-enter")
				defer func() { calls = append(calls, "handle-exit") }()
				return fn(driver.Stream(66))
			}
			withContext := func(fn func() error) error {
				calls = append(calls, "context-enter")
				defer func() { calls = append(calls, "context-restore") }()
				return fn()
			}
			cid, sid := uint32(11), uint32(22)
			if zero {
				cid, sid = 0, 0
			}
			q := nativeIdentityQuery{
				current: func() (driver.Context, error) { calls = append(calls, "current"); return driver.Context(55), nil },
				contextID: func(c driver.Context) (uint32, error) {
					calls = append(calls, "context-id")
					if c != driver.Context(55) {
						t.Fatal("context handle changed")
					}
					return cid, nil
				},
				streamID: func(c driver.Context, s driver.Stream, perThread uint8) (uint32, error) {
					calls = append(calls, "stream-id")
					if c != driver.Context(55) || s != driver.Stream(66) || perThread != 0 {
						t.Fatal("membership/default-stream policy changed", c, s, perThread)
					}
					return sid, nil
				},
			}
			got, err := queryNativeIdentity(withContext, withHandle, q)
			if err != nil || got != (cudaworkload.ContextStreamID{Context: uint64(cid), Stream: uint64(sid)}) {
				t.Fatal("queried IDs not returned", got, err)
			}
			want := []string{"handle-enter", "context-enter", "current", "context-id", "stream-id", "context-restore", "handle-exit"}
			if !reflect.DeepEqual(calls, want) {
				t.Fatal("query escaped lifetime/context scope", calls, want)
			}
		})
	}
}
func TestNativeIdentityQueryErrors(t *testing.T) {
	sentinel := errors.New("controlled identity failure")
	for _, where := range []string{"handle", "context", "current", "context-id", "stream-id", "restore"} {
		t.Run(where, func(t *testing.T) {
			var calls []string
			wh := func(fn func(driver.Stream) error) error {
				calls = append(calls, "handle")
				if where == "handle" {
					return sentinel
				}
				return fn(driver.Stream(66))
			}
			wc := func(fn func() error) error {
				calls = append(calls, "context")
				if where == "context" {
					return sentinel
				}
				err := fn()
				calls = append(calls, "restore")
				if where == "restore" {
					return errors.Join(err, sentinel)
				}
				return err
			}
			q := nativeIdentityQuery{current: func() (driver.Context, error) {
				calls = append(calls, "current")
				if where == "current" {
					return 0, sentinel
				}
				return driver.Context(55), nil
			}, contextID: func(driver.Context) (uint32, error) {
				calls = append(calls, "context-id")
				if where == "context-id" {
					return 0, sentinel
				}
				return 11, nil
			}, streamID: func(driver.Context, driver.Stream, uint8) (uint32, error) {
				calls = append(calls, "stream-id")
				if where == "stream-id" {
					return 0, sentinel
				}
				return 22, nil
			}}
			got, err := queryNativeIdentity(wc, wh, q)
			if !errors.Is(err, sentinel) || got != (cudaworkload.ContextStreamID{}) {
				t.Fatal("failed query published identity", got, err)
			}
			if where == "current" || where == "context-id" || where == "stream-id" || where == "restore" {
				if calls[len(calls)-1] != "restore" {
					t.Fatal("failed query skipped context restoration", calls)
				}
			}
		})
	}
	if _, err := queryNativeIdentity(nil, nil, nativeIdentityQuery{}); err == nil {
		t.Fatal("missing query accepted")
	}
}
func TestNativeWorkerIdentityFailureAndPartial(t *testing.T) {
	sentinel := errors.New("controlled query/launch failure")
	for _, where := range []string{"query", "launch"} {
		t.Run(where, func(t *testing.T) {
			f := &fakeNative{}
			if where == "launch" {
				f.launch = func(int) error { return sentinel }
			}
			e := identityExecutor(f, 1)
			want := cudaworkload.ContextStreamID{Context: 44, Stream: 100}
			e.identity = func(nativeStream) (cudaworkload.ContextStreamID, error) {
				if where == "query" {
					return cudaworkload.ContextStreamID{}, sentinel
				}
				return want, nil
			}
			attempts, err := e.run(testContext(t), 1, planNativeRequests(1, 2)[:1], &nativeCleanup{})
			if !errors.Is(err, sentinel) || len(attempts) != 1 {
				t.Fatal("identity failure lost", attempts, err)
			}
			a := attempts[0]
			if where == "query" {
				if a.annotation != nil || a.trial.ContextStream != nil || f.launches != 0 || a.trial.Timing.Ready != nil {
					t.Fatal("failed query entered launches", a)
				}
			} else {
				if a.annotation == nil || a.annotation.Context != want || a.trial.Correlation != "partial" || a.trial.ContextStream == nil || *a.trial.ContextStream != want {
					t.Fatal("failed launch lost queried partial annotation", a)
				}
			}
			for _, owner := range []string{"input/0", "output/0", "stream/0"} {
				if f.count(owner) != 1 {
					t.Fatal("request owner not closed exactly once", owner, f.count(owner))
				}
			}
		})
	}
}
func TestNativeIdentityUnavailableCompatibility(t *testing.T) {
	t.Run("uninstrumented", func(t *testing.T) {
		f := &fakeNative{}
		e := f.executor()
		e.identity = func(nativeStream) (cudaworkload.ContextStreamID, error) {
			t.Fatal("queried CUPTI without captured attribution")
			return cudaworkload.ContextStreamID{}, nil
		}
		attempts, err := e.run(testContext(t), 1, planNativeRequests(1, 2)[:1], &nativeCleanup{})
		if err != nil || len(attempts) != 1 || attempts[0].trial.ContextStream != nil || attempts[0].trial.Correlation != "uninstrumented" {
			t.Fatal(attempts, err)
		}
	})
	t.Run("legacy-resolver-absent", func(t *testing.T) {
		f := &fakeNative{}
		e := identityExecutor(f, 1)
		attempts, err := e.run(testContext(t), 1, planNativeRequests(1, 2)[:1], &nativeCleanup{})
		if err != nil || len(attempts) != 1 {
			t.Fatal(err)
		}
		a := attempts[0]
		if a.trial.ContextStream != nil || a.annotation == nil || a.annotation.Context != (cudaworkload.ContextStreamID{}) {
			t.Fatal("absent query fabricated identity", a)
		}
		batch := identityBatch(a, 44, 100)
		joins, err := cudaworkload.JoinActivities(batch, []cudaworkload.HostSubmission{*a.annotation})
		if err != nil || joins[0].Join.Outcome == cudaworkload.OutcomeExact {
			t.Fatal("unknown context became exact", joins, err)
		}
	})
}

func TestNativeIdentityCanceledBeforeQuery(t *testing.T) {
	ctx, cancel := context.WithCancel(testContext(t))
	defer cancel()
	f := &fakeNative{upload: func(int) error { cancel(); return nil }}
	e := identityExecutor(f, 1)
	e.identity = func(nativeStream) (cudaworkload.ContextStreamID, error) {
		t.Fatal("identity queried after cancellation")
		return cudaworkload.ContextStreamID{}, nil
	}
	attempts, err := e.run(ctx, 1, planNativeRequests(1, 2)[:1], &nativeCleanup{})
	if !errors.Is(err, context.Canceled) || len(attempts) != 1 || attempts[0].annotation != nil || attempts[0].trial.ContextStream != nil || f.launches != 0 {
		t.Fatal("canceled query entered submission", attempts, err)
	}
	for _, owner := range []string{"input/0", "output/0", "stream/0"} {
		if f.count(owner) != 1 {
			t.Fatal("cancellation lost owner cleanup", owner, f.count(owner))
		}
	}
}
