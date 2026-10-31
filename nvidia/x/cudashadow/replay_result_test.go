package cudashadow_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashadow"
)

// A policy may reuse its own result storage between synchronous calls. The
// evaluator must detach snapshots from that storage before the next call.
type replayResultPolicy struct {
	*cudashadow.ActiveBackendSpy
	calls    int
	infos    int
	scratch  map[string]int64
	cancel   context.CancelFunc
	cancelAt int
	failure  error
}

func newReplayResultPolicy(t *testing.T) *replayResultPolicy {
	t.Helper()
	p := &replayResultPolicy{ActiveBackendSpy: cudashadow.NewActiveBackendSpy(), scratch: make(map[string]int64)}
	t.Cleanup(func() {
		if err := p.AssertZeroMutations(); err != nil {
			t.Error(err)
		}
	})
	return p
}
func (p *replayResultPolicy) Info() cudashadow.PolicyInfo {
	p.infos++
	return cudashadow.PolicyInfo{Name: "replay-result-control", Version: "1", Domain: cudashadow.DomainScheduling, Seed: 7}
}
func (p *replayResultPolicy) Evaluate(context.Context, cudashadow.Input) (cudashadow.Decision, error) {
	p.calls++
	p.scratch["timeslice_us"] = int64(p.calls)
	if p.cancel != nil && p.calls == p.cancelAt {
		p.cancel()
	}
	return cudashadow.Decision{Type: cudashadow.DecisionTimeslice, Parameters: p.scratch, Reason: "constructed observational proposal"}, p.failure
}

func replayResultHistory(t *testing.T, n int) ([]cudashadow.Snapshot, []byte) {
	t.Helper()
	e, err := cudashadow.NewEvaluator(n + 1)
	if err != nil {
		t.Fatal(err)
	}
	e.SetClock(func() int64 { return 100 })
	p := newReplayResultPolicy(t)
	for i := 1; i <= n; i++ {
		provenance := cudashadow.Provenance{DeviceID: 0, DriverVersion: "constructed-host-control", KernelModule: "constructed-host-control", Hook: cudashadow.HookSchedTimeslice, Sequence: uint64(i), TimestampNano: int64(i)}
		in := cudashadow.Input{Provenance: provenance, Domain: cudashadow.DomainScheduling, Scheduling: &cudashadow.SchedulingData{PID: 1234, ChannelID: 1, CurrentSlice: 1}}
		if _, err := e.Evaluate(context.Background(), p, in); err != nil {
			t.Fatal(err)
		}
	}
	var wire bytes.Buffer
	if err := cudashadow.WriteHistory(&wire, e.Snapshots()); err != nil {
		t.Fatal(err)
	}
	history, err := cudashadow.ReadHistory(bytes.NewReader(wire.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	return history, wire.Bytes()
}

func TestReplayKeepsIndependentDecisions(t *testing.T) {
	history, wire := replayResultHistory(t, 3)
	e, err := cudashadow.NewEvaluator(1)
	if err != nil {
		t.Fatal(err)
	}
	p := newReplayResultPolicy(t)
	got, err := e.Replay(context.Background(), p, history)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(history) || p.calls != 3 {
		t.Fatalf("replay length=%d, calls=%d", len(got), p.calls)
	}
	for i := range got {
		if n := got[i].Decision.Parameters["timeslice_us"]; n != int64(i+1) {
			t.Fatalf("replay step %d changed to %d, want %d", i, n, i+1)
		}
	}
	// Later policy reuse and caller edits must not mutate other returned steps
	// or the input history that was read from the canonical wire format.
	p.scratch["timeslice_us"] = 99
	got[0].Decision.Parameters["timeslice_us"] = 88
	for i := 1; i < len(got); i++ {
		if n := got[i].Decision.Parameters["timeslice_us"]; n != int64(i+1) {
			t.Fatalf("replay step %d aliases another result: %d", i, n)
		}
	}
	var after bytes.Buffer
	if err := cudashadow.WriteHistory(&after, history); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(wire, after.Bytes()) {
		t.Fatal("replay or returned-map mutation changed recorded input history")
	}
}

func TestReplayCancellationAfterPolicy(t *testing.T) {
	for _, tt := range []struct{ n, at int }{{1, 1}, {2, 2}, {3, 1}} {
		t.Run(fmt.Sprintf("steps%d_cancel%d", tt.n, tt.at), func(t *testing.T) {
			history, _ := replayResultHistory(t, tt.n)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			p := newReplayResultPolicy(t)
			p.cancel, p.cancelAt = cancel, tt.at
			e, err := cudashadow.NewEvaluator(1)
			if err != nil {
				t.Fatal(err)
			}
			got, err := e.Replay(ctx, p, history)
			if !errors.Is(err, context.Canceled) || got != nil {
				t.Fatalf("Replay after policy canceled context = %d snapshots, %v; want cancellation and nil history", len(got), err)
			}
			if p.calls != tt.at {
				t.Fatalf("policy calls=%d, want %d", p.calls, tt.at)
			}
		})
	}
}

func TestReplayPolicyErrorPrecedesCancellation(t *testing.T) {
	history, _ := replayResultHistory(t, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p := newReplayResultPolicy(t)
	p.cancel, p.cancelAt = cancel, 1
	failure := errors.New("controlled policy failure")
	p.failure = failure
	e, err := cudashadow.NewEvaluator(1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Replay(ctx, p, history)
	if got != nil || !errors.Is(err, failure) || errors.Is(err, context.Canceled) {
		t.Fatalf("policy failure precedence = %d snapshots, %v", len(got), err)
	}
}
