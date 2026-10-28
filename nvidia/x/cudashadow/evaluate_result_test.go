package cudashadow_test

import (
	"bytes"
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashadow"
)

type evaluateResultPolicy struct {
	*cudashadow.ActiveBackendSpy
	calls  int
	params map[string]int64
	fixed  bool
}

func newEvaluateResultPolicy(t *testing.T) *evaluateResultPolicy {
	t.Helper()
	p := &evaluateResultPolicy{ActiveBackendSpy: cudashadow.NewActiveBackendSpy(), params: make(map[string]int64)}
	t.Cleanup(func() {
		if err := p.AssertZeroMutations(); err != nil {
			t.Error(err)
		}
	})
	return p
}

func (p *evaluateResultPolicy) Info() cudashadow.PolicyInfo {
	return cudashadow.PolicyInfo{Name: "evaluate-result-control", Version: "1", Domain: cudashadow.DomainScheduling, Seed: 7}
}

func (p *evaluateResultPolicy) Evaluate(context.Context, cudashadow.Input) (cudashadow.Decision, error) {
	p.calls++
	if !p.fixed {
		p.params["timeslice_us"] = int64(p.calls)
	}
	return cudashadow.Decision{Type: cudashadow.DecisionTimeslice, Parameters: p.params, Reason: "constructed observational proposal"}, nil
}

func evaluateResultInput(seq uint64) cudashadow.Input {
	return cudashadow.Input{
		Provenance: cudashadow.Provenance{DeviceID: 0, DriverVersion: "constructed-host-control", KernelModule: "constructed-host-control", Hook: cudashadow.HookSchedTimeslice, Sequence: seq, TimestampNano: int64(seq)},
		Domain:     cudashadow.DomainScheduling,
		Scheduling: &cudashadow.SchedulingData{PID: 1234, ChannelID: 1, CurrentSlice: 1},
	}
}

func newEvaluateResultEvaluator(t *testing.T) *cudashadow.Evaluator {
	t.Helper()
	e, err := cudashadow.NewEvaluator(4)
	if err != nil {
		t.Fatal(err)
	}
	e.SetClock(func() int64 { return 100 })
	return e
}

func TestEvaluateReturnedHistoryReplays(t *testing.T) {
	e := newEvaluateResultEvaluator(t)
	p := newEvaluateResultPolicy(t)
	var returned []cudashadow.Snapshot
	for seq := uint64(1); seq <= 3; seq++ {
		s, err := e.Evaluate(context.Background(), p, evaluateResultInput(seq))
		if err != nil {
			t.Fatal(err)
		}
		returned = append(returned, s)
	}
	// Internal storage is an independent positive control for the proposals
	// actually made by the policy, not the source of returned history below.
	for i, s := range e.Snapshots() {
		if got := s.Decision.Parameters["timeslice_us"]; got != int64(i+1) {
			t.Fatalf("stored step %d = %d, want %d", i, got, i+1)
		}
	}
	var wire bytes.Buffer
	if err := cudashadow.WriteHistory(&wire, returned); err != nil {
		t.Fatal(err)
	}
	history, err := cudashadow.ReadHistory(bytes.NewReader(wire.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := e.Replay(context.Background(), newEvaluateResultPolicy(t), history)
	if err != nil {
		t.Fatalf("returned-snapshot history does not replay: %v", err)
	}
	for i, s := range replayed {
		if got := s.Decision.Parameters["timeslice_us"]; got != int64(i+1) {
			t.Fatalf("returned step %d = %d, want %d", i, got, i+1)
		}
	}
}

func TestEvaluateResultOwnsDecisionMap(t *testing.T) {
	e := newEvaluateResultEvaluator(t)
	p := newEvaluateResultPolicy(t)
	s, err := e.Evaluate(context.Background(), p, evaluateResultInput(1))
	if err != nil {
		t.Fatal(err)
	}
	s.Decision.Parameters["timeslice_us"] = 88
	if got := p.params["timeslice_us"]; got != 1 {
		t.Fatalf("caller snapshot write changed policy scratch to %d, want 1", got)
	}
	p.params["timeslice_us"] = 99
	if got := s.Decision.Parameters["timeslice_us"]; got != 88 {
		t.Fatalf("policy scratch write changed returned snapshot to %d, want 88", got)
	}
	if got := e.Snapshots()[0].Decision.Parameters["timeslice_us"]; got != 1 {
		t.Fatalf("caller or policy mutation changed internal snapshot to %d, want 1", got)
	}
}

func TestEvaluateDecisionMapShape(t *testing.T) {
	for _, nilMap := range []bool{true, false} {
		name := "empty"
		if nilMap {
			name = "nil"
		}
		t.Run(name, func(t *testing.T) {
			e := newEvaluateResultEvaluator(t)
			p := newEvaluateResultPolicy(t)
			p.fixed = true
			if nilMap {
				p.params = nil
			}
			s, err := e.Evaluate(context.Background(), p, evaluateResultInput(1))
			if err != nil {
				t.Fatal(err)
			}
			if (s.Decision.Parameters == nil) != nilMap {
				t.Fatalf("nil map = %v, want %v", s.Decision.Parameters == nil, nilMap)
			}
			if !nilMap {
				p.params["later"] = 1
			}
			if len(s.Decision.Parameters) != 0 || len(e.Snapshots()[0].Decision.Parameters) != 0 {
				t.Fatal("empty returned or stored map aliases policy storage")
			}
		})
	}
}
