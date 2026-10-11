package cudashadow

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestCorpusBoundAndTransitions(t *testing.T) {
	for _, constructor := range []func(PolicyInfo) (Policy, error){NewFIFOPolicy, NewLFUPolicy, NewMRUPolicy} {
		policy, err := constructor(PolicyInfo{Name: "bounded", Version: "1", Domain: DomainEviction, Parameters: map[string]string{"max_entries": "1"}})
		if err != nil {
			t.Fatal(err)
		}
		input := shadowInput()
		input.Eviction.ChunkAddr = 1
		if _, err := policy.Evaluate(context.Background(), input); err != nil {
			t.Fatal(err)
		}
		input.Eviction.ChunkAddr = 2
		if _, err := policy.Evaluate(context.Background(), input); !errors.Is(err, ErrCapacity) {
			t.Fatalf("capacity: %v", err)
		}
		input.Provenance.Hook = HookBlockAccess
		if _, err := policy.Evaluate(context.Background(), input); err == nil {
			t.Fatal("invalid transition")
		}
	}
}

func TestLFUTieReplay(t *testing.T) {
	info := PolicyInfo{Name: "lfu", Version: "1", Domain: DomainEviction}
	for trial := 0; trial < 20; trial++ {
		policy, _ := NewLFUPolicy(info)
		for _, addr := range []uint64{3, 2, 1} {
			input := shadowInput()
			input.Eviction.ChunkAddr = addr
			if _, err := policy.Evaluate(context.Background(), input); err != nil {
				t.Fatal(err)
			}
		}
		input := shadowInput()
		input.Provenance.Hook = HookEvictPrepare
		decision, err := policy.Evaluate(context.Background(), input)
		if err != nil || decision.Parameters["target_chunk"] != 1 {
			t.Fatalf("non-deterministic tie: %+v %v", decision, err)
		}
	}
}

type waitingPolicy struct {
	entered chan struct{}
	release chan struct{}
}

func (p *waitingPolicy) Info() PolicyInfo {
	return PolicyInfo{Name: "wait", Version: "1", Domain: DomainEviction}
}
func (p *waitingPolicy) Evaluate(ctx context.Context, _ Input) (Decision, error) {
	close(p.entered)
	select {
	case <-p.release:
		return Decision{Type: DecisionSkip, Reason: "released"}, nil
	case <-ctx.Done():
		return Decision{}, ctx.Err()
	}
}

func TestQueuedEvaluationCancellation(t *testing.T) {
	e, _ := NewEvaluator(4)
	e.SetClock(func() int64 { return 20 })
	p := &waitingPolicy{entered: make(chan struct{}), release: make(chan struct{})}
	done := make(chan error, 1)
	go func() { _, err := e.Evaluate(context.Background(), p, shadowInput()); done <- err }()
	<-p.entered
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := e.Evaluate(ctx, p, shadowInput()); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued cancellation: %v", err)
	}
	close(p.release)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("evaluation blocked")
	}
}

func TestHistoryPublicRoundTrip(t *testing.T) {
	e, _ := NewEvaluator(4)
	e.SetClock(func() int64 { return 20 })
	info := PolicyInfo{Name: "fifo", Version: "1", Domain: DomainEviction}
	policy, _ := NewFIFOPolicy(info)
	input := shadowInput()
	input.Provenance.Sequence = 1
	if _, err := e.Evaluate(context.Background(), policy, input); err != nil {
		t.Fatal(err)
	}
	input.Provenance.Sequence = 2
	input.Provenance.Hook = HookEvictPrepare
	if _, err := e.Evaluate(context.Background(), policy, input); err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err := WriteHistory(&encoded, e.Snapshots()); err != nil {
		t.Fatal(err)
	}
	history, err := ReadHistory(bytes.NewReader(encoded.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	fresh, _ := NewFIFOPolicy(info)
	if _, err := e.Replay(context.Background(), fresh, history); err != nil {
		t.Fatal(err)
	}
	for _, mutated := range []string{strings.Replace(encoded.String(), `"version":1`, `"version":2`, 1), strings.Replace(encoded.String(), `"status":"valid"`, `"status":"invented"`, 1), encoded.String() + `{}`} {
		if _, err := ReadHistory(strings.NewReader(mutated)); err == nil {
			t.Fatal("mutated history admitted")
		}
	}
	if _, err := ReadHistory(strings.NewReader(`{"version":2,"Version":1,"snapshots":[]}`)); err == nil {
		t.Fatal("case-aliased version admitted")
	}
	missing := shadowInput()
	missing.Provenance.DriverVersion = ""
	if _, err := e.Evaluate(context.Background(), policy, missing); err == nil {
		t.Fatal("missing input admitted")
	}
	encoded.Reset()
	if err := WriteHistory(&encoded, e.Snapshots()); err != nil {
		t.Fatalf("unsupported observation cannot serialize: %v", err)
	}
}

type invalidDecisionPolicy struct{}

func (invalidDecisionPolicy) Info() PolicyInfo {
	return PolicyInfo{Name: "invalid", Version: "1", Domain: DomainEviction}
}
func (invalidDecisionPolicy) Evaluate(context.Context, Input) (Decision, error) {
	return Decision{Type: "execute_native", Reason: "unsupported"}, nil
}

func TestUnsupportedDecisionAndOutcome(t *testing.T) {
	e, _ := NewEvaluator(4)
	e.SetClock(func() int64 { return 20 })
	s, err := e.Evaluate(context.Background(), invalidDecisionPolicy{}, shadowInput())
	if err == nil || s.Status != StatusUnsupported || s.Decision.Type != DecisionReject {
		t.Fatalf("unsupported decision admitted: %+v %v", s, err)
	}
	p, _ := NewFIFOPolicy(PolicyInfo{Name: "fifo", Version: "1", Domain: DomainEviction})
	s, err = e.Evaluate(context.Background(), p, shadowInput())
	if err != nil {
		t.Fatal(err)
	}
	if err := e.RecordOutcome(s.ID, ObservedOutcome{TimestampNano: 9, Action: "done"}); err == nil {
		t.Fatal("outcome before input admitted")
	}
	outcome := ObservedOutcome{TimestampNano: 21, Action: "done", Details: map[string]string{"control": "baseline"}}
	if err := e.RecordOutcome(s.ID, outcome); err != nil {
		t.Fatal(err)
	}
	if err := e.RecordOutcome(s.ID, outcome); err == nil {
		t.Fatal("outcome overwritten")
	}
}
