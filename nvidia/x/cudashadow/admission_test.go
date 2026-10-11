package cudashadow

import (
	"context"
	"testing"
)

type inspectionPolicy struct{ e *Evaluator }

func (p *inspectionPolicy) Info() PolicyInfo {
	return PolicyInfo{Name: "inspection", Version: "1", Domain: DomainEviction}
}
func (p *inspectionPolicy) Evaluate(_ context.Context, in Input) (Decision, error) {
	p.e.Snapshots() // Policies may inspect prior outcomes without deadlocking.
	in.Eviction.PID = 99
	return Decision{Type: DecisionBypass, Reason: "observed", Parameters: map[string]int64{"value": 1}}, nil
}
func shadowInput() Input {
	return Input{Domain: DomainEviction, Provenance: Provenance{DriverVersion: "1", KernelModule: "uvm", Hook: HookBlockActivate, TimestampNano: 10}, Eviction: &EvictionData{PID: 1}}
}
func TestShadowAdmission(t *testing.T) {
	e, _ := NewEvaluator(4)
	e.SetClock(func() int64 { return 20 })
	p := &inspectionPolicy{e: e}
	in := shadowInput()
	in.Provenance.TimestampNano = 21
	s, err := e.Evaluate(context.Background(), p, in)
	if err != nil || s.Status != StatusStale {
		t.Fatalf("future input admitted: %+v %v", s, err)
	}
	var nilPolicy *inspectionPolicy
	if _, err := e.Evaluate(context.Background(), nilPolicy, shadowInput()); err == nil {
		t.Fatal("typed nil admitted")
	}
	for _, change := range []func(*Input){func(in *Input) { in.Provenance.Hook = "unknown" }, func(in *Input) { in.Provenance.TTL = -1 }, func(in *Input) { in.PTX = &PTXData{} }} {
		in := shadowInput()
		change(&in)
		if err := in.Validate(); err == nil {
			t.Fatal("invalid input admitted")
		}
	}
}
func TestShadowSnapshotsOwnTheirEvidence(t *testing.T) {
	e, _ := NewEvaluator(4)
	e.SetClock(func() int64 { return 20 })
	p := &inspectionPolicy{e: e}
	in := shadowInput()
	s, err := e.Evaluate(context.Background(), p, in)
	if err != nil {
		t.Fatal(err)
	}
	in.Eviction.PID = 77
	s.Decision.Parameters["value"] = 8
	got := e.Snapshots()
	if got[0].Input.Eviction.PID != 1 || got[0].Decision.Parameters["value"] != 1 {
		t.Fatalf("aliased retained evidence: %+v", got)
	}
	got[0].Input.Eviction.PID = 88
	got[0].Decision.Parameters["value"] = 9
	if again := e.Snapshots(); again[0].Input.Eviction.PID != 1 || again[0].Decision.Parameters["value"] != 1 {
		t.Fatal("snapshot read mutated evidence")
	}
}
