package kernelagent

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func diagnosticFixture() *DiagnosticInput {
	return &DiagnosticInput{IncumbentID: "base", WorkloadKey: "work", Target: "sm_fixture", Source: "synthetic caller-bound fixture", Inventory: cuptipcsampling.ReasonInventory{CUPTIVersion: 27, Reasons: []cuptipcsampling.StallReason{{Index: 1, Name: "memory"}, {Index: 2, Name: "selected-zero"}, {Index: 3, Name: "not-selected"}}}, Selected: []uint32{1, 2}, Batch: cuptipcsampling.Batch{CUPTIVersion: 27, Valid: true, TotalSamples: 10, PCs: []cuptipcsampling.PC{{Reasons: []cuptipcsampling.StallSamples{{Index: 1, Samples: 10}, {Index: 2, Samples: 0}}}}}}
}

func TestStallCounterAbsentNotZero(t *testing.T) {
	in := diagnosticFixture()
	s, err := Summarize(in, "base", "work", "sm_fixture", []string{"memory", "selected-zero", "not-selected", "unsupported"})
	if err != nil {
		t.Fatal(err)
	}
	if !s.Counters[1].Available || s.Counters[1].Samples != 0 || s.Counters[2].Available || s.Counters[3].Available || s.Counters[3].Reason != "unsupported on target" {
		t.Fatalf("availability=%+v", s.Counters)
	}
	s, err = Summarize(nil, "base", "work", "sm_fixture", []string{"unsupported"})
	if err != nil || s.Available || s.Counters[0].Available || s.Reason == "" {
		t.Fatalf("unavailable=%+v %v", s, err)
	}
}

func TestInferredBottleneckLabeledInferred(t *testing.T) {
	in := diagnosticFixture()
	s, err := Summarize(in, "base", "work", "sm_fixture", []string{"memory"})
	if err != nil || !s.Inferred || s.Classification != "largest requested stall count: memory" {
		t.Fatalf("classification=%+v %v", s, err)
	}
	in.Batch.DroppedSamples = 1
	s, err = Summarize(in, "base", "work", "sm_fixture", []string{"memory"})
	if err != nil || s.Classification != "unknown" || s.Dropped != 1 || s.Reason == "" {
		t.Fatalf("loss hidden: %+v %v", s, err)
	}
}

func TestDiagnosticBoundsAndIdentity(t *testing.T) {
	for _, change := range []func(*DiagnosticInput){func(x *DiagnosticInput) { x.IncumbentID = "stale" }, func(x *DiagnosticInput) { x.WorkloadKey = "stale" }, func(x *DiagnosticInput) { x.Target = "stale" }, func(x *DiagnosticInput) { x.Batch.Valid = false }, func(x *DiagnosticInput) { x.Batch.PCs = make([]cuptipcsampling.PC, 257) }, func(x *DiagnosticInput) { x.Batch.PCs[0].Reasons[0].Index = 99 }} {
		in := diagnosticFixture()
		change(in)
		if _, err := Summarize(in, "base", "work", "sm_fixture", []string{"memory"}); err == nil {
			t.Fatal("invalid diagnostic admitted")
		}
	}
}
