package cudaworkload

import (
	"context"
	"testing"
)

func TestJoinRejectsIncompleteIdentity(t *testing.T) {
	for _, tc := range []struct {
		name         string
		proc         ProcessGenID
		host, device ContextStreamID
	}{
		{"unknown generation", ProcessGenID{PID: 1}, ContextStreamID{Context: 1}, ContextStreamID{Context: 1}},
		{"different context", ProcessGenID{PID: 1, Generation: 2}, ContextStreamID{Context: 1}, ContextStreamID{Context: 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			j := NewJoiner(JoinConfig{})
			j.SubmitHost(HostSubmission{Process: tc.proc, OpID: 1, NativeCorrelationID: 7, Context: tc.host})
			got := j.SubmitDevice(DeviceActivity{Process: tc.proc, NativeCorrelationID: 7, Context: tc.device})
			if len(got) != 1 || got[0].Outcome == OutcomeExact {
				t.Fatalf("invalid exact join: %+v", got)
			}
		})
	}
}
func TestDeviceFirstDoesNotFlushConsumedActivity(t *testing.T) {
	j := NewJoiner(JoinConfig{})
	proc := ProcessGenID{PID: 1, Generation: 2}
	j.SubmitDevice(DeviceActivity{Process: proc, NativeCorrelationID: 7, StartNs: 5})
	j.SubmitHost(HostSubmission{Process: proc, OpID: 1, NativeCorrelationID: 7})
	if got := j.Flush(); len(got) != 0 {
		t.Fatalf("consumed activity emitted again: %+v", got)
	}
}

func TestConflictingNativeAndRuntimeIDs(t *testing.T) {
	j := NewJoiner(JoinConfig{})
	p := ProcessGenID{PID: 1, Generation: 2}
	j.SubmitHost(HostSubmission{Process: p, OpID: 1, NativeCorrelationID: 7})
	j.SubmitHost(HostSubmission{Process: p, OpID: 2, NativeCorrelationID: 8})
	got := j.SubmitDevice(DeviceActivity{Process: p, NativeCorrelationID: 7, RuntimeCorrelationID: 8})
	if len(got) != 1 || got[0].Outcome != OutcomeAmbiguous {
		t.Fatalf("conflicting IDs admitted: %+v", got)
	}
}

func TestOperationIDsAcrossAnnotators(t *testing.T) {
	proc := ProcessGenID{PID: 7, Generation: 8}
	a := NewAnnotator(proc, "a")
	b := NewAnnotator(proc, "b")
	_, first := a.Begin(context.Background(), "same")
	_, second := b.Begin(context.Background(), "same")
	if first.ID() == second.ID() || first.ID() == 0 || second.ID() == 0 {
		t.Fatal("process-wide operation identity collision")
	}
	ctx, _ := a.Begin(context.Background(), "parent")
	other := NewAnnotator(ProcessGenID{PID: 8, Generation: 9}, "other")
	_, child := other.Begin(ctx, "child")
	if child.ParentID() != 0 {
		t.Fatal("parent linked across process lifetimes")
	}
}
