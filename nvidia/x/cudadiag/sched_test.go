package cudadiag

import (
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func TestSchedEvidence_Matching(t *testing.T) {
	proc := cudaworkload.ProcessGenID{PID: 1234, Generation: 2}
	tid := uint64(5678)

	sched := SchedEvidence{
		Process:      proc,
		TID:          tid,
		ProbePresent: true,
		Intervals: []SchedInterval{
			{
				State: SchedRunning,
				Span:  tracecontract.Interval{Start: 1000, End: 2000},
				CPU:   1,
			},
			{
				State: SchedRunnable,
				Span:  tracecontract.Interval{Start: 2000, End: 2500},
				CPU:   1,
			},
			{
				State: SchedBlocked,
				Span:  tracecontract.Interval{Start: 2500, End: 2800},
				CPU:   1,
			},
		},
	}

	if !sched.Matches(proc, tid) {
		t.Fatalf("expected sched to match target process and TID")
	}

	// Mismatched PID
	wrongProc := cudaworkload.ProcessGenID{PID: 9999, Generation: 2}
	if sched.Matches(wrongProc, tid) {
		t.Fatalf("expected mismatch on PID")
	}

	// Mismatched generation
	wrongGen := cudaworkload.ProcessGenID{PID: 1234, Generation: 3}
	if sched.Matches(wrongGen, tid) {
		t.Fatalf("expected mismatch on generation")
	}

	// Mismatched TID
	if sched.Matches(proc, 9999) {
		t.Fatalf("expected mismatch on TID")
	}

	// Zero values must not match
	if sched.Matches(cudaworkload.ProcessGenID{}, tid) {
		t.Fatalf("zero process must not match")
	}
	if sched.Matches(proc, 0) {
		t.Fatalf("zero TID must not match")
	}

	// Durations check
	runnable, scheduled, wait := sched.Durations()
	if runnable != 500*time.Nanosecond {
		t.Fatalf("runnable got %v, want 500ns", runnable)
	}
	if scheduled != 1000*time.Nanosecond {
		t.Fatalf("scheduled got %v, want 1000ns", scheduled)
	}
	if wait != 300*time.Nanosecond {
		t.Fatalf("wait got %v, want 300ns", wait)
	}
}

func TestSchedEvidence_Validation(t *testing.T) {
	proc := cudaworkload.ProcessGenID{PID: 1234, Generation: 2}
	tid := uint64(5678)

	// Missing probe
	noProbe := SchedEvidence{Process: proc, TID: tid, ProbePresent: false}
	if err := noProbe.Validate(proc, tid); err == nil {
		t.Fatalf("expected error on missing probe")
	}

	// Unknown loss
	unknownLoss := SchedEvidence{Process: proc, TID: tid, ProbePresent: true, UnknownLoss: true}
	if err := unknownLoss.Validate(proc, tid); err == nil {
		t.Fatalf("expected error on unknown loss")
	}

	// Dropped events
	dropped := SchedEvidence{Process: proc, TID: tid, ProbePresent: true, Loss: 5}
	if err := dropped.Validate(proc, tid); err == nil {
		t.Fatalf("expected error on dropped events")
	}

	// Valid
	valid := SchedEvidence{Process: proc, TID: tid, ProbePresent: true}
	if err := valid.Validate(proc, tid); err != nil {
		t.Fatalf("unexpected error on valid sched: %v", err)
	}
}
