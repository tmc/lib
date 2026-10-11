package cudadiag

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func ExampleSchedulerEvidence() {
	proc := cudaworkload.ProcessGenID{PID: 7, Generation: 99}
	events := []SchedulerEvent{{Kind: "sched_wakeup", Timestamp: 10, WakeTID: 7}, {Kind: "sched_switch", Timestamp: 30, NextTID: 7}, {Kind: "sched_switch", Timestamp: 50, PreviousTID: 7, PreviousState: "S"}}
	ev, err := SchedulerEvidence(proc, 7, "linux_monotonic", events, 0, true)
	fmt.Println(len(ev.Intervals), ev.UnknownLoss, err)
	// Output: 2 true <nil>
}

func TestSchedulerEvidenceAdmission(t *testing.T) {
	proc := cudaworkload.ProcessGenID{PID: 7, Generation: 99}
	events := []SchedulerEvent{{Kind: "sched_wakeup", Timestamp: 10, WakeTID: 7}, {Kind: "sched_switch", Timestamp: 30, NextTID: 7}, {Kind: "sched_switch", Timestamp: 50, PreviousTID: 7, PreviousState: "S"}}
	ev, err := SchedulerEvidence(proc, 7, "linux_monotonic", events, 0, true)
	if err != nil || len(ev.Intervals) != 2 {
		t.Fatalf("intervals: %+v %v", ev, err)
	}
	if err := ev.Validate(proc, 7); err == nil {
		t.Fatal("unquantified loss admitted")
	}
	events[1].Timestamp = 9
	if _, err := SchedulerEvidence(proc, 7, "linux_monotonic", events, 0, false); err == nil {
		t.Fatal("clock reversal accepted")
	}
}
