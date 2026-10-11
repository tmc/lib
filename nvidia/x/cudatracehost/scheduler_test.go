package cudatracehost

import "testing"

func TestScheduler(t *testing.T) {
	e, err := ParseScheduler(" target-42 [003] .... 12.25: sched_switch: prev_comm=target prev_pid=42 prev_prio=120 prev_state=R ==> next_comm=idle next_pid=0 next_prio=120")
	if err != nil || e.CPU != 3 || e.Timestamp != 12250000000 || e.PreviousTID != 42 || e.NextTID != 0 || e.PrevState != "R" {
		t.Fatalf("%+v %v", e, err)
	}
	if _, err := ParseScheduler(" target-42 [003] .... 12.25: sched_wakeup: comm=target pid=oops prio=120 target_cpu=003"); err == nil {
		t.Fatal("accepted malformed tid")
	}
}
