package cudadiag

import (
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func TestTimeAccounting_Separation(t *testing.T) {
	wall := tracecontract.Interval{Start: 1000, End: 5000, Clock: "monotonic"}

	// Two kernels running concurrently on different streams:
	// Kernel 1: [2000, 3000] (duration 1000ns)
	// Kernel 2: [2500, 3500] (duration 1000ns)
	// Summed = 2000ns, Union Busy = [2000, 3500] = 1500ns
	kernels := []KernelEvent{
		{
			Span:      tracecontract.Interval{Start: 2000, End: 3000, Clock: "monotonic"},
			StreamID:  1,
			ContextID: 1,
		},
		{
			Span:      tracecontract.Interval{Start: 2500, End: 3500, Clock: "monotonic"},
			StreamID:  2,
			ContextID: 1,
		},
	}

	hostCalls := []HostCallEvent{
		{
			Span:      tracecontract.Interval{Start: 1200, End: 1800, Clock: "monotonic"},
			Name:      "cudaLaunchKernel",
			HasReturn: true,
		},
	}

	proc := cudaworkload.ProcessGenID{PID: 100, Generation: 1}
	tid := uint64(101)
	sched := SchedEvidence{
		Process:      proc,
		TID:          tid,
		ProbePresent: true,
		Intervals: []SchedInterval{
			{
				State: SchedRunning,
				Span:  tracecontract.Interval{Start: 1200, End: 1500, Clock: "monotonic"},
			},
			{
				State: SchedRunnable,
				Span:  tracecontract.Interval{Start: 1500, End: 1700, Clock: "monotonic"},
			},
			{
				State: SchedBlocked,
				Span:  tracecontract.Interval{Start: 1700, End: 1800, Clock: "monotonic"},
			},
		},
	}

	ta := ComputeTimeAccounting(wall, hostCalls, kernels, sched)

	if ta.WallInterval.Duration() != 4000 {
		t.Fatalf("wall duration got %d, want 4000", ta.WallInterval.Duration())
	}
	if ta.SummedKernelDuration != 2000*time.Nanosecond {
		t.Fatalf("summed kernel duration got %v, want 2000ns", ta.SummedKernelDuration)
	}
	if ta.BusyDuration != 1500*time.Nanosecond {
		t.Fatalf("busy duration got %v, want 1500ns", ta.BusyDuration)
	}
	if ta.HostAPIDuration != 600*time.Nanosecond {
		t.Fatalf("host API duration got %v, want 600ns", ta.HostAPIDuration)
	}
	if ta.ScheduledDuration != 300*time.Nanosecond {
		t.Fatalf("scheduled duration got %v, want 300ns", ta.ScheduledDuration)
	}
	if ta.RunnableDuration != 200*time.Nanosecond {
		t.Fatalf("runnable duration got %v, want 200ns", ta.RunnableDuration)
	}
	if ta.WaitDuration != 100*time.Nanosecond {
		t.Fatalf("wait duration got %v, want 100ns", ta.WaitDuration)
	}

	// Invariant: BusyDuration <= SummedKernelDuration
	if ta.BusyDuration > ta.SummedKernelDuration {
		t.Fatalf("invariant violation: busy %v > summed %v", ta.BusyDuration, ta.SummedKernelDuration)
	}
	// Invariant: BusyDuration <= WallInterval.Duration()
	if ta.BusyDuration.Nanoseconds() > ta.WallInterval.Duration() {
		t.Fatalf("invariant violation: busy %v > wall %d", ta.BusyDuration, ta.WallInterval.Duration())
	}
}

func TestUnionDuration_Concurrency(t *testing.T) {
	intervals := []tracecontract.Interval{
		{Start: 100, End: 200},
		{Start: 150, End: 250}, // overlapping
		{Start: 300, End: 400}, // disjoint
		{Start: 400, End: 500}, // contiguous
		{Start: 50, End: 80},   // earlier disjoint
	}

	// Non-overlapping spans:
	// [50, 80] = 30
	// [100, 250] = 150
	// [300, 500] = 200
	// Total union duration = 30 + 150 + 200 = 380ns
	// Sum of individual = 100 + 100 + 100 + 100 + 30 = 430ns

	unionDur := UnionDuration(intervals)
	sumDur := SumDuration(intervals)

	if unionDur != 380*time.Nanosecond {
		t.Fatalf("union duration got %v, want 380ns", unionDur)
	}
	if sumDur != 430*time.Nanosecond {
		t.Fatalf("sum duration got %v, want 430ns", sumDur)
	}
}

func TestComputeIntervalUnion_EmptyAndInverted(t *testing.T) {
	if res := ComputeIntervalUnion(nil); res != nil {
		t.Fatalf("nil intervals got %v, want nil", res)
	}

	inverted := []tracecontract.Interval{
		{Start: 500, End: 400}, // invalid inverted interval
	}
	if res := ComputeIntervalUnion(inverted); len(res) != 0 {
		t.Fatalf("inverted intervals got %v, want empty", res)
	}
	if dur := UnionDuration(inverted); dur != 0 {
		t.Fatalf("inverted duration got %v, want 0", dur)
	}
}
