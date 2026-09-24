package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func kernelRecord(name string, start, end uint64) cuptiactivity.Record {
	return cuptiactivity.Record{
		Kind:     cupti.ActivityKindConcurrentKernel,
		Name:     name,
		Start:    start,
		End:      end,
		Complete: start != 0 && end != 0 && end >= start,
	}
}

// The sum of kernel intervals and the union of them are different quantities,
// and only the union is in the same units as the span. These cases pin that
// difference: a summary that reported the sum as occupancy would report a
// device busier than the span it ran in, and an idle time below zero.
func TestSummarizeActivityIntervals(t *testing.T) {
	tests := []struct {
		name                            string
		records                         []cuptiactivity.Record
		start, end                      uint64
		wantLaunches                    int
		wantSum, wantUnion, wantOverlap int64
		wantIdle                        int64
		wantStatus                      string
	}{
		{
			name:    "serial launches",
			records: []cuptiactivity.Record{kernelRecord("a", 100, 200), kernelRecord("a", 300, 400)},
			start:   50, end: 500,
			wantLaunches: 2, wantSum: 200, wantUnion: 200, wantOverlap: 0, wantIdle: 250,
			wantStatus: statusCollected,
		},
		{
			// Overlapping launches make the sum exceed the union by exactly the
			// concurrency, which is the number that says the sum is not elapsed
			// time. Idle is computed from the union, so it stays a real remainder.
			name:    "overlapping launches",
			records: []cuptiactivity.Record{kernelRecord("a", 100, 300), kernelRecord("b", 200, 400)},
			start:   100, end: 500,
			wantLaunches: 2, wantSum: 400, wantUnion: 300, wantOverlap: 100, wantIdle: 100,
			wantStatus: statusCollected,
		},
		{
			name:    "nested launch",
			records: []cuptiactivity.Record{kernelRecord("a", 100, 400), kernelRecord("b", 150, 200)},
			start:   100, end: 400,
			wantLaunches: 2, wantSum: 350, wantUnion: 300, wantOverlap: 50, wantIdle: 0,
			wantStatus: statusCollected,
		},
		{
			name:    "abutting launches merge",
			records: []cuptiactivity.Record{kernelRecord("a", 100, 200), kernelRecord("a", 200, 250)},
			start:   100, end: 250,
			wantLaunches: 2, wantSum: 150, wantUnion: 150, wantOverlap: 0, wantIdle: 0,
			wantStatus: statusCollected,
		},
		{
			// A record without a usable interval is still a launch and still
			// contributes no duration. Counting it as either alone would be wrong.
			name:    "incomplete record contributes no duration",
			records: []cuptiactivity.Record{kernelRecord("a", 100, 200), {Kind: cupti.ActivityKindConcurrentKernel, Name: "a"}},
			start:   100, end: 300,
			wantLaunches: 1, wantSum: 100, wantUnion: 100, wantOverlap: 0, wantIdle: 100,
			wantStatus: statusCollected,
		},
		{
			// Records outside the bracket that is supposed to contain them mean
			// the span and the timestamps are not the same interval or clock.
			// That is a failure, not an idle time of zero.
			name:    "occupancy exceeding the span fails",
			records: []cuptiactivity.Record{kernelRecord("a", 100, 900)},
			start:   100, end: 300,
			wantLaunches: 1, wantSum: 800, wantUnion: 800, wantOverlap: 0, wantIdle: 0,
			wantStatus: statusFailed,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarizeActivity(cuptiactivity.Result{Records: tt.records, Closed: true}, tt.start, tt.end)
			if got.KernelLaunches != tt.wantLaunches {
				t.Errorf("KernelLaunches = %d, want %d", got.KernelLaunches, tt.wantLaunches)
			}
			if got.BusySumNanos != tt.wantSum {
				t.Errorf("BusySumNanos = %d, want %d", got.BusySumNanos, tt.wantSum)
			}
			if got.OccupiedNanos != tt.wantUnion {
				t.Errorf("OccupiedNanos = %d, want %d", got.OccupiedNanos, tt.wantUnion)
			}
			if got.OverlapNanos != tt.wantOverlap {
				t.Errorf("OverlapNanos = %d, want %d", got.OverlapNanos, tt.wantOverlap)
			}
			if got.IdleNanos != tt.wantIdle {
				t.Errorf("IdleNanos = %d, want %d", got.IdleNanos, tt.wantIdle)
			}
			if got.Status != tt.wantStatus {
				t.Errorf("Status = %q, want %q", got.Status, tt.wantStatus)
			}
			if got.SMOccupancy != "UNMEASURED" {
				t.Errorf("SMOccupancy = %q, want UNMEASURED", got.SMOccupancy)
			}
		})
	}
}

// A loss or an empty collection must not read as a measurement. The failure
// shape this guards is the one where a zero from a tier that collected nothing
// is printed in the same place, and the same shape, as a zero that was observed.
func TestSummarizeActivityFailsClosed(t *testing.T) {
	ok := []cuptiactivity.Record{kernelRecord("a", 100, 200)}
	tests := []struct {
		name   string
		result cuptiactivity.Result
		want   string
	}{
		{name: "clean", result: cuptiactivity.Result{Records: ok, Closed: true}, want: statusCollected},
		{name: "unclosed session", result: cuptiactivity.Result{Records: ok}, want: statusFailed},
		{name: "callback error", result: cuptiactivity.Result{Records: ok, Closed: true, CallbackErrors: 1}, want: statusFailed},
		{name: "dropped records", result: cuptiactivity.Result{Records: ok, Closed: true, DroppedRecords: 3}, want: statusLost},
		{name: "refused buffers", result: cuptiactivity.Result{Records: ok, Closed: true, RefusedBuffers: 1}, want: statusLost},
		{name: "discarded records", result: cuptiactivity.Result{Records: ok, Closed: true, DiscardedRecords: 9}, want: statusLost},
		{name: "pending buffers", result: cuptiactivity.Result{Records: ok, Closed: true, PendingBuffers: 1}, want: statusLost},
		{name: "no records", result: cuptiactivity.Result{Closed: true}, want: statusEmpty},
		{
			name:   "records but no timed kernel",
			result: cuptiactivity.Result{Records: []cuptiactivity.Record{{Kind: cupti.ActivityKindConcurrentKernel}}, Closed: true},
			want:   statusEmpty,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarizeActivity(tt.result, 0, 1000)
			if got.Status != tt.want {
				t.Fatalf("Status = %q, want %q", got.Status, tt.want)
			}
			if tt.want == statusCollected {
				if got.problem() != "" {
					t.Errorf("problem() = %q, want empty", got.problem())
				}
				return
			}
			// Every refusal must say why in the text the capture fails with;
			// the capture directory is never written, so this string is the
			// only place the counters can be read.
			if got.problem() == "" {
				t.Error("problem() is empty for a status that must not be read")
			}
		})
	}
}

// The control. Launch count is the one quantity here with a ground truth, so it
// is what the tier is gated on: the test launches a known number of kernels and
// the activity tier must report exactly that many, not merely more than zero.
//
// The second arm is the near-miss half. A collector that reported a constant,
// or that reported the first arm's count for every run, passes an equality
// check against a single number; it cannot pass two different numbers.
func TestActivityLaunchCount(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	cm, err := nvrtcx.Compile(`extern "C" __global__ void bump(int *out) { out[0] += 1; }`,
		nvrtcx.WithName("activity_control.cu"),
		// A real architecture, not a virtual one: the control loads a CUBIN,
		// and a virtual target compiles to PTX with no CUBIN to load.
		nvrtcx.WithFlag(fmt.Sprintf("--gpu-architecture=sm_%d%d", major, minor)), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("compile control kernel: %v", err)
	}
	mod, _, err := cm.LoadOwnedOutput(dev, nvrtcx.OutputCUBIN)
	if err != nil {
		t.Fatalf("load control kernel: %v", err)
	}
	defer func() {
		if err := mod.Close(); err != nil {
			t.Error(err)
		}
	}()
	fn, err := mod.Function("bump")
	if err != nil {
		t.Fatal(err)
	}
	var p driver.DevicePtr
	if err := dev.WithContext(func() (err error) { p, err = driver.MemAlloc(4); return err }); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.WithContext(p.Free); err != nil {
			t.Error(err)
		}
	}()

	// Two arms with different counts. Both must be exact.
	for _, want := range []int{3, 7} {
		launch := func(context.Context) {
			for i := 0; i < want; i++ {
				if err := fn.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&p)}); err != nil {
					t.Errorf("launch %d: %v", i, err)
					return
				}
			}
			// Close flushes activity buffers; it does not wait for the GPU.
			// Without this the records for the tail of the launches are not
			// guaranteed to exist yet, and the tier would be gated on a race.
			if err := dev.WithContext(driver.CtxSynchronize); err != nil {
				t.Errorf("synchronize: %v", err)
			}
		}
		res, err := withActivity(context.Background(), launch)
		if err != nil {
			// A host whose CUPTI this build is not generated against, or one
			// where another profiler owns the activity APIs, cannot run the
			// control at all. That is an unvalidated tier, not a passing one.
			if res != nil && res.Status == statusFailed && res.KernelLaunches == 0 && res.TotalRecords == 0 {
				t.Skipf("cupti activity collection unavailable on this host, control not executed: %v", err)
			}
			t.Fatalf("withActivity: %v", err)
		}
		if res.KernelLaunches != want {
			t.Fatalf("KernelLaunches = %d, want exactly %d (total records %d, dropped %d, discarded %d, refused buffers %d)",
				res.KernelLaunches, want, res.TotalRecords, res.DroppedRecords, res.DiscardedRecords, res.RefusedBuffers)
		}
		if len(res.Kernels) != 1 || res.Kernels[0].Name != "bump" || res.Kernels[0].Launches != want {
			t.Fatalf("per-kernel rows = %+v, want one row naming bump with %d launches", res.Kernels, want)
		}
		if res.OccupiedNanos <= 0 || res.OccupiedNanos > res.SpanNanos {
			t.Fatalf("occupied %dns is not inside the %dns span", res.OccupiedNanos, res.SpanNanos)
		}
		t.Logf("executed control: %s", res)
	}
}

// TestUnrequestedKindRefusal covers the message a -activity run gets on a host
// whose CUPTI is not the one this build was generated against. The real
// failure on CUDA 13.4 is a Memcpy record size, and this tier requests only
// concurrent kernels, so without the note the flag fails over a record type
// the user never asked for.
func TestUnrequestedKindRefusal(t *testing.T) {
	kernels := []cupti.ActivityKind{cupti.ActivityKindConcurrentKernel}
	memcpy := &cuptiactivity.LayoutError{
		Kind: cupti.ActivityKindMemcpy, NativeSize: 112, GoSize: 96,
	}

	note := unrequestedKindRefusal(memcpy, kernels)
	for _, want := range []string{
		"ActivityKindMemcpy is not among the kinds this tier collects",
		"ActivityKindConcurrentKernel",
		"112", "96", "LD_LIBRARY_PATH",
	} {
		if !strings.Contains(note, want) {
			t.Errorf("note omits %q: %s", want, note)
		}
	}

	// The near-miss half. A refusal of a kind this tier did request is about
	// what it collects, and explaining it away would be worse than silence.
	kernel := &cuptiactivity.LayoutError{
		Kind: cupti.ActivityKindConcurrentKernel, NativeSize: 999, GoSize: 224,
	}
	if note := unrequestedKindRefusal(kernel, kernels); note != "" {
		t.Errorf("explained away a refusal of a requested kind: %s", note)
	}

	// withActivity does not hand this function the error Start returned: it
	// joins Close's error onto it and then wraps the pair. A type assertion
	// would see *errors.joinError and report nothing, silently dropping the
	// note on exactly the path that produces it, so the traversal is the thing
	// under test here -- not the formatting above.
	wrapped := fmt.Errorf("start activity collection: %w",
		errors.Join(error(memcpy), errors.New("close: device busy")))
	if note := unrequestedKindRefusal(wrapped, kernels); note == "" {
		t.Error("note lost through the join-and-wrap withActivity actually performs")
	}
	for _, err := range []error{nil, errors.New("cuptiactivity: unqualified CUPTI record version 120600 (want 13.x)")} {
		if note := unrequestedKindRefusal(err, kernels); note != "" {
			t.Errorf("unrequestedKindRefusal(%v) = %q, want empty", err, note)
		}
	}
}
