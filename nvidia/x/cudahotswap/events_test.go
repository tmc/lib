package cudahotswap

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
	"unsafe"
)

func TestReportEventTimingCapacityBudget(t *testing.T) {
	const limit = 1 << 20
	for _, arm := range []string{"timing", "baseline", "both"} {
		t.Run(arm, func(t *testing.T) {
			r, err := newReportEvents(reportEventLimits{ring: 1, queue: 2, subscribers: 1, bytes: limit})
			if err != nil {
				t.Fatal(err)
			}
			state := eventState()
			report := &Report{Outcome: OutcomeOK}
			n := 130900
			if arm == "both" {
				n /= 2
			}
			if arm != "baseline" {
				report.Timing = &TimingReport{SampleNanos: make([]int64, n)}
			}
			if arm != "timing" {
				report.Baseline = &TimingReport{SampleNanos: make([]int64, n)}
			}
			if _, err := r.append(state, report); err != nil {
				t.Fatal(err)
			}
			e := r.ring[0]
			// Count the retained Go values and capacity-backed arrays. This
			// assertion makes no allocator size-class or heap-peak claim.
			bytes := int(unsafe.Sizeof(e)) + int(unsafe.Sizeof(*e.Report)) + len(e.Name) + len(e.Label) + len(e.Report.Outcome)
			for _, timing := range []*TimingReport{e.Report.Timing, e.Report.Baseline} {
				if timing == nil {
					continue
				}
				if cap(timing.SampleNanos) != len(timing.SampleNanos) {
					t.Fatalf("clone len=%d cap=%d", len(timing.SampleNanos), cap(timing.SampleNanos))
				}
				bytes += int(unsafe.Sizeof(*timing)) + cap(timing.SampleNanos)*8 + len(timing.Source) + len(timing.InputPolicy)
			}
			if bytes > limit {
				t.Fatalf("retained values=%d budget=%d", bytes, limit)
			}
			oversized := &Report{}
			if arm == "both" {
				oversized.Timing = &TimingReport{SampleNanos: make([]int64, limit/16)}
				oversized.Baseline = &TimingReport{SampleNanos: make([]int64, limit/16)}
			} else if arm == "timing" {
				oversized.Timing = &TimingReport{SampleNanos: make([]int64, limit/8)}
			} else {
				oversized.Baseline = &TimingReport{SampleNanos: make([]int64, limit/8)}
			}
			if _, err := r.append(state, oversized); err == nil || r.sequence != 1 {
				t.Fatal("oversized snapshot admitted or changed the sequence")
			}
		})
	}
}

func TestReportEventClonePreservesNilSlices(t *testing.T) {
	r := &Report{Timing: &TimingReport{}, Baseline: &TimingReport{}, Correctness: &CorrectnessReport{}}
	c := cloneEventReport(r)
	if !reflect.DeepEqual(r, c) {
		t.Fatalf("clone changed nil slices: got %+v want %+v", c, r)
	}
}

func testReportEvents(t *testing.T) *reportEvents {
	t.Helper()
	r, err := newReportEvents(reportEventLimits{ring: 2, queue: 3, subscribers: 4, bytes: 4096})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func eventState() State {
	return State{Name: "kernel", Kind: "kernel", Generation: 7, Version: 3, ConfigRevision: 11, Label: "caller label"}
}

func appendReportEvent(t *testing.T, r *reportEvents) uint64 {
	t.Helper()
	n, err := r.append(eventState(), &Report{Task: "task", Outcome: OutcomeWrongAnswer})
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func subscribeReports(t *testing.T, r *reportEvents, since uint64) *reportSubscription {
	t.Helper()
	s, err := r.subscribe(context.Background(), since)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.cancel(context.Canceled) })
	return s
}

func TestReportEventsDropResumeExplicitGap(t *testing.T) {
	r := testReportEvents(t)
	s := subscribeReports(t, r, 0)
	for i := 0; i < 6; i++ {
		if n := appendReportEvent(t, r); n != uint64(i+1) {
			t.Fatalf("sequence %d", n)
		}
	}
	if !errors.Is(s.err(), errSlowSubscriber) {
		t.Fatal("slow subscriber was not explicitly dropped")
	}
	var original []uint64
	for e := range s.events {
		original = append(original, e.Sequence)
	}
	if !reflect.DeepEqual(original, []uint64{1, 2, 3}) {
		t.Fatalf("queued reports = %v", original)
	}
	if len(r.subs) != 0 || r.count != 2 || len(r.ring) != 2 {
		t.Fatal("retention not bounded")
	}
	resumed := subscribeReports(t, r, 3)
	gap := <-resumed.events
	if gap.Sequence != 4 || gap.Report != nil || gap.Gap == nil || *gap.Gap != (Gap{First: 4, Last: 4}) {
		t.Fatalf("lost range not explicit: %+v", gap)
	}
	for _, want := range []uint64{5, 6} {
		e := <-resumed.events
		if e.Sequence != want || e.Gap != nil || e.Report == nil {
			t.Fatalf("resume event %+v", e)
		}
	}
}

func TestReportEventsReplayNoFalseGap(t *testing.T) {
	r := testReportEvents(t)
	empty := subscribeReports(t, r, 0)
	if len(empty.events) != 0 {
		t.Fatal("empty history invented event")
	}
	empty.cancel(context.Canceled)
	appendReportEvent(t, r)
	appendReportEvent(t, r)
	s := subscribeReports(t, r, 0)
	for _, want := range []uint64{1, 2} {
		e := <-s.events
		if e.Gap != nil || e.Sequence != want {
			t.Fatalf("unexpected gap or sequence: %+v", e)
		}
	}
	s.cancel(context.Canceled)
	appendReportEvent(t, r)
	s = subscribeReports(t, r, 1)
	for _, want := range []uint64{2, 3} {
		e := <-s.events
		if e.Gap != nil || e.Sequence != want {
			t.Fatalf("retained report lost: %+v", e)
		}
	}
	current := subscribeReports(t, r, 3)
	if len(current.events) != 0 {
		t.Fatal("current cursor replays old reports")
	}
	appendReportEvent(t, r)
	if e := <-current.events; e.Sequence != 4 || e.Gap != nil {
		t.Fatalf("live report: %+v", e)
	}
}

func TestReportEventsDetachedNestedSnapshotsAndStamps(t *testing.T) {
	r := testReportEvents(t)
	a := subscribeReports(t, r, 0)
	b := subscribeReports(t, r, 0)
	report := &Report{Task: "original", Outcome: OutcomeWrongAnswer, Device: &DeviceReport{Name: "device"}, Compile: &CompileReport{Log: "log"}, Correctness: &CorrectnessReport{Trials: []TrialReport{{Comparison: Comparison{Worst: &Mismatch{Index: 7}}}}}, Timing: &TimingReport{SampleNanos: []int64{10}}, Baseline: &TimingReport{SampleNanos: []int64{20}}, Speedup: &SpeedupReport{Note: "note"}, Unmeasured: []string{"absent"}}
	state := eventState()
	if _, err := r.append(state, report); err != nil {
		t.Fatal(err)
	}
	mutate := func(x *Report) {
		x.Task = "changed"
		x.Device.Name = "changed"
		x.Compile.Log = "changed"
		x.Correctness.Trials[0].Comparison.Worst.Index = 99
		x.Timing.SampleNanos[0] = 99
		x.Baseline.SampleNanos[0] = 99
		x.Speedup.Note = "changed"
		x.Unmeasured[0] = "changed"
	}
	mutate(report)
	first := <-a.events
	if first.Name != state.Name || first.Generation != 7 || first.Version != 3 || first.ConfigRevision != 11 || first.Label != state.Label {
		t.Fatalf("incorrect stamps: %+v", first)
	}
	mutate(first.Report)
	first.Generation = 100
	check := func(e Event) {
		t.Helper()
		x := e.Report
		if x.Task != "original" || x.Device.Name != "device" || x.Compile.Log != "log" || x.Correctness.Trials[0].Comparison.Worst.Index != 7 || x.Timing.SampleNanos[0] != 10 || x.Baseline.SampleNanos[0] != 20 || x.Speedup.Note != "note" || x.Unmeasured[0] != "absent" || e.Generation != 7 {
			t.Fatalf("snapshot changed: %+v", e)
		}
	}
	check(<-b.events)
	replayed := subscribeReports(t, r, 0)
	check(<-replayed.events)
}

func TestReportEventsAdmissionBoundsAndOverflow(t *testing.T) {
	for _, l := range []reportEventLimits{{}, {ring: 0, queue: 3, subscribers: 1, bytes: 4096}, {ring: 2, queue: 2, subscribers: 1, bytes: 4096}, {ring: 2, queue: 3, subscribers: 0, bytes: 4096}, {ring: 1025, queue: 1026, subscribers: 1, bytes: 4096}, {ring: 2, queue: 2049, subscribers: 1, bytes: 4096}, {ring: 2, queue: 3, subscribers: 257, bytes: 4096}, {ring: 2, queue: 3, subscribers: 1, bytes: 1<<20 + 1}} {
		if _, err := newReportEvents(l); err == nil {
			t.Fatalf("invalid limits admitted: %+v", l)
		}
	}
	r := testReportEvents(t)
	if _, err := r.subscribe(context.Background(), 1); err == nil {
		t.Fatal("future cursor accepted")
	}
	if _, err := r.subscribe(nil, 0); err == nil {
		t.Fatal("nil context accepted")
	}
	if _, err := r.append(State{}, &Report{}); err == nil {
		t.Fatal("unstamped report accepted")
	}
	if _, err := r.append(eventState(), nil); err == nil {
		t.Fatal("missing report accepted")
	}
	for _, report := range []*Report{{Reason: strings.Repeat("x", 4096)}, {Correctness: &CorrectnessReport{Trials: make([]TrialReport, 40)}}, {Timing: &TimingReport{SampleNanos: make([]int64, 600)}}, {Unmeasured: make([]string, 300)}, {Outcome: Outcome(strings.Repeat("x", 4096))}} {
		if _, err := r.append(eventState(), report); err == nil {
			t.Fatal("oversized report accepted")
		}
	}
	if r.sequence != 0 || r.count != 0 {
		t.Fatal("failed admission consumed sequence")
	}
	r.sequence = math.MaxUint64
	if _, err := r.append(eventState(), &Report{}); err == nil || r.sequence != math.MaxUint64 || r.count != 0 {
		t.Fatal("overflow admitted or state changed")
	}
	s := subscribeReports(t, r, math.MaxUint64)
	if len(s.events) != 0 {
		t.Fatal("max cursor invented replay")
	}
}

func TestReportEventsCancellationAndSubscriberLimit(t *testing.T) {
	r := testReportEvents(t)
	r.limits.subscribers = 1
	ctx, cancel := context.WithCancel(context.Background())
	s, err := r.subscribe(ctx, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.subscribe(context.Background(), 0); err == nil {
		t.Fatal("subscriber cap ignored")
	}
	cancel()
	<-s.done
	if !errors.Is(s.err(), context.Canceled) {
		t.Fatal("cancellation cause missing")
	}
	if _, ok := <-s.events; ok {
		t.Fatal("canceled channel not closed")
	}
	if _, err := r.subscribe(ctx, 0); !errors.Is(err, context.Canceled) {
		t.Fatal("canceled subscribe admitted")
	}
	next := subscribeReports(t, r, 0)
	if next.err() != nil {
		t.Fatal("new subscriber inherits cancellation")
	}
}

func TestReportEventsConcurrentPublishAndSubscribe(t *testing.T) {
	r := testReportEvents(t)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 30; j++ {
				if _, err := r.append(eventState(), &Report{}); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 30; i++ {
			r.mu.Lock()
			since := r.sequence
			r.mu.Unlock()
			s, err := r.subscribe(context.Background(), since)
			if err != nil {
				t.Error(err)
				return
			}
			s.cancel(context.Canceled)
		}
	}()
	wg.Wait()
	if r.sequence != 120 || r.count != 2 {
		t.Fatalf("sequence/count %d/%d", r.sequence, r.count)
	}
}

func ExampleEvent() {
	e := Event{Sequence: 9, Name: "kernel", Generation: 7, ConfigRevision: 11, Report: &Report{Outcome: OutcomeWrongAnswer}}
	fmt.Println(e.Sequence, e.Name, e.Generation, e.ConfigRevision, e.Report.Outcome)
	// Output: 9 kernel 7 11 wrong_answer
}

func ExampleGap() {
	e := Event{Sequence: 4, Gap: &Gap{First: 2, Last: 4}}
	fmt.Println(e.Gap.First, e.Gap.Last, e.Sequence)
	// Output: 2 4 4
}
