package gaps

import (
	"encoding/json"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/internal/journaltest"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// The NVTX marker flags of an activity record.
const (
	markerFlagInstantaneous = 1 << 0
	markerFlagStart         = 1 << 1
	markerFlagEnd           = 1 << 2
)

// uniformJournal is a five-kernel run with uniform 1000 ns gaps. Every launch
// returns before the previous kernel ends, so every gap is Host Ahead.
func uniformJournal() []byte {
	b := journaltest.New()
	start := uint64(10_000)
	for i := uint32(1); i <= 5; i++ {
		b.Kernel(start, start+4_000, i)
		b.API(start-3_000, start-2_000, i)
		start += 5_000
	}
	return b.Finalize()
}

func TestMergeIntervals(t *testing.T) {
	act := func(start, end int64) timeline.Activity { return timeline.Activity{Start: start, End: end} }
	tests := []struct {
		name string
		in   []timeline.Activity
		want []mergedInterval
	}{
		{
			name: "disjoint",
			in:   []timeline.Activity{act(0, 10), act(20, 30)},
			want: []mergedInterval{{start: 0, end: 10}, {start: 20, end: 30}},
		},
		{
			name: "overlapping across streams",
			in:   []timeline.Activity{act(0, 30), act(10, 20)},
			want: []mergedInterval{{start: 0, end: 30}},
		},
		{
			name: "touching intervals merge",
			in:   []timeline.Activity{act(0, 10), act(10, 20)},
			want: []mergedInterval{{start: 0, end: 20}},
		},
		{
			name: "later end extends running interval",
			in:   []timeline.Activity{act(0, 10), act(5, 25), act(30, 40)},
			want: []mergedInterval{{start: 0, end: 25}, {start: 30, end: 40}},
		},
		{
			name: "empty",
			in:   nil,
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mergeIntervals(tt.in)
			if len(got) != len(tt.want) {
				t.Fatalf("intervals = %d, want %d (%+v)", len(got), len(tt.want), got)
			}
			for i := range got {
				if got[i].start != tt.want[i].start || got[i].end != tt.want[i].end {
					t.Errorf("interval %d = [%d,%d), want [%d,%d)", i, got[i].start, got[i].end, tt.want[i].start, tt.want[i].end)
				}
			}
		})
	}
}

func TestPercentileNanos(t *testing.T) {
	sorted := []int64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	tests := []struct {
		name string
		p    float64
		want int64
	}{
		{"min", 0, 10},
		{"p25", 25, 30},
		{"median", 50, 50},
		{"p75", 75, 70},
		{"p90", 90, 90},
		{"p99", 99, 90},
		{"max", 100, 100},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := percentileNanos(sorted, tt.p); got != tt.want {
				t.Errorf("percentileNanos(%g) = %d, want %d", tt.p, got, tt.want)
			}
		})
	}
	if got := percentileNanos(nil, 50); got != 0 {
		t.Errorf("percentileNanos(nil) = %d, want 0", got)
	}
	if got := percentileNanos([]int64{7}, 99); got != 7 {
		t.Errorf("single sample p99 = %d, want 7", got)
	}
}

func TestDescribeGaps(t *testing.T) {
	got := describeGaps([]int64{100, 1, 2, 3, 4})
	want := Distribution{Min: 1, P25: 2, Median: 3, P75: 4, P90: 4, P99: 4, Max: 100, Mean: 22}
	if got != want {
		t.Errorf("describeGaps = %+v, want %+v", got, want)
	}
	// The mean is 22 ns and the median is 3 ns: one long stall moves the mean
	// past every gap that actually occurred, and the median does not follow.
	if got.Mean <= float64(got.Max)/10 {
		t.Errorf("mean %g should be dragged by the outlier", got.Mean)
	}
}

func TestHostClassification(t *testing.T) {
	// Two merged intervals: [0,100) and [200,300), so one 100 ns gap whose
	// class depends on where the second launch call sits relative to 100.
	acts := []timeline.Activity{{Start: 0, End: 100, Correlation: 1}, {Start: 200, End: 300, Correlation: 2}}
	tests := []struct {
		name         string
		api          *timeline.Span
		wantClass    string
		wantIdle     int64
		wantBehind   int64
		wantDriver   int64
		wantHardware int64
		wantUnattrib int64
	}{
		{
			name:         "host ahead",
			api:          &timeline.Span{Start: 10, End: 50},
			wantClass:    ClassHostAhead,
			wantIdle:     100,
			wantHardware: 100,
		},
		{
			name:         "host during",
			api:          &timeline.Span{Start: 60, End: 140},
			wantClass:    ClassHostDuring,
			wantIdle:     100,
			wantDriver:   40,
			wantHardware: 60,
		},
		{
			name:         "host behind",
			api:          &timeline.Span{Start: 130, End: 170},
			wantClass:    ClassHostBehind,
			wantIdle:     100,
			wantBehind:   30,
			wantDriver:   40,
			wantHardware: 30,
		},
		{
			// Not hardware queue. A gap whose next activity correlated to
			// no API record has no known mechanism, and "hardware queue
			// latency" is a mechanism a reader acts on.
			name:         "unmatched correlation",
			api:          nil,
			wantClass:    ClassUnmatched,
			wantIdle:     100,
			wantUnattrib: 100,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			apis := map[uint32]timeline.Span{}
			if tt.api != nil {
				apis[2] = *tt.api
			}
			s := computeTestStats(acts, apis, "overall", gapSpans{})
			if s.GapCount != 1 || s.IdleNanos != tt.wantIdle {
				t.Fatalf("gaps = %d idle = %d, want 1 gap of %d", s.GapCount, s.IdleNanos, tt.wantIdle)
			}
			for _, c := range s.Concurrency {
				wantCount, wantIdle := 0, int64(0)
				if c.Name == tt.wantClass {
					wantCount, wantIdle = 1, tt.wantIdle
				}
				if c.Count != wantCount || c.IdleNanos != wantIdle {
					t.Errorf("class %s = count %d idle %d, want count %d idle %d", c.Name, c.Count, c.IdleNanos, wantCount, wantIdle)
				}
			}
			a := s.Attribution
			if a.HostBehindNanos != tt.wantBehind || a.DriverSubmitNanos != tt.wantDriver || a.HardwareNanos != tt.wantHardware || a.UnattributedNanos != tt.wantUnattrib {
				t.Errorf("attribution = %+v, want behind %d driver %d hardware %d unattributed %d", a, tt.wantBehind, tt.wantDriver, tt.wantHardware, tt.wantUnattrib)
			}
			if sum := a.HostBehindNanos + a.DriverSubmitNanos + a.HardwareNanos + a.UnattributedNanos; sum != s.IdleNanos {
				t.Errorf("attribution arms sum to %d, want the idle they split, %d", sum, s.IdleNanos)
			}
		})
	}
}

func TestDutyCycleAndDistribution(t *testing.T) {
	cap := journaltest.Capture(map[string][]byte{"capture.journal": uniformJournal()}, journaltest.CompleteStatus())
	report, err := Analyze(cap, Options{WarmupMS: 200, MaxRecords: timeline.MaxRecords})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Devices) != 1 || report.Devices[0].Device != 0 {
		t.Fatalf("devices = %+v, want one device 0", report.Devices)
	}
	s := report.Devices[0].Overall
	// Five 4000 ns kernels separated by four 1000 ns gaps.
	if s.Activities != 5 || s.MergedIntervals != 5 || s.GapCount != 4 {
		t.Fatalf("activities/intervals/gaps = %d/%d/%d, want 5/5/4", s.Activities, s.MergedIntervals, s.GapCount)
	}
	if s.SpanNanos != 24_000 || s.ActiveNanos != 20_000 || s.IdleNanos != 4_000 {
		t.Fatalf("span/active/idle = %d/%d/%d, want 24000/20000/4000", s.SpanNanos, s.ActiveNanos, s.IdleNanos)
	}
	if s.DutyCyclePct < 83.32 || s.DutyCyclePct > 83.34 {
		t.Errorf("duty cycle = %.4f, want 83.33", s.DutyCyclePct)
	}
	if s.Distribution.Median != 1000 || s.Distribution.Mean != 1000 || s.Distribution.Max != 1000 {
		t.Errorf("distribution = %+v, want uniform 1000 ns", s.Distribution)
	}
	ahead := classCount(s, ClassHostAhead)
	if ahead != 4 {
		t.Errorf("host ahead = %d, want 4", ahead)
	}
	if s.Attribution.HardwareNanos != 4_000 {
		t.Errorf("hardware queue = %d, want 4000", s.Attribution.HardwareNanos)
	}
	if report.Activities != 5 || report.APIRecords != 5 || report.Correlated != 100 {
		t.Errorf("report totals = %d activities, %d apis, %.1f%% correlated", report.Activities, report.APIRecords, report.Correlated)
	}
}

func classCount(s *Stats, name string) int {
	for _, c := range s.Concurrency {
		if c.Name == name {
			return c.Count
		}
	}
	return -1
}

func TestMergesStreamsPerDevice(t *testing.T) {
	// Device 0 runs two overlapping streams and device 1 runs one kernel; the
	// overlap must not count twice and the devices must not be merged.
	b := journaltest.New()
	b.Activity("kernel", 0, 1_000, 3_000, 1)
	b.Activity("kernel", 0, 2_000, 4_000, 2)
	b.Activity("kernel", 0, 6_000, 7_000, 3)
	b.Activity("memcpy", 1, 1_000, 2_000, 4)
	b.Activity("kernel", 1, 5_000, 6_000, 5)
	data := b.Finalize()

	cap := journaltest.Capture(map[string][]byte{"capture.journal": data}, journaltest.CompleteStatus())
	report, err := Analyze(cap, Options{WarmupMS: 200, MaxRecords: timeline.MaxRecords})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Devices) != 2 {
		t.Fatalf("devices = %d, want 2", len(report.Devices))
	}
	d0 := report.Devices[0].Overall
	if d0.MergedIntervals != 2 || d0.ActiveNanos != 4_000 || d0.IdleNanos != 2_000 {
		t.Errorf("device 0 = %d intervals, active %d, idle %d; want 2/4000/2000", d0.MergedIntervals, d0.ActiveNanos, d0.IdleNanos)
	}
	d1 := report.Devices[1].Overall
	if d1.MergedIntervals != 2 || d1.ActiveNanos != 2_000 || d1.IdleNanos != 3_000 {
		t.Errorf("device 1 = %d intervals, active %d, idle %d; want 2/2000/3000", d1.MergedIntervals, d1.ActiveNanos, d1.IdleNanos)
	}
}

func TestCompressedJournalMatchesPlain(t *testing.T) {
	plain := uniformJournal()
	tests := []struct {
		name string
		file string
		data []byte
	}{
		{"plain", "capture.journal", plain},
		{"gzip same name", "capture.journal", journaltest.Gzip(plain)},
		{"gzip dot gz name", "capture.journal.gz", journaltest.Gzip(plain)},
	}
	var want string
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := journaltest.Capture(map[string][]byte{tt.file: tt.data}, journaltest.CompleteStatus())
			report, err := Analyze(cap, Options{WarmupMS: 200, MaxRecords: timeline.MaxRecords})
			if err != nil {
				t.Fatal(err)
			}
			if len(report.Journals) != 1 {
				t.Fatalf("journals = %+v", report.Journals)
			}
			if got := report.Journals[0].Compressed; got != (tt.name != "plain") {
				t.Errorf("compressed = %v for %s", got, tt.name)
			}
			if report.Journals[0].Bytes != len(plain) {
				t.Errorf("decoded bytes = %d, want %d", report.Journals[0].Bytes, len(plain))
			}
			body := reportBody(report)
			if want == "" {
				want = body
			} else if body != want {
				t.Errorf("report differs from the plain journal:\n%s\nwant\n%s", body, want)
			}
		})
	}
}

// reportBody renders everything but the journal identity, so a plain and a
// compressed journal can be compared directly.
func reportBody(r *Report) string {
	data, err := json.Marshal(r.Devices)
	if err != nil {
		panic(err)
	}
	return string(data)
}

// TestIgnoresNVTXMarkers covers the tier boundary. NVTX markers share the
// activity record kind with kernels and memcpys but are host annotations, not
// device work: they carry no device time, they can be stamped anywhere on the
// host timeline, and a range's two halves land at its ends rather than across
// it.
//
// Counting one as device activity does three separate wrong things -- it
// inflates Activities, it stretches SpanNanos to the outermost marker, and it
// splits a real device gap in two at the marker's timestamp -- so the control
// is that the same fixture with and without markers reports identically.
func TestIgnoresNVTXMarkers(t *testing.T) {
	kernels := func(b *journaltest.Builder) {
		b.Activity("kernel", 0, 1_000, 3_000, 1)
		b.Activity("kernel", 0, 6_000, 7_000, 2)
	}

	plain := journaltest.New()
	kernels(plain)

	marked := journaltest.New()
	kernels(marked)
	// Well outside the device window on both sides, and squarely inside the
	// 3000..6000 gap: each placement corrupts a different statistic.
	for _, m := range []struct {
		flags uint32
		ts    uint64
	}{
		{markerFlagStart, 100},
		{markerFlagEnd, 9_000},
		{markerFlagInstantaneous, 4_500},
	} {
		marked.Append(wire.KindActivity, map[string]any{
			"kind": "activity", "subtype": "marker", "valid": true, "name": "prefill",
			"marker_domain": "ds4", "marker_id": 1, "marker_flags": m.flags,
			"process_id": 100, "thread_id": 200,
			"start": m.ts, "end": m.ts,
		})
	}

	opts := Options{WarmupMS: 200, MaxRecords: timeline.MaxRecords}
	want, err := Analyze(journaltest.Capture(map[string][]byte{"capture.journal": plain.Finalize()}, journaltest.CompleteStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Analyze(journaltest.Capture(map[string][]byte{"capture.journal": marked.Finalize()}, journaltest.CompleteStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(want.Devices) != 1 || len(got.Devices) != 1 {
		t.Fatalf("devices = %d with markers, %d without; want 1 each: a marker created a device of its own", len(got.Devices), len(want.Devices))
	}
	w, g := want.Devices[0].Overall, got.Devices[0].Overall
	if g.Activities != w.Activities {
		t.Errorf("activities = %d with markers, %d without", g.Activities, w.Activities)
	}
	if g.SpanNanos != w.SpanNanos {
		t.Errorf("span = %d with markers, %d without: a host marker moved an end of the device timeline", g.SpanNanos, w.SpanNanos)
	}
	if g.MergedIntervals != w.MergedIntervals {
		t.Errorf("merged intervals = %d with markers, %d without: a marker split a device gap", g.MergedIntervals, w.MergedIntervals)
	}
	if g.ActiveNanos != w.ActiveNanos || g.IdleNanos != w.IdleNanos {
		t.Errorf("active/idle = %d/%d with markers, %d/%d without", g.ActiveNanos, g.IdleNanos, w.ActiveNanos, w.IdleNanos)
	}
}

// TestIgnoresCaptureRangeEdges guards the capture-range half of the same
// filter. A range edge says when the producer started and stopped collecting;
// it is not device work, and counting one would inflate Activities and move
// the ends of the device timeline.
//
// The fixture marks the edges valid and gives them a span on purpose. The
// producer does not emit them that way today, and a fixture that copied it
// would be caught by the validity check above and would pass with the subtype
// filter deleted -- a test of nothing. Recording a range's extent is the
// obvious next thing to put in this record, and the filter has to be what
// holds when that happens, not the absence of a field.
func TestIgnoresCaptureRangeEdges(t *testing.T) {
	kernels := func(b *journaltest.Builder) {
		b.Activity("kernel", 0, 1_000, 3_000, 1)
		b.Activity("kernel", 0, 6_000, 7_000, 2)
	}

	plain := journaltest.New()
	kernels(plain)

	ranged := journaltest.New()
	kernels(ranged)
	// Before the device window, inside the 3000..6000 gap, and after it:
	// each placement corrupts a different statistic.
	for _, e := range []struct {
		edge  string
		start uint64
		end   uint64
	}{
		{"start", 100, 200},
		{"stop", 4_400, 4_600},
		{"start", 9_000, 9_500},
	} {
		ranged.Append(wire.KindActivity, map[string]any{
			"kind": "activity", "subtype": "capture_range", "valid": true,
			"edge": e.edge, "api": "cudaProfilerStart", "index": 0,
			"toggle_failures": 0, "start": e.start, "end": e.end,
		})
	}

	opts := Options{WarmupMS: 200, MaxRecords: timeline.MaxRecords}
	want, err := Analyze(journaltest.Capture(map[string][]byte{"capture.journal": plain.Finalize()}, journaltest.CompleteStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Analyze(journaltest.Capture(map[string][]byte{"capture.journal": ranged.Finalize()}, journaltest.CompleteStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(want.Devices) != 1 || len(got.Devices) != 1 {
		t.Fatalf("devices = %d with range edges, %d without; want 1 each", len(got.Devices), len(want.Devices))
	}
	w, g := want.Devices[0].Overall, got.Devices[0].Overall
	if g.Activities != w.Activities {
		t.Errorf("activities = %d with range edges, %d without", g.Activities, w.Activities)
	}
	if g.SpanNanos != w.SpanNanos {
		t.Errorf("span = %d with range edges, %d without: an edge moved an end of the device timeline", g.SpanNanos, w.SpanNanos)
	}
	if g.MergedIntervals != w.MergedIntervals {
		t.Errorf("merged intervals = %d with range edges, %d without: an edge split a device gap", g.MergedIntervals, w.MergedIntervals)
	}
	if g.ActiveNanos != w.ActiveNanos || g.IdleNanos != w.IdleNanos {
		t.Errorf("active/idle = %d/%d with range edges, %d/%d without", g.ActiveNanos, g.IdleNanos, w.ActiveNanos, w.IdleNanos)
	}
}

// TestSpanCoversContainedActivities pins the device span against the case
// that breaks the obvious implementation: one activity wholly inside another.
// Activities are sorted by (start, end), so the container sorts first and the
// contained one is last -- taking the last element's end yields a span shorter
// than the active time measured within it, and the report then prints a
// percentage above 100.
//
// A non-overlapping fixture cannot see this, which is why the existing span
// assertions did not: they compare two fixtures that both sort cleanly.
func TestSpanCoversContainedActivities(t *testing.T) {
	b := journaltest.New()
	b.Activity("kernel", 0, 1_000, 5_000, 1)
	b.Activity("kernel", 0, 1_200, 1_400, 2)

	opts := Options{WarmupMS: 200, MaxRecords: timeline.MaxRecords}
	got, err := Analyze(journaltest.Capture(map[string][]byte{"capture.journal": b.Finalize()}, journaltest.CompleteStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	o := got.Devices[0].Overall
	if want := int64(4_000); o.SpanNanos != want {
		t.Errorf("span = %d, want %d: the span must reach the greatest end, not the end of the last activity by start order", o.SpanNanos, want)
	}
	if o.ActiveNanos > o.SpanNanos {
		t.Errorf("active %d exceeds span %d, so the report would print more than 100%% of span", o.ActiveNanos, o.SpanNanos)
	}
	if o.ActiveNanos+o.IdleNanos != o.SpanNanos {
		t.Errorf("active+idle = %d, span = %d: the two must agree", o.ActiveNanos+o.IdleNanos, o.SpanNanos)
	}
}

// TestCountsMemsetsAsBusy pins the consequence the MEMSET tier exists for.
//
// The subtype filter in timeline.Decode is a denylist -- markers and capture
// range edges are dropped and everything else is device work -- so this passes
// the moment memsets reach the journal at all. That is exactly why it is here:
// the whole defect was upstream of any filter. CUPTI was never asked for the
// kind, the stream therefore contained no memsets, and the time one occupied
// on the device was charged to idle by arithmetic that was working correctly
// on the records it had.
//
// The numbers are asserted absolutely, not just compared to the memset-free
// run, because "different from the baseline" would also be satisfied by a
// memset that landed on the wrong device or with the wrong span.
func TestCountsMemsetsAsBusy(t *testing.T) {
	// Two kernels with one 3000 ns hole between them.
	kernels := func(b *journaltest.Builder) {
		b.Activity("kernel", 0, 1_000, 3_000, 1)
		b.Activity("kernel", 0, 6_000, 7_000, 2)
	}

	plain := journaltest.New()
	kernels(plain)

	withMemset := journaltest.New()
	kernels(withMemset)
	// Squarely inside the hole, touching neither kernel, so it splits the one
	// gap into two and removes exactly its own duration from idle.
	withMemset.Activity("memset", 0, 4_000, 5_000, 3)

	opts := Options{WarmupMS: 0, MaxRecords: timeline.MaxRecords}
	before, err := Analyze(journaltest.Capture(map[string][]byte{"capture.journal": plain.Finalize()}, journaltest.CompleteStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	after, err := Analyze(journaltest.Capture(map[string][]byte{"capture.journal": withMemset.Finalize()}, journaltest.CompleteStatus()), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(before.Devices) != 1 || len(after.Devices) != 1 {
		t.Fatalf("devices = %d without the memset, %d with; want 1 each", len(before.Devices), len(after.Devices))
	}
	b, a := before.Devices[0].Overall, after.Devices[0].Overall
	if b.IdleNanos != 3_000 {
		t.Fatalf("baseline idle = %d ns, want 3000: the fixture no longer has the hole this test is about", b.IdleNanos)
	}
	if b.ActiveNanos != 3_000 {
		t.Fatalf("baseline active = %d ns, want 3000 (2000+1000)", b.ActiveNanos)
	}
	// The memset is 1000 ns of device work that used to be reported as idle.
	if a.IdleNanos != 2_000 {
		t.Errorf("idle = %d ns with the memset, want 2000: %d ns of memset time is still charged to idle", a.IdleNanos, a.IdleNanos-2_000)
	}
	if a.ActiveNanos != 4_000 {
		t.Errorf("active = %d ns with the memset, want 4000", a.ActiveNanos)
	}
	if a.Activities != b.Activities+1 {
		t.Errorf("activities = %d with the memset, %d without: the record never reached the timeline", a.Activities, b.Activities)
	}
	// One hole became two, which is what makes the memset visible in the gap
	// report rather than merely subtracted from a total.
	if b.GapCount != 1 || a.GapCount != 2 {
		t.Errorf("gaps = %d without the memset, %d with; want 1 and 2", b.GapCount, a.GapCount)
	}
	if a.SpanNanos != b.SpanNanos {
		t.Errorf("span = %d with the memset, %d without: a contained record moved an end of the timeline", a.SpanNanos, b.SpanNanos)
	}
}

// twoKernelHole is two kernels on device 0 in context 1 with a single 3000 ns
// hole between them. Every test below adds one idle-explaining span inside
// that hole and asserts the hole is still there.
func twoKernelHole() *journaltest.Builder {
	b := journaltest.New()
	b.KernelOn(0, 1, 1_000, 3_000, 1)
	b.KernelOn(0, 1, 6_000, 7_000, 2)
	return b
}

func analyzeFixture(t *testing.T, data []byte) *Report {
	t.Helper()
	report, err := Analyze(journaltest.Capture(map[string][]byte{"capture.journal": data}, journaltest.CompleteStatus()), Options{WarmupMS: 0, MaxRecords: timeline.MaxRecords})
	if err != nil {
		t.Fatal(err)
	}
	return report
}

// TestKeepsOverheadOutOfDeviceBusy is the assertion the OVERHEAD tier
// stands or falls on, and it is an assertion that nothing changed.
//
// The subtype filter in decodeJournal is a denylist: markers and capture-range
// edges are dropped and everything else is device work. Enabling a new kind
// upstream therefore silently enrols it as device busy. For overhead that is
// not a small error, it is an inversion -- CUPTI's own cost would be counted
// as the program keeping the device busy, and the idle the tool itself caused
// would vanish from the number the tool exists to report. Nothing would fail:
// active would rise, idle would fall, and both would still sum to the span.
//
// So the test pins idle, active and the gap count against a run with no
// overhead at all, and then asserts the overlay picked the span up. Exclusion
// without attribution is half the value: a gap the tool reports should be
// labellable "this is CUPTI flushing, not your program".
func TestKeepsOverheadOutOfDeviceBusy(t *testing.T) {
	before := analyzeFixture(t, twoKernelHole().Finalize())
	with := twoKernelHole()
	// Squarely inside the hole, touching neither kernel.
	with.Overhead(4_000, 5_000, 1<<16)
	after := analyzeFixture(t, with.Finalize())

	if len(before.Devices) != 1 || len(after.Devices) != 1 {
		t.Fatalf("devices = %d without the overhead, %d with; want 1 each: an overhead carries no device and must not open a timeline", len(before.Devices), len(after.Devices))
	}
	b, a := before.Devices[0].Overall, after.Devices[0].Overall
	if b.IdleNanos != 3_000 || b.ActiveNanos != 3_000 || b.GapCount != 1 {
		t.Fatalf("baseline idle/active/gaps = %d/%d/%d, want 3000/3000/1: the fixture no longer has the hole this test is about", b.IdleNanos, b.ActiveNanos, b.GapCount)
	}
	if a.IdleNanos != b.IdleNanos {
		t.Errorf("idle = %d ns with the overhead, %d without: %d ns of CUPTI's own cost was counted as device work and subtracted from the idle it caused", a.IdleNanos, b.IdleNanos, b.IdleNanos-a.IdleNanos)
	}
	if a.ActiveNanos != b.ActiveNanos {
		t.Errorf("active = %d ns with the overhead, %d without: an overhead span entered the busy timeline", a.ActiveNanos, b.ActiveNanos)
	}
	if a.GapCount != b.GapCount {
		t.Errorf("gaps = %d with the overhead, %d without: the span split a real device gap in two", a.GapCount, b.GapCount)
	}
	if a.Activities != b.Activities {
		t.Errorf("activities = %d with the overhead, %d without: the count of device work admitted a record that is not device work", a.Activities, b.Activities)
	}
	// Excluded and attributed. Without this the gap is honest and unexplained,
	// which is where the tool already was.
	if a.Overlay.CuptiOverheadNanos != 1_000 {
		t.Errorf("overlay cupti_overhead = %d ns, want 1000: the gap cannot be labelled as the tool's own cost", a.Overlay.CuptiOverheadNanos)
	}
	if a.Overlay.CuptiOverheadGaps != 1 {
		t.Errorf("overlay cupti_overhead gap count = %d, want 1", a.Overlay.CuptiOverheadGaps)
	}
	if b.Overlay.CuptiOverheadNanos != 0 || b.Overlay.CuptiOverheadGaps != 0 {
		t.Errorf("baseline overlay = %d ns over %d gap(s), want zero: the overlay reports overhead in a run that has none", b.Overlay.CuptiOverheadNanos, b.Overlay.CuptiOverheadGaps)
	}
}

// TestKeepsSyncOutOfDeviceBusy is the same denylist trap, in its sharpest
// form. A synchronization span is the host waiting, which is *why* the device
// was idle; counting it as busy would erase precisely the idle it was added to
// explain, and the erasure is exact rather than approximate. A reader would
// see a duty cycle improve as more of the run was spent blocked.
//
// It also covers the device resolution. The record carries a context and no
// device, and the only reason this sync lands on device 0 is that a kernel
// named context 1 on device 0.
func TestKeepsSyncOutOfDeviceBusy(t *testing.T) {
	before := analyzeFixture(t, twoKernelHole().Finalize())
	with := twoKernelHole()
	with.Sync(1, 4_000, 5_500, 3, 9)
	after := analyzeFixture(t, with.Finalize())

	b, a := before.Devices[0].Overall, after.Devices[0].Overall
	if b.IdleNanos != 3_000 {
		t.Fatalf("baseline idle = %d ns, want 3000", b.IdleNanos)
	}
	if a.IdleNanos != b.IdleNanos {
		t.Errorf("idle = %d ns with the sync, %d without: the host's wait was counted as device work, erasing %d ns of the idle it explains", a.IdleNanos, b.IdleNanos, b.IdleNanos-a.IdleNanos)
	}
	if a.ActiveNanos != b.ActiveNanos {
		t.Errorf("active = %d ns with the sync, %d without", a.ActiveNanos, b.ActiveNanos)
	}
	if a.GapCount != b.GapCount {
		t.Errorf("gaps = %d with the sync, %d without", a.GapCount, b.GapCount)
	}
	if a.Overlay.SyncNanos != 1_500 {
		t.Errorf("overlay synchronization = %d ns, want 1500: the idle is reported without the reason for it", a.Overlay.SyncNanos)
	}
	if a.Overlay.SyncGaps != 1 {
		t.Errorf("overlay synchronization gap count = %d, want 1", a.Overlay.SyncGaps)
	}
	if after.UnresolvedSyncs != 0 {
		t.Errorf("unresolved syncs = %d, want 0: a kernel named context 1 on device 0, so this sync is resolvable", after.UnresolvedSyncs)
	}
}

// TestReportsSyncWithUnknownContextAsUnresolved covers the other branch of
// the resolution, which is the one a default would hide.
//
// CUpti_ActivitySynchronization has no device field. Filling one in from
// nothing means writing zero, and zero is a valid device id: on a single-GPU
// run the wrong answer and the right answer are the same number, so the defect
// never surfaces, and on a multi-GPU run one device's idle is explained by
// another device's wait. The fixture therefore runs entirely on device 1, so a
// fabricated device 0 has nowhere to hide.
func TestReportsSyncWithUnknownContextAsUnresolved(t *testing.T) {
	b := journaltest.New()
	b.KernelOn(1, 4, 1_000, 3_000, 1)
	b.KernelOn(1, 4, 6_000, 7_000, 2)
	// Context 9 belongs to no record that carries a device.
	b.Sync(9, 4_000, 5_000, 3, 9)
	report := analyzeFixture(t, b.Finalize())

	if len(report.Devices) != 1 || report.Devices[0].Device != 1 {
		var got []uint32
		for _, d := range report.Devices {
			got = append(got, d.Device)
		}
		t.Fatalf("devices = %v, want [1] only: an unresolvable sync was placed on a device timeline, and device 0 is the timeline it invents", got)
	}
	if report.UnresolvedSyncs != 1 {
		t.Errorf("unresolved syncs = %d, want 1: a sync whose context nothing named was silently accounted for somewhere", report.UnresolvedSyncs)
	}
	if report.UnresolvedSyncNanos != 1_000 {
		t.Errorf("unresolved sync nanos = %d, want 1000", report.UnresolvedSyncNanos)
	}
	s := report.Devices[0].Overall
	if s.IdleNanos != 3_000 {
		t.Errorf("idle = %d ns, want 3000: the unresolvable sync still reached a device timeline", s.IdleNanos)
	}
	if s.Overlay.SyncNanos != 0 {
		t.Errorf("overlay synchronization = %d ns on device 1, want 0: a sync that could not be attributed was attributed anyway", s.Overlay.SyncNanos)
	}
}

// TestKeepsGraphTraceOutOfDeviceBusy covers the third denylist entry.
//
// A graph replay envelope spans the kernel records of the nodes replayed
// inside it, so admitting it to the busy timeline does two things at once: it
// counts that device time twice, and -- because the envelope is contiguous
// across the scheduling gaps between nodes -- it merges over exactly the idle
// a reader would want to see. A graph would appear to run at a 100% duty
// cycle by definition.
//
// The envelope is kept as an overlay instead, where the difference between the
// span and the node time inside it is graph launch and scheduling cost that no
// per-node record holds.
func TestKeepsGraphTraceOutOfDeviceBusy(t *testing.T) {
	// Both kernels name the graph they are nodes of. That is the condition
	// the exclusion rests on, so the fixture has to state it: a capture whose
	// kernels name no graph is the opaque case below, not this one.
	nodes := func() *journaltest.Builder {
		b := journaltest.New()
		b.GraphNode(0, 1, 1_000, 3_000, 1, 5)
		b.GraphNode(0, 1, 6_000, 7_000, 2, 5)
		return b
	}
	before := analyzeFixture(t, nodes().Finalize())
	with := nodes()
	// The envelope covers both kernels and the hole between them, which is
	// what a real replay of two nodes looks like.
	with.GraphTrace(0, 900, 7_100, 5, 3)
	after := analyzeFixture(t, with.Finalize())

	b, a := before.Devices[0].Overall, after.Devices[0].Overall
	if a.IdleNanos != b.IdleNanos {
		t.Errorf("idle = %d ns with the graph trace, %d without: the replay envelope merged over the scheduling gap between its own nodes", a.IdleNanos, b.IdleNanos)
	}
	if a.ActiveNanos != b.ActiveNanos {
		t.Errorf("active = %d ns with the graph trace, %d without: the envelope's device time was counted on top of the nodes inside it", a.ActiveNanos, b.ActiveNanos)
	}
	if a.SpanNanos != b.SpanNanos {
		t.Errorf("span = %d with the graph trace, %d without: the envelope moved an end of the timeline", a.SpanNanos, b.SpanNanos)
	}
	if a.GapCount != b.GapCount {
		t.Errorf("gaps = %d with the graph trace, %d without", a.GapCount, b.GapCount)
	}
	if a.Overlay.GraphReplayNanos != 3_000 {
		t.Errorf("overlay graph_replay = %d ns, want 3000: the idle inside the replay is not attributable to graph scheduling", a.Overlay.GraphReplayNanos)
	}
	if a.Overlay.GraphReplayGaps != 1 {
		t.Errorf("overlay graph_replay gap count = %d, want 1", a.Overlay.GraphReplayGaps)
	}
}

// TestOverlayDoesNotDoubleCountOverlappingSpans pins the one arithmetic
// property an overlay of this shape can get wrong. Two spans covering the same
// idle -- a CUPTI flush that happens during a synchronize is the ordinary case
// -- must contribute their union, not their sum. Summing them produces a
// figure that can exceed the idle it is a part of, which is how a reader
// learns to distrust every number beside it.
func TestOverlayDoesNotDoubleCountOverlappingSpans(t *testing.T) {
	b := twoKernelHole()
	b.Overhead(4_000, 4_800, 1<<16)
	b.Overhead(4_400, 5_000, 1<<16)
	report := analyzeFixture(t, b.Finalize())
	s := report.Devices[0].Overall
	if s.Overlay.CuptiOverheadNanos != 1_000 {
		t.Errorf("overlay cupti_overhead = %d ns, want 1000 (the union of 4000..4800 and 4400..5000): overlapping spans were summed", s.Overlay.CuptiOverheadNanos)
	}
	if s.Overlay.CuptiOverheadGaps != 1 {
		t.Errorf("overlay gap count = %d, want 1: two spans in one gap counted the gap twice", s.Overlay.CuptiOverheadGaps)
	}
	if s.Overlay.CuptiOverheadNanos > s.IdleNanos {
		t.Errorf("overlay cupti_overhead %d ns exceeds idle %d ns", s.Overlay.CuptiOverheadNanos, s.IdleNanos)
	}
}

// TestUnattributedIdleNamesNoMechanism pins the split that a whole campaign
// was misread through.
//
// Idle before an activity that correlated to no API record used to be added,
// whole, to the hardware-queue arm, whose stated meaning is "idle is hardware
// queue latency". That turns "we could not find the launch" into a specific
// mechanism, and a mechanism is what a reader goes and works on. The two arms
// have to be separable in the output, not only in the class table -- a reader
// who sums the attribution lines must not find the unknown inside the known.
func TestUnattributedIdleNamesNoMechanism(t *testing.T) {
	// Three activities, 100 us apart. The middle one correlates; the last
	// does not, so one gap is attributable and one is not.
	acts := []timeline.Activity{
		{Start: 0, End: 100_000, Correlation: 1},
		{Start: 200_000, End: 300_000, Correlation: 2},
		{Start: 400_000, End: 500_000, Correlation: 3},
	}
	apis := map[uint32]timeline.Span{2: {Start: 10_000, End: 50_000}} // host ahead
	s := computeTestStats(acts, apis, "overall", gapSpans{})
	if s.GapCount != 2 || s.IdleNanos != 200_000 {
		t.Fatalf("gaps/idle = %d/%d, want 2/200000", s.GapCount, s.IdleNanos)
	}
	a := s.Attribution
	if a.HardwareNanos != 100_000 || a.UnattributedNanos != 100_000 {
		t.Errorf("attribution = %+v, want hardware 100000 and unattributed 100000 kept apart", a)
	}
	if got := a.HostBehindNanos + a.DriverSubmitNanos + a.HardwareNanos + a.UnattributedNanos; got != s.IdleNanos {
		t.Errorf("arms sum to %d, want %d: the split must partition idle", got, s.IdleNanos)
	}
}

// TestOpaqueGraphReplayIsNotIdle covers the condition that makes the exclusion
// above unsound, which is the one a real capture hit.
//
// The envelope is excluded because it spans the node records. When CUPTI emits
// the envelope and nothing inside it -- no device record names a graph id --
// it spans nothing, and excluding it deletes the entire replay from the
// timeline and republishes its duration as device idle. On the capture that
// found this, 2942.915 ms of graph work read as idle and the duty cycle read
// 14.54% where the work was about 71%. An A/B that disabled graphs moved wall
// clock by 33 ms against a 2904 ms prediction: the work was always there.
//
// Both arms of the discriminator are asserted. A capture whose kernels do name
// their graph must keep the exclusion, or this fix trades one 5x error for
// another.
func TestOpaqueGraphReplayIsNotIdle(t *testing.T) {
	opaque := journaltest.New()
	opaque.KernelOn(0, 1, 1_000, 2_000, 1)  // not a graph node
	opaque.KernelOn(0, 1, 9_000, 10_000, 2) // not a graph node
	opaque.GraphTrace(0, 3_000, 8_000, 5, 3)
	report := analyzeFixture(t, opaque.Finalize())
	if !report.OpaqueGraphReplays {
		t.Fatal("a capture with replay envelopes and no graph-tagged record was not reported as opaque")
	}
	s := report.Devices[0].Overall
	// 1000 + 5000 + 1000 of work in a 9000 ns span: the replay is on the
	// timeline. Excluding it gives 2000 active and 7000 idle.
	if s.ActiveNanos != 7_000 || s.IdleNanos != 2_000 {
		t.Errorf("active/idle = %d/%d, want 7000/2000: the replay was dropped from the timeline and reported as idle", s.ActiveNanos, s.IdleNanos)
	}
	if s.DutyCyclePct < 77.7 || s.DutyCyclePct > 77.8 {
		t.Errorf("duty cycle = %.2f, want 77.78", s.DutyCyclePct)
	}
	// Counted as activity, it is no longer an idle overlay. Reporting it in
	// both places would let a reader add it to a total it is already inside.
	if s.Overlay.GraphReplayNanos != 0 {
		t.Errorf("graph replay overlay = %d ns, want 0: the envelope is on the timeline now", s.Overlay.GraphReplayNanos)
	}

	// The other arm: one graph-tagged kernel is enough to make the envelope
	// the span of something, and the exclusion must survive.
	tagged := journaltest.New()
	tagged.KernelOn(0, 1, 1_000, 2_000, 1)
	tagged.GraphNode(0, 1, 3_000, 4_000, 4, 5)
	tagged.KernelOn(0, 1, 9_000, 10_000, 2)
	tagged.GraphTrace(0, 3_000, 8_000, 5, 3)
	report = analyzeFixture(t, tagged.Finalize())
	if report.OpaqueGraphReplays {
		t.Error("a capture whose kernels name their graph was reported as opaque")
	}
	if s := report.Devices[0].Overall; s.ActiveNanos != 3_000 {
		t.Errorf("active = %d ns, want 3000: the envelope was counted on top of the node inside it", s.ActiveNanos)
	}
}

// TestKernelCensusCountsOnlyKernels keeps memcpy, sync and overhead spans out
// of a per-kernel launch count: a census that counted them would report a
// launch count that moved where no kernel launch did.
func TestKernelCensusCountsOnlyKernels(t *testing.T) {
	b := journaltest.New()
	b.Named("kernel", "k", 1_000, 2_000, 1)
	// Named on purpose: an unnamed memcpy is excluded by the name check
	// rather than by the subtype check, so it cannot tell whether the subtype
	// check does anything.
	b.Named("memcpy", "copyHtoD", 3_000, 4_000, 2)
	b.Overhead(5_000, 6_000, 1)
	cap := journaltest.Capture(map[string][]byte{"capture.journal": b.Finalize()}, journaltest.CompleteStatus())
	report, err := Analyze(cap, Options{MaxRecords: timeline.MaxRecords})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Kernels) != 1 || report.Kernels[0].Name != "k" || report.Kernels[0].Launches != 1 {
		t.Errorf("census = %+v, want one kernel k with one launch", report.Kernels)
	}
	if report.Kernels[0].DeviceNanos != 1_000 {
		t.Errorf("device time = %d ns, want 1000", report.Kernels[0].DeviceNanos)
	}
}

// TestUnnamedKernelRecordsAreCounted keeps a census total honest about what it
// does not cover. A kernel record without a name cannot be a row, and dropping
// it silently makes the rows look like the whole population.
func TestUnnamedKernelRecordsAreCounted(t *testing.T) {
	b := journaltest.New()
	b.Named("kernel", "k", 1_000, 2_000, 1)
	b.Kernel(3_000, 4_000, 2)
	cap := journaltest.Capture(map[string][]byte{"capture.journal": b.Finalize()}, journaltest.CompleteStatus())
	report, err := Analyze(cap, Options{MaxRecords: timeline.MaxRecords})
	if err != nil {
		t.Fatal(err)
	}
	if report.UnnamedKernelRecords != 1 {
		t.Errorf("unnamed kernel records = %d, want 1", report.UnnamedKernelRecords)
	}
	if len(report.Kernels) != 1 {
		t.Errorf("census = %+v, want only the named kernel", report.Kernels)
	}
}

func TestGapEvidenceLimits(t *testing.T) {
	acts := []timeline.Activity{
		{Start: 100, End: 200, Stream: 1},
		{Start: 150, End: 250, Stream: 2, Subtype: "memset"},
		{Start: 300, End: 400, Stream: 1, Correlation: 1},
	}
	stats := computeTestStats(acts, map[uint32]timeline.Span{1: {Start: 260, End: 280}}, "overall", gapSpans{
		sync: []timeline.Span{{Start: 0, End: 500}},
	})
	if stats.ActiveNanos != 250 || stats.IdleNanos != 50 || stats.SpanNanos != 300 {
		t.Fatalf("occupancy = %+v, want active 250, gap 50, span 300", stats)
	}
	if stats.UnknownCauseNanos != 50 || stats.Overlay.SyncNanos != 50 {
		t.Fatalf("synchronization overlap resolved unknown cause: %+v", stats)
	}
	if stats.Scope != "device-wide recorded activity" || stats.Window != "first activity start to last activity end" {
		t.Fatalf("missing evidence bounds: %+v", stats)
	}
}

func TestSteadyEmpty(t *testing.T) {
	if got := Steady(nil, 0, nil); len(got) != 0 {
		t.Fatalf("empty steady activities = %v", got)
	}
}

func computeTestStats(acts []timeline.Activity, apis map[uint32]timeline.Span, label string, spans gapSpans) *Stats {
	return computeStats(acts, func(a timeline.Activity) (timeline.Span, bool) {
		api, ok := apis[a.Correlation]
		return api, ok
	}, label, spans)
}

func TestTimingAPIs(t *testing.T) {
	r := timeline.Records{Calls: []timeline.Call{
		{Journal: "one", Domain: "driver", Correlation: 7, Start: 10, End: 20},
		{Journal: "two", Domain: "driver", Correlation: 7, Start: 30, End: 40},
		{Journal: "one", Domain: "runtime", Correlation: 7, Start: 50, End: 60},
		{Journal: "one", Domain: "driver", Correlation: 8, Start: 70, End: 80},
		{Journal: "one", Domain: "driver", Correlation: 8, Start: 90, End: 100},
	}}
	tests := []struct {
		name, journal, domain, subtype string
		correlation                    uint32
		graph                          uint32
		wantStart                      int64
		want                           bool
	}{
		{"first generation", "one", "driver", "kernel", 7, 0, 10, true},
		{"second generation", "two", "driver", "kernel", 7, 0, 30, true},
		{"runtime domain", "one", "runtime", "kernel", 7, 0, 50, true},
		{"graph timing only", "one", "driver", "kernel", 7, 1, 10, true},
		{"missing journal", "", "driver", "kernel", 7, 0, 0, false},
		{"missing call", "three", "driver", "kernel", 7, 0, 0, false},
		{"reused ID", "one", "driver", "kernel", 8, 0, 0, false},
		{"runtime transfer unsupported", "one", "runtime", "memcpy", 7, 0, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := timeline.Activity{Journal: tt.journal, Correlation: tt.correlation, GraphID: tt.graph, Subtype: tt.subtype}
			span, ok := timingAPIs(r, tt.domain)(a)
			if ok != tt.want || ok && span.Start != tt.wantStart {
				t.Fatalf("timing API = %+v, %v; want start %d, %v", span, ok, tt.wantStart, tt.want)
			}
		})
	}
}

func TestGraphCorrelationMeaning(t *testing.T) {
	b := journaltest.New().GraphNode(0, 1, 100, 200, 7, 1).API(70, 80, 7).Finalize()
	r, err := Analyze(journaltest.Capture(map[string][]byte{"capture.journal": b}, nil), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Correlated != 0 || r.TimingAssociated != 100 || r.APIDomain != "driver" {
		t.Fatalf("graph association = %+v", r)
	}
}
