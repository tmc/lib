package main

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// TestDominanceRecommendationDropsGraphAdviceUnderReplay pins the defect
// measured on GB10: the dominance advisory told a workload that was already a
// CUDA graph replay to capture itself into a CUDA graph, at Critical severity,
// because graphId never reaches the advisor.
func TestDominanceRecommendationDropsGraphAdviceUnderReplay(t *testing.T) {
	const graphClause = "capture the entire layer sequence into a CUDA Graph"

	if rec := dominanceRecommendation(0); !strings.Contains(rec, graphClause) {
		t.Errorf("without graph replays the advisory must still offer graph capture; got %q", rec)
	}
	rec := dominanceRecommendation(5)
	if strings.Contains(rec, graphClause) {
		t.Errorf("under graph replay the advisory must not recommend graph capture; got %q", rec)
	}
	if !strings.Contains(rec, "already replays CUDA graphs") {
		t.Errorf("suppressed advisory must say why it was suppressed; got %q", rec)
	}
}

// TestIdleDominatedCaptureSaysSoFirst pins the finding that keeps the rest of
// the report honest.
//
// advise ranks by device time and launch count. On a capture where the device
// is mostly idle, that ranking is complete and still misleading: it orders the
// minority of the span and is silent about the majority. The quiet arm matters
// as much -- a busy capture must not carry this finding, or it becomes a
// banner readers learn to skip.
func TestIdleDominatedCaptureSaysSoFirst(t *testing.T) {
	// Two kernels per stream, 1 ms each, placed to make the duty cycle.
	build := func(starts []uint64, durNs uint64) *cudacapture.Capture {
		var samples []*profile.Sample
		for i, st := range starts {
			samples = append(samples, &profile.Sample{
				Location: []*profile.Location{{ID: uint64(i + 1), Line: []profile.Line{
					{Function: &profile.Function{ID: 1, Name: "work_kernel"}},
				}}},
				Value: []int64{int64(durNs)},
				Label: map[string][]string{"name": {"work_kernel"}},
				NumLabel: map[string][]int64{
					"stream": {7},
					"start":  {int64(st)},
					"end":    {int64(st + durNs)},
				},
			})
		}
		return &cudacapture.Capture{Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample:     samples,
		}}
	}
	idleFinding := func(t *testing.T, cap *cudacapture.Capture) *AdviceItem {
		t.Helper()
		report, err := AnalyzeCapture(cap)
		if err != nil {
			t.Fatalf("AnalyzeCapture: %v", err)
		}
		for i := range report.Items {
			if strings.Contains(report.Items[i].Finding, "The device was idle for") {
				// It has to be readable before the findings it qualifies.
				for j := 0; j < i; j++ {
					if report.Items[j].Severity != SeverityCritical {
						t.Errorf("idle finding ranks below a %s finding: %s", report.Items[j].Severity, report.Items[j].Finding)
					}
				}
				return &report.Items[i]
			}
		}
		return nil
	}

	// 1 ms of work in every 10 ms.
	idle := build([]uint64{0, 10e6, 20e6, 30e6}, 1e6)
	got := idleFinding(t, idle)
	if got == nil {
		t.Fatal("a capture that is mostly device idle carried no idle finding")
	}
	if got.Severity != SeverityCritical {
		t.Errorf("severity = %s, want Critical", got.Severity)
	}
	streams, _ := timeline.Events(idle)
	duty, _, _, ok := profileDutyCycle(streams)
	if !ok {
		t.Fatal("duty cycle not computable for the fixture the finding fired on")
	}
	// The figure in the prose must be the IDLE share, not the duty cycle.
	// Inverting those two is the one error this sentence can make and still
	// read as sensible, so assert both: the idle number present, and the duty
	// number not standing where it belongs.
	if want := fmt.Sprintf("idle for %.2f%%", 100.0-duty); !strings.Contains(got.Finding, want) {
		t.Errorf("finding does not state %q: %s", want, got.Finding)
	}
	if bad := fmt.Sprintf("idle for %.2f%%", duty); strings.Contains(got.Finding, bad) {
		t.Errorf("finding reports the duty cycle as the idle share: %s", got.Finding)
	}
	if !strings.Contains(got.ActionableRecommendation, "cudaprof gaps") {
		t.Errorf("recommendation does not name the command that attributes idle: %s", got.ActionableRecommendation)
	}

	// The control: back-to-back kernels leave no idle, and must not be told
	// the device was idle.
	if q := idleFinding(t, build([]uint64{0, 1e6, 2e6, 3e6}, 1e6)); q != nil {
		t.Errorf("a fully busy capture reported idle: %s", q.Finding)
	}
}

// TestProfileDutyCycleUnionsOverlappingStreams pins that busy time is a union.
// Summing durations across concurrent streams produces more busy time than
// there is span, which is how a duty cycle exceeds 100%.
func TestProfileDutyCycleUnionsOverlappingStreams(t *testing.T) {
	streams := map[uint32][]timeline.Event{
		7: {{Start: 100, End: 200, DurationNs: 100}, {Start: 300, End: 400, DurationNs: 100}},
		9: {{Start: 100, End: 200, DurationNs: 100}, {Start: 150, End: 250, DurationNs: 100}},
	}
	pct, spanNs, busyNs, ok := profileDutyCycle(streams)
	if !ok {
		t.Fatal("duty cycle not computed for four timestamped activities")
	}
	if spanNs != 300 {
		t.Errorf("span = %d, want 300", spanNs)
	}
	// 100-250 and 300-400: 250 ns of union, not the 400 ns the durations sum to.
	if busyNs != 250 {
		t.Errorf("busy = %d, want 250: concurrent intervals were summed, not unioned", busyNs)
	}
	if pct > 100 {
		t.Errorf("duty cycle = %v, which is impossible and means busy was summed", pct)
	}

	// A timeline whose first activity starts at 0. Exact values, because the
	// span is what a zero-start sentinel gets wrong, and deriving the
	// expectation from the same function would hide it.
	zero := map[uint32][]timeline.Event{7: {
		{Start: 0, End: 1000, DurationNs: 1000},
		{Start: 9000, End: 10000, DurationNs: 1000},
	}}
	pct, spanNs, busyNs, ok = profileDutyCycle(zero)
	if !ok {
		t.Fatal("duty cycle not computed for a timeline starting at 0")
	}
	if spanNs != 10000 || busyNs != 2000 {
		t.Errorf("span/busy = %d/%d, want 10000/2000: the span was measured from the second activity", spanNs, busyNs)
	}
	if pct != 20 {
		t.Errorf("duty cycle = %v, want 20", pct)
	}
}

// TestDutyCycleCountsOnlyDeviceWork covers the three spans a capture carries
// that are not device occupancy.
//
// A synchronization window is the worst of them: the device is idle BECAUSE
// the host is inside that call, so counting it as busy erases exactly the idle
// it explains, and the duty-cycle finding that ranks every other finding goes
// quiet on the captures that need it most.
func TestDutyCycleCountsOnlyDeviceWork(t *testing.T) {
	streams := map[uint32][]timeline.Event{
		0: {
			{Subtype: "kernel", Start: 0, End: 1_000},
			{Subtype: "sync", Start: 1_000, End: 8_000},
			{Subtype: "overhead", Start: 2_000, End: 3_000},
			{Subtype: "kernel", Start: 9_000, End: 10_000},
		},
	}
	pct, span, busy, ok := profileDutyCycle(streams)
	if !ok {
		t.Fatal("duty cycle not computed")
	}
	if span != 10_000 || busy != 2_000 {
		t.Errorf("span/busy = %d/%d, want 10000/2000: a host wait or CUPTI's own cost was counted as device work", span, busy)
	}
	if pct < 19.9 || pct > 20.1 {
		t.Errorf("duty cycle = %.2f, want 20.00", pct)
	}
}

// TestDutyCycleCountsAnUnrecordedGraphReplay pins both arms of the graph rule.
// The envelope is excluded because it spans its nodes' records; a capture that
// holds no such record has had its whole replay deleted from the timeline and
// republished as idle. One real capture read 14.54% where the work was ~71%.
func TestDutyCycleCountsAnUnrecordedGraphReplay(t *testing.T) {
	opaque := map[uint32][]timeline.Event{
		0: {
			{Subtype: "kernel", Start: 0, End: 1_000},
			{Subtype: "graph_trace", GraphID: 5, Start: 2_000, End: 9_000},
			{Subtype: "kernel", Start: 9_000, End: 10_000},
		},
	}
	_, span, busy, ok := profileDutyCycle(opaque)
	if !ok || span != 10_000 || busy != 9_000 {
		t.Errorf("span/busy = %d/%d (ok %v), want 10000/9000: the replay was dropped and reported as idle", span, busy, ok)
	}

	// One kernel naming its graph makes the envelope the span of something,
	// and counting both would charge that device time twice.
	recorded := map[uint32][]timeline.Event{
		0: {
			{Subtype: "kernel", Start: 0, End: 1_000},
			{Subtype: "graph_trace", GraphID: 5, Start: 2_000, End: 9_000},
			{Subtype: "kernel", GraphID: 5, Start: 3_000, End: 4_000},
			{Subtype: "kernel", Start: 9_000, End: 10_000},
		},
	}
	if _, span, busy, ok := profileDutyCycle(recorded); !ok || span != 10_000 || busy != 3_000 {
		t.Errorf("span/busy = %d/%d (ok %v), want 10000/3000: the envelope was counted on top of the node inside it", span, busy, ok)
	}
}

// TestOpaqueGraphCaveatReachesTheDutyCycleFinding pins the caveat's delivery.
// The finding that reranks every other finding must say when its busy figure
// includes time the device may not have been working: with no graph-tagged
// record, a replay counts as busy for its whole envelope, so the idle it
// reports is a floor and not a measurement.
func TestOpaqueGraphCaveatReachesTheDutyCycleFinding(t *testing.T) {
	b := newJournalBuilder()
	b.kernelOn(0, 1, 1_000, 2_000, 1)
	b.graphTrace(0, 3_000, 4_000, 5, 3)
	b.kernelOn(0, 1, 90_000, 91_000, 2)
	cap := testCapture(map[string][]byte{"capture.journal": b.finalize()}, completeStatus())

	streams, opaque := timeline.Events(cap)
	if !opaque {
		t.Fatal("journal with envelopes and no graph-tagged record did not report opaque replays")
	}
	if _, _, busy, ok := profileDutyCycle(streams); !ok || busy != 3_000 {
		t.Fatalf("busy = %d (ok %v), want 3000: the envelope is on the timeline", busy, ok)
	}

	report := &AdviceReport{}
	auditDeviceDutyCycle(cap, report)
	if !strings.Contains(dutyCycleFinding(t, report), "lower bound") {
		t.Errorf("finding does not carry the opacity caveat: %q", dutyCycleFinding(t, report))
	}

	// A capture whose replays are described by their nodes must not carry it:
	// a caveat printed on every capture is one a reader learns to skip.
	c := newJournalBuilder()
	c.kernelOn(0, 1, 1_000, 2_000, 1)
	c.graphNode(0, 1, 3_000, 4_000, 4, 5)
	c.graphTrace(0, 3_000, 4_000, 5, 3)
	c.kernelOn(0, 1, 90_000, 91_000, 2)
	plain := &AdviceReport{}
	auditDeviceDutyCycle(testCapture(map[string][]byte{"capture.journal": c.finalize()}, completeStatus()), plain)
	if strings.Contains(dutyCycleFinding(t, plain), "lower bound") {
		t.Errorf("opacity caveat on a capture whose nodes name their graph: %q", dutyCycleFinding(t, plain))
	}
}

// dutyCycleFinding returns the duty-cycle finding's text. It selects rather
// than indexing: auditDeviceDutyCycle also emits the idle attribution, and a
// test that asserts on Items[0] asserts on whichever of the two happened to
// be appended first.
func dutyCycleFinding(t *testing.T, report *AdviceReport) string {
	t.Helper()
	for _, it := range report.Items {
		if strings.Contains(it.Finding, "The device was idle for") {
			return it.Finding
		}
	}
	t.Fatalf("no duty-cycle finding among %d items", len(report.Items))
	return ""
}

// idleFinding returns the idle-attribution finding's text, selected rather
// than indexed for the reason dutyCycleFinding is.
func idleFinding(t *testing.T, report *AdviceReport) string {
	t.Helper()
	for _, it := range report.Items {
		if strings.Contains(it.Finding, "Idle attribution:") {
			return it.Finding
		}
	}
	t.Fatalf("no idle-attribution finding among %d items", len(report.Items))
	return ""
}

// idleJournal is a five-kernel run of 1000 ns kernels separated by 9000 ns
// gaps -- a 12.2% duty cycle. apiAt places each kernel's launch call relative
// to that kernel's start, which is the only thing that decides which idle arm
// the gap lands in.
func idleJournal(apiAt func(kernelStart uint64) (start, end uint64)) []byte {
	b := newJournalBuilder()
	for i := uint32(1); i <= 5; i++ {
		start := uint64(10_000) + uint64(i-1)*10_000
		b.kernel(start, start+1_000, i)
		s, e := apiAt(start)
		b.api(s, e, i)
	}
	return b.finalize()
}

// TestIdleAttributionNamesTheMechanismItMeasured is a double dissociation: the
// same kernels and the same idle time, with only the launch call moved, must
// produce opposite arms and opposite remedies. A test with one fixture cannot
// tell a correct attribution from a constant one.
func TestIdleAttributionNamesTheMechanismItMeasured(t *testing.T) {
	tests := []struct {
		name        string
		journal     []byte
		wantArm     string
		wantRemedy  string
		notRemedy   string
		wantNoMajor bool
	}{
		{
			// The launch returned before the previous kernel ended, so the
			// work was queued and the device idled anyway.
			name:       "hardware queue",
			journal:    idleJournal(func(k uint64) (uint64, uint64) { return k - 10_000, k - 9_500 }),
			wantArm:    "hardware queue",
			wantRemedy: "Neither batching nor earlier submission",
			notRemedy:  "Submit further ahead",
		},
		{
			// The launch call started 5000 ns after the previous kernel ended.
			name:       "host behind",
			journal:    idleJournal(func(k uint64) (uint64, uint64) { return k - 4_000, k - 3_000 }),
			wantArm:    "host behind",
			wantRemedy: "Submit further ahead",
			notRemedy:  "Neither batching nor earlier submission",
		},
		{
			// One gap of each class. The largest arm holds 40% of the idle,
			// so the finding has to say that acting on it recovers 40% and
			// not the idle time.
			name:        "no majority",
			journal:     mixedIdleJournal(),
			wantArm:     "hardware queue",
			wantRemedy:  "Neither batching nor earlier submission",
			notRemedy:   "Submit further ahead",
			wantNoMajor: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := testCapture(map[string][]byte{"capture.journal": tt.journal}, completeStatus())
			report := &AdviceReport{}
			if !auditIdleAttribution(cap, report) {
				t.Fatal("auditIdleAttribution reported no finding on a 12% duty cycle capture")
			}
			if len(report.Items) != 1 {
				t.Fatalf("items = %d, want 1", len(report.Items))
			}
			item := report.Items[0]
			want := "largest share of that idle -- "
			if !strings.Contains(item.Finding, want) || !strings.Contains(item.Finding, "was "+tt.wantArm+".") {
				t.Errorf("finding does not name %q as the largest arm:\n%s", tt.wantArm, item.Finding)
			}
			if !strings.Contains(item.ActionableRecommendation, tt.wantRemedy) {
				t.Errorf("recommendation lacks %q:\n%s", tt.wantRemedy, item.ActionableRecommendation)
			}
			if strings.Contains(item.ActionableRecommendation, tt.notRemedy) {
				t.Errorf("recommendation carries the other arm's remedy %q:\n%s", tt.notRemedy, item.ActionableRecommendation)
			}
			// The split is stated whole: a reader acting on one arm has to be
			// able to see what the other three hold.
			for _, arm := range []string{"host behind", "driver submission", "hardware queue", "unattributed"} {
				if !strings.Contains(item.Finding, arm) {
					t.Errorf("finding omits the %s arm:\n%s", arm, item.Finding)
				}
			}
			if item.DeviceTimeNanos <= 0 {
				t.Errorf("DeviceTimeNanos = %d, want the idle nanos the finding accounts for", item.DeviceTimeNanos)
			}
			// Every fixture runs the same kernels: 5000 ns of device work
			// inside 41000 ns, so 87.80% of the span was idle. The share this
			// item reports is the share it accounts for -- the idle -- and
			// not the duty cycle, which is the quantity it exists to rank
			// ahead of.
			if want := 87.80; math.Abs(item.ShareOfProfile-want) > 0.01 {
				t.Errorf("ShareOfProfile = %.2f, want the idle share of the span, %.2f", item.ShareOfProfile, want)
			}
			// A plurality is not a majority. The caveat has to appear exactly
			// where it is true: printed on a 100% arm it is false, and
			// omitted on a 40% arm it lets the recommendation read as an
			// account of all the idle time.
			const caveat = "No single mechanism holds a majority"
			if got := strings.Contains(item.Finding, caveat); got != tt.wantNoMajor {
				t.Errorf("majority caveat present = %v, want %v:\n%s", got, tt.wantNoMajor, item.Finding)
			}
		})
	}
}

// mixedIdleJournal gives each of the four idle arms one gap, so no arm holds a
// majority: hardware queue 14500 ns of 36000 ns.
func mixedIdleJournal() []byte {
	b := newJournalBuilder()
	kstart := func(i uint32) uint64 { return uint64(10_000) + uint64(i-1)*10_000 }
	for i := uint32(1); i <= 5; i++ {
		b.kernel(kstart(i), kstart(i)+1_000, i)
	}
	b.api(kstart(1)-5_000, kstart(1)-4_000, 1)
	// Host ahead: the call returned before the previous kernel ended.
	b.api(kstart(2)-10_000, kstart(2)-9_500, 2)
	// Host behind: the call started 8000 ns after the device went idle.
	b.api(kstart(3)-1_000, kstart(3)-500, 3)
	// Correlation 4 is deliberately absent: unattributed.
	// Host during: the previous kernel ended inside the call.
	b.api(kstart(5)-9_100, kstart(5)-5_000, 5)
	return b.finalize()
}

// TestNoIdleAttributionOnABusyCapture keeps the finding off a capture whose
// device was working. Without this the gate is untested in the direction that
// matters: a finding that fires on every capture ranks nothing.
func TestNoIdleAttributionOnABusyCapture(t *testing.T) {
	cap := testCapture(map[string][]byte{"capture.journal": uniformJournal()}, completeStatus())
	report := &AdviceReport{}
	if auditIdleAttribution(cap, report) || len(report.Items) != 0 {
		t.Errorf("attribution finding on an 83%% duty cycle capture: %+v", report.Items)
	}
}

// TestIdleAttributionOutranksAKernelTimeFinding is the whole point of the
// finding: on an idle-dominated capture the reader's first instruction has to
// be about the idle time, not about the largest kernel in the minority of the
// span that remains.
func TestIdleAttributionOutranksAKernelTimeFinding(t *testing.T) {
	journal := idleJournal(func(k uint64) (uint64, uint64) { return k - 10_000, k - 9_500 })
	cap := testCapture(map[string][]byte{"capture.journal": journal}, completeStatus())
	report := &AdviceReport{}
	// A kernel finding holding every nanosecond the device was actually busy.
	report.Items = append(report.Items, AdviceItem{
		Severity:        SeverityCritical,
		Category:        CategoryCompute,
		EvidenceLevel:   EvidenceObserved,
		DeviceTimeNanos: 5_000,
		ShareOfProfile:  100,
		Finding:         "kernel foo holds all of the device time",
		Location:        Location{Kernel: "foo"},
	})
	auditDeviceDutyCycle(cap, report)
	if len(report.Items) != 3 {
		t.Fatalf("items = %d, want the kernel finding plus a duty cycle and an attribution finding", len(report.Items))
	}
	finalizeAndRankFindings(report, map[string]int64{"foo": 5_000}, 5_000)
	first := report.Items[0]
	if !strings.Contains(first.Finding, "Idle attribution:") {
		t.Errorf("first finding is not the idle attribution:\n%s", first.Finding)
	}
	// The duty cycle finding must stop telling the reader to go attribute the
	// idle time once this report has attributed it.
	var duty AdviceItem
	for _, it := range report.Items {
		if strings.Contains(it.Finding, "The device was idle for") {
			duty = it
		}
	}
	if duty.Finding == "" {
		t.Fatal("no duty cycle finding")
	}
	if strings.Contains(duty.ActionableRecommendation, "run cudaprof gaps on this capture") {
		t.Errorf("duty cycle finding still sends the reader to a command this report already ran:\n%s", duty.ActionableRecommendation)
	}
	if !strings.Contains(duty.ActionableRecommendation, "idle attribution in this report") {
		t.Errorf("duty cycle finding does not point at the attribution:\n%s", duty.ActionableRecommendation)
	}
}

// TestIdleMagnitudeIsNotLabelledGPUTime guards the rendered header, which is
// the line a reader ranks findings by. The default parenthetical reads "N% of
// GPU time, M ms", and the attribution's M ms is idle: rendered with the
// default label it reports the time the device did nothing as the time a
// kernel ran.
func TestIdleMagnitudeIsNotLabelledGPUTime(t *testing.T) {
	journal := idleJournal(func(k uint64) (uint64, uint64) { return k - 10_000, k - 9_500 })
	cap := testCapture(map[string][]byte{"capture.journal": journal}, completeStatus())
	report := &AdviceReport{}
	if !auditIdleAttribution(cap, report) {
		t.Fatal("no attribution finding")
	}
	var line string
	for _, l := range strings.Split(report.Markdown(), "\n") {
		if strings.HasPrefix(l, "### ") && strings.Contains(l, "Idle attribution:") {
			line = l
		}
	}
	if line == "" {
		t.Fatalf("attribution finding has no header in:\n%s", report.Markdown())
	}
	if strings.Contains(line, "of GPU time") {
		t.Errorf("idle time rendered as GPU time:\n%s", line)
	}
	if !strings.Contains(line, "of idle") || !strings.Contains(line, "hardware queue:") {
		t.Errorf("header does not say what its number is:\n%s", line)
	}
}

// TestIdleAttributionSeparatesUnseenFromAbsent covers the difference between
// idle a capture could not attribute and idle that is not there. A graph
// replay recorded as an envelope with no per-node records carries no launch
// correlation, so idle beside it lands in the unattributed arm -- with a cause
// that is known and nameable, not unknown. A reader told only "unattributed"
// reasonably concludes the measurement found nothing; the honest statement is
// that the instrument could not see through the replay, and that is fixed by
// re-capturing rather than by changing the program.
func TestIdleAttributionSeparatesUnseenFromAbsent(t *testing.T) {
	const opaque = "graph replay envelopes with no per-node records"

	// Same kernels and same idle in both arms: only the replay records differ.
	plain := idleJournal(func(k uint64) (uint64, uint64) { return k - 5_000, k - 100 })
	report := &AdviceReport{}
	auditDeviceDutyCycle(testCapture(map[string][]byte{"capture.journal": plain}, completeStatus()), report)
	if got := idleFinding(t, report); strings.Contains(got, opaque) {
		t.Errorf("a capture with no graph replays claims replay opacity:\n%s", got)
	}

	b := newJournalBuilder()
	for i := uint32(1); i <= 5; i++ {
		start := uint64(10_000) + uint64(i-1)*10_000
		b.kernel(start, start+1_000, i)
		b.api(start-5_000, start-100, i)
	}
	// An envelope with no nodes inside it: the shape that hides the census.
	b.graphTrace(0, 60_000, 69_000, 7, 100)
	withReplays := b.finalize()

	report = &AdviceReport{}
	auditDeviceDutyCycle(testCapture(map[string][]byte{"capture.journal": withReplays}, completeStatus()), report)
	idle := idleFinding(t, report)
	if !strings.Contains(idle, opaque) {
		t.Errorf("idle finding does not name replay opacity:\n%s", idle)
	}
	if !strings.Contains(idle, "real and unattributable here, not absent") {
		t.Errorf("idle finding does not separate unseen from absent:\n%s", idle)
	}
}

// TestOpaqueDutyCycleIsReportedAsACeiling covers the silent half of the
// opacity problem. The idle finding does not fire above the duty-cycle
// threshold, and a reader who sees no idle finding concludes the device was
// busy. On a capture whose replays are opaque that inference is unavailable:
// each replay counts as busy from its envelope alone, so a run that spent its
// span inside replays reads as busy whatever it did. The absence of the
// finding then says nothing about the run, and nothing in the report said so.
func TestOpaqueDutyCycleIsReportedAsACeiling(t *testing.T) {
	// A span whose busy time is almost entirely one replay envelope: 98%
	// busy as measured, 2% on the records the capture can resolve.
	opaqueJournal := func() []byte {
		b := newJournalBuilder()
		b.kernel(1_000, 2_000, 1)
		b.api(500, 900, 1)
		b.graphTrace(0, 3_000, 98_000, 7, 2)
		b.kernel(99_000, 100_000, 3)
		b.api(98_500, 98_900, 3)
		return b.finalize()
	}

	report := &AdviceReport{}
	auditDeviceDutyCycle(testCapture(map[string][]byte{"capture.journal": opaqueJournal()}, completeStatus()), report)
	ceiling := ceilingFinding(t, report)
	if !strings.Contains(ceiling, "ceiling, not a measurement") {
		t.Errorf("finding does not call the duty cycle a ceiling:\n%s", ceiling)
	}
	// Both numbers have to be present: the ceiling alone is the number that
	// misled the reader, and the floor alone hides how far apart they are.
	if !strings.Contains(ceiling, "97.98%") || !strings.Contains(ceiling, "2.02%") {
		t.Errorf("finding does not report both the measured and resolved shares:\n%s", ceiling)
	}

	// Control 1: the same envelope, resolved into nodes. Nothing is hidden,
	// so there is nothing to warn about.
	b := newJournalBuilder()
	b.kernel(1_000, 2_000, 1)
	b.api(500, 900, 1)
	b.graphTrace(0, 3_000, 98_000, 7, 2)
	b.graphNode(0, 1, 3_000, 98_000, 4, 7)
	b.kernel(99_000, 100_000, 3)
	b.api(98_500, 98_900, 3)
	report = &AdviceReport{}
	auditDeviceDutyCycle(testCapture(map[string][]byte{"capture.journal": b.finalize()}, completeStatus()), report)
	if got := findingContaining(report, "ceiling, not a measurement"); got != "" {
		t.Errorf("ceiling warning on a capture whose nodes are resolved:\n%s", got)
	}

	// Control 2: opaque replays, but the capture is busy on its resolved
	// records alone. Resolving the envelopes cannot change the answer, so
	// saying anything would be noise.
	b = newJournalBuilder()
	b.kernel(1_000, 60_000, 1)
	b.api(500, 900, 1)
	b.graphTrace(0, 61_000, 70_000, 7, 2)
	b.kernel(71_000, 99_000, 3)
	b.api(70_500, 70_900, 3)
	report = &AdviceReport{}
	auditDeviceDutyCycle(testCapture(map[string][]byte{"capture.journal": b.finalize()}, completeStatus()), report)
	if got := findingContaining(report, "ceiling, not a measurement"); got != "" {
		t.Errorf("ceiling warning where the resolved records are already busy:\n%s", got)
	}
}

func findingContaining(report *AdviceReport, want string) string {
	for _, it := range report.Items {
		if strings.Contains(it.Finding, want) {
			return it.Finding
		}
	}
	return ""
}

func ceilingFinding(t *testing.T, report *AdviceReport) string {
	t.Helper()
	if got := findingContaining(report, "ceiling, not a measurement"); got != "" {
		return got
	}
	t.Fatalf("no duty-cycle ceiling finding among %d items", len(report.Items))
	return ""
}

// dominanceSample is one dispatch of a kernel family, with its device time.
func dominanceSample(name string, nanos int64) *profile.Sample {
	return &profile.Sample{
		Value: []int64{nanos},
		Label: map[string][]string{
			"name":    {name},
			"subtype": {"kernel"},
			"stream":  {"1"},
		},
	}
}

// dominanceCapture builds a capture whose profile holds one kernel family of
// count dispatches, each of eachNanos, optionally beside a journal.
func dominanceCapture(count int, eachNanos int64, journal []byte) *cudacapture.Capture {
	samples := make([]*profile.Sample, 0, count)
	for i := 0; i < count; i++ {
		samples = append(samples, dominanceSample(fmt.Sprintf("magma_sgemmEx_kernel_%d", i), eachNanos))
	}
	files := map[string][]byte{}
	if journal != nil {
		files["capture.journal"] = journal
	}
	cap := testCapture(files, completeStatus())
	cap.Profile = &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
		Sample:     samples,
	}
	return cap
}

// submissionIdleJournal builds a timeline whose idle is spent inside the launch
// call: the host was already in the API when the device went idle, which is the
// driver submission arm and the only arm that per-launch batching recovers.
func submissionIdleJournal() []byte {
	b := newJournalBuilder()
	t := uint64(10_000)
	for i := uint32(1); i <= 12; i++ {
		start := t
		end := start + 1_000
		b.kernel(start, end, i)
		if i > 1 {
			// Opened before the previous kernel ended, closed 9000 ns into
			// the 10000 ns gap that follows it.
			b.api(start-10_500, start-1_000, i)
		} else {
			b.api(start-500, start-100, i)
		}
		t = end + 10_000
	}
	return b.finalize()
}

// TestAdvice_DominanceDoesNotBlameLaunchOverheadOnLongDispatches pins the
// separation between what the dominance finding measures and what it infers.
//
// The finding reported a kernel family's share of device time -- measured --
// and in the same sentence, under "Evidence: observed", that fine-grained
// dispatches were incurring submission and launch overhead -- inferred, and on
// a 683 us mean dispatch, false. The capture that raised it attributed 0.27% of
// its idle to driver submission and the kernel was already a batched cuBLASLt
// GEMM, so a reader following the recommendation batched a batched call.
func TestAdvice_DominanceDoesNotBlameLaunchOverheadOnLongDispatches(t *testing.T) {
	const launchOverheadClaim = "launched faster than they execute"
	tests := []struct {
		name            string
		eachNanos       int64
		journal         []byte
		wantOverhead    bool
		wantRecContains string
	}{
		{
			// 683 us per dispatch: launch cost is microseconds and cannot be
			// a dominant share of this.
			name:            "long dispatch is not fine-grained",
			eachNanos:       683_000,
			journal:         submissionIdleJournal(),
			wantOverhead:    false,
			wantRecContains: "the cost is inside the kernel rather than in reaching it",
		},
		{
			// Short dispatches, and the capture measured idle spent inside
			// the launch call: both halves of the claim are supported.
			name:            "short dispatch with measured submission idle",
			eachNanos:       5_000,
			journal:         submissionIdleJournal(),
			wantOverhead:    true,
			wantRecContains: "batched GEMM / GEMV",
		},
		{
			// Short dispatches, but no journal: the submission cost is
			// unmeasured. An unmeasured cost is not a measured one.
			name:            "short dispatch without attribution stays silent",
			eachNanos:       5_000,
			journal:         nil,
			wantOverhead:    false,
			wantRecContains: "the cost is inside the kernel rather than in reaching it",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := dominanceCapture(15, tt.eachNanos, tt.journal)
			kernelTimes, totalTime := computeProfileTimes(cap)
			report := &AdviceReport{}
			auditKernelFamilyDominance(cap, report, kernelTimes, totalTime, hasDeviceTiming(cap))
			if len(report.Items) != 1 {
				t.Fatalf("items = %d, want the dominance finding", len(report.Items))
			}
			item := report.Items[0]
			if !strings.Contains(item.Finding, "dominates GPU execution") {
				t.Fatalf("not the dominance finding:\n%s", item.Finding)
			}
			if got := strings.Contains(item.Finding, launchOverheadClaim); got != tt.wantOverhead {
				t.Errorf("launch-overhead claim = %v, want %v:\n%s", got, tt.wantOverhead, item.Finding)
			}
			if !strings.Contains(item.ActionableRecommendation, tt.wantRecContains) {
				t.Errorf("recommendation does not contain %q:\n%s", tt.wantRecContains, item.ActionableRecommendation)
			}
			// The measured half is always present, whichever way the
			// inferred half went.
			if !strings.Contains(item.Finding, "% of total device time") {
				t.Errorf("finding dropped the measured share:\n%s", item.Finding)
			}
		})
	}
}

// concentratedIdleJournal builds n kernels of kernelNs each separated by
// smallGapNs, with one stallNs gap after the first. Every gap is host behind,
// so the arm is constant and only the concentration and the duty cycle move.
func concentratedIdleJournal(n int, kernelNs, smallGapNs, stallNs uint64) []byte {
	b := newJournalBuilder()
	t := uint64(10_000)
	for i := 1; i <= n; i++ {
		start := t
		end := start + kernelNs
		b.kernel(start, end, uint32(i))
		b.api(start-400, start-100, uint32(i))
		if i == 1 {
			t = end + stallNs
		} else {
			t = end + smallGapNs
		}
	}
	return b.finalize()
}

// TestAdvice_DominantIdleGapReportedAtAnyDutyCycle is a double dissociation on
// the trigger: the same one-time stall, in a capture that is mostly idle and in
// one that is mostly busy, must be reported both times.
//
// The idle finding fired on share of span, so the identical cold start in two
// arms of one program -- differing only in cuBLAS math mode -- was Critical in
// the fast arm and silent in the slow one, where longer kernels diluted it. A
// finding that appears and disappears for reasons unrelated to whether it is
// true cannot be acted on.
func TestAdvice_DominantIdleGapReportedAtAnyDutyCycle(t *testing.T) {
	const stallFinding = "holds"
	tests := []struct {
		name      string
		kernelNs  uint64
		wantBusy  bool
		wantFires bool
	}{
		{name: "idle dominated", kernelNs: 1_000, wantBusy: false, wantFires: true},
		{name: "busy, same stall", kernelNs: 200_000, wantBusy: true, wantFires: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			j := concentratedIdleJournal(6, tt.kernelNs, 1_000, 500_000)
			cap := testCapture(map[string][]byte{"capture.journal": j}, completeStatus())
			attr, ok := captureIdleAttribution(cap)
			if !ok {
				t.Fatal("no idle attribution")
			}
			if busy := attr.duty >= deviceIdleCriticalPct; busy != tt.wantBusy {
				t.Fatalf("duty cycle %.2f%% (busy=%v), want busy=%v -- fixture does not set up the dissociation", attr.duty, busy, tt.wantBusy)
			}
			// The whole report path, so the routing through the duty-cycle
			// branches is what is under test rather than the emitter alone.
			report := &AdviceReport{}
			auditDeviceDutyCycle(cap, report)
			var found bool
			for _, it := range report.Items {
				if strings.Contains(it.Finding, "of idle time, and") && strings.Contains(it.Finding, "A single ") {
					found = true
					if !strings.Contains(it.ActionableRecommendation, "warm-up") {
						t.Errorf("stall finding does not point at warm-up:\n%s", it.ActionableRecommendation)
					}
				}
			}
			if found != tt.wantFires {
				var got []string
				for _, it := range report.Items {
					got = append(got, it.Finding)
				}
				t.Errorf("single-stall finding present = %v, want %v; items:\n%s", found, tt.wantFires, strings.Join(got, "\n"))
			}
		})
	}
}

// TestAdvice_UniformGapsAreNotASingleStall is the negative control for the
// stall finding. Across four uniform gaps the largest necessarily holds 25% of
// the idle, so a share threshold alone reports the arithmetic of "largest of
// four" as a stall. Nothing here is concentrated and nothing should fire.
func TestAdvice_UniformGapsAreNotASingleStall(t *testing.T) {
	j := concentratedIdleJournal(6, 1_000, 100_000, 100_000)
	cap := testCapture(map[string][]byte{"capture.journal": j}, completeStatus())
	attr, ok := captureIdleAttribution(cap)
	if !ok {
		t.Fatal("no idle attribution")
	}
	if attr.gaps < 4 {
		t.Fatalf("gaps = %d, want a handful so the uniform case is exercised", attr.gaps)
	}
	if share, dominant := attr.dominantGap(); dominant {
		t.Errorf("uniform gaps reported as a single stall (largest holds %.1f%% of idle across %d gaps)", share, attr.gaps)
	}
	// The attribution finding still fires: the idle is real, it is just not
	// one stall.
	report := &AdviceReport{}
	if !auditIdleAttribution(cap, report) {
		t.Fatal("no idle attribution finding on a uniform-gap capture")
	}
	for _, it := range report.Items {
		if strings.Contains(it.Finding, "A single ") {
			t.Errorf("stall finding fired on uniform gaps:\n%s", it.Finding)
		}
	}
}
