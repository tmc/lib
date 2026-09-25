package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// dominanceRecommendation words the dominance advisory for what the capture
// shows. Recommending a CUDA Graph to a workload already running as one is
// advice its author cannot act on, so that clause is dropped whenever the
// capture carries graph replays.
// armDriverSubmission names the idle attribution arm that measures per-call
// cost paid inside the launch. It is a constant because a finding that blames
// submission cost looks the arm up by name, and a spelling that drifts would
// silently return "no submission cost measured" for every capture.
const armDriverSubmission = "driver submission"

// dominanceFineGrainedMaxMeanNanos is the mean device duration per dispatch
// below which per-launch cost can plausibly be part of a dominant kernel
// family's total.
//
// Launch and submission cost is a few microseconds per call. Against a
// dispatch averaging hundreds of microseconds it cannot account for a dominant
// share, and a family of 150 dispatches averaging 683 us is not fine-grained
// by any reading. Calling it so sent a reader to batch a call that was already
// a batched cuBLASLt GEMM, and to spend the effort on CUDA graphs for a cost
// the same capture attributed 0.27% of its idle to.
const dominanceFineGrainedMaxMeanNanos = 50_000

// dominanceSubmissionMinIdleShare is the share of idle a capture must
// attribute to driver submission before this report blames submission cost.
// A family can be launched in a tight loop and still cost nothing to submit,
// and the capture measures which it is rather than leaving it to be inferred
// from the launch count.
const dominanceSubmissionMinIdleShare = 5.0

// dispatchSubmissionIdleShare returns the share of the capture's idle time the
// gaps analysis attributes to driver submission, and whether it is known.
//
// A capture with no journal yields nothing here, and nothing is the honest
// answer: an unmeasured submission cost is not a measured absence of one. The
// caller treats an unknown share as insufficient support for blaming
// submission, so the claim is made only where it was measured.
func dispatchSubmissionIdleShare(cap *cudacapture.Capture) (float64, bool) {
	attr, ok := captureIdleAttribution(cap)
	if !ok || attr.idle <= 0 {
		return 0, false
	}
	for _, arm := range attr.arms {
		if arm.name == armDriverSubmission {
			return float64(arm.nanos) / float64(attr.idle) * 100.0, true
		}
	}
	return 0, false
}

// dominanceIsFineGrained reports whether a dominant kernel family's cost can
// be blamed on per-launch overhead, which requires the capture to show both a
// dispatch short enough for the overhead to matter and idle actually spent in
// submission.
//
// The duration test runs first so that the gaps analysis, which parses the
// capture journals, is only paid for on a capture where the answer could be
// yes.
func dominanceIsFineGrained(cap *cudacapture.Capture, familyNanos int64, dispatches int, hasTiming bool) bool {
	if !hasTiming || dispatches <= 0 {
		return false
	}
	if familyNanos/int64(dispatches) >= dominanceFineGrainedMaxMeanNanos {
		return false
	}
	share, ok := dispatchSubmissionIdleShare(cap)
	return ok && share >= dominanceSubmissionMinIdleShare
}

func dominanceRecommendation(graphReplays int) string {
	if graphReplays > 0 {
		return "Restructure per-row dispatches into batched GEMM / GEMV operations. This capture already replays CUDA graphs, so graph capture is not available as a further remedy here."
	}
	return "Restructure per-row dispatches into batched GEMM / GEMV operations or capture the entire layer sequence into a CUDA Graph."
}

func auditKernelFamilyDominance(cap *cudacapture.Capture, report *AdviceReport, kernelTimes map[string]int64, totalTime int64, hasTiming bool) {
	if totalTime <= 0 || len(kernelTimes) == 0 {
		return
	}
	graphReplays := capturedGraphReplays(cap)
	familyTimes := make(map[string]int64)
	familyCounts := make(map[string]int)

	kernelCounts := make(map[string]int)
	if cap.Profile != nil {
		for _, s := range cap.Profile.Sample {
			k := extractKernelNameFromSample(s)
			if k != "" {
				kernelCounts[k]++
			}
		}
	}

	for k, t := range kernelTimes {
		parts := strings.Split(k, "_")
		family := k
		if len(parts) >= 4 {
			family = strings.Join(parts[:len(parts)-1], "_")
		} else if len(parts) >= 2 {
			family = strings.Join(parts[:len(parts)-1], "_")
		}
		familyTimes[family] += t
		familyCounts[family] += kernelCounts[k]
	}

	for family, fTime := range familyTimes {
		pct := float64(fTime) / float64(totalTime) * 100.0
		cnt := familyCounts[family]
		if pct < 50.0 || cnt < 10 {
			continue
		}
		// The share is measured. The cause is not, and the two were
		// previously published in one sentence under "Evidence: observed":
		// the reader could not tell which half the capture supported.
		fineGrained := dominanceIsFineGrained(cap, fTime, cnt, hasTiming)
		var finding string
		if hasTiming {
			finding = fmt.Sprintf("Kernel family %q dominates GPU execution, consuming %.1f%% of total device time (%.2f ms across %d dispatches, %.1f us per dispatch).",
				family, pct, float64(fTime)/1e6, cnt, float64(fTime)/float64(cnt)/1e3)
		} else {
			finding = fmt.Sprintf("Kernel family %q dominates GPU execution, consuming %.1f%% of observed samples (%d samples across %d dispatches).",
				family, pct, fTime, cnt)
		}
		rec := "This finding ranks device time; it does not establish why the family holds it. At this mean dispatch duration the cost is inside the kernel rather than in reaching it, so batching launches or replaying the sequence as a CUDA graph would divide a cost that is not per launch. Check which kernel the library selected for this architecture and compute type -- a library falls back to a generic kernel when it has no tuned one for the requested precision, and the fallback is often several times slower than the tuned path the same GEMM reaches under a different compute type."
		next := "Compare against the same operation at another compute type, and profile the kernel itself with ncu for its achieved occupancy and memory throughput"
		if fineGrained {
			finding += " Dispatches this short are launched faster than they execute, and this capture attributes a measurable share of its idle time to driver submission, so per-launch cost is part of this total."
			rec = dominanceRecommendation(graphReplays)
			next = "Measure wall-clock latency improvement with CUDA graph or batched GEMM/GEMV"
		}
		deviceNanos := fTime
		if !hasTiming {
			deviceNanos = 0
		}
		report.Items = append(report.Items, AdviceItem{
			Severity:        SeverityCritical,
			Category:        CategoryArchitecture,
			EvidenceLevel:   EvidenceObserved,
			DeviceTimeNanos: deviceNanos,
			ShareOfProfile:  pct,
			Location: Location{
				Kernel: family,
			},
			Finding:                  finding,
			ActionableRecommendation: rec,
			NextMeasurement:          next,
		})
	}
}

// deviceIdleCriticalPct is the whole-capture duty cycle below which the idle
// time outweighs everything a kernel-time ranking can see. It is deliberately
// far below lowDutyCyclePct, which judges one short-kernel sequence: this one
// decides whether the report's own ordering is misleading.
const deviceIdleCriticalPct = 50.0

// profileDutyCycle reports the fraction of the captured device span in which
// any device work was running.
//
// Busy time is the UNION of the activity intervals, not their sum. Kernels on
// different streams overlap, and summing durations across streams produces a
// busy figure larger than the span it is divided by -- a duty cycle above
// 100%, which is how this class of number announces that it was summed.
// deviceWorkSubtype reports whether an event occupied the device.
//
// The three it rejects are spans the capture carries for a different purpose:
// CUPTI's own cost, the window the host spent waiting, and the envelope of a
// graph replay. Counting a synchronization window as busy is the worst of the
// three, because it erases exactly the idle it explains -- the device was idle
// BECAUSE the host was in that call.
//
// graphTraceIsWork inverts the graph arm. The envelope is excluded because it
// spans the records of the nodes replayed inside it; when the capture holds no
// such record, it spans nothing, and excluding it deletes the whole replay
// from the timeline and reports its duration as idle.
func deviceWorkSubtype(subtype string, graphTraceIsWork bool) bool {
	switch strings.ToLower(subtype) {
	case "overhead", "sync", "synchronization":
		return false
	case "graph_trace", "graphtrace":
		return graphTraceIsWork
	}
	return true
}

func profileDutyCycle(streams map[uint32][]timeline.Event) (pct float64, spanNs, busyNs uint64, ok bool) {
	// Decided from the capture, not assumed: a replay envelope is redundant
	// with its node records and is the only evidence of the replay without
	// them. See deviceWorkSubtype.
	graphTraceIsWork := true
	for _, evs := range streams {
		for _, ev := range evs {
			if ev.GraphID != 0 && !strings.EqualFold(ev.Subtype, "graph_trace") {
				graphTraceIsWork = false
			}
		}
	}
	type iv struct{ start, end uint64 }
	var ivs []iv
	var first, last uint64
	var haveFirst bool
	for _, evs := range streams {
		for _, ev := range evs {
			if !deviceWorkSubtype(ev.Subtype, graphTraceIsWork) {
				continue
			}
			// End is the sentinel for "this capture carries no activity
			// timestamps", not Start: a device timeline that begins at zero is
			// a legitimate one, and testing Start drops its first activity
			// silently -- which changes both the span and the busy time.
			if ev.End == 0 || ev.End <= ev.Start {
				continue
			}
			ivs = append(ivs, iv{ev.Start, ev.End})
			// Tracked with a flag, not a zero sentinel: a timeline whose
			// first activity starts at 0 would otherwise keep re-taking the
			// earliest start from every later activity, and the span would be
			// measured from the second one.
			if !haveFirst || ev.Start < first {
				first, haveFirst = ev.Start, true
			}
			if ev.End > last {
				last = ev.End
			}
		}
	}
	// One activity has no span worth calling a duty cycle: its span is its own
	// duration and the answer is always 100%.
	if len(ivs) < 2 || last <= first {
		return 0, 0, 0, false
	}
	sort.Slice(ivs, func(i, j int) bool { return ivs[i].start < ivs[j].start })
	cur := ivs[0]
	for _, x := range ivs[1:] {
		if x.start > cur.end {
			busyNs += cur.end - cur.start
			cur = x
			continue
		}
		if x.end > cur.end {
			cur.end = x.end
		}
	}
	busyNs += cur.end - cur.start
	spanNs = last - first
	return float64(busyNs) * 100.0 / float64(spanNs), spanNs, busyNs, true
}

// auditDeviceDutyCycle states, before any kernel finding, when most of the
// captured span was device idle.
//
// advise ranks by device time and launch count, so on an idle-dominated
// capture every finding it emits is ranking the minority of the problem while
// saying nothing about the majority. On one ds4 capture that ordering sent
// three independent readers -- this tool, its operator, and an external
// review -- to micro-kernel batching as the top parity fix, when the device
// was idle 76% of the span and the oracle they were losing to launched 48%
// MORE kernels per token. No count-based remedy could have touched it.
//
// This finding does not diagnose the idle time; it refuses to let the ranking
// below it be read as the whole account, and names the command that does
// attribute idle.
// dutyCycleFloor reports the busy share left when the graph replay envelopes
// are taken out of the busy union, measured against the same span so the two
// percentages are comparable.
//
// An opaque replay is one activity record covering the whole replay, so
// whatever the device did not do between its nodes is counted as work. The
// measured duty cycle is therefore a ceiling. This is the floor beneath it:
// the busy time the capture can actually show. The truth is somewhere in
// between and this capture cannot say where.
func dutyCycleFloor(streams map[uint32][]timeline.Event, spanNs uint64) (float64, bool) {
	if spanNs == 0 {
		return 0, false
	}
	type iv struct{ start, end uint64 }
	var ivs []iv
	for _, evs := range streams {
		for _, ev := range evs {
			// The false is what makes this the floor: deviceWorkSubtype
			// reads it as "a graph replay envelope is not device work", so
			// the envelopes drop out and only records the capture can
			// resolve remain. Do not also skip graph_trace by name here --
			// two spellings of one rule mean neither can be tested.
			if !deviceWorkSubtype(ev.Subtype, false) || ev.End == 0 || ev.End <= ev.Start {
				continue
			}
			ivs = append(ivs, iv{ev.Start, ev.End})
		}
	}
	if len(ivs) == 0 {
		return 0, true
	}
	sort.Slice(ivs, func(i, j int) bool { return ivs[i].start < ivs[j].start })
	var busy uint64
	cur := ivs[0]
	for _, x := range ivs[1:] {
		if x.start > cur.end {
			busy += cur.end - cur.start
			cur = x
			continue
		}
		if x.end > cur.end {
			cur.end = x.end
		}
	}
	busy += cur.end - cur.start
	return float64(busy) * 100.0 / float64(spanNs), true
}

// auditOpaqueDutyCeiling reports a duty cycle that is a ceiling rather than a
// measurement.
//
// The idle finding does not fire above deviceIdleCriticalPct, and a reader who
// sees no idle finding concludes the device was busy. On a capture whose graph
// replays are opaque that inference is not available: each replay counts as
// busy from its envelope alone, so a run that spent most of its span inside
// replays reads as busy whatever it actually did. The absence of the finding
// then says nothing about the run -- the finding could not have fired.
//
// The gate is that removing the envelopes drops the busy share below the
// threshold. Only then could resolving the replays change the answer; a
// capture that is busy on its resolved records alone is busy, and saying
// anything here would be noise.
//
// That gate is also the whole test for opacity, which is why no separate
// opacity flag is consulted. A capture whose replays resolve into nodes has
// its envelopes excluded from the busy union already, so its floor and its
// measured duty cycle are the same number and this returns. Asking a second
// source -- journal record counts rather than the stream events these
// percentages come from -- would let the two disagree about one capture.
func auditOpaqueDutyCeiling(streams map[uint32][]timeline.Event, pct float64, spanNs uint64, report *AdviceReport) {
	floor, ok := dutyCycleFloor(streams, spanNs)
	if !ok || floor >= deviceIdleCriticalPct {
		return
	}
	report.Items = append(report.Items, AdviceItem{
		// Read first by kind: this finding's whole content is that the
		// ordering below it ranks a quantity the capture cannot establish.
		// Ranked by cost it would sort last, having no device time of its
		// own -- which is exactly where it must not be.
		Severity:       SeverityReadFirst,
		severityByKind: true,
		Category:       CategoryArchitectureDispatch,
		EvidenceLevel:  EvidenceUnmeasured,
		magnitude:      fmt.Sprintf("duty cycle %.2f%% measured, %.2f%% on resolved records alone", pct, floor),
		Finding: fmt.Sprintf("The device measured %.2f%% busy, which is why no idle finding appears in this report. That number is a ceiling, not a measurement: this capture counts each graph replay as busy from its envelope alone, so whatever the device did not do between a replay's nodes is counted as work. On the records this capture can resolve, the busy share is %.2f%%. The true duty cycle is somewhere between the two, and a run that is mostly replay reads as busy whichever it is.",
			pct, floor),
		ActionableRecommendation: "Do not read the absence of an idle finding here as evidence that the device was busy: on this capture that finding could not have fired regardless of the idle. Re-capture with graph node records resolved -- device records naming their graph ids -- before concluding anything about this run's duty cycle, and before comparing its duty cycle against a capture whose replays were resolved.",
		NextMeasurement:          "cudaprof gaps <capture> -- it reports the replay opacity directly and splits the idle it can see",
	})
}

func auditDeviceDutyCycle(cap *cudacapture.Capture, report *AdviceReport) {
	streams, graphsOpaque := timeline.Events(cap)
	pct, spanNs, busyNs, ok := profileDutyCycle(streams)
	if !ok {
		return
	}
	report.SpanNanos = int64(spanNs)
	if pct >= deviceIdleCriticalPct {
		// A busy capture can still lose a quarter of its span to a single
		// stall, and the duty cycle is exactly what hides it: the same stall
		// beside longer kernels is a smaller share of a longer span. The
		// concentration is the trigger, not the share.
		auditDominantIdleGap(cap, report)
		// A high duty cycle is why no idle finding follows. On a capture
		// whose replays are opaque that reading is not available: see
		// auditOpaqueDutyCeiling.
		auditOpaqueDutyCeiling(streams, pct, spanNs, report)
		return
	}
	idleNs := spanNs - busyNs
	// Carried forward rather than left to the gaps reader: on such a capture
	// the busy figure includes whatever the device did NOT do between the
	// nodes of a replay, so the idle below is a floor.
	opaque := ""
	if graphsOpaque {
		opaque = " This capture's graph replays are opaque -- no device record names a graph, so each replay counts as busy from its envelope alone and the idle here is a lower bound."
	}
	// The attribution runs first so this finding can say whether the idle it
	// names was attributed, rather than sending the reader to a command that
	// has already run.
	rec := "Attribute the idle time before acting on any kernel finding below: run cudaprof gaps on this capture, which splits idle into host-behind, synchronization, and graph-replay time. A kernel-time ranking cannot distinguish a per-launch tax from a tail, and the two have opposite remedies."
	if auditIdleAttribution(cap, report) {
		rec = "Act on the idle attribution in this report -- the finding naming where this idle went -- before any kernel finding below it. A kernel-time ranking cannot distinguish a per-launch tax from a tail, and the two have opposite remedies."
	}
	report.Items = append(report.Items, AdviceItem{
		Severity:        SeverityCritical,
		Category:        CategoryArchitectureDispatch,
		EvidenceLevel:   EvidenceObserved,
		severityByKind:  true,
		DeviceTimeNanos: int64(busyNs),
		ShareOfProfile:  pct,
		Finding: fmt.Sprintf("The device was idle for %.2f%% of the captured span: %.3f ms idle of a %.3f ms span, with %.3f ms of device work. Every finding below this one ranks that %.2f%% -- none of them can see the idle time, and on this capture the idle time is the larger quantity.%s",
			100.0-pct, float64(idleNs)/1e6, float64(spanNs)/1e6, float64(busyNs)/1e6, pct, opaque),
		ActionableRecommendation: rec,
		NextMeasurement:          "cudaprof gaps <capture> -- idle attribution and the gap distribution",
	})
}

// idleArm is one mechanism of the gaps idle attribution, paired with the
// remedy that mechanism admits.
//
// The remedies are mutually exclusive on purpose. Batching recovers driver
// submission cost and cannot touch hardware queue latency; submitting earlier
// recovers host-behind time and cannot touch either. Naming the wrong arm
// costs the reader the work of trying a remedy that could not have helped,
// and the arms are close enough in a duty-cycle number to be indistinguishable
// without this split.
type idleArm struct {
	name   string
	nanos  int64
	remedy string
}

// idleAttribution is the gaps attribution of one device's idle time, reduced
// to what a finding can state.
type idleAttribution struct {
	device uint32
	idle   int64
	duty   float64
	gaps   int
	median int64
	arms   []idleArm // largest first

	// span is the device timeline this idle was measured over, and maxGap
	// the longest single gap in it. Idle reported as a total and a median
	// cannot distinguish a per-launch tax spread over every gap from one
	// stall that happened once, and those have opposite remedies: see
	// reportDominantIdleGap.
	span   int64
	maxGap int64

	// classes is the host concurrency split, carried for its gap counts.
	// The arms above hold only nanoseconds, so a class that is rare and
	// long reads the same as one that is frequent and short.
	classes []GapClass

	// opaqueGraphs reports that this capture held graph replay envelopes with
	// no per-node records. Those replays carry no launch correlation, so idle
	// beside them lands in the unattributed arm with a cause that is known
	// and nameable rather than unknown. Without saying so, the finding
	// reports that idle as a mechanism the capture could not identify, when
	// what actually happened is that the instrument could not see through the
	// replay.
	opaqueGraphs bool
}

func idleArms(a GapAttribution) []idleArm {
	arms := []idleArm{
		{
			name:   "host behind",
			nanos:  a.HostBehindNanos,
			remedy: "The device finished its work before the host called the next launch, so the host is the one behind. Submit further ahead: enqueue a run of work rather than one op at a time, move host-side work off the dispatch thread, or capture the sequence into a CUDA graph so a single call enqueues all of it.",
		},
		{
			name:   armDriverSubmission,
			nanos:  a.DriverSubmitNanos,
			remedy: "The device went idle while the host was inside the launch call, so the cost is per call and paid in the driver. Make fewer, larger calls: fuse adjacent launches, or replay the sequence as a CUDA graph. Calling earlier cannot help -- the host had already called.",
		},
		{
			name:   "hardware queue",
			nanos:  a.HardwareNanos,
			remedy: "The launch had already returned before the previous op ended, so the work was queued and the device idled anyway. Neither batching nor earlier submission can recover this time: look at what keeps the queued work from starting -- same-stream ordering, an event wait, or a dependency on the op that just finished.",
		},
		{
			name:   "unattributed",
			nanos:  a.UnattributedNanos,
			remedy: "No host launch record correlated to the op that followed this idle, so this capture names no mechanism for it. Find why the correlation ids of these launches do not reach their activities before choosing a remedy: any remedy chosen here would be chosen from ignorance, not from the measurement.",
		},
	}
	sort.SliceStable(arms, func(i, j int) bool { return arms[i].nanos > arms[j].nanos })
	return arms
}

// captureIdleAttribution attributes the idle time of the device that spent the
// most of it, using the same analysis the gaps command reports.
//
// advise otherwise reads the pprof projection, which is an aggregate: it sums
// durations and drops the per-activity timestamps, so no attribution is
// recoverable from it. These numbers exist only in the capture journals. A
// capture without them, or whose journals are truncated, yields nothing here
// rather than an attribution computed from a surviving prefix.
func captureIdleAttribution(cap *cudacapture.Capture) (idleAttribution, bool) {
	gaps, err := analyzeGaps(cap, gapOptions{maxRecords: controllerMaxJournalRecords, journalBytes: cudacapture.MaxCaptureInputBytes})
	if err != nil {
		return idleAttribution{}, false
	}
	var best idleAttribution
	var found bool
	for _, dev := range gaps.Devices {
		s := dev.Overall
		if s == nil || s.IdleNanos <= 0 || (found && s.IdleNanos <= best.idle) {
			continue
		}
		best = idleAttribution{
			device:       dev.Device,
			idle:         s.IdleNanos,
			duty:         s.DutyCyclePct,
			gaps:         s.GapCount,
			median:       s.Distribution.Median,
			span:         s.SpanNanos,
			maxGap:       s.Distribution.Max,
			classes:      s.Concurrency,
			arms:         idleArms(s.Attribution),
			opaqueGraphs: gaps.OpaqueGraphReplays,
		}
		found = true
	}
	return best, found
}

// dominantIdleGapShare is the share of a capture's idle time that one gap must
// hold before that idle is reported as a single stall rather than as a pattern.
//
// Idle reported as a total and a median invites a per-launch reading: so many
// gaps, so much idle, therefore a tax on every launch. A capture whose idle is
// one 63.8 ms gap out of 70.4 ms admits no such remedy -- batching and
// submitting earlier both divide a cost that was paid once. The threshold sits
// well below a majority because a gap holding a quarter of the idle already
// makes the per-launch average a fiction.
const dominantIdleGapShare = 25.0

// minIdleShareOfSpan is the share of the device timeline that must be idle
// before this report decomposes the idle at all. Below it the device was busy,
// and a finding about the shape of a rounding error would rank microseconds
// against the kernels holding the rest of the span.
const minIdleShareOfSpan = 1.0

// dominantGapConcentration is how many times the typical gap the longest gap
// must be before it is called a single stall.
//
// A share threshold alone does not survive a small gap count: across four
// uniform gaps the largest necessarily holds 25% of the idle, and a finding
// that fires there reports the arithmetic of "largest of four" as a stall.
// What separates a stall from a pattern is that the gap is unlike the others,
// so the test is against the mean of the rest rather than against the total.
const dominantGapConcentration = 10.0

// dominantGap reports the share of idle held by the longest single gap, and
// whether that share is large enough to describe the idle as one stall.
func (a idleAttribution) dominantGap() (float64, bool) {
	if a.idle <= 0 || a.maxGap <= 0 || a.span <= 0 {
		return 0, false
	}
	if float64(a.idle)/float64(a.span)*100.0 < minIdleShareOfSpan {
		return 0, false
	}
	share := float64(a.maxGap) / float64(a.idle) * 100.0
	if share < dominantIdleGapShare {
		return 0, false
	}
	if a.gaps <= 1 {
		// One gap is the whole idle: there is no pattern to be unlike.
		return share, true
	}
	restMean := float64(a.idle-a.maxGap) / float64(a.gaps-1)
	if restMean > 0 && float64(a.maxGap) < dominantGapConcentration*restMean {
		return 0, false
	}
	return share, true
}

// gapClassSplit renders the host concurrency classes with their gap counts.
//
// The idle attribution arms carry nanoseconds alone, so a class that is rare
// and long reads exactly like one that is frequent and short. Those are
// different engineering problems: ten gaps holding 70 ms is a stall, and a
// hundred and sixty seven holding 0.36 ms is a well-fed device.
func gapClassSplit(classes []GapClass) string {
	var b strings.Builder
	for _, c := range classes {
		if c.Count == 0 {
			continue
		}
		if b.Len() > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%s %d gaps holding %.3f ms", c.Name, c.Count, float64(c.IdleNanos)/1e6)
	}
	return b.String()
}

// reportDominantIdleGap reports idle concentrated in one gap, and returns
// whether it emitted a finding.
//
// It fires on the concentration alone, deliberately independent of the duty
// cycle. The same one-time stall in a capture whose kernels run longer is a
// smaller share of a longer span, and a threshold on share-of-span therefore
// makes the finding appear and disappear for reasons that have nothing to do
// with whether it is true. Two arms of one program differing only in cuBLAS
// math mode showed exactly that: identical cold starts, one reported and one
// silent, because the slower arm diluted it.
//
// It names no cause. A stall at the head of a capture is usually module load,
// heuristic selection or workspace allocation, but this capture cannot tell
// that from a mid-run stall on a lock or a fault, and the remedies differ.
// What it establishes is the shape: the time was spent once, so no per-launch
// remedy reaches it.
func reportDominantIdleGap(attr idleAttribution, report *AdviceReport) bool {
	share, ok := attr.dominantGap()
	if !ok {
		return false
	}
	spanShare := float64(attr.maxGap) / float64(attr.span) * 100.0
	// A stall that holds a quarter of the whole timeline is the largest
	// single fact about the run; one that holds a quarter of a small idle
	// total is worth naming but not worth outranking the kernels.
	severity := SeverityWarning
	if spanShare >= dominantIdleGapShare {
		severity = SeverityCritical
	}
	finding := fmt.Sprintf("A single %.3f ms gap holds %.1f%% of device %d's %.3f ms of idle time, and %.1f%% of the %.3f ms captured span.",
		float64(attr.maxGap)/1e6, share, attr.device, float64(attr.idle)/1e6,
		spanShare, float64(attr.span)/1e6)
	if attr.gaps > 1 {
		finding += fmt.Sprintf(" The other %d gaps hold %.3f ms between them (median %.3f us), so the idle is not spread across the dispatches.",
			attr.gaps-1, float64(attr.idle-attr.maxGap)/1e6, float64(attr.median)/1e3)
	}
	if split := gapClassSplit(attr.classes); split != "" {
		finding += " Gaps by host concurrency: " + split + "."
	}
	finding += " Idle this concentrated was spent once, so it is not a per-launch cost: batching launches, submitting further ahead and replaying the sequence as a CUDA graph all divide a cost that was not paid per launch, and none of them recovers this time."
	report.Items = append(report.Items, AdviceItem{
		Severity:       severity,
		Category:       CategoryArchitectureDispatch,
		EvidenceLevel:  EvidenceObserved,
		severityByKind: true,
		// Ranked by the idle it accounts for, on the same timeline as the
		// idle attribution finding: this is idle time, not kernel time.
		DeviceTimeNanos:          attr.maxGap,
		ShareOfProfile:           spanShare,
		magnitude:                fmt.Sprintf("longest gap %.3f ms, %.1f%% of idle", float64(attr.maxGap)/1e6, share),
		Finding:                  finding,
		ActionableRecommendation: "Establish whether this stall is one-time before spending anything on the dispatch pattern. Capture again with a warm-up pass, or exclude the first iteration from the profiled range: a stall that disappears is start-up cost -- module load, cuBLAS heuristic selection, workspace allocation -- and belongs outside the measurement, while one that survives is a real stall to attribute. A single-iteration capture cannot tell the two apart.",
		NextMeasurement:          "re-capture with a warm-up iteration ahead of the profiled range, then compare this gap distribution against cudaprof gaps",
	})
	return true
}

// auditDominantIdleGap emits the single-stall finding for a capture the idle
// attribution finding does not reach, which is any capture busy enough to be
// above deviceIdleCriticalPct. A busy capture can still lose a quarter of its
// span to one stall.
func auditDominantIdleGap(cap *cudacapture.Capture, report *AdviceReport) bool {
	attr, ok := captureIdleAttribution(cap)
	if !ok {
		return false
	}
	return reportDominantIdleGap(attr, report)
}

// auditIdleAttribution names where an idle-dominated capture's idle time went,
// and reports whether it emitted a finding.
//
// auditDeviceDutyCycle states that the idle time is the larger quantity; it
// deliberately does not diagnose it. That left the report with no finding a
// reader could act on for the majority of the span, so the only actionable
// items remained the kernel-time ones -- the minority the duty cycle had just
// said not to act on first. This finding closes that: it ranks by the idle
// time it accounts for, so a device idling in the hardware queue outranks the
// kernel holding the largest share of a much smaller busy total.
func auditIdleAttribution(cap *cudacapture.Capture, report *AdviceReport) bool {
	attr, ok := captureIdleAttribution(cap)
	if !ok {
		return false
	}
	// The single-stall finding comes first: where one gap holds the idle, the
	// split below describes a remainder, and a reader who acts on the split
	// without knowing that spends the effort on the wrong quantity.
	//
	// It is emitted before the duty-cycle gate rather than after it. The gate
	// here reads the gaps analysis while the caller branched on the stream
	// events, and the two compute the duty cycle from different records: a
	// capture the caller sent down this path can still be above the threshold
	// by this measure, and the stall would then be reported by neither. The
	// caller takes exactly one of the two paths, so this cannot double up.
	dominant := reportDominantIdleGap(attr, report)
	if attr.duty >= deviceIdleCriticalPct {
		return false
	}
	top := attr.arms[0]
	if top.nanos <= 0 {
		return false
	}
	share := float64(top.nanos) / float64(attr.idle) * 100.0
	var split strings.Builder
	for i, arm := range attr.arms {
		if i != 0 {
			split.WriteString(", ")
		}
		fmt.Fprintf(&split, "%s %.3f ms (%.1f%%)", arm.name, float64(arm.nanos)/1e6, float64(arm.nanos)/float64(attr.idle)*100.0)
	}
	finding := fmt.Sprintf("Device %d was idle %.3f ms across %d gaps (median %.3f us), and the largest share of that idle -- %.1f%% -- was %s. Idle attribution: %s.",
		attr.device, float64(attr.idle)/1e6, attr.gaps, float64(attr.median)/1e3, share, top.name, split.String())
	// Gap counts beside the durations: the arms are nanoseconds alone, and a
	// mechanism holding most of the idle across a handful of gaps is a stall
	// while the same share spread over hundreds is a per-launch tax. The
	// remedies are opposite, and the totals alone do not separate them.
	if split := gapClassSplit(attr.classes); split != "" {
		finding += " Gaps by host concurrency: " + split + "."
	}
	// A plurality is not a majority, and the four remedies are exclusive: a
	// reader who acts on the top arm alone recovers its share and no more.
	// Saying so here keeps the recommendation below from being read as an
	// account of all the idle time.
	if share < 50.0 {
		finding += " No single mechanism holds a majority of the idle time, so a remedy aimed at one arm can recover at most that arm's share."
	}
	// Replay opacity is a property of the instrument, not of the run. Saying
	// so here separates "this idle has no mechanism" from "this capture
	// cannot see the mechanism": the unattributed time is real either way,
	// and only the second is fixed by re-capturing rather than by changing
	// the program.
	if attr.opaqueGraphs {
		finding += " This capture holds graph replay envelopes with no per-node records, which carry no launch correlation, so idle beside a replay cannot be attributed and lands in the unattributed arm. That idle is real and unattributable here, not absent."
	}
	// Where one gap holds the idle, the arm remedies are answers to a question
	// this capture did not ask. "Submit further ahead" divides a cost paid
	// once; the arm it is attached to is real, but it is the mechanism of a
	// stall, not of a dispatch pattern.
	remedy := top.remedy
	if dominant {
		remedy = "Most of this idle is one gap -- see the single-stall finding, which holds the majority of it. Settle that first: until the stall is excluded or explained, this split describes the remainder and its share of the total is not the quantity to act on. Once it is excluded, the remaining idle admits this arm's remedy: " + top.remedy
	}
	report.Items = append(report.Items, AdviceItem{
		Severity:       SeverityCritical,
		Category:       CategoryArchitectureDispatch,
		EvidenceLevel:  EvidenceObserved,
		severityByKind: true,
		// The ranking key is time on the device timeline, and the time this
		// finding accounts for is idle time. Ranking it by the kernel time it
		// does not hold would sort the majority of the span below the
		// minority -- the ordering this finding exists to correct.
		DeviceTimeNanos:          top.nanos,
		ShareOfProfile:           100.0 - attr.duty,
		magnitude:                fmt.Sprintf("%s: %.2f ms, %.1f%% of idle", top.name, float64(top.nanos)/1e6, share),
		Finding:                  finding,
		ActionableRecommendation: remedy,
		NextMeasurement:          "cudaprof gaps <capture> -- the gap distribution and host concurrency classes behind this split",
	})
	return true
}
