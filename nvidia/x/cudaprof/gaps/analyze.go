package gaps

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Options control an analysis.
type Options struct {
	// WarmupMS excludes from the steady state the activities starting within
	// this many milliseconds of the first.
	WarmupMS float64
	// WarmupUntil, when set, replaces WarmupMS: the steady state begins at
	// the first kernel whose name matches this regular expression.
	WarmupUntil string
	// MaxRecords bounds the records decoded from each journal, as in
	// [timeline.Decode].
	MaxRecords int64
	// JournalBytes bounds the decompressed bytes of each journal. Zero means
	// [cudacapture.MaxCaptureInputBytes].
	JournalBytes int64
}

// Analyze reconstructs the device timeline of cap from its journals and
// reports its duty cycle and idle gaps per device.
//
// A capture whose journals [timeline.Journals] refuses is refused with that
// error as is, so a caller can test it against [timeline.ErrJournalOmitted]
// and [timeline.ErrRecordsDropped].
func Analyze(cap *cudacapture.Capture, opts Options) (*Report, error) {
	journals, err := timeline.Journals(cap)
	if err != nil {
		return nil, err
	}
	journalBytes := opts.JournalBytes
	if journalBytes == 0 {
		journalBytes = cudacapture.MaxCaptureInputBytes
	}
	records, err := timeline.Decode(journals, opts.MaxRecords, journalBytes)
	if err != nil {
		return nil, err
	}
	// A graph replay envelope is excluded from the device timeline because it
	// spans the records of the nodes replayed inside it, and counting both
	// would charge that device time twice. When CUPTI emitted the envelopes
	// and no record inside them -- no device record names a graph id -- the
	// envelope spans nothing, so excluding it deletes the whole replay from
	// the timeline and reports its duration as device idle. On one capture
	// that read 14.54% duty cycle where the work was about 71%.
	//
	// The capture carries its own discriminator, so this is decided rather
	// than assumed. With no node records the envelope is the only evidence
	// the graph ran and is counted as the activity it is. What is then lost
	// is the idle BETWEEN nodes, which is counted as busy -- a bounded error
	// in the honest direction, unlike deleting the replay outright.
	//
	// The size of that error is measured, not argued. Two captures of one
	// workload from a single lock hold, one with CUDA graphs disabled so the
	// same work resolves into real kernel records and no promotion happens:
	//
	//	graphs on    duty 72.24%   active 3845.843 ms   169 envelopes
	//	graphs off   duty 70.83%   active 3738.689 ms     0 envelopes
	//
	// 1.4 points apart, the promoted arm high by 107 ms, which is the idle
	// between nodes that this counts as busy. The control needs a second arm
	// of the same workload and so is not reproducible from any one capture.
	graphsAreOpaque := len(records.Graphs) != 0 && records.GraphNodes == 0
	if graphsAreOpaque {
		records.Activities = append(records.Activities, records.Graphs...)
		records.Graphs = nil
	}
	sort.SliceStable(records.Activities, func(i, j int) bool {
		a, b := records.Activities[i], records.Activities[j]
		if a.Start != b.Start {
			return a.Start < b.Start
		}
		return a.End < b.End
	})
	matched := 0
	for _, a := range records.Activities {
		if _, ok := records.APIs[a.Correlation]; ok {
			matched++
		}
	}
	report := &Report{
		Journals:            records.Journals,
		Activities:          len(records.Activities),
		APIRecords:          len(records.APIs),
		Correlated:          float64(matched) / float64(len(records.Activities)) * 100,
		WarmupMS:            opts.WarmupMS,
		WarmupUntil:         opts.WarmupUntil,
		OpaqueGraphReplays:  graphsAreOpaque,
		UnresolvedSyncs:     records.UnresolvedSyncs,
		UnresolvedSyncNanos: records.UnresolvedNanos,
	}
	report.Kernels, report.UnnamedKernelRecords = kernelCensus(records.Activities)
	byDevice := make(map[uint32][]timeline.Activity)
	for _, a := range records.Activities {
		byDevice[a.Device] = append(byDevice[a.Device], a)
	}
	devices := make([]uint32, 0, len(byDevice))
	for d := range byDevice {
		devices = append(devices, d)
	}
	sort.Slice(devices, func(i, j int) bool { return devices[i] < devices[j] })
	warmupNanos := int64(opts.WarmupMS * 1e6)
	var warmupUntil *regexp.Regexp
	if opts.WarmupUntil != "" {
		re, err := regexp.Compile(opts.WarmupUntil)
		if err != nil {
			return nil, fmt.Errorf("warmup kernel pattern: %w", err)
		}
		warmupUntil = re
	}
	for _, d := range devices {
		acts := byDevice[d]
		// Overhead spans carry no device, so every device is offered all of
		// them; the sync and graph spans are filtered to this one.
		spans := gapSpans{
			overhead: records.Overheads,
			sync:     spansForDevice(records.Syncs, d),
			graph:    graphSpansForDevice(records.Graphs, d),
		}
		steady := Steady(acts, warmupNanos, warmupUntil)
		entry := Device{Device: d, Overall: computeStats(acts, records.APIs, "overall", spans), SteadyActivities: len(steady)}
		if len(steady) != 0 && len(steady) != len(acts) {
			entry.SteadyState = computeStats(steady, records.APIs, "steady-state", spans)
		}
		report.Devices = append(report.Devices, entry)
	}
	return report, nil
}

// Steady returns the activities past the warmup: those starting
// at or after the first kernel matching until, or, when until is nil, those
// starting more than warmupNanos after the first activity. acts is sorted by
// start.
func Steady(acts []timeline.Activity, warmupNanos int64, until *regexp.Regexp) []timeline.Activity {
	if until != nil {
		for i, a := range acts {
			if strings.EqualFold(a.Subtype, "kernel") && until.MatchString(a.Name) {
				return acts[i:]
			}
		}
		return nil
	}
	t0 := acts[0].Start
	for i, a := range acts {
		if a.Start-t0 > warmupNanos {
			return acts[i:]
		}
	}
	return nil
}

// Warmup describes what the steady-state window excludes.
func (r *Report) Warmup() string {
	if r.WarmupUntil != "" {
		return fmt.Sprintf("activity before the first kernel matching %q", r.WarmupUntil)
	}
	return fmt.Sprintf("first %g ms", r.WarmupMS)
}

// kernelCensus counts launches and device time per kernel name.
//
// Both numbers come from the same records on purpose. A census that takes
// counts from one source and durations from another can report a kernel whose
// count held still while its time moved when what actually happened is that
// the two sources disagree about which records exist -- and that reading is
// the one a caller acts on, because a launch count that did not move says the
// kernel itself did not change.
func kernelCensus(acts []timeline.Activity) ([]Kernel, int) {
	byName := make(map[string]*Kernel)
	ranks := make(map[string][]int)
	unnamed := 0
	// acts arrives sorted by start, so the running count is the launch order.
	// Unnamed kernel records advance it too: they are launches, and leaving
	// them out would move every later kernel's rank by however many of them
	// the capture happened to carry.
	launches := 0
	for _, a := range acts {
		if !strings.EqualFold(a.Subtype, "kernel") {
			continue
		}
		rank := launches
		launches++
		if a.Name == "" {
			unnamed++
			continue
		}
		c := byName[a.Name]
		if c == nil {
			c = &Kernel{Name: a.Name}
			byName[a.Name] = c
		}
		c.Launches++
		c.DeviceNanos += a.End - a.Start
		ranks[a.Name] = append(ranks[a.Name], rank)
	}
	out := make([]Kernel, 0, len(byName))
	for _, c := range byName {
		// The median, not the mean: one launch at the far end of the run
		// moves a mean past the shift being measured and leaves the median
		// where the bulk of the launches actually are.
		if launches > 1 {
			r := ranks[c.Name]
			c.MedianOrderFrac = float64(r[len(r)/2]) / float64(launches-1)
		}
		out = append(out, *c)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].DeviceNanos != out[j].DeviceNanos {
			return out[i].DeviceNanos > out[j].DeviceNanos
		}
		return out[i].Name < out[j].Name
	})
	return out, unnamed
}

// mergedInterval is one span of contiguous device occupancy. first is the
// activity that opened the interval; its correlation is what the next gap is
// classified against.
type mergedInterval struct {
	start, end int64
	first      timeline.Activity
}

// mergeIntervals collapses overlapping and touching activity intervals into
// the spans during which the device was busy. Activities must be sorted by
// start then end. A new activity that starts at or before the running end
// extends it; anything later opens a new interval.
func mergeIntervals(acts []timeline.Activity) []mergedInterval {
	var merged []mergedInterval
	for _, a := range acts {
		if len(merged) == 0 {
			merged = append(merged, mergedInterval{start: a.Start, end: a.End, first: a})
			continue
		}
		last := &merged[len(merged)-1]
		if a.Start <= last.end {
			if a.End > last.end {
				last.end = a.End
			}
			continue
		}
		merged = append(merged, mergedInterval{start: a.Start, end: a.End, first: a})
	}
	return merged
}

// gapSpans are the non-device spans offered to one device's timeline.
type gapSpans struct {
	overhead, sync, graph []timeline.Span
}

// graphSpansForDevice is spansForDevice over the graph envelopes, which are
// kept as activities so they can be promoted onto the timeline whole.
func graphSpansForDevice(spans []timeline.Activity, device uint32) []timeline.Span {
	var out []timeline.Span
	for _, s := range spans {
		if s.Device == device {
			out = append(out, timeline.Span{Device: s.Device, Start: s.Start, End: s.End})
		}
	}
	return out
}

func spansForDevice(spans []timeline.Span, device uint32) []timeline.Span {
	var out []timeline.Span
	for _, s := range spans {
		if s.Device == device {
			out = append(out, s)
		}
	}
	return out
}

// overlayIdle reports how much of the idle in gaps is covered by spans, and
// how many gaps at least one span touched.
//
// Overlapping spans are counted once: the answer is the measure of the union
// of the intersections, not the sum of them. A CUPTI buffer flush nested
// inside a graph replay would otherwise be charged twice to one gap, and a
// figure derived that way can exceed the idle it is a part of.
func overlayIdle(gaps, spans []timeline.Span) (int64, int) {
	if len(gaps) == 0 || len(spans) == 0 {
		return 0, 0
	}
	var total int64
	var touched int
	for _, g := range gaps {
		var pieces []timeline.Span
		for _, s := range spans {
			start, end := max64(g.Start, s.Start), min64(g.End, s.End)
			if end > start {
				pieces = append(pieces, timeline.Span{Start: start, End: end})
			}
		}
		if len(pieces) == 0 {
			continue
		}
		touched++
		sort.Slice(pieces, func(i, j int) bool { return pieces[i].Start < pieces[j].Start })
		cur := pieces[0]
		for _, p := range pieces[1:] {
			if p.Start <= cur.End {
				if p.End > cur.End {
					cur.End = p.End
				}
				continue
			}
			total += cur.End - cur.Start
			cur = p
		}
		total += cur.End - cur.Start
	}
	return total, touched
}

func computeStats(acts []timeline.Activity, apis map[uint32]timeline.Span, label string, spans gapSpans) *Stats {
	if len(acts) == 0 {
		return nil
	}
	merged := mergeIntervals(acts)
	stats := &Stats{Label: label, Activities: len(acts), MergedIntervals: len(merged)}
	// The span runs to the greatest end, which is not the end of the last
	// activity by start order: acts is sorted by (start, end), so a long
	// activity containing a shorter one sorts first and its end is the larger.
	// Taking acts[len-1].end there reports a span shorter than the busy time
	// measured inside it, and "% of span" then exceeds 100. merged already
	// carries a running maximum, so its last end is the true one.
	stats.SpanNanos = merged[len(merged)-1].end - merged[0].start
	for _, m := range merged {
		stats.ActiveNanos += m.end - m.start
	}
	classGaps := map[string][]int64{}
	var gaps []int64
	var idle []timeline.Span
	for i := 0; i+1 < len(merged); i++ {
		prevEnd, nextStart := merged[i].end, merged[i+1].start
		gap := nextStart - prevEnd
		if gap <= 0 {
			continue
		}
		gaps = append(gaps, gap)
		idle = append(idle, timeline.Span{Start: prevEnd, End: nextStart})
		api, ok := apis[merged[i+1].first.Correlation]
		if !ok {
			classGaps[ClassUnmatched] = append(classGaps[ClassUnmatched], gap)
			stats.Attribution.UnattributedNanos += gap
			continue
		}
		switch {
		case api.End <= prevEnd:
			classGaps[ClassHostAhead] = append(classGaps[ClassHostAhead], gap)
			stats.Attribution.HardwareNanos += gap
		case api.Start <= prevEnd:
			classGaps[ClassHostDuring] = append(classGaps[ClassHostDuring], gap)
			stats.Attribution.DriverSubmitNanos += min64(api.End-prevEnd, gap)
			stats.Attribution.HardwareNanos += max64(0, nextStart-api.End)
		default:
			classGaps[ClassHostBehind] = append(classGaps[ClassHostBehind], gap)
			starved := min64(api.Start-prevEnd, gap)
			stats.Attribution.HostBehindNanos += starved
			stats.Attribution.DriverSubmitNanos += min64(api.End-api.Start, max64(0, gap-starved))
			stats.Attribution.HardwareNanos += max64(0, nextStart-api.End)
		}
	}
	for _, g := range gaps {
		stats.IdleNanos += g
	}
	stats.GapCount = len(gaps)
	stats.Distribution = describeGaps(gaps)
	stats.Overlay.CuptiOverheadNanos, stats.Overlay.CuptiOverheadGaps = overlayIdle(idle, spans.overhead)
	stats.Overlay.SyncNanos, stats.Overlay.SyncGaps = overlayIdle(idle, spans.sync)
	stats.Overlay.GraphReplayNanos, stats.Overlay.GraphReplayGaps = overlayIdle(idle, spans.graph)
	if total := stats.ActiveNanos + stats.IdleNanos; total > 0 {
		stats.DutyCyclePct = float64(stats.ActiveNanos) / float64(total) * 100
	}
	for _, name := range []string{ClassHostAhead, ClassHostDuring, ClassHostBehind, ClassUnmatched} {
		g := classGaps[name]
		class := Class{Name: name, Meaning: classMeaning[name], Count: len(g)}
		for _, v := range g {
			class.IdleNanos += v
		}
		if len(g) != 0 {
			sorted := append([]int64(nil), g...)
			sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
			class.Median = percentileNanos(sorted, 50)
			class.Mean = float64(class.IdleNanos) / float64(len(g))
		}
		stats.Concurrency = append(stats.Concurrency, class)
	}
	return stats
}

// describeGaps reports the gap distribution. The median is reported alongside
// the mean everywhere: on sub-millisecond work one scheduling hiccup moves the
// mean past the effect being measured and leaves the median where it was.
func describeGaps(gaps []int64) Distribution {
	if len(gaps) == 0 {
		return Distribution{}
	}
	sorted := append([]int64(nil), gaps...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	var sum int64
	for _, g := range sorted {
		sum += g
	}
	return Distribution{
		Min:    sorted[0],
		P25:    percentileNanos(sorted, 25),
		Median: percentileNanos(sorted, 50),
		P75:    percentileNanos(sorted, 75),
		P90:    percentileNanos(sorted, 90),
		P99:    percentileNanos(sorted, 99),
		Max:    sorted[len(sorted)-1],
		Mean:   float64(sum) / float64(len(sorted)),
	}
}

// percentileNanos indexes a sorted sample. It is the nearest-rank position
// used by the reference analyzer: no interpolation, so every reported value is
// an observed gap rather than one no gap ever took.
func percentileNanos(sorted []int64, p float64) int64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(p / 100 * float64(len(sorted)-1))
	if i < 0 {
		i = 0
	}
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
