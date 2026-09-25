package main

import (
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/gaps"
)

// Comparing two gap reports is what makes the idle axis usable in a
// before/after test. Summed per-kernel device time cannot see that axis at
// all: a change that moves work out of inter-kernel idle and into the kernels
// themselves raises device-active time while lowering what the run costs, and
// a profiler reporting only the sum scores such a win as a regression. Span,
// active and idle are therefore always reported together, and the span is what
// arbitrates -- it is what the run cost; active and idle only say where it went.
//
// The two captures are analyzed under one set of options, so the steady-state
// windows are cut at the same warmup by construction. diffGapReports checks
// that anyway, because a report can also arrive from somewhere else, and two
// steady-state sections cut at different warmups are unlike windows that
// subtract to a number meaning nothing.

// GapIntDelta is one integer quantity before and after. Delta is target minus
// base, so a negative delta is always less of whatever Name says.
type GapIntDelta struct {
	Name   string `json:"name"`
	Base   int64  `json:"base"`
	Target int64  `json:"target"`
	Delta  int64  `json:"delta"`
}

// GapFloatDelta is one fractional quantity before and after.
type GapFloatDelta struct {
	Name   string  `json:"name"`
	Base   float64 `json:"base"`
	Target float64 `json:"target"`
	Delta  float64 `json:"delta"`
}

func intDelta(name string, base, target int64) GapIntDelta {
	return GapIntDelta{Name: name, Base: base, Target: target, Delta: target - base}
}

func floatDelta(name string, base, target float64) GapFloatDelta {
	return GapFloatDelta{Name: name, Base: base, Target: target, Delta: target - base}
}

// GapDiff is the whole result of one "gaps --diff_base" run.
type GapDiff struct {
	Base     *GapReport        `json:"base"`
	Target   *GapReport        `json:"target"`
	WarmupMS float64           `json:"warmup_ms"`
	Devices  []GapDeviceDiff   `json:"devices"`
	Kernels  *KernelCensusDiff `json:"kernels,omitempty"`

	// Cudaprof is the build that computed the diff.
	Cudaprof buildStamp `json:"cudaprof"`
}

// KernelDelta is one kernel's launch count, device time and place in the run,
// in both arms.
type KernelDelta struct {
	Name              string `json:"name"`
	BaseLaunches      int    `json:"base_launches"`
	TargetLaunches    int    `json:"target_launches"`
	BaseNanos         int64  `json:"base_device_ns"`
	TargetNanos       int64  `json:"target_device_ns"`
	DeltaNanos        int64  `json:"delta_device_ns"`
	LaunchesUnchanged bool   `json:"launches_unchanged"`

	// BaseOrderFrac and TargetOrderFrac are where the kernel's median launch
	// sat in each arm's launch order, and PositionShift is the difference.
	// Relocated marks a kernel whose launch count held still while its
	// launches moved to a different part of the run.
	BaseOrderFrac   float64 `json:"base_median_order_frac"`
	TargetOrderFrac float64 `json:"target_median_order_frac"`
	PositionShift   float64 `json:"median_order_shift"`
	Relocated       bool    `json:"relocated,omitempty"`
}

// relocationShiftFrac is how far a kernel's median launch has to move through
// the run's launch order, as a fraction of it, before the kernel's launch
// count holding still stops being evidence that a window containing it holds
// the same work in both arms.
//
// It decides emphasis, not visibility: the two positions are printed for
// every row whose count did not change, so a reader can see a shift below the
// threshold and judge it against their own window.
const relocationShiftFrac = 0.05

// KernelCensusDiff splits the per-kernel device-time deltas by whether the
// kernel's launch count moved.
//
// The two halves have opposite causes and only one of them is about the change
// under test. A kernel whose launch count moved took a different path: that is
// the change working, or failing to. A kernel whose launch count is identical
// in both arms was not touched -- nothing about it changed -- so device time
// that moved there is pressure the rest of the change put on the device:
// clocks, cache, memory system, occupancy.
//
// Ranking the two together by delta alone hides this, and reliably puts the
// untouched kernel on top: on one A/B that removed launches and got slower,
// the largest single row was a kernel with the same 645 launches in both arms,
// and the actual result -- 74 ms of removed work against 188 ms spread across
// four kernels nobody had touched -- was invisible until launch count was held
// constant.
type KernelCensusDiff struct {
	Changed   []KernelDelta `json:"launches_changed"`
	Unchanged []KernelDelta `json:"launches_unchanged"`
	// Relocated holds the kernels whose launch count is identical and whose
	// launches sit in a different part of the run.
	//
	// Two things put a kernel here: the change moved its launches, or the
	// change altered what runs around it and carried its position along.
	// This does not separate those, and does not claim to. Either one breaks
	// the same thing -- a rate computed over part of the run is dividing by a
	// window that no longer holds the same work in both arms -- and that is
	// what the row is reported for.
	Relocated      []KernelDelta `json:"launches_relocated"`
	ChangedNanos   int64         `json:"launches_changed_delta_ns"`
	UnchangedNanos int64         `json:"launches_unchanged_delta_ns"`
	RelocatedNanos int64         `json:"launches_relocated_delta_ns"`
	// Kernel records whose name the journal did not carry. They are in
	// neither list and in no total, so they are counted here rather than
	// leaving the lists to imply they cover every kernel record.
	UnnamedBase   int `json:"unnamed_kernel_records_base,omitempty"`
	UnnamedTarget int `json:"unnamed_kernel_records_target,omitempty"`

	// Caveat is set when the two arms did not record kernels the same way,
	// which makes a launch count difference a recording difference rather
	// than a change in what ran.
	Caveat string `json:"caveat,omitempty"`

	// PositionCaveat is set when kernels that ran in both arms moved
	// through the run. It is a different claim from Caveat: the records
	// agree about what ran, and disagree about where.
	PositionCaveat string `json:"position_caveat,omitempty"`
}

// censusCaveat names a launch count difference the change did not cause.
//
// A capture whose graph replays are opaque holds no device record for the
// kernels replayed inside them: they ran, and this census counts zero
// launches for them. Diffed against an arm where the same kernels resolve
// into records -- which is exactly what toggling CUDA graphs produces -- every
// such kernel lands at the top of "launch count changed" with its whole device
// time apparently removed. On the go_graph/go_nograph pair that is 36 of the
// 45 changed rows and 2.9 seconds of device time that did not go anywhere.
func censusCaveat(base, target *GapReport) string {
	if base.OpaqueGraphReplays == target.OpaqueGraphReplays {
		return ""
	}
	opaque, resolved := "base", "target"
	if target.OpaqueGraphReplays {
		opaque, resolved = "target", "base"
	}
	return fmt.Sprintf("the %s capture's graph replays are opaque and the %s capture's are not: kernels that ran inside a replay have no device record in the %s arm, so they show a launch count of zero there. A row that falls to or rises from zero is that difference in recording, not a launch the change removed",
		opaque, resolved, opaque)
}

// positionCaveat states that the two arms disagree about WHERE work happened.
//
// Every total in this census is a whole-capture total, and a relocation moves
// no whole-capture total at all: the launch count is identical and the device
// time is whatever it was. A rate measured over a window is not so stable.
// Work that leaves a window has left that rate's denominator without leaving
// the run, which reads as a speedup the run did not get -- on the pair that
// motivated this, 24 repack launches moved out of the prefill window while
// repack's whole-capture time rose 89.934 ms, and the windowed metric called
// it an improvement.
//
// This has to print above the table rather than beside the rows it describes.
// The preamble already carries the general rule, but a sentence printed on
// every report says nothing about THIS pair; a reader learns that relocation
// occurred here only if something fires on it. By the time the section is
// reached the totals have been read and the conclusion drawn.
func positionCaveat(rows []KernelDelta) string {
	if len(rows) == 0 {
		return ""
	}
	worst := rows[0]
	for _, r := range rows[1:] {
		if math.Abs(r.PositionShift) > math.Abs(worst.PositionShift) {
			worst = r
		}
	}
	subject := fmt.Sprintf("%d kernels kept their launch counts", len(rows))
	if len(rows) == 1 {
		subject = "one kernel kept its launch count"
	}
	return fmt.Sprintf("%s and moved through the run, %s furthest at %+.1f points of launch order. The two arms disagree about where work happened, not whether it happened, so these whole-capture totals are unmoved by it and any rate measured over a window -- prefill tokens/s, a phase total -- is not. Check the window before reading a speedup",
		subject, truncateKernelName(worst.Name, 44), worst.PositionShift*100)
}

// diffKernelCensus pairs the two censuses by kernel name.
//
// A kernel present in one arm only is a launch count change from or to zero,
// which is what it is: the change added or removed it. It belongs in Changed
// and never in Unchanged, where "the launch count did not move" would be a
// claim about a kernel that did not run.
func diffKernelCensus(base, target *GapReport) *KernelCensusDiff {
	if len(base.Kernels) == 0 && len(target.Kernels) == 0 {
		return nil
	}
	index := func(rows []gaps.Kernel) map[string]gaps.Kernel {
		m := make(map[string]gaps.Kernel, len(rows))
		for _, r := range rows {
			m[r.Name] = r
		}
		return m
	}
	b, t := index(base.Kernels), index(target.Kernels)
	names := make([]string, 0, len(b)+len(t))
	for n := range b {
		names = append(names, n)
	}
	for n := range t {
		if _, ok := b[n]; !ok {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	out := &KernelCensusDiff{
		UnnamedBase:   base.UnnamedKernelRecords,
		UnnamedTarget: target.UnnamedKernelRecords,
		Caveat:        censusCaveat(base, target),
	}
	for _, n := range names {
		bk, tk := b[n], t[n]
		d := KernelDelta{
			Name:              n,
			BaseLaunches:      bk.Launches,
			TargetLaunches:    tk.Launches,
			BaseNanos:         bk.DeviceNanos,
			TargetNanos:       tk.DeviceNanos,
			DeltaNanos:        tk.DeviceNanos - bk.DeviceNanos,
			LaunchesUnchanged: bk.Launches == tk.Launches,
			BaseOrderFrac:     bk.MedianOrderFrac,
			TargetOrderFrac:   tk.MedianOrderFrac,
		}
		if !d.LaunchesUnchanged {
			out.Changed = append(out.Changed, d)
			out.ChangedNanos += d.DeltaNanos
			continue
		}
		// A position is only comparable between two arms that both ran the
		// kernel. Present in one arm only is a count change, handled above.
		d.PositionShift = d.TargetOrderFrac - d.BaseOrderFrac
		if math.Abs(d.PositionShift) >= relocationShiftFrac {
			d.Relocated = true
			out.Relocated = append(out.Relocated, d)
			out.RelocatedNanos += d.DeltaNanos
			continue
		}
		out.Unchanged = append(out.Unchanged, d)
		out.UnchangedNanos += d.DeltaNanos
	}
	rank := func(rows []KernelDelta) {
		sort.SliceStable(rows, func(i, j int) bool {
			a, b := rows[i].DeltaNanos, rows[j].DeltaNanos
			if a < 0 {
				a = -a
			}
			if b < 0 {
				b = -b
			}
			if a != b {
				return a > b
			}
			return rows[i].Name < rows[j].Name
		})
	}
	rank(out.Changed)
	rank(out.Unchanged)
	rank(out.Relocated)
	out.PositionCaveat = positionCaveat(out.Relocated)
	return out
}

// GapDeviceDiff compares the timelines of one device. A window is absent
// rather than empty when either capture does not have it: a steady-state
// section that exists on one side only is reported as missing, never compared
// against the other side's whole run.
type GapDeviceDiff struct {
	Device      uint32        `json:"device"`
	Overall     *GapStatsDiff `json:"overall,omitempty"`
	SteadyState *GapStatsDiff `json:"steady_state,omitempty"`
	Missing     []string      `json:"missing,omitempty"`
}

// GapStatsDiff compares one window of one device.
type GapStatsDiff struct {
	Label        string        `json:"label"`
	Span         GapIntDelta   `json:"span_ns"`
	Active       GapIntDelta   `json:"active_ns"`
	Idle         GapIntDelta   `json:"idle_ns"`
	DutyCyclePct GapFloatDelta `json:"duty_cycle_pct"`
	GapCount     GapIntDelta   `json:"gap_count"`
	Distribution []GapIntDelta `json:"distribution_ns"`
	DistMean     GapFloatDelta `json:"distribution_mean_ns"`
	Concurrency  []GapIntDelta `json:"host_concurrency_idle_ns"`
	Attribution  []GapIntDelta `json:"idle_attribution_ns"`
	Verdict      string        `json:"verdict"`
}

// diffGapReports compares two reports produced under the same options.
func diffGapReports(base, target *GapReport) (*GapDiff, error) {
	if base.WarmupMS != target.WarmupMS || base.WarmupUntil != target.WarmupUntil {
		return nil, fmt.Errorf("warmup windows differ: base excludes the %s and target the %s, so their steady-state sections cover unlike windows", base.Warmup(), target.Warmup())
	}
	diff := &GapDiff{Base: base, Target: target, WarmupMS: base.WarmupMS, Cudaprof: currentBuild()}
	baseByDevice := indexGapDevices(base)
	targetByDevice := indexGapDevices(target)
	for _, id := range mergedDeviceIDs(baseByDevice, targetByDevice) {
		b, haveBase := baseByDevice[id]
		t, haveTarget := targetByDevice[id]
		entry := GapDeviceDiff{Device: id}
		switch {
		case !haveBase:
			entry.Missing = append(entry.Missing, "base capture recorded no activity on this device")
		case !haveTarget:
			entry.Missing = append(entry.Missing, "target capture recorded no activity on this device")
		default:
			entry.Overall = diffGapStats(b.Overall, t.Overall)
			entry.SteadyState = diffGapStats(b.SteadyState, t.SteadyState)
			if entry.Overall == nil {
				entry.Missing = append(entry.Missing, "overall window missing from one capture")
			}
			if entry.SteadyState == nil {
				entry.Missing = append(entry.Missing, steadyStateAbsence(b, t, base.Warmup()))
			}
		}
		diff.Devices = append(diff.Devices, entry)
	}
	if len(diff.Devices) == 0 {
		return nil, fmt.Errorf("neither capture holds a device timeline to compare")
	}
	diff.Kernels = diffKernelCensus(base, target)
	return diff, nil
}

// steadyStateAbsence says which side lacks a steady-state section. A run whose
// activities all fall inside the warmup has no such section, and reporting the
// side that is missing keeps the absence from reading as a zero.
func steadyStateAbsence(base, target gaps.Device, warmup string) string {
	switch {
	case base.SteadyState == nil && target.SteadyState == nil:
		return fmt.Sprintf("neither capture has activity past the %s, so there is no steady-state window", warmup)
	case base.SteadyState == nil:
		return fmt.Sprintf("base capture has no activity past the %s, so its steady-state window does not exist", warmup)
	default:
		return fmt.Sprintf("target capture has no activity past the %s, so its steady-state window does not exist", warmup)
	}
}

func indexGapDevices(r *GapReport) map[uint32]gaps.Device {
	out := make(map[uint32]gaps.Device, len(r.Devices))
	for _, d := range r.Devices {
		out[d.Device] = d
	}
	return out
}

func mergedDeviceIDs(a, b map[uint32]gaps.Device) []uint32 {
	seen := make(map[uint32]bool, len(a)+len(b))
	var ids []uint32
	for _, m := range []map[uint32]gaps.Device{a, b} {
		for id := range m {
			if !seen[id] {
				seen[id] = true
				ids = append(ids, id)
			}
		}
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

func diffGapStats(base, target *gaps.Stats) *GapStatsDiff {
	if base == nil || target == nil {
		return nil
	}
	d := &GapStatsDiff{
		Label:        base.Label,
		Span:         intDelta("timeline span", base.SpanNanos, target.SpanNanos),
		Active:       intDelta("device active", base.ActiveNanos, target.ActiveNanos),
		Idle:         intDelta("device idle", base.IdleNanos, target.IdleNanos),
		DutyCyclePct: floatDelta("duty cycle", base.DutyCyclePct, target.DutyCyclePct),
		GapCount:     intDelta("gap count", int64(base.GapCount), int64(target.GapCount)),
		DistMean:     floatDelta("mean", base.Distribution.Mean, target.Distribution.Mean),
	}
	bd, td := base.Distribution, target.Distribution
	d.Distribution = []GapIntDelta{
		intDelta("min", bd.Min, td.Min),
		intDelta("p25", bd.P25, td.P25),
		intDelta("median", bd.Median, td.Median),
		intDelta("p75", bd.P75, td.P75),
		intDelta("p90", bd.P90, td.P90),
		intDelta("p99", bd.P99, td.P99),
		intDelta("max", bd.Max, td.Max),
	}
	baseClass := indexGapClasses(base)
	targetClass := indexGapClasses(target)
	for _, name := range []string{gaps.ClassHostAhead, gaps.ClassHostDuring, gaps.ClassHostBehind, gaps.ClassUnmatched} {
		d.Concurrency = append(d.Concurrency, intDelta(name, baseClass[name], targetClass[name]))
	}
	d.Attribution = []GapIntDelta{
		intDelta("host behind", base.Attribution.HostBehindNanos, target.Attribution.HostBehindNanos),
		intDelta("driver submission", base.Attribution.DriverSubmitNanos, target.Attribution.DriverSubmitNanos),
		intDelta("hardware queue", base.Attribution.HardwareNanos, target.Attribution.HardwareNanos),
		// The fourth arm is listed with the other three because the four
		// partition idle. A diff that drops one prints rows that no longer
		// sum to the idle delta above them.
		intDelta("unattributed", base.Attribution.UnattributedNanos, target.Attribution.UnattributedNanos),
	}
	d.Verdict = gapVerdict(d.Span, d.Active, d.Idle)
	return d
}

func indexGapClasses(s *gaps.Stats) map[string]int64 {
	out := make(map[string]int64, len(s.Concurrency))
	for _, c := range s.Concurrency {
		out[c.Name] = c.IdleNanos
	}
	return out
}

// gapVerdict says what moved, in the terms a summed-kernel-time view cannot
// express. The case it exists for is a run whose kernels got slower and whose
// idle got shorter by more: the span falls, the run is faster, and a diff of
// per-kernel device time alone calls it a regression. The reverse case -- a
// span that grew while kernel time shrank -- is the same mistake with the sign
// flipped, and is named just as explicitly.
func gapVerdict(span, active, idle GapIntDelta) string {
	switch {
	case span.Delta < 0 && active.Delta > 0:
		return fmt.Sprintf("span fell by %s while device-active time rose by %s: the win is inter-kernel idle (%s), and summed per-kernel device time alone would score this run a regression",
			gapDuration(span.Delta), gapDuration(active.Delta), signedDuration(idle.Delta))
	case span.Delta > 0 && active.Delta < 0:
		return fmt.Sprintf("span rose by %s while device-active time fell by %s: the loss is inter-kernel idle (%s), and summed per-kernel device time alone would score this run a speedup",
			gapDuration(span.Delta), gapDuration(active.Delta), signedDuration(idle.Delta))
	case span.Delta < 0:
		return fmt.Sprintf("span fell by %s: device active %s, device idle %s", gapDuration(span.Delta), signedDuration(active.Delta), signedDuration(idle.Delta))
	case span.Delta > 0:
		return fmt.Sprintf("span rose by %s: device active %s, device idle %s", gapDuration(span.Delta), signedDuration(active.Delta), signedDuration(idle.Delta))
	case active.Delta == 0 && idle.Delta == 0:
		return "span, device active and device idle are unchanged"
	default:
		return fmt.Sprintf("span is unchanged and the run traded busy time for idle time: device active %s, device idle %s", signedDuration(active.Delta), signedDuration(idle.Delta))
	}
}

// gapDuration prints the magnitude of a duration at a scale that keeps its
// digits. The tables above are held to milliseconds so their columns compare,
// but a verdict is a sentence: a 400 ns change rendered as "0.000 ms" states
// the opposite of what was measured.
func gapDuration(ns int64) string {
	if ns < 0 {
		ns = -ns
	}
	switch {
	case ns >= 1e6:
		return fmt.Sprintf("%.3f ms", float64(ns)/1e6)
	case ns >= 1e3:
		return fmt.Sprintf("%.3f us", float64(ns)/1e3)
	default:
		return fmt.Sprintf("%d ns", ns)
	}
}

func signedDuration(ns int64) string {
	if ns < 0 {
		return "-" + gapDuration(ns)
	}
	return "+" + gapDuration(ns)
}

func signedMicros(ns int64) string { return fmt.Sprintf("%+.3f", float64(ns)/1e3) }

// ---------------------------------------------------------------------------
// reading and rendering

// gapReportForPath reads one capture and analyzes it under opts. Both sides of
// a diff go through it, so neither side can be analyzed under options the other
// did not see.
func gapReportForPath(path string, opts gapOptions) (*GapReport, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	cap, err := readCapturePathLimits(abs, opts.limits)
	if err != nil {
		return nil, fmt.Errorf("read capture %s: %w", path, err)
	}
	report, err := analyzeGaps(cap, opts)
	if err != nil {
		return nil, fmt.Errorf("analyze %s: %w", path, err)
	}
	report.Capture = abs
	return report, nil
}

func (d *GapDiff) text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "cudaprof gaps diff\n")
	fmt.Fprintf(&b, "  base   %s\n", d.Base.Capture)
	fmt.Fprintf(&b, "  target %s\n", d.Target.Capture)
	fmt.Fprintf(&b, "  activities %d -> %d, api records %d -> %d\n",
		d.Base.Activities, d.Target.Activities, d.Base.APIRecords, d.Target.APIRecords)
	for _, dev := range d.Devices {
		for _, s := range []*GapStatsDiff{dev.Overall, dev.SteadyState} {
			if s == nil {
				continue
			}
			fmt.Fprintf(&b, "\ndevice %d, %s", dev.Device, s.Label)
			if s.Label == "steady-state" {
				fmt.Fprintf(&b, " (%s excluded from both)", d.Base.Warmup())
			}
			b.WriteString("\n")
			writeGapStatsDiff(&b, s)
		}
		for _, m := range dev.Missing {
			fmt.Fprintf(&b, "\ndevice %d: %s\n", dev.Device, m)
		}
	}
	writeKernelCensusDiff(&b, d.Kernels)
	return b.String()
}

// kernelCensusRows is how many rows of each half are printed. The totals below
// them are over every row, not over the printed ones.
const kernelCensusRows = 10

func writeKernelCensusDiff(b *strings.Builder, k *KernelCensusDiff) {
	if k == nil {
		return
	}
	b.WriteString("\nper-kernel census (whole capture, not a window)\n")
	// Printed above the two halves, not below them: it says what the first
	// half's largest rows mean, and a reader who has already read them has
	// already drawn the conclusion it exists to prevent.
	if k.Caveat != "" {
		fmt.Fprintf(b, "  caveat: %s\n", k.Caveat)
	}
	if k.PositionCaveat != "" {
		fmt.Fprintf(b, "  caveat: %s\n", k.PositionCaveat)
	}
	section := func(title, note string, rows []KernelDelta, total int64, position bool) {
		fmt.Fprintf(b, "  %s: %s (%s)\n", title, signedDuration(total), note)
		if len(rows) == 0 {
			b.WriteString("    none\n")
			return
		}
		fmt.Fprintf(b, "    %-44s %9s %9s %12s %12s %12s", "kernel", "launches", "->", "device ms", "->", "delta")
		if position {
			fmt.Fprintf(b, "  %15s", "median rank")
		}
		b.WriteString("\n")
		for i, r := range rows {
			if i == kernelCensusRows {
				fmt.Fprintf(b, "    (%d more)\n", len(rows)-kernelCensusRows)
				break
			}
			fmt.Fprintf(b, "    %-44s %9d %9d %12s %12s %12s",
				truncateKernelName(r.Name, 44), r.BaseLaunches, r.TargetLaunches,
				millis(r.BaseNanos), millis(r.TargetNanos), signedDuration(r.DeltaNanos))
			if position {
				fmt.Fprintf(b, "  %5.1f%% -> %5.1f%%", r.BaseOrderFrac*100, r.TargetOrderFrac*100)
			}
			b.WriteString("\n")
		}
	}
	section("launch count changed", "the code path moved: this is the change", k.Changed, k.ChangedNanos, false)
	section("launch count unchanged, position moved", "same launches, different part of the run: a rate measured over a window will disagree with this total, and the window is the thing to check", k.Relocated, k.RelocatedNanos, true)
	section("launch count unchanged, position held", "these kernels were not touched: this is what the change cost elsewhere on the device", k.Unchanged, k.UnchangedNanos, true)
	if k.UnnamedBase != 0 || k.UnnamedTarget != 0 {
		fmt.Fprintf(b, "  %d -> %d kernel records carried no name and are in neither half\n", k.UnnamedBase, k.UnnamedTarget)
	}
}

// truncateKernelName keeps the column aligned without hiding which kernel a
// row is: mangled C++ names differ in their tail, so the tail is what is kept.
func truncateKernelName(name string, width int) string {
	if len(name) <= width {
		return name
	}
	return "..." + name[len(name)-(width-3):]
}

func writeGapStatsDiff(b *strings.Builder, s *GapStatsDiff) {
	writeNanosRow(b, s.Span)
	writeNanosRow(b, s.Active)
	writeNanosRow(b, s.Idle)
	fmt.Fprintf(b, "  %-17s %11.2f%% -> %11.2f%%  (%+.2f pp)\n", s.DutyCyclePct.Name, s.DutyCyclePct.Base, s.DutyCyclePct.Target, s.DutyCyclePct.Delta)
	fmt.Fprintf(b, "  %-17s %12d -> %12d  (%+d)\n", s.GapCount.Name, s.GapCount.Base, s.GapCount.Target, s.GapCount.Delta)
	fmt.Fprintf(b, "  gap distribution (us)\n")
	for _, p := range s.Distribution {
		fmt.Fprintf(b, "    %-8s %10s -> %10s  (%s)\n", p.Name, micros(p.Base), micros(p.Target), signedMicros(p.Delta))
	}
	fmt.Fprintf(b, "    %-8s %10.3f -> %10.3f  (%+.3f)\n", s.DistMean.Name, s.DistMean.Base/1e3, s.DistMean.Target/1e3, s.DistMean.Delta/1e3)
	fmt.Fprintf(b, "  idle by host concurrency class\n")
	for _, c := range s.Concurrency {
		fmt.Fprintf(b, "    %-12s %12s -> %12s  (%s)\n", c.Name, millis(c.Base), millis(c.Target), signedDuration(c.Delta))
	}
	fmt.Fprintf(b, "  idle attribution\n")
	for _, a := range s.Attribution {
		fmt.Fprintf(b, "    %-17s %12s -> %12s  (%s)\n", a.Name, millis(a.Base), millis(a.Target), signedDuration(a.Delta))
	}
	fmt.Fprintf(b, "  verdict: %s\n", s.Verdict)
}

func writeNanosRow(b *strings.Builder, v GapIntDelta) {
	fmt.Fprintf(b, "  %-17s %12s -> %12s  (%s)\n", v.Name, millis(v.Base), millis(v.Target), signedDuration(v.Delta))
}

// ---------------------------------------------------------------------------
// the decomposition printed beside a duration delta

// gapDecompositionMarkdown renders the span/active/idle decomposition of two
// captures for "cudaprof diff --explain". It sits beside the summed kernel
// time because that sum is blind to inter-kernel idle: on its own it scores a
// run that trades kernel time for idle time backwards.
//
// A capture without a usable journal is named with the reason the timeline
// could not be reconstructed, and the rest of the explanation still prints.
func gapDecompositionMarkdown(baseCap, targetCap *cudacapture.Capture, basePath, targetPath string) string {
	opts := gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords}
	baseReport, baseErr := analyzeGaps(baseCap, opts)
	targetReport, targetErr := analyzeGaps(targetCap, opts)
	return gapDecompositionReportsMarkdown(baseReport, baseErr, targetReport, targetErr, basePath, targetPath)
}

// gapDecompositionReportsMarkdown is gapDecompositionMarkdown over gap reports
// already made, with the error each analysis returned.
func gapDecompositionReportsMarkdown(baseReport *GapReport, baseErr error, targetReport *GapReport, targetErr error, basePath, targetPath string) string {
	var sb strings.Builder
	sb.WriteString("## Device Timeline Decomposition\n\n")
	if baseErr != nil || targetErr != nil {
		sb.WriteString("The summed kernel time above cannot see inter-kernel idle. That decomposition is unavailable here:\n\n")
		for _, e := range []struct {
			path string
			err  error
		}{{basePath, baseErr}, {targetPath, targetErr}} {
			if e.err != nil {
				sb.WriteString(fmt.Sprintf("- `%s`: %v\n", e.path, e.err))
			}
		}
		sb.WriteString("\nA change that moves work out of idle and into the kernels therefore cannot be told apart from a regression by the table above alone.\n\n")
		return sb.String()
	}
	baseReport.Capture, targetReport.Capture = basePath, targetPath
	diff, err := diffGapReports(baseReport, targetReport)
	if err != nil {
		sb.WriteString(fmt.Sprintf("Device timelines could not be compared: %v\n\n", err))
		return sb.String()
	}
	sb.WriteString("Span is what the run cost; active and idle say where it went. Summed per-kernel device time is the active column alone.\n\n")
	sb.WriteString("| Device | Window | Metric | Base | Target | Delta |\n|---|---|---|---|---|---|\n")
	for _, dev := range diff.Devices {
		for _, s := range []*GapStatsDiff{dev.Overall, dev.SteadyState} {
			if s == nil {
				continue
			}
			for _, v := range []GapIntDelta{s.Span, s.Active, s.Idle} {
				sb.WriteString(fmt.Sprintf("| %d | %s | %s | %s | %s | %s |\n",
					dev.Device, s.Label, v.Name, millis(v.Base), millis(v.Target), signedDuration(v.Delta)))
			}
			sb.WriteString(fmt.Sprintf("| %d | %s | duty cycle | %.2f%% | %.2f%% | %+.2f pp |\n",
				dev.Device, s.Label, s.DutyCyclePct.Base, s.DutyCyclePct.Target, s.DutyCyclePct.Delta))
			sb.WriteString(fmt.Sprintf("| %d | %s | gap count | %d | %d | %+d |\n",
				dev.Device, s.Label, s.GapCount.Base, s.GapCount.Target, s.GapCount.Delta))
		}
	}
	sb.WriteString("\n")
	for _, dev := range diff.Devices {
		for _, s := range []*GapStatsDiff{dev.Overall, dev.SteadyState} {
			if s == nil {
				continue
			}
			sb.WriteString(fmt.Sprintf("- **device %d, %s**: %s\n", dev.Device, s.Label, s.Verdict))
		}
		for _, m := range dev.Missing {
			sb.WriteString(fmt.Sprintf("- **device %d**: %s\n", dev.Device, m))
		}
	}
	sb.WriteString("\n")
	writeKernelCensusMarkdown(&sb, diff.Kernels)
	return sb.String()
}

// writeKernelCensusMarkdown renders the census beside the duration delta,
// where an A/B is actually read. The duration table above it ranks every
// kernel together by delta, which puts a kernel the change never touched at
// the top whenever the change cost more elsewhere than it saved.
func writeKernelCensusMarkdown(sb *strings.Builder, k *KernelCensusDiff) {
	if k == nil || (len(k.Changed) == 0 && len(k.Relocated) == 0 && len(k.Unchanged) == 0) {
		return
	}
	sb.WriteString("## Per-Kernel Census\n\n")
	sb.WriteString("Whole capture, not a window. A kernel whose launch count moved took a different path: that is the change. A kernel that kept its launch count but whose launches sit in a different part of the run is not untouched -- a rate measured over a window will disagree with these totals. Only a kernel that kept both its count and its place was left alone, and device time that moved there is what the change cost elsewhere on the device.\n\n")
	if k.Caveat != "" {
		sb.WriteString(fmt.Sprintf("**Caveat**: %s.\n\n", k.Caveat))
	}
	if k.PositionCaveat != "" {
		sb.WriteString(fmt.Sprintf("**Caveat**: %s.\n\n", k.PositionCaveat))
	}
	sb.WriteString(fmt.Sprintf("- **Launch count changed**: %s\n", signedDuration(k.ChangedNanos)))
	sb.WriteString(fmt.Sprintf("- **Launch count unchanged, position moved**: %s\n", signedDuration(k.RelocatedNanos)))
	sb.WriteString(fmt.Sprintf("- **Launch count unchanged, position held**: %s\n\n", signedDuration(k.UnchangedNanos)))
	sb.WriteString("| Kernel | Launches | Base | Target | Delta | Median rank | Class |\n|---|---|---|---|---|---|---|\n")
	for _, half := range [][]KernelDelta{k.Changed, k.Relocated, k.Unchanged} {
		for i, r := range half {
			if i == kernelCensusRows {
				sb.WriteString(fmt.Sprintf("| _(%d more)_ | | | | | | |\n", len(half)-kernelCensusRows))
				break
			}
			state, rank := "changed", ""
			if r.LaunchesUnchanged {
				state = "unchanged, position held"
				if r.Relocated {
					state = "unchanged, position moved"
				}
				rank = fmt.Sprintf("%.1f%% -> %.1f%%", r.BaseOrderFrac*100, r.TargetOrderFrac*100)
			}
			sb.WriteString(fmt.Sprintf("| `%s` | %d -> %d | %s | %s | %s | %s | %s |\n",
				r.Name, r.BaseLaunches, r.TargetLaunches, millis(r.BaseNanos), millis(r.TargetNanos), signedDuration(r.DeltaNanos), rank, state))
		}
	}
	sb.WriteString("\n")
}
