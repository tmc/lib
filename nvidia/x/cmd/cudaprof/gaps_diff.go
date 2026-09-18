package main

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
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
	Base     *GapReport      `json:"base"`
	Target   *GapReport      `json:"target"`
	WarmupMS float64         `json:"warmup_ms"`
	Devices  []GapDeviceDiff `json:"devices"`
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
	if base.WarmupMS != target.WarmupMS {
		return nil, fmt.Errorf("warmup windows differ: base excludes the first %g ms and target the first %g ms, so their steady-state sections cover unlike windows", base.WarmupMS, target.WarmupMS)
	}
	diff := &GapDiff{Base: base, Target: target, WarmupMS: base.WarmupMS}
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
				entry.Missing = append(entry.Missing, steadyStateAbsence(b, t, base.WarmupMS))
			}
		}
		diff.Devices = append(diff.Devices, entry)
	}
	if len(diff.Devices) == 0 {
		return nil, fmt.Errorf("neither capture holds a device timeline to compare")
	}
	return diff, nil
}

// steadyStateAbsence says which side lacks a steady-state section. A run whose
// activities all fall inside the warmup has no such section, and reporting the
// side that is missing keeps the absence from reading as a zero.
func steadyStateAbsence(base, target GapDevice, warmupMS float64) string {
	switch {
	case base.SteadyState == nil && target.SteadyState == nil:
		return fmt.Sprintf("neither capture has activity past the first %g ms, so there is no steady-state window", warmupMS)
	case base.SteadyState == nil:
		return fmt.Sprintf("base capture has no activity past the first %g ms, so its steady-state window does not exist", warmupMS)
	default:
		return fmt.Sprintf("target capture has no activity past the first %g ms, so its steady-state window does not exist", warmupMS)
	}
}

func indexGapDevices(r *GapReport) map[uint32]GapDevice {
	out := make(map[uint32]GapDevice, len(r.Devices))
	for _, d := range r.Devices {
		out[d.Device] = d
	}
	return out
}

func mergedDeviceIDs(a, b map[uint32]GapDevice) []uint32 {
	seen := make(map[uint32]bool, len(a)+len(b))
	var ids []uint32
	for _, m := range []map[uint32]GapDevice{a, b} {
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

func diffGapStats(base, target *GapStats) *GapStatsDiff {
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
	for _, name := range []string{classHostAhead, classHostDuring, classHostBehind, classUnmatched} {
		d.Concurrency = append(d.Concurrency, intDelta(name, baseClass[name], targetClass[name]))
	}
	d.Attribution = []GapIntDelta{
		intDelta("host behind", base.Attribution.HostBehindNanos, target.Attribution.HostBehindNanos),
		intDelta("driver submission", base.Attribution.DriverSubmitNanos, target.Attribution.DriverSubmitNanos),
		intDelta("hardware queue", base.Attribution.HardwareNanos, target.Attribution.HardwareNanos),
	}
	d.Verdict = gapVerdict(d.Span, d.Active, d.Idle)
	return d
}

func indexGapClasses(s *GapStats) map[string]int64 {
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
				fmt.Fprintf(&b, " (first %g ms excluded from both)", d.WarmupMS)
			}
			b.WriteString("\n")
			writeGapStatsDiff(&b, s)
		}
		for _, m := range dev.Missing {
			fmt.Fprintf(&b, "\ndevice %d: %s\n", dev.Device, m)
		}
	}
	return b.String()
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
// captures for "cudaprof diff --explain". It sits beside the summed per-kernel
// duration delta because that sum is blind to inter-kernel idle: on its own it
// scores a run that trades kernel time for idle time backwards.
//
// A capture without a usable journal is named with the reason the timeline
// could not be reconstructed, and the rest of the explanation still prints:
// the duration delta is not wrong, it is only incomplete.
func gapDecompositionMarkdown(baseCap, targetCap *cudacapture.Capture, basePath, targetPath string) string {
	opts := gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords}
	var sb strings.Builder
	sb.WriteString("## Device Timeline Decomposition\n\n")
	baseReport, baseErr := analyzeGaps(baseCap, opts)
	targetReport, targetErr := analyzeGaps(targetCap, opts)
	if baseErr != nil || targetErr != nil {
		sb.WriteString("The duration delta above sums per-kernel device time, which cannot see inter-kernel idle. That decomposition is unavailable here:\n\n")
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
	return sb.String()
}
