package main

import (
	"fmt"
	"math"
	"strings"

	"github.com/google/pprof/profile"
)

// maxCoverageSkewPct is the largest difference, in percentage points, between
// the two arms' kernel coverage at which their summed kernel times are still
// compared as a ratio. Beyond it the arms did not record the same share of
// their device work as kernels, and a ratio of the sums measures the
// recording, not the run.
const maxCoverageSkewPct = 1.0

// armTotals is one arm of a diff: its summed kernel time beside the device
// timeline that time must be read against.
type armTotals struct {
	// summedNanos is the sum of the profile's kernel time samples over every
	// kernel, whatever --kernel selects.
	summedNanos int64

	// activeNanos is device-active time summed across devices and spanNanos
	// is the longest device timeline span, both from the capture journal.
	// They are meaningful only when unread is empty.
	activeNanos int64
	spanNanos   int64

	// opaqueGraphs reports that the journal held graph replay envelopes and
	// no node records, so activeNanos counts whole replays.
	opaqueGraphs bool

	// unread is why the device timeline could not be read. Empty when it was.
	unread string
}

// coveragePct is summed kernel time as a percentage of device-active time.
// It can exceed 100 when kernels overlap on a device.
func (a armTotals) coveragePct() float64 {
	return float64(a.summedNanos) * 100 / float64(a.activeNanos)
}

// readArmTotals reads one arm's totals from its profile and its gap report.
// A report that failed, or that holds no device timeline, leaves the arm
// UNREAD with the reason.
func readArmTotals(p *profile.Profile, r *GapReport, err error) armTotals {
	a := armTotals{summedNanos: summedKernelNanos(p)}
	if err != nil {
		a.unread = err.Error()
		return a
	}
	read := false
	for _, d := range r.Devices {
		if d.Overall == nil {
			continue
		}
		read = true
		a.activeNanos += d.Overall.ActiveNanos
		a.spanNanos = max(a.spanNanos, d.Overall.SpanNanos)
	}
	if !read {
		a.unread = "the journal holds no device timeline"
	}
	a.opaqueGraphs = r.OpaqueGraphReplays
	return a
}

// summedKernelNanos sums p's kernel time samples over every kernel.
func summedKernelNanos(p *profile.Profile) int64 {
	idx := findTimeIndex(p)
	if idx < 0 {
		return 0
	}
	var n int64
	for _, s := range p.Sample {
		n += s.Value[idx]
	}
	return n
}

// summedKernelVerdict reports whether the two arms' summed kernel times may be
// compared as a ratio, and when not, why. A ratio is printed only when both
// device timelines were read, both arms recorded kernel time against nonzero
// device-active time, their coverage agrees within maxCoverageSkewPct, and
// the captured span did not move against the summed kernel time.
func summedKernelVerdict(base, target armTotals) (ok bool, reason string) {
	for _, arm := range []struct {
		name string
		t    armTotals
	}{{"base", base}, {"target", target}} {
		switch {
		case arm.t.unread != "":
			return false, fmt.Sprintf("%s coverage UNREAD: %s", arm.name, arm.t.unread)
		case arm.t.activeNanos <= 0:
			return false, fmt.Sprintf("%s device-active time is zero, so its coverage is undefined", arm.name)
		case arm.t.summedNanos <= 0:
			return false, fmt.Sprintf("%s recorded no kernel time", arm.name)
		}
	}
	if skew := math.Abs(base.coveragePct() - target.coveragePct()); skew > maxCoverageSkewPct {
		return false, fmt.Sprintf("kernel coverage differs by %.2f pp (base %.2f%%, target %.2f%%), so the sums do not describe the same share of device work",
			skew, base.coveragePct(), target.coveragePct())
	}
	summed := target.summedNanos - base.summedNanos
	span := target.spanNanos - base.spanNanos
	if summed != 0 && span != 0 && (summed > 0) != (span > 0) {
		return false, fmt.Sprintf("summed kernel time moved %s but captured span moved %s",
			signedDuration(summed), signedDuration(span))
	}
	return true, ""
}

// printedDeltaVerdict is summedKernelVerdict for the delta a report prints.
// With --kernel that delta covers only the selected kernels, and it must
// agree with the captured span as the whole sum must: a selected kernel whose
// time rose while the span fell is not a comparison the span supports.
func printedDeltaVerdict(c *DurationComparison, base, target armTotals) (ok bool, reason string) {
	if ok, reason := summedKernelVerdict(base, target); !ok {
		return false, reason
	}
	span := target.spanNanos - base.spanNanos
	if c.DeltaNs != 0 && span != 0 && (c.DeltaNs > 0) != (span > 0) {
		return false, fmt.Sprintf("the selected kernels' summed time moved %s but captured span moved %s",
			signedDuration(c.DeltaNs), signedDuration(span))
	}
	return true, ""
}

// writeSummedKernelMarkdown renders the summed kernel time of both arms with
// the coverage and captured span each must be read against, and the
// comparison c of the kernels --kernel selected. The ratio and the per-kernel
// ratios are printed only when printedDeltaVerdict allows it.
func writeSummedKernelMarkdown(sb *strings.Builder, c *DurationComparison, base, target armTotals, kernelFilter string) {
	sb.WriteString("## Summed Kernel Time\n\n")
	if c.BaseTotalNs <= 0 && c.TargetTotalNs <= 0 {
		sb.WriteString("No kernel time samples recorded in profiles.\n\n")
		return
	}
	sb.WriteString("Summed kernel time adds each arm's kernel time samples. It is not wall time: overlapping kernels add, idle does not, and kernels the capture did not resolve are absent. Read it against the device-active time it covers and the span the capture spent.\n\n")
	writeArmLine(sb, "base", base)
	writeArmLine(sb, "target", target)

	ok, reason := printedDeltaVerdict(c, base, target)
	label := "summed kernel time"
	if kernelFilter != "" {
		label = fmt.Sprintf("summed kernel time of kernels matching %q", kernelFilter)
	}
	delta := formatDeltaUs(float64(c.DeltaNs) / 1000)
	if ok && c.BaseTotalNs > 0 && c.TargetTotalNs > 0 {
		fmt.Fprintf(sb, "- **Delta**: %s %s (%+.1f%%, base/target %.2f×)\n\n",
			label, delta, float64(c.DeltaNs)*100/float64(c.BaseTotalNs), c.Speedup)
	} else if ok {
		fmt.Fprintf(sb, "- **Delta**: %s %s; no ratio: one arm recorded none of these kernels\n\n", label, delta)
	} else {
		fmt.Fprintf(sb, "- **Delta**: %s %s; no ratio: %s\n\n", label, delta, reason)
	}

	if len(c.Kernels) > 1 {
		sb.WriteString("| Kernel | Base (µs) | Target (µs) | Delta (µs) | Base/Target |\n")
		sb.WriteString("|---|---|---|---|---|\n")
		for _, k := range c.Kernels {
			var ratio string
			switch {
			case k.TargetNs == 0:
				ratio = "only in base"
			case k.BaseNs == 0:
				ratio = "only in target"
			case !ok:
				ratio = "withheld"
			default:
				ratio = fmt.Sprintf("%.2f×", k.Speedup)
			}
			fmt.Fprintf(sb, "| `%s` | %.2f | %.2f | %+.2f | %s |\n",
				k.Name, float64(k.BaseNs)/1000, float64(k.TargetNs)/1000, float64(k.DeltaNs)/1000, ratio)
		}
		sb.WriteString("\n")
	}
}

func writeArmLine(sb *strings.Builder, name string, a armTotals) {
	summed := formatRuntime(float64(a.summedNanos) / 1000)
	if a.unread != "" {
		fmt.Fprintf(sb, "- **%s**: summed kernel time %s; device active, coverage and captured span UNREAD: %s\n", name, summed, a.unread)
		return
	}
	coverage := "coverage UNREAD (no device-active time)"
	if a.activeNanos > 0 {
		coverage = fmt.Sprintf("coverage %.2f%%", a.coveragePct())
	}
	fmt.Fprintf(sb, "- **%s**: summed kernel time %s, device active %s, %s, captured span %s\n",
		name, summed, millis(a.activeNanos), coverage, millis(a.spanNanos))
	if a.opaqueGraphs {
		fmt.Fprintf(sb, "  - graph replays were recorded as envelopes only: device active counts each whole replay and summed kernel time holds none of its kernels\n")
	}
}

// durationSynthesis is the synthesis line for summed kernel time: its
// direction when the arms may be compared, and the reason when not.
func durationSynthesis(c *DurationComparison, base, target armTotals) string {
	if c == nil {
		return ""
	}
	ok, reason := printedDeltaVerdict(c, base, target)
	switch {
	case !ok:
		return "Summed kernel time is not compared: " + reason + "."
	case c.DeltaNs < 0:
		return fmt.Sprintf("Summed kernel time fell by %s with coverage and captured span in agreement.", millis(-c.DeltaNs))
	case c.DeltaNs > 0:
		return fmt.Sprintf("Summed kernel time rose by %s with coverage and captured span in agreement.", millis(c.DeltaNs))
	}
	return ""
}
