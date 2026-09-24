package main

import (
	"fmt"
	"sort"
	"strings"
)

func severityRank(s Severity) int {
	switch s {
	case SeverityReadFirst:
		return 0
	case SeverityCost:
		return 1
	case SeverityContext:
		return 2
	default:
		return 3
	}
}

func finalizeAndRankFindings(report *AdviceReport, kernelTimes map[string]int64, totalTime int64) {
	finalizeAndRankFindingsWithTiming(report, kernelTimes, totalTime, true)
}

func finalizeAndRankFindingsWithTiming(report *AdviceReport, kernelTimes map[string]int64, totalTime int64, hasTiming bool) {
	if report == nil || len(report.Items) == 0 {
		return
	}
	report.Items = foldKernelPatterns(report.Items)
	for i := range report.Items {
		item := &report.Items[i]
		if hasTiming {
			if item.DeviceTimeNanos == 0 && item.Location.Kernel != "" {
				item.DeviceTimeNanos = kernelTimes[item.Location.Kernel]
			}
			if totalTime > 0 && item.DeviceTimeNanos > 0 && item.ShareOfProfile == 0 {
				item.ShareOfProfile = float64(item.DeviceTimeNanos) / float64(totalTime) * 100.0
			}
		} else {
			// Pure PC-sample or sample profile: do not assign sample counts to DeviceTimeNanos!
			// DeviceTimeNanos remains 0.
			if totalTime > 0 && item.ShareOfProfile == 0 && item.Location.Kernel != "" {
				kSamples := kernelTimes[item.Location.Kernel]
				if kSamples > 0 {
					item.ShareOfProfile = float64(kSamples) / float64(totalTime) * 100.0
				}
			}
		}
	}

	// Read first is earned by kind, never by cost. The largest item in a list
	// is still an item in the list: it tells the reader what to look at next,
	// not that the report's own ordering is untrustworthy. Cost used to be
	// able to buy the top tier, guarded only by a share floor, and the floor
	// was a patch over the conflation -- it kept a 0.006 ms finding out of
	// first place without making first place mean anything.
	for i := range report.Items {
		item := &report.Items[i]
		if item.Severity == SeverityReadFirst && !item.severityByKind {
			item.Severity = SeverityCost
		}
		for j := range item.Patterns {
			if p := &item.Patterns[j]; p.Severity == SeverityReadFirst {
				p.Severity = SeverityCost
			}
		}
	}

	// Findings too small to act on stop competing with findings that are not.
	// They stay in the report -- each may still be true -- but they rank below
	// everything worth a reader's attention instead of beside it.
	//
	// Small has to mean both things at once. An absolute floor alone demotes
	// every finding in a short capture, including one holding all of it; a
	// share floor alone keeps a 0.006 ms finding in a 0.01 ms capture ranked
	// beside real expense. A finding is negligible when it is a rounding
	// error in wall time AND a rounding error in the profile.
	for i := range report.Items {
		item := &report.Items[i]
		if item.Severity != SeverityCost || item.severityByKind {
			continue
		}
		if item.DeviceTimeNanos <= 0 || item.DeviceTimeNanos >= negligibleDeviceNanos {
			continue
		}
		if item.ShareOfProfile >= negligibleSharePct {
			continue
		}
		item.Severity = SeverityContext
	}

	sort.SliceStable(report.Items, func(i, j int) bool {
		a, b := report.Items[i], report.Items[j]
		if a.Severity != b.Severity {
			return severityRank(a.Severity) < severityRank(b.Severity)
		}
		if ra, rb := max(a.DeviceTimeNanos, a.HostTimeNanos), max(b.DeviceTimeNanos, b.HostTimeNanos); ra != rb {
			return ra > rb
		}
		if a.ShareOfProfile != b.ShareOfProfile {
			return a.ShareOfProfile > b.ShareOfProfile
		}
		if a.Category != b.Category {
			return a.Category < b.Category
		}
		return a.Finding < b.Finding
	})
}

// takesKernelTime reports whether item is a pattern in one kernel's code
// whose size is that kernel's time, filled in when the report is ranked.
// A dispatch finding names a kernel for how it is launched, not for its code.
func takesKernelTime(item AdviceItem) bool {
	return item.Location.Kernel != "" && !strings.Contains(item.Location.Kernel, " -> ") &&
		!isDispatch(item.Category) &&
		item.DeviceTimeNanos == 0 && item.HostTimeNanos == 0 && item.ShareOfProfile == 0 &&
		item.magnitude == "" && !item.severityByKind && len(item.Patterns) == 0
}

func isDispatch(c Category) bool {
	return c == CategoryDispatch || c == CategoryArchitectureDispatch
}

// foldKernelPatterns replaces the findings that take their size from the same
// kernel with one finding for the kernel that holds them as patterns. Each
// would otherwise be charged the kernel's whole time, and a kernel showing
// five patterns ranked as five findings each worth all of it. A kernel with
// one such finding keeps it as it is.
func foldKernelPatterns(items []AdviceItem) []AdviceItem {
	byKernel := make(map[string][]AdviceItem)
	for _, item := range items {
		if takesKernelTime(item) {
			byKernel[item.Location.Kernel] = append(byKernel[item.Location.Kernel], item)
		}
	}
	var out []AdviceItem
	folded := make(map[string]bool)
	for _, item := range items {
		k := item.Location.Kernel
		patterns := byKernel[k]
		if !takesKernelTime(item) || len(patterns) < 2 {
			out = append(out, item)
			continue
		}
		if folded[k] {
			continue
		}
		folded[k] = true
		kernel := AdviceItem{
			Severity: SeverityContext,
			Category: CategoryKernel,
			Location: Location{Kernel: k},
			Finding:  fmt.Sprintf("%d patterns in kernel %q; the kernel's time is counted once", len(patterns), k),
			Patterns: patterns,
		}
		for _, p := range patterns {
			if severityRank(p.Severity) < severityRank(kernel.Severity) {
				kernel.Severity = p.Severity
			}
		}
		out = append(out, kernel)
	}
	return out
}

// negligibleDeviceNanos is the device time below which a cost finding ranks as
// context rather than beside findings worth acting on. Half a millisecond is
// not a claim about what matters on every capture; it is the point below which
// a reader cannot act on the difference, and three findings worth 0.01 ms,
// 0.01 ms and 0.00 ms rendered as peers of a 9.52 ms one is how a report
// spends attention it did not need to.
const negligibleDeviceNanos = 500_000

// negligibleSharePct is the share of profiled device time below which a cost
// finding is a rounding error in the profile as well as in wall time. Both
// conditions must hold before a finding is demoted: see the demotion loop in
// finalizeAndRankFindingsWithTiming.
const negligibleSharePct = 1.0
