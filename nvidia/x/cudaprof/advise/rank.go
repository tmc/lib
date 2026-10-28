package advise

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

func finalizeAndRankFindings(report *Report, kernelTimes map[string]int64, totalTime int64) {
	finalizeAndRankFindingsWithTiming(report, kernelTimes, totalTime, true)
}

func finalizeAndRankFindingsWithTiming(report *Report, kernelTimes map[string]int64, totalTime int64, hasTiming bool) {
	if report == nil || len(report.Items) == 0 {
		return
	}
	report.Items = foldKernelPatterns(report.Items)
	for i := range report.Items {
		item := &report.Items[i]
		if hasTiming {
			if item.DeviceTimeNanos == 0 && item.Location.Kernel != "" {
				item.DeviceTimeNanos = kernelTimes[item.Location.kernelKey()]
			}
			if totalTime > 0 && item.DeviceTimeNanos > 0 && item.ShareOfProfile == 0 {
				item.ShareOfProfile = float64(item.DeviceTimeNanos) / float64(totalTime) * 100.0
			}
		} else {
			// Pure PC-sample or sample profile: do not assign sample counts to DeviceTimeNanos!
			// DeviceTimeNanos remains 0.
			if totalTime > 0 && item.ShareOfProfile == 0 && item.Location.Kernel != "" {
				kSamples := kernelTimes[item.Location.kernelKey()]
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

	rankItems(report.Items)
	report.Items = groupRules(report.Items)
	rankItems(report.Items)
}

// rankItems sorts items by severity, then size.
func rankItems(items []Item) {
	sort.SliceStable(items, func(i, j int) bool {
		a, b := items[i], items[j]
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

// ruleNames describes each groupable rule, for the finding that holds its
// findings.
var ruleNames = map[string]string{
	"single-warp":         "single-warp occupancy",
	"imad-wide":           "scalar address arithmetic (IMAD.WIDE)",
	"fastmath":            "precise float math with local memory",
	"spill":               "register spill",
	"gpr-limit":           "register limit",
	"gpr-cliff":           "register occupancy cliff",
	"ugpr":                "uniform register pressure",
	"ptx-scalar-load":     "PTX scalar global load",
	"ptx-mad-wide":        "PTX scalar address arithmetic",
	"frame-size":          "local stack frame",
	"maxreg":              "compiler register limit",
	"occupancy":           "theoretical occupancy",
	"kernel-patterns":     "kernel code pattern",
	"unmeasured-sequence": "short-kernel sequence without timing",
	"short-sequence":      "short-kernel sequence duty cycle",
	"host-loop":           "repeated host loop dispatch",
	"round-trip":          "intermediate tensor round-trip",
	"shared-activation":   "shared activation load",
	"fusion":              "pointwise fusion",
	"tensor-path":         "tensor path",
	"warp-mma-tcgen05":    "warp-level MMA where tcgen05 applies",
	"warp-mma-wgmma":      "warp-level MMA where wgmma applies",
	"fp4-dequant":         "FP4 decoded in registers",
}

// groupRules replaces the findings of each rule that fired more than once
// with one finding that states the pattern once and lists its locations.
// Rules that fire once per kernel or per kernel pair produced 9,635 findings
// on one training capture, 6,017 of them the same IMAD.WIDE note in a
// different kernel, and ten of the top twelve on another the same
// single-warp paragraph; a reader needs the pattern, its total, and where.
//
// The group takes the place of its highest-ranked finding. Its device time
// is the sum over its findings only when each names a different kernel;
// kernel pairs overlap, so a group of pairs reports its largest instead.
func groupRules(items []Item) []Item {
	count := make(map[string]int)
	for _, item := range items {
		if ruleNames[item.rule] != "" {
			count[item.rule]++
		}
	}
	held := make(map[string][]Item)
	for _, item := range items {
		if count[item.rule] > 1 {
			held[item.rule] = append(held[item.rule], item)
		}
	}
	var out []Item
	done := make(map[string]bool)
	for _, item := range items {
		r := item.rule
		if count[r] <= 1 {
			out = append(out, item)
			continue
		}
		if done[r] {
			continue
		}
		done[r] = true
		out = append(out, groupItem(r, held[r]))
	}
	return out
}

// groupItem returns the finding that holds held, the findings of rule r in
// rank order.
func groupItem(r string, held []Item) Item {
	name := ruleNames[r]
	first := held[0]
	g := Item{
		Severity:                 first.Severity,
		severityByKind:           first.severityByKind,
		Category:                 first.Category,
		EvidenceLevel:            first.EvidenceLevel,
		ActionableRecommendation: first.ActionableRecommendation,
		NextMeasurement:          first.NextMeasurement,
		rule:                     r,
		grouped:                  true,
		Patterns:                 held,
	}
	kernels := make(map[string]bool)
	distinct := true
	var total, largest int64
	var share float64
	for _, item := range held {
		k := item.Location.kernelKey()
		if k == "" || strings.Contains(k, " -> ") || kernels[k] {
			distinct = false
		}
		kernels[k] = true
		if item.Category != g.Category {
			g.Category = CategoryKernel
		}
		d := max(item.DeviceTimeNanos, item.HostTimeNanos)
		total += d
		largest = max(largest, d)
		share += item.ShareOfProfile
	}
	switch {
	case distinct && total > 0:
		g.Finding = fmt.Sprintf("%d kernels, %.2f ms total: %s", len(held), float64(total)/1e6, name)
		g.DeviceTimeNanos = total
		g.ShareOfProfile = share
	case distinct:
		g.Finding = fmt.Sprintf("%d kernels: %s", len(held), name)
		g.ShareOfProfile = share
	default:
		g.Finding = fmt.Sprintf("%d findings: %s", len(held), name)
		if largest > 0 {
			g.magnitude = fmt.Sprintf("%d findings, the largest %.3f ms", len(held), float64(largest)/1e6)
		}
	}
	return g
}

// takesKernelTime reports whether item is a pattern in one kernel's code
// whose size is that kernel's time, filled in when the report is ranked.
// A dispatch finding names a kernel for how it is launched, not for its code.
func takesKernelTime(item Item) bool {
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
func foldKernelPatterns(items []Item) []Item {
	byKernel := make(map[string][]Item)
	for _, item := range items {
		if takesKernelTime(item) {
			k := item.Location.kernelKey()
			byKernel[k] = append(byKernel[k], item)
		}
	}
	var out []Item
	folded := make(map[string]bool)
	for _, item := range items {
		k := item.Location.kernelKey()
		patterns := byKernel[k]
		if !takesKernelTime(item) || len(patterns) < 2 {
			out = append(out, item)
			continue
		}
		if folded[k] {
			continue
		}
		folded[k] = true
		kernel := Item{
			Severity: SeverityContext,
			Category: CategoryKernel,
			rule:     "kernel-patterns",
			Location: Location{Kernel: item.Location.Kernel, Module: item.Location.Module},
			Finding:  fmt.Sprintf("%d patterns in kernel %q; the kernel's time is counted once", len(patterns), item.Location.Kernel),
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
