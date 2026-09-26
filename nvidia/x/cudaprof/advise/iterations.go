package advise

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
)

// AddIterations adds the findings of a per-iteration analysis to r, after
// the findings to read first: they are costs of the loop being measured,
// and they are ranked among themselves by time per iteration, not by the
// whole capture's device time the other findings rank by. A failed
// iteration gate is itself a finding to read first.
func (r *Report) AddIterations(it *iteration.Report) {
	var items []Item
	if !it.Gate.Pass {
		items = append(items, Item{
			Severity:                 SeverityReadFirst,
			Category:                 CategoryDispatch,
			EvidenceLevel:            EvidenceObserved,
			Finding:                  "iteration gate failed: " + strings.Join(it.Gate.Failures, "; "),
			ActionableRecommendation: fmt.Sprintf("Per-iteration figures below average iterations that did different work. Check -iteration-marker %s and -skip-iterations %d against the program's loop, or capture a range of identical iterations.", it.Marker, it.Skip),
			severityByKind:           true,
		})
	}
	n := int64(it.Steady.Iterations)
	for _, f := range it.Advise() {
		item := Item{
			Severity:                 SeverityCost,
			Category:                 CategoryDispatch,
			EvidenceLevel:            EvidenceObserved,
			Finding:                  "every iteration: " + f.Finding,
			ActionableRecommendation: f.Action,
		}
		if w := it.Steady.WindowNanos / n; w > 0 {
			item.magnitude = fmt.Sprintf("%.3f ms per iteration, %.1f%% of the iteration", float64(f.Nanos)/1e6, 100*float64(f.Nanos)/float64(w))
		}
		switch f.Rule {
		case "idle", "host-sync":
			item.DeviceTimeNanos = f.Nanos * n
		default:
			item.HostTimeNanos = f.Nanos * n
		}
		items = append(items, item)
	}
	// The whole-capture idle findings measure warm-up with the loop: on a
	// training capture the largest was a 789 ms start-up gap. The
	// per-iteration idle finding measures the loop alone, so they are
	// demoted to context beneath it.
	demoted := false
	for i := range r.Items {
		item := &r.Items[i]
		if item.rule != "capture-idle" {
			continue
		}
		item.Severity = SeverityContext
		item.severityByKind = false
		item.Finding += fmt.Sprintf(" This counts the whole capture, warm-up included; the per-iteration findings skip the first %d iterations.", it.Skip)
		demoted = true
	}
	if demoted {
		sort.SliceStable(r.Items, func(i, j int) bool {
			return severityRank(r.Items[i].Severity) < severityRank(r.Items[j].Severity)
		})
	}
	i := 0
	for i < len(r.Items) && r.Items[i].Severity == SeverityReadFirst {
		i++
	}
	r.Items = append(r.Items[:i], append(items, r.Items[i:]...)...)
}
