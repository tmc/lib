package cudaschedule

import (
	"fmt"
	"math"
)

// Replay applies a recorded subset of injection decisions to pristine PTX.
// Every site must exactly match Analyze(src), and no site may occur twice.
// Decisions retain their supplied order. Options supplies the original seed,
// intensity and delay ceiling; Focus does not redraw or change decisions.
// Replaying the complete Perturb decision list with the original options
// produces the same PTX and report. An empty subset returns src unchanged.
func Replay(src string, opts Options, decisions []InjectionDecision) (Rewrite, error) {
	if math.IsNaN(opts.Intensity) || math.IsInf(opts.Intensity, 0) || opts.Intensity < 0 {
		return Rewrite{}, fmt.Errorf("cudaschedule: intensity must be finite and non-negative")
	}
	sites, err := Analyze(src)
	if err != nil {
		return Rewrite{}, err
	}
	if len(decisions) > len(sites) {
		return Rewrite{}, fmt.Errorf("cudaschedule: replay has more decisions than source sites")
	}
	maxSleep := opts.MaxSleepNs
	if maxSleep == 0 {
		maxSleep = DefaultMaxSleepNs
	}
	ceiling := scaledMax(opts.Intensity, maxSleep)
	seen := make(map[int]bool, len(decisions))
	copyDecisions := make([]InjectionDecision, len(decisions))
	for i, decision := range decisions {
		ordinal := decision.Site.Ordinal
		if ordinal < 0 || ordinal >= len(sites) || !sameReplaySite(decision.Site, sites[ordinal]) {
			return Rewrite{}, fmt.Errorf("cudaschedule: replay decision %d does not match source site", i)
		}
		if seen[ordinal] {
			return Rewrite{}, fmt.Errorf("cudaschedule: replay decision %d repeats site %d", i, ordinal)
		}
		seen[ordinal] = true
		if decision.BeforeNs > ceiling || decision.AfterNs > ceiling ||
			(opts.Intensity == 0 && (decision.BeforeNs != 0 || decision.AfterNs != 0)) {
			return Rewrite{}, fmt.Errorf("cudaschedule: replay decision %d exceeds delay ceiling", i)
		}
		if decision.Site.Kind == SiteBackedge && decision.AfterNs != 0 {
			return Rewrite{}, fmt.Errorf("cudaschedule: replay decision %d sleeps after backedge", i)
		}
		copyDecisions[i] = decision
		if decision.Site.BlockIndex != nil {
			block := *decision.Site.BlockIndex
			copyDecisions[i].Site.BlockIndex = &block
		}
	}
	return rewriteDecisions(src, opts.Seed, opts.Intensity, len(sites), copyDecisions), nil
}

func sameReplaySite(a, b Site) bool {
	if (a.BlockIndex == nil) != (b.BlockIndex == nil) {
		return false
	}
	if a.BlockIndex != nil && *a.BlockIndex != *b.BlockIndex {
		return false
	}
	a.BlockIndex, b.BlockIndex = nil, nil
	return a == b
}
