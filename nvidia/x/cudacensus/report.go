package cudacensus

import (
	"fmt"
	"sort"
)

// Phase describes a named execution interval on the device clock.
type Phase struct {
	Name    string
	StartNs int64
	EndNs   int64
}

// DurationNs returns the duration between StartNs and EndNs.
func (p Phase) DurationNs() int64 {
	return p.EndNs - p.StartNs
}

// Phase returns a Phase for the first span with the given name, if present.
func (c *Capture) Phase(name string) (Phase, bool) {
	s, ok := c.Span(name)
	if !ok {
		return Phase{}, false
	}
	return Phase{Name: s.Name, StartNs: s.StartNs, EndNs: s.EndNs}, true
}

// TransfersWindow returns the transfers that started at or after startNs and
// before endNs. Both bounds are on the device clock.
func (c *Capture) TransfersWindow(startNs, endNs int64) []Transfer {
	lo := sort.Search(len(c.Transfers), func(i int) bool { return c.Transfers[i].StartNs >= startNs })
	hi := sort.Search(len(c.Transfers), func(i int) bool { return c.Transfers[i].StartNs >= endNs })
	return c.Transfers[lo:hi]
}

// ActiveNs computes the union-of-device-active duration across launches,
// merging any overlapping or contiguous intervals on the device clock.
//
// When kernels execute concurrently across multiple streams, summing their
// individual durations yields a figure larger than the physical device time
// elapsed. ActiveNs returns the actual elapsed device-busy interval.
func ActiveNs(launches []Launch) int64 {
	if len(launches) == 0 {
		return 0
	}
	intervals := make([][2]int64, len(launches))
	for i, l := range launches {
		intervals[i] = [2]int64{l.StartNs, l.EndNs}
	}
	return ActiveIntervalNs(intervals)
}

// TransfersActiveNs computes the union-of-device-active duration across transfers.
func TransfersActiveNs(transfers []Transfer) int64 {
	if len(transfers) == 0 {
		return 0
	}
	intervals := make([][2]int64, len(transfers))
	for i, t := range transfers {
		intervals[i] = [2]int64{t.StartNs, t.EndNs}
	}
	return ActiveIntervalNs(intervals)
}

// ActiveIntervalNs computes the union duration across arbitrary [start, end] intervals.
func ActiveIntervalNs(intervals [][2]int64) int64 {
	if len(intervals) == 0 {
		return 0
	}
	sorted := make([][2]int64, len(intervals))
	copy(sorted, intervals)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i][0] == sorted[j][0] {
			return sorted[i][1] < sorted[j][1]
		}
		return sorted[i][0] < sorted[j][0]
	})

	var total int64
	curStart := sorted[0][0]
	curEnd := sorted[0][1]

	for _, it := range sorted[1:] {
		if it[0] <= curEnd {
			if it[1] > curEnd {
				curEnd = it[1]
			}
		} else {
			total += curEnd - curStart
			curStart = it[0]
			curEnd = it[1]
		}
	}
	total += curEnd - curStart
	return total
}

// FamilyRule defines a stable, versioned rule mapping a kernel symbol to an
// architectural reporting family.
type FamilyRule struct {
	Name  string
	Match func(symbol string) bool
}

// ClassifySymbol matches a symbol against rules in order, returning the
// matching family name or "unknown" if no rule matches.
func ClassifySymbol(symbol string, rules []FamilyRule) string {
	for _, r := range rules {
		if r.Match != nil && r.Match(symbol) {
			return r.Name
		}
	}
	return "unknown"
}

// FamilySummary aggregates metrics for all kernels matching one reporting family.
type FamilySummary struct {
	Name           string
	Launches       int
	TotalDeviceNs  int64
	ActiveDeviceNs int64
	Bytes          uint64
	FLOPs          uint64
	Kernels        int // Count of distinct kernel names
}

// PhaseReport summarizes the execution metrics and family decomposition within one phase.
type PhaseReport struct {
	Phase          Phase
	Launches       int
	TotalDeviceNs  int64 // Sum of individual kernel durations
	ActiveDeviceNs int64 // Union of concurrent device intervals
	Transfers      int
	TransferBytes  uint64
	Families       []FamilySummary
	Entries        []Entry
	CoverageCount  int // Launches classified into a known family
	UnknownCount   int // Launches classified as "unknown"
}

// Report is diagnostic output over one or more phases in a capture.
type Report struct {
	Complete      bool     // True if all phases had valid bounds
	Missing       []string // Explanations of invalid or missing phase bounds
	UnknownLaunch int      // Total launches across all phases with no matching family rule
	Phases        []PhaseReport
}

// Analyze evaluates the capture across the provided phases using the given
// coster and family classification rules.
func (c *Capture) Analyze(phases []Phase, coster Coster, rules ...FamilyRule) Report {
	rep := Report{
		Complete: true,
	}
	if c == nil {
		rep.Complete = false
		rep.Missing = append(rep.Missing, "capture is nil")
		return rep
	}

	for _, p := range phases {
		pr := PhaseReport{
			Phase: p,
		}
		if p.EndNs <= p.StartNs {
			rep.Complete = false
			rep.Missing = append(rep.Missing, fmt.Sprintf("phase %q has non-positive duration [%d, %d]", p.Name, p.StartNs, p.EndNs))
			rep.Phases = append(rep.Phases, pr)
			continue
		}

		launches := c.Window(p.StartNs, p.EndNs)
		transfers := c.TransfersWindow(p.StartNs, p.EndNs)

		pr.Launches = len(launches)
		pr.TotalDeviceNs = 0
		for _, l := range launches {
			pr.TotalDeviceNs += l.DurationNs()
		}
		pr.ActiveDeviceNs = ActiveNs(launches)

		pr.Transfers = len(transfers)
		for _, t := range transfers {
			pr.TransferBytes += t.Bytes
		}

		pr.Entries = Census(launches, coster)

		// Group by family
		type famAcc struct {
			summary  FamilySummary
			launches []Launch
		}
		byFam := make(map[string]*famAcc)
		var famOrder []string

		for _, e := range pr.Entries {
			fam := ClassifySymbol(e.Name, rules)
			acc := byFam[fam]
			if acc == nil {
				acc = &famAcc{
					summary: FamilySummary{Name: fam},
				}
				byFam[fam] = acc
				famOrder = append(famOrder, fam)
			}
			acc.summary.Launches += e.Launches
			acc.summary.TotalDeviceNs += e.DeviceNs
			acc.summary.Bytes += e.Bytes
			acc.summary.FLOPs += e.FLOPs
			acc.summary.Kernels++

			if fam == "unknown" {
				pr.UnknownCount += e.Launches
				rep.UnknownLaunch += e.Launches
			} else {
				pr.CoverageCount += e.Launches
			}
		}

		// Calculate ActiveDeviceNs for each family from its underlying launches
		for _, l := range launches {
			fam := ClassifySymbol(l.Name, rules)
			if acc := byFam[fam]; acc != nil {
				acc.launches = append(acc.launches, l)
			}
		}

		for _, fam := range famOrder {
			acc := byFam[fam]
			acc.summary.ActiveDeviceNs = ActiveNs(acc.launches)
			pr.Families = append(pr.Families, acc.summary)
		}

		rep.Phases = append(rep.Phases, pr)
	}

	return rep
}
