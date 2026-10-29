package advise

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// A Rank is one rank's capture in a multi-rank analysis.
type Rank struct {
	// Name identifies the rank in findings, such as "rank3" or a capture
	// file name.
	Name    string
	Capture *cudacapture.Capture

	// OffsetNanos is added to this rank's device timestamps under
	// [AlignClock], to bring captures taken on different hosts onto one
	// clock. It is ignored under [AlignCollective], which measures the
	// offset.
	OffsetNanos int64
}

// An Alignment says how [AnalyzeRanks] puts the ranks' timelines on one
// clock.
type Alignment string

const (
	// AlignCollective shifts each rank so that matched collectives end
	// together. A collective cannot complete on any rank until the last
	// rank has arrived, so on an NVLink domain the ends agree to within the
	// transfer's tail while the starts differ by exactly the arrival skew
	// being measured. The shift is the median over matched collectives, so
	// one collective that ended late on one rank does not move the rest.
	AlignCollective Alignment = "collective"

	// AlignClock takes each rank's timestamps as they are, plus its
	// OffsetNanos. It is correct for ranks captured on one host, whose
	// CUPTI timestamps share a clock.
	AlignClock Alignment = "clock"
)

// RankSummary is one rank's share of a multi-rank analysis.
type RankSummary struct {
	Name string `json:"name"`

	// OffsetNanos is the shift applied to this rank's timestamps.
	OffsetNanos int64 `json:"offset_nanos"`

	SpanNanos    int64 `json:"span_nanos"`
	ComputeNanos int64 `json:"compute_nanos"`
	CommNanos    int64 `json:"comm_nanos"`

	// ExposedCommNanos is communication time during which this rank ran no
	// compute on any stream.
	ExposedCommNanos int64 `json:"exposed_comm_nanos"`

	// WaitNanos is time this rank spent inside matched collectives before
	// the last rank arrived: the time it waited on others.
	WaitNanos int64 `json:"wait_nanos"`

	// Last counts the matched collectives this rank entered last, which
	// are the ones every other rank waited on it for.
	Last int `json:"last"`

	// MeanLateNanos is how far after the first rank this rank entered a
	// matched collective, on average.
	MeanLateNanos int64 `json:"mean_late_nanos"`
}

// RankTimeline is the cross-rank part of a multi-rank report.
type RankTimeline struct {
	Alignment Alignment     `json:"alignment"`
	Ranks     []RankSummary `json:"ranks"`

	// Collectives is the number of collectives matched on every rank, and
	// Unmatched the number seen on some ranks and not others. A collective
	// is matched by kernel name and its order among collectives of that
	// name.
	Collectives int `json:"collectives"`
	Unmatched   int `json:"unmatched,omitempty"`

	// MeanSkewNanos and MaxSkewNanos are the spread between the first and
	// last rank to enter a matched collective.
	MeanSkewNanos int64 `json:"mean_skew_nanos"`
	MaxSkewNanos  int64 `json:"max_skew_nanos"`

	// Straggler is the rank that entered the most matched collectives last,
	// or "" when no rank did so in a majority of them.
	Straggler string `json:"straggler,omitempty"`
}

// AnalyzeRanks analyzes the captures of the ranks of one job together. It
// reports how far apart the ranks enter their collectives, which rank the
// others wait on, and each rank's exposed communication. The kernel findings
// of each rank are [Analyze]'s to report, one capture at a time.
func AnalyzeRanks(ranks []Rank, align Alignment) (*Report, error) {
	if len(ranks) < 2 {
		return nil, errors.New("analyze ranks: need at least two ranks")
	}
	if align != AlignCollective && align != AlignClock {
		return nil, fmt.Errorf("analyze ranks: unknown alignment %q", align)
	}
	type rankData struct {
		streams map[uint32][]timeline.Event
		comm    map[string][]timeline.Event // by kernel name, in start order
		summary commSummary
		hasComm bool
	}
	data := make([]rankData, len(ranks))
	for i, r := range ranks {
		if r.Capture == nil {
			return nil, fmt.Errorf("analyze ranks: rank %s: nil capture", r.Name)
		}
		streams, _ := timeline.Events(r.Capture)
		d := rankData{streams: streams, comm: map[string][]timeline.Event{}}
		for _, evs := range streams {
			for _, ev := range evs {
				if ev.End > ev.Start && commKind(ev) != "" {
					d.comm[ev.Name] = append(d.comm[ev.Name], ev)
				}
			}
		}
		for _, evs := range d.comm {
			sort.Slice(evs, func(i, j int) bool { return evs[i].Start < evs[j].Start })
		}
		d.summary, d.hasComm = summarizeCommunication(streams)
		data[i] = d
	}

	// Match the k-th collective of each name across ranks.
	type match struct{ evs []timeline.Event }
	var matches []match
	tl := &RankTimeline{Alignment: align}
	names := map[string]bool{}
	for _, d := range data {
		for n := range d.comm {
			names[n] = true
		}
	}
	sorted := make([]string, 0, len(names))
	for n := range names {
		sorted = append(sorted, n)
	}
	sort.Strings(sorted)
	for _, n := range sorted {
		lo, hi := -1, 0
		for _, d := range data {
			c := len(d.comm[n])
			if lo < 0 || c < lo {
				lo = c
			}
			hi = max(hi, c)
		}
		tl.Unmatched += hi - lo
		for k := 0; k < lo; k++ {
			m := match{evs: make([]timeline.Event, len(data))}
			for i, d := range data {
				m.evs[i] = d.comm[n][k]
			}
			matches = append(matches, m)
		}
	}
	tl.Collectives = len(matches)

	offsets := make([]int64, len(ranks))
	switch align {
	case AlignClock:
		for i, r := range ranks {
			offsets[i] = r.OffsetNanos
		}
	case AlignCollective:
		if len(matches) == 0 {
			return nil, errors.New("analyze ranks: no collective appears on every rank, so there is nothing to align on; use clock alignment")
		}
		for i := range ranks {
			d := make([]int64, len(matches))
			for k, m := range matches {
				d[k] = int64(m.evs[0].End) - int64(m.evs[i].End)
			}
			sort.Slice(d, func(a, b int) bool { return d[a] < d[b] })
			offsets[i] = d[len(d)/2]
		}
	}

	sums := make([]RankSummary, len(ranks))
	late := make([]int64, len(ranks))
	for i, r := range ranks {
		s := data[i].summary
		sums[i] = RankSummary{Name: r.Name, OffsetNanos: offsets[i]}
		if data[i].hasComm {
			sums[i].SpanNanos = int64(s.span)
			sums[i].ComputeNanos = int64(s.compute)
			sums[i].CommNanos = int64(s.comm)
			sums[i].ExposedCommNanos = int64(s.exposed)
		}
	}
	var skewTotal int64
	for _, m := range matches {
		first, last, lastRank := int64(0), int64(0), 0
		for i, ev := range m.evs {
			start := int64(ev.Start) + offsets[i]
			if i == 0 || start < first {
				first = start
			}
			if i == 0 || start > last {
				last, lastRank = start, i
			}
		}
		skew := last - first
		skewTotal += skew
		tl.MaxSkewNanos = max(tl.MaxSkewNanos, skew)
		sums[lastRank].Last++
		for i, ev := range m.evs {
			start := int64(ev.Start) + offsets[i]
			late[i] += start - first
			// A rank cannot wait past its own collective's end.
			sums[i].WaitNanos += min(last-start, int64(ev.End-ev.Start))
		}
	}
	if n := int64(len(matches)); n > 0 {
		tl.MeanSkewNanos = skewTotal / n
		for i := range sums {
			sums[i].MeanLateNanos = late[i] / n
		}
	}
	best := 0
	for i := range sums {
		if sums[i].Last > sums[best].Last {
			best = i
		}
	}
	if len(matches) > 0 && sums[best].Last*2 > len(matches) {
		tl.Straggler = sums[best].Name
	}
	tl.Ranks = sums

	report := &Report{Items: []Item{}, Ranks: tl}
	evidence := EvidenceObserved
	alignNote := "on their capture clocks"
	if align == AlignCollective {
		// The skew is measured against an alignment it assumed: that
		// matched collectives end together.
		evidence = EvidenceSuspected
		alignNote = "after aligning the ranks on the ends of matched collectives, which assumes a collective completes on every rank at once"
	}
	if len(matches) > 0 {
		var waited int64
		for i := range sums {
			waited += sums[i].WaitNanos
		}
		finding := fmt.Sprintf("Across %d collectives matched on all %d ranks, the ranks entered each collective a mean %.1f us apart (max %.1f us), %s; together they spent %.3f ms waiting inside collectives for the last rank to arrive.",
			len(matches), len(ranks), float64(tl.MeanSkewNanos)/1e3, float64(tl.MaxSkewNanos)/1e3, alignNote, float64(waited)/1e6)
		rec := "No rank arrives last in a majority of collectives, so the skew is spread rather than owned: look at what varies per step -- host jitter, data-dependent work such as MoE routing, or launch order -- before treating one rank as slow."
		if tl.Straggler != "" {
			s := sums[best]
			finding += fmt.Sprintf(" Rank %s entered %d of them last, a mean %.1f us after the first rank: the others wait on it.", s.Name, s.Last, float64(s.MeanLateNanos)/1e3)
			rec = fmt.Sprintf("Find what rank %s does before its collectives that the others do not: run advise on its capture alone and compare its compute time and idle against another rank's. A consistently late rank sets every collective's duration for all of them, so its pre-collective work is worth more than any transfer tuning.", s.Name)
		}
		if tl.Unmatched > 0 {
			finding += fmt.Sprintf(" %d collectives appeared on some ranks and not others and were not matched.", tl.Unmatched)
		}
		report.Items = append(report.Items, Item{
			rule:                     "rank-skew",
			Severity:                 SeverityCost,
			Category:                 CategoryCommunication,
			EvidenceLevel:            evidence,
			DeviceTimeNanos:          waited,
			magnitude:                fmt.Sprintf("%.3f ms waited across ranks", float64(waited)/1e6),
			Finding:                  finding,
			ActionableRecommendation: rec,
			NextMeasurement:          "per-rank NVTX ranges around each step's pre-collective work, captured on every rank",
		})
	}
	for i, s := range sums {
		if s.ExposedCommNanos <= 0 || s.SpanNanos <= 0 {
			continue
		}
		pct := float64(s.ExposedCommNanos) / float64(s.SpanNanos) * 100
		finding := fmt.Sprintf("Rank %s: %.3f ms of %.3f ms communication was exposed (%.1f%% of its %.3f ms span). Communication: %s.",
			s.Name, float64(s.ExposedCommNanos)/1e6, float64(s.CommNanos)/1e6, pct, float64(s.SpanNanos)/1e6, data[i].summary.describeKernels())
		if s.WaitNanos > 0 {
			finding += fmt.Sprintf(" %.3f ms of its time in matched collectives was waiting for the last rank, so at most %.3f ms of the exposed time was transfer.",
				float64(s.WaitNanos)/1e6, float64(max(0, s.ExposedCommNanos-s.WaitNanos))/1e6)
		}
		severity := SeverityCost
		if pct < 1 {
			severity = SeverityContext
		}
		report.Items = append(report.Items, Item{
			rule:                     "exposed-comm",
			Severity:                 severity,
			Category:                 CategoryCommunication,
			EvidenceLevel:            EvidenceObserved,
			DeviceTimeNanos:          s.ExposedCommNanos,
			ShareOfProfile:           pct,
			magnitude:                fmt.Sprintf("rank %s, %.1f%% of its span exposed", s.Name, pct),
			Finding:                  finding,
			ActionableRecommendation: "Overlap the exposed collectives with independent compute on another stream, or fuse them into the producing kernel's epilogue; time spent waiting for a late rank is recovered at that rank, not here.",
			NextMeasurement:          "re-capture every rank after the change and compare exposed communication per rank",
		})
	}
	sort.SliceStable(report.Items, func(a, b int) bool {
		return report.Items[a].DeviceTimeNanos > report.Items[b].DeviceTimeNanos
	})
	return report, nil
}

// writeRanks renders the per-rank table of a multi-rank report.
func (tl *RankTimeline) writeRanks(sb *strings.Builder) {
	fmt.Fprintf(sb, "## Ranks\n\n%d ranks, %d matched collectives, aligned by %s; mean skew %.1f us, max %.1f us",
		len(tl.Ranks), tl.Collectives, tl.Alignment, float64(tl.MeanSkewNanos)/1e3, float64(tl.MaxSkewNanos)/1e3)
	if tl.Straggler != "" {
		fmt.Fprintf(sb, "; straggler %s", tl.Straggler)
	}
	sb.WriteString(".\n\n| rank | span ms | compute ms | comm ms | exposed ms | waited ms | last | mean late us |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range tl.Ranks {
		fmt.Fprintf(sb, "| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %d | %.1f |\n",
			r.Name, float64(r.SpanNanos)/1e6, float64(r.ComputeNanos)/1e6, float64(r.CommNanos)/1e6,
			float64(r.ExposedCommNanos)/1e6, float64(r.WaitNanos)/1e6, r.Last, float64(r.MeanLateNanos)/1e3)
	}
	sb.WriteString("\n")
}
