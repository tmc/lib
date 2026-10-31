package advise

import (
	"fmt"
	"sort"
	"strings"
)

// A report ranked against a base orders the target's findings by how much of
// the gap to a base capture each could explain. The gap is the Baseline's:
// its windows, its decomposition and its refusals. Each finding moves one or
// more terms of the comparison's identity:
//
//   - a dispatch finding moves idle, the time the device waits on the host;
//   - a synchronous copy moves memcpy and idle;
//   - a finding in a kernel's code moves the exclusive time of the kernel's op;
//   - any other finding moves no term.
//
// A finding's bound is the delta of its terms, capped at the finding's own
// time when that is known. It is an upper bound on what the finding could
// explain, not an estimate of what fixing it saves, and findings that share
// a term share its bound.

// A Baseline is a comparison of the analyzed capture, the target, against a
// base capture. Every delta is target minus base, in nanoseconds.
type Baseline interface {
	// String describes the two captures, their windows and the identity the
	// terms satisfy, as lines of plain text each ending in a newline.
	String() string

	// TermDelta returns the delta of the named term of the comparison's
	// identity: "window", "kernel", "memcpy", "memset", "idle", "lead_in",
	// "tail" or "unresolved". A term the comparison lacks has a delta of zero.
	TermDelta(term string) int64

	// OpDelta returns the op the named kernel's time is counted under and
	// the delta of that op's exclusive time. It reports ok false when the
	// comparison did not measure ops.
	OpDelta(kernel string) (op string, delta int64, ok bool)
}

// GapBound is what one finding could explain of the gap to a base capture.
type GapBound struct {
	// Term names the comparison terms the finding moves, such as "idle" or
	// "op rms", joined by " + ".
	Term           string `json:"term"`
	TermDeltaNanos int64  `json:"term_delta_ns"`

	// BoundNanos is at most TermDeltaNanos. It is zero or negative when the
	// term moved in the target's favour, and then the finding explains none
	// of the gap.
	BoundNanos int64 `json:"bound_ns"`
}

// The sections of a report ranked against a base, in order.
const (
	gapReadFirst = iota
	gapExplains
	gapNoTerm
	gapAgainst
)

var gapHeadings = [...]string{
	gapReadFirst: "## Read first",
	gapExplains:  "## Findings that could explain the gap, by bound",
	gapNoTerm:    "## Findings that move no term of the comparison",
	gapAgainst:   "## Findings that do not explain the gap: their term did not grow",
}

func gapSection(item Item) int {
	switch {
	case item.Severity == SeverityReadFirst:
		return gapReadFirst
	case item.Gap == nil:
		return gapNoTerm
	case item.Gap.BoundNanos > 0:
		return gapExplains
	}
	return gapAgainst
}

// RankAgainst sets r.Base to base, sets each item's Gap from it, and orders
// the items by section, and within a section by bound. Read first stays
// first: it says how to read the rest, whatever its bound.
func (r *Report) RankAgainst(base Baseline) {
	clearNoRuntimeAttribution(r.Items, nil)
	r.Base = base
	for i := range r.Items {
		r.Items[i].Gap = gapBound(r.Items[i], base)
	}
	sort.SliceStable(r.Items, func(i, j int) bool {
		a, b := r.Items[i], r.Items[j]
		if sa, sb := gapSection(a), gapSection(b); sa != sb {
			return sa < sb
		}
		return a.Gap != nil && b.Gap != nil && a.Gap.BoundNanos > b.Gap.BoundNanos
	})
}

// gapBound returns what item could explain of base's gap, or nil when it
// moves no term.
func gapBound(item Item, base Baseline) *GapBound {
	if item.NoRuntimeAttribution {
		return nil
	}
	var names []string
	var delta int64
	switch {
	case item.terms != nil:
		for _, t := range item.terms {
			names = append(names, t)
			delta += base.TermDelta(t)
		}
	case isDispatch(item.Category):
		names = []string{"idle"}
		delta = base.TermDelta("idle")
	case item.Location.Kernel != "":
		// A fusion finding names a pair; it moves each distinct op once. An
		// op absent from both windows has a delta of zero.
		seen := make(map[string]bool)
		for _, k := range strings.Split(item.Location.Kernel, " -> ") {
			op, d, ok := base.OpDelta(k)
			if !ok {
				return nil
			}
			if seen[op] {
				continue
			}
			seen[op] = true
			names = append(names, "op "+op)
			delta += d
		}
	default:
		return nil
	}
	bound := delta
	if own := max(item.DeviceTimeNanos, item.HostTimeNanos); delta > 0 && own > 0 {
		bound = min(delta, own)
	}
	return &GapBound{Term: strings.Join(names, " + "), TermDeltaNanos: delta, BoundNanos: bound}
}

// writeBase writes the comparison that ranks the report.
func (r *Report) writeBase(sb *strings.Builder) {
	b := r.Base
	sb.WriteString("## Against base\n\n```\n")
	sb.WriteString(b.String())
	sb.WriteString("```\n\n")
	ms := func(name string) string { return fmt.Sprintf("%+.2f", float64(b.TermDelta(name))/1e6) }
	between := b.TermDelta("idle") - b.TermDelta("lead_in") - b.TermDelta("tail")
	fmt.Fprintf(sb, "Terms, ms, target minus base: window %s = kernel %s + memcpy %s + memset %s + idle %s (lead-in %s, between %+.2f, tail %s) + unresolved %s.\n\n",
		ms("window"), ms("kernel"), ms("memcpy"), ms("memset"), ms("idle"), ms("lead_in"), float64(between)/1e6, ms("tail"), ms("unresolved"))
	sb.WriteString("Each finding names the terms it could move and their delta. Its bound is that delta, capped at the finding's own time: " +
		"an upper bound on how much of the gap the finding could explain, not an estimate of what fixing it saves. " +
		"Findings that share a term share its bound, so bounds do not add. " +
		"The findings are read from the whole target capture; the terms cover the compare window only.\n\n")
}

// gapLine is the body line stating an item's term and bound.
func gapLine(g *GapBound) string {
	bound := "none of the gap"
	if g.BoundNanos > 0 {
		bound = fmt.Sprintf("at most %.2f ms of the gap", float64(g.BoundNanos)/1e6)
	}
	return fmt.Sprintf("explains %s (term %s, Δ %+.2f ms)", bound, g.Term, float64(g.TermDeltaNanos)/1e6)
}
