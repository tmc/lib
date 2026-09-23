package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// Advise with -base ranks the target's findings by how much of the gap to a
// base capture each could explain. The gap is compare's: the same windows,
// the same decomposition and the same refusals. Each finding moves one or
// more terms of compare's identity:
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

// adviseAgainstBase compares the target capture at path against opts.base and
// ranks report by the comparison.
func adviseAgainstBase(report *AdviceReport, target *cudacapture.Capture, path string, opts adviseOptions, limits cudacapture.Limits) error {
	var rules opRules
	if opts.ops != "" {
		f, err := os.Open(opts.ops)
		if err != nil {
			return err
		}
		rules, err = parseOpRules(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", opts.ops, err)
		}
	}
	basePath, err := filepath.Abs(opts.base)
	if err != nil {
		return err
	}
	base, err := readCapturePathLimits(basePath, limits)
	if err != nil {
		return fmt.Errorf("read base capture %s: %w", opts.base, err)
	}
	c, err := compareCaptures(base, target, rules, nil, compareOptions{phase: opts.phase, opsPath: opts.ops})
	if err != nil {
		return fmt.Errorf("compare: %w", err)
	}
	c.Base.Capture, c.Target.Capture = basePath, path
	rankAgainstBase(report, c, rules)
	return nil
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

func gapSection(item AdviceItem) int {
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

// rankAgainstBase sets each item's Gap from c and orders the items by
// section, and within a section by bound. Read first stays first: it says how
// to read the rest, whatever its bound.
func rankAgainstBase(report *AdviceReport, c *Comparison, rules opRules) {
	report.Base = c
	ops := make(map[string]int64)
	for _, o := range c.Ops {
		ops[o.Op] = o.DeltaExclusiveNanos
	}
	for i := range report.Items {
		report.Items[i].Gap = gapBound(report.Items[i], c, ops, rules)
	}
	sort.SliceStable(report.Items, func(i, j int) bool {
		a, b := report.Items[i], report.Items[j]
		if sa, sb := gapSection(a), gapSection(b); sa != sb {
			return sa < sb
		}
		return a.Gap != nil && b.Gap != nil && a.Gap.BoundNanos > b.Gap.BoundNanos
	})
}

// gapBound returns what item could explain of c's gap, or nil when it moves
// no term. ops holds each op's exclusive delta.
func gapBound(item AdviceItem, c *Comparison, ops map[string]int64, rules opRules) *GapBound {
	var names []string
	var delta int64
	switch {
	case item.terms != nil:
		for _, t := range item.terms {
			names = append(names, t)
			delta += c.term(t).Delta
		}
	case isDispatch(item.Category):
		names = []string{"idle"}
		delta = c.term("idle").Delta
	case item.Location.Kernel != "" && c.OpStatus == "MEASURED":
		// A fusion finding names a pair; it moves each distinct op once. An
		// op absent from both windows has a delta of zero.
		seen := make(map[string]bool)
		for _, k := range strings.Split(item.Location.Kernel, " -> ") {
			op, _ := rules.kernelOp(k)
			if seen[op] {
				continue
			}
			seen[op] = true
			names = append(names, "op "+op)
			delta += ops[op]
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
func (r *AdviceReport) writeBase(sb *strings.Builder) {
	c := r.Base
	sb.WriteString("## Against base\n\n```\n")
	fmt.Fprintf(sb, "base    %s\n        %s\n", c.Base.Capture, c.Base.windowLine())
	fmt.Fprintf(sb, "target  %s\n        %s\n", c.Target.Capture, c.Target.windowLine())
	for _, n := range c.Notes {
		fmt.Fprintf(sb, "note: %s\n", n)
	}
	sb.WriteString(c.identity())
	sb.WriteString("```\n\n")
	ms := func(name string) string { return fmt.Sprintf("%+.2f", float64(c.term(name).Delta)/1e6) }
	between := c.term("idle").Delta - c.term("lead_in").Delta - c.term("tail").Delta
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
