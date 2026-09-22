package cudalineinfo

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// A Side is one build in a comparison: its attribution, the label that names
// it in a report, and, when the build was also run, what a profile of it
// measured. The profile is optional, and a comparison of two builds neither
// of which was run is still a comparison of what the compiler did.
type Side struct {
	*Attribution
	Label   string
	Profile *Profile
}

// A SiteDiff is the change in one site's instructions between two
// attributions of the same unit: the counts on each side and the opcodes that
// appear only on one, as a multiset difference. Register allocation and
// scheduling change operands freely, so the opcode is the unit of comparison.
//
// BaseValue and TargetValue are what each side's profile measured at the
// site, and are zero on a side that has none. Instructions removed are not
// work removed: a line can lose instructions and gain time, and only the two
// measurements say which happened.
type SiteDiff struct {
	Site         Site
	Base, Target int
	BaseValue    int64
	TargetValue  int64
	Removed      []string // opcodes in base and not in target, with multiplicity
	Added        []string // opcodes in target and not in base, with multiplicity
}

// Diff compares two sides site by site, in file and line order. A site
// present on one side only has a zero count on the other.
func Diff(base, target Side) []SiteDiff {
	seen := map[Site]bool{}
	var sites []Site
	for _, a := range []*Attribution{base.Attribution, target.Attribution} {
		for _, s := range a.Sites() {
			if !seen[s] {
				seen[s] = true
				sites = append(sites, s)
			}
		}
	}
	for _, side := range []Side{base, target} {
		if side.Profile == nil {
			continue
		}
		for s := range side.Profile.Sites {
			if !seen[s] {
				seen[s] = true
				sites = append(sites, s)
			}
		}
	}
	sortSites(sites)
	var diffs []SiteDiff
	for _, s := range sites {
		b, t := opcodes(base.SASS[s]), opcodes(target.SASS[s])
		diffs = append(diffs, SiteDiff{
			Site:        s,
			Base:        len(base.SASS[s]),
			Target:      len(target.SASS[s]),
			BaseValue:   base.value(s),
			TargetValue: target.value(s),
			Removed:     subtract(b, t),
			Added:       subtract(t, b),
		})
	}
	return diffs
}

// value is what the side's profile measured at a site, or zero when the side
// carries no profile.
func (s Side) value(site Site) int64 {
	if s.Profile == nil {
		return 0
	}
	return s.Profile.Sites[site]
}

// measure names what the two sides measured, or is empty when neither was
// run. Two profiles of different measures do not compare, so the second is
// reported only when it measures what the first does.
func measure(base, target Side) string {
	for _, s := range []Side{base, target} {
		if s.Profile != nil {
			return s.Profile.Measure
		}
	}
	return ""
}

// WriteDiff writes the per-site comparison of two sides, then the change in
// the remainder. A side that carries a profile adds a column of what it
// measured, so a line's change in instructions can be read next to the change
// in what the device did with them.
func WriteDiff(w io.Writer, base, target Side) error {
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%s: %s (%d of %d) vs %s (%d of %d)\n", base.Unit.Name,
		base.Label, base.Attributed, base.Instructions,
		target.Label, target.Attributed, target.Instructions)
	m := measure(base, target)
	tw := tabwriter.NewWriter(bw, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "  SITE\t%s\t%s", strings.ToUpper(base.Label), strings.ToUpper(target.Label))
	if m != "" {
		fmt.Fprintf(tw, "\t%s\t%s", strings.ToUpper(base.Label+" "+m), strings.ToUpper(target.Label+" "+m))
	}
	fmt.Fprint(tw, "\tREMOVED\tADDED\n")
	diffs := Diff(base, target)
	sites := make([]Site, len(diffs))
	for i, d := range diffs {
		sites[i] = d.Site
	}
	names := SiteNames(sites)
	for _, d := range diffs {
		fmt.Fprintf(tw, "  %s\t%d\t%d", names[d.Site], d.Base, d.Target)
		if m != "" {
			fmt.Fprintf(tw, "\t%s\t%s", count(base, d.BaseValue), count(target, d.TargetValue))
		}
		fmt.Fprintf(tw, "\t%s\t%s\n", summarize(d.Removed), summarize(d.Added))
	}
	tw.Flush()
	fmt.Fprintf(bw, "  remainder: %s: %s\n", base.Label, base.Remainder)
	fmt.Fprintf(bw, "  remainder: %s: %s\n", target.Label, target.Remainder)
	return bw.Flush()
}

// count renders one side's measurement. A side that was not run measured
// nothing, which is not the same as having measured zero, so it prints as
// unmeasured rather than as a number.
func count(s Side, n int64) string {
	if s.Profile == nil {
		return "-"
	}
	return strconv.FormatInt(n, 10)
}

func opcodes(insts []cudasass.Instruction) map[string]int {
	m := map[string]int{}
	for _, in := range insts {
		m[cudasass.BaseOpcode(in.Opcode)]++
	}
	return m
}

// subtract returns the opcodes of a not matched in b, with multiplicity,
// sorted.
func subtract(a, b map[string]int) []string {
	var out []string
	for op, n := range a {
		for i := b[op]; i < n; i++ {
			out = append(out, op)
		}
	}
	sort.Strings(out)
	return out
}

// summarize renders an opcode list as "FFMA×3 MUFU.EX2", keeping the table
// readable when a site loses twenty instructions.
func summarize(ops []string) string {
	if len(ops) == 0 {
		return "-"
	}
	var parts []string
	for i := 0; i < len(ops); {
		j := i
		for j < len(ops) && ops[j] == ops[i] {
			j++
		}
		if j-i > 1 {
			parts = append(parts, fmt.Sprintf("%s×%d", ops[i], j-i))
		} else {
			parts = append(parts, ops[i])
		}
		i = j
	}
	return strings.Join(parts, " ")
}
