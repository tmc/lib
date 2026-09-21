package cudalineinfo

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// A SiteDiff is the change in one site's instructions between two
// attributions of the same unit: the counts on each side and the opcodes that
// appear only on one, as a multiset difference. Register allocation and
// scheduling change operands freely, so the opcode is the unit of comparison.
type SiteDiff struct {
	Site         Site
	Base, Target int
	Removed      []string // opcodes in base and not in target, with multiplicity
	Added        []string // opcodes in target and not in base, with multiplicity
}

// Diff compares two attributions site by site, in file and line order. A
// site present on one side only has a zero count on the other.
func Diff(base, target *Attribution) []SiteDiff {
	seen := map[Site]bool{}
	var sites []Site
	for _, a := range []*Attribution{base, target} {
		for _, s := range a.Sites() {
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
			Site:    s,
			Base:    len(base.SASS[s]),
			Target:  len(target.SASS[s]),
			Removed: subtract(b, t),
			Added:   subtract(t, b),
		})
	}
	return diffs
}

// WriteDiff writes the per-site comparison of two attributions, then the
// change in the remainder. The labels name the two sides.
func WriteDiff(w io.Writer, base, target *Attribution, baseLabel, targetLabel string) error {
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%s: %s (%d of %d) vs %s (%d of %d)\n", base.Unit.Name,
		baseLabel, base.Attributed, base.Instructions,
		targetLabel, target.Attributed, target.Instructions)
	tw := tabwriter.NewWriter(bw, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "  SITE\t%s\t%s\tREMOVED\tADDED\n", strings.ToUpper(baseLabel), strings.ToUpper(targetLabel))
	for _, d := range Diff(base, target) {
		fmt.Fprintf(tw, "  %s\t%d\t%d\t%s\t%s\n", d.Site, d.Base, d.Target, summarize(d.Removed), summarize(d.Added))
	}
	tw.Flush()
	fmt.Fprintf(bw, "  remainder: %s: %s\n", baseLabel, base.Remainder)
	fmt.Fprintf(bw, "  remainder: %s: %s\n", targetLabel, target.Remainder)
	return bw.Flush()
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
