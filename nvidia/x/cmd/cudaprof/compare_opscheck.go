package main

import (
	"fmt"
	"io"
	"slices"
	"strings"
	"text/tabwriter"
)

// OpsCheck reports how the lines of an ops file matched the kernels of both
// captures.
type OpsCheck struct {
	Rules []OpsRuleCheck `json:"rules"`
	Terms []OpsTermCheck `json:"combine_terms,omitempty"`
}

// OpsRuleCheck is one rule line. A line wins the kernels it is the first to
// match; the kernels it matches that an earlier line won are shadowed, and
// ShadowedBy lists the lines that won them. Kernels are distinct match keys.
type OpsRuleCheck struct {
	Line                  int    `json:"line"`
	Pattern               string `json:"pattern"`
	Op                    string `json:"op"`
	BaseKernels           int    `json:"base_kernels"`
	TargetKernels         int    `json:"target_kernels"`
	BaseLaunches          int    `json:"base_launches"`
	TargetLaunches        int    `json:"target_launches"`
	BaseShadowedKernels   int    `json:"base_shadowed_kernels"`
	TargetShadowedKernels int    `json:"target_shadowed_kernels"`
	ShadowedBy            []int  `json:"shadowed_by,omitempty"`
}

// OpsTermCheck is one term of a combine line: the launches it took.
type OpsTermCheck struct {
	Line           int    `json:"line"`
	Combine        string `json:"combine"`
	Term           string `json:"term"`
	BaseLaunches   int    `json:"base_launches"`
	TargetLaunches int    `json:"target_launches"`
}

// checkOps matches every rule against every kernel of both sides.
func checkOps(rules opRules, combines []opCombine, b, t *CompareSide) *OpsCheck {
	c := &OpsCheck{Rules: make([]OpsRuleCheck, len(rules))}
	shadowedBy := make([]map[int]bool, len(rules))
	for i, r := range rules {
		c.Rules[i] = OpsRuleCheck{Line: r.Line, Pattern: r.Pattern, Op: r.Op}
		shadowedBy[i] = make(map[int]bool)
	}
	for _, side := range []*CompareSide{b, t} {
		for _, keys := range side.kernels {
			for key, k := range keys {
				if key == "" {
					key = "(unnamed)"
				}
				winner := -1
				for i, r := range rules {
					if !r.re.MatchString(key) {
						continue
					}
					rc := &c.Rules[i]
					if winner < 0 {
						winner = i
						if side == b {
							rc.BaseKernels++
							rc.BaseLaunches += k.launches
						} else {
							rc.TargetKernels++
							rc.TargetLaunches += k.launches
						}
						continue
					}
					if side == b {
						rc.BaseShadowedKernels++
					} else {
						rc.TargetShadowedKernels++
					}
					shadowedBy[i][rules[winner].Line] = true
				}
			}
		}
	}
	for i := range c.Rules {
		for line := range shadowedBy[i] {
			c.Rules[i].ShadowedBy = append(c.Rules[i].ShadowedBy, line)
		}
		slices.Sort(c.Rules[i].ShadowedBy)
	}
	for i, cb := range combines {
		for j, term := range cb.Terms {
			c.Terms = append(c.Terms, OpsTermCheck{
				Line:           cb.Line,
				Combine:        cb.Name,
				Term:           term.String(),
				BaseLaunches:   b.combines[i].termHits[j],
				TargetLaunches: t.combines[i].termHits[j],
			})
		}
	}
	return c
}

// err reports the rules that won no kernel in either capture and the
// combine terms that took none.
func (c *OpsCheck) err() error {
	if c == nil {
		return nil
	}
	var msgs []string
	var lines []string
	for _, r := range c.Rules {
		if r.BaseKernels == 0 && r.TargetKernels == 0 {
			lines = append(lines, fmt.Sprint(r.Line))
		}
	}
	if lines != nil {
		msgs = append(msgs, "lines "+strings.Join(lines, ", ")+" won no kernel in either capture")
	}
	lines = nil
	for _, t := range c.Terms {
		if t.BaseLaunches == 0 && t.TargetLaunches == 0 {
			lines = append(lines, fmt.Sprintf("%d (%s)", t.Line, t.Term))
		}
	}
	if lines != nil {
		msgs = append(msgs, "combine terms on lines "+strings.Join(lines, ", ")+" took no kernel in either capture")
	}
	if msgs == nil {
		return nil
	}
	return fmt.Errorf("%s", strings.Join(msgs, "; "))
}

func (c *Comparison) writeOpsCheck(w io.Writer) error {
	if c.OpsCheck == nil {
		return nil
	}
	fmt.Fprintln(w, "\nops check: kernels (distinct match keys) and launches each line won by matching first, and the kernels")
	fmt.Fprintln(w, "it matched that an earlier line won. NOTHING: the line won no kernel in either capture.")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "kernels\t\tlaunches\t\tshadowed\t\t\t")
	fmt.Fprintln(tw, "base\ttarget\tbase\ttarget\tbase\ttarget\tline\t  op pattern [shadowed by lines]")
	for _, r := range c.OpsCheck.Rules {
		by := make([]string, len(r.ShadowedBy))
		for i, l := range r.ShadowedBy {
			by[i] = fmt.Sprint(l)
		}
		mark := ""
		if len(by) > 0 {
			mark = " [" + strings.Join(by, ",") + "]"
		}
		if r.BaseKernels == 0 && r.TargetKernels == 0 {
			mark += " NOTHING"
		}
		fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t  %s %s%s\n",
			r.BaseKernels, r.TargetKernels, r.BaseLaunches, r.TargetLaunches,
			r.BaseShadowedKernels, r.TargetShadowedKernels, r.Line, r.Op, r.Pattern, mark)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	if len(c.OpsCheck.Terms) == 0 {
		return nil
	}
	fmt.Fprintln(w, "\ncombine terms: launches each took. NOTHING: none in either capture.")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "base\ttarget\tline\t  combine: term")
	for _, t := range c.OpsCheck.Terms {
		mark := ""
		if t.BaseLaunches == 0 && t.TargetLaunches == 0 {
			mark = " NOTHING"
		}
		fmt.Fprintf(tw, "%d\t%d\t%d\t  %s: %s%s\n", t.BaseLaunches, t.TargetLaunches, t.Line, t.Combine, t.Term, mark)
	}
	return tw.Flush()
}
