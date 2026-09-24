package main

import (
	"fmt"
	"io"
	"math"
	"slices"
	"sort"
	"text/tabwriter"
)

// breakdownNote is stated with every breakdown, in JSON and in text.
const breakdownNote = "each column is a median over the pairs taken on its own: a row's delta median is not its G median minus its C median, and the rows' medians do not sum to the kernel total's"

// qualifyBreakdown is a campaign's pairs broken down by op and by stage.
// Times are kernel exclusive milliseconds, as compare charges them.
type qualifyBreakdown struct {
	// Pairs is how many pairs every row is taken over: all the pairs the
	// verdict scores, in block order G1-C1 then G2-C2.
	Pairs  int          `json:"pairs"`
	Note   string       `json:"note"`
	Notes  []string     `json:"notes,omitempty"`
	Kernel qualifyRow   `json:"kernel_exclusive"`
	Ops    []qualifyRow `json:"ops,omitempty"`
	Stages []qualifyRow `json:"stages,omitempty"`
}

// qualifyRow is one op or stage over every pair. A pair in which the row is
// absent counts it as zero launches and zero time.
type qualifyRow struct {
	Name           string  `json:"name"`
	BaseLaunches   [2]int  `json:"c_launches_min_max"`
	TargetLaunches [2]int  `json:"g_launches_min_max"`
	BaseMS         float64 `json:"c_median_ms"`
	TargetMS       float64 `json:"g_median_ms"`
	DeltaMS        float64 `json:"delta_median_ms"`
	DeltaMinMS     float64 `json:"delta_min_ms"`
	DeltaMaxMS     float64 `json:"delta_max_ms"`
}

// pairCell is one row in one pair.
type pairCell struct {
	baseLaunches, targetLaunches int
	base, target                 int64
}

// breakDownPairs joins each pair's partitioned captures, C as base and G as
// target, and summarizes each op and stage over the pairs. entries holds
// whole blocks in run order, each measured with its side.
func breakDownPairs(entries []qualifyEntry, rules opRules, stages stageBoundaries) (*qualifyBreakdown, error) {
	var pairs []*Comparison
	var labels []string
	for i := 0; i+len(qualifyArms) <= len(entries); i += len(qualifyArms) {
		c1, g1, g2, c2 := entries[i], entries[i+1], entries[i+2], entries[i+3]
		for _, p := range [][2]qualifyEntry{{c1, g1}, {c2, g2}} {
			label := fmt.Sprintf("block %s %s-%s", p[0].Block, p[1].Arm, p[0].Arm)
			c, err := compareSides(p[0].side, p[1].side, rules, stages, compareOptions{})
			if err != nil {
				return nil, fmt.Errorf("%s: %w", label, err)
			}
			pairs = append(pairs, c)
			labels = append(labels, label)
		}
	}
	bd := &qualifyBreakdown{Pairs: len(pairs), Note: breakdownNote}
	var kernel []pairCell
	for _, c := range pairs {
		kernel = append(kernel, pairCell{c.Base.KernelCount, c.Target.KernelCount, c.Base.KernelNanos, c.Target.KernelNanos})
	}
	bd.Kernel = summarizeRow("kernel exclusive", kernel)
	if rules != nil {
		cells := map[string][]pairCell{}
		for i, c := range pairs {
			for _, o := range c.Ops {
				if cells[o.Op] == nil {
					cells[o.Op] = make([]pairCell, len(pairs))
				}
				cells[o.Op][i] = pairCell{o.BaseLaunches, o.TargetLaunches, o.BaseExclusiveNanos, o.TargetExclusiveNanos}
			}
		}
		for name, cs := range cells {
			bd.Ops = append(bd.Ops, summarizeRow(name, cs))
		}
		sort.Slice(bd.Ops, func(i, j int) bool {
			a, b := bd.Ops[i], bd.Ops[j]
			if da, db := math.Abs(a.DeltaMS), math.Abs(b.DeltaMS); da != db {
				return da > db
			}
			return a.Name < b.Name
		})
	}
	if stages != nil {
		var order []string
		cells := map[string][]pairCell{}
		for i, c := range pairs {
			for _, st := range c.Stages {
				if cells[st.Stage] == nil {
					cells[st.Stage] = make([]pairCell, len(pairs))
					order = append(order, st.Stage)
				}
				cells[st.Stage][i] = pairCell{st.BaseLaunches, st.TargetLaunches, st.BaseExclusiveNanos, st.TargetExclusiveNanos}
				if st.BaseEntries != st.TargetEntries {
					bd.Notes = append(bd.Notes, fmt.Sprintf("%s: stage %s began %d times in C and %d in G: the boundaries did not align", labels[i], st.Stage, st.BaseEntries, st.TargetEntries))
				}
			}
		}
		for _, name := range order {
			bd.Stages = append(bd.Stages, summarizeRow(name, cells[name]))
		}
	}
	return bd, nil
}

// summarizeRow takes the medians and ranges of one row over its pairs, of
// which there is at least one.
func summarizeRow(name string, cells []pairCell) qualifyRow {
	r := qualifyRow{
		Name:           name,
		BaseLaunches:   [2]int{cells[0].baseLaunches, cells[0].baseLaunches},
		TargetLaunches: [2]int{cells[0].targetLaunches, cells[0].targetLaunches},
	}
	var base, target, delta []float64
	for _, c := range cells {
		r.BaseLaunches = [2]int{min(r.BaseLaunches[0], c.baseLaunches), max(r.BaseLaunches[1], c.baseLaunches)}
		r.TargetLaunches = [2]int{min(r.TargetLaunches[0], c.targetLaunches), max(r.TargetLaunches[1], c.targetLaunches)}
		base = append(base, float64(c.base)/1e6)
		target = append(target, float64(c.target)/1e6)
		delta = append(delta, float64(c.target-c.base)/1e6)
	}
	r.BaseMS, r.TargetMS, r.DeltaMS = median(base), median(target), median(delta)
	r.DeltaMinMS, r.DeltaMaxMS = slices.Min(delta), slices.Max(delta)
	return r
}

func (bd *qualifyBreakdown) write(w io.Writer) {
	fmt.Fprintf(w, "\nbreakdown over %d pairs, kernel exclusive ms (delta is G - C; not gated)\n", bd.Pairs)
	fmt.Fprintf(w, "note: %s\n", bd.Note)
	for _, n := range bd.Notes {
		fmt.Fprintf(w, "note: %s\n", n)
	}
	table := func(title string, rows []qualifyRow) {
		fmt.Fprintln(w)
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
		fmt.Fprintln(tw, "launches\t\tmedian\t\t\trange\t\t")
		fmt.Fprintln(tw, "C\tG\tC\tG\tdelta\tmin\tmax\t  "+title)
		for _, r := range append([]qualifyRow{bd.Kernel}, rows...) {
			fmt.Fprintf(tw, "%s\t%s\t%.3f\t%.3f\t%+.3f\t%+.3f\t%+.3f\t  %s\n",
				launchRange(r.BaseLaunches), launchRange(r.TargetLaunches),
				r.BaseMS, r.TargetMS, r.DeltaMS, r.DeltaMinMS, r.DeltaMaxMS, r.Name)
		}
		tw.Flush()
	}
	if bd.Ops != nil {
		table("op", bd.Ops)
	}
	if bd.Stages != nil {
		table("stage", bd.Stages)
	}
}

func launchRange(r [2]int) string {
	if r[0] == r[1] {
		return fmt.Sprint(r[0])
	}
	return fmt.Sprintf("%d..%d", r[0], r[1])
}
