package main

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Groups -by charges kernels to, besides a range's own name.
const (
	groupForward   = "forward"
	groupBackward  = "backward"
	groupOptimizer = "optimizer"
	groupOther     = "(other range)"      // inside ranges no phase rule names
	groupNoRange   = "(no range)"         // launched with no range open
	groupNoLaunch  = "(no launch record)" // no host API record to place it
)

// phaseOrder is the order the phase table lists its rows in.
var phaseOrder = []string{groupForward, groupBackward, groupOptimizer, groupOther, groupNoRange, groupNoLaunch}

// phaseOf names the phase of a stack of open NVTX ranges, outermost first:
// the first range whose name says forward (":fwd", "forward"), backward
// (":bwd", "backward", as in PyTorch's autograd "MulBackward0") or
// optimizer ("optim") decides it.
func phaseOf(stack []timeline.Range) string {
	for _, r := range stack {
		n := strings.ToLower(r.Name)
		switch {
		case strings.Contains(n, ":fwd") || strings.Contains(n, "forward"):
			return groupForward
		case strings.Contains(n, ":bwd") || strings.Contains(n, "backward"):
			return groupBackward
		case strings.Contains(n, "optim"):
			return groupOptimizer
		}
	}
	if len(stack) == 0 {
		return groupNoRange
	}
	return groupOther
}

// rangeLabeler names the group of each instant of one host thread.
type rangeLabeler struct {
	bounds []int64  // sorted, distinct
	labels []string // labels[i] holds over [bounds[i], bounds[i+1])
}

func newRangeLabeler(ranges []timeline.Range, by string) *rangeLabeler {
	var bounds []int64
	for _, r := range ranges {
		bounds = append(bounds, r.Start, r.End)
	}
	sort.Slice(bounds, func(i, j int) bool { return bounds[i] < bounds[j] })
	l := &rangeLabeler{}
	for i, b := range bounds {
		if i > 0 && b == bounds[i-1] {
			continue
		}
		var stack []timeline.Range
		for _, r := range ranges {
			if r.Start <= b && b < r.End {
				stack = append(stack, r)
			}
		}
		// Outermost first: earliest start, then longest.
		sort.Slice(stack, func(i, j int) bool {
			if stack[i].Start != stack[j].Start {
				return stack[i].Start < stack[j].Start
			}
			return stack[i].End > stack[j].End
		})
		label := groupNoRange
		switch {
		case by == "phase":
			label = phaseOf(stack)
		case len(stack) > 0:
			label = stack[len(stack)-1].Name
		}
		l.bounds = append(l.bounds, b)
		l.labels = append(l.labels, label)
	}
	return l
}

// at returns the label of instant t.
func (l *rangeLabeler) at(t int64) string {
	i := sort.Search(len(l.bounds), func(i int) bool { return l.bounds[i] > t }) - 1
	if i < 0 {
		return groupNoRange
	}
	return l.labels[i]
}

// GroupAttribution says how a side's kernels were placed in -by groups.
type GroupAttribution struct {
	Ranges          int `json:"ranges"`
	UnpairedMarkers int `json:"unpaired_markers,omitempty"`
	// OwnThread launches were placed by the ranges of the thread that
	// launched them; OtherThread launches came from a thread with no ranges
	// and were placed by the ranges of every thread; NoLaunchRecord
	// launches have no host API record.
	OwnThread      int `json:"own_thread"`
	OtherThread    int `json:"other_thread"`
	NoLaunchRecord int `json:"no_launch_record"`
}

// assignGroups charges each kernel of recs to the group its launch fell in
// and returns the groups by name. owned is the exclusive time of each record.
func (s *CompareSide) assignGroups(recs []compareRecord, owned []int64, records timeline.Records, by string) map[string]*stageTotals {
	a := &GroupAttribution{Ranges: len(records.Ranges), UnpairedMarkers: records.UnpairedMarkers}
	s.Groups = a
	byThread := make(map[uint32][]timeline.Range)
	for _, r := range records.Ranges {
		byThread[r.Thread] = append(byThread[r.Thread], r)
	}
	labelers := make(map[uint32]*rangeLabeler)
	for tid, rs := range byThread {
		labelers[tid] = newRangeLabeler(rs, by)
	}
	all := newRangeLabeler(records.Ranges, by)
	out := make(map[string]*stageTotals)
	for i, r := range recs {
		if r.class != classKernel {
			continue
		}
		label := groupNoLaunch
		if api, ok := records.APIs[r.corr]; ok && r.corr != 0 {
			if l := labelers[api.Thread]; l != nil {
				label = l.at(api.Start)
				a.OwnThread++
			} else {
				label = all.at(api.Start)
				a.OtherThread++
			}
		} else {
			a.NoLaunchRecord++
		}
		t := out[label]
		if t == nil {
			t = &stageTotals{stage: label}
			out[label] = t
		}
		t.add(r.op, owned[i], r.end-r.start)
	}
	return out
}

// CompareGroup is one -by group in both captures.
type CompareGroup struct {
	Group                string `json:"group"`
	BaseLaunches         int    `json:"base_launches"`
	TargetLaunches       int    `json:"target_launches"`
	BaseExclusiveNanos   int64  `json:"base_exclusive_ns"`
	TargetExclusiveNanos int64  `json:"target_exclusive_ns"`
	DeltaExclusiveNanos  int64  `json:"delta_exclusive_ns"`
	BaseSummedNanos      int64  `json:"base_summed_ns"`
	TargetSummedNanos    int64  `json:"target_summed_ns"`

	// Ops is the group's time by op; the cells of an op over all groups sum
	// to its row in Comparison.Ops.
	Ops []CompareOp `json:"ops"`
}

// joinGroups lists every group either side has: phases in phaseOrder,
// ranges largest |Δ| first.
func joinGroups(by string, base, target map[string]*stageTotals) []CompareGroup {
	names := make(map[string]bool)
	for n := range base {
		names[n] = true
	}
	for n := range target {
		names[n] = true
	}
	var rows []CompareGroup
	for n := range names {
		b, t := base[n], target[n]
		if b == nil {
			b = &stageTotals{}
		}
		if t == nil {
			t = &stageTotals{}
		}
		rows = append(rows, CompareGroup{
			Group:                n,
			BaseLaunches:         b.launches,
			TargetLaunches:       t.launches,
			BaseExclusiveNanos:   b.exclusive,
			TargetExclusiveNanos: t.exclusive,
			DeltaExclusiveNanos:  t.exclusive - b.exclusive,
			BaseSummedNanos:      b.summed,
			TargetSummedNanos:    t.summed,
			Ops:                  joinOps(b.ops, t.ops, nil, nil),
		})
	}
	rank := func(g string) int {
		for i, p := range phaseOrder {
			if p == g {
				return i
			}
		}
		return len(phaseOrder)
	}
	sort.Slice(rows, func(i, j int) bool {
		if by == "phase" {
			return rank(rows[i].Group) < rank(rows[j].Group)
		}
		di, dj := abs64(rows[i].DeltaExclusiveNanos), abs64(rows[j].DeltaExclusiveNanos)
		if di != dj {
			return di > dj
		}
		return rows[i].Group < rows[j].Group
	})
	return rows
}

// checkGroups checks that the groups partition kernel exclusive time, and
// that each op's cells sum to its row.
func checkGroups(c *Comparison) error {
	if c.Groups == nil {
		return nil
	}
	var base, target int64
	var rows []cellRow
	for _, g := range c.Groups {
		base += g.BaseExclusiveNanos
		target += g.TargetExclusiveNanos
		rows = append(rows, cellRow{g.Group, g.BaseExclusiveNanos, g.TargetExclusiveNanos, g.Ops})
	}
	if base != c.Base.KernelNanos || target != c.Target.KernelNanos {
		return fmt.Errorf("-by %s: groups exclusive %d, %d ns != kernel exclusive %d, %d ns", c.By, base, target, c.Base.KernelNanos, c.Target.KernelNanos)
	}
	return checkCellRows(rows, c.Ops)
}

// compareGroupCells is how many (group, op) cells, largest |Δ| first, the
// text lists. JSON gives every cell.
const compareGroupCells = 20

func (c *Comparison) writeGroups(w io.Writer) error {
	if c.Groups == nil {
		return nil
	}
	fmt.Fprintf(w, "\nper %s, ns. exclusive as above; it sums to kernel exclusive. a launch is placed by the NVTX ranges\n", c.By)
	fmt.Fprintln(w, "open on its host thread when the launch call began.")
	for _, s := range []struct {
		name string
		side *CompareSide
	}{{"base", c.Base}, {"target", c.Target}} {
		a := s.side.Groups
		fmt.Fprintf(w, "%s: %d ranges, %d unpaired markers; launches placed by own thread %d, by every thread %d, unplaced (no launch record) %d\n",
			s.name, a.Ranges, a.UnpairedMarkers, a.OwnThread, a.OtherThread, a.NoLaunchRecord)
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "launches\t\texclusive\t\t\tsummed\t\t")
	fmt.Fprintf(tw, "base\ttarget\tbase\ttarget\tΔ\tbase\ttarget\t  %s\n", c.By)
	for _, g := range c.Groups {
		fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%+d\t%d\t%d\t  %s\n",
			g.BaseLaunches, g.TargetLaunches,
			g.BaseExclusiveNanos, g.TargetExclusiveNanos, g.DeltaExclusiveNanos,
			g.BaseSummedNanos, g.TargetSummedNanos, g.Group)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	type cell struct {
		group string
		op    CompareOp
	}
	var cells []cell
	for _, g := range c.Groups {
		for _, o := range g.Ops {
			cells = append(cells, cell{g.Group, o})
		}
	}
	sort.SliceStable(cells, func(i, j int) bool {
		return abs64(cells[i].op.DeltaExclusiveNanos) > abs64(cells[j].op.DeltaExclusiveNanos)
	})
	n := min(len(cells), compareGroupCells)
	fmt.Fprintf(w, "\nper %s and op, the %d cells of largest |Δ exclusive|:\n", c.By, n)
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "launches\t\texclusive\t\t\tΔ split\t\t\t")
	fmt.Fprintf(tw, "base\ttarget\tbase\ttarget\tΔ\tcount\tper-launch\t  %s\t  op\n", c.By)
	for _, x := range cells[:n] {
		o := x.op
		fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%+d\t%+d\t%+d\t  %s\t  %s\n",
			o.BaseLaunches, o.TargetLaunches,
			o.BaseExclusiveNanos, o.TargetExclusiveNanos, o.DeltaExclusiveNanos,
			o.DeltaFromCountNanos, o.DeltaFromPerLaunchNanos, x.group, o.Op)
	}
	return tw.Flush()
}
