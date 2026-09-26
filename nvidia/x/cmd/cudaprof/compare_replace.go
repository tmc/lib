package main

import (
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"
	"text/tabwriter"
)

// ClockRatio is the SM clock both sides read at one window edge.
type ClockRatio struct {
	Edge             string  `json:"edge"` // "start" or "stop"
	BaseSMClockMHz   uint32  `json:"base_sm_clock_mhz"`
	TargetSMClockMHz uint32  `json:"target_sm_clock_mhz"`
	Ratio            float64 `json:"ratio"` // target / base
}

// KernelReplacement is a set of kernels that ran in the base only, paired
// with a set that ran in the target only: a rename, or a fusion or split.
//
// When Launches is nonzero, every kernel on both sides ran that many times,
// so one launch of each kernel of a side is one step of the same work, and
// the PerLaunch fields are a side's summed time over Launches. Otherwise
// the row holds the kernels of an op that paired with nothing of equal
// count, and there is no per-launch time.
type KernelReplacement struct {
	Op                   string   `json:"op"`
	Base                 []string `json:"base"`
	Target               []string `json:"target"`
	Launches             int      `json:"launches_per_kernel,omitempty"`
	BaseLaunches         int      `json:"base_launches"`
	TargetLaunches       int      `json:"target_launches"`
	BaseExclusiveNanos   int64    `json:"base_exclusive_ns"`
	TargetExclusiveNanos int64    `json:"target_exclusive_ns"`
	DeltaExclusiveNanos  int64    `json:"delta_exclusive_ns"`
	BaseSummedNanos      int64    `json:"base_summed_ns"`
	TargetSummedNanos    int64    `json:"target_summed_ns"`
	BasePerLaunchNanos   int64    `json:"base_per_launch_ns,omitempty"`
	TargetPerLaunchNanos int64    `json:"target_per_launch_ns,omitempty"`
	DeltaPerLaunchNanos  int64    `json:"delta_per_launch_ns,omitempty"`
}

// replacementPrefix is how many leading bytes, ignoring case, two kernel
// names must share to pair when no -ops file groups them.
const replacementPrefix = 5

// pairReplacements pairs the kernels that ran on one side only. Two such
// kernels, one per side, pair when they ran equally often and, with an ops
// file, belong to the same op, or, without one, share a name prefix of
// replacementPrefix bytes. Kernels connected by pairs form one row, so two
// kernels fused into one make a 2:1 row. With an ops file, an op's
// one-sided kernels that paired with nothing form one more row, when both
// sides have some.
func pairReplacements(base, target map[string]map[string]*opTotals, grouped bool) []KernelReplacement {
	type kernel struct {
		op, key string
		target  bool
		t       *opTotals
	}
	var gone, added []kernel
	for op, keys := range base {
		for key, t := range keys {
			if target[op][key] == nil {
				gone = append(gone, kernel{op, key, false, t})
			}
		}
	}
	for op, keys := range target {
		for key, t := range keys {
			if base[op][key] == nil {
				added = append(added, kernel{op, key, true, t})
			}
		}
	}
	all := append(gone, added...)
	sort.Slice(all, func(i, j int) bool {
		if all[i].target != all[j].target {
			return !all[i].target
		}
		if all[i].op != all[j].op {
			return all[i].op < all[j].op
		}
		return all[i].key < all[j].key
	})

	parent := make([]int, len(all))
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(i int) int {
		if parent[i] != i {
			parent[i] = find(parent[i])
		}
		return parent[i]
	}
	paired := make([]bool, len(all))
	nGone := len(gone)
	for i := range nGone {
		for j := nGone; j < len(all); j++ {
			g, a := all[i], all[j]
			if g.t.launches != a.t.launches {
				continue
			}
			if grouped && g.op != a.op || !grouped && !sharePrefix(g.key, a.key, replacementPrefix) {
				continue
			}
			parent[find(j)] = find(i)
			paired[i], paired[j] = true, true
		}
	}

	var rows []KernelReplacement
	add := func(members []kernel, launches int) {
		r := KernelReplacement{Op: members[0].op, Launches: launches}
		for _, k := range members {
			if k.target {
				r.Target = append(r.Target, k.key)
				r.TargetLaunches += k.t.launches
				r.TargetExclusiveNanos += k.t.exclusive
				r.TargetSummedNanos += k.t.summed
			} else {
				r.Base = append(r.Base, k.key)
				r.BaseLaunches += k.t.launches
				r.BaseExclusiveNanos += k.t.exclusive
				r.BaseSummedNanos += k.t.summed
			}
		}
		r.DeltaExclusiveNanos = r.TargetExclusiveNanos - r.BaseExclusiveNanos
		if launches > 0 {
			r.BasePerLaunchNanos = r.BaseSummedNanos / int64(launches)
			r.TargetPerLaunchNanos = r.TargetSummedNanos / int64(launches)
			r.DeltaPerLaunchNanos = r.TargetPerLaunchNanos - r.BasePerLaunchNanos
		}
		rows = append(rows, r)
	}
	components := make(map[int][]kernel)
	var roots []int
	for i, k := range all {
		if !paired[i] {
			continue
		}
		root := find(i)
		if components[root] == nil {
			roots = append(roots, root)
		}
		components[root] = append(components[root], k)
	}
	for _, root := range roots {
		members := components[root]
		if !grouped {
			// Without an ops file each kernel is its own op.
			for i := range members {
				members[i].op = ""
			}
		}
		add(members, members[0].t.launches)
	}
	if grouped {
		left := make(map[string][]kernel)
		var ops []string
		for i, k := range all {
			if paired[i] {
				continue
			}
			if left[k.op] == nil {
				ops = append(ops, k.op)
			}
			left[k.op] = append(left[k.op], k)
		}
		slices.Sort(ops)
		for _, op := range ops {
			ks := left[op]
			if !ks[0].target && ks[len(ks)-1].target {
				add(ks, 0)
			}
		}
	}
	sort.SliceStable(rows, func(i, j int) bool {
		return abs64(rows[i].DeltaExclusiveNanos) > abs64(rows[j].DeltaExclusiveNanos)
	})
	return rows
}

// sharePrefix reports whether a and b begin with the same n bytes,
// ignoring case.
func sharePrefix(a, b string, n int) bool {
	return len(a) >= n && len(b) >= n && strings.EqualFold(a[:n], b[:n])
}

// writeReplacements writes the replacement rows, if any.
func (c *Comparison) writeReplacements(w io.Writer) error {
	if len(c.Replacements) == 0 {
		return nil
	}
	fmt.Fprintln(w, "\nkernels replaced: base-only kernels paired with target-only kernels of equal launch count")
	fmt.Fprintln(w, "(same op with -ops, else a shared name prefix). per launch: a side's summed ns over the launches per kernel;")
	fmt.Fprintln(w, "rows with launches - are an op's leftover one-sided kernels of unequal counts.")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "launches\tper launch\t\t\texclusive\t\t\t")
	fmt.Fprintln(tw, "each\tbase\ttarget\tΔ\tbase\ttarget\tΔ\t  kernels")
	for _, r := range c.Replacements {
		each, bp, tp, dp := "-", "-", "-", "-"
		if r.Launches > 0 {
			each = fmt.Sprint(r.Launches)
			bp, tp, dp = fmt.Sprint(r.BasePerLaunchNanos), fmt.Sprint(r.TargetPerLaunchNanos), fmt.Sprintf("%+d", r.DeltaPerLaunchNanos)
		}
		names := strings.Join(r.Base, " + ") + " -> " + strings.Join(r.Target, " + ")
		if r.Op != "" {
			names = r.Op + ": " + names
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%d\t%+d\t  %s\n", each, bp, tp, dp,
			r.BaseExclusiveNanos, r.TargetExclusiveNanos, r.DeltaExclusiveNanos, names)
	}
	return tw.Flush()
}
