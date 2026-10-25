package okcprofile

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Elasticity is one kernel's share of the log score.
type Elasticity struct {
	Kernel    string
	Truncated bool
	Merged    bool // see Kernel.Merged
	Calls     int  // visible launches summed over profiled cases

	// E is d log(score) / d x when the kernel's time shrinks by the
	// fraction x on every shape: cutting it by 10% raises log(score) by
	// about 0.1·E.
	E float64
}

// ElasticityTable is the per-kernel elasticity table for one profile.
type ElasticityTable struct {
	Kernels []Elasticity // by E descending

	// Score is the geometric mean over loads of Σ weight·BaselineNS /
	// Σ weight·MedianNS: the score the elasticities are derivatives of.
	// It is the judge's score only to the extent that the judge's paired
	// estimate agrees with this ratio of medians.
	Score float64
	Loads int

	// Coverage is Σ E: the fraction of judged time the kernel tables
	// account for, averaged over loads. Below 1 the rest is gaps and
	// launch overhead the tables do not show, or hidden groups; above 1
	// overlapping kernels count their waits.
	Coverage float64
	Hidden   int
	Overlap  bool
}

// Elasticities computes each kernel's elasticity of the log score.
//
// The score is a geometric mean over L loads of B_l / C_l, where C_l is the
// weighted sum Σ w·MedianNS of the load's shapes, so
//
//	E_k = (1/L) Σ_l Σ_{s in l} w_s·t_{k,s} / C_l
//
// with t_{k,s} the kernel's time in the profile of shape s. The kernel
// times come from the profile reports and C_l from the benchmarks; a
// profiled load with no benchmark is an error. Shapes that were timed but
// not profiled contribute nothing, which lowers Coverage.
//
// The formula applies to any weighted-geomean benchmark; only the inputs are
// judge-specific.
func Elasticities(f *File) (ElasticityTable, error) {
	type load struct{ base, cand float64 }
	loads := map[string]*load{}
	var order []string
	for i, b := range f.Benchmarks {
		w := b.Case.Weight
		if !(w > 0) || math.IsInf(w, 0) || !(b.BaselineNS > 0) || !(b.MedianNS > 0) {
			return ElasticityTable{}, fmt.Errorf("benchmark %d: weight %v, baseline %v ns, median %v ns; want finite and positive", i, w, b.BaselineNS, b.MedianNS)
		}
		l := loads[b.Case.Load]
		if l == nil {
			l = new(load)
			loads[b.Case.Load] = l
			order = append(order, b.Case.Load)
		}
		l.base += w * b.BaselineNS
		l.cand += w * b.MedianNS
	}
	if len(loads) == 0 {
		return ElasticityTable{}, errors.New("no benchmarks")
	}
	var t ElasticityTable
	t.Loads = len(loads)
	var logSum float64
	for _, name := range order {
		logSum += math.Log(loads[name].base / loads[name].cand)
	}
	t.Score = math.Exp(logSum / float64(t.Loads))

	rows := map[string]*Elasticity{}
	for i, p := range f.Profiles {
		rep, err := ParseReport(p.Report)
		if err != nil {
			return ElasticityTable{}, fmt.Errorf("profile %d: %w", i, err)
		}
		l := loads[p.Case.Load]
		if l == nil {
			return ElasticityTable{}, fmt.Errorf("profile %d: load %q has no benchmark", i, p.Case.Load)
		}
		w := p.Case.Weight
		if !(w > 0) || math.IsInf(w, 0) {
			return ElasticityTable{}, fmt.Errorf("profile %d: weight %v, want finite and positive", i, w)
		}
		t.Hidden += rep.Hidden
		t.Overlap = t.Overlap || rep.Overlap
		seen := map[string]bool{}
		for _, row := range rep.Rows {
			e := rows[row.Kernel]
			if e == nil {
				e = &Elasticity{Kernel: row.Kernel}
				rows[row.Kernel] = e
			}
			e.Truncated = e.Truncated || row.Truncated
			e.Merged = e.Merged || seen[row.Kernel]
			seen[row.Kernel] = true
			e.Calls += row.Calls
			de := w * float64(row.Time) / l.cand / float64(t.Loads)
			e.E += de
			t.Coverage += de
		}
	}
	for _, e := range rows {
		t.Kernels = append(t.Kernels, *e)
	}
	sort.Slice(t.Kernels, func(i, j int) bool {
		a, b := t.Kernels[i], t.Kernels[j]
		if a.E != b.E {
			return a.E > b.E
		}
		return a.Kernel < b.Kernel
	})
	return t, nil
}
