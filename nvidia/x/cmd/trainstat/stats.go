package main

import (
	"fmt"
	"math"
	"slices"
	"strconv"

	"github.com/aclements/go-moremath/stats"
	"golang.org/x/perf/benchmath"
)

// A sample is one table cell's values.
type sample struct {
	xs   []float64 // values; for merged replicates, in replicate order
	reps []string  // replicate of each value, or nil if every result is a value
	vals []string  // validation set hash of each replicate
	bad  string    // why the cell has no sample, or ""
}

// A summary is a sample's center and confidence interval.
// Lo and Hi are NaN when there is no interval.
type summary struct {
	n              int
	center, lo, hi float64
	normal         bool   // mean and t interval; otherwise benchmath
	pct            string // benchmath's ± range, for !normal
}

// A comparison is the difference of two samples, new minus base.
// Lo and Hi bound the difference and are NaN when there is no interval;
// P is NaN when there is no test.
type comparison struct {
	n1, n2        int
	delta, lo, hi float64
	p             float64
	method        string // paired, welch, point, or benchmath's (U or exact)
	note          string // why a comparison is not paired, or ""
	fmtDelta      string // benchmath's delta, for method U and exact
}

// A stat computes summaries and comparisons at one confidence level.
type stat struct {
	confidence, alpha float64
	thresholds        benchmath.Thresholds
}

func newStat(confidence, alpha float64) *stat {
	t := benchmath.DefaultThresholds
	t.CompareAlpha = alpha
	return &stat{confidence: confidence, alpha: alpha, thresholds: t}
}

// tQuantile returns the two-sided critical value of Student's t.
func (st *stat) tQuantile(dof float64) float64 {
	return stats.InvCDF(stats.TDist{V: dof})(1 - (1-st.confidence)/2)
}

// summarize summarizes s: the mean and a t interval if normal, else
// benchmath's summary under assumption a.
func (st *stat) summarize(s *sample, normal bool, a benchmath.Assumption) summary {
	n := len(s.xs)
	if normal {
		sum := summary{n: n, center: stats.Mean(s.xs), lo: math.NaN(), hi: math.NaN(), normal: true}
		if n >= 2 {
			hw := st.tQuantile(float64(n-1)) * stats.StdDev(s.xs) / math.Sqrt(float64(n))
			sum.lo, sum.hi = sum.center-hw, sum.center+hw
		}
		return sum
	}
	bs := benchmath.NewSample(slices.Clone(s.xs), &st.thresholds)
	bsum := a.Summary(bs, st.confidence)
	return summary{n: n, center: bsum.Center, lo: bsum.Lo, hi: bsum.Hi, pct: bsum.PctRangeString()}
}

// compare compares base and s. Normal samples are compared by a paired t
// test on per-replicate differences when pair is set and both samples
// have the same replicates, and by Welch's t test otherwise. Others are
// compared by benchmath under assumption a, with the difference bounded by
// interval arithmetic on the two summaries.
func (st *stat) compare(base, s *sample, normal, pair bool, a benchmath.Assumption) comparison {
	n1, n2 := len(base.xs), len(s.xs)
	c := comparison{n1: n1, n2: n2, lo: math.NaN(), hi: math.NaN(), p: math.NaN()}
	if !normal {
		b1 := benchmath.NewSample(slices.Clone(base.xs), &st.thresholds)
		b2 := benchmath.NewSample(slices.Clone(s.xs), &st.thresholds)
		s1, s2 := a.Summary(b1, st.confidence), a.Summary(b2, st.confidence)
		c.delta = s2.Center - s1.Center
		if n1 == 1 && n2 == 1 {
			c.method = "point"
			return c
		}
		cmp := a.Compare(b1, b2)
		c.p = cmp.P
		c.lo, c.hi = s2.Lo-s1.Hi, s2.Hi-s1.Lo
		c.method = "U"
		if a == benchmath.AssumeExact {
			c.method = "exact"
		}
		c.fmtDelta = cmp.FormatDelta(s1.Center, s2.Center)
		return c
	}

	c.delta = stats.Mean(s.xs) - stats.Mean(base.xs)
	var se, dof float64
	switch {
	case pair && n1 >= 2 && slices.Equal(base.reps, s.reps) && base.reps != nil:
		if i := valMismatch(base, s); i >= 0 {
			c.note = "unpaired: validation sets differ for replicate " + base.reps[i]
		} else {
			d := make([]float64, n1)
			for i := range d {
				d[i] = s.xs[i] - base.xs[i]
			}
			c.method = "paired"
			c.delta = stats.Mean(d)
			se, dof = stats.StdDev(d)/math.Sqrt(float64(n1)), float64(n1-1)
			break
		}
		fallthrough
	case n1 >= 2 && n2 >= 2:
		v1, v2 := stats.Variance(base.xs)/float64(n1), stats.Variance(s.xs)/float64(n2)
		c.method = "welch"
		se = math.Sqrt(v1 + v2)
		dof = (v1 + v2) * (v1 + v2) / (v1*v1/float64(n1-1) + v2*v2/float64(n2-1))
	default:
		c.method = "point"
		return c
	}
	if se == 0 {
		c.lo, c.hi, c.p = c.delta, c.delta, 1
		if c.delta != 0 {
			c.p = 0
		}
		return c
	}
	hw := st.tQuantile(dof) * se
	c.lo, c.hi = c.delta-hw, c.delta+hw
	c.p = 2 * stats.TDist{V: dof}.CDF(-math.Abs(c.delta/se))
	return c
}

// valMismatch returns the index of the first replicate whose validation
// set hash differs between a and b, or -1.
func valMismatch(a, b *sample) int {
	for i := range a.vals {
		if a.vals[i] != b.vals[i] {
			return i
		}
	}
	return -1
}

// significant reports whether c rejects "no difference" at level alpha.
func (c comparison) significant(alpha float64) bool {
	return c.p < alpha
}

// info returns the test and sample sizes of c, such as "p=0.013 n=5 paired".
func (c comparison) info() string {
	n := strconv.Itoa(c.n1)
	if c.n1 != c.n2 {
		n = fmt.Sprintf("%d+%d", c.n1, c.n2)
	}
	switch c.method {
	case "point":
		if c.n1 == 1 && c.n2 == 1 {
			return "n=1"
		}
		return "n=" + n + " no CI"
	case "paired", "welch":
		return fmt.Sprintf("%s n=%s %s", formatP(c.p), n, c.method)
	}
	return fmt.Sprintf("%s n=%s", formatP(c.p), n)
}

// formatP formats a p-value to three decimals, or as p<0.001.
func formatP(p float64) string {
	if p < 0.001 {
		return "p<0.001"
	}
	return fmt.Sprintf("p=%.3f", p)
}

// decimals returns the number of decimal places that show two
// significant digits of a half-width hw, or -1 if hw is not positive and
// finite.
func decimals(hw float64) int {
	if !(hw > 0) || math.IsInf(hw, 0) {
		return -1
	}
	return min(max(1-int(math.Floor(math.Log10(hw))), 0), 12)
}

// format formats v with dec decimal places, or to 4 significant digits if
// dec < 0. Signed values carry a sign.
func format(v float64, dec int, signed bool) string {
	switch {
	case dec < 0 && signed:
		return fmt.Sprintf("%+.4g", v)
	case dec < 0:
		return fmt.Sprintf("%.4g", v)
	case signed:
		return fmt.Sprintf("%+.*f", dec, v)
	}
	return fmt.Sprintf("%.*f", dec, v)
}
