package qualify

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
)

// A KernelRecord is one measured kernel execution.
type KernelRecord struct {
	Name       string  `json:"name"`
	DurationNs float64 `json:"duration_ns"`
	Bytes      float64 `json:"bytes"` // DRAM bytes moved
	FLOPs      float64 `json:"flops"`
}

// A Ceiling is a device's measured peak memory bandwidth and compute rate,
// such as from a STREAM-style copy and an MMA loop, not a datasheet value.
type Ceiling struct {
	Device       string  `json:"device"`
	BandwidthGBs float64 `json:"bandwidth_gbs"` // 1e9 bytes per second
	FLOPS        float64 `json:"flops"`         // floating-point operations per second
}

func (c Ceiling) check() error {
	if !(c.BandwidthGBs > 0) || !(c.FLOPS > 0) {
		return fmt.Errorf("ceiling %v GB/s, %v FLOP/s", c.BandwidthGBs, c.FLOPS)
	}
	return nil
}

// A Run is a set of kernel records measured against one ceiling. Records
// with the same name are repeated samples of that kernel.
type Run struct {
	Ceiling Ceiling        `json:"ceiling"`
	Records []KernelRecord `json:"records"`
}

// A RooflinePoint places one kernel record under the roofline.
type RooflinePoint struct {
	Name      string
	Intensity float64 // FLOPs per byte; +Inf with no bytes
	Bound     string  // "memory" or "compute": the ceiling that limits the kernel
	// IdealNs is the roofline time max(Bytes/bandwidth, FLOPs/FLOPS).
	IdealNs float64
	// Fraction is IdealNs/DurationNs: 1 at the roofline, smaller below
	// it. A value above 1 means the record or the ceiling is wrong.
	Fraction float64
}

// Roofline reports each record's position under c.
func Roofline(records []KernelRecord, c Ceiling) ([]RooflinePoint, error) {
	if err := c.check(); err != nil {
		return nil, fmt.Errorf("roofline: %w", err)
	}
	out := make([]RooflinePoint, 0, len(records))
	for _, r := range records {
		if !(r.DurationNs > 0) || r.Bytes < 0 || r.FLOPs < 0 || r.Bytes == 0 && r.FLOPs == 0 {
			return nil, fmt.Errorf("roofline: %s: duration %v ns, %v bytes, %v FLOPs", r.Name, r.DurationNs, r.Bytes, r.FLOPs)
		}
		memNs := r.Bytes / c.BandwidthGBs // bytes / (1e9 B/s) = ns
		cmpNs := r.FLOPs / c.FLOPS * 1e9
		p := RooflinePoint{Name: r.Name, Intensity: math.Inf(1), Bound: "compute", IdealNs: cmpNs}
		if r.Bytes > 0 {
			p.Intensity = r.FLOPs / r.Bytes
		}
		if memNs >= cmpNs {
			p.Bound, p.IdealNs = "memory", memNs
		}
		p.Fraction = p.IdealNs / r.DurationNs
		out = append(out, p)
	}
	return out, nil
}

// RegressionConfig configures [CompareRuns].
type RegressionConfig struct {
	Threshold        float64 // relative median slowdown that counts; 0 means 0.05
	Alpha            float64 // one-sided sign-test level; 0 means 0.05
	CeilingTolerance float64 // relative ceiling difference allowed; 0 means 0.01
}

func (c RegressionConfig) norm() RegressionConfig {
	if c.Threshold == 0 {
		c.Threshold = 0.05
	}
	if c.Alpha == 0 {
		c.Alpha = 0.05
	}
	if c.CeilingTolerance == 0 {
		c.CeilingTolerance = 0.01
	}
	return c
}

// ErrCeilingMismatch reports a baseline measured against a different ceiling.
var ErrCeilingMismatch = errors.New("baseline ceiling differs")

// A Regression compares one kernel's samples with its baseline.
type Regression struct {
	Name             string
	BaselineMedianNs float64
	CurrentMedianNs  float64
	Ratio            float64 // CurrentMedianNs / BaselineMedianNs
	// Above and Below count current samples above and below the baseline
	// median; samples equal to it are dropped.
	Above, Below int
	PValue       float64 // one-sided sign-test p-value in the direction of Ratio
	Regressed    bool
	Improved     bool
	Note         string // why a kernel was not compared
}

// CompareRuns compares the durations of cur with those of base, kernel by
// kernel. A kernel regressed when both
//
//   - its current median is more than Threshold above the baseline median, and
//   - a one-sided sign test of the current samples against the baseline
//     median rejects "no slowdown" at level Alpha: the probability of at
//     least Above of Above+Below fair coin flips landing above is ≤ Alpha.
//
// Improvement is the mirror image. Using medians and signs, a few outlier
// samples move neither test, and a regression needs enough samples to be
// significant: with every sample slower, at least 5 samples at Alpha 0.05.
//
// CompareRuns returns an error wrapping [ErrCeilingMismatch] when the two
// runs' devices differ or their ceilings differ by more than
// CeilingTolerance, since the durations are then not comparable. Kernels
// present in only one run are reported with a Note and not compared.
func CompareRuns(base, cur Run, cfg RegressionConfig) ([]Regression, error) {
	cfg = cfg.norm()
	if err := base.Ceiling.check(); err != nil {
		return nil, fmt.Errorf("compare runs: baseline: %w", err)
	}
	if err := cur.Ceiling.check(); err != nil {
		return nil, fmt.Errorf("compare runs: current: %w", err)
	}
	rel := func(a, b float64) float64 { return math.Abs(a-b) / b }
	if base.Ceiling.Device != cur.Ceiling.Device ||
		rel(cur.Ceiling.BandwidthGBs, base.Ceiling.BandwidthGBs) > cfg.CeilingTolerance ||
		rel(cur.Ceiling.FLOPS, base.Ceiling.FLOPS) > cfg.CeilingTolerance {
		return nil, fmt.Errorf("compare runs: %w: baseline %+v, current %+v", ErrCeilingMismatch, base.Ceiling, cur.Ceiling)
	}
	bs, cs := samples(base.Records), samples(cur.Records)
	names := make([]string, 0, len(bs)+len(cs))
	for n := range bs {
		names = append(names, n)
	}
	for n := range cs {
		if _, ok := bs[n]; !ok {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	var out []Regression
	for _, n := range names {
		b, c := bs[n], cs[n]
		switch {
		case len(b) == 0:
			out = append(out, Regression{Name: n, Note: "no baseline samples"})
			continue
		case len(c) == 0:
			out = append(out, Regression{Name: n, Note: "no current samples"})
			continue
		}
		r := Regression{Name: n, BaselineMedianNs: median(b), CurrentMedianNs: median(c)}
		r.Ratio = r.CurrentMedianNs / r.BaselineMedianNs
		for _, v := range c {
			switch {
			case v > r.BaselineMedianNs:
				r.Above++
			case v < r.BaselineMedianNs:
				r.Below++
			}
		}
		if r.Ratio >= 1 {
			r.PValue = binomialTail(r.Above, r.Above+r.Below)
			r.Regressed = r.Ratio > 1+cfg.Threshold && r.PValue <= cfg.Alpha
		} else {
			r.PValue = binomialTail(r.Below, r.Above+r.Below)
			r.Improved = r.Ratio < 1-cfg.Threshold && r.PValue <= cfg.Alpha
		}
		out = append(out, r)
	}
	return out, nil
}

func samples(recs []KernelRecord) map[string][]float64 {
	m := map[string][]float64{}
	for _, r := range recs {
		m[r.Name] = append(m[r.Name], r.DurationNs)
	}
	return m
}

func median(v []float64) float64 {
	s := slices.Clone(v)
	slices.Sort(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// binomialTail returns P(X ≥ k) for X ~ Binomial(n, 1/2), and 1 when n is 0.
func binomialTail(k, n int) float64 {
	if n == 0 {
		return 1
	}
	p := 0.0
	for i := k; i <= n; i++ {
		lg, _ := math.Lgamma(float64(n + 1))
		li, _ := math.Lgamma(float64(i + 1))
		lj, _ := math.Lgamma(float64(n - i + 1))
		p += math.Exp(lg - li - lj - float64(n)*math.Ln2)
	}
	return min(p, 1)
}
