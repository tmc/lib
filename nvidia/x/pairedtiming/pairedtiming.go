package pairedtiming

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

// A Block is one ABBA block: baseline A and candidate B timed in the order
// A1, B1, B2, A2.
type Block struct {
	A1, B1, B2, A2 float64
}

// A ControlBlock times a graph of one trivial operation and a graph of two
// identical ones in the order One1, Two1, Two2, One2.
type ControlBlock struct {
	One1, Two1, Two2, One2 float64
}

// An Estimator selects how a block becomes a ratio.
type Estimator int

const (
	// Geometric is sqrt((A1/B1) * (A2/B2)), after intercept removal.
	Geometric Estimator = iota
	// Sum is (A1+A2) / (B1+B2), after intercept removal.
	Sum
)

func (e Estimator) String() string {
	switch e {
	case Geometric:
		return "geometric"
	case Sum:
		return "sum"
	}
	return fmt.Sprintf("Estimator(%d)", int(e))
}

// ParseEstimator returns the estimator named "geometric" or "sum".
func ParseEstimator(s string) (Estimator, error) {
	switch s {
	case "geometric":
		return Geometric, nil
	case "sum":
		return Sum, nil
	}
	return 0, fmt.Errorf("unknown estimator %q", s)
}

// Intercept returns the median over blocks of (2*One1 - Two1 + 2*One2 - Two2) / 2,
// the time one call costs beyond the work it does. It returns an error if
// there are no blocks or the estimate is not finite and positive; a negative
// estimate means the control did not resolve the intercept, and clamping it
// to zero would hide that.
//
// The one-op samples carry weight 2, so the block order does not cancel
// drift: a linear drift of d per sample adds 1.5d to a block's estimate.
// Take enough blocks that drift in both directions averages out.
func Intercept(blocks []ControlBlock) (float64, error) {
	if len(blocks) == 0 {
		return 0, errors.New("intercept: no control blocks")
	}
	h := make([]float64, len(blocks))
	for i, b := range blocks {
		h[i] = (2*b.One1 - b.Two1 + 2*b.One2 - b.Two2) / 2
	}
	m := median(h)
	if math.IsNaN(m) || math.IsInf(m, 0) || m <= 0 {
		return 0, fmt.Errorf("intercept: unresolved estimate %v", m)
	}
	return m, nil
}

// Ratio returns the block's baseline-to-candidate ratio after subtracting the
// intercept h from every sample. A ratio above 1 means the candidate is
// faster. It returns an error if any sample is not finite or not larger than h.
func (b Block) Ratio(h float64, e Estimator) (float64, error) {
	a1, b1, b2, a2 := b.A1-h, b.B1-h, b.B2-h, b.A2-h
	for _, x := range [...]float64{a1, b1, b2, a2} {
		if math.IsNaN(x) || math.IsInf(x, 0) || x <= 0 {
			return 0, fmt.Errorf("block %v: sample not above intercept %v", b, h)
		}
	}
	switch e {
	case Geometric:
		return math.Sqrt((a1 / b1) * (a2 / b2)), nil
	case Sum:
		return (a1 + a2) / (b1 + b2), nil
	}
	return 0, fmt.Errorf("unknown estimator %v", e)
}

// A Result is the paired estimate for one shape.
type Result struct {
	Ratio     float64   // median block ratio; above 1 means the candidate is faster
	MinRatio  float64   // smallest block ratio
	Baseline  float64   // median baseline sample minus the intercept
	Candidate float64   // Baseline / Ratio; derived, not measured
	Ratios    []float64 // per-block ratios in input order
}

// A Summary is the paired estimate and block-win sign test for one shape's
// ABBA blocks.
type Summary struct {
	Result
	Wins   int     // blocks with ratio > 1
	Losses int     // blocks with ratio < 1
	SignP  float64 // exact two-sided binomial p-value from SignTest(Wins, Losses)
}

// Analyze computes [Paired] with the [Geometric] estimator and [SignTest] over
// the per-block ratios.
func Analyze(blocks []Block, h float64) (Summary, error) {
	r, err := Paired(blocks, h, Geometric)
	if err != nil {
		return Summary{}, err
	}
	s := Summary{Result: r}
	for _, x := range r.Ratios {
		switch {
		case x > 1:
			s.Wins++
		case x < 1:
			s.Losses++
		}
	}
	s.SignP = SignTest(s.Wins, s.Losses)
	return s, nil
}

// Paired returns the paired estimate for blocks after removing the
// intercept h, which may be zero.
func Paired(blocks []Block, h float64, e Estimator) (Result, error) {
	if len(blocks) == 0 {
		return Result{}, errors.New("paired: no blocks")
	}
	ratios := make([]float64, len(blocks))
	base := make([]float64, 0, 2*len(blocks))
	for i, b := range blocks {
		r, err := b.Ratio(h, e)
		if err != nil {
			return Result{}, fmt.Errorf("paired: %w", err)
		}
		ratios[i] = r
		base = append(base, b.A1-h, b.A2-h)
	}
	r := Result{
		Ratio:    median(ratios),
		MinRatio: slices.Min(ratios),
		Baseline: median(base),
		Ratios:   ratios,
	}
	r.Candidate = r.Baseline / r.Ratio
	return r, nil
}

// A Shape is one benchmark case's baseline and candidate time.
type Shape struct {
	Load      string  // group the shape is scored in, such as "decode"
	Weight    float64 // weight of the shape within its load
	Baseline  float64
	Candidate float64
}

// A LoadScore is the weighted ratio of one load.
type LoadScore struct {
	Load      string
	Baseline  float64 // sum of Weight*Baseline
	Candidate float64 // sum of Weight*Candidate
	Speedup   float64 // Baseline / Candidate
}

// A Scored is a suite score and its per-load parts, sorted by load name.
type Scored struct {
	Speedup float64 // geometric mean of the per-load speedups
	Loads   []LoadScore
}

// Score returns the geometric mean over loads of the ratio of weighted
// baseline time to weighted candidate time. Shapes are never averaged as
// ratios, so a short shape cannot outvote a long one in the same load.
func Score(shapes []Shape) (Scored, error) {
	if len(shapes) == 0 {
		return Scored{}, errors.New("score: no shapes")
	}
	idx := map[string]int{}
	var s Scored
	for _, sh := range shapes {
		if !(sh.Weight > 0) || !(sh.Baseline > 0) || !(sh.Candidate > 0) ||
			math.IsInf(sh.Weight+sh.Baseline+sh.Candidate, 0) {
			return Scored{}, fmt.Errorf("score: shape %+v: weight and times must be finite and positive", sh)
		}
		i, ok := idx[sh.Load]
		if !ok {
			i = len(s.Loads)
			idx[sh.Load] = i
			s.Loads = append(s.Loads, LoadScore{Load: sh.Load})
		}
		s.Loads[i].Baseline += sh.Weight * sh.Baseline
		s.Loads[i].Candidate += sh.Weight * sh.Candidate
	}
	slices.SortFunc(s.Loads, func(a, b LoadScore) int {
		if a.Load < b.Load {
			return -1
		}
		if a.Load > b.Load {
			return 1
		}
		return 0
	})
	var sum float64
	for i := range s.Loads {
		l := &s.Loads[i]
		l.Speedup = l.Baseline / l.Candidate
		sum += math.Log(l.Speedup)
	}
	s.Speedup = math.Exp(sum / float64(len(s.Loads)))
	return s, nil
}

// A Series is the ABBA blocks measured for one shape.
type Series struct {
	Name      string
	Load      string
	Weight    float64
	Intercept float64 // replay intercept for this shape; zero for none
	Blocks    []Block
}

// ScoreSeries computes [Paired] for each series and then [Score].
func ScoreSeries(series []Series, e Estimator) (Scored, []Result, error) {
	shapes := make([]Shape, len(series))
	results := make([]Result, len(series))
	for i, s := range series {
		r, err := Paired(s.Blocks, s.Intercept, e)
		if err != nil {
			return Scored{}, nil, fmt.Errorf("series %s: %w", s.Name, err)
		}
		results[i] = r
		shapes[i] = Shape{Load: s.Load, Weight: s.Weight, Baseline: r.Baseline, Candidate: r.Candidate}
	}
	sc, err := Score(shapes)
	if err != nil {
		return Scored{}, nil, err
	}
	return sc, results, nil
}

// An Interval is a percentile bootstrap interval.
type Interval struct {
	Lo, Hi float64
}

// Bootstrap returns the 95% percentile interval of the [ScoreSeries]
// speedup over draws resamples. Each resample draws, independently for each
// series, as many blocks as the series has, with replacement. A block is the
// resampling unit because its four samples share drift. The same seed gives
// the same interval.
func Bootstrap(series []Series, e Estimator, draws int, seed uint64) (Interval, error) {
	if draws < 1 {
		return Interval{}, fmt.Errorf("bootstrap: draws %d < 1", draws)
	}
	if _, _, err := ScoreSeries(series, e); err != nil {
		return Interval{}, fmt.Errorf("bootstrap: %w", err)
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	sample := make([]Series, len(series))
	for i, s := range series {
		sample[i] = s
		sample[i].Blocks = make([]Block, len(s.Blocks))
	}
	speedups := make([]float64, draws)
	for d := range speedups {
		for i, s := range series {
			for j := range sample[i].Blocks {
				sample[i].Blocks[j] = s.Blocks[rng.IntN(len(s.Blocks))]
			}
		}
		sc, _, err := ScoreSeries(sample, e)
		if err != nil {
			return Interval{}, fmt.Errorf("bootstrap: %w", err)
		}
		speedups[d] = sc.Speedup
	}
	slices.Sort(speedups)
	return Interval{
		Lo: speedups[int(0.025*float64(draws))],
		Hi: speedups[min(draws-1, int(0.975*float64(draws)))],
	}, nil
}

// SignTest returns the exact two-sided binomial p-value for wins and losses
// under the null hypothesis that each outcome has probability 1/2. Ties are
// excluded by the caller. With no outcomes it returns 1.
func SignTest(wins, losses int) float64 {
	n := wins + losses
	if n == 0 {
		return 1
	}
	k := min(wins, losses)
	// Sum C(n,i)/2^n for i <= k in log space so large n does not underflow.
	lgn, _ := math.Lgamma(float64(n + 1))
	var p float64
	for i := 0; i <= k; i++ {
		a, _ := math.Lgamma(float64(i + 1))
		b, _ := math.Lgamma(float64(n - i + 1))
		p += math.Exp(lgn - a - b - float64(n)*math.Ln2)
	}
	return min(1, 2*p)
}

func median(x []float64) float64 {
	s := slices.Clone(x)
	slices.Sort(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}
