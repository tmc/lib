package main

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"time"
)

// ABBAPair records one paired trial comparing candidate (A) against baseline (B)
// under either AB or BA ordering.
type ABBAPair struct {
	Order             string        // "AB" or "BA"
	CandidateTPOT     time.Duration // median per-token decode latency of candidate A
	BaselineTPOT      time.Duration // median per-token decode latency of baseline B
	CandidateFallback int           // fallback count for A (must be 0)
	BaselineFallback  int           // fallback count for B (must be 0)
	TokensMatch       bool          // whether greedy token sequences or parity check matched
}

// ABBASummary holds the statistical evaluation of a set of paired ABBA trials.
type ABBASummary struct {
	Pairs               int
	CandidateWins       int
	BaselineWins        int
	Ties                int
	MedianCandidateTPOT time.Duration
	MedianBaselineTPOT  time.Duration
	MedianImprovement   float64 // (MedianBaselineTPOT - MedianCandidateTPOT) / MedianBaselineTPOT
	OneSidedPValue      float64 // exact binomial P(X >= CandidateWins | n = Pairs, p = 0.5)
	ZeroFallback        bool
	AllTokensMatch      bool
	Pass                bool
}

// SummarizeStepDurations summarizes per-step decode durations for a single
// trial. When len(steps) > 1, it drops steps[0] (the first decode step where
// graph build or warmup occurs) so that steady-state steps 1..N-1 are
// summarized. It returns an error if steps is empty or contains any
// non-positive duration.
func SummarizeStepDurations(steps []time.Duration) (min, median, mean, max time.Duration, err error) {
	if len(steps) == 0 {
		return 0, 0, 0, 0, errors.New("empty step durations")
	}
	for i, d := range steps {
		if d <= 0 {
			return 0, 0, 0, 0, fmt.Errorf("step %d: non-positive duration %v", i, d)
		}
	}
	steady := steps
	if len(steps) > 1 {
		steady = steps[1:]
	}
	sorted := slices.Clone(steady)
	slices.Sort(sorted)

	min = sorted[0]
	max = sorted[len(sorted)-1]
	median = medianDuration(sorted)

	var total time.Duration
	for _, d := range sorted {
		total += d
	}
	mean = total / time.Duration(len(sorted))
	return min, median, mean, max, nil
}

// EvaluateABBAPairs evaluates a slice of paired ABBA trials against the
// preregistered retain criteria: at least 10 pairs, >= 9/10 pair wins for the
// candidate (one-sided sign test p <= 11/1024 < 0.05 at n=10), >= 5% median
// TPOT improvement, zero fallbacks in either arm, and matching tokens across
// all pairs.
func EvaluateABBAPairs(pairs []ABBAPair) (ABBASummary, error) {
	if len(pairs) == 0 {
		return ABBASummary{}, errors.New("empty ABBA pairs")
	}
	cand := make([]time.Duration, len(pairs))
	base := make([]time.Duration, len(pairs))
	summary := ABBASummary{
		Pairs:          len(pairs),
		ZeroFallback:   true,
		AllTokensMatch: true,
	}
	for i, p := range pairs {
		if p.Order != "AB" && p.Order != "BA" {
			return ABBASummary{}, fmt.Errorf("pair %d: invalid order %q (want AB or BA)", i, p.Order)
		}
		if p.CandidateTPOT <= 0 {
			return ABBASummary{}, fmt.Errorf("pair %d: non-positive candidate TPOT %v", i, p.CandidateTPOT)
		}
		if p.BaselineTPOT <= 0 {
			return ABBASummary{}, fmt.Errorf("pair %d: non-positive baseline TPOT %v", i, p.BaselineTPOT)
		}
		if p.CandidateFallback < 0 || p.BaselineFallback < 0 {
			return ABBASummary{}, fmt.Errorf("pair %d: negative fallback count (%d, %d)", i, p.CandidateFallback, p.BaselineFallback)
		}
		cand[i] = p.CandidateTPOT
		base[i] = p.BaselineTPOT
		switch {
		case p.CandidateTPOT < p.BaselineTPOT:
			summary.CandidateWins++
		case p.CandidateTPOT > p.BaselineTPOT:
			summary.BaselineWins++
		default:
			summary.Ties++
		}
		if p.CandidateFallback != 0 || p.BaselineFallback != 0 {
			summary.ZeroFallback = false
		}
		if !p.TokensMatch {
			summary.AllTokensMatch = false
		}
	}

	slices.Sort(cand)
	slices.Sort(base)
	summary.MedianCandidateTPOT = medianDuration(cand)
	summary.MedianBaselineTPOT = medianDuration(base)
	summary.MedianImprovement = float64(summary.MedianBaselineTPOT-summary.MedianCandidateTPOT) / float64(summary.MedianBaselineTPOT)
	summary.OneSidedPValue = binomialUpperTailHalf(summary.Pairs, summary.CandidateWins)

	summary.Pass = summary.Pairs >= 10 &&
		summary.CandidateWins*10 >= 9*summary.Pairs &&
		summary.OneSidedPValue <= 11.0/1024.0 &&
		summary.MedianImprovement >= 0.05 &&
		summary.ZeroFallback &&
		summary.AllTokensMatch
	return summary, nil
}

func medianDuration(sorted []time.Duration) time.Duration {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return sorted[n/2]
	}
	lo, hi := sorted[n/2-1], sorted[n/2]
	return lo + (hi-lo)/2
}

// binomialUpperTailHalf returns the exact one-sided binomial sign-test
// p-value P(X >= k | X ~ Binomial(n, 0.5)) = sum_{i=k}^n C(n, i) * 0.5^n.
func binomialUpperTailHalf(n, k int) float64 {
	if k <= 0 {
		return 1.0
	}
	if k > n {
		return 0.0
	}
	var sum float64
	for i := k; i <= n; i++ {
		sum += binomialCoeff(n, i)
	}
	return math.Ldexp(sum, -n)
}

func binomialCoeff(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	if k > n-k {
		k = n - k
	}
	c := 1.0
	for i := 1; i <= k; i++ {
		c = c * float64(n-k+i) / float64(i)
	}
	return math.Round(c)
}
