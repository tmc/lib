package dsv4eval

import (
	"fmt"
	"math"
)

// Perplexity is a summary of token log probabilities.
type Perplexity struct {
	Tokens, Scored   int
	NLL, AvgNLL, PPL float64
}

// ScorePerplexity scores one logit vector per target token. Log-sum-exp is
// accumulated in float64; vectors must be finite and targets in range.
func ScorePerplexity(logits [][]float32, targets []int) (Perplexity, error) {
	if len(logits) != len(targets) {
		return Perplexity{}, fmt.Errorf("score perplexity: %d vectors for %d targets", len(logits), len(targets))
	}
	p := Perplexity{Tokens: len(targets), Scored: len(targets)}
	for i, row := range logits {
		if len(row) == 0 || targets[i] < 0 || targets[i] >= len(row) {
			return Perplexity{}, fmt.Errorf("score perplexity: target %d at row %d is out of range", targets[i], i)
		}
		lp, err := LogProb(row, targets[i])
		if err != nil {
			return Perplexity{}, fmt.Errorf("score perplexity row %d: %w", i, err)
		}
		p.NLL -= lp
	}
	if p.Scored > 0 {
		p.AvgNLL = p.NLL / float64(p.Scored)
		p.PPL = math.Exp(p.AvgNLL)
	}
	return p, nil
}

// LogProb returns the natural log probability of target using a stable
// float64 log-sum-exp over float32 logits.
func LogProb(logits []float32, target int) (float64, error) {
	if target < 0 || target >= len(logits) || len(logits) == 0 {
		return 0, fmt.Errorf("target %d out of range", target)
	}
	maxLogit := math.Inf(-1)
	for _, x := range logits {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return 0, fmt.Errorf("non-finite logit")
		}
		if float64(x) > maxLogit {
			maxLogit = float64(x)
		}
	}
	var sum float64
	for _, x := range logits {
		sum += math.Exp(float64(x) - maxLogit)
	}
	return float64(logits[target]) - maxLogit - math.Log(sum), nil
}

// Consistency compares live and fresh logits at matching decode positions.
type Consistency struct {
	Positions   int
	MaxAbs, RMS float64
	Top1Matches int
}

// Compare returns numerical and top-1 agreement for two equal-shaped traces.
func Compare(live, fresh [][]float32) (Consistency, error) {
	if len(live) != len(fresh) {
		return Consistency{}, fmt.Errorf("compare consistency: %d positions versus %d", len(live), len(fresh))
	}
	c := Consistency{Positions: len(live)}
	var sum float64
	elements := 0
	for i := range live {
		if len(live[i]) != len(fresh[i]) || len(live[i]) == 0 {
			return Consistency{}, fmt.Errorf("compare consistency: position %d has mismatched logits", i)
		}
		elements += len(live[i])
		li, fi := argmax(live[i]), argmax(fresh[i])
		if li == fi {
			c.Top1Matches++
		}
		for j := range live[i] {
			d := math.Abs(float64(live[i][j]) - float64(fresh[i][j]))
			if d > c.MaxAbs {
				c.MaxAbs = d
			}
			sum += d * d
		}
	}
	if elements > 0 {
		c.RMS = math.Sqrt(sum / float64(elements))
	}
	return c, nil
}
func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}
