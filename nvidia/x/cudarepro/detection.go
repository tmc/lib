package cudarepro

import (
	"fmt"
	"math"
)

// DetectionProbability calculates the probability of observing at least one reproduction
// across attempts independent trials, where each trial reproduces with probability
// p = float64(reproductions) / float64(repeats).
//
// The calculation assumes independent identically distributed (i.i.d.) attempts.
// An empirical frequency does not establish an i.i.d. model or provide a statistical confidence bound.
func DetectionProbability(reproductions, repeats, attempts int) (float64, error) {
	if repeats <= 0 {
		return 0, fmt.Errorf("calculate detection probability: invalid replay count %d", repeats)
	}
	if reproductions < 0 || reproductions > repeats {
		return 0, fmt.Errorf("calculate detection probability: invalid reproductions %d for %d replays", reproductions, repeats)
	}
	if attempts < 0 {
		return 0, fmt.Errorf("calculate detection probability: negative attempts %d", attempts)
	}
	if attempts == 0 || reproductions == 0 {
		return 0, nil
	}
	if reproductions == repeats {
		return 1, nil
	}
	rate := float64(reproductions) / float64(repeats)
	// P = 1 - (1 - rate)^attempts.
	// Use -expm1(attempts * log1p(-rate)) for numerical stability.
	prob := -math.Expm1(float64(attempts) * math.Log1p(-rate))
	if prob > 1 {
		prob = 1
	} else if prob < 0 {
		prob = 0
	}
	return prob, nil
}

// DetectionAttempts plans a finite export run under an explicit model:
// independent attempts, each reproducing with probability equal to the measured
// frequency (reproductions / repeats). An empirical frequency does not establish that
// model or provide a confidence bound. The caller must retain this assumption with the export.
func DetectionAttempts(identity Identity, measured Measurement, repeats int, targetProbability float64, maxAttempts int) (int, error) {
	if repeats <= 0 || len(measured.Runs) != repeats || maxAttempts <= 0 || math.IsNaN(targetProbability) || targetProbability <= 0 || targetProbability >= 1 {
		return 0, fmt.Errorf("plan detection: invalid replay count, probability, or attempt limit")
	}
	if _, err := AcceptRate(identity, measured, measured, true); err != nil {
		return 0, fmt.Errorf("plan detection: %w", err)
	}
	if measured.Reproductions == repeats {
		return 1, nil
	}
	logMiss := math.Log1p(-float64(measured.Reproductions) / float64(repeats))
	logTarget := math.Log1p(-targetProbability)
	count := math.Ceil(logTarget / logMiss)
	// Avoid float-to-int conversion overflow at maximum int boundary.
	if math.IsInf(count, 0) || count < 1 || count >= float64(int(^uint(0)>>1)) || count > float64(maxAttempts) {
		return 0, fmt.Errorf("plan detection: required attempts exceed the declared limit")
	}
	n := int(count)
	if float64(n)*logMiss > logTarget {
		if n == maxAttempts {
			return 0, fmt.Errorf("plan detection: required attempts exceed the declared limit")
		}
		n++
	}
	return n, nil
}

// DetectionProbability calculates the detection probability for the given number
// of attempts under the assumption of independent attempts with rate m.Reproductions / len(m.Runs).
func (m Measurement) DetectionProbability(attempts int) (float64, error) {
	return DetectionProbability(m.Reproductions, len(m.Runs), attempts)
}

// DetectionAttempts calculates the number of attempts needed to achieve targetProbability,
// bounded by maxAttempts, verifying that m matches identity.
func (m Measurement) DetectionAttempts(identity Identity, targetProbability float64, maxAttempts int) (int, error) {
	return DetectionAttempts(identity, m, len(m.Runs), targetProbability, maxAttempts)
}
