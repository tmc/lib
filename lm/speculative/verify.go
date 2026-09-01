// SPDX-License-Identifier: Apache-2.0

package speculative

import (
	"math/rand"
)

// VerificationResult holds accepted candidate tokens and the final transition token.
type VerificationResult struct {
	AcceptedTokens []int32
	BonusToken     int32
	TotalEmitted   int
	AcceptedCount  int
}

// VerifyGreedy performs greedy speculative verification against top predictions of target model.
func VerifyGreedy(targetPredictions []int32, draftTokens []int32) VerificationResult {
	var accepted []int32
	for i, draftTok := range draftTokens {
		if i >= len(targetPredictions) {
			break
		}
		targetTok := targetPredictions[i]
		if draftTok == targetTok {
			accepted = append(accepted, draftTok)
		} else {
			return VerificationResult{
				AcceptedTokens: accepted,
				BonusToken:     targetTok,
				TotalEmitted:   len(accepted) + 1,
				AcceptedCount:  len(accepted),
			}
		}
	}

	bonus := int32(-1)
	if len(targetPredictions) > len(draftTokens) {
		bonus = targetPredictions[len(draftTokens)]
	}

	total := len(accepted)
	if bonus != -1 {
		total++
	}

	return VerificationResult{
		AcceptedTokens: accepted,
		BonusToken:     bonus,
		TotalEmitted:   total,
		AcceptedCount:  len(accepted),
	}
}

// VerifyStochastic performs rejection sampling for temperature-based speculative verification.
func VerifyStochastic(draftProbs [][]float32, targetProbs [][]float32, draftTokens []int32, rng *rand.Rand) VerificationResult {
	var accepted []int32
	for i, draftTok := range draftTokens {
		if i >= len(targetProbs) || i >= len(draftProbs) {
			break
		}
		tProb := targetProbs[i][draftTok]
		dProb := draftProbs[i][draftTok]

		acceptRatio := float32(1.0)
		if dProb > 0 {
			acceptRatio = tProb / dProb
		}

		r := float32(1.0)
		if rng != nil {
			r = rng.Float32()
		}

		if r <= acceptRatio {
			accepted = append(accepted, draftTok)
		} else {
			bonus := sampleResidual(targetProbs[i], draftProbs[i], rng)
			return VerificationResult{
				AcceptedTokens: accepted,
				BonusToken:     bonus,
				TotalEmitted:   len(accepted) + 1,
				AcceptedCount:  len(accepted),
			}
		}
	}

	bonus := int32(-1)
	if len(targetProbs) > len(draftTokens) {
		bonus = sampleDistribution(targetProbs[len(draftTokens)], rng)
	}

	total := len(accepted)
	if bonus != -1 {
		total++
	}

	return VerificationResult{
		AcceptedTokens: accepted,
		BonusToken:     bonus,
		TotalEmitted:   total,
		AcceptedCount:  len(accepted),
	}
}

func sampleDistribution(probs []float32, rng *rand.Rand) int32 {
	if len(probs) == 0 {
		return 0
	}
	r := float32(0.5)
	if rng != nil {
		r = rng.Float32()
	}
	var cdf float32
	for i, p := range probs {
		cdf += p
		if r <= cdf {
			return int32(i)
		}
	}
	return int32(len(probs) - 1)
}

func sampleResidual(targetProbs, draftProbs []float32, rng *rand.Rand) int32 {
	res := make([]float32, len(targetProbs))
	var sum float32
	for i := range targetProbs {
		diff := targetProbs[i] - draftProbs[i]
		if diff > 0 {
			res[i] = diff
			sum += diff
		}
	}
	if sum > 0 {
		for i := range res {
			res[i] /= sum
		}
	}
	return sampleDistribution(res, rng)
}
