package cudaspec

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
// targetPredictions must have length len(draftTokens) + 1.
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
			// Speculative mismatch: emit correct target prediction and stop
			return VerificationResult{
				AcceptedTokens: accepted,
				BonusToken:     targetTok,
				TotalEmitted:   len(accepted) + 1,
				AcceptedCount:  len(accepted),
			}
		}
	}

	// All draft tokens were accepted: emit next bonus token from position len(draftTokens)
	bonus := int32(-1)
	if len(targetPredictions) > len(draftTokens) {
		bonus = targetPredictions[len(draftTokens)]
	}

	return VerificationResult{
		AcceptedTokens: accepted,
		BonusToken:     bonus,
		TotalEmitted:   len(accepted) + 1,
		AcceptedCount:  len(accepted),
	}
}

// VerifyStochastic implements speculative sampling via rejection sampling.
func VerifyStochastic(targetProbs [][]float32, draftProbs [][]float32, draftTokens []int32, rng *rand.Rand) VerificationResult {
	var accepted []int32

	for i, draftTok := range draftTokens {
		pTarget := targetProbs[i][draftTok]
		pDraft := draftProbs[i][draftTok]

		acceptProb := float32(1.0)
		if pDraft > 0 {
			acceptProb = pTarget / pDraft
		}

		if rng.Float32() <= acceptProb {
			accepted = append(accepted, draftTok)
		} else {
			// Reject draft token: sample adjusted difference distribution
			adjustedProbs := make([]float32, len(targetProbs[i]))
			sum := float32(0.0)
			for vocabIdx := range adjustedProbs {
				diff := targetProbs[i][vocabIdx] - draftProbs[i][vocabIdx]
				if diff > 0 {
					adjustedProbs[vocabIdx] = diff
					sum += diff
				}
			}

			bonus := sampleCategorical(adjustedProbs, sum, rng)
			return VerificationResult{
				AcceptedTokens: accepted,
				BonusToken:     bonus,
				TotalEmitted:   len(accepted) + 1,
				AcceptedCount:  len(accepted),
			}
		}
	}

	// Sample bonus token from final target distribution
	finalIdx := len(draftTokens)
	var bonus int32
	if finalIdx < len(targetProbs) {
		sum := float32(0.0)
		for _, p := range targetProbs[finalIdx] {
			sum += p
		}
		bonus = sampleCategorical(targetProbs[finalIdx], sum, rng)
	}

	return VerificationResult{
		AcceptedTokens: accepted,
		BonusToken:     bonus,
		TotalEmitted:   len(accepted) + 1,
		AcceptedCount:  len(accepted),
	}
}

func sampleCategorical(probs []float32, sum float32, rng *rand.Rand) int32 {
	if sum <= 0 {
		return 0
	}
	r := rng.Float32() * sum
	cum := float32(0.0)
	for i, p := range probs {
		cum += p
		if cum >= r {
			return int32(i)
		}
	}
	return int32(len(probs) - 1)
}
