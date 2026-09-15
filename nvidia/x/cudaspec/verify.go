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
	emitted := len(accepted)
	if len(targetPredictions) > len(draftTokens) {
		bonus = targetPredictions[len(draftTokens)]
		emitted++
	}

	return VerificationResult{
		AcceptedTokens: accepted,
		BonusToken:     bonus,
		TotalEmitted:   emitted,
		AcceptedCount:  len(accepted),
	}
}

// VerifyStochastic implements speculative sampling via rejection sampling.
//
// Draft token i is accepted with probability min(1, p_target/p_draft). On
// rejection it samples the residual distribution max(0, p_target - p_draft),
// which is what makes the emitted sequence distributed exactly as the target
// model's own samples. Verification stops early at the first position not
// covered by both distributions.
func VerifyStochastic(targetProbs [][]float32, draftProbs [][]float32, draftTokens []int32, rng *rand.Rand) VerificationResult {
	var accepted []int32

	i := 0
	for ; i < len(draftTokens); i++ {
		draftTok := draftTokens[i]
		if i >= len(targetProbs) || i >= len(draftProbs) {
			break
		}
		if draftTok < 0 || int(draftTok) >= len(targetProbs[i]) || int(draftTok) >= len(draftProbs[i]) {
			break
		}

		pTarget := targetProbs[i][draftTok]
		pDraft := draftProbs[i][draftTok]

		// A token the draft model could not have produced carries no
		// information about the target, so accept it only if the target model
		// gives it positive mass.
		var acceptProb float32
		switch {
		case pDraft > 0:
			acceptProb = pTarget / pDraft
		case pTarget > 0:
			acceptProb = 1
		}

		if rng.Float32() < acceptProb {
			accepted = append(accepted, draftTok)
			continue
		}

		// Reject draft token: sample the residual distribution. If the target
		// puts no mass outside the draft, fall back to the target itself
		// rather than emitting token 0.
		residual := make([]float32, len(targetProbs[i]))
		var sum float32
		for vocabIdx := range residual {
			diff := targetProbs[i][vocabIdx]
			if vocabIdx < len(draftProbs[i]) {
				diff -= draftProbs[i][vocabIdx]
			}
			if diff > 0 {
				residual[vocabIdx] = diff
				sum += diff
			}
		}
		if sum <= 0 {
			residual = targetProbs[i]
			sum = 0
			for _, p := range residual {
				if p > 0 {
					sum += p
				}
			}
		}

		return VerificationResult{
			AcceptedTokens: accepted,
			BonusToken:     sampleCategorical(residual, sum, rng),
			TotalEmitted:   len(accepted) + 1,
			AcceptedCount:  len(accepted),
		}
	}

	// Sample bonus token from the target distribution at the first unverified
	// position, if the caller supplied one.
	if i < len(targetProbs) {
		var sum float32
		for _, p := range targetProbs[i] {
			if p > 0 {
				sum += p
			}
		}
		return VerificationResult{
			AcceptedTokens: accepted,
			BonusToken:     sampleCategorical(targetProbs[i], sum, rng),
			TotalEmitted:   len(accepted) + 1,
			AcceptedCount:  len(accepted),
		}
	}

	return VerificationResult{
		AcceptedTokens: accepted,
		BonusToken:     -1,
		TotalEmitted:   len(accepted),
		AcceptedCount:  len(accepted),
	}
}

// sampleCategorical draws an index from probs, whose positive entries total
// sum. It skips zero-probability entries: a token the distribution rules out
// must never be emitted.
func sampleCategorical(probs []float32, sum float32, rng *rand.Rand) int32 {
	if sum <= 0 {
		return -1
	}
	r := rng.Float32() * sum
	cum := float32(0.0)
	for i, p := range probs {
		if p <= 0 {
			continue
		}
		cum += p
		if r < cum {
			return int32(i)
		}
	}
	// Rounding can leave r just past the total; fall back to the last
	// admissible token rather than an excluded one.
	for i := len(probs) - 1; i >= 0; i-- {
		if probs[i] > 0 {
			return int32(i)
		}
	}
	return -1
}
