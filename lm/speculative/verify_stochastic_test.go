// SPDX-License-Identifier: Apache-2.0

package speculative

import (
	"math"
	"math/rand"
	"testing"
)

// drawFrom samples an index from a normalized distribution.
func drawFrom(probs []float32, rng *rand.Rand) int32 {
	r := rng.Float32()
	var cum float32
	for i, p := range probs {
		cum += p
		if r < cum {
			return int32(i)
		}
	}
	return int32(len(probs) - 1)
}

// TestVerifyStochasticMatchesTarget is the defining property of speculative
// sampling: whatever the draft model proposes, the token actually emitted must
// be distributed exactly as a sample from the target model. The main
// acceptance path already held this property; this pins it against regressions
// from the degenerate-case fixes below.
func TestVerifyStochasticMatchesTarget(t *testing.T) {
	target := []float32{0.10, 0.20, 0.30, 0.40}
	draft := []float32{0.40, 0.30, 0.20, 0.10}

	rng := rand.New(rand.NewSource(1))
	const trials = 200000

	counts := make([]int, len(target))
	for i := 0; i < trials; i++ {
		tok := drawFrom(draft, rng)
		res := VerifyStochastic([][]float32{target}, [][]float32{draft}, []int32{tok}, rng)

		var emitted int32
		if len(res.AcceptedTokens) > 0 {
			emitted = res.AcceptedTokens[0]
		} else {
			emitted = res.BonusToken
		}
		if emitted < 0 || int(emitted) >= len(counts) {
			t.Fatalf("trial %d emitted token %d, outside the vocabulary", i, emitted)
		}
		counts[emitted]++
	}

	for tok, want := range target {
		got := float64(counts[tok]) / trials
		// Three standard errors of a binomial proportion, plus slack.
		tol := 3*math.Sqrt(float64(want)*(1-float64(want))/trials) + 1e-3
		if math.Abs(got-float64(want)) > tol {
			t.Errorf("token %d emitted %.4f of the time, want %.4f (tolerance %.4f)", tok, got, want, tol)
		}
	}
}

// TestVerifyStochasticRejectsImpossibleToken covers the pDraft == 0 branch. The
// old code set the acceptance probability to 1 whenever the draft model gave a
// token no mass, so a token that BOTH models rule out was accepted outright.
func TestVerifyStochasticRejectsImpossibleToken(t *testing.T) {
	target := []float32{1, 0}
	draft := []float32{1, 0}

	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 100; i++ {
		// Token 1 has zero probability under both distributions.
		res := VerifyStochastic([][]float32{target}, [][]float32{draft}, []int32{1}, rng)
		if len(res.AcceptedTokens) != 0 {
			t.Fatalf("trial %d accepted token 1, which neither model can produce", i)
		}
		if res.BonusToken != 0 {
			t.Fatalf("trial %d emitted token %d, want 0 (the only token with mass)", i, res.BonusToken)
		}
	}
}

// TestVerifyStochasticEmptyResidual covers the rejection path where the
// residual max(0, target-draft) is entirely zero. The old sampler returned
// token 0 in that case regardless of whether the target gave it any mass.
func TestVerifyStochasticEmptyResidual(t *testing.T) {
	// The target rules out tokens 0 and 1 and is dominated by the draft
	// everywhere, so a rejection leaves no residual mass to sample.
	target := []float32{0, 0, 0.25, 0.25}
	draft := []float32{0, 0, 0.50, 0.50}

	rng := rand.New(rand.NewSource(3))
	rejections := 0
	for i := 0; i < 2000; i++ {
		res := VerifyStochastic([][]float32{target}, [][]float32{draft}, []int32{2}, rng)
		if len(res.AcceptedTokens) > 0 {
			continue
		}
		rejections++
		if res.BonusToken != 2 && res.BonusToken != 3 {
			t.Fatalf("trial %d emitted token %d on rejection, which the target forbids", i, res.BonusToken)
		}
	}
	if rejections == 0 {
		t.Fatal("no rejections occurred, so the empty-residual path was never reached")
	}
	t.Logf("%d of 2000 trials rejected and sampled the fallback", rejections)
}

// TestVerifyStochasticRaggedInput checks that mismatched shapes stop
// verification instead of panicking on an out-of-range index.
func TestVerifyStochasticRaggedInput(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	target := [][]float32{{0.5, 0.5}}
	draft := [][]float32{{0.5, 0.5}}

	// More draft tokens than probability rows.
	res := VerifyStochastic(target, draft, []int32{0, 1, 0}, rng)
	if res.AcceptedCount > 1 {
		t.Errorf("AcceptedCount = %d, want at most 1 with a single probability row", res.AcceptedCount)
	}

	// A draft token outside the vocabulary.
	res = VerifyStochastic(target, draft, []int32{7}, rng)
	if res.AcceptedCount != 0 {
		t.Errorf("AcceptedCount = %d, want 0 for an out-of-vocabulary token", res.AcceptedCount)
	}

	// No probabilities at all.
	res = VerifyStochastic(nil, nil, []int32{0}, rng)
	if res.AcceptedCount != 0 || res.TotalEmitted != 0 || res.BonusToken != -1 {
		t.Errorf("got %+v, want an empty result with BonusToken -1", res)
	}
}

// TestVerifyGreedyNoBonusAvailable covers the emitted count when the caller
// supplies no prediction past the draft: nothing extra is emitted, so
// TotalEmitted must not count a bonus that does not exist.
func TestVerifyGreedyNoBonusAvailable(t *testing.T) {
	res := VerifyGreedy([]int32{10, 20}, []int32{10, 20})
	if res.AcceptedCount != 2 {
		t.Errorf("AcceptedCount = %d, want 2", res.AcceptedCount)
	}
	if res.BonusToken != -1 {
		t.Errorf("BonusToken = %d, want -1", res.BonusToken)
	}
	if res.TotalEmitted != 2 {
		t.Errorf("TotalEmitted = %d, want 2 (no bonus was available)", res.TotalEmitted)
	}
}
