package dsv4

import (
	"errors"
	"fmt"
	"math"
)

// ExactRandom supplies the random draws used by ExactVerify. math/rand.Rand
// satisfies this interface; keeping it small also makes seeded replay tests
// independent of a particular random-number implementation.
type ExactRandom interface {
	Float64() float64
}

// ExactVerify verifies one point-mass draft against a probability vector.
// With probability p[draft], it returns the draft and accepted=true. Otherwise
// it returns a replacement sampled from the remaining probability mass and
// accepted=false. The resulting token has exactly the input distribution.
//
// The caller owns the verifier state transition: this function only defines
// the sampling decision, so it is usable by CPU and future device verifiers.
func ExactVerify(probabilities []float64, draft int, random ExactRandom) (token int, accepted bool, err error) {
	if len(probabilities) == 0 {
		return 0, false, errors.New("exact sampling requires probabilities")
	}
	if draft < 0 || draft >= len(probabilities) {
		return 0, false, fmt.Errorf("draft token %d is outside probability vector", draft)
	}
	if random == nil {
		return 0, false, errors.New("exact sampling requires a random source")
	}
	total := 0.0
	for i, p := range probabilities {
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 {
			return 0, false, fmt.Errorf("probability %d is invalid", i)
		}
		total += p
	}
	if total <= 0 || math.IsInf(total, 0) {
		return 0, false, errors.New("probabilities have no finite mass")
	}
	draftMass := probabilities[draft]
	draw, err := exactRandomDraw(random)
	if err != nil {
		return 0, false, err
	}
	if draw < draftMass/total {
		return draft, true, nil
	}
	remaining := total - draftMass
	if remaining <= 0 {
		// The rejection branch is unreachable for a probability-one draft,
		// apart from a caller's out-of-contract random source returning >= 1.
		return draft, true, nil
	}
	target, err := exactRandomDraw(random)
	if err != nil {
		return 0, false, err
	}
	target *= remaining
	for i, p := range probabilities {
		if i == draft {
			continue
		}
		if target < p {
			return i, false, nil
		}
		target -= p
	}
	// Floating-point subtraction can leave a tiny residual at the endpoint.
	for i := len(probabilities) - 1; i >= 0; i-- {
		if i != draft && probabilities[i] > 0 {
			return i, false, nil
		}
	}
	return 0, false, errors.New("replacement sampling found no remaining mass")
}

func exactRandomDraw(random ExactRandom) (float64, error) {
	draw := random.Float64()
	if math.IsNaN(draw) || math.IsInf(draw, 0) || draw < 0 || draw >= 1 {
		return 0, errors.New("exact sampling random draw is outside [0,1)")
	}
	return draw, nil
}
