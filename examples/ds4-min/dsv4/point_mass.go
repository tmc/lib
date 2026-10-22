package dsv4

import (
	"errors"
	"fmt"
	"math"
)

// PointMassRandom supplies float32 draws for PointMassVerify. For upstream
// seeded replay, use sampler.RNG's 24-bit stream; matching this interface alone
// does not make another random source's seeds equivalent.
type PointMassRandom interface {
	Float32() float32
}

// PointMassVerify implements the 11e8cad point-mass proposal protocol for an
// already filtered, normalized float32 probability vector. Unlike ExactVerify,
// it does not renormalize acceptance mass. Probability-zero and probability-one
// acceptance consume no draw; interior acceptance includes equality. Rejection
// masks the draft and walks the remaining mass with float32 arithmetic and
// inclusive boundaries, falling back to the lowest-index maximum.
//
// The input is not modified. Malformed extents and nonfinite or negative
// probabilities return errors without consuming randomness. The caller
// owns state transitions and must retain or roll back the random frontier.
func PointMassVerify(probabilities []float32, draft int, random PointMassRandom) (token int, accepted bool, err error) {
	if len(probabilities) == 0 {
		return 0, false, errors.New("point-mass sampling requires probabilities")
	}
	if draft < 0 || draft >= len(probabilities) {
		return 0, false, fmt.Errorf("draft token %d is outside probability vector", draft)
	}
	if random == nil {
		return 0, false, errors.New("point-mass sampling requires a random source")
	}
	for i, p := range probabilities {
		if p < 0 || !pointMassFinite(p) {
			return 0, false, fmt.Errorf("probability %d is invalid", i)
		}
	}
	p := probabilities[draft]
	if p >= 1 {
		return draft, true, nil
	}
	if p > 0 {
		draw, err := pointMassDraw(random)
		if err != nil {
			return 0, false, err
		}
		if draw <= p {
			return draft, true, nil
		}
	}
	token, err = pointMassReplacement(probabilities, draft, random)
	return token, false, err
}

// This is sample_probabilities with the point-mass draft masked, without
// modifying the caller's vector. Both accumulation and scan stay float32.
func pointMassReplacement(probabilities []float32, draft int, random PointMassRandom) (int, error) {
	var sum float32
	best, bestP := 0, probabilities[0]
	if draft == 0 {
		bestP = 0
	}
	for i, p := range probabilities {
		if i == draft {
			p = 0
		}
		if p > 0 && pointMassFinite(p) {
			sum = float32(sum + p)
		}
		if p > bestP {
			best, bestP = i, p
		}
	}
	if sum <= 0 || !pointMassFinite(sum) {
		return best, nil
	}
	draw, err := pointMassDraw(random)
	if err != nil {
		return 0, err
	}
	r := float32(draw * sum)
	for i, p := range probabilities {
		if i == draft || p <= 0 || !pointMassFinite(p) {
			continue
		}
		r = float32(r - p)
		if r <= 0 {
			return i, nil
		}
	}
	return best, nil
}

func pointMassDraw(random PointMassRandom) (float32, error) {
	draw := random.Float32()
	if !pointMassFinite(draw) || draw < 0 || draw >= 1 {
		return 0, errors.New("point-mass random draw is outside [0,1)")
	}
	return draw, nil
}

func pointMassFinite(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}
