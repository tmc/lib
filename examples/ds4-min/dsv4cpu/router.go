package dsv4cpu

import (
	"errors"
	"math"
)

// RouterProbabilities converts router logits to sqrt(softplus(logit)).
func RouterProbabilities(dst, logits []float32) error {
	if len(dst) == 0 || len(dst) != len(logits) {
		return errors.New("invalid router shape")
	}
	for i, x := range logits {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return errors.New("nonfinite router logit")
		}
		var p float32
		if x > 20 {
			p = x
		} else if x < -20 {
			p = float32(math.Exp(float64(x)))
		} else {
			p = float32(math.Log1p(float64(float32(math.Exp(float64(x))))))
		}
		dst[i] = float32(math.Sqrt(float64(p)))
	}
	return nil
}

// SelectExperts ranks with optional correction bias, but normalizes weights
// from unbiased probabilities. Equal scores retain ascending expert order.
func SelectExperts(ids []int, weights, probs, bias []float32, scale float32) error {
	if len(ids) == 0 || len(ids) != len(weights) || len(ids) > len(probs) || (bias != nil && len(bias) != len(probs)) || !(scale > 0) || math.IsInf(float64(scale), 0) {
		return errors.New("invalid expert selection shape")
	}
	for _, p := range probs {
		if !(p >= 0) || math.IsInf(float64(p), 0) {
			return errors.New("invalid expert probability")
		}
	}
	for i, b := range bias {
		if math.IsNaN(float64(b)) || math.IsInf(float64(b), 0) || math.IsInf(float64(probs[i]+b), 0) {
			return errors.New("nonfinite expert correction")
		}
	}
	for i := range ids {
		ids[i] = -1
	}
	score := func(i int) float32 {
		if bias != nil {
			return probs[i] + bias[i]
		}
		return probs[i]
	}
	for i := range probs {
		for j := range ids {
			if ids[j] < 0 || score(i) > score(ids[j]) {
				copy(ids[j+1:], ids[j:len(ids)-1])
				ids[j] = i
				break
			}
		}
	}
	return ExpertWeights(weights, probs, ids, scale)
}

// ExpertWeights computes normalized weights for a supplied hash-routing list.
func ExpertWeights(dst, probs []float32, ids []int, scale float32) error {
	if len(ids) == 0 || len(dst) != len(ids) || !(scale > 0) || math.IsInf(float64(scale), 0) {
		return errors.New("invalid expert weights shape")
	}
	var sum float32
	for _, i := range ids {
		if i < 0 || i >= len(probs) || !(probs[i] >= 0) || math.IsInf(float64(probs[i]), 0) {
			return errors.New("invalid selected expert")
		}
		sum += probs[i]
	}
	if sum < 6.103515625e-5 {
		sum = 6.103515625e-5
	}
	for j, i := range ids {
		dst[j] = float32(probs[i]/sum) * scale
	}
	return nil
}
