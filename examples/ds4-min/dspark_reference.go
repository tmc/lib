package main

import (
	"fmt"
	"math"
)

// dsparkMainReference projects concatenated selected-layer HC means and RMS
// normalizes them, the support stage-zero feature input (ds4.c:33181).
func dsparkMainReference(features, projection, norm []float32, width int, epsilon float32) ([]float32, error) {
	if width < 1 || len(features) == 0 || len(norm) != width || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) || epsilon <= 0 {
		return nil, errShape
	}
	for _, a := range [][]float32{features, projection, norm} {
		for _, v := range a {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("dspark main: nonfinite input")
			}
		}
	}
	x, err := matvec(projection, features, width, len(features))
	if err != nil {
		return nil, err
	}
	return rmsNorm(x, norm, epsilon)
}

// dsparkHeadReference evaluates actual support-head weights after the support
// transformer produces its hidden rows and unmodified logits. W1 and W2 are
// [vocab,rank] GGUF row-major matrices; confidence is [hidden+rank]. This
// implements ds4.c:35259-35317 and 35328-35379 at 11e8cad, not the missing
// support transformer. No target-model prediction is used as a draft.
func dsparkHeadReference(logits, hidden, w1, w2, confidence []float32, rows, vocab, rank, width, previous int, threshold float32) ([]int, []float32, error) {
	if rows < 1 || rows > 16 || vocab < 1 || rank < 1 || width < 1 || previous < 0 || previous >= vocab || math.IsNaN(float64(threshold)) || threshold < 0 || threshold > 1 {
		return nil, nil, errShape
	}
	if len(logits)/rows != vocab || len(logits)%rows != 0 || len(hidden)/rows != width || len(hidden)%rows != 0 || len(w1)/vocab != rank || len(w1)%vocab != 0 || len(w2) != len(w1) || len(confidence) != width+rank {
		return nil, nil, errShape
	}
	for _, a := range [][]float32{logits, hidden, w1, w2, confidence} {
		for _, x := range a {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return nil, nil, fmt.Errorf("dspark head: nonfinite input")
			}
		}
	}
	proposal := make([]int, 0, rows)
	probabilities := make([]float32, 0, rows)
	for row := 0; row < rows; row++ {
		state := w1[previous*rank : (previous+1)*rank]
		best, bestValue := 0, float32(math.Inf(-1))
		for token := 0; token < vocab; token++ {
			bias := float32(0)
			for k := 0; k < rank; k++ {
				bias += w2[token*rank+k] * state[k]
			}
			value := logits[row*vocab+token] + bias
			if math.IsInf(float64(value), 0) || math.IsNaN(float64(value)) {
				return nil, nil, fmt.Errorf("dspark head: nonfinite bias")
			}
			if value > bestValue {
				best, bestValue = token, value
			}
		}
		z := float32(0)
		for k := 0; k < width; k++ {
			z += confidence[k] * hidden[row*width+k]
		}
		for k := 0; k < rank; k++ {
			z += confidence[width+k] * state[k]
		}
		if math.IsNaN(float64(z)) || math.IsInf(float64(z), 0) {
			return nil, nil, fmt.Errorf("dspark head: nonfinite confidence")
		}
		p := float32(1 / (1 + math.Exp(-float64(z))))
		probabilities = append(probabilities, p)
		if p < threshold {
			break
		}
		proposal = append(proposal, best)
		previous = best
	}
	return proposal, probabilities, nil
}

// dsparkHCMean copies the post-FFN HC state into the support feature layout.
// Upstream assigns every HC lane weight 1/N_HC (ds4.c:17300), then captures
// selected layers after FFN. The input layout is [lane,embedding], as read
// by ds4_cuda.cu:11813 hc_weighted_sum_kernel.
func dsparkHCMean(state []float32, width, lanes int) ([]float32, error) {
	if width < 1 || lanes < 1 || len(state)/width != lanes || len(state)%width != 0 {
		return nil, errShape
	}
	mean := make([]float32, width)
	weight := float32(1) / float32(lanes)
	for i := range mean {
		for lane := 0; lane < lanes; lane++ {
			x := state[lane*width+i]
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return nil, fmt.Errorf("dspark capture: nonfinite HC value")
			}
			mean[i] += x * weight
		}
	}
	return mean, nil
}
