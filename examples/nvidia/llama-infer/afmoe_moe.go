package main

import (
	"fmt"
	"math"
	"sort"
)

// afmoeRoute applies sigmoid routing to post-linear logits. BF16 logits and
// biases are passed as their exact float32 values. Bias changes selection only;
// selected weights come from the original scores.
//
// Equal selection scores use increasing expert IDs. This scalar convention is
// deterministic, but does not specify the tie ordering of torch.topk.
func afmoeRoute(logits, bias []float32, k int, normalize bool, eps, scale float32) ([]int, []float32, error) {
	if len(logits) == 0 || k < 1 || k > len(logits) {
		return nil, nil, fmt.Errorf("afmoe route: invalid top-k %d for %d experts", k, len(logits))
	}
	if bias != nil && len(bias) != len(logits) {
		return nil, nil, fmt.Errorf("afmoe route: %d biases for %d experts", len(bias), len(logits))
	}
	if !afmoeFinite(eps) || eps < 0 || !afmoeFinite(scale) || scale <= 0 {
		return nil, nil, fmt.Errorf("afmoe route: invalid epsilon or scale")
	}
	scores := make([]float32, len(logits))
	selection := make([]float32, len(logits))
	ids := make([]int, len(logits))
	for i, x := range logits {
		if !afmoeFinite(x) || bias != nil && !afmoeFinite(bias[i]) {
			return nil, nil, fmt.Errorf("afmoe route: non-finite expert %d", i)
		}
		scores[i] = float32(1 / (1 + math.Exp(-float64(x))))
		selection[i] = scores[i]
		if bias != nil {
			selection[i] += bias[i]
		}
		if !afmoeFinite(selection[i]) {
			return nil, nil, fmt.Errorf("afmoe route: non-finite selection score %d", i)
		}
		ids[i] = i
	}
	sort.Slice(ids, func(i, j int) bool {
		a, b := ids[i], ids[j]
		if selection[a] == selection[b] {
			return a < b
		}
		return selection[a] > selection[b]
	})
	ids = ids[:k:k]
	weights := make([]float32, k)
	var sum float32
	for i, id := range ids {
		weights[i] = scores[id]
		sum += scores[id]
	}
	if normalize {
		denominator := sum + eps
		if !afmoeFinite(denominator) || denominator <= 0 {
			return nil, nil, fmt.Errorf("afmoe route: invalid normalization denominator")
		}
		for i := range weights {
			weights[i] /= denominator
		}
	}
	for i := range weights {
		weights[i] *= scale
	}
	return ids, weights, nil
}

// afmoeWeightedExperts rounds each float32 weighted expert contribution to
// BF16 before accumulation. outputs and weights are in selected-route order.
func afmoeWeightedExperts(outputs [][]float32, weights []float32) ([][]float32, error) {
	if len(outputs) == 0 || len(outputs) != len(weights) {
		return nil, fmt.Errorf("afmoe weight experts: mismatched outputs and weights")
	}
	n := len(outputs[0])
	if n == 0 {
		return nil, fmt.Errorf("afmoe weight experts: empty output")
	}
	result := make([][]float32, len(outputs))
	for i, output := range outputs {
		if len(output) != n || !afmoeFinite(weights[i]) {
			return nil, fmt.Errorf("afmoe weight experts: invalid route %d", i)
		}
		result[i] = make([]float32, n)
		for j, x := range output {
			product := x * weights[i]
			if !afmoeFinite(product) {
				return nil, fmt.Errorf("afmoe weight experts: non-finite route %d element %d", i, j)
			}
			result[i][j] = afmoeBF16(product)
		}
	}
	return result, nil
}

// afmoeCombineExperts models two explicit accumulation policies, both in
// increasing expert-ID order. When wide is false every addition rounds to
// BF16; when true the float32 sum rounds only at the end. The official model's
// scatter_add does not promise either policy on every backend.
func afmoeCombineExperts(shared []float32, weighted [][]float32, ids []int, wide bool) ([]float32, error) {
	if len(shared) == 0 || len(weighted) == 0 || len(weighted) != len(ids) {
		return nil, fmt.Errorf("afmoe combine experts: mismatched shapes")
	}
	order := make([]int, len(ids))
	seen := make(map[int]bool, len(ids))
	for i, id := range ids {
		if id < 0 || seen[id] || len(weighted[i]) != len(shared) {
			return nil, fmt.Errorf("afmoe combine experts: invalid route %d", i)
		}
		seen[id] = true
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return ids[order[i]] < ids[order[j]] })
	result := append([]float32(nil), shared...)
	for j, x := range result {
		if !afmoeFinite(x) {
			return nil, fmt.Errorf("afmoe combine experts: non-finite shared element %d", j)
		}
	}
	for _, i := range order {
		for j, x := range weighted[i] {
			sum := result[j] + x
			if !afmoeFinite(sum) {
				return nil, fmt.Errorf("afmoe combine experts: non-finite route %d element %d", i, j)
			}
			if !wide {
				sum = afmoeBF16(sum)
			}
			result[j] = sum
		}
	}
	if wide {
		for j := range result {
			result[j] = afmoeBF16(result[j])
		}
	}
	return result, nil
}
