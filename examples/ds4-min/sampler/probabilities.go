package sampler

import (
	"errors"
	"sort"
)

type probabilityCandidate struct {
	candidate
	prob float32
}

// Probabilities builds the filtered distribution for the upstream exact
// speculative verifier (11e8cad), in vocabulary order. It draws no randomness
// and returns a detached slice. Its heap/filter float32 summation follows
// sample_build_probabilities; rounding can differ from Sample's direct draw
// path. Greedy or degenerate distributions are point masses. Nonfinite
// parameters or logits are errors. It does not read or modify sampler scratch.
func (s *Sampler) Probabilities(logits []float32, p Params) ([]float32, error) {
	if s == nil || len(logits) == 0 {
		return nil, errors.New("sampling probabilities require logits and sampler")
	}
	if !isFinite(p.Temperature) || !isFinite(p.TopP) || !isFinite(p.MinP) {
		return nil, errors.New("sampling probabilities require finite parameters")
	}
	for _, v := range logits {
		if !isFinite(v) {
			return nil, errors.New("sampling probabilities require finite logits")
		}
	}
	out := make([]float32, len(logits))
	point := func() ([]float32, error) { out[Argmax(logits)] = 1; return out, nil }
	if p.Temperature <= 0 {
		return point()
	}
	topP := p.TopP
	if topP <= 0 || topP > 1 {
		topP = 1
	}
	minP := max(p.MinP, 0)
	topK := min(max(p.TopK, 0), 1024, len(logits))
	capacity := 512
	if topK > 0 {
		capacity = topK
	}
	heap := make([]probabilityCandidate, 0, min(capacity, len(logits)))
	maximum := negInf
	for _, v := range logits {
		maximum = max(maximum, v)
	}
	var fullSum, heapSum float32
	scale := newTemperatureScale(p.Temperature)
	for id, v := range logits {
		raw := Exp(scale.apply(float32(v - maximum)))
		fullSum += raw
		c := probabilityCandidate{candidate{id, v}, raw}
		if len(heap) < capacity {
			heap = append(heap, c)
			heapSum += raw
			probabilityHeapUp(heap, len(heap)-1)
		} else if better(c.candidate, heap[0].candidate) {
			heapSum -= heap[0].prob
			heap[0] = c
			heapSum += raw
			probabilityHeapDown(heap, 0)
		}
	}
	if !(fullSum > 0) || !isFinite(fullSum) || len(heap) == 0 {
		return point()
	}
	sort.Slice(heap, func(i, j int) bool { return better(heap[i].candidate, heap[j].candidate) })
	base := fullSum
	if topK > 0 {
		base = heapSum
	}
	end, filtered, stopped, reached := probabilityFilter(heap, base, topP, minP)
	complete := topK > 0 || len(heap) == len(logits) || reached || stopped && heap[len(heap)-1].prob < heap[0].prob*minP
	if !complete {
		heap = make([]probabilityCandidate, len(logits))
		for id, v := range logits {
			heap[id] = probabilityCandidate{candidate{id, v}, Exp(scale.apply(float32(v - maximum)))}
		}
		sort.Slice(heap, func(i, j int) bool { return better(heap[i].candidate, heap[j].candidate) })
		end, filtered, _, _ = probabilityFilter(heap, fullSum, topP, minP)
	}
	if end == 0 || !(filtered > 0) || !isFinite(filtered) {
		out[heap[0].id] = 1
		return out, nil
	}
	inverse := float32(1 / filtered)
	for _, c := range heap[:end] {
		out[c.id] = c.prob * inverse
	}
	return out, nil
}

func probabilityHeapUp(heap []probabilityCandidate, i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !better(heap[parent].candidate, heap[i].candidate) {
			return
		}
		heap[parent], heap[i] = heap[i], heap[parent]
		i = parent
	}
}

func probabilityHeapDown(heap []probabilityCandidate, i int) {
	for {
		left, right, smallest := i*2+1, i*2+2, i
		if left < len(heap) && better(heap[smallest].candidate, heap[left].candidate) {
			smallest = left
		}
		if right < len(heap) && better(heap[smallest].candidate, heap[right].candidate) {
			smallest = right
		}
		if smallest == i {
			return
		}
		heap[smallest], heap[i] = heap[i], heap[smallest]
		i = smallest
	}
}

func probabilityFilter(candidates []probabilityCandidate, base, topP, minP float32) (end int, sum float32, stopped, reached bool) {
	minimum := candidates[0].prob * minP
	for i, c := range candidates {
		if i > 0 && c.prob < minimum {
			stopped = true
			break
		}
		sum += c.prob
		end++
		if sum/base >= topP {
			reached = true
			break
		}
	}
	return
}
