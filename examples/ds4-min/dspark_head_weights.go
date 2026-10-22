package main

import (
	"fmt"
	"math"
)

// headCPU projects the actual support hidden through the base output weight,
// then applies the support's sequential Markov and confidence heads. It never
// calls target inference to manufacture a draft. A zero threshold disables
// confidence rejection, as upstream permits. This CPU reference does not
// qualify the native implementation or implement scheduler acceptance.
func (m *dsparkModel) headCPU(base *dsparkModel, hidden [][]float32, previous, count int, threshold float32) ([]int, []float32, error) {
	if m == nil || base == nil || m.stages < 1 || previous < 0 || previous >= Flash.Vocab || count < 1 || count > m.block || len(hidden) != m.block || threshold < 0 || threshold > 1 || !allFinite([]float32{threshold}) {
		return nil, nil, errShape
	}
	output, ok := base.tensor("output.weight")
	if !ok || len(output.Dimensions) != 2 || output.Dimensions[0] != uint64(Flash.Embed) || output.Dimensions[1] != uint64(Flash.Vocab) {
		return nil, nil, fmt.Errorf("dspark head: base output shape mismatch")
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", m.stages-1, suffix) }
	confidence, err := m.denseRow(name("confidence_head.proj.weight"), 0)
	if err != nil {
		return nil, nil, err
	}
	if len(confidence) != Flash.Embed+m.rank {
		return nil, nil, errShape
	}
	proposal := make([]int, 0, count)
	probabilities := make([]float32, 0, count)
	for row := 0; row < count; row++ {
		if len(hidden[row]) != Flash.Embed || !allFinite(hidden[row]) {
			return nil, nil, errShape
		}
		state, err := m.denseRow(name("markov_head.markov_w1.weight"), previous)
		if err != nil {
			return nil, nil, err
		}
		if len(state) != m.rank {
			return nil, nil, errShape
		}
		logits, err := base.denseMatvec("output.weight", hidden[row])
		if err != nil {
			return nil, nil, err
		}
		bias, err := m.denseMatvec(name("markov_head.markov_w2.weight"), state)
		if err != nil {
			return nil, nil, err
		}
		if len(bias) != Flash.Vocab {
			return nil, nil, errShape
		}
		best, bestValue := 0, float32(math.Inf(-1))
		for token, value := range logits {
			value += bias[token]
			if !allFinite([]float32{value}) {
				return nil, nil, fmt.Errorf("dspark head: nonfinite biased logit")
			}
			if value > bestValue {
				best, bestValue = token, value
			}
		}
		z := dot(confidence[:Flash.Embed], hidden[row]) + dot(confidence[Flash.Embed:], state)
		if !allFinite([]float32{z}) {
			return nil, nil, fmt.Errorf("dspark head: nonfinite confidence")
		}
		probability := sigmoid(z)
		probabilities = append(probabilities, probability)
		if threshold > 0 && probability < threshold {
			break
		}
		proposal = append(proposal, best)
		previous = best
	}
	return proposal, probabilities, nil
}
