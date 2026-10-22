package main

import "fmt"

// proposalCPU is the reference producer: real base embedding rows feed the
// support transformer, whose hidden feeds the real base output and support
// Markov/confidence weights. The target backend must supply the selected HC
// capture and a qualified trusted support-cache window. Native dispatch and
// accepted-prefix cache ownership are intentionally not hidden in this API.
func (s *dsparkState) proposalCPU(base *dsparkModel, token, position, count int, threshold float32, appendStart, visibleStart, visibleRows int) (proposal []int, probabilities []float32, err error) {
	if s == nil || base == nil || token < 0 || token >= Flash.Vocab || position != s.position || count < 1 || count > s.model.block {
		return nil, nil, errShape
	}
	embedding, ok := base.tensor("token_embd.weight")
	if !ok || len(embedding.Dimensions) != 2 || embedding.Dimensions[0] != uint64(Flash.Embed) || embedding.Dimensions[1] != uint64(Flash.Vocab) {
		return nil, nil, fmt.Errorf("dspark proposal: base embedding shape mismatch")
	}
	embeddings := make([][]float32, s.model.block)
	for row := range embeddings {
		id := s.model.noise
		if row == 0 {
			id = token
		}
		embeddings[row], err = base.denseRow("token_embd.weight", id)
		if err != nil {
			return nil, nil, err
		}
	}
	before := s.clone()
	defer func() {
		if err != nil {
			for stage := range s.raw {
				copy(s.raw[stage], before.raw[stage])
			}
		}
	}()
	hidden, err := s.forwardCPU(embeddings, appendStart, visibleStart, visibleRows)
	if err != nil {
		return nil, nil, err
	}
	return s.model.headCPU(base, hidden, token, count, threshold)
}
