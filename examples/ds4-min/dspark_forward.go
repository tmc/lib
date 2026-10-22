package main

import "fmt"

// forwardCPU evaluates the admitted support transformer. The feature row is
// fixed across stages; only draft rows are replaced (ds4.c:33999). Explicit
// ring-window arguments keep this reference separate from the scheduler's
// accepted-prefix bookkeeping and from the target model's State.
//
// This is a CPU value-reference path, not CUDA/Metal production integration.
// It does not implement target-logit projection or draft acceptance.
func (s *dsparkState) forwardCPU(embeddings [][]float32, appendStart, visibleStart, visibleRows int) (hidden [][]float32, err error) {
	input, positions, err := s.stageInput(embeddings)
	if err != nil {
		return nil, err
	}
	before := s.clone()
	defer func() {
		if err != nil {
			for stage := range s.raw {
				copy(s.raw[stage], before.raw[stage])
			}
		}
	}()
	width := Flash.Embed * Flash.HC
	var rows [][]float32
	for stage := 0; stage < s.model.stages; stage++ {
		rows, err = s.stageAttention(stage, appendStart, visibleStart, visibleRows, input, positions)
		if err != nil {
			return nil, fmt.Errorf("dspark stage %d attention: %w", stage, err)
		}
		rows, err = s.stageFFN(stage, rows)
		if err != nil {
			return nil, fmt.Errorf("dspark stage %d FFN: %w", stage, err)
		}
		for row, hc := range rows {
			copy(input[(row+1)*width:(row+2)*width], hc)
		}
	}
	return s.model.finalHidden(rows)
}

// finalHidden follows ds4.c:34800-34912: RMS over flat HC, learned lane
// weights, weighted lane sum, then learned embedding RMS.
func (m *dsparkModel) finalHidden(input [][]float32) ([][]float32, error) {
	if m == nil || len(input) != m.block || m.stages < 1 {
		return nil, errShape
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", m.stages-1, suffix) }
	base, err := m.denseRow(name("hc_head_base.weight"), 0)
	if err != nil {
		return nil, err
	}
	scale, err := m.denseRow(name("hc_head_scale.weight"), 0)
	if err != nil {
		return nil, err
	}
	norm, err := m.denseRow(name("norm.weight"), 0)
	if err != nil {
		return nil, err
	}
	ones := make([]float32, Flash.HC*Flash.Embed)
	for i := range ones {
		ones[i] = 1
	}
	out := make([][]float32, len(input))
	for row, hc := range input {
		flat, err := rmsNorm(hc, ones, Flash.RMSEps)
		if err != nil {
			return nil, err
		}
		mix, err := m.denseMatvec(name("hc_head_fn.weight"), flat)
		if err != nil {
			return nil, err
		}
		weights := make([]float32, Flash.HC)
		for lane := range weights {
			weights[lane] = sigmoid(mix[lane]*scale[0]+base[lane]) + Flash.HCEps
		}
		value, err := hcWeightedSum(hc, weights, Flash.Embed)
		if err != nil {
			return nil, err
		}
		out[row], err = rmsNorm(value, norm, Flash.RMSEps)
		if err != nil || !allFinite(out[row]) {
			return nil, fmt.Errorf("dspark final hidden: invalid RMS output")
		}
	}
	return out, nil
}
