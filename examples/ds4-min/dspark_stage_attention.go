package main

import "fmt"

// stageAttention evaluates support attention on all visible rows, then the
// grouped A/B output projections and HC residual. Cache coordinates are
// explicit: the feature row precedes the draft rows in the support ring,
// although its RoPE position equals the first draft's position. The caller
// owns accepted-prefix truncation; this is not a target-State commit.
//
// This CPU reference decodes admitted weights and uses float activations.
// It does not reproduce CUDA's route-dependent activation quantization or
// reduction order and must pass the physical numerical gate before use.
func (s *dsparkState) stageAttention(stage, appendStart, visibleStart, visibleRows int, input []float32, positions []int) (output [][]float32, err error) {
	if s == nil || stage < 0 || stage >= len(s.raw) || appendStart < 0 || appendStart >= s.capacity || visibleStart < 0 || visibleStart >= s.capacity || visibleRows < s.model.block+1 || visibleRows > s.capacity {
		return nil, errShape
	}
	if (appendStart-visibleStart+s.capacity)%s.capacity+s.model.block+1 != visibleRows {
		return nil, fmt.Errorf("dspark attention: appended rows must end visible window")
	}
	queries, kv, post, combine, err := s.stageQKV(stage, input, positions)
	if err != nil {
		return nil, err
	}
	// A failed stage cannot leave a partially updated private support cache.
	previous := append([]float32(nil), s.raw[stage]...)
	defer func() {
		if err != nil {
			copy(s.raw[stage], previous)
		}
	}()
	flatKV := make([]float32, 0, len(kv)*Flash.HeadDim)
	for _, row := range kv {
		flatKV = append(flatKV, row...)
	}
	if err = s.writeRaw(stage, appendStart, flatKV); err != nil {
		return nil, err
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", stage, suffix) }
	sinks, err := s.model.denseRow(name("attn_sinks.weight"), 0)
	if err != nil {
		return nil, err
	}
	width := Flash.Embed * Flash.HC
	groupDim := Flash.Heads * Flash.HeadDim / Flash.OutGroups
	for row, query := range queries {
		heads := make([]float32, len(query))
		for head := 0; head < Flash.Heads; head++ {
			value, e := s.attend(stage, visibleStart, visibleRows, query[head*Flash.HeadDim:(head+1)*Flash.HeadDim], sinks[head])
			if e != nil {
				return nil, e
			}
			if e := dsparkRope(value, Flash.RopeDim, positions[row+1], true); e != nil {
				return nil, e
			}
			copy(heads[head*Flash.HeadDim:], value)
		}
		low := make([]float32, Flash.OutGroups*Flash.OutLoRA)
		for group := 0; group < Flash.OutGroups; group++ {
			values := heads[group*groupDim : (group+1)*groupDim]
			for rank := 0; rank < Flash.OutLoRA; rank++ {
				weights, e := s.model.denseRow(name("attn_output_a.weight"), group*Flash.OutLoRA+rank)
				if e != nil {
					return nil, e
				}
				low[group*Flash.OutLoRA+rank] = dot(weights, values)
			}
		}
		projection, e := s.model.denseMatvec(name("attn_output_b.weight"), low)
		if e != nil {
			return nil, e
		}
		residual := input[(row+1)*width : (row+2)*width]
		hc, e := hcPost(projection, residual, post[row], combine[row])
		if e != nil {
			return nil, e
		}
		if !allFinite(hc) {
			return nil, fmt.Errorf("dspark attention: nonfinite HC output")
		}
		output = append(output, hc)
	}
	return output, nil
}
