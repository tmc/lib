package main

import "fmt"

// dsparkState is the drafter's private request state, not a target verifier.
// Target State snapshot integration must include this value when connected.
type dsparkState struct {
	model    *dsparkModel
	capacity int
	position int
	features []float32
	captured []bool
	raw      [][]float32
}

func newDSparkState(model *dsparkModel, capacity int) (*dsparkState, error) {
	if model == nil || model.stages < 3 || len(model.targets) == 0 || capacity < model.block+1 || capacity > 1<<20 {
		return nil, fmt.Errorf("dspark state: invalid capacity or model")
	}
	s := &dsparkState{model: model, capacity: capacity, position: -1, features: make([]float32, len(model.targets)*Flash.Embed), captured: make([]bool, len(model.targets)), raw: make([][]float32, model.stages)}
	for i := range s.raw {
		s.raw[i] = make([]float32, capacity*Flash.HeadDim)
	}
	return s, nil
}

func (s *dsparkState) clone() *dsparkState {
	if s == nil {
		return nil
	}
	c := *s
	c.features = append([]float32(nil), s.features...)
	c.captured = append([]bool(nil), s.captured...)
	c.raw = make([][]float32, len(s.raw))
	for i, a := range s.raw {
		c.raw[i] = append([]float32(nil), a...)
	}
	return &c
}

// beginCapture starts one position's selected post-FFN feature set. The target
// backend calls capture after its pending HC expansion is flushed, matching
// ds4.c:29222-29228. Mixing positions cannot yield a complete feature set.
func (s *dsparkState) beginCapture(position int) error {
	if s == nil || position < 0 || position > 2147483647-s.model.block {
		return fmt.Errorf("dspark capture: invalid position")
	}
	s.position = position
	clear(s.captured)
	clear(s.features)
	return nil
}
func (s *dsparkState) capture(layer, position int, hc []float32) error {
	if s == nil || position != s.position {
		return fmt.Errorf("dspark capture: position mismatch")
	}
	for slot, selected := range s.model.targets {
		if layer == selected {
			mean, err := dsparkHCMean(hc, Flash.Embed, Flash.HC)
			if err != nil {
				return err
			}
			copy(s.features[slot*Flash.Embed:], mean)
			s.captured[slot] = true
			return nil
		}
	}
	return nil
}

// stageInput binds the admitted stage-zero projection and norm to the captured
// features. Embeddings are actual decoded target-model rows for token followed
// by noise; no target predictions stand in for the support transformer.
func (s *dsparkState) stageInput(embeddings [][]float32) ([]float32, []int, error) {
	if s == nil || s.position < 0 || len(embeddings) != s.model.block {
		return nil, nil, fmt.Errorf("dspark input: incomplete block")
	}
	for _, ok := range s.captured {
		if !ok {
			return nil, nil, fmt.Errorf("dspark input: missing target feature")
		}
	}
	for _, e := range embeddings {
		if len(e) != Flash.Embed || !allFinite(e) {
			return nil, nil, fmt.Errorf("dspark input: invalid embedding")
		}
	}
	projection := make([]float32, 0, Flash.Embed*len(s.features))
	for row := 0; row < Flash.Embed; row++ {
		a, err := s.model.denseRow("mtp.0.main_proj.weight", row)
		if err != nil {
			return nil, nil, err
		}
		projection = append(projection, a...)
	}
	norm, err := s.model.denseRow("mtp.0.main_norm.weight", 0)
	if err != nil {
		return nil, nil, err
	}
	main, err := dsparkMainReference(s.features, projection, norm, Flash.Embed, Flash.RMSEps)
	if err != nil {
		return nil, nil, err
	}
	hcWidth := Flash.Embed * Flash.HC
	input := make([]float32, (s.model.block+1)*hcWidth)
	positions := make([]int, s.model.block+1)
	positions[0] = s.position
	for lane := 0; lane < Flash.HC; lane++ {
		copy(input[lane*Flash.Embed:], main)
	}
	for row, e := range embeddings {
		positions[row+1] = s.position + row
		for lane := 0; lane < Flash.HC; lane++ {
			copy(input[(row+1)*hcWidth+lane*Flash.Embed:], e)
		}
	}
	return input, positions, nil
}

// writeRaw stores post-FP8, half-rounded rows in the support-only ring.
// Validation precedes writes; quantization is performed by the stage forward,
// not inferred from the pointer or silently omitted here.
func (s *dsparkState) writeRaw(stage, start int, rows []float32) error {
	if s == nil || stage < 0 || stage >= len(s.raw) || start < 0 || start >= s.capacity || len(rows) == 0 || len(rows)%Flash.HeadDim != 0 || len(rows)/Flash.HeadDim > s.capacity || !allFinite(rows) {
		return fmt.Errorf("dspark raw: invalid rows")
	}
	for row := 0; row < len(rows)/Flash.HeadDim; row++ {
		copy(s.raw[stage][((start+row)%s.capacity)*Flash.HeadDim:], rows[row*Flash.HeadDim:(row+1)*Flash.HeadDim])
	}
	return nil
}
