package main

import "fmt"

// legacyMTPState owns the causal support ring. The target model supplies
// decoded embedding rows and its actual output projection separately.
type legacyMTPState struct {
	model       *legacyMTPModel
	raw         []float32
	count, next int
}

func newLegacyMTPState(m *legacyMTPModel) (*legacyMTPState, error) {
	if m == nil || m.weights == nil {
		return nil, errShape
	}
	return &legacyMTPState{model: m, raw: make([]float32, Flash.SlidingWindow*Flash.HeadDim)}, nil
}

func (s *legacyMTPState) clone() *legacyMTPState {
	if s == nil {
		return nil
	}
	c := *s
	c.raw = append([]float32(nil), s.raw...)
	return &c
}

// step executes actual decoded support attention and MoE weights. Every
// failure leaves the request's support ring unchanged. Float activations
// make this a CPU reference, not a native reduction/quantization receipt.
func (s *legacyMTPState) step(embedding, hidden []float32, position int) (head, nextHidden []float32, err error) {
	if s == nil || s.model == nil || s.model.weights == nil || len(s.raw) != Flash.SlidingWindow*Flash.HeadDim || s.count < 0 || s.count > Flash.SlidingWindow || position < 0 || position != s.next {
		return nil, nil, fmt.Errorf("legacy mtp: position mismatch")
	}
	input, err := s.model.input(embedding, hidden)
	if err != nil {
		return nil, nil, err
	}
	view := *s.model.weights
	view.stages, view.block = 1, 1
	work := &dsparkState{model: &view, capacity: Flash.SlidingWindow, position: position, raw: [][]float32{append([]float32(nil), s.raw...)}}
	// The generic QKV reference excludes its feature row from Q. Supplying
	// the same input twice produces the one legacy query and KV without
	// inventing any feature or storing an extra cache row.
	paired := append(append([]float32(nil), input...), input...)
	q, kv, post, combine, err := work.stageQKV(0, paired, []int{position, position})
	if err != nil {
		return nil, nil, err
	}
	row := position % work.capacity
	copy(work.raw[0][row*Flash.HeadDim:], kv[1])
	count := min(s.count+1, work.capacity)
	start := (row - count + 1 + work.capacity) % work.capacity
	sinks, err := view.denseRow("mtp.0.attn_sinks.weight", 0)
	if err != nil {
		return nil, nil, err
	}
	heads := make([]float32, Flash.Heads*Flash.HeadDim)
	for h := 0; h < Flash.Heads; h++ {
		v, e := work.attend(0, start, count, q[0][h*Flash.HeadDim:(h+1)*Flash.HeadDim], sinks[h])
		if e != nil {
			return nil, nil, e
		}
		if e = dsparkRope(v, Flash.RopeDim, position, true); e != nil {
			return nil, nil, e
		}
		copy(heads[h*Flash.HeadDim:], v)
	}
	groupDim := len(heads) / Flash.OutGroups
	low := make([]float32, Flash.OutGroups*Flash.OutLoRA)
	for group := 0; group < Flash.OutGroups; group++ {
		for rank := 0; rank < Flash.OutLoRA; rank++ {
			w, e := view.denseRow("mtp.0.attn_output_a.weight", group*Flash.OutLoRA+rank)
			if e != nil {
				return nil, nil, e
			}
			low[group*Flash.OutLoRA+rank] = dot(w, heads[group*groupDim:(group+1)*groupDim])
		}
	}
	projected, err := view.denseMatvec("mtp.0.attn_output_b.weight", low)
	if err != nil {
		return nil, nil, err
	}
	after, err := hcPost(projected, input, post[0], combine[0])
	if err != nil {
		return nil, nil, err
	}
	ffn, err := work.stageFFN(0, [][]float32{after})
	if err != nil {
		return nil, nil, err
	}
	head, err = s.model.headInput(ffn[0])
	if err != nil {
		return nil, nil, err
	}
	s.raw, s.count, s.next = work.raw[0], count, position+1
	return head, ffn[0], nil
}
