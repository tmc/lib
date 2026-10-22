package dsv41

import "fmt"

// Checkpoint is an immutable, in-process checkpoint at a completed token
// boundary. It borrows the original Model and cannot be restored into another
// model or context. It is not an upstream KVC payload or a portable disk codec.
// The zero value is invalid. The borrowed model source must remain alive.
type Checkpoint struct{ state *State }

// Checkpoint copies all mutable request storage. A partially executed or failed
// token cannot be checkpointed. The result remains independent of later Eval,
// Reset and Restore calls on the source request.
func (s *State) Checkpoint() (*Checkpoint, error) {
	if !s.complete() {
		return nil, fmt.Errorf("V4.1 state not at a completed token boundary")
	}
	return &Checkpoint{state: cloneState(s)}, nil
}

// Restore atomically replaces this request's history with a checkpoint from
// the same Model and context. It also recovers a failed request. Invalid
// checkpoints leave the destination unchanged; storage is never shared.
func (s *State) Restore(c *Checkpoint) error {
	if s == nil || s.model == nil || c == nil || !c.state.complete() || c.state.model != s.model || c.state.context != s.context {
		return fmt.Errorf("incompatible V4.1 checkpoint")
	}
	copy := cloneState(c.state)
	*s = *copy
	return nil
}

func (s *State) complete() bool {
	return s != nil && s.model != nil && s.valid && s.engram != nil && s.kv != nil && s.position >= 0 && s.position <= s.context && s.kv.context == s.context && s.kv.position == s.position && s.kv.layer == 0 && len(s.tokens) == s.position && (s.position == 0 && len(s.logits) == 0 || s.position > 0 && len(s.logits) == 129280)
}

func cloneState(s *State) *State {
	dst := *s
	dst.tokens = append([]int32(nil), s.tokens...)
	dst.logits = append([]float32(nil), s.logits...)
	kv := *s.kv
	for i := range kv.raw {
		kv.raw[i] = append([]float32(nil), s.kv.raw[i]...)
	}
	for i := range kv.owners {
		kv.owners[i].kv = append([]float32(nil), s.kv.owners[i].kv...)
		kv.owners[i].index = append([]float32(nil), s.kv.owners[i].index...)
	}
	dst.kv = &kv
	engram := *s.engram
	for i := range engram.rows {
		engram.rows[i] = append([]float32(nil), s.engram.rows[i]...)
	}
	dst.engram = &engram
	for i := range dst.pool {
		dst.pool[i].kv = append([]float32(nil), s.pool[i].kv...)
		dst.pool[i].scores = append([]float32(nil), s.pool[i].scores...)
	}
	dst.candidates.mask = append([]bool(nil), s.candidates.mask...)
	dst.candidates.selected = append([]int(nil), s.candidates.selected...)
	return &dst
}
