package dsv41

import (
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv41engram"
	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
)

// EngramWeights borrows immutable table and projection views and gate weights.
// Projection maps 24*256 decoded features to five width-element rows.
type EngramWeights struct {
	Table      *dsv41engram.Table
	Projection *dsv4cpu.Matrix
	Query, Key []float32
}

// EngramState owns one request's hash history and decoded token rows. It must
// not be shared between requests. Weights and Layout are borrowed immutable
// resources; their readers must outlive this state.
type EngramState struct {
	layout      *dsv41engram.Layout
	weights     [2]EngramWeights
	history     dsv41engram.History
	rows        [2][]float32
	text, ready bool
}

// NewEngramState binds the two Engram insertion sites. The zero value is not
// usable. No allocation is proportional to the complete SSD tables.
func NewEngramState(layout *dsv41engram.Layout, weights [2]EngramWeights) (*EngramState, error) {
	if layout == nil {
		return nil, fmt.Errorf("missing engram layout")
	}
	for _, w := range weights {
		if w.Table == nil || w.Projection == nil || len(w.Query) == 0 || len(w.Query) != len(w.Key) || len(w.Query)%4 != 0 {
			return nil, fmt.Errorf("invalid engram weights")
		}
	}
	s := &EngramState{layout: layout, weights: weights}
	for i := range s.rows {
		s.rows[i] = make([]float32, dsv41engram.Columns*dsv41engram.Dim)
	}
	return s, nil
}

// Reset restarts the token history and discards the prepared token.
func (s *EngramState) Reset() { s.history.Reset(); s.ready = false }

// Prepare hashes one token and reads its two sets of table rows. On failure
// no history is committed, and Apply refuses the incomplete preparation.
func (s *EngramState) Prepare(token int32, text bool) error {
	s.ready = false
	h := s.history
	var ids [2 * dsv41engram.Columns]uint32
	mask := byte(0)
	if text {
		mask = 1
	}
	if err := s.layout.Hash(&h, []int32{token}, []byte{mask}, ids[:]); err != nil {
		return err
	}
	if text {
		for i, w := range s.weights {
			if err := w.Table.ReadRows(ids[i*dsv41engram.Columns:(i+1)*dsv41engram.Columns], s.rows[i]); err != nil {
				return fmt.Errorf("read engram table %d: %w", i, err)
			}
		}
	}
	s.history = h
	s.text = text
	s.ready = true
	return nil
}

// Apply executes the source's pre-attention Engram projection and residual
// gate at layers 1 and 14. Other layers and image tokens leave residual alone.
func (s *EngramState) Apply(layer int, residual []float32) error {
	l, err := Schedule(layer)
	if err != nil {
		return err
	}
	if !s.ready {
		return fmt.Errorf("engram token not prepared")
	}
	if l.Engram < 0 || !s.text {
		return nil
	}
	w := s.weights[l.Engram]
	width := len(w.Query) / 4
	if len(residual) != 4*width {
		return fmt.Errorf("invalid engram residual")
	}
	x := append([]float32(nil), s.rows[l.Engram]...)
	for i := range x {
		x[i] = BF16(x[i])
	}
	kv := make([]float32, 5*width)
	if err := w.Projection.Mul(kv, x); err != nil {
		return fmt.Errorf("project engram: %w", err)
	}
	// ds41_matmul(..., true) rounds the projection output before the gate.
	for i := range kv {
		kv[i] = BF16(kv[i])
	}
	return AddEngram(residual, kv, w.Query, w.Key, width, 1e-20)
}
