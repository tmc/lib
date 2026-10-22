//go:build linux

package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Returned views are borrowed from the consumer owner, not independent
// allocations. Callers must not Free them or retain them after owner.Close.
func shapeConsumerView[T any](s *cudaShapeConsumers, name string) (cudamem.Slice[T], error) {
	if s == nil || s.closed {
		return cudamem.Slice[T]{}, errors.New("cuda shape consumers: closed owner")
	}
	if err := s.r.checkOpen(); err != nil {
		return cudamem.Slice[T]{}, err
	}
	v, ok := s.buffers[name]
	if !ok || v.Ptr() == 0 {
		return cudamem.Slice[T]{}, fmt.Errorf("cuda shape consumers: missing buffer %s", name)
	}
	return cudamem.Reinterpret[T](v)
}

func (s *cudaShapeConsumers) attentionQuant() (shapeAttentionQuantBuffers, error) {
	var q shapeAttentionQuantBuffers
	for _, v := range []struct {
		name string
		out  *cudamem.Slice[byte]
	}{{"attention-input", &q.input}, {"attention-qr", &q.qr}, {"attention-heads", &q.heads}, {"attention-low", &q.low}} {
		b, err := shapeConsumerView[byte](s, v.name)
		if err != nil {
			return shapeAttentionQuantBuffers{}, err
		}
		*v.out = b
	}
	return q, checkShapeAttentionQuantExtents(s.config, q)
}

func (s *cudaShapeConsumers) ffnBuffers() (shapeFFNBuffers, error) {
	var b shapeFFNBuffers
	for _, v := range []struct {
		name string
		out  *cudamem.Slice[float32]
	}{{"router-logits", &b.logits}, {"router-probs", &b.probs}, {"router-weights", &b.weights}, {"expert-gate", &b.gate}, {"expert-up", &b.up}, {"expert-down", &b.down}} {
		x, err := shapeConsumerView[float32](s, v.name)
		if err != nil {
			return shapeFFNBuffers{}, err
		}
		*v.out = x
	}
	for _, v := range []struct {
		name string
		out  *cudamem.Slice[byte]
	}{{"ffn-input", &b.inputQ81}, {"ffn-experts", &b.expertsQ81}, {"shared-mid", &b.sharedQ80}} {
		x, err := shapeConsumerView[byte](s, v.name)
		if err != nil {
			return shapeFFNBuffers{}, err
		}
		*v.out = x
	}
	var err error
	b.ids, err = shapeConsumerView[int32](s, "router-ids")
	if err != nil {
		return shapeFFNBuffers{}, err
	}
	b.invalid, err = shapeConsumerView[uint32](s, "router-invalid")
	return b, err
}
