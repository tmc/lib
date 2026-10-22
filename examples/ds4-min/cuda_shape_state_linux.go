//go:build linux

package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// newCUDAShapeState allocates one independent serial decode State. The caller
// owns the uploaded immutable weights and foundation until cleanup completes.
// This private constructor does not admit the family through public Supports.
func newCUDAShapeState(r *cudaFoundation, c dsv4shape.Config, binding *dsv4shape.Binding, norms map[string]cudamem.Slice[float32], hcWeights map[string]shapeHCWeights, weights shapeDecodeWeights, context, reserve uint64) (*cudaShapeDecode, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if err := validateShapeRopeMetadata(r.gguf); err != nil {
		return nil, err
	}
	_, bytes, err := planShapeConsumerBuffers(c, context)
	if err != nil {
		return nil, err
	}
	forwardReserve, err := shapeStateReserve(reserve, bytes)
	if err != nil {
		return nil, err
	}
	return newCUDAShapeStateWithOperations(binding, weights, shapeStateOperations{
		forward: func() (*cudaShapeForward, error) {
			return newCUDAShapeForward(r, c, binding, norms, context, forwardReserve)
		},
		consumers: func() (*cudaShapeConsumers, error) {
			return newCUDAShapeConsumers(r, c, context, reserve+192)
		},
		hc:  func() (*cudaShapeHC, error) { return newCUDAShapeHC(r, c) },
		mix: func() (*cudaShapeHCMix, error) { return newCUDAShapeHCMix(r, c, binding, hcWeights) },
	})
}

type shapeStateOperations struct {
	forward   func() (*cudaShapeForward, error)
	consumers func() (*cudaShapeConsumers, error)
	hc        func() (*cudaShapeHC, error)
	mix       func() (*cudaShapeHCMix, error)
}

func newCUDAShapeStateWithOperations(binding *dsv4shape.Binding, weights shapeDecodeWeights, ops shapeStateOperations) (*cudaShapeDecode, error) {
	if ops.forward == nil || ops.consumers == nil || ops.hc == nil || ops.mix == nil {
		return nil, errors.New("cuda shape state: missing constructor")
	}
	s := new(cudaShapeDecode)
	fail := func(stage string, err error) (*cudaShapeDecode, error) {
		return nil, finishBackendCleanup(fmt.Errorf("cuda shape state: %s: %w", stage, err), s)
	}
	var err error
	if s.forward, err = ops.forward(); err != nil {
		return fail("forward", err)
	}
	if s.forward == nil {
		return fail("forward", errors.New("missing owner"))
	}
	if s.consumers, err = ops.consumers(); err != nil {
		return fail("consumers", err)
	}
	if s.consumers == nil {
		return fail("consumers", errors.New("missing owner"))
	}
	if s.hc, err = ops.hc(); err != nil {
		return fail("hc", err)
	}
	if s.hc == nil {
		return fail("hc", errors.New("missing owner"))
	}
	if s.mix, err = ops.mix(); err != nil {
		return fail("mix", err)
	}
	if s.mix == nil {
		return fail("mix", errors.New("missing owner"))
	}
	state, err := newCUDAShapeDecode(s.forward, s.consumers, s.hc, s.mix, binding, weights)
	if err != nil {
		return fail("attach", err)
	}
	return state, nil
}
