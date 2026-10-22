//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// legacyMTPCUDABinding borrows one exact target request. It does not admit
// support weights, allocate kernels, or enable a native MTP execution route.
type legacyMTPCUDABinding struct {
	owner              *cudaStateAdapter
	request            *cudaRequestState
	key                *legacyMTPRequestKey
	modelSHA, quantSHA [32]byte
}

// The expected hashes come from the admitted target manifest. Quantization
// identity is the native tensor-layout hash, not the JSON profile digest.
func newLegacyMTPCUDABinding(owner *cudaStateAdapter, request *cudaRequestState, expectedModel, expectedQuant [32]byte) (*legacyMTPCUDABinding, error) {
	if owner == nil || request == nil || request.owner != owner {
		return nil, fmt.Errorf("legacy mtp binding: request owner mismatch")
	}
	if _, err := request.speculativeEpoch(); err != nil {
		return nil, err
	}
	model, quant, err := request.runtime.cudaFoundation.nativeStateIdentity()
	if err != nil {
		return nil, fmt.Errorf("legacy mtp binding target identity: %w", err)
	}
	if err := validateLegacyMTPIdentity(model, quant, expectedModel, expectedQuant); err != nil {
		return nil, err
	}
	return &legacyMTPCUDABinding{owner: owner, request: request, key: new(legacyMTPRequestKey), modelSHA: model, quantSHA: quant}, nil
}

func (b *legacyMTPCUDABinding) frontier(history []int) (legacyMTPFrontier, error) {
	if b == nil || b.request == nil || b.owner == nil || b.request.owner != b.owner {
		return legacyMTPFrontier{}, fmt.Errorf("legacy mtp binding: request owner mismatch")
	}
	epoch, err := b.request.speculativeEpoch()
	if err != nil {
		return legacyMTPFrontier{}, err
	}
	model, quant, err := b.request.runtime.cudaFoundation.nativeStateIdentity()
	if err != nil {
		return legacyMTPFrontier{}, fmt.Errorf("legacy mtp binding target identity: %w", err)
	}
	if err := validateLegacyMTPIdentity(model, quant, b.modelSHA, b.quantSHA); err != nil {
		return legacyMTPFrontier{}, err
	}
	return newLegacyMTPFrontier(b.key, epoch, b.request.runtime.pos, history)
}

// capture publishes no preparation. The caller must revalidate again after
// producing support rows, immediately before publishing its derived result.
// Request operations are serialized by the owner; this is not a concurrency
// lock or a promise that concurrent State calls are safe.
func (b *legacyMTPCUDABinding) capture(expected legacyMTPFrontier, history []int) ([]float32, error) {
	current, err := b.frontier(history)
	if err != nil {
		return nil, err
	}
	if err := expected.validate(current); err != nil {
		return nil, err
	}
	hidden, err := captureLegacyMTPHC(b.request, expected.position)
	if err != nil {
		return nil, err
	}
	current, err = b.frontier(history)
	if err != nil {
		return nil, err
	}
	if err := expected.validate(current); err != nil {
		return nil, err
	}
	return hidden, nil
}

// captureLegacyMTPHC reads the committed terminal target HC after successful
// Eval. It does not change target state or add a persistent request buffer.
// Scalar eval leaves hc[0] intact after the output head and increments pos
// only after synchronization and finite logits admission (2178-2217).
func captureLegacyMTPHC(s *cudaRequestState, nextPosition int) ([]float32, error) {
	if s == nil || s.closed || s.runtime == nil {
		return nil, dsv4.ErrClosed
	}
	r := s.runtime
	if r.failed != nil {
		return nil, fmt.Errorf("legacy mtp capture: %w", r.failed)
	}
	if r.cudaFoundation == nil || r.model == nil || r.state.hc[0].Len() != cudaHCWidth {
		return nil, fmt.Errorf("legacy mtp capture: missing target HC resources")
	}
	if nextPosition <= 0 || r.pos != nextPosition {
		return nil, fmt.Errorf("legacy mtp capture: target frontier is not committed")
	}
	if err := checkFiniteLogits("legacy mtp capture", r.logits); err != nil {
		return nil, err
	}
	if len(r.logits) == 0 {
		return nil, fmt.Errorf("legacy mtp capture: missing target logits")
	}
	if err := syncStream(r.stream()); err != nil {
		return nil, fmt.Errorf("legacy mtp capture synchronize: %w", err)
	}
	hidden := make([]float32, cudaHCWidth)
	if err := r.state.hc[0].CopyToHost(hidden); err != nil {
		return nil, fmt.Errorf("legacy mtp capture HC: %w", err)
	}
	if !allFinite(hidden) {
		return nil, fmt.Errorf("legacy mtp capture: nonfinite target HC")
	}
	return hidden, nil
}
