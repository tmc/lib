//go:build darwin && arm64

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestMetalDSparkThresholdZero(t *testing.T) {
	ids, err := metalDSparkProposalResult([]int32{1, 2}, []float32{0, 0}, []uint32{2, 0}, 2, 0)
	if err != nil || len(ids) != 2 {
		t.Errorf("zero-confidence terminal result rejected: %v", err)
	}
	if _, err := metalDSparkProposalResult([]int32{1, 2}, []float32{0, 0}, []uint32{1, 0}, 2, 0); err == nil {
		t.Error("zero threshold admitted a filtered prefix")
	}
	a := &metalStateAdapter{}
	s := &metalRequestState{owner: a, runtime: &metalRuntime{}}
	w := &metalDSparkWeights{model: &dsparkModel{block: 4}}
	reached := errors.New("reached feature acquisition")
	factory := metalDSparkConstructorFactory{
		features: func(*metalRequestState, *metalDSparkWeights) (*metalDSparkFeatures, error) { return nil, reached },
		forward: func(*metalRequestState, *metalDSparkWeights) (*metalDSparkNative, error) {
			t.Fatal("forward after failed feature acquisition")
			return nil, nil
		},
		verifier: func(*metalRequestState) (dsv4.SpeculativeVerifier, error) {
			t.Fatal("verifier after failed acquisition")
			return nil, nil
		},
	}
	_, _, err = newMetalDSparkSpeculatorWithFactory(a, s, w, 0, factory)
	if !errors.Is(err, reached) {
		t.Errorf("zero was rejected before actual constructor acquisition: %v", err)
	}
}
