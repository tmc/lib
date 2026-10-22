//go:build linux

package main

import "github.com/tmc/lib/examples/ds4-min/dsv4"

func openServerBackend(model string, context int, ballast float64, warmWeights bool) (dsv4.IndependentStates, error) {
	r, err := newCUDARuntimeWithOptions(model, context, cudaRuntimeOptions{
		ballastGib: ballast, ballastSet: ballast != 0,
		warmWeights: warmWeights, warmWeightsSet: warmWeights,
	})
	if err != nil {
		return nil, err
	}
	return &cudaStateAdapter{runtime: r}, nil
}
