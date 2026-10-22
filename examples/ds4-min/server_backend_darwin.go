//go:build darwin && arm64

package main

import (
	"errors"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func openServerBackend(model string, context int, ballast float64, warmWeights bool) (dsv4.IndependentStates, error) {
	if ballast != 0 {
		return nil, errors.New("ballast-gib requires CUDA")
	}
	if warmWeights {
		return nil, errors.New("warm-weights requires CUDA")
	}
	r, err := newMetalRuntime(model, context)
	if err != nil {
		return nil, err
	}
	return &metalStateAdapter{runtime: r}, nil
}
