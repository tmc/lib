//go:build !linux && (!darwin || !arm64)

package main

import "github.com/tmc/lib/examples/ds4-min/dsv4"

func openConfiguredServerBackend(model string, context int, ballast float64, warmWeights bool, streaming streamingOptions) (dsv4.IndependentStates, error) {
	if err := streaming.validate("cuda"); err != nil {
		return nil, err
	}
	return openServerBackend(model, context, ballast, warmWeights)
}
