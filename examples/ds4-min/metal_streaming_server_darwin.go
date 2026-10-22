//go:build darwin && arm64

package main

import (
	"errors"
	"io"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func openConfiguredServerBackend(model string, context int, ballast float64, warmWeights bool, streaming streamingOptions) (dsv4.IndependentStates, error) {
	if !streaming.enabled {
		if err := streaming.validate("metal"); err != nil {
			return nil, err
		}
		return openServerBackend(model, context, ballast, warmWeights)
	}
	if err := streaming.validate("metal"); err != nil {
		return nil, err
	}
	if ballast != 0 || warmWeights {
		return nil, errors.New("CUDA residency options unavailable on Metal")
	}
	return openMetalStreamingServer(model, context, streaming, newMetalRuntimeStreamingLogged)
}

func openMetalStreamingServer(model string, context int, streaming streamingOptions, create func(string, int, io.Writer, streamingOptions) (*metalRuntime, error)) (dsv4.IndependentStates, error) {
	r, err := create(model, context, nil, streaming)
	if err != nil {
		if r != nil {
			err = finishBackendCleanup(err, r)
		}
		return nil, err
	}
	if r == nil {
		return nil, errors.New("Metal constructor returned no owner")
	}
	return &metalStateAdapter{runtime: r}, nil
}
