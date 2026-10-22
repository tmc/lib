//go:build !linux && !(darwin && arm64)

package main

import (
	"errors"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func openServerBackend(string, int, float64, bool) (dsv4.IndependentStates, error) {
	return nil, errors.New("server backend requires CUDA on linux or Metal on darwin arm64")
}
