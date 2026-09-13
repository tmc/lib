//go:build !linux

package main

import "errors"

func stopCUDAProfiler() error {
	return errors.New("cuProfilerStop: not available on this platform")
}
