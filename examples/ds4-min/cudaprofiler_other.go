//go:build !linux

package main

import "errors"

func startCUDAProfiler() error {
	return errors.New("cuProfilerStart: not available on this platform")
}

func stopCUDAProfiler() error {
	return errors.New("cuProfilerStop: not available on this platform")
}
