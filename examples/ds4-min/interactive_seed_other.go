//go:build !darwin && !linux

package main

import "errors"

func interactiveClockSeed() (uint64, error) {
	return 0, errors.New("interactive CPU-clock seed unsupported on this platform")
}
