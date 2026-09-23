//go:build !unix

package main

// monotonicNanos reports that this system has no CLOCK_MONOTONIC to share
// with an agent.
func monotonicNanos() (uint64, bool) { return 0, false }
