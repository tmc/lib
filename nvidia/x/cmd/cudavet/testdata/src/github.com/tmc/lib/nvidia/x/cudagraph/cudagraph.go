// Package cudagraph is a declaration-only stand-in for the real x/cudagraph
// package, whose Capture records the work of a function on a stream.
package cudagraph

import "github.com/tmc/lib/nvidia/x/cuda"

type Graph struct{}

func Capture(s *cuda.Stream, fn func()) (*Graph, error) { return nil, nil }
