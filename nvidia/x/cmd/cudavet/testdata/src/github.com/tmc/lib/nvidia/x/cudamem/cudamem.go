// Package cudamem is a declaration-only stand-in for the real cudamem
// package, whose asynchronous copies pin the host buffer themselves.
package cudamem

import "github.com/tmc/lib/nvidia/x/cuda"

type Slice[T any] struct {
	alloc *allocation
}

type allocation struct{}

func (s Slice[T]) Free() error                                          { return nil }
func (s Slice[T]) CopyFromHostAsync(src []T, stream *cuda.Stream) error { return nil }
func (s Slice[T]) CopyToHostAsync(dst []T, stream *cuda.Stream) error   { return nil }
