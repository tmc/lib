// Package cudamem is a declaration-only stand-in for the real cudamem
// package, whose asynchronous copies pin the host buffer themselves.
package cudamem

import "github.com/tmc/lib/nvidia/driver"

type Slice[T any] struct{}

func (s Slice[T]) Free() error                                           { return nil }
func (s Slice[T]) CopyFromHostAsync(src []T, stream driver.Stream) error { return nil }
func (s Slice[T]) CopyToHostAsync(dst []T, stream driver.Stream) error   { return nil }
