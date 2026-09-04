package cudamem

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
)

// Arena groups device allocations that share a lifetime. Allocations made through
// [AllocIn] are checked once, with [Arena.Err], and released together by [Arena.Free],
// so a structure holding a dozen buffers needs one error check and one cleanup path
// instead of one per buffer.
//
// The zero Arena is not usable; obtain one from [NewArena]. An Arena is not safe for
// concurrent use.
type Arena struct {
	dev   *cuda.Device
	err   error
	frees []func() error
}

// NewArena returns an empty Arena allocating from dev.
func NewArena(dev *cuda.Device) *Arena {
	return &Arena{dev: dev}
}

// Device returns the device the arena allocates from.
func (a *Arena) Device() *cuda.Device {
	return a.dev
}

// AllocIn allocates a Slice[T] of length elements from a and records it for release
// by [Arena.Free]. The name identifies the allocation in the arena's error.
//
// After the first failure AllocIn allocates nothing further and returns the zero Slice,
// which packs as a null device pointer, so callers may defer error handling to a single
// [Arena.Err] check.
func AllocIn[T any](a *Arena, name string, length int) Slice[T] {
	if a.err != nil {
		return Slice[T]{}
	}
	s, err := Alloc[T](a.dev, length)
	if err != nil {
		a.err = fmt.Errorf("cudamem: arena alloc %s: %w", name, err)
		return Slice[T]{}
	}
	a.frees = append(a.frees, s.Free)
	return s
}

// Err returns the first error from an allocation made through the arena, or nil if
// every allocation succeeded.
func (a *Arena) Err() error {
	return a.err
}

// Free releases every allocation the arena still holds, in the reverse order it made
// them, and reports the errors from any releases that failed. Freeing an arena twice is
// safe: the second call has nothing to release.
func (a *Arena) Free() error {
	frees := a.frees
	a.frees = nil
	var errs []error
	for i := len(frees) - 1; i >= 0; i-- {
		if err := frees[i](); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
