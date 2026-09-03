package metalmem

import (
	"fmt"

	applemetal "github.com/tmc/lib/apple/x/metal"
)

// Arena groups allocations that share a lifetime. Allocations made through
// [AllocIn] and [UploadIn] are checked once, with [Arena.Err], and released
// together by [Arena.Free], so a structure holding a dozen buffers needs one
// error check and one cleanup path instead of one per buffer.
//
// The zero Arena is not usable; obtain one from [NewArena]. An Arena is not
// safe for concurrent use.
type Arena struct {
	dev   *applemetal.Device
	err   error
	frees []func()
}

// NewArena returns an empty Arena allocating from dev.
func NewArena(dev *applemetal.Device) *Arena {
	return &Arena{dev: dev}
}

// Device returns the device the arena allocates from.
func (a *Arena) Device() *applemetal.Device {
	return a.dev
}

// AllocIn allocates a Slice[T] of length elements from a and records it for
// release by [Arena.Free]. The name identifies the allocation in the arena's
// error.
//
// After the first failure AllocIn allocates nothing further and returns the
// zero Slice, which binds a null pointer, so callers may defer error handling
// to a single [Arena.Err] check.
func AllocIn[T any](a *Arena, name string, length int) Slice[T] {
	if a.err != nil {
		return Slice[T]{}
	}
	s, err := Alloc[T](a.dev, length)
	if err != nil {
		a.err = fmt.Errorf("metalmem: arena alloc %s: %w", name, err)
		return Slice[T]{}
	}
	a.keep(s.owner)
	return s
}

// UploadIn allocates a Slice[T] holding a copy of data and records it for
// release by [Arena.Free]. It is [AllocIn] followed by the copy, which is one
// step on unified memory. The name identifies the allocation in the arena's
// error.
func UploadIn[T any](a *Arena, name string, data []T) Slice[T] {
	if a.err != nil {
		return Slice[T]{}
	}
	s, err := AllocFromSlice[T](a.dev, data)
	if err != nil {
		a.err = fmt.Errorf("metalmem: arena upload %s: %w", name, err)
		return Slice[T]{}
	}
	a.keep(s.owner)
	return s
}

// keep records an allocation's owner for release by Free.
func (a *Arena) keep(o *owner) {
	if o != nil {
		a.frees = append(a.frees, o.release)
	}
}

// Err returns the first error from an allocation made through the arena, or
// nil if every allocation succeeded.
func (a *Arena) Err() error {
	return a.err
}

// Free releases every allocation the arena still holds, in the reverse order
// it made them. Freeing an arena twice is safe: the second call has nothing to
// release.
//
// Unlike its CUDA counterpart Free reports nothing, because releasing a Metal
// buffer cannot fail.
func (a *Arena) Free() {
	frees := a.frees
	a.frees = nil
	for i := len(frees) - 1; i >= 0; i-- {
		frees[i]()
	}
}
