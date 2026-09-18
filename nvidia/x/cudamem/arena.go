package cudamem

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
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
	dev     *cuda.Device
	tracker *Tracker
	err     error
	frees   []func() error

	// chunk, when nonzero, is the size of the blocks the arena carves
	// allocations out of. block is the current one and offset the bytes of it
	// already handed out.
	chunk  int
	block  Slice[byte]
	offset int
}

// NewArena returns an empty Arena allocating from dev.
func NewArena(dev *cuda.Device) *Arena {
	return &Arena{dev: dev}
}

// NewChunkedArena returns an arena that carves allocations out of blocks of chunk
// bytes rather than making one device allocation per [AllocIn]. A caller that
// allocates tens of thousands of small buffers with one lifetime pays one
// cudaMalloc per block instead of one per buffer; at roughly six microseconds
// each that is the difference between milliseconds and a second.
//
// Sub-allocations are aligned to 256 bytes, which is what the driver returns for
// an allocation of its own, so a kernel cannot tell the two apart. They share the
// block's ownership, exactly as [Slice.Slice] results share their parent's: freeing
// one releases the whole block. [Arena.Free] is the intended release path and
// releases every block.
//
// A request larger than chunk gets a block of its own.
func NewChunkedArena(dev *cuda.Device, chunk int) *Arena {
	return &Arena{dev: dev, chunk: chunk}
}

// NewTrackedArena returns an arena whose allocations are accounted by tracker.
func NewTrackedArena(dev *cuda.Device, tracker *Tracker) *Arena {
	return &Arena{dev: dev, tracker: tracker}
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
	if a.chunk > 0 {
		s, err := allocInBlock[T](a, length)
		if err != nil {
			a.err = fmt.Errorf("cudamem: arena alloc %s: %w", name, err)
			return Slice[T]{}
		}
		return s
	}
	s, err := AllocTracked[T](a.dev, a.tracker, length)
	if err != nil {
		a.err = fmt.Errorf("cudamem: arena alloc %s: %w", name, err)
		return Slice[T]{}
	}
	a.frees = append(a.frees, s.Free)
	return s
}

// arenaBlockAlign is the alignment sub-allocations are given. The driver aligns
// its own allocations to at least this, so nothing downstream -- a vectorized
// load, a cp.async -- sees a weaker pointer than it would have.
const arenaBlockAlign = 256

func allocInBlock[T any](a *Arena, length int) (Slice[T], error) {
	if length < 0 {
		return Slice[T]{}, errors.New("cudamem: negative allocation length")
	}
	var t T
	elemSize := unsafe.Sizeof(t)
	if elemSize == 0 {
		elemSize = 1
	}
	if uintptr(length) > uintptr(^uint(0))/elemSize {
		return Slice[T]{}, fmt.Errorf("cudamem: allocation size overflows for %d elements", length)
	}
	n := uintptr(length) * elemSize
	size := int((n + arenaBlockAlign - 1) / arenaBlockAlign * arenaBlockAlign)
	if size == 0 {
		size = arenaBlockAlign
	}
	if a.block.len-a.offset < size {
		block, err := AllocTracked[byte](a.dev, a.tracker, max(size, a.chunk))
		if err != nil {
			return Slice[T]{}, err
		}
		a.frees = append(a.frees, block.Free)
		a.block, a.offset = block, 0
	}
	s := Slice[T]{
		ptr:      a.block.ptr + driver.DevicePtr(a.offset),
		len:      length,
		cap:      length,
		elemSize: elemSize,
		alloc:    a.block.alloc,
	}
	a.offset += size
	return s, nil
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
