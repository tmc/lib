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

	// spans records each device allocation the arena made, so [Arena.Fill] can
	// reach memory that AllocIn has handed out but nothing has written yet.
	spans []arenaSpan
}

// arenaSpan is one device allocation an arena made, in bytes.
type arenaSpan struct {
	ptr   driver.DevicePtr
	bytes int
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
	if a.chunk > 0 {
		if a.err != nil {
			return Slice[T]{}
		}
		s, err := allocInBlock[T](a, length)
		if err != nil {
			a.err = fmt.Errorf("cudamem: arena alloc %s: %w", name, err)
			return Slice[T]{}
		}
		return s
	}
	s := allocIn(a, name, func() (Slice[T], error) { return AllocTracked[T](a.dev, a.tracker, length) })
	if a.err != nil {
		return Slice[T]{}
	}
	a.spans = append(a.spans, arenaSpan{s.ptr, length * int(unsafe.Sizeof(*new(T)))})
	return s
}

func allocIn[T any](a *Arena, name string, allocate func() (Slice[T], error)) Slice[T] {
	if a.err != nil {
		return Slice[T]{}
	}
	s, err := allocate()
	if s.Ptr() != 0 {
		a.frees = append(a.frees, s.Free)
	}
	if err != nil {
		a.err = fmt.Errorf("cudamem: arena alloc %s: %w", name, err)
		return Slice[T]{}
	}
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
		a.spans = append(a.spans, arenaSpan{block.ptr, block.len})
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

// Fill sets every byte the arena has allocated to the low byte of word. Device
// memory comes back holding whatever the last owner left in it, so a buffer an
// arena has handed out but nothing has written yet reads as that. Filling it
// with a known value turns a read of uninitialized memory from something that
// depends on the machine's history into something repeatable: zero makes the
// buffer behave as if it had been cleared, and any other value makes a kernel
// that depends on it fail the same way every time.
//
// Fill is not part of allocation. An arena whose buffers are all written before
// they are read does not need it, and paying for it on every allocation would
// charge that arena for a property it already has.
func (a *Arena) Fill(word uint32) error {
	if a.err != nil {
		return a.err
	}
	return a.dev.WithContext(func() error {
		for _, s := range a.spans {
			if n := uint(s.bytes / 4); n > 0 {
				if err := s.ptr.MemsetD32(word, n); err != nil {
					return fmt.Errorf("cudamem: arena fill: %w", err)
				}
			}
			if rem := s.bytes % 4; rem > 0 {
				p := s.ptr + driver.DevicePtr(s.bytes-rem)
				if err := p.MemsetD8(uint8(word), uint(rem)); err != nil {
					return fmt.Errorf("cudamem: arena fill tail: %w", err)
				}
			}
		}
		return nil
	})
}

// Err returns the first error from an allocation made through the arena, or nil if
// every allocation succeeded.
func (a *Arena) Err() error {
	return a.err
}

// Free releases every allocation the arena still holds, in the reverse order it made
// them, and reports the errors from any releases that failed. Failed releases are
// retained for another Free; successful releases are never repeated.
func (a *Arena) Free() error {
	frees := a.frees
	a.spans = nil
	var errs []error
	for i := len(frees) - 1; i >= 0; i-- {
		if err := frees[i](); err != nil {
			errs = append(errs, err)
		} else {
			frees[i] = nil
		}
	}
	a.frees = nil
	for _, free := range frees {
		if free != nil {
			a.frees = append(a.frees, free)
		}
	}
	return errors.Join(errs...)
}
