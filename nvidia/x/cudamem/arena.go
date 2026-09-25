package cudamem

import (
	"errors"
	"fmt"
	"slices"
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
	// allocations out of. blocks holds every block in the order allocations
	// use them; blocks[cur] is the current one and offset the bytes of it
	// already handed out. Release and Reset move cur and offset back, and
	// later allocations reuse the blocks after them.
	chunk  int
	blocks []Slice[byte]
	cur    int
	offset int

	// newBlock allocates a block and freeBlock frees one; tests replace
	// them to count calls.
	newBlock  func(bytes int) (Slice[byte], error)
	freeBlock func(Slice[byte]) error

	// inUse is the bytes handed out and not released, and highWater the
	// most inUse has been.
	inUse, highWater int

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
	bytes := length * int(unsafe.Sizeof(*new(T)))
	a.spans = append(a.spans, arenaSpan{s.ptr, bytes})
	a.use(bytes)
	return s
}

func (a *Arena) use(bytes int) {
	a.inUse += bytes
	a.highWater = max(a.highWater, a.inUse)
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
	if len(a.blocks) == 0 || a.blocks[a.cur].len-a.offset < size {
		if err := a.nextBlock(size); err != nil {
			return Slice[T]{}, err
		}
	}
	block := a.blocks[a.cur]
	s := Slice[T]{
		ptr:      block.ptr + driver.DevicePtr(a.offset),
		len:      length,
		cap:      length,
		elemSize: elemSize,
		alloc:    block.alloc,
	}
	a.offset += size
	a.use(size)
	return s, nil
}

// nextBlock makes the block after the current one current, allocating a new
// one there if the next block is missing or smaller than size. With the
// blocks reused in order, a sequence of requests repeated after a Reset gets
// the same pointers and allocates nothing.
//
// A next block smaller than size holds nothing live, since allocations fill
// blocks in order, so it is freed and replaced rather than kept behind the
// new one. Otherwise steps whose large requests vary in size would leave
// a block of every size seen and the arena would grow without bound.
func (a *Arena) nextBlock(size int) error {
	next := a.cur + 1
	if len(a.blocks) == 0 {
		next = 0
	}
	if next < len(a.blocks) && a.blocks[next].len >= size {
		a.cur, a.offset = next, 0
		return nil
	}
	alloc := a.newBlock
	if alloc == nil {
		alloc = func(bytes int) (Slice[byte], error) { return AllocTracked[byte](a.dev, a.tracker, bytes) }
	}
	block, err := alloc(max(size, a.chunk))
	if err != nil {
		return err
	}
	a.spans = append(a.spans, arenaSpan{block.ptr, block.len})
	if next < len(a.blocks) {
		old := a.blocks[next]
		a.blocks[next] = block
		a.spans = slices.DeleteFunc(a.spans, func(s arenaSpan) bool { return s.ptr == old.ptr })
		if err := a.releaseBlock(old); err != nil {
			a.frees = append(a.frees, func() error { return a.releaseBlock(old) })
		}
	} else {
		a.blocks = append(a.blocks, block)
	}
	a.cur, a.offset = next, 0
	return nil
}

func (a *Arena) releaseBlock(b Slice[byte]) error {
	if a.freeBlock != nil {
		return a.freeBlock(b)
	}
	return b.Free()
}

// A Mark records a point in an arena's allocations for [Arena.Release].
// The zero Mark is the arena's start.
type Mark struct {
	block, offset int // chunked arenas: the cursor
	frees, spans  int // other arenas: the allocations made before the mark
	inUse         int
}

// Mark returns the current point in the arena's allocations.
func (a *Arena) Mark() Mark {
	if a.chunk > 0 {
		return Mark{block: a.cur, offset: a.offset, inUse: a.inUse}
	}
	return Mark{frees: len(a.frees), spans: len(a.spans), inUse: a.inUse}
}

// Release releases everything allocated through the arena since m was
// taken. Slices allocated since then must no longer be used, and marks
// taken since then are invalid. Release also clears [Arena.Err], since the
// failed allocation it reports came after the mark or was never made.
//
// A chunked arena ([NewChunkedArena]) keeps its blocks and hands the
// released space out again, so a training step that calls Reset, or
// Release to a mark taken at its start, and then makes the same requests
// as the step before gets the same pointers and makes no driver
// allocations. That is what lets a captured CUDA graph keep pointing at
// valid buffers. The device memory is not cleared; see [Arena.Fill] and
// [Slice.MemsetAsync].
//
// An unchunked arena frees the allocations made since m, most recent
// first, and keeps any whose free fails for [Arena.Free].
func (a *Arena) Release(m Mark) error {
	if m.inUse > a.inUse {
		return errors.New("cudamem: arena release to a mark past the current allocations")
	}
	a.err = nil
	a.inUse = m.inUse
	if a.chunk > 0 {
		a.cur, a.offset = m.block, m.offset
		return nil
	}
	if m.frees > len(a.frees) || m.spans > len(a.spans) {
		return errors.New("cudamem: arena release to a mark from another arena or a freed one")
	}
	var errs []error
	released := slices.Clone(a.frees[m.frees:])
	a.frees = a.frees[:m.frees]
	var failed []func() error
	for i := len(released) - 1; i >= 0; i-- {
		if err := released[i](); err != nil {
			errs = append(errs, err)
			failed = append(failed, released[i])
		}
	}
	slices.Reverse(failed)
	a.frees = append(a.frees, failed...)
	a.spans = a.spans[:m.spans]
	return errors.Join(errs...)
}

// Reset releases everything allocated through the arena, as Release to the
// zero Mark does. A chunked arena keeps its blocks for reuse; [Arena.Free]
// returns them to the driver.
func (a *Arena) Reset() error {
	return a.Release(Mark{})
}

// ArenaStats describes an arena's memory use in bytes.
type ArenaStats struct {
	// InUse is the memory handed out and not yet released. A chunked
	// arena counts each allocation rounded up to its 256-byte alignment.
	InUse int
	// HighWater is the most InUse has been since the arena was created
	// or last freed. It is the size to give a chunked arena's blocks so
	// a step fits in one.
	HighWater int
	// Reserved is the device memory the arena holds: its blocks for a
	// chunked arena, its live allocations otherwise.
	Reserved int
}

// Stats returns the arena's memory use.
func (a *Arena) Stats() ArenaStats {
	st := ArenaStats{InUse: a.inUse, HighWater: a.highWater}
	for _, s := range a.spans {
		st.Reserved += s.bytes
	}
	return st
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
	for _, b := range a.blocks {
		frees = append(frees, func() error { return a.releaseBlock(b) })
	}
	a.spans = nil
	a.blocks, a.cur, a.offset = nil, 0, 0
	a.inUse, a.highWater = 0, 0
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
