package metalmem

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	applemetal "github.com/tmc/lib/apple/x/metal"
)

// Slice is a typed, contiguous run of elements in Apple unified memory.
//
// A Slice is a value: copying one shares the underlying MTLBuffer rather than
// duplicating it, and so does [Slice.Slice]. Every copy shares one owner, so
// the buffer is released by the first [Slice.Free] and later calls do nothing.
//
// The zero Slice has no buffer and no elements. [Slice.Buffer] returns nil for
// it, which binds a null pointer into a kernel's argument table, so a kernel
// that accepts an optional input can be given a zero Slice directly.
type Slice[T any] struct {
	buf      metal.MTLBuffer
	off      int // element offset of the first element within buf
	len      int
	cap      int
	elemSize uintptr
	owner    *owner
}

// owner releases the resources behind one buffer exactly once. Slices derived
// from an allocation share its owner, so freeing any of them releases the
// allocation and freeing the rest is a no-op.
type owner struct {
	once     sync.Once
	released atomic.Bool
	buf      metal.MTLBuffer
	mmap     []byte // the mapping to unmap, for zero-copy slices
	charge   *charge
}

func (o *owner) release() {
	if o == nil {
		return
	}
	o.once.Do(func() {
		o.released.Store(true)
		if o.buf != nil && o.buf.GetID() != 0 {
			objectivec.ObjectFromID(o.buf.GetID()).Release()
		}
		if o.mmap != nil {
			_ = syscall.Munmap(o.mmap)
		}
		o.charge.release()
	})
}

// elemSizeOf reports the byte size of T, treating an empty type as one byte so
// that a length still maps to a distinct allocation size.
func elemSizeOf[T any]() uintptr {
	var t T
	if size := unsafe.Sizeof(t); size != 0 {
		return size
	}
	return 1
}

// byteLen returns the number of bytes needed for length elements. Metal's
// allocation entry points take int byte counts, so reject values that would
// wrap when converted from uintptr.
func byteLen(length int, elemSize uintptr) (int, error) {
	if length < 0 {
		return 0, fmt.Errorf("negative allocation length %d", length)
	}
	const maxInt = int(^uint(0) >> 1)
	if uintptr(length) > uintptr(maxInt)/elemSize {
		return 0, fmt.Errorf("allocation size overflows for %d elements", length)
	}
	return int(uintptr(length) * elemSize), nil
}

// Alloc allocates a Slice[T] of length elements in shared unified memory. A
// length of zero allocates nothing and returns the zero Slice.
func Alloc[T any](dev *applemetal.Device, length int) (Slice[T], error) {
	return alloc[T](dev, length, nil)
}

func alloc[T any](dev *applemetal.Device, length int, tracker *Tracker) (Slice[T], error) {
	elemSize := elemSizeOf[T]()
	totalBytes, err := byteLen(length, elemSize)
	if err != nil {
		return Slice[T]{}, fmt.Errorf("metalmem: %w", err)
	}
	if length == 0 {
		return Slice[T]{}, nil
	}
	if dev == nil {
		return Slice[T]{}, fmt.Errorf("metalmem: nil device")
	}
	buf, err := dev.AllocBuffer(totalBytes)
	if err != nil {
		return Slice[T]{}, fmt.Errorf("metalmem: alloc %d elements: %w", length, err)
	}
	return newSlice[T](buf, length, elemSize, nil, tracker.acquire(uint64(totalBytes))), nil
}

// AllocFromSlice allocates a Slice[T] holding a copy of data. An empty data
// allocates nothing and returns the zero Slice.
func AllocFromSlice[T any](dev *applemetal.Device, data []T) (Slice[T], error) {
	return allocFromSlice[T](dev, data, nil)
}

func allocFromSlice[T any](dev *applemetal.Device, data []T, tracker *Tracker) (Slice[T], error) {
	if len(data) == 0 {
		return Slice[T]{}, nil
	}
	if dev == nil {
		return Slice[T]{}, fmt.Errorf("metalmem: nil device")
	}
	elemSize := elemSizeOf[T]()
	totalBytes := uintptr(len(data)) * elemSize
	buf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&data[0]), uint(totalBytes), metal.MTLResourceStorageModeShared)
	if buf == nil || buf.GetID() == 0 {
		return Slice[T]{}, fmt.Errorf("metalmem: alloc buffer for %d elements failed", len(data))
	}
	return newSlice[T](buf, len(data), elemSize, nil, tracker.acquire(uint64(totalBytes))), nil
}

// AllocMMapZeroCopy maps length elements of f starting at offset and wraps the
// mapping in a Metal buffer. [Slice.Free] unmaps it.
//
// The buffer holds a copy of the mapping rather than aliasing it: the Metal
// call that would alias host pages requires a page-aligned pointer and length
// and a deallocator, which this does not yet use. Writes through
// [Slice.HostSlice] therefore do not reach the file.
func AllocMMapZeroCopy[T any](dev *applemetal.Device, f *os.File, offset int64, length int) (Slice[T], error) {
	if dev == nil {
		return Slice[T]{}, fmt.Errorf("metalmem: nil device")
	}
	elemSize := elemSizeOf[T]()
	totalBytes, err := byteLen(length, elemSize)
	if err != nil {
		return Slice[T]{}, fmt.Errorf("metalmem: %w", err)
	}
	if length == 0 {
		return Slice[T]{}, fmt.Errorf("metalmem: invalid zero-copy length %d", length)
	}
	if f == nil {
		return Slice[T]{}, fmt.Errorf("metalmem: nil mmap file")
	}

	data, err := syscall.Mmap(int(f.Fd()), offset, totalBytes, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return Slice[T]{}, fmt.Errorf("metalmem: mmap file: %w", err)
	}

	buf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&data[0]), uint(totalBytes), metal.MTLResourceStorageModeShared)
	if buf == nil || buf.GetID() == 0 {
		_ = syscall.Munmap(data)
		return Slice[T]{}, fmt.Errorf("metalmem: create Metal buffer from mapping of %d bytes failed", totalBytes)
	}
	return newSlice[T](buf, length, elemSize, data, nil), nil
}

// newSlice wraps a freshly created buffer, which the returned Slice owns.
func newSlice[T any](buf metal.MTLBuffer, length int, elemSize uintptr, mmap []byte, charge *charge) Slice[T] {
	return Slice[T]{
		buf:      buf,
		len:      length,
		cap:      length,
		elemSize: elemSize,
		owner:    &owner{buf: buf, mmap: mmap, charge: charge},
	}
}

// Buffer returns the underlying MTLBuffer, or nil for the zero Slice. Binding
// a nil buffer gives the kernel a null pointer, so an optional input needs no
// branch at the call site. Bind it at [Slice.Offset], not at zero.
func (s Slice[T]) Buffer() metal.MTLBuffer {
	if s.owner != nil && s.owner.released.Load() {
		return nil
	}
	return s.buf
}

// Offset returns the byte offset of the slice's first element within
// [Slice.Buffer]. It is zero unless the slice came from [Slice.Slice].
func (s Slice[T]) Offset() uint {
	return uint(uintptr(s.off) * s.elemSize)
}

// Len returns the number of elements in the slice.
func (s Slice[T]) Len() int {
	return s.len
}

// Cap returns the number of elements from the slice's first element to the end
// of the allocation.
func (s Slice[T]) Cap() int {
	return s.cap
}

// ElemSize returns the byte size of one element.
func (s Slice[T]) ElemSize() uintptr {
	return s.elemSize
}

// TotalBytes returns the bytes occupied by the slice's elements.
func (s Slice[T]) TotalBytes() uint {
	return uint(uintptr(s.len) * s.elemSize)
}

// Free releases the allocation the slice came from. Every slice derived from
// one allocation shares its owner, so the first Free releases it and the rest
// do nothing; freeing the zero Slice does nothing.
func (s Slice[T]) Free() {
	s.owner.release()
}

// HostSlice returns a Go slice aliasing the unified buffer, or nil if the
// slice is empty. Writes through it are visible to the GPU.
func (s Slice[T]) HostSlice() []T {
	if s.owner != nil && s.owner.released.Load() {
		return nil
	}
	if s.len == 0 || s.buf == nil || s.buf.GetID() == 0 {
		return nil
	}
	ptr := s.buf.Contents()
	if ptr == nil {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Add(ptr, s.Offset())), s.len)
}

// Slice returns the elements of s from start to end. The result shares s's
// allocation and its owner, and reports its position within the buffer through
// [Slice.Offset].
func (s Slice[T]) Slice(start, end int) (Slice[T], error) {
	if s.owner != nil && s.owner.released.Load() {
		return Slice[T]{}, fmt.Errorf("metalmem: slice of released allocation")
	}
	if start < 0 || end < start || end > s.cap {
		return Slice[T]{}, fmt.Errorf("metalmem: slice bounds out of range [%d:%d] with cap %d", start, end, s.cap)
	}
	sub := s
	sub.off = s.off + start
	sub.len = end - start
	sub.cap = s.cap - start
	return sub, nil
}

// CopyFromHost copies src into the slice. It reports an error if src is longer
// than the slice.
func (s Slice[T]) CopyFromHost(src []T) error {
	if len(src) == 0 {
		return nil
	}
	if len(src) > s.len {
		return fmt.Errorf("metalmem: source length %d exceeds destination length %d", len(src), s.len)
	}
	dst := s.HostSlice()
	if dst == nil {
		return fmt.Errorf("metalmem: buffer contents pointer is nil")
	}
	copy(dst, src)
	return nil
}

// CopyToHost copies the slice into dst. It reports an error if dst is shorter
// than the slice.
func (s Slice[T]) CopyToHost(dst []T) error {
	if s.len == 0 {
		return nil
	}
	if len(dst) < s.len {
		return fmt.Errorf("metalmem: destination length %d smaller than source length %d", len(dst), s.len)
	}
	src := s.HostSlice()
	if src == nil {
		return fmt.Errorf("metalmem: buffer contents pointer is nil")
	}
	copy(dst, src)
	return nil
}

// Pool recycles buffers by power-of-two size class, for transient activations
// whose size repeats from step to step.
//
// The zero Pool is not usable; obtain one from [NewPool]. A Pool is safe for
// concurrent use.
type Pool struct {
	mu    sync.Mutex
	dev   *applemetal.Device
	slabs map[int][]metal.MTLBuffer
	owned map[uintptr]struct{}
}

// NewPool returns an empty Pool allocating from dev.
func NewPool(dev *applemetal.Device) *Pool {
	return &Pool{
		dev:   dev,
		slabs: make(map[int][]metal.MTLBuffer),
		owned: make(map[uintptr]struct{}),
	}
}

// Get returns a recycled buffer of at least minBytes, allocating one if the
// pool holds none.
func (p *Pool) Get(minBytes int) (metal.MTLBuffer, error) {
	if p == nil {
		return nil, fmt.Errorf("metalmem: nil pool")
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	bucket, err := sizeClass(minBytes)
	if err != nil {
		return nil, fmt.Errorf("metalmem: pool get: %w", err)
	}
	if list := p.slabs[bucket]; len(list) > 0 {
		buf := list[len(list)-1]
		p.slabs[bucket] = list[:len(list)-1]
		delete(p.owned, uintptr(buf.GetID()))
		return buf, nil
	}
	if p.dev == nil {
		return nil, fmt.Errorf("metalmem: pool has nil device")
	}
	return p.dev.AllocBuffer(bucket)
}

// Put returns a buffer of the given capacity to the pool for reuse. On error,
// ownership remains with the caller. Putting the same buffer twice is an
// error.
func (p *Pool) Put(buf metal.MTLBuffer, capacity int) error {
	if p == nil {
		return fmt.Errorf("metalmem: nil pool")
	}
	if buf == nil || buf.GetID() == 0 {
		return fmt.Errorf("metalmem: pool put nil buffer")
	}
	bucket, err := sizeClass(capacity)
	if err != nil {
		return fmt.Errorf("metalmem: pool put: %w", err)
	}
	if buf.Length() < uint(bucket) {
		return fmt.Errorf("metalmem: buffer length %d is smaller than size class %d", buf.Length(), bucket)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	id := uintptr(buf.GetID())
	if _, ok := p.owned[id]; ok {
		return fmt.Errorf("metalmem: buffer is already in pool")
	}
	p.slabs[bucket] = append(p.slabs[bucket], buf)
	p.owned[id] = struct{}{}
	return nil
}

// Free releases every buffer the pool holds.
func (p *Pool) Free() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, list := range p.slabs {
		for _, buf := range list {
			if buf != nil && buf.GetID() != 0 {
				objectivec.ObjectFromID(buf.GetID()).Release()
			}
		}
	}
	clear(p.slabs)
	clear(p.owned)
}

// sizeClass rounds n up to the power-of-two bucket the pool stores it in.
func sizeClass(n int) (int, error) {
	if n < 0 {
		return 0, fmt.Errorf("negative buffer size %d", n)
	}
	bucket := 1024
	for bucket < n {
		if bucket > int(^uint(0)>>1)/2 {
			return 0, fmt.Errorf("buffer size %d is too large", n)
		}
		bucket *= 2
	}
	return bucket, nil
}
