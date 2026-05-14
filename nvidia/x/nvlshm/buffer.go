package nvlshm

import (
	"errors"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

// A DistBuffer is a distributed buffer of WorldSize chunks mapped
// contiguously; chunk i is physically resident on rank i's GPU. All chunks
// are read-write from this rank. Create with [DistAlloc] or
// [DistAllocMulticast].
type DistBuffer struct {
	base       driver.DevicePtr
	chunkBytes uintptr
	world      int
	rank       int
	dev        driver.Device

	mcVA     driver.DevicePtr // 0 when the buffer has no multicast mapping
	mcHandle uint64

	mu     sync.Mutex
	closed bool
}

// Ptr returns the base of the contiguous mapping of all chunks.
func (b *DistBuffer) Ptr() driver.DevicePtr { return b.base }

// ChunkBytes returns the size of one rank's chunk in bytes.
func (b *DistBuffer) ChunkBytes() uintptr { return b.chunkBytes }

// TotalBytes returns the size of the full mapping (ChunkBytes × WorldSize).
func (b *DistBuffer) TotalBytes() uintptr { return b.chunkBytes * uintptr(b.world) }

// WorldSize returns the number of chunks.
func (b *DistBuffer) WorldSize() int { return b.world }

// Rank returns this process's rank.
func (b *DistBuffer) Rank() int { return b.rank }

// ChunkPtr returns the address of rank r's chunk within the mapping.
// It panics if r is out of range.
func (b *DistBuffer) ChunkPtr(r int) driver.DevicePtr {
	if r < 0 || r >= b.world {
		panic("nvlshm: chunk rank out of range")
	}
	return b.base + driver.DevicePtr(uintptr(r)*b.chunkBytes)
}

// LocalPtr returns the address of this rank's own chunk.
func (b *DistBuffer) LocalPtr() driver.DevicePtr { return b.ChunkPtr(b.rank) }

// McPtr returns the multicast (multimem) address and true when the buffer
// was created with [DistAllocMulticast]. A store to McPtr()+off is fanned
// out by the NVSwitch to ChunkPtr(r)+off on every rank r.
func (b *DistBuffer) McPtr() (driver.DevicePtr, bool) {
	return b.mcVA, b.mcVA != 0
}

// Close synchronizes the context and releases the buffer's mappings: the
// multicast view first, then every chunk mapping, then the address
// reservation. Close is idempotent. Callers must ensure no peer is still
// accessing this rank's chunk — tear down behind a barrier.
func (b *DistBuffer) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.closed = true

	// Device-wide sync (not current-stream): in-flight kernels on any
	// stream must finish before their memory is unmapped.
	errs := []error{driver.CtxSynchronize()}

	if b.mcVA != 0 {
		errs = append(errs,
			b.mcVA.MemUnmap(uint(b.chunkBytes)),
			b.mcVA.MemAddressFree(uint(b.chunkBytes)),
			driver.MemRelease(b.mcHandle))
		b.mcVA, b.mcHandle = 0, 0
	}
	for i := 0; i < b.world; i++ {
		va := b.base + driver.DevicePtr(uintptr(i)*b.chunkBytes)
		errs = append(errs, va.MemUnmap(uint(b.chunkBytes)))
	}
	errs = append(errs, b.base.MemAddressFree(uint(b.TotalBytes())))
	b.base = 0
	return errors.Join(errs...)
}

// An OwnedBuffer is a buffer physically resident on one owner rank's GPU,
// mapped read-write into every rank's address space. Create with
// [SingleOwnerAlloc].
type OwnedBuffer struct {
	base  driver.DevicePtr
	bytes uintptr
	owner int
	dev   driver.Device

	mu     sync.Mutex
	closed bool
}

// Ptr returns the buffer's address in this process.
func (b *OwnedBuffer) Ptr() driver.DevicePtr { return b.base }

// Bytes returns the buffer size in bytes.
func (b *OwnedBuffer) Bytes() uintptr { return b.bytes }

// Owner returns the rank whose GPU holds the physical memory.
func (b *OwnedBuffer) Owner() int { return b.owner }

// Close synchronizes the context and releases the mapping. Close is
// idempotent. As with [DistBuffer.Close], sequence it behind a barrier.
func (b *OwnedBuffer) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.closed = true
	errs := []error{
		driver.CtxSynchronize(),
		b.base.MemUnmap(uint(b.bytes)),
		b.base.MemAddressFree(uint(b.bytes)),
	}
	b.base = 0
	return errors.Join(errs...)
}
