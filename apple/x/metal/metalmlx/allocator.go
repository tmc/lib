package metalmlx

import (
	"fmt"
	"sync"

	"github.com/tmc/apple/metal"
)

// resourceOptions are the options every allocation is made with. Shared
// storage puts the buffer in unified memory; hazard tracking is off because
// the command encoder inserts its own barriers and fences.
const resourceOptions = metal.MTLResourceStorageModeShared | metal.MTLResourceHazardTrackingModeUntracked

const (
	// smallSize is the size below which allocations go on the heap until it
	// is full. It is the minimum size of a buffer allocated from a heap, so a
	// heap holds at most heapSize/smallSize buffers.
	smallSize = 256
	heapSize  = 1 << 20
)

// Allocator is a caching allocator for Metal buffers. Port of MetalAllocator
// in mlx/backend/metal/allocator.cpp.
type Allocator struct {
	dev       metal.MTLDevice
	heap      metal.MTLHeap
	residency *ResidencySets

	mu    sync.Mutex
	cache *bufferCache[metal.MTLBuffer]

	blockLimit    uint64
	gcLimit       uint64
	activeMemory  uint64
	peakMemory    uint64
	maxPoolSize   uint64
	wiredLimit    uint64
	numResources  uint64
	resourceLimit uint64
}

func newAllocator(d *Device) *Allocator {
	dev := d.raw
	a := &Allocator{dev: dev, residency: d.residency}
	a.cache = newBufferCache(uint64(vmPageSize()),
		func(b metal.MTLBuffer) uint64 { return uint64(b.Length()) },
		func(b metal.MTLBuffer) {
			if !valid(b.Heap()) {
				a.residency.Erase(b)
			}
			release(b)
		})

	memSize := hostMemorySize()
	maxRecSize := dev.RecommendedMaxWorkingSetSize()
	a.resourceLimit = resourceLimit()
	a.blockLimit = min(uint64(1.5*float64(maxRecSize)), uint64(0.95*float64(memSize)))
	a.gcLimit = min(uint64(0.95*float64(maxRecSize)), a.blockLimit)
	a.maxPoolSize = a.blockLimit
	if dev.Name() == "Apple Paravirtual device" {
		return a // no heap in a VM
	}
	desc := metal.NewMTLHeapDescriptor()
	desc.SetResourceOptions(resourceOptions)
	desc.SetSize(heapSize)
	a.heap = dev.NewHeapWithDescriptor(desc)
	if valid(a.heap) {
		a.residency.Insert(a.heap, uint64(a.heap.AllocatedSize()))
	}
	return a
}

// Malloc returns a buffer of at least size bytes, from the cache when a
// suitable one is there. A zero size returns a nil buffer: Metal does not
// allow empty buffers.
func (a *Allocator) Malloc(size uint64) (metal.MTLBuffer, error) {
	if size == 0 {
		return nil, nil
	}
	if maxLen := uint64(a.dev.MaxBufferLength()); size > maxLen {
		return nil, fmt.Errorf("metalmlx: allocating %d bytes exceeds the maximum buffer size of %d bytes", size, maxLen)
	}
	page := uint64(vmPageSize())
	if size > page {
		size = page * ((size + page - 1) / page)
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	buf, hit := a.cache.reuseFromCache(size)
	if !hit {
		memRequired := a.activeMemory + a.cache.cacheSize() + size

		// Under memory pressure try to reclaim memory from the cache.
		if memRequired >= a.gcLimit || a.numResources >= a.resourceLimit {
			a.numResources -= uint64(a.cache.releaseCachedBuffers(memRequired - a.gcLimit))
		}
		if a.numResources >= a.resourceLimit {
			return nil, fmt.Errorf("metalmlx: resource limit (%d) exceeded", a.resourceLimit)
		}
		a.mu.Unlock()
		if size < smallSize && valid(a.heap) {
			buf = a.heap.NewBufferWithLengthOptions(uint(size), resourceOptions)
		}
		if !valid(buf) {
			buf = a.dev.NewBufferWithLengthOptions(uint(size), resourceOptions)
		}
		a.mu.Lock()
		if !valid(buf) {
			return nil, fmt.Errorf("metalmlx: unable to allocate %d bytes", size)
		}
		a.numResources++
		if !valid(buf.Heap()) {
			a.residency.Insert(buf, uint64(buf.AllocatedSize()))
		}
	}

	a.activeMemory += uint64(buf.Length())
	a.peakMemory = max(a.peakMemory, a.activeMemory)

	// Keep the cache below the requested limit.
	if a.cache.cacheSize() > a.maxPoolSize {
		a.numResources -= uint64(a.cache.releaseCachedBuffers(a.cache.cacheSize() - a.maxPoolSize))
	}
	return buf, nil
}

// Free returns a buffer to the cache, or releases it when the cache is full.
func (a *Allocator) Free(buf metal.MTLBuffer) {
	if !valid(buf) {
		return
	}
	a.mu.Lock()
	a.activeMemory -= uint64(buf.Length())
	if a.cache.cacheSize() < a.maxPoolSize {
		a.cache.recycleToCache(buf)
		a.mu.Unlock()
		return
	}
	a.numResources--
	if !valid(buf.Heap()) {
		a.residency.Erase(buf)
	}
	a.mu.Unlock()
	release(buf)
}

// Release frees a buffer without caching it.
func (a *Allocator) Release(buf metal.MTLBuffer) {
	if !valid(buf) {
		return
	}
	a.mu.Lock()
	a.activeMemory -= uint64(buf.Length())
	a.numResources--
	a.residency.Erase(buf)
	a.mu.Unlock()
	release(buf)
}

// ActiveMemory returns the bytes held by live buffers.
func (a *Allocator) ActiveMemory() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.activeMemory
}

// PeakMemory returns the high water mark of ActiveMemory.
func (a *Allocator) PeakMemory() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.peakMemory
}

// ResetPeakMemory sets the peak memory to the active memory.
func (a *Allocator) ResetPeakMemory() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.peakMemory = 0
}

// CacheMemory returns the bytes held by cached buffers.
func (a *Allocator) CacheMemory() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.cache.cacheSize()
}

// SetCacheLimit sets the maximum cache size and returns the previous limit.
func (a *Allocator) SetCacheLimit(limit uint64) uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.maxPoolSize, limit = limit, a.maxPoolSize
	return limit
}

// SetMemoryLimit sets the soft limit on active memory, above which the cache
// is reclaimed, and returns the previous limit.
func (a *Allocator) SetMemoryLimit(limit uint64) uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.blockLimit, limit = limit, a.blockLimit
	a.gcLimit = min(a.blockLimit, uint64(0.95*float64(a.dev.RecommendedMaxWorkingSetSize())))
	return limit
}

// MemoryLimit returns the current memory limit.
func (a *Allocator) MemoryLimit() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.blockLimit
}

// SetWiredLimit sets how many bytes stay GPU-resident and returns the
// previous limit.
func (a *Allocator) SetWiredLimit(limit uint64) (uint64, error) {
	if maxRec := a.dev.RecommendedMaxWorkingSetSize(); limit > maxRec {
		return 0, fmt.Errorf("metalmlx: wired limit %d is larger than the maximum working set size %d", limit, maxRec)
	}
	a.mu.Lock()
	a.wiredLimit, limit = limit, a.wiredLimit
	wired := a.wiredLimit
	a.mu.Unlock()
	a.residency.Resize(wired)
	return limit, nil
}

// ClearCache releases every cached buffer.
func (a *Allocator) ClearCache() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.numResources -= uint64(a.cache.clear())
}

func (a *Allocator) close() {
	a.mu.Lock()
	a.numResources -= uint64(a.cache.clear())
	if valid(a.heap) {
		a.residency.Erase(a.heap)
		release(a.heap)
		a.heap = nil
	}
	a.mu.Unlock()
}
