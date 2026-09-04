package cudamlx

import (
	"fmt"
	"math/bits"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// Port of allocator.cpp.
const (
	pageSize = 16384
	// Allocations at or below this size come from the small pool.
	smallBlockSize = 8
	// The small pool size in bytes, a multiple of the host page size and
	// smallBlockSize.
	smallPoolSize = 4 * pageSize
)

// A Buffer is one allocation tracked by the Allocator. Data is a pointer
// valid on both host and device for unified buffers (Device == -1), and a
// device pointer otherwise. Port of CudaBuffer.
type Buffer struct {
	Data   uintptr
	Size   uint64
	Device int // -1 for unified (managed or pinned host) memory

	small bool // owned by the small pool
	idx   int32
}

// Ptr returns the buffer's device pointer.
func (b *Buffer) Ptr() driver.DevicePtr {
	return driver.DevicePtr(b.Data)
}

type cudaBuffer = Buffer

// smallSizePool carves one unified allocation into smallBlockSize blocks.
// Port of SmallSizePool.
type smallSizePool struct {
	data     uintptr
	blocks   []Buffer
	freeList []int32
}

func newSmallSizePool(a *Allocator) (*smallSizePool, error) {
	p := &smallSizePool{}
	var err error
	p.data, err = a.unifiedMalloc(smallPoolSize)
	if err != nil {
		return nil, err
	}
	if a.managed {
		for _, d := range a.devices {
			if d.ConcurrentManagedAccess() {
				loc := driver.MemLocation{Type: driver.MemLocationTypeDevice, Id: int32(d.Ordinal())}
				if err := driver.DevicePtr(p.data).MemAdvise(smallPoolSize, driver.MemAdviseSetAccessedBy, loc); err != nil {
					return nil, fmt.Errorf("mem advise: %w", err)
				}
			}
		}
	}
	numBlocks := smallPoolSize / smallBlockSize
	p.blocks = make([]Buffer, numBlocks)
	p.freeList = make([]int32, 0, numBlocks)
	for i := numBlocks - 1; i >= 0; i-- {
		p.freeList = append(p.freeList, int32(i))
	}
	return p, nil
}

func (p *smallSizePool) malloc() *Buffer {
	if len(p.freeList) == 0 {
		return nil
	}
	i := p.freeList[len(p.freeList)-1]
	p.freeList = p.freeList[:len(p.freeList)-1]
	b := &p.blocks[i]
	*b = Buffer{
		Data:   p.data + uintptr(i)*smallBlockSize,
		Size:   smallBlockSize,
		Device: -1,
		small:  true,
		idx:    i,
	}
	return b
}

func (p *smallSizePool) free(b *Buffer) {
	p.freeList = append(p.freeList, b.idx)
}

// Allocator is the caching CUDA allocator. Port of CudaAllocator.
type Allocator struct {
	devices []*Device
	managed bool // all devices support concurrent managed access

	mu          sync.Mutex
	memoryLimit uint64
	freeLimit   uint64
	totalMemory uint64
	maxPoolSize uint64
	cache       *bufferCache
	active      uint64
	peak        uint64
	freeStreams []driver.Stream
	memPools    []driver.MemoryPool
	smallPool   *smallSizePool
}

var (
	allocOnce sync.Once
	allocator *Allocator
	allocErr  error
)

// GetAllocator returns the process-wide allocator, creating it on first
// use.
func GetAllocator() (*Allocator, error) {
	allocOnce.Do(func() {
		allocator, allocErr = newAllocator()
	})
	return allocator, allocErr
}

func newAllocator() (*Allocator, error) {
	devs, err := Devices()
	if err != nil {
		return nil, err
	}
	if len(devs) == 0 {
		return nil, fmt.Errorf("no cuda devices")
	}
	a := &Allocator{devices: devs, managed: true}
	for _, d := range devs {
		// Windows, WSL and Tegra report no concurrentManagedAccess and
		// managed memory does not work there; fall back to pinned host
		// memory for unified buffers.
		if !d.ConcurrentManagedAccess() {
			a.managed = false
		}
	}
	a.cache = newBufferCache(pageSize,
		func(b *Buffer) uint64 { return b.Size },
		func(b *Buffer) { a.freeCudaBuffer(b) })

	err = devs[0].WithCurrent(func() error {
		_, total, err := driver.MemGetInfo()
		if err != nil {
			return fmt.Errorf("mem get info: %w", err)
		}
		a.totalMemory = uint64(total)
		a.memoryLimit = uint64(float64(a.totalMemory) * 0.95)
		a.freeLimit = a.totalMemory - a.memoryLimit
		a.maxPoolSize = a.memoryLimit

		a.freeStreams = make([]driver.Stream, len(devs))
		a.memPools = make([]driver.MemoryPool, len(devs))
		for i, d := range devs {
			if !d.MemoryPools() {
				continue
			}
			if err := d.MakeCurrent(); err != nil {
				return err
			}
			s, err := driver.NewStream(uint32(driver.StreamNonBlocking))
			if err != nil {
				return fmt.Errorf("free stream: %w", err)
			}
			a.freeStreams[i] = s
			pool, err := d.Raw().GetDefaultMemPool()
			if err != nil {
				return fmt.Errorf("default mem pool: %w", err)
			}
			a.memPools[i] = pool
		}
		if err := devs[0].MakeCurrent(); err != nil {
			return err
		}
		a.smallPool, err = newSmallSizePool(a)
		return err
	})
	if err != nil {
		return nil, err
	}
	return a, nil
}

func (a *Allocator) unifiedMalloc(size uint64) (uintptr, error) {
	if a.managed {
		p, err := driver.MemAllocManaged(uint(size), uint32(driver.MemAttachGlobal))
		if err != nil {
			return 0, fmt.Errorf("malloc managed %d: %w", size, err)
		}
		return uintptr(p), nil
	}
	var p uintptr
	if err := driver.MemAllocHost(&p, uint(size)); err != nil {
		return 0, fmt.Errorf("malloc host %d: %w", size, err)
	}
	return p, nil
}

func (a *Allocator) unifiedFree(p uintptr) error {
	if a.managed {
		return driver.DevicePtr(p).MemFree()
	}
	return driver.MemFreeHost(p)
}

// roundSize maps a request to its allocation size class.
func roundSize(size uint64) uint64 {
	switch {
	case size <= smallBlockSize:
		return smallBlockSize
	case size < pageSize:
		return 1 << bits.Len64(size-1)
	default:
		return pageSize * ((size + pageSize - 1) / pageSize)
	}
}

// Malloc allocates a unified buffer.
func (a *Allocator) Malloc(size uint64) (*Buffer, error) {
	return a.MallocAsync(size, -1, driver.Stream(0))
}

// MallocAsync allocates size bytes on device, stream-ordered on stream
// when the device supports memory pools. Pass device -1 (and a zero
// stream) for unified memory.
func (a *Allocator) MallocAsync(size uint64, device int, stream driver.Stream) (*Buffer, error) {
	if size == 0 {
		return &Buffer{Device: -1}, nil
	}
	size = roundSize(size)
	if size <= smallBlockSize || stream == 0 {
		device = -1
	}

	a.mu.Lock()
	buf := a.cache.reuseFromCache(size)
	if buf == nil {
		// Under memory pressure, reclaim from the cache first.
		if used := a.active + a.cache.cacheSize() + size; used > a.memoryLimit {
			a.cache.releaseCachedBuffers(used - a.memoryLimit)
		}
		if size <= smallBlockSize {
			buf = a.smallPool.malloc()
		}
		a.mu.Unlock()
		if buf == nil {
			var data uintptr
			err := a.devices[max(device, 0)].WithCurrent(func() error {
				var err error
				if device == -1 {
					data, err = a.unifiedMalloc(size)
					return err
				}
				if a.memPools[device] != 0 {
					p, err := stream.MemAllocAsync(uint(size))
					data = uintptr(p)
					return err
				}
				p, err := driver.MemAlloc(uint(size))
				data = uintptr(p)
				return err
			})
			if err != nil {
				return nil, fmt.Errorf("malloc %d bytes: %w", size, err)
			}
			buf = &Buffer{Data: data, Size: size, Device: device}
		}
		a.mu.Lock()

		// If any device pool reserves too much, trim the cache so graph
		// and kernel execution do not fail with OOM.
		if a.cache.cacheSize() > 0 {
			for _, p := range a.memPools {
				if p == 0 {
					continue
				}
				var used uint64
				if err := p.GetAttribute(driver.MempoolAttrReservedMemCurrent, uintptr(unsafe.Pointer(&used))); err == nil {
					if used > a.totalMemory-a.freeLimit {
						a.cache.releaseCachedBuffers(a.freeLimit)
						break
					}
				}
			}
		}
	}
	a.active += buf.Size
	a.peak = max(a.active, a.peak)
	if cs := a.cache.cacheSize(); cs > a.maxPoolSize {
		a.cache.releaseCachedBuffers(cs - a.maxPoolSize)
	}
	a.mu.Unlock()

	if buf.Device >= 0 && buf.Device != device {
		if err := a.MoveToUnifiedMemory(buf, stream); err != nil {
			return nil, err
		}
	}
	return buf, nil
}

// Free returns the buffer to the cache, or releases it when the cache is
// full.
func (a *Allocator) Free(buf *Buffer) {
	if buf == nil || buf.Size == 0 {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.active -= buf.Size
	if a.cache.cacheSize() < a.maxPoolSize {
		a.cache.recycleToCache(buf)
	} else {
		a.freeCudaBuffer(buf)
	}
}

// MoveToUnifiedMemory replaces the buffer's device memory with unified
// memory and copies the data over. Pass a nonzero stream to copy
// asynchronously.
func (a *Allocator) MoveToUnifiedMemory(buf *Buffer, stream driver.Stream) error {
	if buf.Device == -1 {
		return nil
	}
	dev := a.devices[buf.Device]
	return dev.WithCurrent(func() error {
		data, err := a.unifiedMalloc(buf.Size)
		if err != nil {
			return err
		}
		src := driver.DevicePtr(buf.Data)
		if stream != 0 && a.memPools[buf.Device] != 0 {
			if err := driver.DevicePtr(data).MemcpyAsync(src, uint(buf.Size), stream); err != nil {
				return err
			}
			if err := a.freeAsync(buf, stream); err != nil {
				return err
			}
		} else {
			if err := driver.DevicePtr(data).Memcpy(src, uint(buf.Size)); err != nil {
				return err
			}
			if err := a.freeAsync(buf, 0); err != nil {
				return err
			}
		}
		buf.Data = data
		buf.Device = -1
		return nil
	})
}

// freeCudaBuffer must be called with mu held.
func (a *Allocator) freeCudaBuffer(buf *Buffer) {
	if buf.small {
		a.smallPool.free(buf)
		return
	}
	a.freeAsync(buf, 0)
}

func (a *Allocator) freeAsync(buf *Buffer, stream driver.Stream) error {
	if buf.Device == -1 {
		return a.devices[0].WithCurrent(func() error {
			return a.unifiedFree(buf.Data)
		})
	}
	return a.devices[buf.Device].WithCurrent(func() error {
		if a.memPools[buf.Device] != 0 {
			s := stream
			if s == 0 {
				s = a.freeStreams[buf.Device]
			}
			return driver.DevicePtr(buf.Data).MemFreeAsync(s)
		}
		return driver.DevicePtr(buf.Data).MemFree()
	})
}

// ActiveMemory returns the bytes currently allocated to live buffers.
func (a *Allocator) ActiveMemory() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.active
}

// PeakMemory returns the maximum of ActiveMemory since the last reset.
func (a *Allocator) PeakMemory() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.peak
}

// ResetPeakMemory sets the peak memory counter to zero.
func (a *Allocator) ResetPeakMemory() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.peak = 0
}

// MemoryLimit returns the allocation limit in bytes.
func (a *Allocator) MemoryLimit() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.memoryLimit
}

// SetMemoryLimit sets the allocation limit and returns the old one.
func (a *Allocator) SetMemoryLimit(limit uint64) uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	old := a.memoryLimit
	a.memoryLimit = limit
	return old
}

// CacheMemory returns the bytes held in the buffer cache.
func (a *Allocator) CacheMemory() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.cache.cacheSize()
}

// SetCacheLimit sets the cache size limit and returns the old one.
func (a *Allocator) SetCacheLimit(limit uint64) uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	old := a.maxPoolSize
	a.maxPoolSize = limit
	return old
}

// ClearCache releases all cached buffers.
func (a *Allocator) ClearCache() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.cache.clear()
}
