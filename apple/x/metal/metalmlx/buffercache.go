package metalmlx

// bufferCache holds freed buffers for reuse, keyed by size, evicting in
// least-recently-inserted order. Port of mlx/backend/common/buffer_cache.h
// with the multimap replaced by a size-sorted bucket map.
type bufferCache[T comparable] struct {
	pageSize uint64
	getSize  func(T) uint64
	free     func(T)

	pool     map[uint64][]*holder[T] // size -> holders, most recent last
	sizes    []uint64                // sorted keys of pool
	head     *holder[T]
	tail     *holder[T]
	poolSize uint64
}

type holder[T comparable] struct {
	prev, next *holder[T]
	buf        T
	size       uint64
}

func newBufferCache[T comparable](pageSize uint64, getSize func(T) uint64, free func(T)) *bufferCache[T] {
	return &bufferCache[T]{
		pageSize: pageSize,
		getSize:  getSize,
		free:     free,
		pool:     make(map[uint64][]*holder[T]),
	}
}

// reuseFromCache returns a cached buffer of at least size bytes, if there is one.
// A buffer is only reused when its size is below min(2*size, size+2*pages).
func (c *bufferCache[T]) reuseFromCache(size uint64) (T, bool) {
	var zero T
	limit := min(2*size, size+2*c.pageSize)
	for _, s := range c.sizes {
		if s < size {
			continue
		}
		if s >= limit {
			return zero, false
		}
		hs := c.pool[s]
		h := hs[len(hs)-1]
		c.removeBucket(s, h)
		c.removeFromList(h)
		c.poolSize -= s
		return h.buf, true
	}
	return zero, false
}

func (c *bufferCache[T]) recycleToCache(buf T) {
	h := &holder[T]{buf: buf, size: c.getSize(buf)}
	c.addAtHead(h)
	if _, ok := c.pool[h.size]; !ok {
		c.insertSize(h.size)
	}
	c.pool[h.size] = append(c.pool[h.size], h)
	c.poolSize += h.size
}

// releaseCachedBuffers frees least-recently-cached buffers until at least
// minBytesToFree bytes are freed, and returns the number released.
func (c *bufferCache[T]) releaseCachedBuffers(minBytesToFree uint64) int {
	if float64(minBytesToFree) >= 0.9*float64(c.poolSize) {
		return c.clear()
	}
	n := 0
	var freed uint64
	for c.tail != nil && freed < minBytesToFree {
		h := c.tail
		freed += h.size
		c.free(h.buf)
		n++
		c.removeBucket(h.size, h)
		c.removeFromList(h)
	}
	c.poolSize -= freed
	return n
}

func (c *bufferCache[T]) clear() int {
	n := 0
	for h := c.head; h != nil; h = h.next {
		c.free(h.buf)
		n++
	}
	c.pool = make(map[uint64][]*holder[T])
	c.sizes = c.sizes[:0]
	c.head, c.tail = nil, nil
	c.poolSize = 0
	return n
}

func (c *bufferCache[T]) cacheSize() uint64 {
	return c.poolSize
}

func (c *bufferCache[T]) insertSize(size uint64) {
	i := 0
	for i < len(c.sizes) && c.sizes[i] < size {
		i++
	}
	c.sizes = append(c.sizes, 0)
	copy(c.sizes[i+1:], c.sizes[i:])
	c.sizes[i] = size
}

func (c *bufferCache[T]) removeBucket(size uint64, h *holder[T]) {
	hs := c.pool[size]
	for i, e := range hs {
		if e == h {
			hs = append(hs[:i], hs[i+1:]...)
			break
		}
	}
	if len(hs) == 0 {
		delete(c.pool, size)
		for i, s := range c.sizes {
			if s == size {
				c.sizes = append(c.sizes[:i], c.sizes[i+1:]...)
				break
			}
		}
	} else {
		c.pool[size] = hs
	}
}

func (c *bufferCache[T]) addAtHead(h *holder[T]) {
	if c.head == nil {
		c.head, c.tail = h, h
		return
	}
	c.head.prev = h
	h.next = c.head
	c.head = h
}

func (c *bufferCache[T]) removeFromList(h *holder[T]) {
	if h.prev != nil {
		h.prev.next = h.next
	} else {
		c.head = h.next
	}
	if h.next != nil {
		h.next.prev = h.prev
	} else {
		c.tail = h.prev
	}
	h.prev, h.next = nil, nil
}
