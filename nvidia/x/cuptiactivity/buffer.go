package cuptiactivity

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// bufferPool retains every buffer handed to CUPTI until its completion callback
// has consumed the records. Go pinning is sufficient here: these are host-side
// activity buffers, not application CUDA DMA allocations.
type bufferPool struct {
	mu      sync.Mutex
	bytes   int
	limit   int
	buffers map[uintptr]*activityBuffer
	refused uint64
}

type activityBuffer struct {
	words      []uint64
	pin        runtime.Pinner
	processing bool
}

func newBufferPool(bytes, limit int) (*bufferPool, error) {
	if bytes < 8 || bytes%8 != 0 {
		return nil, errors.New("cuptiactivity: buffer size must be a positive multiple of eight")
	}
	if limit <= 0 || bytes > int(^uint(0)>>1)/limit {
		return nil, errors.New("cuptiactivity: invalid buffer limit")
	}
	return &bufferPool{bytes: bytes, limit: limit, buffers: make(map[uintptr]*activityBuffer)}, nil
}

// request returns an aligned host buffer, or zero when the explicit allocation
// budget is exhausted. Refusals must be reported alongside CUPTI dropped counts.
func (p *bufferPool) request() (uintptr, uint) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.buffers) >= p.limit {
		p.refused++
		return 0, 0
	}
	b := &activityBuffer{words: make([]uint64, p.bytes/8)}
	b.pin.Pin(&b.words[0])
	ptr := uintptr(unsafe.Pointer(&b.words[0]))
	p.buffers[ptr] = b
	return ptr, uint(p.bytes)
}

// finish invokes consume only after bounds and ownership checks. The byte view
// is borrowed for consume's duration; records with pointers into it must be
// copied before consume returns. Pending includes callbacks still consuming.
func (p *bufferPool) finish(ptr uintptr, size, valid uint, consume func([]byte) error) error {
	p.mu.Lock()
	b := p.buffers[ptr]
	if b == nil {
		p.mu.Unlock()
		return errors.New("cuptiactivity: completion returned an unknown buffer")
	}
	if b.processing {
		p.mu.Unlock()
		return errors.New("cuptiactivity: duplicate buffer completion")
	}
	b.processing = true
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		b.pin.Unpin()
		delete(p.buffers, ptr)
	}()
	if size != uint(p.bytes) || valid > size {
		return fmt.Errorf("cuptiactivity: invalid completed buffer bounds size=%d valid=%d", size, valid)
	}
	if valid == 0 {
		return nil
	}
	if consume == nil {
		return errors.New("cuptiactivity: missing activity consumer")
	}
	return consume(unsafe.Slice((*byte)(unsafe.Pointer(&b.words[0])), int(valid)))
}

func (p *bufferPool) status() (pending int, refused uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.buffers), p.refused
}
