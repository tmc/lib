package cuptiactivity

import (
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/cupti"
)

// collection receives completed buffers. Native callbacks may run concurrently;
// records and diagnostics are retained independently of callback buffer memory.
// The record limit bounds host memory even when a consumer does not drain it.
type collection struct {
	mu                                             sync.Mutex
	pool                                           *bufferPool
	layouts                                        map[cupti.ActivityKind]uint
	next                                           recordIterator
	name                                           func(uintptr) (string, error)
	dropped                                        func(cupti.Context, uint32) (uint, error)
	limit                                          int
	records                                        []Record
	incomplete, discarded, nativeDropped, failures uint64
	firstError                                     error
}

func (c *collection) complete(context cupti.Context, stream uint32, ptr uintptr, size, valid uint) {
	err := c.pool.finish(ptr, size, valid, func(data []byte) error {
		return decodeBuffer(data, c.layouts, c.next, c.name, func(r Record) error {
			c.mu.Lock()
			defer c.mu.Unlock()
			if !r.Complete {
				c.incomplete++
			}
			if len(c.records) >= c.limit {
				c.discarded++
				return nil // Keep decoding to return the whole buffer safely.
			}
			c.records = append(c.records, r)
			return nil
		})
	})
	c.recordError(err)
	// CUPTI resets this native count on each query, so accumulate rather than
	// replacing it. Query even if decoding failed or the buffer was empty.
	n, err := c.dropped(context, stream)
	if err != nil {
		c.recordError(fmt.Errorf("cuptiactivity: dropped records: %w", err))
		return
	}
	c.mu.Lock()
	c.nativeDropped += uint64(n)
	c.mu.Unlock()
}

func (c *collection) recordError(err error) {
	if err == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failures++
	if c.firstError == nil {
		c.firstError = err
	}
}
