// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"fmt"
	"math"
	"sync"
)

// SlidingWindowConfig configures an attention sink and rolling window KV buffer.
type SlidingWindowConfig struct {
	SinkTokens int // Number of initial tokens preserved indefinitely (typically 4)
	WindowSize int // Number of rolling window tokens preserved (e.g. 4096)
}

// RollingKVCache manages a fixed-size ring buffer for unbounded streaming attention with attention sinks.
type RollingKVCache struct {
	sinkTokens int
	windowSize int
	capacity   int
	totalSeen  int
	mu         sync.RWMutex
}

// NewRollingKVCache creates a sliding window KV cache preserving sink tokens and rolling window.
func NewRollingKVCache(cfg SlidingWindowConfig) (*RollingKVCache, error) {
	if cfg.SinkTokens < 0 || cfg.WindowSize <= 0 {
		return nil, fmt.Errorf("cudakv: invalid sliding window config: sink=%d, window=%d", cfg.SinkTokens, cfg.WindowSize)
	}
	return &RollingKVCache{
		sinkTokens: cfg.SinkTokens,
		windowSize: cfg.WindowSize,
		capacity:   cfg.SinkTokens + cfg.WindowSize,
		totalSeen:  0,
	}, nil
}

// PhysicalSlot maps a logical sequence position to its physical buffer slot in
// the ring buffer. It returns -1 for a negative position.
//
// The mapping is many-to-one: every windowSize positions share a slot, so a
// slot holds the requested position only while that position is resident. Use
// Slot to get the mapping and that check together.
func (c *RollingKVCache) PhysicalSlot(logicalPos int) int {
	if logicalPos < 0 {
		return -1
	}

	// Initial sink tokens occupy fixed slots 0..sinkTokens-1
	if logicalPos < c.sinkTokens {
		return logicalPos
	}

	// Rolling tokens map into ring buffer slots [sinkTokens .. sinkTokens + windowSize - 1]
	offset := logicalPos - c.sinkTokens
	slot := c.sinkTokens + (offset % c.windowSize)
	return slot
}

// WindowStart returns the oldest logical position still held in the rolling
// window. Positions below it have been overwritten, except for the sink.
func (c *RollingKVCache) WindowStart() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.windowStart()
}

// windowStart requires c.mu to be held.
func (c *RollingKVCache) windowStart() int {
	if start := c.totalSeen - c.windowSize; start > c.sinkTokens {
		return start
	}
	return c.sinkTokens
}

// Resident reports whether logicalPos is still addressable. A position is
// resident if it has been written, and either lies in the sink or has not yet
// been lapped by the rolling window.
func (c *RollingKVCache) Resident(logicalPos int) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if logicalPos < 0 || logicalPos >= c.totalSeen {
		return false
	}
	if logicalPos < c.sinkTokens {
		return true
	}
	return logicalPos >= c.windowStart()
}

// Slot returns the physical slot holding logicalPos, and whether that slot
// still holds it. Callers must not read the slot when ok is false: the ring
// has reused it for a later position.
func (c *RollingKVCache) Slot(logicalPos int) (slot int, ok bool) {
	if !c.Resident(logicalPos) {
		return -1, false
	}
	return c.PhysicalSlot(logicalPos), true
}

// Advance records that n new tokens have been written into the rolling cache.
func (c *RollingKVCache) Advance(n int) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if n < 0 || n > math.MaxInt-c.totalSeen {
		return fmt.Errorf("cudakv: invalid advance %d from %d", n, c.totalSeen)
	}
	c.totalSeen += n
	return nil
}

// ActiveTokens returns the total number of currently active tokens.
func (c *RollingKVCache) ActiveTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.totalSeen <= c.capacity {
		return c.totalSeen
	}
	return c.capacity
}

// TotalSeen returns the total number of tokens processed across the stream.
func (c *RollingKVCache) TotalSeen() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.totalSeen
}
