// SPDX-License-Identifier: Apache-2.0

package metalkv

import (
	"fmt"
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
		return nil, fmt.Errorf("metalkv: invalid sliding window config: sink=%d, window=%d", cfg.SinkTokens, cfg.WindowSize)
	}
	return &RollingKVCache{
		sinkTokens: cfg.SinkTokens,
		windowSize: cfg.WindowSize,
		capacity:   cfg.SinkTokens + cfg.WindowSize,
		totalSeen:  0,
	}, nil
}

// PhysicalSlot maps a logical sequence position to its physical buffer slot in the ring buffer.
func (c *RollingKVCache) PhysicalSlot(logicalPos int) int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Initial sink tokens occupy fixed slots 0..sinkTokens-1
	if logicalPos < c.sinkTokens {
		return logicalPos
	}

	// Rolling tokens map into ring buffer slots [sinkTokens .. sinkTokens + windowSize - 1]
	offset := logicalPos - c.sinkTokens
	slot := c.sinkTokens + (offset % c.windowSize)
	return slot
}

// Advance records that new tokens have been written into the rolling cache.
func (c *RollingKVCache) Advance(n int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.totalSeen += n
}

// ActiveTokens returns the total number of currently active tokens (at most SinkTokens + WindowSize).
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
