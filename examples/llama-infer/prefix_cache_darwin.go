//go:build darwin

package main

import (
	"encoding/binary"
	"sync"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalkv"
)

// PrefixCacheMetrics reports cache activity from actual KV snapshot restores.
type PrefixCacheMetrics struct {
	Entries     int
	Bytes       int64
	Hits        int64
	Misses      int64
	SavedTokens int64
}

type prefixSnapshot struct {
	tokens   int
	current  int32
	k        [][]float32
	v        [][]float32
	bytes    int64
	lastUsed time.Time
}

type prefixCache struct {
	mu sync.Mutex

	radix       *metalkv.RadixTree
	blockMap    map[int]*prefixSnapshot
	tokenBlocks map[string][]int
	entries     map[string]*prefixSnapshot
	maxBytes    int64
	maxItems    int
	bytes       int64
	hits        int64
	misses      int64
	saved       int64
	nextBlock   int
}

func newPrefixCache(maxBytes int64, maxItems int) *prefixCache {
	return &prefixCache{
		radix:       metalkv.NewRadixTree(maxItems*4, 1),
		blockMap:    make(map[int]*prefixSnapshot),
		tokenBlocks: make(map[string][]int),
		entries:     make(map[string]*prefixSnapshot),
		maxBytes:    maxBytes,
		maxItems:    maxItems,
	}
}

func prefixKey(tokens []int32) string {
	data := make([]byte, 4*len(tokens))
	for i, token := range tokens {
		binary.LittleEndian.PutUint32(data[4*i:], uint32(token))
	}
	return string(data)
}

func (c *prefixCache) load(tokens []int32) (*prefixSnapshot, int, bool) {
	if c == nil || len(tokens) == 0 {
		return nil, 0, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.radix != nil {
		matchedCount, matchedBlocks, _ := c.radix.MatchPrefix(tokens)
		if matchedCount > 0 && len(matchedBlocks) > 0 {
			lastBlock := matchedBlocks[len(matchedBlocks)-1]
			if entry, ok := c.blockMap[lastBlock]; ok && entry != nil {
				entry.lastUsed = time.Now()
				c.hits++
				c.saved += int64(matchedCount)
				return entry, matchedCount, true
			}
		}
	}

	for count := len(tokens); count > 0; count-- {
		entry := c.entries[prefixKey(tokens[:count])]
		if entry == nil {
			continue
		}
		entry.lastUsed = time.Now()
		c.hits++
		c.saved += int64(count)
		return entry, count, true
	}
	c.misses++
	return nil, 0, false
}

func (c *prefixCache) evictLocked(key string, snap *prefixSnapshot) {
	delete(c.entries, key)
	c.bytes -= snap.bytes
	if blocks, ok := c.tokenBlocks[key]; ok {
		for _, b := range blocks {
			delete(c.blockMap, b)
		}
		delete(c.tokenBlocks, key)
	}
}

func (c *prefixCache) store(tokens []int32, snapshot *prefixSnapshot) {
	if c == nil || len(tokens) == 0 || snapshot == nil || snapshot.bytes <= 0 || snapshot.bytes > c.maxBytes {
		return
	}
	key := prefixKey(tokens)
	c.mu.Lock()
	defer c.mu.Unlock()
	if previous := c.entries[key]; previous != nil {
		c.evictLocked(key, previous)
	}
	for (len(c.entries) >= c.maxItems || c.bytes+snapshot.bytes > c.maxBytes) && len(c.entries) > 0 {
		var oldestKey string
		var oldest *prefixSnapshot
		for candidateKey, candidate := range c.entries {
			if oldest == nil || candidate.lastUsed.Before(oldest.lastUsed) {
				oldestKey, oldest = candidateKey, candidate
			}
		}
		c.evictLocked(oldestKey, oldest)
	}
	if c.bytes+snapshot.bytes > c.maxBytes {
		return
	}
	snapshot.lastUsed = time.Now()
	c.entries[key] = snapshot
	c.bytes += snapshot.bytes

	if c.radix != nil {
		blockID := c.nextBlock + 1
		c.nextBlock += len(tokens)
		blocks := make([]int, len(tokens))
		for i := range blocks {
			blocks[i] = blockID + i
			c.blockMap[blocks[i]] = snapshot
		}
		c.tokenBlocks[key] = blocks
		_ = c.radix.Insert(tokens, blocks)
	}
}

func (c *prefixCache) metrics() PrefixCacheMetrics {
	if c == nil {
		return PrefixCacheMetrics{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return PrefixCacheMetrics{
		Entries:     len(c.entries),
		Bytes:       c.bytes,
		Hits:        c.hits,
		Misses:      c.misses,
		SavedTokens: c.saved,
	}
}

func (c *prefixCache) reset() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.radix = metalkv.NewRadixTree(c.maxItems*4, 1)
	c.blockMap = make(map[int]*prefixSnapshot)
	c.tokenBlocks = make(map[string][]int)
	c.entries = make(map[string]*prefixSnapshot)
	c.bytes = 0
	c.hits = 0
	c.misses = 0
	c.saved = 0
	c.nextBlock = 0
}
