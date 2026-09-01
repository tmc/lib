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

	radix     *metalkv.RadixTree
	blockMap  map[int]*prefixSnapshot
	entries   map[string]*prefixSnapshot
	maxBytes  int64
	maxItems  int
	bytes     int64
	hits      int64
	misses    int64
	saved     int64
	nextBlock int
}

func newPrefixCache(maxBytes int64, maxItems int) *prefixCache {
	return &prefixCache{
		radix:    metalkv.NewRadixTree(maxItems*4, 1),
		blockMap: make(map[int]*prefixSnapshot),
		entries:  make(map[string]*prefixSnapshot),
		maxBytes: maxBytes,
		maxItems: maxItems,
	}
}

func prefixKey(tokens []int32) string {
	data := make([]byte, 4*len(tokens))
	for i, token := range tokens {
		binary.LittleEndian.PutUint32(data[4*i:], uint32(token))
	}
	return string(data)
}

func (c *prefixCache) load(tokens []int32) (*prefixSnapshot, bool) {
	if c == nil || len(tokens) == 0 {
		return nil, false
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
				return entry, true
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
		return entry, true
	}
	c.misses++
	return nil, false
}

func (c *prefixCache) store(tokens []int32, snapshot *prefixSnapshot) {
	if c == nil || len(tokens) == 0 || snapshot == nil || snapshot.bytes <= 0 || snapshot.bytes > c.maxBytes {
		return
	}
	key := prefixKey(tokens)
	c.mu.Lock()
	defer c.mu.Unlock()
	if previous := c.entries[key]; previous != nil {
		c.bytes -= previous.bytes
	}
	for (len(c.entries) >= c.maxItems || c.bytes+snapshot.bytes > c.maxBytes) && len(c.entries) > 0 {
		var oldestKey string
		var oldest *prefixSnapshot
		for candidateKey, candidate := range c.entries {
			if oldest == nil || candidate.lastUsed.Before(oldest.lastUsed) {
				oldestKey, oldest = candidateKey, candidate
			}
		}
		delete(c.entries, oldestKey)
		c.bytes -= oldest.bytes
	}
	if c.bytes+snapshot.bytes > c.maxBytes {
		return
	}
	snapshot.lastUsed = time.Now()
	c.entries[key] = snapshot
	c.bytes += snapshot.bytes

	if c.radix != nil {
		c.nextBlock++
		blockID := c.nextBlock
		blocks := make([]int, len(tokens))
		for i := range blocks {
			blocks[i] = blockID + i
			c.blockMap[blocks[i]] = snapshot
		}
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
