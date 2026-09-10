package main

// A prefix cache over the paged KV pool, behind LLAMA_INFER_PAGED_PREFIX=1.
//
// UNQUALIFIED. No GPU has run this. It needs the paged path
// (LLAMA_INFER_PAGED_KV=1, itself unqualified) and is refused without it. The
// join sequencing here is host-tested against a fake pool
// (prefix_cache_paged_test.go); the test that would qualify it on a device,
// TestPagedPrefixServingReuse in serving_prefix_linux_test.go, has never been
// executed by its author.
//
// What it does. When a serving request finishes, the whole blocks its sequence
// wrote are handed to a cudakv.RadixTree keyed by the tokens they hold, and
// retained in the pool on the tree's behalf. A later request whose prompt
// begins with a stored sequence starts on those blocks and computes only the
// rest, its last prompt token always included so that prefill still produces
// a next token. The tree is bounded in segments; the pool is bounded in
// blocks, and the cache gives blocks back, least recently matched first, when
// the pool runs short.
//
// This file has no build tag: nothing in it touches a device. The pool is an
// interface that *cudakv.PagedCache satisfies; the device half is the serving
// backend in serving_backend_linux.go and the sequence constructor in
// paged_kv_step_cuda.go.

import (
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

// pagedPrefixEnabled reports whether the paged prefix cache was explicitly
// requested. It is opt-in through the environment like the paged path itself.
func pagedPrefixEnabled() bool { return os.Getenv("LLAMA_INFER_PAGED_PREFIX") == "1" }

// defaultPagedPrefixNodes bounds the tree in stored segments. Blocks, the
// scarce resource, are bounded by the pool and reclaimed on demand, so this
// only caps bookkeeping.
const defaultPagedPrefixNodes = 256

// prefixPool is the part of cudakv.PagedCache the prefix cache needs: holder
// counting on blocks and sequence tables. The fake in the test is the other
// implementation.
type prefixPool interface {
	Retain(blocks []int) error
	Release(blocks []int) error
	NumFreeBlocks() int
}

// pagedPrefixCache maps token prefixes to pool blocks that already hold their
// keys and values. Safe for concurrent use; every method takes the lock.
type pagedPrefixCache struct {
	mu         sync.Mutex
	pool       prefixPool
	tree       *cudakv.RadixTree
	blockSize  int
	blockBytes int64

	// retained counts blocks the pool holds for the tree: taken by Insert and
	// not yet given back by an eviction.
	retained int
	hits     int64
	misses   int64
	saved    int64

	// err is the first bookkeeping failure. After one the tree and the pool
	// may disagree about who holds what, so the cache refuses further joins
	// rather than hand out blocks it cannot vouch for.
	err error
}

// newPagedPrefixCache builds a cache over pool for blocks of blockSize tokens
// and blockBytes bytes each, storing at most maxNodes segments.
func newPagedPrefixCache(pool prefixPool, blockSize, maxNodes int, blockBytes int64) (*pagedPrefixCache, error) {
	if pool == nil {
		return nil, errors.New("paged prefix: nil pool")
	}
	if blockSize <= 0 || maxNodes <= 0 || blockBytes <= 0 {
		return nil, fmt.Errorf("paged prefix: invalid shape (block size %d, nodes %d, block bytes %d)", blockSize, maxNodes, blockBytes)
	}
	return &pagedPrefixCache{pool: pool, tree: cudakv.NewRadixTree(maxNodes, blockSize), blockSize: blockSize, blockBytes: blockBytes}, nil
}

func (c *pagedPrefixCache) fail(err error) error {
	if c.err == nil {
		c.err = fmt.Errorf("paged prefix: %w", err)
	}
	return c.err
}

// join looks up the longest stored prefix of prompt that leaves at least the
// last token to compute, and pins it. It returns the blocks holding the
// prefix, how many tokens they cover, always a whole number of blocks, and the
// node to pass to leave. A miss returns no blocks and a nil node.
func (c *pagedPrefixCache) join(prompt []int32) (blocks []int, matched int, node *cudakv.RadixNode, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err != nil {
		return nil, 0, nil, c.err
	}
	if len(prompt) >= 2 {
		matched, blocks, node = c.tree.Acquire(prompt[:len(prompt)-1])
	}
	if matched == 0 {
		c.misses++
		return nil, 0, nil, nil
	}
	c.hits++
	c.saved += int64(matched)
	return append([]int(nil), blocks...), matched, node, nil
}

// abandon undoes a join whose sequence was never started.
func (c *pagedPrefixCache) abandon(node *cudakv.RadixNode) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tree.Release(node)
}

// leave stores a finished sequence: tokens are every token whose keys and
// values the sequence wrote, in position order, and table is its block table,
// which must still be live in the pool. The whole blocks are inserted; blocks
// the tree takes are retained on its behalf, and blocks the tree gives up to
// make room are released. The join's node is unpinned last, so the sequence's
// own path cannot be what is evicted to make room for it.
func (c *pagedPrefixCache) leave(node *cudakv.RadixNode, tokens []int32, table []int) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	defer c.tree.Release(node)
	if c.err != nil {
		return c.err
	}
	n := len(tokens) / c.blockSize
	if n == 0 {
		return nil
	}
	if len(table) < n {
		return c.fail(fmt.Errorf("sequence wrote %d tokens but holds %d blocks of %d", len(tokens), len(table), c.blockSize))
	}
	taken, evicted, err := c.tree.Insert(tokens[:n*c.blockSize], table[:n])
	if err != nil {
		return c.fail(err)
	}
	if err := c.pool.Retain(taken); err != nil {
		return c.fail(err)
	}
	c.retained += len(taken)
	if err := c.pool.Release(evicted); err != nil {
		return c.fail(err)
	}
	c.retained -= len(evicted)
	return nil
}

// ensureFree gives blocks back to the pool, least recently matched segment
// first, until at least n are free. It reports whether it got there: pinned
// segments, those a live sequence joined on, are never evicted.
func (c *pagedPrefixCache) ensureFree(n int) (bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err != nil {
		return false, c.err
	}
	for c.pool.NumFreeBlocks() < n {
		freed := c.tree.Evict(1)
		if len(freed) == 0 {
			return false, nil
		}
		if err := c.pool.Release(freed); err != nil {
			return false, c.fail(err)
		}
		c.retained -= len(freed)
	}
	return true, nil
}

// reset drops every stored segment not pinned by a live sequence and zeroes
// the counters. A benchmark calls it between samples so that no sample reuses
// what an earlier one computed.
func (c *pagedPrefixCache) reset() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err != nil {
		return c.err
	}
	for {
		freed := c.tree.Evict(c.tree.Len())
		if len(freed) == 0 {
			break
		}
		if err := c.pool.Release(freed); err != nil {
			return c.fail(err)
		}
		c.retained -= len(freed)
	}
	c.hits, c.misses, c.saved = 0, 0, 0
	return nil
}

func (c *pagedPrefixCache) metrics() PrefixCacheMetrics {
	c.mu.Lock()
	defer c.mu.Unlock()
	return PrefixCacheMetrics{
		Entries:     c.tree.Len(),
		Bytes:       int64(c.retained) * c.blockBytes,
		Hits:        c.hits,
		Misses:      c.misses,
		SavedTokens: c.saved,
	}
}
