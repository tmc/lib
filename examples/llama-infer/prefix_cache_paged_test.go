package main

import (
	"fmt"
	"reflect"
	"testing"
)

// fakePool counts holders on blocks the way cudakv's allocator does, so the
// cache's Retain and Release calls can be checked without a device.
type fakePool struct {
	holders []int
	free    []int
	next    int64
	tables  map[int64][]int
}

func newFakePool(total int) *fakePool {
	p := &fakePool{holders: make([]int, total), tables: make(map[int64][]int)}
	for b := total - 1; b >= 0; b-- {
		p.free = append(p.free, b)
	}
	return p
}

func (p *fakePool) held(b int) error {
	if b < 0 || b >= len(p.holders) {
		return fmt.Errorf("block %d out of range", b)
	}
	if p.holders[b] == 0 {
		return fmt.Errorf("block %d is free", b)
	}
	return nil
}

func (p *fakePool) Retain(blocks []int) error {
	for _, b := range blocks {
		if err := p.held(b); err != nil {
			return err
		}
	}
	for _, b := range blocks {
		p.holders[b]++
	}
	return nil
}

func (p *fakePool) Release(blocks []int) error {
	for _, b := range blocks {
		if err := p.held(b); err != nil {
			return err
		}
	}
	for _, b := range blocks {
		if p.holders[b]--; p.holders[b] == 0 {
			p.free = append(p.free, b)
		}
	}
	return nil
}

func (p *fakePool) NumFreeBlocks() int { return len(p.free) }

// alloc starts a sequence on prefix blocks plus enough fresh ones for tokens.
func (p *fakePool) alloc(t *testing.T, prefix []int, tokens, blockSize int) (int64, []int) {
	t.Helper()
	need := (tokens+blockSize-1)/blockSize - len(prefix)
	if need > len(p.free) {
		t.Fatalf("alloc %d tokens on %d prefix blocks: %d fresh blocks needed, %d free", tokens, len(prefix), need, len(p.free))
	}
	table := append([]int(nil), prefix...)
	for _, b := range prefix {
		p.holders[b]++
	}
	for i := 0; i < need; i++ {
		b := p.free[len(p.free)-1]
		p.free = p.free[:len(p.free)-1]
		p.holders[b] = 1
		table = append(table, b)
	}
	p.next++
	p.tables[p.next] = table
	return p.next, table
}

func (p *fakePool) freeSeq(id int64) {
	for _, b := range p.tables[id] {
		if p.holders[b]--; p.holders[b] == 0 {
			p.free = append(p.free, b)
		}
	}
	delete(p.tables, id)
}

func (p *fakePool) live() int {
	n := 0
	for _, h := range p.holders {
		if h > 0 {
			n++
		}
	}
	return n
}

func seq(n int, from int32) []int32 {
	s := make([]int32, n)
	for i := range s {
		s[i] = from + int32(i)
	}
	return s
}

func TestPagedPrefixSwitch(t *testing.T) {
	t.Setenv("LLAMA_INFER_PAGED_PREFIX", "")
	if pagedPrefixEnabled() {
		t.Fatal("enabled with the variable unset")
	}
	t.Setenv("LLAMA_INFER_PAGED_PREFIX", "1")
	if !pagedPrefixEnabled() {
		t.Fatal("not enabled by 1")
	}
	if _, err := newPagedPrefixCache(nil, 4, 8, 1); err == nil {
		t.Fatal("nil pool accepted")
	}
	if _, err := newPagedPrefixCache(newFakePool(1), 0, 8, 1); err == nil {
		t.Fatal("zero block size accepted")
	}
}

// TestPagedPrefixJoinLeave walks one request through: a miss, a full prefill,
// a leave that stores the whole blocks, then a second request that starts on
// them and computes only its tail.
func TestPagedPrefixJoinLeave(t *testing.T) {
	const bs = 4
	pool := newFakePool(16)
	c, err := newPagedPrefixCache(pool, bs, 8, 100)
	if err != nil {
		t.Fatal(err)
	}

	prompt := seq(10, 1) // 2 whole blocks and a half
	blocks, matched, node, err := c.join(prompt)
	if err != nil || blocks != nil || matched != 0 || node != nil {
		t.Fatalf("join on an empty cache = %v, %d, %v, %v; want a miss", blocks, matched, node, err)
	}
	// Two decoded tokens land at positions 10 and 11: the sequence wrote 12,
	// which its third block still covers.
	written := append(append([]int32(nil), prompt...), 200, 201)
	id, table := pool.alloc(t, nil, len(written), bs)
	if err := c.leave(node, written, table); err != nil {
		t.Fatal(err)
	}
	pool.freeSeq(id)
	m := c.metrics()
	if m.Entries != 1 || m.Misses != 1 || m.Hits != 0 || m.Bytes != 3*100 {
		t.Fatalf("after one leave: %+v", m)
	}
	if got := pool.live(); got != 3 {
		t.Fatalf("%d blocks live after the sequence ended, want the 3 the tree kept", got)
	}

	// A prompt extending the stored sequence reuses all three blocks; its
	// remaining tokens, 12..14, are what prefill computes.
	prompt2 := append(append([]int32(nil), written...), 300, 301, 302)
	blocks, matched, node, err = c.join(prompt2)
	if err != nil {
		t.Fatal(err)
	}
	if want := table[:3]; !reflect.DeepEqual(blocks, want) || matched != 12 || node == nil {
		t.Fatalf("join = %v, %d, %v; want %v, 12, a node", blocks, matched, node, want)
	}
	m = c.metrics()
	if m.Hits != 1 || m.SavedTokens != 12 {
		t.Fatalf("after one hit: %+v", m)
	}
	// It will write its 15 prompt tokens plus one decode: 16, a fourth block.
	written2 := append(append([]int32(nil), prompt2...), 400)
	id2, table2 := pool.alloc(t, blocks, len(written2), bs)
	for _, b := range blocks {
		if pool.holders[b] != 2 {
			t.Fatalf("shared block %d has %d holders, want the tree and the sequence", b, pool.holders[b])
		}
	}
	// While joined, the shared path cannot be evicted from under it.
	if ok, err := c.ensureFree(len(pool.holders)); err != nil || ok {
		t.Fatalf("ensureFree(all) while a sequence is joined = %v, %v; want false", ok, err)
	}
	for _, b := range blocks {
		if pool.holders[b] != 2 {
			t.Fatalf("ensureFree dropped the tree's hold on shared block %d: %d holders", b, pool.holders[b])
		}
	}
	if err := c.leave(node, written2, table2); err != nil {
		t.Fatal(err)
	}
	pool.freeSeq(id2)
	m = c.metrics()
	if m.Entries != 2 || m.Bytes != 4*100 {
		t.Fatalf("after the extension left: %+v", m)
	}

	// Reset gives every block back.
	if err := c.reset(); err != nil {
		t.Fatal(err)
	}
	if got := pool.live(); got != 0 {
		t.Fatalf("%d blocks live after reset, want 0", got)
	}
	if m := c.metrics(); m != (PrefixCacheMetrics{}) {
		t.Fatalf("metrics after reset: %+v", m)
	}
}

// TestPagedPrefixLastTokenIsComputed pins the rule that a join never covers the
// whole prompt: prefill must still run at least the last token.
func TestPagedPrefixLastTokenIsComputed(t *testing.T) {
	const bs = 2
	pool := newFakePool(8)
	c, err := newPagedPrefixCache(pool, bs, 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	tokens := seq(4, 1)
	id, table := pool.alloc(t, nil, len(tokens), bs)
	if err := c.leave(nil, tokens, table); err != nil {
		t.Fatal(err)
	}
	pool.freeSeq(id)
	for _, test := range []struct {
		prompt []int32
		want   int
	}{
		{seq(4, 1), 2}, // whole prompt stored: keep the last block to compute
		{seq(5, 1), 4}, // one past the stored sequence: reuse all of it
		{seq(3, 1), 2},
		{seq(2, 1), 0}, // only one token would be left, which is not a whole block
		{seq(1, 1), 0},
		{nil, 0},
	} {
		_, matched, node, err := c.join(test.prompt)
		if err != nil {
			t.Fatal(err)
		}
		if matched != test.want {
			t.Errorf("join(%v) matched %d tokens, want %d", test.prompt, matched, test.want)
		}
		c.abandon(node)
	}
}

// TestPagedPrefixEnsureFree checks that pool pressure evicts stored segments
// least recently matched first and stops at what live sequences pin.
func TestPagedPrefixEnsureFree(t *testing.T) {
	const bs = 2
	pool := newFakePool(6)
	c, err := newPagedPrefixCache(pool, bs, 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	store := func(tokens []int32) []int {
		id, table := pool.alloc(t, nil, len(tokens), bs)
		if err := c.leave(nil, tokens, table); err != nil {
			t.Fatal(err)
		}
		pool.freeSeq(id)
		return table
	}
	a := store([]int32{1, 1, 1, 1})
	b := store([]int32{2, 2})
	if free := pool.NumFreeBlocks(); free != 3 {
		t.Fatalf("%d free after storing 3 blocks of 6, want 3", free)
	}
	// Touch b so a is the least recently matched.
	_, _, node, _ := c.join([]int32{2, 2, 9})
	c.abandon(node)
	if ok, err := c.ensureFree(4); err != nil || !ok {
		t.Fatalf("ensureFree(4) = %v, %v", ok, err)
	}
	for _, blk := range a {
		if pool.holders[blk] != 0 {
			t.Errorf("block %d of the least recently matched segment is still held", blk)
		}
	}
	if pool.holders[b[0]] != 1 {
		t.Errorf("the recently matched segment was evicted first")
	}
	// Pin b through a join; nothing more can be freed.
	_, _, node, _ = c.join([]int32{2, 2, 9})
	if ok, err := c.ensureFree(6); err != nil || ok {
		t.Fatalf("ensureFree(6) with a pinned segment = %v, %v; want false, nil", ok, err)
	}
	c.abandon(node)
	if ok, err := c.ensureFree(6); err != nil || !ok {
		t.Fatalf("ensureFree(6) after the join ended = %v, %v", ok, err)
	}
	if m := c.metrics(); m.Entries != 0 || m.Bytes != 0 {
		t.Fatalf("metrics with everything evicted: %+v", m)
	}
}

// TestPagedPrefixFailsClosed checks that a bookkeeping error poisons the cache
// rather than leaving the tree and the pool in silent disagreement.
func TestPagedPrefixFailsClosed(t *testing.T) {
	pool := newFakePool(4)
	c, err := newPagedPrefixCache(pool, 2, 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	// A table naming a free block cannot be retained.
	if err := c.leave(nil, []int32{1, 1}, []int{3}); err == nil {
		t.Fatal("leave with a free block succeeded")
	}
	if _, _, _, err := c.join([]int32{1, 1, 1}); err == nil {
		t.Fatal("join after a bookkeeping failure succeeded")
	}
	if err := c.reset(); err == nil {
		t.Fatal("reset after a bookkeeping failure succeeded")
	}
}
