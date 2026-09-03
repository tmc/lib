//go:build darwin

package metalprefix

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/lm/kvcache"
)

func testCache(t *testing.T, maxBytes int64, maxEntries int) *Cache {
	t.Helper()
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal unavailable: %v", err)
	}
	c, err := New(Config{Device: dev, MaxBytes: maxBytes, MaxEntries: maxEntries, BlockTokens: 2, Layers: 2, KVHeads: 1, HeadDim: 2})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := c.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return c
}

func kv(tokens int, seed float32) ([][]float32, [][]float32) {
	k := make([][]float32, 2)
	v := make([][]float32, 2)
	for layer := range k {
		k[layer] = make([]float32, tokens*2)
		v[layer] = make([]float32, tokens*2)
		for i := range k[layer] {
			k[layer][i] = seed + float32(layer*100+i)
			v[layer][i] = -seed - float32(layer*100+i)
		}
	}
	return k, v
}

func TestCachePartialBlockRoundTrip(t *testing.T) {
	c := testCache(t, 4096, 4)
	tokens := []int32{1, 2, 3}
	k, v := kv(len(tokens), 7)
	if err := c.Put(context.Background(), tokens, 9, k, v, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	l, ok, err := c.Acquire(append(tokens, 4))
	if err != nil || !ok {
		t.Fatalf("Acquire = %v, %v", ok, err)
	}
	defer l.Release()
	gotK, gotV := kv(len(tokens), 0)
	for layer := range gotK {
		clear(gotK[layer])
		clear(gotV[layer])
	}
	if err := l.Restore(gotK, gotV, len(tokens)); err != nil {
		t.Fatal(err)
	}
	for layer := range k {
		for i := range k[layer] {
			if gotK[layer][i] != k[layer][i] || gotV[layer][i] != v[layer][i] {
				t.Fatalf("layer %d value %d = (%v,%v), want (%v,%v)", layer, i, gotK[layer][i], gotV[layer][i], k[layer][i], v[layer][i])
			}
		}
	}
	if l.Tokens() != 3 || l.Current() != 9 {
		t.Fatalf("lease metadata = %d/%d, want 3/9", l.Tokens(), l.Current())
	}
	for _, id := range l.Blocks() {
		if id.ID == 0 || id.Generation == 0 || id.Start < 0 || id.End <= id.Start || id.Layer < 0 {
			t.Fatalf("invalid block identity: %+v", id)
		}
	}
}

func TestCacheCopyOnWriteSuffix(t *testing.T) {
	c := testCache(t, 4096, 4)
	k4, v4 := kv(4, 1)
	if err := c.Put(context.Background(), []int32{1, 2, 3, 4}, 5, k4, v4, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	before := c.Stats()
	k6, v6 := kv(6, 1)
	if err := c.Put(context.Background(), []int32{1, 2, 3, 4, 8, 9}, 10, k6, v6, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	after := c.Stats()
	blockBytes := int64(2 * 2 * 2 * 2 * 4) // K+V, layers, tokens, values/token, bytes.
	if after.Bytes-before.Bytes != blockBytes {
		t.Fatalf("COW suffix allocated %d bytes, want %d", after.Bytes-before.Bytes, blockBytes)
	}
}

func TestCacheReplacesEntryAtLimit(t *testing.T) {
	c := testCache(t, 4096, 1)
	tokens := []int32{1, 2}
	k, v := kv(len(tokens), 1)
	if err := c.Put(context.Background(), tokens, 3, k, v, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	if err := c.Put(context.Background(), tokens, 4, k, v, CompletedFence()); err != nil {
		t.Fatalf("replace entry: %v", err)
	}
	l, ok, err := c.Acquire(tokens)
	if err != nil || !ok {
		t.Fatalf("Acquire = %v, %v", ok, err)
	}
	defer l.Release()
	if got := l.Current(); got != 4 {
		t.Fatalf("current = %d, want 4", got)
	}
}

func TestEvictionRemovesLookupButKeepsLeasedBlocks(t *testing.T) {
	c := testCache(t, 4096, 1)
	k, v := kv(2, 1)
	if err := c.Put(context.Background(), []int32{1, 2}, 3, k, v, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	l, ok, err := c.Acquire([]int32{1, 2})
	if err != nil || !ok {
		t.Fatalf("Acquire = %v, %v", ok, err)
	}
	old := c.Stats()
	k2, v2 := kv(2, 20)
	if err := c.Put(context.Background(), []int32{8, 9}, 10, k2, v2, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := c.Acquire([]int32{1, 2}); err != nil || ok {
		t.Fatalf("evicted Acquire = %v, %v; want miss", ok, err)
	}
	if got := c.Stats(); got.Bytes <= old.Bytes {
		t.Fatalf("live lease bytes = %d, want greater than %d", got.Bytes, old.Bytes)
	}
	l.Release()
	if got := c.Stats(); got.Bytes != old.Bytes {
		t.Fatalf("released physical bytes = %d, want %d", got.Bytes, old.Bytes)
	}
}

func TestPutCancellationIsTransactional(t *testing.T) {
	c := testCache(t, 4096, 2)
	done := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	k, v := kv(2, 1)
	err := c.Put(ctx, []int32{1, 2}, 3, k, v, ChannelFence(done))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Put error = %v, want context.Canceled", err)
	}
	if got := c.Stats(); got != (Stats{}) {
		t.Fatalf("stats after cancellation = %+v, want zero", got)
	}
}

func TestAcquireAllRollsBackOnStaleConstituent(t *testing.T) {
	c := testCache(t, 4096, 2)
	k, v := kv(2, 1)
	tokens := []int32{1, 2}
	if err := c.Put(context.Background(), tokens, 3, k, v, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	c.mu.Lock()
	e := c.entries[tokenKey(tokens)]
	bad := e.blocks[len(e.blocks)-1]
	if err := c.store.Evict(context.Background(), bad.lease, func(context.Context, kvcache.Descriptor) error { return nil }); err != nil {
		c.mu.Unlock()
		t.Fatal(err)
	}
	c.mu.Unlock()

	if _, ok, err := c.Acquire(tokens); err == nil || ok {
		t.Fatalf("Acquire stale constituent = %v, %v; want error", ok, err)
	}
	c.mu.Lock()
	first := e.blocks[0]
	desc, err := c.store.Describe(first.lease)
	if err != nil {
		c.mu.Unlock()
		t.Fatal(err)
	}
	if desc.References != 1 {
		c.mu.Unlock()
		t.Fatalf("rolled-back references = %d, want 1", desc.References)
	}
	// Repair the deliberately corrupted test fixture so cleanup can reclaim all
	// physical buffers and close the logical store.
	e.blocks = e.blocks[:len(e.blocks)-1]
	bad.b.k.Free()
	bad.b.v.Free()
	delete(c.blocks, [2]uint64{uint64(bad.b.id), uint64(bad.b.gen)})
	c.bytes -= int64(bad.b.k.Len()+bad.b.v.Len()) * 4
	c.mu.Unlock()
}

func TestCloseWaitsForLiveLease(t *testing.T) {
	c := testCache(t, 4096, 2)
	k, v := kv(2, 1)
	if err := c.Put(context.Background(), []int32{1, 2}, 3, k, v, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	l, _, err := c.Acquire([]int32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); !errors.Is(err, kvcache.ErrBusy) {
		t.Fatalf("Close with lease = %v, want ErrBusy", err)
	}
	if _, _, err := c.Acquire([]int32{1, 2}); !errors.Is(err, kvcache.ErrClosed) {
		t.Fatalf("Acquire after Close = %v, want ErrClosed", err)
	}
	l.Release()
	if err := c.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}
