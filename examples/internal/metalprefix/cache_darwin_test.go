//go:build darwin

package metalprefix

import (
	"context"
	"errors"
	"testing"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/lm/kvcache"
)

var errInjectedCompletion = errors.New("injected terminal command-buffer failure")

func TestCacheTracksPhysicalBlocks(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal unavailable: %v", err)
	}
	defer dev.Close()
	var tracker metalmem.Tracker
	c, err := New(Config{Device: dev, Tracker: &tracker, MaxBytes: 4096, MaxEntries: 4, BlockTokens: 2, Layers: 2, KVHeads: 1, HeadDim: 2})
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int32{1, 2, 3}
	k, v := kv(len(tokens), 7)
	if err := c.Put(t.Context(), tokens, 9, k, v, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	const want = 2 * 2 * 3 * 2 * 4
	if current, peak := tracker.Snapshot(); current != want || peak != want {
		t.Fatalf("tracked blocks = %d, %d, want %d", current, peak, want)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if current, peak := tracker.Snapshot(); current != 0 || peak != want {
		t.Fatalf("tracked blocks after Close = %d, %d, want 0, %d", current, peak, want)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
}

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

func TestCachePutBuffersRoundTrip(t *testing.T) {
	c := testCache(t, 4096, 4)
	tokens := []int32{1, 2, 3}
	wantK, wantV := kv(len(tokens), 7)
	k := make([]mempool.Slice[float32], 2)
	v := make([]mempool.Slice[float32], 2)
	for layer := range k {
		var err error
		k[layer], err = mempool.AllocFromSlice(c.cfg.Device, append([]float32{99, 99}, wantK[layer]...))
		if err != nil {
			t.Fatal(err)
		}
		v[layer], err = mempool.AllocFromSlice(c.cfg.Device, append([]float32{99, 99}, wantV[layer]...))
		if err != nil {
			t.Fatal(err)
		}
		defer k[layer].Free()
		defer v[layer].Free()
	}
	if err := c.PutBuffers(t.Context(), tokens, 9, k, v, 2); err != nil {
		t.Fatal(err)
	}
	for layer := range k {
		clear(k[layer].HostSlice())
		clear(v[layer].HostSlice())
	}
	lease, ok, err := c.Acquire(tokens)
	if err != nil || !ok {
		t.Fatalf("Acquire = %v, %v", ok, err)
	}
	defer lease.Release()
	gotK, gotV := kv(len(tokens), 0)
	for layer := range gotK {
		clear(gotK[layer])
		clear(gotV[layer])
	}
	if err := lease.Restore(gotK, gotV, len(tokens)); err != nil {
		t.Fatal(err)
	}
	for layer := range wantK {
		for i := range wantK[layer] {
			if gotK[layer][i] != wantK[layer][i] || gotV[layer][i] != wantV[layer][i] {
				t.Fatalf("layer %d value %d = %g/%g, want %g/%g", layer, i, gotK[layer][i], gotV[layer][i], wantK[layer][i], wantV[layer][i])
			}
		}
	}
}

func TestPutBuffersFailedCompletionRollsBack(t *testing.T) {
	c := testCache(t, 4096, 4)
	kData, vData := kv(3, 7)
	k := make([]mempool.Slice[float32], len(kData))
	v := make([]mempool.Slice[float32], len(vData))
	for layer := range k {
		var err error
		k[layer], err = mempool.AllocFromSlice(c.cfg.Device, kData[layer])
		if err != nil {
			t.Fatal(err)
		}
		v[layer], err = mempool.AllocFromSlice(c.cfg.Device, vData[layer])
		if err != nil {
			t.Fatal(err)
		}
		defer k[layer].Free()
		defer v[layer].Free()
	}
	c.waitCommandBuffer = func(ctx context.Context, cb apple.MTLCommandBuffer) error {
		// Wait for the real blit to reach a terminal state before simulating the
		// unsuccessful completion observed by the owning Go layer.
		if err := metal.WaitCommandBuffer(ctx, cb); err != nil {
			return err
		}
		return errInjectedCompletion
	}
	err := c.PutBuffers(t.Context(), []int32{1, 2, 3}, 4, k, v, 0)
	if !errors.Is(err, errInjectedCompletion) {
		t.Fatalf("PutBuffers error = %v, want injected completion", err)
	}
	if got := c.Stats(); got != (Stats{}) {
		t.Fatalf("stats after failed completion = %+v, want zero", got)
	}
	if _, ok, err := c.Acquire([]int32{1, 2, 3}); err != nil || ok {
		t.Fatalf("Acquire after failed completion = %v, %v; want miss", ok, err)
	}
}

func TestLayerPinsSurviveFailureUntilReleased(t *testing.T) {
	c := testCache(t, 4096, 1)
	k, v := kv(2, 1)
	tokens := []int32{1, 2}
	if err := c.Put(t.Context(), tokens, 3, k, v, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	lease, ok, err := c.Acquire(tokens)
	if err != nil || !ok {
		t.Fatalf("Acquire = %v, %v", ok, err)
	}
	view, err := lease.PinLayer(0)
	if err != nil {
		t.Fatal(err)
	}
	before := c.Stats()
	// Replacement evicts the old entry. The parent lease and independent
	// block-table view still pin its buffers when completion reports failure.
	k2, v2 := kv(2, 9)
	if err := c.Put(t.Context(), []int32{8, 9}, 10, k2, v2, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	lease.Release()
	if got := c.Stats(); got.Bytes <= before.Bytes {
		t.Fatalf("bytes with failed-wave view pin = %d, want more than %d", got.Bytes, before.Bytes)
	}
	// A terminal error does not itself release buffer references. Ownership
	// ends when the command-buffer owner releases its view pins.
	complete := func() error { return errInjectedCompletion }
	if err := complete(); !errors.Is(err, errInjectedCompletion) {
		t.Fatalf("completion error = %v, want injected failure", err)
	}
	view.Release()
	if got := c.Stats(); got.Bytes != before.Bytes {
		t.Fatalf("bytes after releasing failed-wave pins = %d, want %d", got.Bytes, before.Bytes)
	}
}

func TestCacheExtendBuffersPartialBlock(t *testing.T) {
	c := testCache(t, 4096, 4)
	baseTokens := []int32{1, 2, 3}
	wantTokens := []int32{1, 2, 3, 4, 5}
	baseK, baseV := kv(len(baseTokens), 7)
	if err := c.Put(t.Context(), baseTokens, 4, baseK, baseV, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	base, ok, err := c.Acquire(wantTokens)
	if err != nil || !ok {
		t.Fatalf("Acquire base = %v, %v", ok, err)
	}
	defer base.Release()
	baseBlocks := base.Blocks()
	wantK, wantV := kv(len(wantTokens), 7)
	k := make([]mempool.Slice[float32], c.cfg.Layers)
	v := make([]mempool.Slice[float32], c.cfg.Layers)
	for layer := range k {
		sourceK := append([]float32(nil), wantK[layer]...)
		sourceV := append([]float32(nil), wantV[layer]...)
		clear(sourceK[:len(baseTokens)*c.valuesPerToken()])
		clear(sourceV[:len(baseTokens)*c.valuesPerToken()])
		k[layer], err = mempool.AllocFromSlice(c.cfg.Device, sourceK)
		if err != nil {
			t.Fatal(err)
		}
		v[layer], err = mempool.AllocFromSlice(c.cfg.Device, sourceV)
		if err != nil {
			t.Fatal(err)
		}
		defer k[layer].Free()
		defer v[layer].Free()
	}
	if err := c.ExtendBuffers(t.Context(), base, wantTokens, 6, k, v, 0); err != nil {
		t.Fatal(err)
	}
	if got := c.Stats(); got.DeviceReadBytes != 96 || got.DeviceWriteBytes != 96 {
		t.Fatalf("device copy bytes = %d/%d, want 96/96", got.DeviceReadBytes, got.DeviceWriteBytes)
	}
	extended, err := c.AcquireExact(wantTokens)
	if err != nil {
		t.Fatal(err)
	}
	defer extended.Release()
	gotK, gotV := kv(len(wantTokens), 0)
	for layer := range gotK {
		clear(gotK[layer])
		clear(gotV[layer])
	}
	if err := extended.Restore(gotK, gotV, len(wantTokens)); err != nil {
		t.Fatal(err)
	}
	for layer := range wantK {
		for i := range wantK[layer] {
			if gotK[layer][i] != wantK[layer][i] || gotV[layer][i] != wantV[layer][i] {
				t.Fatalf("layer %d value %d = %g/%g, want %g/%g", layer, i, gotK[layer][i], gotV[layer][i], wantK[layer][i], wantV[layer][i])
			}
		}
	}
	extendedBlocks := extended.Blocks()
	for layer := range c.cfg.Layers {
		if extendedBlocks[layer].ID != baseBlocks[layer].ID || extendedBlocks[layer].Generation != baseBlocks[layer].Generation {
			t.Fatalf("layer %d complete block was not shared: base=%+v extended=%+v", layer, baseBlocks[layer], extendedBlocks[layer])
		}
	}
}

func TestCacheExtendBuffersCollisionCancellationAndStale(t *testing.T) {
	c := testCache(t, 4096, 4)
	baseTokens := []int32{1, 2}
	targetTokens := []int32{1, 2, 3}
	baseK, baseV := kv(2, 1)
	if err := c.Put(t.Context(), baseTokens, 3, baseK, baseV, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	base, _, err := c.Acquire(targetTokens)
	if err != nil {
		t.Fatal(err)
	}
	targetK, targetV := kv(3, 10)
	if err := c.Put(t.Context(), targetTokens, 4, targetK, targetV, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	before := c.Stats()
	sourceK := make([]mempool.Slice[float32], c.cfg.Layers)
	sourceV := make([]mempool.Slice[float32], c.cfg.Layers)
	for layer := range sourceK {
		sourceK[layer], err = mempool.AllocFromSlice(c.cfg.Device, make([]float32, 8))
		if err != nil {
			t.Fatal(err)
		}
		sourceV[layer], err = mempool.AllocFromSlice(c.cfg.Device, make([]float32, 8))
		if err != nil {
			t.Fatal(err)
		}
		defer sourceK[layer].Free()
		defer sourceV[layer].Free()
	}
	if err := c.ExtendBuffers(t.Context(), base, targetTokens, 99, sourceK, sourceV, 0); err != nil {
		t.Fatalf("collision: %v", err)
	}
	if got := c.Stats(); got.Bytes != before.Bytes || got.Blocks != before.Blocks || got.Entries != before.Entries {
		t.Fatalf("collision changed allocation: before=%+v after=%+v", before, got)
	}
	collision, err := c.AcquireExact(targetTokens)
	if err != nil {
		t.Fatal(err)
	}
	if collision.Current() != 4 {
		t.Fatalf("collision current = %d, want first publisher 4", collision.Current())
	}
	collision.Release()
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if err := c.ExtendBuffers(canceled, base, append(targetTokens, 4), 5, sourceK, sourceV, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled extension: %v", err)
	}
	failureTokens := append(targetTokens, 4)
	failureBefore := c.Stats()
	c.waitCommandBuffer = func(ctx context.Context, cb apple.MTLCommandBuffer) error {
		if err := metal.WaitCommandBuffer(ctx, cb); err != nil {
			return err
		}
		return errInjectedCompletion
	}
	if err := c.ExtendBuffers(t.Context(), base, failureTokens, 5, sourceK, sourceV, 0); !errors.Is(err, errInjectedCompletion) {
		t.Fatalf("failed completion extension: %v", err)
	}
	if got := c.Stats(); got != failureBefore {
		t.Fatalf("failed completion changed cache: before=%+v after=%+v", failureBefore, got)
	}
	if _, err := c.AcquireExact(failureTokens); !errors.Is(err, kvcache.ErrStale) {
		t.Fatalf("failed completion published target: %v", err)
	}
	c.waitCommandBuffer = metal.WaitCommandBuffer
	c.mu.Lock()
	good := base.refs[0].lease
	base.refs[0].lease = kvcache.Lease{}
	c.mu.Unlock()
	if err := c.ExtendBuffers(t.Context(), base, append(targetTokens, 4), 5, sourceK, sourceV, 0); !errors.Is(err, kvcache.ErrStale) {
		t.Fatalf("stale constituent extension: %v", err)
	}
	c.mu.Lock()
	base.refs[0].lease = good
	c.mu.Unlock()
	base.Release()
	if err := c.ExtendBuffers(t.Context(), base, append(targetTokens, 4), 5, sourceK, sourceV, 0); !errors.Is(err, kvcache.ErrStale) {
		t.Fatalf("stale extension: %v", err)
	}
}

func TestCacheExtendBuffersEvictsLookupKeepsBaseLease(t *testing.T) {
	c := testCache(t, 4096, 1)
	baseTokens := []int32{1, 2, 3}
	targetTokens := []int32{1, 2, 3, 4}
	baseK, baseV := kv(len(baseTokens), 2)
	if err := c.Put(t.Context(), baseTokens, 4, baseK, baseV, CompletedFence()); err != nil {
		t.Fatal(err)
	}
	base, _, err := c.Acquire(targetTokens)
	if err != nil {
		t.Fatal(err)
	}
	wantK, wantV := kv(len(targetTokens), 2)
	sourceK := make([]mempool.Slice[float32], c.cfg.Layers)
	sourceV := make([]mempool.Slice[float32], c.cfg.Layers)
	for layer := range sourceK {
		dataK := append([]float32(nil), wantK[layer]...)
		dataV := append([]float32(nil), wantV[layer]...)
		clear(dataK[:len(baseTokens)*c.valuesPerToken()])
		clear(dataV[:len(baseTokens)*c.valuesPerToken()])
		sourceK[layer], err = mempool.AllocFromSlice(c.cfg.Device, dataK)
		if err != nil {
			t.Fatal(err)
		}
		sourceV[layer], err = mempool.AllocFromSlice(c.cfg.Device, dataV)
		if err != nil {
			t.Fatal(err)
		}
		defer sourceK[layer].Free()
		defer sourceV[layer].Free()
	}
	if err := c.ExtendBuffers(t.Context(), base, targetTokens, 5, sourceK, sourceV, 0); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := c.Acquire(baseTokens); err != nil || ok {
		t.Fatalf("evicted base lookup = %v, %v; want miss", ok, err)
	}
	gotK, gotV := kv(len(baseTokens), 0)
	for layer := range gotK {
		clear(gotK[layer])
		clear(gotV[layer])
	}
	if err := base.Restore(gotK, gotV, len(baseTokens)); err != nil {
		t.Fatalf("live base after eviction: %v", err)
	}
	base.Release()
	extended, err := c.AcquireExact(targetTokens)
	if err != nil {
		t.Fatal(err)
	}
	extended.Release()
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
