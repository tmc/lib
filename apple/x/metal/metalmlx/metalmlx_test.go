package metalmlx

import (
	"testing"
	"unsafe"
)

type fakeBuf struct{ size uint64 }

func TestBufferCacheReuse(t *testing.T) {
	const page = 16384
	freed := 0
	c := newBufferCache(uint64(page),
		func(b *fakeBuf) uint64 { return b.size },
		func(*fakeBuf) { freed++ })

	b := &fakeBuf{size: 4096}
	c.recycleToCache(b)
	if got := c.cacheSize(); got != 4096 {
		t.Fatalf("cacheSize = %d, want 4096", got)
	}

	// A hit must be below min(2*size, size+2*pages).
	if _, ok := c.reuseFromCache(8192); ok {
		t.Fatal("reuseFromCache(8192) hit, want miss")
	}
	got, ok := c.reuseFromCache(4096)
	if !ok || got != b {
		t.Fatalf("reuseFromCache(4096) = %v, %v, want %v, true", got, ok, b)
	}
	if got := c.cacheSize(); got != 0 {
		t.Fatalf("cacheSize after reuse = %d, want 0", got)
	}

	// Eviction runs least-recently-cached first.
	first := &fakeBuf{size: page}
	second := &fakeBuf{size: 2 * page}
	third := &fakeBuf{size: 4 * page}
	c.recycleToCache(first)
	c.recycleToCache(second)
	c.recycleToCache(third)
	if n := c.releaseCachedBuffers(1); n != 1 {
		t.Fatalf("releaseCachedBuffers(1) = %d, want 1", n)
	}
	if got := c.cacheSize(); got != 6*page {
		t.Fatalf("cacheSize = %d, want %d", got, 6*page)
	}
	if _, ok := c.reuseFromCache(page); ok {
		t.Error("first buffer should have been evicted")
	}
	if n := c.clear(); n != 2 {
		t.Fatalf("clear = %d, want 2", n)
	}
	if freed != 3 {
		t.Fatalf("freed = %d, want 3", freed)
	}
}

func TestSizeDefaultsToOne(t *testing.T) {
	got := Size{Width: 8}.mtl()
	if got.Width != 8 || got.Height != 1 || got.Depth != 1 {
		t.Errorf("Size{8}.mtl() = %+v, want {8 1 1}", got)
	}
}

func TestCommandBufferLimitsEmptyArchitecture(t *testing.T) {
	ops, mb := commandBufferLimits("")
	if ops != 40 || mb != 40 {
		t.Fatalf("commandBufferLimits(empty) = %d, %d, want 40, 40", ops, mb)
	}
}

func TestBytesHelpers(t *testing.T) {
	v := uint32(0x04030201)
	b, err := ValueBytes(&v)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 4 || b[0] != 1 || b[3] != 4 {
		t.Errorf("ValueBytes = %v", b)
	}
	s := []int64{1, 2, 3}
	b, err = SliceBytes(s)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 24 {
		t.Fatalf("SliceBytes len = %d, want 24", len(b))
	}
	if got := *(*int64)(unsafe.Pointer(&b[16])); got != 3 {
		t.Errorf("SliceBytes[2] = %d, want 3", got)
	}
	b, err = SliceBytes([]int32(nil))
	if err != nil {
		t.Fatal(err)
	}
	if b != nil {
		t.Error("SliceBytes(nil) should be nil")
	}
}

func TestSysctlUint64(t *testing.T) {
	n, err := sysctlUint64("hw.memsize")
	if err != nil {
		t.Fatal(err)
	}
	if n < 1<<30 {
		t.Errorf("hw.memsize = %d, want at least 1 GiB", n)
	}
	if hostMemorySize() != n {
		t.Errorf("hostMemorySize = %d, want %d", hostMemorySize(), n)
	}
	if resourceLimit() == 0 {
		t.Error("resourceLimit = 0")
	}
}
