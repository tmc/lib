package cudamlx

import (
	"path/filepath"
	"testing"
	"unsafe"
)

func TestRoundSize(t *testing.T) {
	tests := []struct {
		in, want uint64
	}{
		{1, 8},
		{8, 8},
		{9, 16},
		{100, 128},
		{pageSize - 1, pageSize},
		{pageSize, pageSize},
		{pageSize + 1, 2 * pageSize},
		{3 * pageSize, 3 * pageSize},
	}
	for _, tt := range tests {
		if got := roundSize(tt.in); got != tt.want {
			t.Errorf("roundSize(%d) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

func TestBufferCacheReuse(t *testing.T) {
	freed := 0
	c := newBufferCache(pageSize,
		func(b *Buffer) uint64 { return b.Size },
		func(*Buffer) { freed++ })

	b := &Buffer{Size: 4096}
	c.recycleToCache(b)
	if got := c.cacheSize(); got != 4096 {
		t.Fatalf("cacheSize = %d, want 4096", got)
	}

	// A hit must be below min(2*size, size+2*pages).
	if got := c.reuseFromCache(8192); got != nil {
		t.Fatalf("reuseFromCache(8192) = %v, want nil", got)
	}
	if got := c.reuseFromCache(4096); got != b {
		t.Fatalf("reuseFromCache(4096) = %v, want %v", got, b)
	}
	if got := c.cacheSize(); got != 0 {
		t.Fatalf("cacheSize after reuse = %d, want 0", got)
	}

	// Eviction runs least-recently-cached first.
	first := &Buffer{Size: pageSize}
	second := &Buffer{Size: 2 * pageSize}
	third := &Buffer{Size: 4 * pageSize}
	c.recycleToCache(first)
	c.recycleToCache(second)
	c.recycleToCache(third)
	if n := c.releaseCachedBuffers(1); n != 1 {
		t.Fatalf("releaseCachedBuffers(1) = %d, want 1", n)
	}
	if got := c.cacheSize(); got != 6*pageSize {
		t.Fatalf("cacheSize = %d, want %d", got, 6*pageSize)
	}
	if got := c.reuseFromCache(pageSize); got != nil {
		t.Fatalf("first buffer should have been evicted, got %v", got)
	}
	if n := c.clear(); n != 2 {
		t.Fatalf("clear = %d, want 2", n)
	}
	if freed != 3 {
		t.Fatalf("freed = %d, want 3", freed)
	}
}

func TestKernelArgsShapePadding(t *testing.T) {
	var args KernelArgs
	if err := args.AppendShape([]int32{2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if err := args.AppendStrides([]int64{12, 4, 1}); err != nil {
		t.Fatal(err)
	}
	ptrs := args.Args()
	if len(ptrs) != 2 {
		t.Fatalf("len(args) = %d, want 2", len(ptrs))
	}
	shape := (*Shape)(ptrs[0])
	want := Shape{2, 3, 4}
	if *shape != want {
		t.Errorf("shape arg = %v, want %v", *shape, want)
	}
	strides := (*Strides)(ptrs[1])
	if strides[0] != 12 || strides[3] != 0 {
		t.Errorf("strides arg = %v", *strides)
	}
	if unsafe.Sizeof(*shape) != 4*MaxNDim {
		t.Errorf("sizeof(Shape) = %d", unsafe.Sizeof(*shape))
	}

	if err := args.AppendShape(make([]int32, MaxNDim+1)); err == nil {
		t.Error("AppendShape with ndim > MaxNDim should fail")
	}
}

func TestPTXCacheRoundTrip(t *testing.T) {
	dir := t.TempDir()
	ptx := []byte("fake ptx\x00")
	kernels := [][2]string{{"kern", "_Z4kernPi"}}
	writeCachedPTX(dir, "mod", ptx, kernels, "__global__ void kern(int*) {}")

	gotPTX, gotKernels, ok := readCachedPTX(dir, "mod")
	if !ok {
		t.Fatal("readCachedPTX failed")
	}
	if string(gotPTX) != string(ptx) {
		t.Errorf("ptx = %q, want %q", gotPTX, ptx)
	}
	if len(gotKernels) != 1 || gotKernels[0] != kernels[0] {
		t.Errorf("kernels = %v, want %v", gotKernels, kernels)
	}

	if _, _, ok := readCachedPTX(filepath.Join(dir, "missing"), "mod"); ok {
		t.Error("readCachedPTX on missing dir should fail")
	}
	if _, _, ok := readCachedPTX("", "mod"); ok {
		t.Error("readCachedPTX with empty dir should fail")
	}
}
