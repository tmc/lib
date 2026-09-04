package nvrtcx

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestComputeCacheKeyOrdersHeaders(t *testing.T) {
	a := map[string]string{"a.h": "a", "b.h": "b", "c.h": "c"}
	b := map[string]string{"c.h": "c", "a.h": "a", "b.h": "b"}
	if got, want := computeCacheKey("src", "name", []string{"-f"}, a), computeCacheKey("src", "name", []string{"-f"}, b); got != want {
		t.Fatalf("cache keys differ: %s != %s", got, want)
	}
}

func TestComputeCacheKeyFramesFields(t *testing.T) {
	tests := []struct {
		name string
		a    string
		b    string
	}{
		{
			name: "header fields",
			a:    computeCacheKey("src", "name", nil, map[string]string{"a": "b=c"}),
			b:    computeCacheKey("src", "name", nil, map[string]string{"a=b": "c"}),
		},
		{
			name: "option sections",
			a:    computeCacheKey("src", "name", []string{"a", "b"}, nil),
			b:    computeCacheKey("src", "name", nil, map[string]string{"a": "b"}),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.a == test.b {
				t.Fatalf("distinct inputs share cache key %s", test.a)
			}
		})
	}
}

func TestCUBINReturnsCopy(t *testing.T) {
	module := &CompiledModule{cubin: []byte{1, 2, 3}}
	got := module.CUBIN()
	got[0] = 9
	if !slices.Equal(module.CUBIN(), []byte{1, 2, 3}) {
		t.Fatalf("CUBIN aliases cached image: %v", module.CUBIN())
	}
}

func TestClearCachePreservesUnrelatedFiles(t *testing.T) {
	useTestCache(t)
	dir, err := getCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"a.ptx", "b.cubin", "keep.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(name), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := ClearCache(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "keep.txt")); err != nil {
		t.Fatalf("unrelated file removed: %v", err)
	}
	for _, name := range []string{"a.ptx", "b.cubin"} {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Fatalf("cache file %q remains: %v", name, err)
		}
	}
}

func TestLookupCacheRejectsEmptyImage(t *testing.T) {
	useTestCache(t)
	dir, err := getCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	const key = "empty"
	if err := os.WriteFile(filepath.Join(dir, key+".cache"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if module, ok := lookupCache(key); ok {
		t.Fatalf("lookupCache returned empty module %v", module)
	}
}

func TestCacheEntryRoundTrip(t *testing.T) {
	want := &CompiledModule{ptx: "ptx", cubin: []byte{1, 2, 3}, key: "key"}
	data, err := encodeCacheEntry(want)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeCacheEntry(want.key, data)
	if err != nil {
		t.Fatal(err)
	}
	if got.ptx != want.ptx || !bytes.Equal(got.cubin, want.cubin) || got.key != want.key {
		t.Fatalf("decodeCacheEntry = %#v, want %#v", got, want)
	}
}

func TestLookupCacheRejectsTruncatedEntry(t *testing.T) {
	useTestCache(t)
	dir, err := getCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	data, err := encodeCacheEntry(&CompiledModule{ptx: "valid", key: "key"})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "key.cache"), data[:len(data)-1], 0o600); err != nil {
		t.Fatal(err)
	}
	if module, ok := lookupCache("key"); ok {
		t.Fatalf("lookupCache returned truncated module %v", module)
	}
}

func TestLookupCacheRejectsCorruptEntry(t *testing.T) {
	useTestCache(t)
	dir, err := getCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	data, err := encodeCacheEntry(&CompiledModule{ptx: "valid", key: "key"})
	if err != nil {
		t.Fatal(err)
	}
	data[len(cacheMagic)+16] ^= 0xff
	if err := os.WriteFile(filepath.Join(dir, "key.cache"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	if module, ok := lookupCache("key"); ok {
		t.Fatalf("lookupCache returned corrupt module %v", module)
	}
}

func TestStoreCachePublishesOneEntry(t *testing.T) {
	useTestCache(t)
	if err := storeCache("key", &CompiledModule{ptx: "ptx", cubin: []byte{1}, key: "key"}); err != nil {
		t.Fatal(err)
	}
	dir, err := getCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "key.cache" {
		t.Fatalf("cache entries = %v, want key.cache", entries)
	}
}

func TestCompileRejectsInvalidOptions(t *testing.T) {
	tests := []struct {
		name string
		src  string
		opts []Option
	}{
		{name: "empty source"},
		{name: "nil option", src: "x", opts: []Option{nil}},
		{name: "empty name", src: "x", opts: []Option{WithName("")}},
		{name: "invalid architecture", src: "x", opts: []Option{WithArch(0, 0)}},
		{name: "empty flag", src: "x", opts: []Option{WithFlag("")}},
		{name: "empty header name", src: "x", opts: []Option{WithHeader("", "x")}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := Compile(test.src, test.opts...); err == nil {
				t.Fatal("Compile succeeded")
			}
		})
	}
}
