package nvrtcx

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestCompilerCacheKey(t *testing.T) {
	input := computeCacheKey("source", "name.cu", nil, nil)
	base := compilerCacheKey(input, "nvrtc 13.0 /cuda/libnvrtc.so")
	if base != compilerCacheKey(input, "nvrtc 13.0 /cuda/libnvrtc.so") {
		t.Fatal("same compiler identity changed key")
	}
	for _, identity := range []string{"nvrtc 13.1 /cuda/libnvrtc.so", "nvrtc 13.0 /other/libnvrtc.so"} {
		if base == compilerCacheKey(input, identity) {
			t.Fatalf("compiler identity %q reused key", identity)
		}
	}
	if base == input {
		t.Fatal("legacy unqualified key reused")
	}
}

func TestCompilerFileIdentityChangesKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "libnvrtc.so.13.0.88")
	if err := os.WriteFile(path, []byte("nvrtc build 1"), 0o644); err != nil {
		t.Fatal(err)
	}
	input := computeCacheKey("source", "name.cu", nil, nil)
	key := func() string {
		t.Helper()
		id := libraryIdentity(path)
		if id == "" {
			t.Fatalf("libraryIdentity(%s) = \"\"", path)
		}
		return compilerCacheKey(input, "nvrtc 13.0 "+path+" "+id)
	}
	base := key()
	if key() != base {
		t.Fatal("untouched library changed key")
	}

	// In place, same length: only the modification time records the write.
	f, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt([]byte("nvrtc build 2"), 0); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	inPlace := key()
	if inPlace == base {
		t.Fatal("in-place rewrite reused key")
	}

	// Renamed over, with the modification time put back: only the inode
	// records the replacement.
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	tmp := path + ".new"
	if err := os.WriteFile(tmp, []byte("nvrtc build 3"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(tmp, fi.ModTime(), fi.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp, path); err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "linux" || runtime.GOOS == "darwin" {
		if key() == inPlace {
			t.Fatal("replacement by rename reused key")
		}
	}
}

func TestMappedLibraryIn(t *testing.T) {
	maps := `ffff80000000-ffff80400000 r--p 00000000 103:02 5001 /usr/local/cuda/lib64/libnvrtc.so.13.0.88
ffff80400000-ffff81000000 r-xp 00400000 103:02 5001 /usr/local/cuda/lib64/libnvrtc.so.13.0.88
ffff82000000-ffff83000000 r-xp 00000000 103:02 5002 /usr/local/cuda/lib64/libnvrtc-builtins.so.13.0.88
ffff84000000-ffff85000000 r-xp 00000000 103:02 6001 /usr/lib/aarch64-linux-gnu/libcuda.so.580.95.05 (deleted)
ffff88000000-ffff88021000 rw-p 00000000 00:00 0                          [heap]
`
	tests := []struct {
		soname, maps, want string
	}{
		{"libnvrtc.so.13", maps, "/usr/local/cuda/lib64/libnvrtc.so.13.0.88"},
		{"libnvrtc-builtins.so.13", maps, "/usr/local/cuda/lib64/libnvrtc-builtins.so.13.0.88"},
		{"libcuda.so.1", maps, "/usr/lib/aarch64-linux-gnu/libcuda.so.580.95.05"},
		{"libnvJitLink.so.13", maps, ""},
		{"libnvrtc.so.13", maps + "ffff90000000-ffff90001000 r-xp 00000000 103:02 5003 /opt/libnvrtc.so.13.1.0\n", ""},
		{"libnvrtc.so.13", maps + "ffff90000000-ffff90001000 r-xp 00000000 103:02 5004 /usr/local/cuda-12/lib64/libnvrtc.so.12.8.93\n", "/usr/local/cuda/lib64/libnvrtc.so.13.0.88"},
	}
	for _, tt := range tests {
		if got := mappedLibraryIn(tt.maps, tt.soname); got != tt.want {
			t.Errorf("mappedLibraryIn(%s) = %q, want %q", tt.soname, got, tt.want)
		}
	}
}
