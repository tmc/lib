//go:build linux || darwin

package cudacache

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// coarseIdentity models a filesystem whose timestamps are too coarse to
// record a write, and, when reuseInode is set, one that gives a replacing
// file the inode of the file it replaced. The files and their bytes are
// real; only what stat can distinguish is reduced.
func coarseIdentity(reuseInode bool) identityFunc {
	return func(path string, fi os.FileInfo) (ToolFileIdentity, error) {
		id, err := toolFileIdentity(path, fi)
		if err != nil {
			return ToolFileIdentity{}, err
		}
		id.ModTimeNanos, id.ChangeTimeNanos = 0, 0
		if reuseInode {
			id.Inode = 0
		}
		return id, nil
	}
}

func writeToolFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func rewriteAt(t *testing.T, path, content string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt([]byte(content), 0); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func renameOver(t *testing.T, path, content string) {
	t.Helper()
	writeToolFile(t, path+".new", content)
	if err := os.Rename(path+".new", path); err != nil {
		t.Fatal(err)
	}
}

// TestVerifyToolRehashesUnderCoarseStat rewrites a tool to different bytes
// of the same size where stat reports the recorded identity unchanged, so
// only the bytes record the write.
func TestVerifyToolRehashesUnderCoarseStat(t *testing.T) {
	tests := []struct {
		name       string
		rewrite    func(t *testing.T, path, content string)
		reuseInode bool
	}{
		{"in place", rewriteAt, false},
		{"replaced, inode reused", renameOver, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			identity := coarseIdentity(tt.reuseInode)
			path := filepath.Join(t.TempDir(), "libnvrtc.so.13")
			writeToolFile(t, path, "nvrtc build 1")
			d, err := digestTool("nvrtc", path, identity)
			if err != nil {
				t.Fatal(err)
			}
			if !d.File.Unix {
				t.Fatalf("digestTool(%s): no Unix identity", path)
			}
			if err := verifyTool(d, identity); err != nil {
				t.Fatalf("verifyTool on untouched file: %v", err)
			}

			tt.rewrite(t, path, "nvrtc build 2")
			after, err := statTool(path, identity)
			if err != nil {
				t.Fatal(err)
			}
			if after != d.File {
				t.Fatalf("fixture: coarse identity changed %+v -> %+v; it must hide the write", d.File, after)
			}
			if err := verifyTool(d, identity); !errors.Is(err, ErrToolIdentityChanged) {
				t.Fatalf("verifyTool after %s rewrite under unchanged stat = %v, want ErrToolIdentityChanged", tt.name, err)
			}

			// Adjacent: the original bytes written back verify again.
			tt.rewrite(t, path, "nvrtc build 1")
			if err := verifyTool(d, identity); err != nil {
				t.Fatalf("verifyTool after restoring the bytes: %v", err)
			}
		})
	}
}

// TestProbeLoadedToolRehashesUnderCoarseStat probes a mapped stand-in
// library twice around an in-place write that stat does not record. The
// second probe must report the new bytes.
func TestProbeLoadedToolRehashesUnderCoarseStat(t *testing.T) {
	identity := coarseIdentity(false)
	path := filepath.Join(t.TempDir(), "libnvrtc.so.13")
	writeToolFile(t, path, "nvrtc build 1")
	// LoadedLibraryPath on Linux accepts only a mapped file.
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	mem, err := syscall.Mmap(int(f.Fd()), 0, len("nvrtc build 1"), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(mem)

	first, err := probeLoadedTool("nvrtc", path, identity)
	if err != nil {
		t.Fatal(err)
	}
	rewriteAt(t, path, "nvrtc build 2")
	second, err := probeLoadedTool("nvrtc", path, identity)
	if err != nil {
		t.Fatal(err)
	}
	if second.File != first.File {
		t.Fatalf("fixture: coarse identity changed %+v -> %+v; it must hide the write", first.File, second.File)
	}
	if second.SHA256 == first.SHA256 {
		t.Fatalf("probe after in-place write = sha256 %s, unchanged", second.SHA256)
	}
	if err := verifyTool(first, identity); !errors.Is(err, ErrToolIdentityChanged) {
		t.Fatalf("verifyTool of first probe = %v, want ErrToolIdentityChanged", err)
	}
	if err := verifyTool(second, identity); err != nil {
		t.Fatalf("verifyTool of second probe: %v", err)
	}
}
