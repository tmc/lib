//go:build linux

package nvrtcx

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"golang.org/x/sys/unix"
)

func TestLoadedCompilerMappingRejectsStaleFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "libnvrtc fixture.so")
	if err := os.WriteFile(path, []byte("compiler bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	st := fi.Sys().(*syscall.Stat_t)
	line := fmt.Sprintf("1000-2000 r-xp 0000 %02x:%02x %d %s", unix.Major(uint64(st.Dev)), unix.Minor(uint64(st.Dev)), st.Ino, path)
	if got, ok, err := checkedCompilerMapping(line, path); err != nil || !ok || got != path {
		t.Fatalf("intact mapping: %q %t %v", got, ok, err)
	}
	for _, maps := range []string{"", line + " (deleted)", strings.Replace(line, fmt.Sprint(st.Ino), "0", 1)} {
		if _, _, err := checkedCompilerMapping(maps, path); err == nil {
			t.Fatal("unmapped/deleted/wrong inode accepted")
		}
	}
	if err := os.WriteFile(path+".new", []byte("replacement"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(path+".new", path); err != nil {
		t.Fatal(err)
	}
	if _, _, err := checkedCompilerMapping(line, path); err == nil {
		t.Fatal("replacement bytes attributed to old mapping")
	}
}

func TestLoadedCompilerIdentityActualHostMapping(t *testing.T) {
	for _, mode := range []string{"replace", "in place", "delete"} {
		t.Run(mode, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "libnvrtc-fixture.so")
			data := make([]byte, 4096)
			copy(data, "compiler original")
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			mapped, err := syscall.Mmap(int(f.Fd()), 0, len(data), syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				t.Fatal(err)
			}
			defer syscall.Munmap(mapped)
			id, supported, err := loadedCompilerIdentity(path)
			if err != nil || !supported {
				t.Fatalf("actual host mapping: %q %t %v", id, supported, err)
			}
			h := sha256.Sum256(data)
			if !strings.Contains(id, fmt.Sprintf("sha256=%x", h)) {
				t.Fatal("wrong descriptor digest")
			}
			if again, _, err := loadedCompilerIdentity(path); err != nil || again != id {
				t.Fatal("stable mapping changed")
			}
			switch mode {
			case "replace":
				replacement := path + ".new"
				if err := os.WriteFile(replacement, data, 0600); err != nil {
					t.Fatal(err)
				}
				if err := os.Rename(replacement, path); err != nil {
					t.Fatal(err)
				}
			case "in place":
				data[0] = 'X'
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
			case "delete":
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
			}
			if err := verifyLoadedCompilerIdentity(path, id); err == nil {
				t.Fatal("changed mapped compiler accepted")
			}
		})
	}
}
