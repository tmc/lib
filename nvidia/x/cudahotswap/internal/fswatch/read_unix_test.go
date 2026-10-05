//go:build unix

package fswatch

import (
	"golang.org/x/sys/unix"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestReadFileRefusesFIFOAndSymlink(t *testing.T) {
	dir := t.TempDir()
	fifo := filepath.Join(dir, "fifo")
	if err := unix.Mkfifo(fifo, 0600); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := ReadFile(fifo, 8<<20); done <- err }()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("FIFO accepted")
		}
	case <-time.After(time.Second):
		t.Fatal("FIFO open blocked")
	}
	regular := filepath.Join(dir, "regular")
	if err := os.WriteFile(regular, []byte("data"), 0600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(regular, link); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadFile(link, 8<<20); err == nil {
		t.Fatal("symlink followed")
	}
	if _, err := ReadFile(regular, 3); err == nil {
		t.Fatal("size limit ignored")
	}
	if b, err := ReadFile(regular, 4); err != nil || string(b) != "data" {
		t.Fatal(string(b), err)
	}
}

func TestScanBoundsBeforeReading(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "large")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate((8 << 20) + 1); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := scan([]string{dir}); err == nil {
		t.Fatal("oversized scan accepted")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 9; i++ {
		f, err := os.CreateTemp(dir, "limit-")
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Truncate(8 << 20); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := scan([]string{dir}); err == nil {
		t.Fatal("aggregate scan accepted")
	}
}
