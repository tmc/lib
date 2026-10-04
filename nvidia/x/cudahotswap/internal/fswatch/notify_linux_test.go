package fswatch

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestInotifyRenameSave(t *testing.T) {
	dir := t.TempDir()
	n, err := openNotify([]string{dir}, false)
	if err != nil {
		t.Fatal(err)
	}
	defer n.close()
	if n.fd < 0 {
		t.Fatal("default backend is not inotify")
	}
	path := filepath.Join(dir, "temporary")
	if err := os.WriteFile(path, []byte("kernel"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(path, filepath.Join(dir, "kernel.cu")); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := n.wait(ctx, time.Hour); err != nil {
		t.Fatalf("rename did not notify: %v", err)
	}
	poll, err := openNotify([]string{dir}, true)
	if err != nil {
		t.Fatal(err)
	}
	defer poll.close()
	if poll.fd >= 0 {
		t.Fatal("poll mode created an inotify descriptor")
	}
}
