//go:build unix

package main

import (
	"golang.org/x/sys/unix"
	"path/filepath"
	"testing"
	"time"
)

func TestClientSourceFIFORefusal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fifo")
	if err := unix.Mkfifo(path, 0600); err != nil {
		t.Fatal(err)
	}
	result := make(chan error, 1)
	go func() { _, err := readClientSource(path); result <- err }()
	select {
	case err := <-result:
		if err == nil {
			t.Fatal("FIFO accepted")
		}
	case <-time.After(time.Second):
		t.Fatal("FIFO blocked source read")
	}
}
