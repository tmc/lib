package cudatracehost

import (
	"context"
	"errors"
	"github.com/tmc/lib/ebpfhost/bpfprobe"
	"os"
	"testing"
	"time"
)

func TestCloseUnblocksNext(t *testing.T) {
	c := &Collector{done: make(chan struct{}), events: make(chan transportEvent), group: &bpfprobe.Group{}}
	result := make(chan error, 1)
	go func() { _, err := c.nextBPF(context.Background()); result <- err }()
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-result:
		if !errors.Is(err, os.ErrClosed) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("close did not unblock read")
	}
	if _, err := c.Next(context.Background()); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("next after close: %v", err)
	}
}
func TestTerminalErrorRetained(t *testing.T) {
	c := &Collector{done: make(chan struct{}), events: make(chan transportEvent)}
	want := errors.New("reader failed")
	c.markTerminal(want)
	if _, err := c.Next(context.Background()); !errors.Is(err, want) {
		t.Fatal(err)
	}
}
func TestDeviceIdentity(t *testing.T) {
	if !sameDevice(0x801, "08:01") || sameDevice(0x801, "08:02") || sameDevice(0x801, "bad") {
		t.Fatal("device mismatch accepted")
	}
}
