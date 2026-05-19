package fuse_test

import (
	"os"
	"testing"

	"github.com/tmc/lib/fuse"
)

func RequireKernelGate(t *testing.T) {
	t.Helper()
	if os.Getenv("FUSE_TEST_KERNEL") != "1" {
		t.Skip("set FUSE_TEST_KERNEL=1 and run on a host with /dev/fuse access")
	}
}

var RunKernelGate func(*testing.T)

func runKernelGate(t *testing.T) {
	dev, err := fuse.OpenDevice(os.Getenv("FUSE_TEST_DEVICE"))
	if err != nil {
		t.Fatalf("open fuse device: %v", err)
	}
	defer dev.Close()
	if dev.FD() < 0 {
		t.Fatal("invalid fuse fd")
	}
}

func TestIntegrationKernelGate(t *testing.T) {
	RequireKernelGate(t)
	run := runKernelGate
	if RunKernelGate != nil {
		run = RunKernelGate
	}
	run(t)
}
