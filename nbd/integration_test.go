package nbd_test

import (
	"os"
	"testing"

	"github.com/tmc/lib/nbd"
)

// RequireKernelGate checks whether the Linux kernel NBD manual gate is enabled.
func RequireKernelGate(t *testing.T) {
	t.Helper()
	if os.Getenv("NBD_TEST_KERNEL") != "1" {
		t.Skip("set NBD_TEST_KERNEL=1 and run as root on a host with the nbd module loaded")
	}
}

// RunKernelGate is set by handwritten kernel integration tests.
var RunKernelGate func(*testing.T)

func runKernelGate(t *testing.T) {
	path := os.Getenv("NBD_TEST_DEVICE")
	if path == "" {
		path = "/dev/nbd0"
	}
	dev, err := nbd.OpenKernelDevice(path)
	if err != nil {
		t.Fatalf("open kernel device: %v", err)
	}
	defer dev.Close()
	if dev.FD() < 0 {
		t.Fatal("kernel device fd is invalid")
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
