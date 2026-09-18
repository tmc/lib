package cudaplan_test

import (
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func requireNativePlanDevice(t *testing.T) *cuda.Device {
	t.Helper()
	// See cudashared/native_test.go: the lock exports the holder's pid, not "1".
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("enabled native plan test requires running under the GPU lock")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("enabled native plan test requires a CUDA device: %v", err)
	}
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Errorf("close native device: %v", err)
		}
	})
	return dev
}
