//go:build linux && cupti_physical

package cuptipmsampling_test

import (
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

func withContext(t *testing.T) {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA driver unavailable: %v", err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("retain primary context: %v", err)
	}
	t.Cleanup(func() {
		if err := dev.PrimaryCtxRelease(); err != nil {
			t.Errorf("release context: %v", err)
		}
	})
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("set current context: %v", err)
	}
}

func TestPhysicalCounterAvailability(t *testing.T) {
	withContext(t)

	if _, err := cuptipmsampling.CheckVersion(); err != nil {
		t.Skipf("CUPTI unavailable or unqualified version: %v", err)
	}

	img, err := cuptipmsampling.CounterAvailability(0)
	if err != nil {
		t.Fatalf("CounterAvailability(0): %v", err)
	}
	t.Logf("CounterAvailability image size: %d bytes", len(img))
}
