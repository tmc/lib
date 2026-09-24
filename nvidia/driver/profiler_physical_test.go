package driver_test

import (
	"os"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

// TestNativeProfilerControl checks that the profiler symbols bind and that
// stopping and restarting collection succeeds against a current context. No
// profiler is attached, which is the case the calls must still tolerate.
func TestNativeProfilerControl(t *testing.T) {
	if os.Getenv("CUDA_DRIVER_HARDWARE_TEST") != "1" {
		t.Skip("set CUDA_DRIVER_HARDWARE_TEST=1 for native driver validation")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		fn   func() error
	}{
		{"stop", driver.ProfilerStop},
		{"start", driver.ProfilerStart},
		{"stop again", driver.ProfilerStop},
	} {
		if err := tt.fn(); err != nil {
			t.Errorf("%s: %v", tt.name, err)
		}
	}
}
