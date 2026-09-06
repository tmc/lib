package driver_test

import (
	"github.com/tmc/lib/nvidia/driver"
	"os"
	"runtime"
	"testing"
)

func TestNativeLegacyCapability(t *testing.T) {
	if os.Getenv("CUDA_DRIVER_HARDWARE_TEST") != "1" {
		t.Skip("set CUDA_DRIVER_HARDWARE_TEST=1 for native driver validation")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	major, err := dev.GetAttribute(driver.DevAttrComputeCapabilityMajor)
	if err != nil {
		t.Fatal(err)
	}
	minor, err := dev.GetAttribute(driver.DevAttrComputeCapabilityMinor)
	if err != nil {
		t.Fatal(err)
	}
	var oldMajor, oldMinor int32
	if err := dev.ComputeCapability(&oldMajor, &oldMinor); err != nil {
		t.Fatal(err)
	}
	if major <= 0 || int(oldMajor) != major || int(oldMinor) != minor {
		t.Fatalf("legacy capability %d.%d, modern attributes %d.%d", oldMajor, oldMinor, major, minor)
	}
	t.Logf("legacy and modern capability agree: %d.%d", major, minor)
}
