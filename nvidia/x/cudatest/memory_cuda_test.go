package cudatest_test

import (
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestDeviceMemoryCheckCUDA(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") == "" {
		t.Skip("LIB_CUDA_HARDWARE_TEST is empty")
	}
	dev := cudatest.Require(t)
	defer dev.Close()
	// The CUDA driver retained 458,752 bytes after the first clean allocation
	// on the GB10 test host. Keep that observed driver cache below an explicit
	// 1 MiB allowance while testing a distinct 4 MiB retained allocation.
	const driverCacheTolerance = 1 << 20

	t.Run("clean", func(t *testing.T) {
		check, err := cudatest.BeginDeviceMemoryCheck(dev, driverCacheTolerance)
		if err != nil {
			t.Fatal(err)
		}
		var ptr driver.DevicePtr
		if err := dev.WithContext(func() error {
			var err error
			ptr, err = driver.MemAlloc(4 << 20)
			return err
		}); err != nil {
			t.Fatal(err)
		}
		if err := dev.WithContext(ptr.MemFree); err != nil {
			t.Fatal(err)
		}
		if err := check.Check(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("intentional allocation", func(t *testing.T) {
		check, err := cudatest.BeginDeviceMemoryCheck(dev, driverCacheTolerance)
		if err != nil {
			t.Fatal(err)
		}
		var ptr driver.DevicePtr
		if err := dev.WithContext(func() error {
			var err error
			ptr, err = driver.MemAlloc(4 << 20)
			return err
		}); err != nil {
			t.Fatal(err)
		}
		err = check.Check()
		if freeErr := dev.WithContext(ptr.MemFree); freeErr != nil {
			t.Fatal(freeErr)
		}
		if err == nil || !strings.Contains(err.Error(), "retained device memory grew") {
			t.Fatalf("Check() error = %v, want retained growth", err)
		}
	})
}
