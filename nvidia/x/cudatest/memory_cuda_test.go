package cudatest_test

import (
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestDeviceMemoryCheckCUDA(t *testing.T) {
	// This gate is not a device probe. The check reads process-global free
	// bytes through cuMemGetInfo, and the mutex it holds serializes only
	// within this process, so any other process using the GPU is attributed
	// to the test: under an 8 MiB alloc/free loop from a second process it
	// failed 2 runs in 10, both directions -- reporting another process's
	// allocation as retained growth, and having its own deliberate leak
	// masked by another process's free. It needs the GPU to itself.
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") == "" {
		t.Skip("LIB_CUDA_HARDWARE_TEST is empty: this test measures process-global device memory and needs exclusive use of the GPU")
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
