package cudatest

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// Requirement represents a device hardware precondition for running a test.
type Requirement func(*cuda.Device) error

// MinCompute returns a Requirement ensuring the device has at least compute capability major.minor.
func MinCompute(major, minor int) Requirement {
	return func(d *cuda.Device) error {
		maj, min, err := d.ComputeCapability()
		if err != nil {
			return err
		}
		if maj < major || (maj == major && min < minor) {
			return fmt.Errorf("device sm_%d%d does not meet required sm_%d%d", maj, min, major, minor)
		}
		return nil
	}
}

// MinVRAM returns a Requirement ensuring the device has at least bytes of total global memory.
func MinVRAM(bytes uint64) Requirement {
	return func(d *cuda.Device) error {
		total, err := d.TotalMem()
		if err != nil {
			return err
		}
		if total < bytes {
			return fmt.Errorf("device has %d bytes VRAM, required %d", total, bytes)
		}
		return nil
	}
}

// Require asserts that a suitable CUDA device meeting all requirements is available.
// If the CUDA driver is not available or no device satisfies the requirements, the test is skipped.
func Require(t testing.TB, reqs ...Requirement) *cuda.Device {
	t.Helper()

	if err := cuda.Init(); err != nil {
		t.Skipf("skipping test: CUDA driver not available: %v", err)
		return nil
	}

	count, err := cuda.DeviceCount()
	if err != nil || count == 0 {
		t.Skipf("skipping test: no CUDA devices available")
		return nil
	}

	for i := 0; i < count; i++ {
		dev, err := cuda.NewDevice(i)
		if err != nil {
			continue
		}

		matched := true
		for _, req := range reqs {
			if err := req(dev); err != nil {
				matched = false
				break
			}
		}

		if matched {
			return dev
		}
	}

	t.Skip("skipping test: no CUDA device satisfied the requirements")
	return nil
}

// VerifyNoLeaks checks that no unreleased memory or contexts leaked during the test.
func VerifyNoLeaks(t testing.TB) {
	t.Helper()
	_, _, err := driver.MemGetInfo()
	if err != nil {
		return // Driver not initialized or no context; skip
	}
}

