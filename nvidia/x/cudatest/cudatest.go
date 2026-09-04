package cudatest

import (
	"errors"
	"fmt"
	"sync"
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

type memoryTelemetry interface {
	Synchronize() error
	Memory() (free, total uint64, err error)
}

type deviceMemoryTelemetry struct{ device *cuda.Device }

func (m deviceMemoryTelemetry) Synchronize() error {
	return m.device.WithContext(driver.CtxSynchronize)
}

func (m deviceMemoryTelemetry) Memory() (uint64, uint64, error) {
	var free, total uint
	err := m.device.WithContext(func() error {
		var err error
		free, total, err = driver.MemGetInfo()
		return err
	})
	return uint64(free), uint64(total), err
}

var memoryScopeLocks sync.Map // device ordinal -> *sync.Mutex

// DeviceMemoryCheck compares free device memory before and after a test scope.
// It does not observe contexts, modules, host allocations, or allocation
// ownership. Check serializes participating scopes for the same device, but
// allocations made outside cudatest can still affect its process-global
// measurement.
type DeviceMemoryCheck struct {
	telemetry memoryTelemetry
	lock      *sync.Mutex
	baseline  uint64
	total     uint64
	tolerance uint64

	mu   sync.Mutex
	done bool
}

// BeginDeviceMemoryCheck synchronizes device, captures its free-memory
// baseline, and starts a serialized check scope. tolerance is the maximum
// permitted decrease in free device memory. The caller must call Check even
// when the test body fails so the scope lock is released.
func BeginDeviceMemoryCheck(device *cuda.Device, tolerance uint64) (*DeviceMemoryCheck, error) {
	if device == nil {
		return nil, errors.New("cudatest: nil device")
	}
	lock, _ := memoryScopeLocks.LoadOrStore(device.Ordinal(), new(sync.Mutex))
	return beginDeviceMemoryCheck(deviceMemoryTelemetry{device}, lock.(*sync.Mutex), tolerance)
}

func beginDeviceMemoryCheck(telemetry memoryTelemetry, lock *sync.Mutex, tolerance uint64) (*DeviceMemoryCheck, error) {
	if telemetry == nil || lock == nil {
		return nil, errors.New("cudatest: invalid memory check backend")
	}
	lock.Lock()
	if err := telemetry.Synchronize(); err != nil {
		lock.Unlock()
		return nil, fmt.Errorf("cudatest: synchronize before memory check: %w", err)
	}
	free, total, err := telemetry.Memory()
	if err != nil {
		lock.Unlock()
		return nil, fmt.Errorf("cudatest: capture memory baseline: %w", err)
	}
	if total == 0 || free > total {
		lock.Unlock()
		return nil, errors.New("cudatest: invalid memory baseline")
	}
	return &DeviceMemoryCheck{telemetry: telemetry, lock: lock, baseline: free, total: total, tolerance: tolerance}, nil
}

// Check synchronizes the device and reports a retained decrease in free
// device memory greater than the configured tolerance. Check must be called
// exactly once.
func (c *DeviceMemoryCheck) Check() error {
	if c == nil {
		return errors.New("cudatest: nil memory check")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.done {
		return errors.New("cudatest: memory check already completed")
	}
	c.done = true
	defer c.lock.Unlock()
	if err := c.telemetry.Synchronize(); err != nil {
		return fmt.Errorf("cudatest: synchronize after memory check: %w", err)
	}
	free, total, err := c.telemetry.Memory()
	if err != nil {
		return fmt.Errorf("cudatest: capture final memory state: %w", err)
	}
	if total != c.total || free > total {
		return fmt.Errorf("cudatest: device memory telemetry changed: total %d, want %d", total, c.total)
	}
	if free >= c.baseline || c.baseline-free <= c.tolerance {
		return nil
	}
	return fmt.Errorf("cudatest: retained device memory grew by %d bytes (tolerance %d)", c.baseline-free, c.tolerance)
}
