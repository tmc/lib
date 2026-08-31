package cuda

import (
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

var (
	initOnce sync.Once
	initErr  error
)

// Init initializes the CUDA driver API. Calling Init multiple times is safe and returns
// the result of the first initialization attempt. Most functions in this package call
// Init automatically if it has not already been called.
func Init() error {
	initOnce.Do(func() {
		initErr = driver.Init(0)
	})
	if initErr != nil {
		return fmt.Errorf("cuda: init: %w", initErr)
	}
	return nil
}

// DeviceCount returns the number of compute-capable CUDA devices available on the system.
func DeviceCount() (int, error) {
	if err := Init(); err != nil {
		return 0, err
	}
	count, err := driver.DeviceGetCount()
	if err != nil {
		return 0, fmt.Errorf("cuda: get device count: %w", err)
	}
	return count, nil
}

// Devices returns a list of all available CUDA devices on the system.
func Devices() ([]*Device, error) {
	count, err := DeviceCount()
	if err != nil {
		return nil, err
	}
	devs := make([]*Device, count)
	for i := 0; i < count; i++ {
		d, err := NewDevice(i)
		if err != nil {
			return nil, err
		}
		devs[i] = d
	}
	return devs, nil
}
