package cuptipmsampling

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

var (
	profilerOnce sync.Once
	profilerErr  error
)

func initProfiler() error {
	profilerOnce.Do(func() {
		var params cupti.ProfilerInitializeParams
		params.StructSize = uint(unsafe.Offsetof(params.PPriv) + unsafe.Sizeof(params.PPriv))
		if err := cupti.ProfilerInitialize(&params); err != nil {
			profilerErr = fmt.Errorf("cuptipmsampling: profiler initialize: %w", err)
		}
	})
	return profilerErr
}

// CounterAvailability queries the hardware counter availability image for the specified device.
// This image is used by the host profiler to filter out unavailable raw hardware metrics.
func CounterAvailability(deviceIndex uint32) ([]byte, error) {
	if _, err := CheckVersion(); err != nil {
		return nil, err
	}
	if err := initProfiler(); err != nil {
		return nil, err
	}

	// 1. Query required buffer size
	var params cupti.PmSamplingGetCounterAvailabilityParams
	params.StructSize = uint(unsafe.Offsetof(params.PCounterAvailabilityImage) + unsafe.Sizeof(params.PCounterAvailabilityImage))
	params.DeviceIndex = uint(deviceIndex)

	if err := cupti.PmSamplingGetCounterAvailability(&params); err != nil {
		return nil, fmt.Errorf("cuptipmsampling: query counter availability size: %w", err)
	}

	size := params.CounterAvailabilityImageSize
	if size == 0 {
		return nil, nil
	}

	// 2. Allocate buffer and retrieve availability image
	buf := make([]byte, size)
	params.PCounterAvailabilityImage = uintptr(unsafe.Pointer(&buf[0]))

	if err := cupti.PmSamplingGetCounterAvailability(&params); err != nil {
		runtime.KeepAlive(buf)
		return nil, fmt.Errorf("cuptipmsampling: get counter availability image: %w", err)
	}
	runtime.KeepAlive(buf)

	return buf, nil
}
