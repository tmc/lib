package cuptirange

import (
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"runtime"
	"unsafe"
)

// CounterAvailability returns a bounded counter availability image for dev's
// current primary context. It queries context-level counters only. The device
// must be open. The returned image filters host metric configuration; it does
// not establish permission to run a later collection.
func CounterAvailability(dev *cuda.Device, maxBytes int) (image []byte, err error) {
	if dev == nil || maxBytes <= 0 {
		return nil, errors.New("cuptirange: invalid counter availability arguments")
	}
	version, err := cupti.GetVersion()
	if err != nil {
		return nil, err
	}
	if !qualifiedLayout(version) || unsafe.Sizeof(uintptr(0)) != 8 {
		return nil, fmt.Errorf("cuptirange: unqualified counter availability version %d", version)
	}
	err = dev.WithContext(func() error {
		init := cupti.ProfilerInitializeParams{}
		init.StructSize = uint(unsafe.Offsetof(init.PPriv) + unsafe.Sizeof(init.PPriv))
		if err := cupti.ProfilerInitialize(&init); err != nil {
			return fmt.Errorf("cuptirange: initialize profiler: %w", err)
		}
		ctx, err := driver.NewContextGetCurrent()
		if err != nil {
			return err
		}
		image, err = counterAvailability(cupti.Context(ctx), maxBytes, cupti.ProfilerGetCounterAvailability)
		return err
	})
	return image, err
}

func counterAvailability(ctx cupti.Context, maxBytes int, query func(*cupti.ProfilerGetCounterAvailabilityParams) error) ([]byte, error) {
	if ctx == 0 || maxBytes <= 0 || query == nil {
		return nil, errors.New("cuptirange: invalid counter availability query")
	}
	p := cupti.ProfilerGetCounterAvailabilityParams{Ctx: ctx}
	// Use the original prefix, ending at pCounterAvailabilityImage, documented in
	// CUDA 12.9 and retained by later versions. Omit bAllowDeviceLevelCounters:
	// context-only filtering must not implicitly enable device-level counters.
	// https://docs.nvidia.com/cupti/12.9/api/structCUpti__Profiler__GetCounterAvailability__Params.html
	p.StructSize = uint(unsafe.Offsetof(p.PCounterAvailabilityImage) + unsafe.Sizeof(p.PCounterAvailabilityImage))
	if err := query(&p); err != nil {
		return nil, fmt.Errorf("cuptirange: counter availability size: %w", err)
	}
	size := p.CounterAvailabilityImageSize
	if size == 0 || size > uint(maxBytes) {
		return nil, fmt.Errorf("cuptirange: counter availability size %d exceeds bounds", size)
	}
	image := make([]byte, int(size))
	var pin runtime.Pinner
	pin.Pin(&image[0])
	defer pin.Unpin()
	p.PCounterAvailabilityImage = uintptr(unsafe.Pointer(&image[0]))
	if err := query(&p); err != nil {
		return nil, fmt.Errorf("cuptirange: counter availability image: %w", err)
	}
	if p.CounterAvailabilityImageSize != size {
		return nil, errors.New("cuptirange: counter availability size changed")
	}
	return image, nil
}
