package legacycupti

import "github.com/tmc/lib/nvidia/cupti"

func legacy() error {
	return cupti.EventGroupCreate() // want `EventGroupCreate always returns ErrLegacyProfilerNotSupported since CUDA 13.0; use the range profiler API, starting with RangeProfilerEnable`
}

func current() error {
	return cupti.RangeProfilerEnable()
}
