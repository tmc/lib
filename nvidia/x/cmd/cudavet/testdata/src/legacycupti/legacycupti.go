package legacycupti

import (
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
)

func legacy(ctx driver.Context) (cupti.EventGroup, error) {
	return cupti.NewEventGroup(ctx, 0) // want `NewEventGroup always returns ErrLegacyProfilerNotSupported since CUDA 13.0; use the range profiler API, starting with RangeProfilerEnable`
}

func current(p *cupti.RangeProfilerEnableParams) error {
	return cupti.RangeProfilerEnable(p)
}
