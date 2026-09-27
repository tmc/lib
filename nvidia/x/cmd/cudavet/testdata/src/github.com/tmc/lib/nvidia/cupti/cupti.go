// Package cupti is a declaration-only stand-in for the real cupti package.
package cupti

import "github.com/tmc/lib/nvidia/driver"

type EventGroup uintptr

// Deprecated: Unsupported since CUDA 13.0, where it always returns [ErrLegacyProfilerNotSupported]; use the range profiler API, starting with [RangeProfilerEnable].
//
// Wraps cuptiEventGroupCreate.
func NewEventGroup(context driver.Context, flags uint32) (EventGroup, error) { return 0, nil }

type RangeProfilerEnableParams struct{}

func RangeProfilerEnable(pParams *RangeProfilerEnableParams) error { return nil }
