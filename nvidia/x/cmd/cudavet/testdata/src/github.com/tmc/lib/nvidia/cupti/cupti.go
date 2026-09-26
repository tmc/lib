// Package cupti is a declaration-only stand-in for the real cupti package.
package cupti

// EventGroupCreate creates a new event group.
//
// Deprecated: Unsupported since CUDA 13.0, where it always returns
// ErrLegacyProfilerNotSupported; use the range profiler API, starting with
// RangeProfilerEnable.
func EventGroupCreate() error { return nil }

// RangeProfilerEnable enables the range profiler.
func RangeProfilerEnable() error { return nil }
