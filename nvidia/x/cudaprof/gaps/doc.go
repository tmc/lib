// Package gaps measures the idle gaps in the device timeline of a cudaprof
// capture.
//
// pprof samples are aggregates: they sum durations and drop the start and stop
// timestamps, so inter-kernel idle time is not recoverable from them. [Analyze]
// reconstructs the timeline from the capture journals with the timeline
// package and reports, per device, the duty cycle, the distribution of idle
// gaps, what the host was doing at each gap, and a census of the kernels that
// ran.
//
// A capture whose journals are absent, omitted, truncated, or lossy is refused
// rather than analyzed, because a duty cycle computed from a surviving prefix
// reads exactly like one computed from the whole run.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package gaps
