// Package gaps measures gaps in the recorded device timeline of a cudaprof
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
// All streams on a device are merged before finding gaps. A stream-local
// gap covered by work on another stream is therefore recorded occupancy.
// Kernels, copies, and memsets contribute occupancy; synchronization and
// collection overhead are overlapping evidence, not occupancy. Graph replay
// envelopes contribute occupancy only when no graph node records exist.
//
// The analysis window begins at the first device activity and ends at the
// last activity end. Capture-edge gaps are not measured. Absence of recorded
// activity does not establish hardware idle: disabled activity kinds and
// unsupported collection can leave work unseen even in a lossless journal.
// API timing classes and synchronization overlaps do not prove causation.
// Every gap remains causally unknown in Stats.UnknownCauseNanos.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package gaps
