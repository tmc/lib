// Package metalprof provides Apple Metal capture helpers, hardware timestamp
// sampling, caller-supplied event recording, and pprof integration.
//
// # Context Labels & pprof.Do Integration
//
// When dispatching Metal compute passes, metalprof propagates active [runtime/pprof.Labels]
// from calling Go goroutines into GPU sample tags and Metal command encoder debug scopes
// using [Do]. This enables multidimensional filtering by model layer, request ID, or
// operation name across both Go CPU goroutines and Metal GPU execution.
//
// # Continuous GPU Flight Recording
//
// [FlightRecorder] stores caller-supplied Metal event durations and threadgroup
// dimensions in an in-memory ring buffer. It does not measure those durations;
// callers must retain the clock source separately. In-memory buffers can
// be snapshotted on demand (e.g. during a latency spike, numerical anomaly, or test
// failure) into standard gzipped profile.proto (for go tool pprof) or Perfetto .pftrace
// formats.
//
// # Programmatic Captures & Headless Replay
//
// [StartCapture] and [ReplayTrace] enable automated GPU trace captures (.gputrace)
// and headless replay profiling via macOS MTLReplayer without opening Xcode.
package metalprof
