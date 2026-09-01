// Package metalprof provides deep Apple Metal GPU profiling, hardware timestamp
// sampling, and pprof integration for Go applications on Apple Silicon.
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
// [FlightRecorder] provides continuous in-memory ring-buffer capture of Metal kernel
// execution events, hardware durations, and threadgroup dimensions. In-memory buffers can
// be snapshotted on demand (e.g. during a latency spike, numerical anomaly, or test
// failure) into standard gzipped profile.proto (for go tool pprof) or Perfetto .pftrace
// formats.
//
// # Programmatic Captures & Headless Replay
//
// [StartCapture] and [ReplayTrace] enable automated GPU trace captures (.gputrace)
// and headless replay profiling via macOS MTLReplayer without opening Xcode.
package metalprof
