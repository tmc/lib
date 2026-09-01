// Package cudaprof enables deep Go to CUDA, PTX, and SASS profiling integration
// compatible with standard Go tooling (go tool pprof) and Perfetto timelines.
//
// # Context Labels & pprof.Do Integration
//
// When profiling GPU workloads, cudaprof propagates active [runtime/pprof.Labels] from
// the calling Go goroutines into GPU sample tags using [Do]. This enables multidimensional
// filtering and attribution by model layer, request ID, or tenant across Go host execution
// and GPU device kernels.
//
// # Continuous GPU Flight Recording
//
// [FlightRecorder] provides continuous in-memory ring-buffer capture of GPU execution
// events, kernel activity, and SM PC sampling. In-memory buffers can be snapshotted on demand
// (e.g. during an error, panic, or latency spike) into standard gzipped profile.proto or
// Perfetto .pftrace formats with zero disk overhead during normal operation.
//
// # Cross-Language Tracebacks
//
// cudaprof constructs unified multi-frame callstacks spanning Go host caller functions,
// CUDA C++ device kernels (.cu), JIT-compiled PTX virtual instructions, and hardware SASS opcodes.
package cudaprof

