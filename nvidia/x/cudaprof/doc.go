// Package cudaprof writes caller-supplied CUDA execution observations as Go
// profiles, Perfetto traces, and bounded in-memory recordings.
//
// The package does not collect CUPTI events or device counters. Callers remain
// responsible for measuring GPU work and passing those observations to a
// Session or FlightRecorder. Do attaches Go pprof labels and, when available,
// brackets its callback with an NVTX range.
package cudaprof
