// Package perfetto implements a high-performance, streaming wire-format encoder for Perfetto
// trace files (.pftrace) compatible with the Chrome / Perfetto UI trace viewer.
//
// It generates standard Perfetto protobuf trace packets (TrackDescriptor, TrackEvent)
// with zero external dependencies, allowing real-time timeline visualization of GPU kernels,
// stream memory copies, and host goroutines.
//
// # Visualization
//
// Trace files emitted by this package can be opened directly at https://ui.perfetto.dev.
package perfetto
