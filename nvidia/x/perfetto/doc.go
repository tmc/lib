// Package perfetto implements a streaming wire-format encoder for Perfetto
// trace files (.pftrace) compatible with the Chrome / Perfetto UI trace viewer.
//
// It generates standard Perfetto protobuf trace packets (TrackDescriptor, TrackEvent)
// with zero external dependencies. Callers supply the events and timestamps;
// the package does not observe GPU kernels or memory copies itself.
//
// # Visualization
//
// Trace files emitted by this package can be opened directly at https://ui.perfetto.dev.
package perfetto
