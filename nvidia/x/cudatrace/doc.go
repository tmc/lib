// Package cudatrace records host-side CUDA submission regions and explicit
// timestamped GPU spans in Perfetto traces.
//
// # Timeline Correlation
//
// Span and SpanSource use the host clock around the supplied function. They do
// not insert CUDA events or synchronize a stream, so they do not measure GPU
// execution. To place real device intervals on the timeline, bracket the work
// with CUDA events, convert them with a Clock from Calibrate, and pass the
// resulting host-domain timestamps to RecordSpan.
//
// # Go runtime/trace Integration
//
// [Tracer.Span] creates a runtime/trace region around the supplied host function.
// Runtime trace regions and Perfetto GPU spans are separate outputs; this
// package does not create runtime/trace tasks or automatically align device clocks.
package cudatrace
